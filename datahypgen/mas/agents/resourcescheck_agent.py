'''
资源检查智能体

这个智能体只验证上一阶段任务计划显式声明的工具、基础数据和外部数据 requirement。
数据集检索只解析已声明 requirement；不可强绑定时返回诊断，由 StepPlan 决定如何重规划。
工具检索：如果缺少可以调用工具检索注册。

'''
import logging
import re
import signal
import threading
import time
import json
import asyncio
import os
import copy
from pathlib import Path
from typing import Dict, Any, List, Optional
from tqdm import tqdm
from .base_agent import BaseAgent, AgentExecutionError
from . import plan_utils

# 引入提供的模块
from ..tools.geo_data_search import GeoDataSearch
from ..tools.dataset_source_clients import probe_dataset
from ..tools.dataset_resolution_semantics import (
    classify_region_match,
    detect_family_matches,
    extract_region_terms,
    infer_identity_concepts,
    infer_observable_identity,
    is_climate_identity_text,
    is_known_product_family,
    is_land_cover_identity_text,
    normalize_frozen_identity_dimension,
    normalize_region_name,
    normalize_temporal_coverage,
    normalize_temporal_resolution,
)
from ..tools.global_coverage_evidence import assess_global_coverage_evidence
from ..tools.gee_auth import (
    initialize_gee_from_service_account,
    set_gee_deadline,
)
from ..tools.operation_semantics import evaluate_operation_compatibility
from ..tools.capability_details import CapabilityDetailStore
from ..tools.dataset_binding_proof import (
    build_dataset_binding_proof,
    validate_dataset_binding_proof,
)
from ..tools.tool_finder import get_tool_finder
from ..workflow.data_type import _PROBE_OK_STATUSES, is_remote_manifest_source

logger = logging.getLogger(__name__)


def step_input_data_types(step):
    """读取明确声明的类型；缺少类型是证据缺口，不是格式失败。"""
    inputs = step.get("inputs")
    if not isinstance(inputs, list):
        return []
    return [
        item["data_type"].strip() for item in inputs
        if isinstance(item, dict) and isinstance(item.get("data_type"), str)
        and item["data_type"].strip()
    ]


def step_required_dataset_ids(step):
    """只读取真实声明的数据引用，不补造缺失引用。"""
    inputs = step.get("inputs")
    if not isinstance(inputs, list):
        return []
    return [
        item["source_id"].strip() for item in inputs
        if isinstance(item, dict) and item.get("source_kind") == "required_dataset"
        and isinstance(item.get("source_id"), str) and item["source_id"].strip()
    ]


class _GEEProbeTimeoutError(TimeoutError):
    """Raised when the complete synchronous GEE probe exceeds its deadline."""


_PACKAGE_IMPORT_ALIASES = {
    "earthengine-api": "ee",
    "earthengine": "ee",
    "google-earth-engine": "ee",
    "scikit-learn": "sklearn",
    "scikit-image": "skimage",
    "pytorch": "torch",
    "pillow": "PIL",
    "opencv-python": "cv2",
    "pyyaml": "yaml",
    "gdal": "osgeo.gdal",
}

_IDENTITY_GENERIC_TOKENS = {
    "classification",
    "collection",
    "data",
    "dataset",
    "global",
    "grid",
    "image",
    "map",
    "product",
    "raster",
    "static",
    "timeseries",
}


class ResourcesCheck_Agent(BaseAgent):
    """
    资源检查智能体 (ResourcesCheck_Agent)
    
    职责:
    1. 资源检查: 对照 Task Plan 检查工具和数据集是否就绪。
    2. 资源获取: 
       - 若缺少数据，调用 GeoDataSearch 检索。
    """

    # GEE 探测支持的构造器全集，也是类型未知时的候选求证顺序
    _GEE_PROBE_CTORS = ("ee.ImageCollection", "ee.Image", "ee.FeatureCollection")

    def __init__(self, model, config: Dict[str, Any]):
        super().__init__(model, config)
        
        # 定义工具数据存储路径
        self.tool_data_dir = os.path.join("datahypgen", "mas", "tools", "data")
        self.tool_schema_dir = os.path.join(self.tool_data_dir, "tool_schemas")
        self.tool_desc_dir = self.tool_schema_dir
        
        os.makedirs(self.tool_data_dir, exist_ok=True)
        os.makedirs(self.tool_schema_dir, exist_ok=True)
        self.tool_finder = get_tool_finder()
        self.geo_search = GeoDataSearch(cache_enabled=True)

        global_config = config.get("_global_config", {})

        # Commit 2: 远程探测配置与缓存
        probe_config = config.get("remote_probe", {})
        self.probe_timeout = probe_config.get("timeout_seconds", 10)
        self.probe_max_retries = probe_config.get("max_retries", 2)
        self.probe_cache_ttl = probe_config.get("cache_ttl_seconds", 3600)  # 1小时
        self.remote_probe_cache: Dict[str, Dict[str, Any]] = {}  # 会话级缓存
        logger.debug(
            f"Remote probe config: timeout={self.probe_timeout}s, "
            f"retries={self.probe_max_retries}, cache_ttl={self.probe_cache_ttl}s"
        )

        # GEE 认证配置：支持每次访问前从环境文件刷新变量
        gee_config = global_config.get("tools", {}).get("gee", {}) if isinstance(global_config, dict) else {}
        self.gee_env_file = gee_config.get("env_file", ".env")
        self.gee_reload_env_each_call = bool(gee_config.get("reload_env_each_call", True))
        self.gee_api_key_env = gee_config.get("api_key_env", "GEE_API_KEY")
        self.gee_project_env = gee_config.get("project_env", "EARTHENGINE_PROJECT")
        logger.debug(
            "GEE auth config: env_file=%s, reload_each_call=%s, api_key_env=%s, project_env=%s",
            self.gee_env_file,
            self.gee_reload_env_each_call,
            self.gee_api_key_env,
            self.gee_project_env,
        )

        # Commit 4: capability index 加载（用于 step-level 精确验证）
        cap_cfg = global_config.get("tools", {}).get("capability_index", {}) if isinstance(global_config, dict) else {}
        self._cap_index_enabled: bool = cap_cfg.get("enabled", True)
        self._cap_index_path: str = cap_cfg.get(
            "path", "datahypgen/mas/tools/data/geo_capability_index.json"
        )
        self._cap_entries: List[Dict[str, Any]] = []
        self._cap_fn_lookup: Dict[str, Dict[str, Any]] = {}  # function_name → entry
        self._cap_detail_store: Optional[CapabilityDetailStore] = None
        if self._cap_index_enabled:
            self._load_capability_index()

        # Commit 4: task_type 闭集（唯一语义基准来自 core.py）
        try:
            from ..tools.core import TASK_TYPE_KEYWORDS
            self._task_type_closed_set: set = set(TASK_TYPE_KEYWORDS.keys())
        except ImportError:
            logger.warning("Failed to import TASK_TYPE_KEYWORDS from core.py; task_type closed set unavailable.")
            self._task_type_closed_set = set()

        # 定义系统提示词
        self.system_prompt = config.get("system_prompt", """
你的任务是根据给定的"任务执行计划"，针对特定的地理数据集进行深入分析。
在执行之前，你具备自我检查能力：如果发现工具缺失或数据缺失，你会主动要求获取。
""")
        self.semantic_resolution_retries = int(config.get("semantic_resolution_retries", 2) or 2)
        self.recall_min_score = float(config.get("recall_min_score", 0.18) or 0.18)
        self.weak_bind_min_score = float(config.get("weak_bind_min_score", 0.40) or 0.40)
        self.strong_bind_min_score = float(config.get("strong_bind_min_score", 0.65) or 0.65)
        self.allow_online_gee_fallback = bool(config.get("allow_online_gee_fallback", True))

    def _load_capability_index(self) -> None:
        """
        Commit 4: 加载 geo_capability_index.json 并构建 function_name → entry 快速查表。
        Fail-closed：加载失败时清空，不影响其他阶段。
        """
        try:
            idx_path = self._cap_index_path
            if not os.path.isabs(idx_path):
                idx_path = os.path.join(os.getcwd(), idx_path)
            if not os.path.exists(idx_path):
                logger.warning(f"[ResourceCheck] capability index not found: {idx_path}")
                return
            with open(idx_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                self._cap_entries = data
            elif isinstance(data, dict) and "entries" in data:
                self._cap_entries = data["entries"]
            else:
                logger.warning("[ResourceCheck] capability index has unexpected format.")
                return
            # 构建 function_name → entry lookup
            for entry in self._cap_entries:
                fn = entry.get("function_name", "").strip()
                if fn:
                    self._cap_fn_lookup[fn] = entry
            self._cap_detail_store = CapabilityDetailStore(
                Path(idx_path).parent / "geo_capability_details"
            )
            logger.debug(
                f"[ResourceCheck] capability index loaded: {len(self._cap_entries)} entries, "
                f"{len(self._cap_fn_lookup)} unique functions."
            )
        except Exception as e:
            logger.warning(f"[ResourceCheck] failed to load capability index: {e}")
            self._cap_entries = []
            self._cap_fn_lookup = {}
            self._cap_detail_store = None

    def _capability_entry_with_detail(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        if self._cap_detail_store is None:
            return dict(entry)
        detail = self._cap_detail_store.get(entry)
        return self._cap_detail_store.merge(entry, detail)

    # ------------------------------------------------------------------
    # Commit 4: step-level implementation validation
    # ------------------------------------------------------------------

    def _validate_step_implementations(
        self, task_plan: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        对每个步骤分别验证实现可达性与操作语义兼容性。

        打分规则（每步）：
          - high  (1.0) : function_name 精确命中 capability index 且 pip_verified=True
          - medium(0.6) : 函数命中 capability index（未 pip_verified），或步骤声明了 tools
                          （无论 implementation_hint 是否存在，只要未给出 function_name）
          - low   (0.3) : 仅靠 LLM 推断，不计入可执行通过

        不做本地环境 import 探测：capability index 命中即可用性证据。

        操作语义兼容性区分两类结果：
          - 硬失败（operation_mismatch / forbidden_function）：存在正向矛盾证据，
            operation_compatible=False，计入 step_failures 并拖垮 compatibility_passed。
          - advisory（operation_unproven）：仅缺少确定性证据（numpy/pandas/xarray
            等通用库函数常不在操作语义索引里），operation_compatible 保持 True，
            仅以 operation_compatibility_advisory=True 记录警告，不判失败。

        计划级得分 = 所有步骤得分的**最小值**（非平均值）。

        gap 区分：
          - coverage_gap  : task_type 可归入闭集，但找不到匹配函数
          - capability_gap: task_type 自身无法归入闭集（core.py TASK_TYPE_KEYWORDS）

        返回：
          {
              "step_validations": [...],    # per-step 验证记录
              "reachability_score": float,  # min(step reachability scores)
              "compatibility_passed": bool,
              "weakest_step_id": int|None,
              "step_failures": [...],        # 低置信步骤
              "failure_class": str,
          }
        """
        steps = task_plan if isinstance(task_plan, list) else []
        step_validations: List[Dict[str, Any]] = []
        step_failures: List[Dict[str, Any]] = []
        step_scores: List[float] = []
        compatibility_results: List[bool] = []

        for step in steps:
            step_id = step.get("id")
            step_name = step.get("step_name", "")
            hint = step.get("implementation_hint") or {}
            task_type = step.get("task_type", "")
            tools = step.get("tools", [])
            if isinstance(tools, str):
                tools = [tools]

            fn_name = hint.get("function_name", "") if isinstance(hint, dict) else ""
            pkg = hint.get("package", "") if isinstance(hint, dict) else ""

            # 判断 task_type 是否在闭集中
            task_type_in_closed_set = bool(
                task_type and self._task_type_closed_set and task_type in self._task_type_closed_set
            )

            # Phase 1: 精确命中 capability index
            score = 0.3  # default: low (LLM inference only)
            validation_level = "llm_inference"
            gap_type = ""
            index_entry = self._cap_fn_lookup.get(fn_name) if fn_name else None

            if index_entry and index_entry.get("pip_verified", False):
                # 高置信：精确命中且 pip_verified
                score = 1.0
                validation_level = "verified"
            elif index_entry:
                # 函数在 index 中但未 pip_verified：index 命中即可用性证据，给中等分
                score = 0.6
                validation_level = "index_unverified"
            elif fn_name:
                # 函数不在 index 中 → gap 区分
                if task_type_in_closed_set:
                    gap_type = "coverage_gap"
                elif task_type:
                    gap_type = "capability_gap"
                else:
                    gap_type = "coverage_gap"  # 无 task_type 时保守归类
            else:
                # 未给出 function_name（hint 缺失，或 hint 中只有 package 等部分信息）
                if not hint:
                    validation_level = "no_hint"
                # 步骤声明了 tools 即视为可用性证据，给中等分；不要求 hint 缺失，
                # 否则带部分 hint 的步骤反而比完全没有 hint 的步骤得分更低，
                # 被误判 tool_unreachable（numpy/pandas/xarray 等通用库最常见）。
                if tools:
                    score = 0.6
                    validation_level = "tools_declared"

            compatibility_entry = (
                self._capability_entry_with_detail(index_entry)
                if isinstance(index_entry, dict)
                else (hint if isinstance(hint, dict) else {})
            )
            compatibility = evaluate_operation_compatibility(step, compatibility_entry)
            # operation_unproven 只是"缺少确定性证据"（通用库函数不在操作语义
            # 索引里时的常态），不构成正向矛盾；只有 operation_mismatch /
            # forbidden_function 这类矛盾证据才算硬不兼容。缺证据降级为
            # advisory，不拖垮 compatibility_passed，也不触发 step_failures。
            # 优先读校验器自带的 advisory 标记（散文启发式分歧等负向判定均
            # 自标 advisory），按状态字符串枚举仅作旧版兜底。
            compatibility_advisory = not compatibility["compatible"] and bool(
                compatibility.get(
                    "advisory",
                    compatibility["status"] == "operation_unproven",
                )
            )
            hard_incompatible = (
                not compatibility["compatible"] and not compatibility_advisory
            )
            compatibility_results.append(not hard_incompatible)

            step_scores.append(score)

            validation_record = {
                "step_id": step_id,
                "step_name": step_name,
                "function_name": fn_name,
                "package": pkg,
                "task_type": task_type,
                "task_type_in_closed_set": task_type_in_closed_set,
                "validation_level": validation_level,
                "score": score,
                "gap_type": gap_type,
                "reachability_passed": score >= 0.6,
                "verification_gaps": ["tool_reachability_unverified"] if score < 0.6 else [],
                "operation_compatible": not hard_incompatible,
                "operation_compatibility_status": compatibility["status"],
                "operation_compatibility_advisory": compatibility_advisory,
                "operation_compatibility_reason": compatibility["reason"],
                "required_operations": compatibility["required_operations"],
                "candidate_operations": compatibility["candidate_operations"],
            }
            step_validations.append(validation_record)

            # 目录覆盖不足不证明工具不可达；这里只拦确定的操作语义冲突。
            # 真实导入/探测失败由资源检查的 adjustments 与 manifest 判定。
            if hard_incompatible:
                failure_type = "implementation_incompatible"
                step_failures.append({
                    "step_id": step_id,
                    "step_name": step_name,
                    "failure_type": failure_type,
                    "detail": (
                        f"validation_level={validation_level}, score={score}, "
                        f"gap_type={gap_type or 'none'}, "
                        f"compatibility={compatibility['status']}: {compatibility['reason']}"
                    ),
                    "tools_involved": tools,
                    "datasets_involved": [],
                    "gap_type": gap_type,
                    "suggested_task_type": task_type if gap_type == "capability_gap" else "",
                })

        # 计划级评分：最小值
        reachability_score = min(step_scores) if step_scores else 1.0
        weakest_step_id = None
        if step_scores:
            min_idx = step_scores.index(min(step_scores))
            if step_scores[min_idx] < 1.0:
                weakest_step_id = step_validations[min_idx].get("step_id")
        if weakest_step_id is None:
            weakest_step_id = next(
                (
                    record.get("step_id")
                    for record in step_validations
                    if not record.get("operation_compatible", False)
                ),
                None,
            )

        # failure_class 汇总
        failure_class = ""
        if step_failures:
            if any(
                failure.get("failure_type") == "implementation_incompatible"
                for failure in step_failures
            ):
                failure_class = "implementation_incompatible"
            gap_types = {f.get("gap_type", "") for f in step_failures}
            gap_types.discard("")
            if failure_class:
                pass
            elif len(gap_types) == 1:
                failure_class = "coverage_gap" if "coverage_gap" in gap_types else "capability_gap"
            elif len(gap_types) > 1:
                failure_class = "mixed"
            else:
                failure_class = "tool_unreachable"

        return {
            "step_validations": step_validations,
            "reachability_score": round(reachability_score, 2),
            "compatibility_passed": all(compatibility_results),
            "weakest_step_id": weakest_step_id,
            "step_failures": step_failures,
            "failure_class": failure_class,
        }

    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行全流程：资源可达性与操作语义检查 → manifest 构建与终止判定。
        """
        task_plan = context.get("task_plan", [])
        dataset_info = context.get("dataset_info", {})
        goal = context.get("goal", "Undefined Goal")
        active_application_id = (
            context.get("active_application_id")
            or context.get("application_context", {}).get("idea_id")
        )

        if not active_application_id:
            raise AgentExecutionError(
                "ResourcesCheck_Agent requires active_application_id for application-scoped validation"
            )
        if not isinstance(task_plan, list):
            raise AgentExecutionError(
                "ResourcesCheck_Agent expects an application-scoped task_plan list"
            )
        if not task_plan:
            logger.warning(
                f"Agent {self.name}: No task plan provided for application {active_application_id}."
            )
            return {"status": "skipped", "reason": "No task plan", "active_application_id": active_application_id}
        # 资源检查不运行 DAG/schema 格式门禁；仅消费计划中的资源声明。

        logger.info(
            f"Agent {self.name} starting workflow for goal: {goal} "
            f"(application={active_application_id})"
        )

        adjustments: List[Dict[str, Any]] = []
        adjusted_plan = task_plan
        updated_dataset_info = dataset_info

        require_binding_proof = (
            context.get("resource_check_mode") != "single_pass_report"
        )

        try:
            # 阶段B：工具验证 + 主数据集路径验证 + 辅助数据集批量检查
            adjusted_plan, updated_dataset_info, adjustments = await self._check_and_prepare_resources(
                task_plan,
                dataset_info,
                require_binding_proof=require_binding_proof,
            )
        finally:
            await self.geo_search.close()

        # 阶段C：构建 data_access_manifest + 远程探测 + 统一终止判定
        manifest = self._build_data_access_manifest(adjusted_plan, updated_dataset_info, context)

        # Commit 2: 远程可达性探测
        manifest = await self._probe_remote_connectivity(manifest)

        termination = self._evaluate_termination_condition_with_manifest(
            adjusted_plan, updated_dataset_info, adjustments, manifest, context
        )
        dataset_resolution_failures = self._build_dataset_resolution_failures(updated_dataset_info)

        # Commit 4: step-level implementation validation
        impl_validation = self._validate_step_implementations(adjusted_plan)
        logger.info(
            f"[ResourceCheck] step implementation validation: "
            f"reachability_score={impl_validation['reachability_score']}, "
            f"compatibility_passed={impl_validation['compatibility_passed']}, "
            f"weakest_step_id={impl_validation['weakest_step_id']}, "
            f"failures={len(impl_validation['step_failures'])}"
        )

        status = "terminated" if termination["should_terminate"] else "completed"
        if termination["should_terminate"]:
            logger.error(
                f"Agent {self.name}: Terminating workflow. "
                f"Rule={termination['terminal_rule']}, Reason={termination['reason']}"
            )

        overall_failure_class = str(termination.get("failure_class", "") or "").strip()
        validation_failure_class = str(impl_validation.get("failure_class", "") or "").strip()
        if validation_failure_class:
            if overall_failure_class and overall_failure_class != validation_failure_class:
                overall_failure_class = "mixed"
            elif not overall_failure_class:
                overall_failure_class = validation_failure_class

        return {
            "status": status,
            "active_application_id": active_application_id,
            "task_plan": adjusted_plan,
            "dataset_info": updated_dataset_info,
            "adjustments": adjustments,
            "missing_datasets": updated_dataset_info.get("unresolvable_datasets", []),
            "termination": termination,
            "guaranteed_datasets": termination.get("guaranteed_datasets", []),
            "data_access_manifest": manifest,
            "dataset_resolution_failures": dataset_resolution_failures,
            # Commit 4: step-level validation results
            "step_tool_validations": impl_validation["step_validations"],
            "tool_reachability_score": impl_validation["reachability_score"],
            "implementation_compatibility_passed": impl_validation["compatibility_passed"],
            "weakest_step_id": impl_validation["weakest_step_id"],
            "step_failures": impl_validation["step_failures"],
            "failure_class": overall_failure_class,
        }
    @staticmethod
    def _normalize_local_data_path(data_path: Any) -> str:
        """Normalize a local dataset reference to an absolute project path."""
        raw_path = str(data_path or "").strip()
        if not raw_path:
            return ""
        while raw_path.upper().startswith("LOCAL:"):
            raw_path = raw_path[6:].strip()
        if not raw_path:
            return ""
        if raw_path.upper().startswith(
            ("GEE:", "NASA_CMR:", "PLANETARY_COMPUTER:", "EARTH_SEARCH:", "HTTP:", "HTTPS:")
        ):
            return raw_path
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[3] / path
        return str(path.resolve(strict=False))

    @classmethod
    def _validate_data_path(cls, data_path: str) -> bool:
        """Return whether the declared base-data path is an existing file or directory."""
        normalized_path = cls._normalize_local_data_path(data_path)
        is_valid = os.path.isfile(normalized_path) or os.path.isdir(normalized_path)
        if is_valid:
            logger.info(f"Data path validated: {normalized_path}")
        else:
            logger.error(f"Declared data path does not exist: {normalized_path}")
        return is_valid

    @staticmethod
    def _package_name_without_version(package_name: str) -> str:
        return re.split(r"[<>=!~]", str(package_name or "").strip(), maxsplit=1)[0].strip()

    @classmethod
    def _normalize_package_import_name(cls, package_name: str) -> str:
        package_key = cls._package_name_without_version(package_name).casefold()
        if package_key in _PACKAGE_IMPORT_ALIASES:
            return _PACKAGE_IMPORT_ALIASES[package_key]
        return package_key.replace("-", "_")

    @classmethod
    def _capability_entry_matches_tool(
        cls,
        entry: Dict[str, Any],
        tool_name: str,
    ) -> bool:
        requested_root = cls._normalize_package_import_name(tool_name).split(".", 1)[0].casefold()
        entry_package_root = cls._normalize_package_import_name(
            entry.get("package", "")
        ).split(".", 1)[0].casefold()
        function_root = str(entry.get("function_name", "") or "").split(".", 1)[0].casefold()
        return bool(
            requested_root
            and requested_root in {entry_package_root, function_root}
        )

    def _resolve_step_tools(
        self, task_steps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        B-1: Per-step tool resolution via capability index search.

        For each step:
          - normalize install names, import names, and module paths
          - search the capability index by normalized import name
          - keep capability entries as function hints

        注意：不做本地环境 import 探测。capability index 是工具可用性的唯一证据源；
        本地 venv 缺包不应阻塞计划校验（计划描述的是目标执行环境能力）。

        Returns:
          {
            "resolved": [cap_entry, ...],
            "adjustments": [...],
            "prompt_hints": str,   # function_name + signature for prompt injection
          }
        """
        resolved: List[Dict[str, Any]] = []
        adjustments: List[Dict[str, Any]] = []
        seen_fns: set = set()

        for step in task_steps:
            hint = step.get("implementation_hint") or {}
            selected_package = (
                str(hint.get("package") or "").strip()
                if isinstance(hint, dict)
                else ""
            )
            required_runtime_tools = step.get("required_runtime_tools") or []
            if isinstance(required_runtime_tools, str):
                required_runtime_tools = [required_runtime_tools]
            tools = [selected_package] if selected_package else []
            tools.extend(
                str(tool).strip()
                for tool in required_runtime_tools
                if str(tool).strip()
            )
            if not tools:
                legacy_tools = step.get("tools", [])
                tools = [legacy_tools] if isinstance(legacy_tools, str) else list(legacy_tools)
            tools = list(dict.fromkeys(tools))
            input_types = step_input_data_types(step)

            for tool_name in tools:
                import_name = self._normalize_package_import_name(str(tool_name))
                if " " in import_name:
                    import_name = str(tool_name).strip()
                search_keyword = import_name.split(".", 1)[0] or str(tool_name)
                raw_results = self.tool_finder.search(search_keyword, input_types, top_k=5)
                results = [
                    entry
                    for entry in raw_results
                    if self._capability_entry_matches_tool(entry, str(tool_name))
                ]

                if not results:
                    marker_key = f"package:{import_name}"
                    if marker_key not in seen_fns:
                        seen_fns.add(marker_key)
                        resolved.append({
                            "function_name": import_name,
                            "package": self._package_name_without_version(str(tool_name)),
                            "signature": "",
                            "pip_verified": True,
                            "resolution_source": "declared_package",
                        })
                    continue

                for entry in results:
                    fn = entry.get("function_name", "")
                    if fn in seen_fns:
                        continue
                    seen_fns.add(fn)
                    resolved.append(entry)

        hint_lines = []
        for entry in resolved:
            if entry.get("resolution_source") == "declared_package":
                continue
            fn = entry.get("function_name", "")
            sig = entry.get("signature", "")
            desc = (entry.get("raw_description") or "")[:120]
            hint_lines.append(f"{fn}{sig}")
            if desc:
                hint_lines.append(f"# {desc}")
        prompt_hints = "\n".join(hint_lines)

        return {
            "resolved": resolved,
            "adjustments": adjustments,
            "prompt_hints": prompt_hints,
        }

    async def _check_and_prepare_resources(
        self,
        task_plan: List[Dict],
        dataset_info: Dict,
        *,
        require_binding_proof: bool = True,
    ) -> tuple:
        """
        阶段B：工具验证 + 主数据集路径验证 + 辅助数据集批量检查与统一调整。

        B-1: 工具包按 capability index 解析（不做本地 import 探测）
        B-2: 主数据集声明路径的精确文件系统验证
        B-3: 对显式外部数据 requirement 做批量检索与强绑定
        """
        logger.info("Starting Resource Check Phase (strict data gateway)...")
        adjustments: List[Dict[str, Any]] = []

        # ── B-1: 按 step 检索工具，携带 inputs 中声明的数据类型 ─────────────
        task_steps = task_plan if isinstance(task_plan, list) else task_plan.get("task_plan", [])
        tool_resolution = self._resolve_step_tools(task_steps)
        adjustments.extend(tool_resolution["adjustments"])

        if tool_resolution["prompt_hints"]:
            dataset_info["tool_prompt_hints"] = tool_resolution["prompt_hints"]

        logger.info(
            f"resolved={len(tool_resolution['resolved'])} functions, "
            f"adjustments={len(tool_resolution['adjustments'])}"
        )

        # ── B-2: 主数据集文件系统路径验证 ────────────────────────────────────
        declared_primary_data_path = (
            dataset_info.get("data_path")
            or (dataset_info.get("base_dataset") or {}).get("data_path")
        )
        primary_data_path = self._normalize_local_data_path(declared_primary_data_path)
        if declared_primary_data_path:
            dataset_info["primary_data_original_path"] = declared_primary_data_path
            if primary_data_path:
                dataset_info["data_path"] = primary_data_path
                if isinstance(dataset_info.get("base_dataset"), dict):
                    dataset_info["base_dataset"]["data_path"] = primary_data_path
            if not self._validate_data_path(primary_data_path):
                logger.error(f"PRIMARY DATA PATH NOT FOUND: {primary_data_path}")
                dataset_info["primary_data_status"] = "path_not_found"
                adjustments.append({
                    "type": "primary_data_path_invalid",
                    "path": declared_primary_data_path,
                    "normalized_path": primary_data_path,
                    "terminal": True,
                    "note": "主数据集路径不存在，后续终止判定将阻断分析。",
                })
            else:
                dataset_info["primary_data_status"] = "verified"
        else:
            # 无本地路径：不能默认 verified，标记为 no_local_path
            # 只有在线检索结果能提供可达性保证时才由下游覆盖此状态
            dataset_info.setdefault("primary_data_status", "no_local_path")

        # ── B-3: 显式外部数据 requirement 批量检查 ──────────────────────────
        adjusted_plan, updated_dataset_info, dataset_adjustments = await self._evaluate_dataset_requirements(
            task_plan=task_plan,
            dataset_info=dataset_info,
            require_binding_proof=require_binding_proof,
        )
        adjustments.extend(dataset_adjustments)

        return adjusted_plan, updated_dataset_info, adjustments

    async def resolve_plan_dataset_requirements(
        self,
        *,
        required_datasets: List[Dict[str, Any]],
        dataset_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Resolve requirement drafts before StepPlan may generate a final graph."""
        if not isinstance(required_datasets, list):
            raise AgentExecutionError(
                "plan-time dataset resolution requires required_datasets as a list"
            )
        if any(not isinstance(item, dict) for item in required_datasets):
            raise AgentExecutionError(
                "plan-time dataset resolution requires structured requirement objects"
            )
        requirement_ids = [
            self._get_required_dataset_id(item) for item in required_datasets
        ]
        if len(requirement_ids) != len(set(requirement_ids)):
            raise AgentExecutionError(
                "plan-time dataset resolution requires unique requirement ids"
            )

        working_dataset_info = copy.deepcopy(dataset_info)
        working_dataset_info["required_datasets"] = copy.deepcopy(
            required_datasets
        )
        _, resolved_info, _ = await self._evaluate_dataset_requirements(
            task_plan=[],
            dataset_info=working_dataset_info,
            require_binding_proof=False,
        )
        reports = resolved_info.get("dataset_resolution_report") or {}
        resolved_entries = resolved_info.get("resolved_datasets") or {}
        proofed_requirements: List[Dict[str, Any]] = []
        downgraded_optional_requirements: List[Dict[str, Any]] = []
        hard_unresolvable_ids: List[str] = []

        for requirement in resolved_info.get("required_datasets") or []:
            requirement_copy = copy.deepcopy(requirement)
            requirement_id = self._get_required_dataset_id(requirement_copy)
            report = reports.get(requirement_id) or {}
            if report.get("status") == "resolved_strong":
                resolved_entry = resolved_entries.get(requirement_id)
                try:
                    requirement_copy["dataset_binding_proof"] = (
                        build_dataset_binding_proof(
                            requirement_copy,
                            resolved_entry,
                            report,
                        )
                    )
                except (TypeError, ValueError) as exc:
                    raise AgentExecutionError(
                        "failed to build dataset binding proof for "
                        f"{requirement_id}: {exc}"
                    ) from exc
                proofed_requirements.append(requirement_copy)
                continue
            execution_level = self._get_requirement_execution_requirement_level(
                requirement_copy
            )
            unbound_class = str(report.get("unbound_class") or "").strip()
            if execution_level == "required" and unbound_class == "no_candidate":
                requirement_copy["execution_requirement_level"] = "optional"
                requirement_copy["resolution_status"] = str(
                    report.get("binding_status") or report.get("status") or "unbound"
                )
                requirement_copy["resolved_dataset"] = None
                requirement_copy["dataset_resolution_failure"] = copy.deepcopy(report)
                requirement_copy["plan_time_resolution_action"] = (
                    "downgraded_to_optional_no_candidate"
                )
                downgraded_optional_requirements.append(requirement_copy)
                continue
            if execution_level == "optional":
                requirement_copy["resolution_status"] = str(
                    report.get("binding_status") or report.get("status") or "unbound"
                )
                requirement_copy["resolved_dataset"] = None
                requirement_copy["dataset_resolution_failure"] = copy.deepcopy(report)
                requirement_copy["plan_time_resolution_action"] = (
                    "omitted_from_frozen_contract_optional_unresolved"
                )
                downgraded_optional_requirements.append(requirement_copy)
                continue
            hard_unresolvable_ids.append(requirement_id)

        return {
            "status": (
                "resolved_strong" if not hard_unresolvable_ids else "unresolved"
            ),
            "required_datasets": proofed_requirements,
            "unresolvable_datasets": hard_unresolvable_ids,
            "omitted_optional_datasets": downgraded_optional_requirements,
            "dataset_resolution_report": copy.deepcopy(reports),
            "resolution_log": copy.deepcopy(
                resolved_info.get("resolution_log") or []
            ),
        }

    async def _evaluate_dataset_requirements(
        self,
        task_plan: List[Dict[str, Any]],
        dataset_info: Dict[str, Any],
        require_binding_proof: bool = False,
    ) -> tuple:
        """
        B-3: 显式外部数据 requirement 批量检查与绑定。

        检查阶段：对每个 required_dataset 顺序尝试三种检索策略，仅记录结果，不修改计划。
        写回阶段：只增加 strong resolved bindings；未强绑定项保留原计划并输出重规划诊断。
        """
        if "required_datasets" not in dataset_info or not isinstance(
            dataset_info["required_datasets"], list
        ):
            raise AgentExecutionError(
                "ResourceCheck dataset evaluation requires the validated required_datasets list"
            )
        required_datasets = dataset_info["required_datasets"]

        search_limit = max(50, int(self.config.get("dataset_search_limit", 50) or 50))

        resolved: Dict[str, Dict[str, Any]] = {}
        unresolvable: List[str] = []
        unresolvable_names: List[str] = []
        unresolvable_details: Dict[str, Dict[str, Any]] = {}
        resolution_log: List[Dict[str, Any]] = []
        resolution_report: Dict[str, Dict[str, Any]] = {}
        adjusted_plan = task_plan if isinstance(task_plan, list) else []
        adjustments: List[Dict[str, Any]] = []

        for requirement_index, req_item in enumerate(required_datasets):
            if not isinstance(req_item, dict):
                raise AgentExecutionError(
                    f"required_datasets[{requirement_index}] must be a structured object"
                )
            dataset_name = self._get_required_dataset_name(req_item)
            requirement_id = self._get_required_dataset_id(req_item)
            req = self._upgrade_legacy_required_dataset_contract(req_item)

            if require_binding_proof:
                proof_valid, proof_reason, proof_entry = (
                    validate_dataset_binding_proof(req)
                )
                if proof_valid:
                    resolved[requirement_id] = proof_entry
                    proof = req["dataset_binding_proof"]
                    resolution_report[requirement_id] = {
                        "status": "resolved_strong",
                        "dataset_name": dataset_name,
                        "binding_status": "resolved_strong",
                        "bind_status_reason": "verified_binding_proof_reused",
                        "resolution_source": "binding_proof_reuse",
                        "constraint_report": copy.deepcopy(
                            proof.get("constraint_report") or {}
                        ),
                        "blocking_reasons": [],
                        "penalty_report": copy.deepcopy(
                            proof_entry.get("penalty_report") or {}
                        ),
                        "best_candidate": {
                            "name": proof_entry.get("name")
                            or proof_entry.get("title"),
                            "id": proof_entry.get("id"),
                            "type": proof_entry.get("type"),
                        },
                        "search_stage": "binding_proof_reuse",
                        "online_attempted": False,
                        "failure_status": "",
                        "unbound_class": "",
                        "top_candidates": [],
                        "frozen_constraints_satisfied": bool(
                            proof.get("frozen_constraints_satisfied", False)
                        ),
                    }
                    resolution_log.append(
                        {
                            "dataset": dataset_name,
                            "requirement_id": requirement_id,
                            "status": "resolved_strong",
                            "strategy": "binding_proof_reuse",
                            "reasons": ["verified_binding_proof_reused"],
                            "confidence_flags": copy.deepcopy(
                                proof.get("constraint_report") or {}
                            ),
                            "top_candidates": [],
                            "attempts": [],
                        }
                    )
                else:
                    resolution_report[requirement_id] = {
                        "status": "unresolved_binding_proof",
                        "dataset_name": dataset_name,
                        "binding_status": "unresolved_binding_proof",
                        "bind_status_reason": proof_reason,
                        "resolution_source": "binding_proof_validation",
                        "constraint_report": {},
                        "blocking_reasons": [proof_reason],
                        "penalty_report": {},
                        "best_candidate": None,
                        "search_stage": "binding_proof_validation",
                        "online_attempted": False,
                        "failure_status": "unresolved_binding_proof",
                        "unbound_class": "binding_proof_invalid",
                        "top_candidates": [],
                        "frozen_constraints_satisfied": False,
                    }
                    resolution_log.append(
                        {
                            "dataset": dataset_name,
                            "requirement_id": requirement_id,
                            "status": "unresolved_binding_proof",
                            "unbound_class": "binding_proof_invalid",
                            "reasons": [proof_reason],
                            "confidence_flags": {},
                            "top_candidates": [],
                            "attempts": [],
                        }
                    )
                continue

            modality = req.get("modality", "unknown")
            retrieval_hint = str(req.get("retrieval_hint", "auto")).lower()
            attempts: List[Dict[str, Any]] = []
            search_intent = self._build_dataset_search_intent(req, dataset_info)
            search_intent["search_k"] = search_limit

            best_result: Dict[str, Any] = {
                "status": "unresolved_real_missing",
                "candidate": None,
                "top_candidates": [],
                "binding_decision": {},
                "reasons": ["no_candidates"],
                "confidence_flags": {},
                "penalty_report": {},
                "frozen_constraints_satisfied": False,
            }
            best_strategy = ""

            primary_resolution = self._resolve_primary_dataset_requirement(req, dataset_info)
            if primary_resolution.get("matched"):
                if primary_resolution.get("entry"):
                    entry = dict(primary_resolution["entry"])
                    entry["binding_status"] = "resolved_strong"
                    entry["bind_status_reason"] = "primary_dataset_reused"
                    entry["resolution_source"] = "primary_dataset"
                    entry["constraint_report"] = {}
                    entry["blocking_reasons"] = []
                    entry["penalty_report"] = {
                        "region_penalty": 0.0,
                        "time_penalty": 0.0,
                        "role_penalty": 0.0,
                    }
                    entry["frozen_constraints_satisfied"] = True
                    resolved[requirement_id] = entry
                    resolution_report[requirement_id] = {
                        "status": "resolved_strong",
                        "dataset_name": dataset_name,
                        "binding_status": "resolved_strong",
                        "bind_status_reason": "primary_dataset_reused",
                        "resolution_source": "primary_dataset",
                        "constraint_report": {},
                        "blocking_reasons": [],
                        "penalty_report": entry["penalty_report"],
                        "best_candidate": None,
                        "search_stage": "primary_dataset_reuse",
                        "online_attempted": False,
                        "failure_status": "",
                        "unbound_class": "",
                        "top_candidates": [],
                        "frozen_constraints_satisfied": True,
                    }
                    resolution_log.append({
                        "dataset": dataset_name,
                        "requirement_id": requirement_id,
                        "status": "resolved_strong",
                        "strategy": "primary_dataset_reuse",
                        "reasons": ["primary_dataset_reused"],
                        "confidence_flags": {},
                        "top_candidates": [],
                        "attempts": [],
                    })
                else:
                    resolution_report[requirement_id] = {
                        "status": "unresolved_real_missing",
                        "dataset_name": dataset_name,
                        "binding_status": "unresolved_real_missing",
                        "bind_status_reason": "primary_dataset_unavailable",
                        "resolution_source": "primary_dataset",
                        "constraint_report": {},
                        "blocking_reasons": ["primary_dataset_unavailable"],
                        "penalty_report": {},
                        "best_candidate": None,
                        "search_stage": "primary_dataset_reuse",
                        "online_attempted": False,
                        "failure_status": "unresolved_real_missing",
                        "unbound_class": "real_unreachable",
                        "top_candidates": [],
                        "frozen_constraints_satisfied": False,
                    }
                    resolution_log.append({
                        "dataset": dataset_name,
                        "requirement_id": requirement_id,
                        "status": "unresolved_real_missing",
                        "unbound_class": "real_unreachable",
                        "reasons": ["primary_dataset_unavailable"],
                        "confidence_flags": {},
                        "top_candidates": [],
                        "attempts": [],
                    })
                continue

            seen_local_candidate_ids: set = set()
            for query in self._build_search_queries_from_intent(search_intent):
                attempt_intent = {
                    **search_intent,
                    "query_text": query,
                    "raw_query": query,
                    "search_k": search_limit,
                }
                local_result = self.geo_search.resolve_local_dataset(attempt_intent)
                local_result = await self._validate_resolution_candidate_identity(
                    local_result,
                    req=req,
                    dataset_info=dataset_info,
                    modality=modality,
                    retrieval_hint=retrieval_hint,
                    search_intent=search_intent,
                )
                attempts.append({
                    "attempt": len(attempts) + 1,
                    "strategy": "local_resolution",
                    "keyword": query,
                    "found": local_result.get("candidate") is not None,
                    "status": local_result.get("status"),
                    "reasons": local_result.get("reasons", []),
                })
                if self._is_better_resolution_result(local_result, best_result):
                    best_result = local_result
                    best_strategy = "local_resolution"
                if local_result.get("status") == "resolved_strong":
                    best_result = local_result
                    best_strategy = "local_resolution"
                    break
                # 本地已给出同一个 weak 候选时，换 query 词再打在线目录不会改善结论：
                # 在线要么返回同一 asset，要么返回 family 不匹配的别的东西，而无 blocking
                # 的 weak 候选循环结束后本就会被接受。每次在线要并行等 GEE/CMR/PC 三源，
                # 网络差时单次数十秒，一个 requirement 重复 10 次就是十几分钟。
                local_candidate_id = self._resolution_candidate_id(local_result)
                repeated_local_candidate = (
                    local_candidate_id is not None
                    and local_candidate_id in seen_local_candidate_ids
                    and local_result.get("status") == "resolved_weak"
                )
                if local_candidate_id is not None:
                    seen_local_candidate_ids.add(local_candidate_id)
                if repeated_local_candidate:
                    attempts.append({
                        "attempt": len(attempts) + 1,
                        "strategy": "online_catalog_resolution",
                        "keyword": query,
                        "found": False,
                        "status": "skipped",
                        "reasons": ["repeated_local_weak_candidate"],
                    })
                    continue
                if self.allow_online_gee_fallback and local_result.get("status") != "resolved_strong":
                    online_result = await self.geo_search.resolve_online_dataset(attempt_intent)
                    online_result = await self._validate_resolution_candidate_identity(
                        online_result,
                        req=req,
                        dataset_info=dataset_info,
                        modality=modality,
                        retrieval_hint=retrieval_hint,
                        search_intent=search_intent,
                    )
                    attempts.append({
                        "attempt": len(attempts) + 1,
                        "strategy": "online_catalog_resolution",
                        "keyword": query,
                        "found": online_result.get("candidate") is not None,
                        "status": online_result.get("status"),
                        "reasons": online_result.get("reasons", []),
                    })
                    if self._is_better_resolution_result(online_result, best_result):
                        best_result = online_result
                        best_strategy = "online_catalog_resolution"
                    if online_result.get("status") == "resolved_strong":
                        best_result = online_result
                        best_strategy = "online_catalog_resolution"
                        break

            if best_result.get("status") != "resolved_strong":
                family_result = await self._try_family_substitution(
                    req,
                    search_intent,
                    search_limit,
                    dataset_info=dataset_info,
                    modality=modality,
                    retrieval_hint=retrieval_hint,
                )
                if family_result is not None:
                    attempts.append({
                        "attempt": len(attempts) + 1,
                        "strategy": "family_substitution",
                        "keyword": family_result.get("family_substitution_query", ""),
                        "found": family_result.get("candidate") is not None,
                        "status": family_result.get("status"),
                        "reasons": family_result.get("reasons", []),
                    })
                    if self._is_better_resolution_result(family_result, best_result):
                        best_result = family_result
                        best_strategy = "family_substitution"

            final_status = str(best_result.get("status") or "unresolved_real_missing")
            binding_decision = best_result.get("binding_decision", {}) or {}
            blocking_reasons = list(
                binding_decision.get("blocking_reasons", best_result.get("reasons", [])) or []
            )
            verification_gaps = list(
                binding_decision.get("verification_gaps", []) or []
            )
            soft_reasons = list(binding_decision.get("soft_reasons", []) or [])
            penalty_report = dict(best_result.get("penalty_report", {}) or binding_decision.get("penalty_report", {}) or {})
            constraint_report = dict(binding_decision.get("constraint_report", {}) or {})
            unsupported_identity_constraints = list(
                search_intent.get("identity_constraint_unsupported") or []
            )
            if unsupported_identity_constraints:
                constraint_report["identity_constraint_unsupported"] = (
                    unsupported_identity_constraints
                )
            top_candidates = list(best_result.get("top_candidates", []) or [])
            best_candidate_summary = top_candidates[0] if top_candidates else None
            if best_candidate_summary is None and best_result.get("candidate") is not None:
                candidate = best_result["candidate"]
                candidate_properties = getattr(candidate, "properties", None)
                if not isinstance(candidate_properties, dict) and isinstance(candidate, dict):
                    candidate_properties = candidate.get("properties") or candidate
                if not isinstance(candidate_properties, dict):
                    candidate_properties = {}
                best_candidate_summary = {
                    "name": (
                        candidate_properties.get("title")
                        or getattr(candidate, "name", None)
                        or candidate_properties.get("name")
                    ),
                    "id": (
                        candidate_properties.get("id")
                        or candidate_properties.get("ee_id")
                        or getattr(candidate, "feature_id", None)
                    ),
                    "type": (
                        candidate_properties.get("type")
                        or getattr(candidate, "feature_type", None)
                    ),
                }

            # 对外部源（NASA_CMR / PLANETARY_COMPUTER / EARTH_SEARCH）放宽绑定条件：
            # 只要达到 weak 阈值且无阻断原因，就接受为可绑定（升级到 strong）。
            # 这些源缺少时空详情，无法满足冻结约束，但不应因此被拒绝。
            candidate = best_result.get("candidate")
            is_external_source = False
            if candidate is not None:
                candidate_source = getattr(candidate, "source", "")
                candidate_props = getattr(candidate, "properties", {}) or {}
                dataset_source = candidate_props.get("dataset_source", "")
                is_external_source = any(
                    ext_src in candidate_source or ext_src in dataset_source
                    for ext_src in ("NASA_CMR", "PLANETARY_COMPUTER", "EARTH_SEARCH")
                )

            if (
                final_status == "resolved_weak"
                and candidate is not None
                and not blocking_reasons
            ):
                final_status = "resolved_strong"
                if is_external_source:
                    if "external_source_binding_relaxed" not in soft_reasons:
                        soft_reasons.append("external_source_binding_relaxed")
                else:
                    if "accepted_nonblocking_candidate_as_strong" not in soft_reasons:
                        soft_reasons.append("accepted_nonblocking_candidate_as_strong")

            if final_status in {"resolved_strong", "resolved_weak"} and best_result.get("candidate") is not None:
                entry = await self._build_retrieved_entry(
                    best_result["candidate"],
                    req,
                    dataset_info,
                    modality,
                    retrieval_hint,
                    note=(
                        "resolved_via_family_substitution"
                        if best_strategy == "family_substitution"
                        else "resolved_via_catalog_resolution"
                    ),
                )
                entry["binding_status"] = final_status
                entry["bind_status_reason"] = (
                    "strong_binding_accepted"
                    if final_status == "resolved_strong"
                    else "weak_binding_requires_step_plan_review"
                )
                entry["resolution_source"] = best_strategy
                entry["constraint_report"] = constraint_report
                entry["blocking_reasons"] = blocking_reasons
                entry["verification_gaps"] = verification_gaps
                entry["penalty_report"] = penalty_report
                entry["frozen_constraints_satisfied"] = bool(best_result.get("frozen_constraints_satisfied", False))
                semantic_failure_reason = self._validate_requirement_identity_semantics(
                    intent=search_intent,
                    resolved_entry=entry,
                )
                if semantic_failure_reason:
                    final_status = "unresolved_semantic"
                    blocking_reasons.append(semantic_failure_reason)
                elif final_status == "resolved_strong":
                    resolution_report[requirement_id] = {
                        "status": final_status,
                        "dataset_name": dataset_name,
                        "binding_status": final_status,
                        "bind_status_reason": entry["bind_status_reason"],
                        "resolution_source": best_strategy,
                        "constraint_report": constraint_report,
                        "blocking_reasons": blocking_reasons,
                        "verification_gaps": verification_gaps,
                        "penalty_report": penalty_report,
                        "best_candidate": best_candidate_summary,
                        "search_stage": best_strategy,
                        "online_attempted": False,
                        "failure_status": "",
                        "unbound_class": "",
                        "top_candidates": top_candidates,
                        "frozen_constraints_satisfied": entry["frozen_constraints_satisfied"],
                    }
                    resolved[requirement_id] = entry
                    resolution_log.append({
                        "dataset": dataset_name,
                        "requirement_id": requirement_id,
                        "status": final_status,
                        "strategy": best_strategy,
                        "reasons": blocking_reasons or verification_gaps or soft_reasons,
                        "confidence_flags": constraint_report,
                        "top_candidates": top_candidates,
                        "attempts": attempts,
                    })
                    continue
                else:
                    best_candidate_summary = best_candidate_summary or {
                        "name": entry.get("title") or entry.get("name"),
                        "id": entry.get("id"),
                        "type": entry.get("type"),
                    }
                    if "weak_binding_requires_step_plan_review" not in soft_reasons:
                        soft_reasons.append(
                            "weak_binding_requires_step_plan_review"
                        )

            hard_conflict_reasons = {
                "region_conflict",
                "type_mismatch_hard",
                "family_mismatch_hard",
                "temporal_no_overlap",
                "temporal_coverage_incomplete",
                "temporal_resolution_incompatible",
                "frozen_region_conflict",
                "frozen_modality_mismatch",
                "frozen_candidate_type_mismatch",
            }
            if final_status == "resolved_weak" and verification_gaps:
                unbound_class = "metadata_unverified"
            elif final_status == "resolved_weak":
                unbound_class = "weak_binding"
            elif final_status == "unresolved_semantic":
                unbound_class = (
                    "hard_conflict"
                    if any(reason in hard_conflict_reasons for reason in blocking_reasons)
                    else "semantic_rejection"
                )
            elif "no_candidates" in set(blocking_reasons or best_result.get("reasons", []) or []):
                unbound_class = "no_candidate"
            else:
                unbound_class = "real_unreachable"
            resolution_report[requirement_id] = {
                "status": final_status,
                "dataset_name": dataset_name,
                "binding_status": final_status,
                "bind_status_reason": (
                    (blocking_reasons or verification_gaps)[0]
                    if blocking_reasons or verification_gaps
                    else final_status
                ),
                "resolution_source": best_strategy,
                "constraint_report": constraint_report,
                "blocking_reasons": blocking_reasons,
                "verification_gaps": verification_gaps,
                "penalty_report": penalty_report,
                "best_candidate": best_candidate_summary,
                "search_stage": best_strategy or "search_exhausted",
                "online_attempted": False,
                "failure_status": final_status,
                "unbound_class": unbound_class,
                "top_candidates": top_candidates,
                "frozen_constraints_satisfied": bool(best_result.get("frozen_constraints_satisfied", False)),
            }
            resolution_log.append({
                "dataset": dataset_name,
                "requirement_id": requirement_id,
                "status": final_status,
                "unbound_class": unbound_class,
                "reasons": blocking_reasons or verification_gaps or soft_reasons,
                "confidence_flags": constraint_report,
                "top_candidates": top_candidates,
                "attempts": attempts,
            })

        for req_item in required_datasets:
            if not isinstance(req_item, dict):
                continue
            requirement_id = self._get_required_dataset_id(req_item)
            dataset_name = self._get_required_dataset_name(req_item)
            req_upgraded = self._upgrade_legacy_required_dataset_contract(req_item)
            report = resolution_report.get(requirement_id, {}) or {}
            status = str(report.get("status") or "pending")
            if status == "resolved_strong":
                continue

            unresolvable.append(requirement_id)
            unresolvable_names.append(dataset_name)
            affected_step_ids = [
                step.get("id")
                for step in adjusted_plan
                if isinstance(step, dict)
                and requirement_id in step_required_dataset_ids(step)
            ]
            report["needs_step_plan_replan"] = True
            report["repair_context"] = {
                "affected_requirement": {
                    "requirement_id": requirement_id,
                    "requirement_name": req_upgraded.get("name", ""),
                    "allowed_product_families": req_upgraded.get("allowed_product_families", []) or [],
                    "binding_notes": self._get_requirement_binding_notes(req_upgraded),
                    "aliases": req_upgraded.get("aliases", []) or [],
                },
                "affected_step_ids": affected_step_ids,
                "diagnostics": {
                    "best_candidate_found": report.get("best_candidate"),
                    "blocking_reasons": report.get("blocking_reasons", []),
                    "top_candidates": report.get("top_candidates", []),
                },
                "step_plan_actions": [
                    "keep the requirement only if a resolved_strong binding can be declared",
                    "replace or remove the requirement when the accepted idea semantics permit it",
                    "synchronize required_dataset input references on every affected step",
                ],
                "must_preserve": ["accepted idea semantics"],
            }
            unresolvable_details[requirement_id] = dict(report)

        updated_required_datasets = self._annotate_required_datasets_with_resolution(
            required_datasets=required_datasets,
            resolved=resolved,
            unresolvable_ids=unresolvable,
            unresolvable_details=unresolvable_details,
        )
        adjusted_plan = self._bind_resolved_datasets_to_task_plan(
            task_plan=adjusted_plan,
            resolved=resolved,
        )

        # 写入 dataset_info（仅在此处写入一次，循环内不写入）
        dataset_info["resolved_datasets"] = resolved
        dataset_info["required_datasets"] = updated_required_datasets
        dataset_info["unresolvable_datasets"] = unresolvable
        dataset_info["unresolvable_dataset_names"] = unresolvable_names
        dataset_info["unresolvable_dataset_details"] = unresolvable_details
        dataset_info["dataset_resolution_report"] = resolution_report
        dataset_info["missing_datasets"] = unresolvable        # 保持接口兼容
        dataset_info["resolution_log"] = resolution_log
        dataset_info["retrieved_datasets"] = list(resolved.values())
        dataset_info["available_datasets"] = (
            [entry.get("name", req_id) for req_id, entry in resolved.items()]
            + [dataset_info.get("name") or dataset_info.get("base_dataset", {}).get("name", "")]
        )

        for requirement_id, report in resolution_report.items():
            candidate = report.get("best_candidate") or {}
            logger.info(
                "[ResourceCheck] dataset requirement checked: "
                "requirement_id=%s, status=%s, search_stage=%s, "
                "best_candidate=%s, reasons=%s",
                requirement_id,
                report.get("status"),
                report.get("search_stage"),
                candidate.get("id") or candidate.get("name") or "",
                report.get("blocking_reasons") or report.get("verification_gaps") or [],
            )

        return adjusted_plan, dataset_info, adjustments

    async def _build_retrieved_entry(
        self, best, req: Dict, dataset_info: Dict,
        modality: str, retrieval_hint: str, note: str = ""
    ) -> Dict[str, Any]:
        """构建已解析数据集的标准条目。"""
        dataset_name = self._get_required_dataset_name(req)
        requirement_id = self._get_required_dataset_id(req)
        retrieval_method = "keyword"

        props = getattr(best, "properties", {}) or {}
        dataset_id = props.get("id") or props.get("ee_id") or getattr(best, "feature_id", "")
        # 候选自身的真实 metadata，与下面 name/title 的"缺失则回退到需求名"分开保存。
        # entry["name"] / entry["title"] 是给人看的展示字段，允许回退；身份与区域
        # 判定只能用这里的原始值，否则需求名会变成候选的自证证据（见
        # candidate_identity_text 处的说明）。
        candidate_own_name = str(getattr(best, "name", "") or "")
        candidate_own_title = str(props.get("title") or "")
        entry: Dict[str, Any] = {
            "requirement_id": requirement_id,
            "name": getattr(best, "name", dataset_name),
            "for_requirement": dataset_name,
            "title": props.get("title") or getattr(best, "name", dataset_name),
            "id": dataset_id,
            "type": props.get("type") or getattr(best, "feature_type", ""),
            "ee_id": props.get("ee_id"),
            "ee_object_type": props.get("ee_object_type"),
            "ee_ctor": props.get("ee_ctor"),
            "source": getattr(best, "source", ""),
            # datasets.json 中的真实数据来源，用于验证阶段分发到对应客户端
            "dataset_source": props.get("source") or "",
            "modality": modality,
            "retrieval_method": retrieval_method,
            "geometry": getattr(best, "geometry", None),
            "docs": props.get("docs"),
            "doc_url": props.get("docs"),
            "doi": props.get("doi"),
            "relations": props.get("relations", []),
            "parameters": props.get("parameters", []),
            "metadata_url": props.get("metadata_url"),
            "data_url": props.get("data_url"),
            "is_collection": props.get("is_collection"),
            "stac_url": props.get("stac_url"),
            "thumbnail_url": props.get("thumbnail_url"),
            "terms_of_use": props.get("terms_of_use"),
            "sample_code": props.get("sample_code"),
            "provider": props.get("provider"),
            "tags": props.get("tags", []),
            "thematic_group": props.get("thematic_group"),
            "region_name": props.get("region_name"),
            "bbox": props.get("bbox"),
            "license": props.get("license"),
            "execution_requirement_level": self._get_requirement_execution_requirement_level(req),
            "binding_notes": self._get_requirement_binding_notes(req),
        }
        candidate_tags = entry.get("tags", []) or []
        if isinstance(candidate_tags, str):
            candidate_tags = [candidate_tags]
        # 身份/区域证据只允许来自候选自身的 metadata。此前这里读 entry["name"] 与
        # entry["title"]，而两者在候选缺失该字段时会回退成 dataset_name（需求名），
        # 于是需求里的词被当成候选的属性。实测后果：需求名含 "Global" 时，一个真实
        # metadata 完全没有区域信息的候选，其 region status 从 unspecified 变成
        # global，凭需求名自证了全球覆盖并被判 resolved_strong。
        candidate_identity_text = " ".join(
            str(part)
            for part in [
                candidate_own_name,
                candidate_own_title,
                entry.get("id", ""),
                entry.get("source", ""),
                entry.get("provider", ""),
                entry.get("thematic_group", ""),
                entry.get("region_name", ""),
                " ".join(candidate_tags),
            ]
            if part
        )
        candidate_regions = extract_region_terms(
            candidate_identity_text,
            explicit_region=entry.get("region_name"),
        )
        # 全球覆盖走分层证据：bbox（强）> title/id/tags（中）> description（弱），
        # 且 bbox 存在并明确非全球时否决全部文本声明。此前这里只看 bbox，
        # description 与 keywords 虽已在归一化时保留却从未参与判定——实测 10821 条
        # catalog 记录中可证明全球覆盖的比例因此只有 26.9%，接入后升至 54.4%。
        global_evidence = assess_global_coverage_evidence(
            bbox=entry.get("bbox"),
            short_fields=[
                candidate_own_name,
                candidate_own_title,
                entry.get("id", ""),
                " ".join(str(tag).strip() for tag in candidate_tags if str(tag).strip()),
            ],
            description=props.get("description"),
        )
        if global_evidence["proves_global"]:
            candidate_regions.append("global")
        elif global_evidence["bbox_refutes"]:
            candidate_regions = [
                region for region in candidate_regions
                if str(region).strip().casefold() != "global"
            ]
        entry["candidate_regions"] = list(dict.fromkeys(candidate_regions))
        # 留档判定依据，便于解释"description 写了 global 却未判全球"这类结果。
        entry["global_coverage_evidence"] = global_evidence
        # 供下游身份判定复用的干净证据：不含 name/title 的需求名回退值。
        # `_validate_requirement_identity_semantics` 以此替代重新拼接 entry 字段。
        entry["candidate_identity_evidence"] = candidate_identity_text
        if note:
            entry["note"] = note
        return entry

    def _get_required_dataset_name(self, req_item: Any) -> str:
        """提取结构化数据需求名称。"""
        if not isinstance(req_item, dict):
            raise AgentExecutionError("required dataset must be a structured object")
        return str(req_item.get("name", req_item.get("id", ""))).strip()

    def _get_required_dataset_id(self, req_item: Any) -> str:
        """提取 StepPlan 显式声明的数据需求 ID。"""
        if not isinstance(req_item, dict):
            raise AgentExecutionError("required dataset must be a structured object")
        requirement_id = str(req_item.get("id", "")).strip()
        if not requirement_id:
            raise AgentExecutionError("required dataset must contain a non-empty id")
        return requirement_id

    def _get_requirement_execution_requirement_level(self, req_item: Any) -> str:
        if not isinstance(req_item, dict):
            return ""
        level = str(req_item.get("execution_requirement_level") or "").strip().lower()
        if level == "hard":
            return "required"
        if level == "soft":
            return "optional"
        return level

    def _get_requirement_binding_notes(self, req_item: Any) -> str:
        if not isinstance(req_item, dict):
            return ""
        return str(req_item.get("binding_notes") or "").strip()

    def _upgrade_legacy_required_dataset_contract(self, entry: Any) -> Dict[str, Any]:
        """
        单一边界兼容：notes → binding_notes 等，并删除已废弃的数据角色字段。
        历史 alias `hard/soft` 在边界归一到公开契约值域 `required/optional`。
        """
        if not isinstance(entry, dict):
            return {}
        upgraded = dict(entry)
        if not upgraded.get("binding_notes") and upgraded.get("notes"):
            upgraded["binding_notes"] = upgraded.get("notes")
        level = str(upgraded.get("execution_requirement_level") or "").strip().lower()
        if level == "hard":
            upgraded["execution_requirement_level"] = "required"
        elif level == "soft":
            upgraded["execution_requirement_level"] = "optional"
        elif level not in {"required", "recommended", "optional"}:
            upgraded["execution_requirement_level"] = ""
        upgraded.pop("research_role", None)
        upgraded.pop("question_role", None)
        frozen_constraints = upgraded.get("frozen_constraints")
        if isinstance(frozen_constraints, dict):
            frozen_constraints = dict(frozen_constraints)
            frozen_constraints.pop("research_role", None)
            frozen_constraints.pop("question_role", None)
            upgraded["frozen_constraints"] = frozen_constraints
        return upgraded

    def _is_requirement_mandatory(self, req_item: Any) -> bool:
        level = self._get_requirement_execution_requirement_level(req_item)
        if level == "required":
            return True
        if level in {"recommended", "optional"}:
            return False
        return False

    def _is_primary_dataset_requirement(self, req_item: Dict[str, Any], dataset_info: Dict[str, Any]) -> bool:
        if not isinstance(req_item, dict):
            return False
        # 统一走 plan_utils 的 canonical_id 优先识别（含 identity / name token 兜底），
        # 消除本文件原有的重复 token 匹配（会被 base 简称击穿）。
        if plan_utils.is_base_dataset_requirement(req_item, dataset_info):
            return True
        # 兜底：resourcescheck 自有的 binding_notes 提取（可能处理别名/嵌套结构）。
        notes = self._get_requirement_binding_notes(req_item).lower()
        if "base dataset already available" in notes or "base dataset already available locally" in notes:
            return True
        return False

    def _build_primary_dataset_requirement_entry(
        self,
        req_item: Dict[str, Any],
        dataset_info: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        primary_status = dataset_info.get("primary_data_status", "unknown")
        base_dataset = dataset_info.get("base_dataset") or {}
        primary_path = self._normalize_local_data_path(
            dataset_info.get("data_path") or base_dataset.get("data_path")
        )
        primary_source = dataset_info.get("source") or base_dataset.get("source")
        primary_name = dataset_info.get("name") or base_dataset.get("name") or self._get_required_dataset_name(req_item)
        if not primary_path and not primary_source:
            return None
        if primary_path:
            source_value = f"LOCAL:{primary_path}"
        else:
            source_value = str(primary_source or "").strip()

        return {
            "requirement_id": self._get_required_dataset_id(req_item),
            "name": primary_name,
            "for_requirement": self._get_required_dataset_name(req_item),
            "title": primary_name,
            "id": str(base_dataset.get("id") or dataset_info.get("id") or "primary_dataset"),
            "type": str(base_dataset.get("type") or dataset_info.get("type") or ""),
            "ee_id": base_dataset.get("ee_id") or dataset_info.get("ee_id"),
            "ee_object_type": base_dataset.get("ee_object_type") or dataset_info.get("ee_object_type"),
            "ee_ctor": base_dataset.get("ee_ctor") or dataset_info.get("ee_ctor"),
            "dataset_source": base_dataset.get("dataset_source") or dataset_info.get("dataset_source"),
            "doc_url": base_dataset.get("doc_url") or dataset_info.get("doc_url"),
            "stac_url": base_dataset.get("stac_url") or dataset_info.get("stac_url"),
            "overpass_ql": base_dataset.get("overpass_ql") or dataset_info.get("overpass_ql"),
            "endpoint": base_dataset.get("endpoint") or dataset_info.get("endpoint"),
            "source": source_value,
            "modality": str(req_item.get("modality") or base_dataset.get("modality") or dataset_info.get("modality") or ""),
            "retrieval_method": "provided_primary",
            "geometry": dataset_info.get("geometry") or base_dataset.get("geometry"),
            "docs": base_dataset.get("docs") or dataset_info.get("docs"),
            "sample_code": base_dataset.get("sample_code") or dataset_info.get("sample_code"),
            "provider": base_dataset.get("provider") or dataset_info.get("provider"),
            "tags": base_dataset.get("tags", []) or dataset_info.get("tags", []),
            "thematic_group": base_dataset.get("thematic_group") or dataset_info.get("thematic_group"),
            "license": base_dataset.get("license") or dataset_info.get("license"),
            "execution_requirement_level": self._get_requirement_execution_requirement_level(req_item),
            "binding_notes": self._get_requirement_binding_notes(req_item),
            "binding_status": "provided_primary" if primary_status == "verified" or primary_source else "unbound",
            "bind_status_reason": (
                "primary_dataset_reused"
                if primary_status == "verified" or primary_source
                else f"primary_dataset_status:{primary_status}"
            ),
            "resolution_source": "primary_dataset",
            "constraint_report": {},
            "note": "Requirement matched the provided base dataset and was bound without external search.",
        }

    def _resolve_primary_dataset_requirement(
        self,
        req_item: Dict[str, Any],
        dataset_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not self._is_primary_dataset_requirement(req_item, dataset_info):
            return {"matched": False, "entry": None}
        return {
            "matched": True,
            "entry": self._build_primary_dataset_requirement_entry(req_item, dataset_info),
        }

    @staticmethod
    def _normalize_semantic_text(value: Any) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", str(value or "").lower()))

    @staticmethod
    def _extract_explicit_dataset_identifiers(*values: Any) -> List[str]:
        identifiers: List[str] = []
        seen: set[str] = set()
        pattern = re.compile(r"[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+){1,}")
        for value in values:
            for raw_match in pattern.findall(str(value or "")):
                match = raw_match.rstrip(".,;:!?)]}\"'")
                if not match:
                    continue
                normalized = match.casefold()
                if normalized in seen:
                    continue
                seen.add(normalized)
                identifiers.append(match)
        return identifiers

    @staticmethod
    def _extract_quoted_retrieval_terms(value: Any) -> List[str]:
        terms: List[str] = []
        seen: set[str] = set()
        pattern = re.compile(r"(['\"])([^'\"\r\n]{1,80})\1")
        for match in pattern.finditer(str(value or "")):
            term = " ".join(match.group(2).split()).strip(".,;:!?()[]{}")
            if not term or "/" in term or not re.search(r"[A-Za-z0-9]", term):
                continue
            normalized = term.casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            terms.append(term)
        return terms

    def _tokenize_semantic_text(self, value: Any) -> set[str]:
        return {
            token
            for token in self._normalize_semantic_text(value).split()
            if token and len(token) > 1
        }

    def _build_dataset_search_intent(
        self,
        req: Dict[str, Any],
        dataset_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        dataset_name = self._get_required_dataset_name(req)
        binding_notes = self._get_requirement_binding_notes(req)
        canonical_requirement_name = str(
            req.get("canonical_requirement_name") or dataset_name or ""
        ).strip()
        execution_requirement_level = self._get_requirement_execution_requirement_level(req)
        modality = str(req.get("modality", "") or "").strip()
        temporal_coverage = normalize_temporal_coverage(
            req.get("temporal_coverage")
        )
        temporal_resolution = normalize_temporal_resolution(
            req.get("temporal_resolution")
        )
        aliases = req.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        allowed_product_families = req.get("allowed_product_families") or []
        if isinstance(allowed_product_families, str):
            allowed_product_families = [allowed_product_families]
        frozen_constraints = req.get("frozen_constraints") or {}
        soft_constraints = req.get("soft_constraints") or {}
        explicit_dataset_ids = self._extract_explicit_dataset_identifiers(
            req.get("asset_id"),
            req.get("ee_id"),
            req.get("retrieval_hint"),
            *aliases,
        )
        retrieval_product_terms = self._extract_quoted_retrieval_terms(
            req.get("retrieval_hint")
        )
        base_dataset = dataset_info.get("base_dataset") or {}
        region_name = ""
        if isinstance(base_dataset.get("data_description"), dict):
            base_spatial_coverage = (base_dataset.get("data_description") or {}).get("spatial_coverage")
            if isinstance(base_spatial_coverage, dict):
                region_name = str(base_spatial_coverage.get("region_name") or "").strip()
            elif isinstance(base_spatial_coverage, str):
                region_name = base_spatial_coverage.strip()
        if not region_name and isinstance(dataset_info.get("data_description"), dict):
            region_name = str(
                ((dataset_info.get("data_description") or {}).get("spatial_coverage") or {}).get("region_name") or ""
            ).strip()
        frozen_region = str(
            frozen_constraints.get("region", "")
            if isinstance(frozen_constraints, dict)
            else ""
        ).strip()
        if frozen_region:
            region_terms = extract_region_terms("", explicit_region=frozen_region)
        elif not region_name:
            spatial_extent = dataset_info.get("spatial_extent") or dataset_info.get("task", {}).get("spatial_extent", {})
            region_terms = self._extract_region_terms_from_spatial_info(spatial_extent)
        else:
            region_terms = extract_region_terms("", explicit_region=region_name)

        declared_region_text = " ".join(
            part
            for part in [dataset_name, canonical_requirement_name, binding_notes]
            if part
        )
        declared_region_terms = extract_region_terms(declared_region_text)
        if "global" in declared_region_terms:
            region_terms = ["global"]

        semantic_requirement_text = " ".join(
            part for part in [dataset_name, binding_notes, modality] if part
        ).strip()
        time_terms: List[str] = []
        if temporal_coverage:
            time_terms.append("time_series")
        if temporal_resolution:
            cadence_label = {
                "year": "annual",
                "month": "monthly",
                "day": "daily",
            }.get(temporal_resolution.get("unit"))
            if cadence_label:
                time_terms.append(cadence_label)

        identity_text = self._build_requirement_identity_text(
            {
                "dataset_name": dataset_name,
                "canonical_requirement_name": canonical_requirement_name,
                "modality": modality,
                "preferred_family": str(req.get("preferred_family", "") or "").strip() or None,
                "allowed_product_families": [
                    str(family).strip() for family in allowed_product_families if str(family).strip()
                ],
                "aliases": [str(alias).strip() for alias in aliases if str(alias).strip()],
            }
        )
        inferred_identity = infer_observable_identity(
            " ".join(
                part
                for part in [identity_text, binding_notes]
                if part
            )
        )
        identity_constraints: Dict[str, List[str]] = {}
        unsupported_identity_constraints: List[str] = []
        for dimension in ("entity", "observable", "measure"):
            explicit_value = (
                frozen_constraints.get(dimension)
                if isinstance(frozen_constraints, dict)
                else None
            )
            if explicit_value is None:
                identity_constraints[dimension] = list(
                    inferred_identity.get(dimension, [])
                )
                continue
            # 同时接受裸值/列表与 {"values": [...], "match": ...} 两种写法；
            # match 语义由 GeoDataSearch 的 identity gate 消费，这里只取值做规范化。
            explicit_values, _ = normalize_frozen_identity_dimension(explicit_value)
            identity_constraints[dimension] = [
                self._normalize_semantic_text(value).replace(" ", "_")
                for value in explicit_values
                if self._normalize_semantic_text(value)
            ]
            supported_values = set(
                infer_observable_identity(" ".join(identity_constraints[dimension])).get(
                    dimension,
                    [],
                )
            )
            unsupported_identity_constraints.extend(
                f"{dimension}:{value}"
                for value in identity_constraints[dimension]
                if value not in supported_values
            )
        return {
            "dataset_name": dataset_name,
            "canonical_requirement_name": canonical_requirement_name,
            "query_text": canonical_requirement_name or dataset_name,
            "raw_query": canonical_requirement_name or dataset_name,
            "binding_notes": binding_notes,
            "aliases": [str(alias).strip() for alias in aliases if str(alias).strip()],
            "explicit_dataset_ids": explicit_dataset_ids,
            "retrieval_product_terms": retrieval_product_terms,
            "allowed_product_families": [
                str(family).strip() for family in allowed_product_families if str(family).strip()
            ],
            "preferred_family": str(req.get("preferred_family", "") or "").strip() or None,
            "frozen_constraints": dict(frozen_constraints) if isinstance(frozen_constraints, dict) else {},
            "soft_constraints": dict(soft_constraints) if isinstance(soft_constraints, dict) else {},
            "execution_requirement_level": execution_requirement_level,
            "modality": modality,
            "temporal_coverage": temporal_coverage,
            "temporal_resolution": temporal_resolution,
            "semantic_requirement_text": semantic_requirement_text,
            "region_terms": region_terms,
            "required_region": region_terms[0] if len(region_terms) == 1 else None,
            "time_terms": time_terms,
            "requirement_identity_text": identity_text,
            "identity_constraints": identity_constraints,
            "identity_constraint_unsupported": unsupported_identity_constraints,
            "recall_min_score": self.recall_min_score,
            "weak_bind_min_score": self.weak_bind_min_score,
            "strong_bind_min_score": self.strong_bind_min_score,
        }

    def _build_search_queries_from_intent(self, intent: Dict[str, Any]) -> List[str]:
        dataset_name = str(intent.get("dataset_name", "") or "").strip()
        canonical = str(intent.get("canonical_requirement_name", "") or dataset_name).strip()
        aliases = intent.get("aliases") or []
        families = intent.get("allowed_product_families") or []
        explicit_dataset_ids = intent.get("explicit_dataset_ids") or []
        retrieval_product_terms = intent.get("retrieval_product_terms") or []
        semantic_requirement_text = str(intent.get("semantic_requirement_text", "") or "").strip()
        region_name = str(intent.get("required_region", "") or "").replace("_", " ").strip()

        queries: List[str] = []
        for dataset_id in explicit_dataset_ids:
            dataset_id_text = str(dataset_id).strip()
            if dataset_id_text:
                queries.append(dataset_id_text)

        for product_term in retrieval_product_terms:
            product_text = str(product_term).strip()
            if product_text:
                queries.append(product_text)

        for alias in aliases[:3]:
            alias_text = str(alias).strip()
            if alias_text:
                queries.append(f"{alias_text} {region_name}".strip())
                queries.append(alias_text)

        if dataset_name:
            queries.append(f"{dataset_name} {region_name}".strip())
            queries.append(dataset_name)

        preferred_family = str(intent.get("preferred_family", "") or "").strip()
        if preferred_family:
            queries.append(f"{preferred_family} {region_name}".strip())
            queries.append(preferred_family)
        for family in families[:2]:
            family_text = str(family).strip()
            if family_text:
                queries.append(f"{family_text} {region_name}".strip())
                queries.append(family_text)

        if canonical:
            queries.append(f"{canonical} {region_name}".strip())
            queries.append(canonical)

        normalized_req = self._normalize_semantic_text(semantic_requirement_text)
        if any(token in normalized_req for token in {"drought", "spi", "pdsi", "spei"}):
            queries.append(f"{canonical or dataset_name} drought index spi pdsi {region_name}".strip())
        if any(token in normalized_req for token in {"precipitation", "precip", "rainfall", "temperature", "temp", "climate"}):
            queries.append(f"{canonical or dataset_name} precipitation temperature climate {region_name}".strip())

        deduped: List[str] = []
        seen: set[str] = set()
        for query in queries:
            normalized_query = self._normalize_semantic_text(query)
            if not normalized_query or normalized_query in seen:
                continue
            seen.add(normalized_query)
            deduped.append(query)
        return deduped

    @staticmethod
    def _resolution_candidate_id(result: Dict[str, Any]) -> Optional[str]:
        candidate = result.get("candidate")
        if candidate is None:
            return None
        properties = getattr(candidate, "properties", None)
        if not isinstance(properties, dict) and isinstance(candidate, dict):
            properties = candidate.get("properties") or candidate
        if not isinstance(properties, dict):
            properties = {}
        candidate_id = (
            properties.get("id")
            or properties.get("ee_id")
            or getattr(candidate, "feature_id", None)
        )
        return str(candidate_id) if candidate_id else None

    @staticmethod
    def _resolution_status_rank(status: str) -> int:
        return {
            "resolved_strong": 4,
            "resolved_weak": 3,
            "unresolved_semantic": 2,
            "unresolved_real_missing": 1,
        }.get(str(status or ""), 0)

    @classmethod
    def _resolution_result_rank(cls, result: Dict[str, Any]) -> tuple:
        binding_decision = result.get("binding_decision") or {}
        blocking_reasons = list(
            binding_decision.get("blocking_reasons", result.get("reasons", [])) or []
        )
        verification_gaps = list(binding_decision.get("verification_gaps", []) or [])
        candidate = result.get("candidate")
        properties = getattr(candidate, "properties", None)
        if not isinstance(properties, dict) and isinstance(candidate, dict):
            properties = candidate.get("properties") or candidate
        if not isinstance(properties, dict):
            properties = {}
        try:
            final_score = float(
                properties.get("final_score", properties.get("relevance_score", 0.0))
                or 0.0
            )
        except (TypeError, ValueError):
            final_score = 0.0
        return (
            cls._resolution_status_rank(result.get("status", "")),
            int(bool(binding_decision.get("accepted_for_binding", False))),
            int(bool(result.get("frozen_constraints_satisfied", False))),
            -len(blocking_reasons),
            -len(verification_gaps),
            final_score,
        )

    @classmethod
    def _is_better_resolution_result(
        cls,
        candidate_result: Dict[str, Any],
        current_result: Dict[str, Any],
    ) -> bool:
        return cls._resolution_result_rank(candidate_result) > cls._resolution_result_rank(
            current_result
        )

    def _extract_resolution_constraints(self, binding_notes: str) -> Dict[str, Any]:
        """从 binding_notes 中提取空间分辨率约束关键词"""
        normalized = binding_notes.lower()
        constraints = {
            "require_sub_meter": False,
            "require_vhr": False,
            "require_drone": False,
            "max_resolution_meters": None,
        }

        if any(kw in normalized for kw in ["sub-meter", "submeter", "sub meter", "<1m", "< 1m"]):
            constraints["require_sub_meter"] = True
            constraints["max_resolution_meters"] = 1.0

        if any(kw in normalized for kw in ["vhr", "very high resolution", "very-high-resolution"]):
            constraints["require_vhr"] = True
            if constraints["max_resolution_meters"] is None:
                constraints["max_resolution_meters"] = 5.0

        if any(kw in normalized for kw in ["drone", "uav", "unmanned aerial"]):
            constraints["require_drone"] = True
            if constraints["max_resolution_meters"] is None:
                constraints["max_resolution_meters"] = 2.0

        import re
        meter_pattern = r'(?:less than|<|under|below)\s*(\d+\.?\d*)\s*m(?:eter)?s?'
        matches = re.findall(meter_pattern, normalized)
        if matches:
            explicit_limit = float(matches[0])
            if constraints["max_resolution_meters"] is None or explicit_limit < constraints["max_resolution_meters"]:
                constraints["max_resolution_meters"] = explicit_limit

        return constraints

    def _extract_candidate_resolution_meters(self, entry: Dict[str, Any]) -> Optional[float]:
        """从候选 entry 中提取空间分辨率（米）"""
        spatial_resolution = entry.get("spatial_resolution")
        if not spatial_resolution:
            return None

        if isinstance(spatial_resolution, dict):
            meters = spatial_resolution.get("resolution_meters")
            if meters is not None:
                return float(meters)

        if isinstance(spatial_resolution, str):
            import re
            match = re.search(r'(\d+\.?\d*)\s*m(?:eter)?s?', spatial_resolution.lower())
            if match:
                return float(match.group(1))

        return None

    def _validate_resolution_constraints(
        self,
        candidate_entry: Dict[str, Any],
        constraints: Dict[str, Any],
        binding_notes: str,
    ) -> Optional[str]:
        """验证候选数据集是否满足自然语言约束"""
        max_allowed = constraints.get("max_resolution_meters")
        if max_allowed is None:
            return None

        candidate_resolution = self._extract_candidate_resolution_meters(candidate_entry)
        if candidate_resolution is None:
            return None

        if candidate_resolution > max_allowed:
            require_types = []
            if constraints.get("require_sub_meter"):
                require_types.append("sub-meter")
            if constraints.get("require_drone"):
                require_types.append("drone")
            if constraints.get("require_vhr"):
                require_types.append("VHR")

            type_text = "/".join(require_types) if require_types else f"<={max_allowed}m"
            return (
                f"spatial_resolution_incompatible: requirement specifies {type_text} "
                f"({binding_notes}), but candidate has {candidate_resolution}m resolution"
            )

        return None

    async def _validate_resolution_candidate_identity(
        self,
        result: Dict[str, Any],
        *,
        req: Dict[str, Any],
        dataset_info: Dict[str, Any],
        modality: str,
        retrieval_hint: str,
        search_intent: Dict[str, Any],
    ) -> Dict[str, Any]:
        if result.get("status") not in {"resolved_strong", "resolved_weak"}:
            return result
        if result.get("candidate") is None:
            return result

        binding_decision = dict(result.get("binding_decision", {}) or {})
        entry = await self._build_retrieved_entry(
            result["candidate"],
            req,
            dataset_info,
            modality,
            retrieval_hint,
        )
        entry["constraint_report"] = dict(binding_decision.get("constraint_report", {}) or {})
        semantic_failure = self._validate_requirement_identity_semantics(
            intent=search_intent,
            resolved_entry=entry,
        )
        if not semantic_failure:
            binding_notes = str(search_intent.get("binding_notes", "") or "").strip()
            if binding_notes:
                resolution_constraints = self._extract_resolution_constraints(binding_notes)
                constraint_failure = self._validate_resolution_constraints(
                    candidate_entry=entry,
                    constraints=resolution_constraints,
                    binding_notes=binding_notes,
                )
                if constraint_failure:
                    semantic_failure = constraint_failure

        if not semantic_failure:
            return result

        rejected = dict(result)
        rejected["status"] = "unresolved_semantic"
        rejected_reasons = list(result.get("reasons", []) or [])
        if semantic_failure not in rejected_reasons:
            rejected_reasons.append(semantic_failure)
        rejected["reasons"] = rejected_reasons
        blocking_reasons = list(binding_decision.get("blocking_reasons", []) or [])
        if semantic_failure not in blocking_reasons:
            blocking_reasons.append(semantic_failure)
        binding_decision["accepted_for_binding"] = False
        binding_decision["binding_grade"] = "unresolved_semantic"
        binding_decision["blocking_reasons"] = blocking_reasons
        rejected["binding_decision"] = binding_decision
        return rejected

    async def _try_family_substitution(
        self,
        req: Dict[str, Any],
        search_intent: Dict[str, Any],
        search_limit: int,
        *,
        dataset_info: Dict[str, Any],
        modality: str,
        retrieval_hint: str,
    ) -> Optional[Dict[str, Any]]:
        families = req.get("allowed_product_families") or search_intent.get("allowed_product_families") or []
        if isinstance(families, str):
            families = [families]
        frozen_constraints = search_intent.get("frozen_constraints") or {}
        region = str(
            frozen_constraints.get("region")
            or search_intent.get("required_region")
            or ""
        ).replace("_", " ").strip()

        for family in families:
            family_text = str(family).strip()
            if not family_text:
                continue
            query = f"{family_text} {region}".strip()
            attempt_intent = {
                **search_intent,
                "query_text": query,
                "raw_query": query,
                "search_k": search_limit,
            }
            local_result = self.geo_search.resolve_local_dataset(attempt_intent)
            candidate_result = local_result
            if (
                self.allow_online_gee_fallback
                and candidate_result.get("status") not in {"resolved_strong", "resolved_weak"}
            ):
                online_result = await self.geo_search.resolve_online_dataset(attempt_intent)
                if self._is_better_resolution_result(online_result, candidate_result):
                    candidate_result = online_result
            candidate_result = await self._validate_resolution_candidate_identity(
                candidate_result,
                req=req,
                dataset_info=dataset_info,
                modality=modality,
                retrieval_hint=retrieval_hint,
                search_intent=search_intent,
            )
            if candidate_result.get("status") not in {"resolved_strong", "resolved_weak"}:
                continue
            if not candidate_result.get("frozen_constraints_satisfied", False):
                continue
            candidate_result = dict(candidate_result)
            candidate_result["family_substitution_query"] = query
            return candidate_result
        return None

    async def _llm_rewrite_resolution_query(
        self,
        intent: Dict[str, Any],
        rejected_reasons: Optional[List[str]] = None,
    ) -> List[str]:
        """
        当严格本地解析失败时，用 LLM 把科学语义改写成更像真实产品族的查询词。
        输入是 requirement 的科学语义和拒绝原因，输出 1-3 个不同 family 的产品词查询。
        """
        dataset_name = str(intent.get("dataset_name", "") or "").strip()
        binding_notes = str(intent.get("binding_notes", "") or "").strip()
        modality = str(intent.get("modality", "") or "").strip()
        semantic_requirement_text = str(intent.get("semantic_requirement_text", "") or "").strip()
        required_region = str(intent.get("required_region", "") or "").replace("_", " ").strip()
        rejected_reasons = rejected_reasons or []
        rejection_block = ""
        if rejected_reasons:
            rejection_text = "\n".join(f"- {reason}" for reason in rejected_reasons)
            rejection_block = (
                "Previously resolved candidates were rejected for these reasons:\n"
                f"{rejection_text}\n"
                "Generate queries that would find a DIFFERENT dataset family that avoids these issues.\n\n"
            )

        prompt = (
            "You are a GEE (Google Earth Engine) catalog expert.\n"
            "Given a dataset requirement that failed strict local resolution, translate it into "
            "1-3 specific GEE dataset product names or keywords likely to exist in the catalog.\n"
            "Focus on actual product families (e.g. TERRACLIMATE, GRIDMET, MODIS, ERA5, PRISM), "
            "not academic terminology.\n"
            "Generate queries that would find a DIFFERENT dataset family from previously rejected candidates.\n"
            "Do not embed research-layer role labels (driver/context/validation, etc.) into the output queries.\n\n"
            f"Dataset requirement name: {dataset_name}\n"
            f"Identity-oriented requirement text: {semantic_requirement_text}\n"
            f"Binding notes: {binding_notes}\n"
            f"Modality: {modality}\n"
            f"Target region: {required_region}\n\n"
            f"{rejection_block}"
            'Output JSON: {"queries": ["<product_name_1>", "<product_name_2>"]}'
        )
        schema = {
            "type": "object",
            "properties": {
                "queries": {"type": "array", "items": {"type": "string"}, "maxItems": 3}
            },
            "required": ["queries"],
        }
        try:
            result = await self._call_model(prompt=prompt, schema=schema, temperature=0.2)
            queries = result.get("queries", [])
            deduped_queries: List[str] = []
            seen_queries: set[str] = set()
            for query in queries:
                sanitized_query = str(query).strip()
                normalized_query = self._normalize_semantic_text(sanitized_query)
                if not normalized_query or normalized_query in seen_queries:
                    continue
                seen_queries.add(normalized_query)
                deduped_queries.append(sanitized_query)
            return deduped_queries
        except Exception as e:
            logger.warning(f"[ResourceCheck] LLM query rewrite failed for {dataset_name}: {e}")
            return []

    def _build_requirement_identity_text(self, intent: Dict[str, Any]) -> str:
        if not isinstance(intent, dict):
            return ""
        identity_parts: List[str] = [
            str(intent.get("dataset_name", "") or "").strip(),
            str(intent.get("canonical_requirement_name", "") or "").strip(),
            str(intent.get("modality", "") or "").strip(),
            str(intent.get("preferred_family", "") or "").strip(),
        ]
        for family in (intent.get("allowed_product_families") or []):
            family_text = str(family).strip()
            if family_text:
                identity_parts.append(family_text)
        for alias in (intent.get("aliases") or [])[:5]:
            alias_text = str(alias).strip()
            if alias_text:
                identity_parts.append(alias_text)
        return " ".join(part for part in identity_parts if part)

    def _validate_requirement_identity_semantics(
        self,
        intent: Dict[str, Any],
        resolved_entry: Dict[str, Any],
    ) -> str:
        req_text = self._build_requirement_identity_text(intent)
        candidate_tags = resolved_entry.get("tags", []) or []
        if isinstance(candidate_tags, str):
            candidate_tags = [candidate_tags]
        # 优先使用 _build_retrieved_entry 留下的干净证据。回退到重新拼接时，
        # name/title 可能是需求名的回退值——这会让需求文本变成候选的自证证据，
        # 因此拼接分支只在 entry 未经 _build_retrieved_entry 处理时使用。
        cand_text = str(resolved_entry.get("candidate_identity_evidence") or "").strip()
        if not cand_text:
            cand_text = " ".join(
                str(part)
                for part in [
                    resolved_entry.get("name", ""),
                    resolved_entry.get("title", ""),
                    resolved_entry.get("id", ""),
                    resolved_entry.get("source", ""),
                    resolved_entry.get("thematic_group", ""),
                    resolved_entry.get("provider", ""),
                    " ".join(candidate_tags),
                ]
                if part
            )
        req_tokens = self._tokenize_semantic_text(req_text)
        cand_tokens = self._tokenize_semantic_text(cand_text)

        candidate_regions = resolved_entry.get("candidate_regions") or extract_region_terms(
            cand_text,
            explicit_region=resolved_entry.get("region_name"),
        )
        for required_region in (intent.get("region_terms") or []):
            region_match = classify_region_match(required_region, candidate_regions)
            region_status = str(region_match.get("status") or "unspecified")
            # 全球覆盖要求下，只有"候选明确声明了不兼容的区域"才是语义拒绝。
            # 候选没有任何区域信息（status == "unspecified"）属于 metadata 缺口：
            # 既未证明也未证伪全球覆盖，不构成 hard conflict。真实案例是 GEBCO
            # 全球水深格网——全球覆盖是其定义属性，却因 catalog 未写 region 被
            # 判 "does not prove required global spatial coverage" 而杀掉整个计划。
            # 该缺口由 _build_verification_gaps 记为 global_coverage_unverified。
            if (
                normalize_region_name(required_region) == "global"
                and region_status not in {"global", "unspecified"}
            ):
                return "resolved candidate does not prove required global spatial coverage"
            if region_status == "conflict":
                return "resolved candidate spatial coverage conflicts with the requested region"

        explicit_dataset_ids = {
            str(dataset_id).strip().casefold()
            for dataset_id in (intent.get("explicit_dataset_ids") or [])
            if str(dataset_id).strip()
        }
        candidate_dataset_ids = {
            str(dataset_id).strip().casefold()
            for dataset_id in [resolved_entry.get("id"), resolved_entry.get("ee_id")]
            if str(dataset_id or "").strip()
        }
        explicit_id_matched = bool(
            explicit_dataset_ids.intersection(candidate_dataset_ids)
        )

        req_is_climate = is_climate_identity_text(req_text)
        req_is_land_cover = is_land_cover_identity_text(req_text)
        candidate_is_climate = is_climate_identity_text(cand_text)
        candidate_is_land_cover = is_land_cover_identity_text(cand_text)
        req_concepts = set(infer_identity_concepts(req_text))
        candidate_concepts = set(infer_identity_concepts(cand_text))
        candidate_observable_identity = infer_observable_identity(cand_text)
        for dimension, required_values in (
            intent.get("identity_constraints") or {}
        ).items():
            normalized_required = {
                self._normalize_semantic_text(value).replace(" ", "_")
                for value in (required_values or [])
                if self._normalize_semantic_text(value)
            }
            if not normalized_required:
                continue
            supported_required = set(
                infer_observable_identity(" ".join(normalized_required)).get(
                    dimension,
                    [],
                )
            )
            if not supported_required:
                continue
            candidate_values = {
                self._normalize_semantic_text(value).replace(" ", "_")
                for value in candidate_observable_identity.get(dimension, [])
                if self._normalize_semantic_text(value)
            }
            if not candidate_values:
                if explicit_id_matched:
                    continue
                return (
                    f"resolved candidate does not prove required {dimension} identity: "
                    f"{sorted(normalized_required)}"
                )
            if supported_required.isdisjoint(candidate_values):
                return (
                    f"resolved candidate {dimension} identity conflicts with requirement: "
                    f"required={sorted(supported_required)}, candidate={sorted(candidate_values)}"
                )

        if (
            req_is_climate
            and not req_is_land_cover
            and candidate_is_land_cover
            and not candidate_is_climate
        ):
            return "resolved candidate is a land-cover product but requirement is climate-like"
        if (
            req_is_land_cover
            and not req_is_climate
            and candidate_is_climate
            and not candidate_is_land_cover
        ):
            return "resolved candidate is a climate product but requirement is land-cover-like"

        if (
            "climate_classification" in req_concepts
            and "climate_classification" not in candidate_concepts
        ):
            return "resolved candidate does not provide the requested climate classification identity"
        if (
            "categorical_variable" in req_concepts
            and "continuous_aggregate" in candidate_concepts
        ):
            return "resolved candidate is a continuous aggregate, not the requested categorical variable"
        if "lithology_category" in req_concepts:
            if (
                "bedrock_depth" in candidate_concepts
                and "lithology_category" not in candidate_concepts
            ):
                return "resolved candidate measures bedrock depth, not categorical lithology"
            if "lithology_category" not in candidate_concepts:
                return "resolved candidate does not provide the requested categorical lithology identity"

        if explicit_id_matched:
            return ""

        allowed_families = intent.get("allowed_product_families") or []
        known_families = [
            family for family in allowed_families if is_known_product_family(family)
        ]
        family_matches = detect_family_matches(intent, cand_text)
        if (
            allowed_families
            and len(known_families) == len(allowed_families)
            and not family_matches
        ):
            return "resolved candidate does not match any requested known product family"

        if not family_matches and req_tokens:
            informative_req_tokens = req_tokens - _IDENTITY_GENERIC_TOKENS
            informative_candidate_tokens = cand_tokens - _IDENTITY_GENERIC_TOKENS
            overlap = informative_req_tokens.intersection(informative_candidate_tokens)
            if not overlap:
                return "resolved candidate lacks dataset identity overlap with the requested requirement identity"

        return ""

    def _extract_region_terms_from_spatial_info(self, spatial_info: Any) -> List[str]:
        if not isinstance(spatial_info, dict):
            return []

        extracted_terms: List[str] = []
        for key in ["region_name", "region", "name", "location", "area", "description"]:
            value = spatial_info.get(key)
            if not isinstance(value, str) or not value.strip():
                continue
            normalized_value = normalize_region_name(value)
            if normalized_value:
                extracted_terms.append(normalized_value)
            extracted_terms.extend(extract_region_terms("", explicit_region=value))

        deduped_terms: List[str] = []
        seen_terms = set()
        for term in extracted_terms:
            if not term or term in seen_terms:
                continue
            seen_terms.add(term)
            deduped_terms.append(term)
        return deduped_terms

    def _detect_semantic_resolution_mismatch(
        self,
        req: Dict[str, Any],
        resolved_entry: Dict[str, Any],
        dataset_info: Dict[str, Any],
    ) -> str:
        compat_intent = self._build_dataset_search_intent(req, dataset_info)
        return self._validate_requirement_identity_semantics(
            intent=compat_intent,
            resolved_entry=resolved_entry,
        )

    def _annotate_required_datasets_with_resolution(
        self,
        required_datasets: List[Any],
        resolved: Dict[str, Dict[str, Any]],
        unresolvable_ids: List[str],
        unresolvable_details: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[Any]:
        """将 ResourcesCheck 的解析结果写回 required_datasets。"""
        annotated: List[Any] = []
        unresolved_set = set(unresolvable_ids)
        unresolvable_details = unresolvable_details or {}

        for req_item in required_datasets:
            if not isinstance(req_item, dict):
                annotated.append(req_item)
                continue

            req_copy = self._upgrade_legacy_required_dataset_contract(req_item)
            requirement_id = self._get_required_dataset_id(req_copy)
            resolved_entry = resolved.get(requirement_id)

            if resolved_entry:
                req_copy["resolution_status"] = "resolved_strong"
                req_copy["resolved_dataset"] = dict(resolved_entry)
            elif requirement_id in unresolved_set:
                detail = unresolvable_details.get(requirement_id, {}) or {}
                req_copy["resolution_status"] = str(detail.get("binding_status") or "unbound")
                req_copy["resolved_dataset"] = None
                req_copy["dataset_resolution_failure"] = dict(detail)
            else:
                req_copy["resolution_status"] = "pending"
            annotated.append(req_copy)

        return annotated

    def _build_dataset_resolution_failures(self, dataset_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """汇总未绑定 requirement 的结构化失败诊断，供 workflow / refinement / replanning 复用。"""
        failures: List[Dict[str, Any]] = []
        unresolvable_ids = dataset_info.get("unresolvable_datasets", []) or []
        unresolvable_details = dataset_info.get("unresolvable_dataset_details", {}) or {}

        for requirement_id in unresolvable_ids:
            detail = dict(unresolvable_details.get(requirement_id, {}) or {})
            failures.append(
                {
                    "requirement_id": requirement_id,
                    "dataset_name": detail.get("dataset_name", requirement_id),
                    "binding_status": detail.get("binding_status", "unbound"),
                    "bind_status_reason": detail.get("bind_status_reason", ""),
                    "resolution_source": detail.get("resolution_source", ""),
                    "constraint_report": detail.get("constraint_report", {}),
                    "blocking_reasons": detail.get("blocking_reasons", []),
                    "best_candidate": detail.get("best_candidate"),
                    "search_stage": detail.get("search_stage", ""),
                    "online_attempted": bool(detail.get("online_attempted", False)),
                    "failure_status": detail.get("failure_status", ""),
                    "unbound_class": detail.get("unbound_class", ""),
                    "needs_step_plan_replan": bool(detail.get("needs_step_plan_replan", False)),
                    "repair_context": detail.get("repair_context"),
                }
            )

        return failures

    def _bind_resolved_datasets_to_task_plan(
        self,
        task_plan: List[Dict[str, Any]],
        resolved: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Bind strong results only to explicit required_dataset DAG inputs."""
        if not isinstance(task_plan, list):
            return task_plan

        bound_plan: List[Dict[str, Any]] = []
        for step in task_plan:
            step_copy = dict(step)
            normalized_required_ids = step_required_dataset_ids(step_copy)
            step_copy["resolved_dataset_bindings"] = [
                dict(resolved[req_id])
                for req_id in normalized_required_ids
                if req_id in resolved
            ]
            bound_plan.append(step_copy)

        return bound_plan

    def _resolve_search_backend(self, req: Optional[Dict[str, Any]]) -> Optional[str]:
        """
        按 required_dataset.source_preference 解析目标后端。

        说明：
        - 仅在 ResourcesCheck 内部封装,不改变外部接口。
        - 未匹配到明确后端时返回 None,交由 GeoDataSearch 默认策略处理。
        - STAC 现在是独立后端,可以路由到专门的 STAC catalog 检索路径(如果
          GeoDataSearch 实现了的话),或先由 auto 兜底、待后续接入真实 STAC 检索。
        """
        if not isinstance(req, dict):
            return None

        source_preference = str(req.get("source_preference", "")).strip().lower()
        backend_mapping = {
            "pangaea": "pangaea",
            "gee": "gee",
            "osm": "osm",
            "openstreetmap": "osm",
            "stac": "stac",
        }
        return backend_mapping.get(source_preference)

    async def _search_by_routed_backend(
        self,
        keyword: str,
        req: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[Any]:
        """
        统一检索入口：按 required_dataset.source_preference 路由检索后端。

        说明：
        - `gee` -> 仅走 GEE catalog 检索
        - `osm` / `openstreetmap` -> 仅走 OSM metadata 检索
        - 未声明或无法映射 -> 交由 GeoDataSearch 默认策略自动选择
        """
        backend = self._resolve_search_backend(req)
        source_preference = None
        if isinstance(req, dict):
            raw_source_preference = str(req.get("source_preference", "")).strip()
            if raw_source_preference:
                source_preference = raw_source_preference

        return await self.geo_search.search_by_keyword(
            keyword=keyword,
            limit=limit,
            backend=backend,
            source_preference=source_preference,
        )

    def _step_needs_any(
        self,
        step: Dict,
        missing_requirement_ids: Optional[List[str]] = None,
    ) -> bool:
        """判断步骤是否显式引用给定 requirement IDs。"""
        required_ids = step_required_dataset_ids(step)
        return bool(
            set(str(item) for item in required_ids)
            & set(missing_requirement_ids or [])
        )

    def _strip_manifest_source_prefix(self, value: str, prefix: str) -> str:
        """去除 manifest source 中重复前缀。"""
        normalized_value = str(value or "").strip()
        normalized_prefix = str(prefix or "").strip()
        while normalized_prefix and normalized_value.upper().startswith(normalized_prefix.upper()):
            normalized_value = normalized_value[len(normalized_prefix):].strip()
        return normalized_value

    def _build_remote_manifest_source(self, source_value: str, prefix: str) -> str:
        """按统一前缀格式构造远程 manifest source。"""
        source_body = self._strip_manifest_source_prefix(source_value, prefix)
        return f"{prefix}{source_body}" if source_body else prefix.rstrip(":")

    # GEE 资产可能带的 manifest source 前缀：catalog 直读（按来源细分）、
    # 在线 catalog 回退、alias 命中，以及历史上的裸 GEE: 形式。
    _GEE_SOURCE_PREFIXES = (
        "GEE_OFFICIAL:",
        "GEE_COMMUNITY:",
        "GEE_ONLINE:",
        "GEE_ALIAS:",
        "GEE:",
    )

    def _strip_gee_source_prefix(self, source_value: str) -> str:
        """从 GEE 类 manifest source 中取出 Earth Engine asset id。"""
        normalized = str(source_value or "").strip()
        for prefix in self._GEE_SOURCE_PREFIXES:
            if normalized.upper().startswith(prefix):
                return normalized[len(prefix):].strip()
        return normalized

    def _create_gee_manifest_item(
        self,
        requirement_id: str,
        mandatory: bool,
        dataset_title: str,
        ds_entry: Dict[str, Any],
        geometry: Any,
        note: str,
    ) -> Dict[str, Any]:
        """根据已解析 GEE 数据集构造显式 manifest 项。"""
        ee_id = str(ds_entry.get("ee_id") or ds_entry.get("id") or "").strip()
        asset_type = str(ds_entry.get("type", "") or "").strip()
        # ds_entry.type 可能是来源类别（gee_dataset）而非 asset 类型，
        # 这里统一解析出可用于探测的构造器，避免下游按来源类别猜 asset 形态。
        ee_ctor = self._resolve_gee_ctor(ds_entry, ee_id)
        ee_object_type = ee_ctor.split(".", 1)[1] if ee_ctor else ds_entry.get("ee_object_type")
        return {
            "requirement_id": requirement_id,
            "mandatory": mandatory,
            "dataset_title": dataset_title,
            "source": self._build_remote_manifest_source(
                ee_id or self._strip_gee_source_prefix(ds_entry.get("source", "")),
                "GEE:",
            ),
            "access_mode": "direct_read",
            "bind_status": "bindable",
            "binding_status": str(ds_entry.get("binding_status") or "resolved_strong"),
            "bind_status_reason": str(ds_entry.get("bind_status_reason") or "strong_binding_accepted"),
            "resolution_source": str(ds_entry.get("resolution_source") or "local_strict_resolution"),
            "constraint_report": dict(ds_entry.get("constraint_report", {}) or {}),
            "local_stage_path": None,
            "spatial_coverage": geometry,
            "resource_role": "input",
            "id": str(ds_entry.get("id") or ee_id or "").strip() or None,
            "type": asset_type or None,
            "catalog_type": asset_type or None,
            "ee_id": ee_id or None,
            "ee_object_type": ee_object_type,
            "ee_ctor": ee_ctor,
            "docs": ds_entry.get("docs"),
            "sample_code": ds_entry.get("sample_code"),
            "provider": ds_entry.get("provider"),
            "tags": ds_entry.get("tags", []),
            "thematic_group": ds_entry.get("thematic_group"),
            "connectivity_probe": None,
            "note": note,
        }

    # 由 datasets.json 导入的非 GEE 数据源集合。
    # 只收录能在不下载数据的前提下确认数据集存在的源；Geofabrik PBF、
    # WorldClim zip 这类纯静态下载镜像不在此列（它们必须下载才能确认，
    # 不适合作为检索候选，已从 datasets.json 中剔除）。
    _EXTERNAL_DATASET_SOURCES: frozenset[str] = frozenset({
        "PANGAEA",
        "NASA_CMR",
        "PLANETARY_COMPUTER",
        "EARTH_SEARCH",
        "API",
        "OSM_TAGINFO",
        "OSM_OVERPASS",
    })

    def _create_external_manifest_item(
        self,
        requirement_id: str,
        mandatory: bool,
        dataset_title: str,
        ds_entry: Dict[str, Any],
        geometry: Any,
        note: str,
    ) -> Dict[str, Any]:
        """根据已解析的非 GEE 外部数据集构造显式 manifest 项。"""
        dataset_source = str(ds_entry.get("dataset_source") or "").strip().upper()
        dataset_id = str(ds_entry.get("id") or "").strip()
        source_prefix = f"{dataset_source}:" if dataset_source else "REMOTE:"
        return {
            "requirement_id": requirement_id,
            "mandatory": mandatory,
            "dataset_title": dataset_title,
            "source": self._build_remote_manifest_source(dataset_id or ds_entry.get("source", ""), source_prefix),
            "access_mode": "direct_read",
            "bind_status": "bindable",
            "binding_status": str(ds_entry.get("binding_status") or "resolved_strong"),
            "bind_status_reason": str(ds_entry.get("bind_status_reason") or "strong_binding_accepted"),
            "resolution_source": str(ds_entry.get("resolution_source") or "external_catalog_resolution"),
            "constraint_report": dict(ds_entry.get("constraint_report", {}) or {}),
            "local_stage_path": None,
            "spatial_coverage": geometry,
            "resource_role": "input",
            "id": dataset_id or None,
            "type": ds_entry.get("type") or None,
            "dataset_source": dataset_source,
            "doc_url": ds_entry.get("doc_url") or ds_entry.get("docs"),
            "doi": ds_entry.get("doi"),
            "relations": ds_entry.get("relations", []),
            "parameters": ds_entry.get("parameters", []),
            "metadata_url": ds_entry.get("metadata_url"),
            "data_url": ds_entry.get("data_url"),
            "is_collection": ds_entry.get("is_collection"),
            "stac_url": ds_entry.get("stac_url"),
            "thumbnail_url": ds_entry.get("thumbnail_url"),
            "terms_of_use": ds_entry.get("terms_of_use"),
            "provider": ds_entry.get("provider"),
            "tags": ds_entry.get("tags", []),
            "thematic_group": ds_entry.get("thematic_group"),
            "overpass_ql": ds_entry.get("overpass_ql"),
            "endpoint": ds_entry.get("endpoint"),
            "connectivity_probe": None,
            "note": note,
        }

    def _build_data_access_manifest(
        self,
        task_plan: List[Dict],
        dataset_info: Dict,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Commit 1: 构建数据访问清单（data_access_manifest）。

        为每个数据需求生成 manifest 项，标记其绑定状态：
        - requirement_id: 数据需求标识
        - mandatory: 是否必需（True/False）
        - source: 数据源类型（LOCAL:/path, GEE:collection, API:endpoint）
        - access_mode: 访问模式（direct_read, stage_on_demand）
        - bind_status: 绑定状态（bindable, unbound）
        - local_stage_path: 本地暂存路径（可空）
        - spatial_coverage: 空间覆盖范围（用于一致性检查）
        """
        manifest: List[Dict[str, Any]] = []

        # 1. 主数据集
        primary_status = dataset_info.get("primary_data_status", "unknown")
        base_dataset = dataset_info.get("base_dataset") or {}
        primary_path = self._normalize_local_data_path(
            dataset_info.get("data_path") or base_dataset.get("data_path")
        )
        primary_source = dataset_info.get("source") or base_dataset.get("source")
        primary_name = dataset_info.get("name") or base_dataset.get("name", "primary")
        primary_geometry = dataset_info.get("geometry") or base_dataset.get("geometry")
        primary_entry = {
            **base_dataset,
            **dataset_info,
        }
        composite_components = (
            [component for component in (base_dataset.get("dataset_components") or []) if isinstance(component, dict)]
            if base_dataset.get("type") == "composite_dataset"
            else []
        )
        if composite_components:
            first_component = composite_components[0]
            primary_path = self._normalize_local_data_path(first_component.get("data_path") or "")
            primary_source = first_component.get("source") or ""
            primary_name = first_component.get("name") or primary_name
            primary_geometry = first_component.get("geometry")
            primary_entry = first_component

        if primary_path:
            manifest.append({
                "requirement_id": "primary_dataset",
                "mandatory": True,
                "source": f"LOCAL:{primary_path}",
                "access_mode": "direct_read",
                "bind_status": "bindable" if primary_status == "verified" else "unbound",
                "binding_status": "provided_primary" if primary_status == "verified" else "unbound",
                "bind_status_reason": (
                    "primary_dataset_verified"
                    if primary_status == "verified"
                    else f"primary_dataset_status:{primary_status}"
                ),
                "resolution_source": "primary_dataset",
                "constraint_report": {},
                "local_stage_path": primary_path if primary_status == "verified" else None,
                "spatial_coverage": primary_geometry,
                "resource_role": "input",
                "connectivity_probe": None,  # Commit 2: 本地数据无需探测
                "note": f"Primary dataset: {primary_name}"
            })
        elif primary_source:
            # 在线数据源（GEE/API）
            source_type = primary_source
            primary_dataset_source = str(primary_entry.get("dataset_source") or "").strip().upper()
            if primary_dataset_source in self._EXTERNAL_DATASET_SOURCES:
                manifest.append(self._create_external_manifest_item(
                    requirement_id="primary_dataset",
                    mandatory=True,
                    dataset_title=primary_name,
                    ds_entry=primary_entry,
                    geometry=primary_geometry,
                    note=f"Primary dataset from online source: {primary_name}",
                ))
            elif "GEE" in source_type.upper() or "EARTH ENGINE" in source_type.upper():
                manifest.append(self._create_gee_manifest_item(
                    requirement_id="primary_dataset",
                    mandatory=True,
                    dataset_title=primary_name,
                    ds_entry=primary_entry,
                    geometry=primary_geometry,
                    note=f"Primary dataset from online source: {primary_name}",
                ))
            elif "API" in source_type.upper():
                manifest.append({
                    "requirement_id": "primary_dataset",
                    "mandatory": True,
                    "dataset_title": primary_name,
                    "source": self._build_remote_manifest_source(source_type, "API:"),
                    "access_mode": "direct_read",
                    "bind_status": "bindable",
                    "binding_status": "provided_primary",
                    "bind_status_reason": "primary_dataset_online_source",
                    "resolution_source": "primary_dataset",
                    "constraint_report": {},
                    "local_stage_path": None,
                    "spatial_coverage": primary_geometry,
                    "resource_role": "input",
                    "connectivity_probe": None,  # Commit 2: 待探测
                    "note": f"Primary dataset from online source: {primary_name}"
                })
            else:
                manifest.append({
                    "requirement_id": "primary_dataset",
                    "mandatory": True,
                    "dataset_title": primary_name,
                    "source": self._build_remote_manifest_source(source_type, "REMOTE:"),
                    "access_mode": "direct_read",
                    "bind_status": "bindable",
                    "binding_status": "provided_primary",
                    "bind_status_reason": "primary_dataset_online_source",
                    "resolution_source": "primary_dataset",
                    "constraint_report": {},
                    "local_stage_path": None,
                    "spatial_coverage": primary_geometry,
                    "resource_role": "input",
                    "connectivity_probe": None,  # Commit 2: 待探测
                    "note": f"Primary dataset from online source: {primary_name}"
                })
        else:
            manifest.append({
                "requirement_id": "primary_dataset",
                "mandatory": True,
                "dataset_title": primary_name,
                "source": "UNRESOLVED:primary_dataset",
                "access_mode": "direct_read",
                "bind_status": "unbound",
                "binding_status": "unbound",
                "bind_status_reason": "primary_dataset_missing_source",
                "resolution_source": "primary_dataset",
                "constraint_report": {},
                "local_stage_path": None,
                "spatial_coverage": primary_geometry,
                "resource_role": "input",
                "connectivity_probe": None,
                "note": (
                    "Primary dataset is missing both local data_path and online source. "
                    "Resource check must fail closed."
                ),
            })

        # 1b. Composite anchors expose every declared component as a mandatory input.
        for index, component in enumerate(composite_components[1:], start=2):
            requirement_id = f"anchor_component_{index}"
            component_source = str(component.get("source") or "").strip()
            component_path = str(component.get("data_path") or "").strip()
            component_name = component.get("name") or requirement_id
            component_geometry = component.get("geometry")
            component_entry = {**component, "title": component_name, "name": component_name}
            component_dataset_source = str(component.get("dataset_source") or "").strip().upper()
            if component_path:
                manifest.append({
                    "requirement_id": requirement_id,
                    "mandatory": True,
                    "dataset_title": component_name,
                    "source": f"LOCAL:{component_path}",
                    "access_mode": "direct_read",
                    "bind_status": "bindable",
                    "binding_status": "provided_anchor_component",
                    "bind_status_reason": "anchor_component_declared",
                    "resolution_source": "anchor_component",
                    "constraint_report": {},
                    "local_stage_path": component_path,
                    "spatial_coverage": component_geometry,
                    "resource_role": component.get("role") or "input",
                    "connectivity_probe": None,
                    "note": f"Composite anchor component: {component_name}",
                })
            elif component_dataset_source in self._EXTERNAL_DATASET_SOURCES:
                manifest.append(self._create_external_manifest_item(
                    requirement_id, True, component_name, component_entry, component_geometry,
                    f"Composite anchor component from online source: {component_name}",
                ))
            elif "GEE" in component_source.upper() or "EARTH ENGINE" in component_source.upper():
                manifest.append(self._create_gee_manifest_item(
                    requirement_id, True, component_name, component_entry, component_geometry,
                    f"Composite anchor component from online source: {component_name}",
                ))
            elif "API" in component_source.upper() or component_source.startswith("http"):
                manifest.append({
                    "requirement_id": requirement_id,
                    "mandatory": True,
                    "dataset_title": component_name,
                    "source": self._build_remote_manifest_source(component_source, "API:"),
                    "access_mode": "direct_read",
                    "bind_status": "bindable",
                    "binding_status": "provided_anchor_component",
                    "bind_status_reason": "anchor_component_online_source",
                    "resolution_source": "anchor_component",
                    "constraint_report": {},
                    "local_stage_path": None,
                    "spatial_coverage": component_geometry,
                    "resource_role": component.get("role") or "input",
                    "connectivity_probe": None,
                    "note": f"Composite anchor component from API source: {component_name}",
                })
            elif component_source:
                manifest.append({
                    "requirement_id": requirement_id,
                    "mandatory": True,
                    "dataset_title": component_name,
                    "source": self._build_remote_manifest_source(component_source, "REMOTE:"),
                    "access_mode": "direct_read",
                    "bind_status": "bindable",
                    "binding_status": "provided_anchor_component",
                    "bind_status_reason": "anchor_component_online_source",
                    "resolution_source": "anchor_component",
                    "constraint_report": {},
                    "local_stage_path": None,
                    "spatial_coverage": component_geometry,
                    "resource_role": component.get("role") or "input",
                    "connectivity_probe": None,
                    "note": f"Composite anchor component from online source: {component_name}",
                })
            else:
                manifest.append({
                    "requirement_id": requirement_id,
                    "mandatory": True,
                    "dataset_title": component_name,
                    "source": f"UNRESOLVED:{requirement_id}",
                    "access_mode": "direct_read",
                    "bind_status": "unbound",
                    "binding_status": "unbound",
                    "bind_status_reason": "anchor_component_missing_source",
                    "resolution_source": "anchor_component",
                    "constraint_report": {},
                    "local_stage_path": None,
                    "spatial_coverage": component_geometry,
                    "resource_role": component.get("role") or "input",
                    "connectivity_probe": None,
                    "note": f"Composite anchor component is missing a source: {component_name}",
                })

        # 2. 已解析的辅助数据集（mandatory 仅由 execution_requirement_level 判定）
        resolved_datasets = dataset_info.get("resolved_datasets", {})
        for requirement_id, ds_entry in resolved_datasets.items():
            is_mandatory = self._is_requirement_mandatory(ds_entry)
            source_str = ds_entry.get("source", "")
            geometry = ds_entry.get("geometry")
            dataset_title = ds_entry.get("title") or ds_entry.get("name") or requirement_id

            # 判断数据源类型
            if source_str.startswith("LOCAL:") or (source_str and os.path.exists(source_str)):
                local_path = self._strip_manifest_source_prefix(source_str, "LOCAL:")
                source = f"LOCAL:{local_path}"
                access_mode = "direct_read"
                bind_status = "bindable"
                manifest.append({
                    "requirement_id": requirement_id,
                    "mandatory": is_mandatory,
                    "dataset_title": dataset_title,
                    "source": source,
                    "access_mode": access_mode,
                    "bind_status": bind_status,
                    "binding_status": str(ds_entry.get("binding_status") or "resolved_strong"),
                    "bind_status_reason": str(ds_entry.get("bind_status_reason") or "strong_binding_accepted"),
                    "resolution_source": str(ds_entry.get("resolution_source") or "local_strict_resolution"),
                    "constraint_report": dict(ds_entry.get("constraint_report", {}) or {}),
                    "local_stage_path": local_path,
                    "spatial_coverage": geometry,
                    "resource_role": "input",
                    "connectivity_probe": None,
                    "note": ds_entry.get("note", "")
                })
            # 外部源判定必须排在 GEE 之前。GEE 分支靠 `"GEE" in source` 这种
            # 子串匹配，而 catalog 的 source 值本身可能带 GEE 字样；用 catalog
            # 的 dataset_source 精确判定，才不会把 STAC 源误路由到 ee 探测。
            elif str(ds_entry.get("dataset_source") or "").strip().upper() in self._EXTERNAL_DATASET_SOURCES:
                manifest.append(self._create_external_manifest_item(
                    requirement_id=requirement_id,
                    mandatory=is_mandatory,
                    dataset_title=dataset_title,
                    ds_entry=ds_entry,
                    geometry=geometry,
                    note=ds_entry.get("note", ""),
                ))
            elif "GEE" in source_str.upper() or "EARTH ENGINE" in source_str.upper():
                manifest.append(self._create_gee_manifest_item(
                    requirement_id=requirement_id,
                    mandatory=is_mandatory,
                    dataset_title=dataset_title,
                    ds_entry=ds_entry,
                    geometry=geometry,
                    note=ds_entry.get("note", ""),
                ))
            elif "API" in source_str.upper() or source_str.startswith("http"):
                source = self._build_remote_manifest_source(source_str, "API:")
                access_mode = "direct_read"
                bind_status = "bindable"
                manifest.append({
                    "requirement_id": requirement_id,
                    "mandatory": is_mandatory,
                    "dataset_title": dataset_title,
                    "source": source,
                    "access_mode": access_mode,
                    "bind_status": bind_status,
                    "binding_status": str(ds_entry.get("binding_status") or "resolved_strong"),
                    "bind_status_reason": str(ds_entry.get("bind_status_reason") or "strong_binding_accepted"),
                    "resolution_source": str(ds_entry.get("resolution_source") or "api_resolution"),
                    "constraint_report": dict(ds_entry.get("constraint_report", {}) or {}),
                    "local_stage_path": None,
                    "spatial_coverage": geometry,
                    "resource_role": "input",
                    "connectivity_probe": None,
                    "note": ds_entry.get("note", "")
                })
            else:
                source = self._build_remote_manifest_source(source_str, "REMOTE:")
                access_mode = "direct_read"
                bind_status = "bindable"
                manifest.append({
                    "requirement_id": requirement_id,
                    "mandatory": is_mandatory,
                    "dataset_title": dataset_title,
                    "source": source,
                    "access_mode": access_mode,
                    "bind_status": bind_status,
                    "binding_status": str(ds_entry.get("binding_status") or "resolved_strong"),
                    "bind_status_reason": str(ds_entry.get("bind_status_reason") or "strong_binding_accepted"),
                    "resolution_source": str(ds_entry.get("resolution_source") or "remote_resolution"),
                    "constraint_report": dict(ds_entry.get("constraint_report", {}) or {}),
                    "local_stage_path": None,
                    "spatial_coverage": geometry,
                    "resource_role": "input",
                    "connectivity_probe": None,
                    "note": ds_entry.get("note", "")
                })

        # 3. 不可达数据集（标记为 unbound）
        req_requirement_lookup: Dict[str, Dict[str, Any]] = {}
        for req_item in (dataset_info.get("required_datasets") or []):
            if isinstance(req_item, dict):
                rid = self._get_required_dataset_id(req_item)
                req_requirement_lookup[rid] = dict(req_item)

        unresolvable = dataset_info.get("unresolvable_datasets", [])
        unresolvable_details = dataset_info.get("unresolvable_dataset_details", {}) or {}
        for requirement_id in unresolvable:
            is_mandatory_unresolvable = self._is_requirement_mandatory(
                req_requirement_lookup.get(requirement_id, {})
            )
            detail = unresolvable_details.get(requirement_id, {}) or {}
            dataset_title = str(detail.get("dataset_name") or requirement_id).strip()
            unbound_class = str(detail.get("unbound_class") or "real_unreachable").strip()
            failure_status = str(detail.get("failure_status") or "unresolved").strip()
            manifest.append({
                "requirement_id": requirement_id,
                "dataset_title": dataset_title,
                "mandatory": is_mandatory_unresolvable,
                "source": f"UNRESOLVED:{requirement_id}",
                "access_mode": "stage_on_demand",
                "bind_status": "unbound",
                "binding_status": str(detail.get("binding_status") or "unbound"),
                "bind_status_reason": str(detail.get("bind_status_reason") or failure_status),
                "resolution_source": str(detail.get("resolution_source") or ""),
                "constraint_report": dict(detail.get("constraint_report", {}) or {}),
                "local_stage_path": None,
                "spatial_coverage": None,
                "resource_role": "input",
                "unbound_class": unbound_class,
                "failure_status": failure_status,
                "penalty_report": dict(detail.get("penalty_report", {}) or {}),
                "needs_step_plan_replan": bool(detail.get("needs_step_plan_replan", False)),
                "connectivity_probe": None,  # Commit 2: 无法探测
                "note": (
                    f"Dataset '{dataset_title}' is unbound after intent-driven local resolution "
                    f"({failure_status}, class={unbound_class})"
                ),
            })

        # 4. 输出/临时目录（由编排层提供会话目录，标记为 output/temp）
        session_id = context.get("application_context", {}).get("task_id", "unknown")
        results_dir = os.path.join(os.getcwd(), "results", str(session_id))
        temp_dir = os.path.join(os.getcwd(), "results", str(session_id), "temp")
        manifest.append({
            "requirement_id": "session_output_dir",
            "mandatory": False,
            "source": f"LOCAL:{results_dir}",
            "access_mode": "direct_write",
            "bind_status": "bindable",
            "binding_status": "output_contract",
            "bind_status_reason": "session_output_bindable",
            "resolution_source": "session_output_contract",
            "constraint_report": {},
            "local_stage_path": results_dir,
            "spatial_coverage": None,
            "resource_role": "output",
            "connectivity_probe": None,
            "note": "Session output directory for analysis results"
        })
        manifest.append({
            "requirement_id": "session_temp_dir",
            "mandatory": False,
            "source": f"LOCAL:{temp_dir}",
            "access_mode": "direct_write",
            "bind_status": "bindable",
            "binding_status": "temp_contract",
            "bind_status_reason": "session_temp_bindable",
            "resolution_source": "session_temp_contract",
            "constraint_report": {},
            "local_stage_path": temp_dir,
            "spatial_coverage": None,
            "resource_role": "temp",
            "connectivity_probe": None,
            "note": "Session temporary directory for intermediate files"
        })

        logger.info(f"Built data_access_manifest with {len(manifest)} items")
        return manifest

    async def _probe_remote_connectivity(self, manifest: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Commit 2: 远程可达性探测子阶段。

        对 manifest 中所有远程项（GEE/API）执行连通性探测：
        - GEE: 最小认证 + 轻量 metadata 查询
        - API: HEAD /health 或最小 GET 探针
        - 使用会话级缓存避免重复探测
        - 超时与重试控制
        """
        import time as time_module
        from datetime import datetime

        for item in manifest:
            source = item.get("source", "")

            # 跳过本地数据和已不可达的项
            if not source or source.startswith("LOCAL:") or source.startswith("UNRESOLVED:"):
                continue

            # 检查缓存
            if source in self.remote_probe_cache:
                cached = self.remote_probe_cache[source]
                cache_age = time_module.time() - cached.get("cached_at", 0)
                if cache_age < self.probe_cache_ttl:
                    item["connectivity_probe"] = {
                        "status": cached.get("status", "unknown"),
                        "checked_at": cached.get("checked_at", datetime.now().isoformat()),
                        "reason": cached.get("reason", "") + " (cached)",
                        "latency_ms": cached.get("latency_ms", 0)
                    }
                    logger.info(f"Using cached probe result for {source}: {cached.get('status', 'unknown')}")
                    continue

            # 执行探测：传输层瞬时故障在此原地重试，而不是把单次超时当成
            # 数据不可达送进 T6 -> refinement -> StepPlan replan 循环（replan 对网络
            # 抖动无效，只会白白消耗十几分钟的 LLM 出稿时间）。
            probe_result = await self._probe_source_once(source, item)
            attempts = 1
            while (
                self._is_transient_probe_failure(probe_result)
                and attempts <= self.probe_max_retries
            ):
                logger.warning(
                    f"Transient probe failure for {source} (attempt {attempts}/"
                    f"{self.probe_max_retries + 1}), retrying: "
                    f"{probe_result.get('reason', '')[:160]}"
                )
                await asyncio.sleep(min(2.0 * attempts, 6.0))
                probe_result = await self._probe_source_once(source, item)
                attempts += 1
            probe_result["attempts"] = attempts

            item["connectivity_probe"] = probe_result

            # 更新缓存：只缓存确定性结论。传输层故障（SSL 中断、超时、连接/代理错误）
            # 是探测未完成而非数据不可达，缓存它会把一次瞬时抖动放大成整任务失败——
            # 同一 task 的多个 application 共享 primary_dataset，首次探测撞上抖动后
            # 其余 application 全部读到缓存的失败结论，产出 5/5 的假 T6。
            if self._is_transient_probe_failure(probe_result):
                logger.warning(
                    f"Transient probe failure for {source} persisted after {attempts} attempts, "
                    f"not cached: {probe_result.get('reason', '')[:160]}"
                )
            else:
                self.remote_probe_cache[source] = {
                    **probe_result,
                    "cached_at": time_module.time()
                }

        return manifest

    async def _probe_source_once(self, source: str, item: Dict[str, Any]) -> Dict[str, Any]:
        from datetime import datetime

        dataset_source = str(item.get("dataset_source") or "").strip().upper()
        if dataset_source in self._EXTERNAL_DATASET_SOURCES:
            # datasets.json 中的非 GEE 源：按 source 调用对应客户端探测
            record = {
                "id": item.get("id"),
                "source": dataset_source,
                "doc_url": item.get("doc_url"),
                "stac_url": item.get("stac_url"),
                "overpass_ql": item.get("overpass_ql"),
                "endpoint": item.get("endpoint"),
            }
            return await probe_dataset(record)
        if source.upper().startswith(("GEE:", "GEE_OFFICIAL:", "GEE_COMMUNITY:", "GEE_ONLINE:", "GEE_ALIAS:")):
            # catalog 的 source 前缀带具体来源（GEE_OFFICIAL / GEE_COMMUNITY），
            # 在线与 alias 通道另有各自前缀；它们都走 Earth Engine 运行时探测。
            try:
                return await self._probe_gee(source, manifest_item=item)
            except TypeError:
                return await self._probe_gee(source)
        if source.startswith("API:"):
            return await self._probe_api(source)
        # REMOTE: 类型，跳过探测
        return {
            "status": "unknown",
            "checked_at": datetime.now().isoformat(),
            "reason": "Remote source type not supported for probing",
            "latency_ms": 0
        }

    # 传输层故障标志：这些错误说明探测没能完成，与数据本身是否可达无关。
    # 覆盖 _probe_gee 与 _probe_api 两条通道的失败文案。
    _TRANSIENT_PROBE_MARKERS = (
        "ssl",
        "eof occurred",
        "timeout",
        "timed out",
        "connection",
        "proxy",
        "temporarily unavailable",
        "max retries exceeded",
        "broken pipe",
        "reset by peer",
        "handshake",
    )

    def _is_transient_probe_failure(self, probe_result: Dict[str, Any]) -> bool:
        """判断探测失败是否属于可重试的传输层故障。

        只对 status == "fail" 生效；成功结果与确定性失败（asset 不存在、
        依赖库缺失、asset id 为空等）都返回 False，仍按原逻辑缓存——
        对这些结论重试没有意义，缓存反而能省掉同一 task 内的重复探测。
        """
        if not isinstance(probe_result, dict):
            return False
        if str(probe_result.get("status", "")).strip().lower() != "fail":
            return False
        reason = str(probe_result.get("reason", "")).lower()
        if not reason:
            return False
        # GEE 语义错误（EEException）走 "GEE error:" 前缀，属确定性结论；
        # 但 EEException 也可能包裹传输层异常，所以仍按标志词判定而非只看前缀。
        return any(marker in reason for marker in self._TRANSIENT_PROBE_MARKERS)

    def _reload_gee_env(self) -> None:
        """
        每次访问 GEE 前可选刷新环境变量。

        说明：
        - 默认读取项目根目录 .env；
        - 若配置关闭 reload_env_each_call，则跳过；
        - 加载失败不抛异常，由后续 ee.Initialize 给出具体错误。
        """
        if not self.gee_reload_env_each_call:
            return

        env_file = str(self.gee_env_file or "").strip()
        if not env_file:
            return

        env_path = os.path.expanduser(env_file)
        if not os.path.isabs(env_path):
            env_path = os.path.join(os.getcwd(), env_path)
        if not os.path.exists(env_path):
            return

        try:
            from dotenv import load_dotenv

            load_dotenv(dotenv_path=env_path, override=True)
        except Exception as e:
            logger.warning(f"Failed to load GEE env file '{env_path}': {e}")

    def _initialize_gee(self, ee) -> None:
        """
        统一 GEE 初始化入口。

        约定 self.gee_api_key_env 指向的变量承载服务账号凭证：JSON 内容本身（以 `{`
        开头）或该 JSON 的文件路径；EARTHENGINE_PROJECT 可选，用于指定初始化 project。
        """
        self._reload_gee_env()
        initialize_gee_from_service_account(
            ee,
            credential_env=self.gee_api_key_env,
            project_env=self.gee_project_env,
        )

    def _build_gee_probe_target(self, ee: Any, ee_ctor: str, ee_id: str) -> Any:
        """按 ee_ctor 构造最小 GEE 探测对象。"""
        if ee_ctor == "ee.Image":
            return ee.Image(ee_id)
        if ee_ctor == "ee.ImageCollection":
            return ee.ImageCollection(ee_id)
        if ee_ctor == "ee.FeatureCollection":
            return ee.FeatureCollection(ee_id)
        raise ValueError(f"Unsupported ee_ctor for probe: {ee_ctor}")

    def _normalize_gee_ctor(self, value: Any) -> Optional[str]:
        """把 catalog type / ee_object_type / ee_ctor 归一化为 ee 构造器名。

        只接受能判定 asset 形态的取值。`gee_dataset`、`local_dataset` 这类是
        数据来源类别而非 asset 类型，一律归一化为 None，交由探测阶段按候选
        构造器求证——臆断构造器会把可达资产判成不可达。
        """
        normalized = str(value or "").strip().lower()
        if normalized in {"ee.image", "image"}:
            return "ee.Image"
        if normalized in {"ee.imagecollection", "imagecollection", "image_collection"}:
            return "ee.ImageCollection"
        if normalized in {
            "ee.featurecollection",
            "featurecollection",
            "feature_collection",
            "table",
        }:
            return "ee.FeatureCollection"
        return None

    def _resolve_gee_ctor(self, entry: Dict[str, Any], ee_id: str) -> Optional[str]:
        """按证据强弱确定 GEE 构造器：条目显式声明 > 本地 catalog 查表。

        两条通道都给不出可判定类型时返回 None，表示"未知"而非某个默认值。
        """
        for key in ("ee_ctor", "ee_object_type", "catalog_type", "type"):
            ctor = self._normalize_gee_ctor(entry.get(key))
            if ctor:
                return ctor
        geo_search = getattr(self, "geo_search", None)
        if not ee_id or geo_search is None:
            return None
        return self._normalize_gee_ctor(
            geo_search.get_local_catalog_asset_type(ee_id)
        )

    def _probe_gee_with_ctor_fallback(
        self, ee: Any, ee_ctor: Optional[str], ee_id: str
    ) -> str:
        """按候选构造器依次探测 GEE asset，类型不匹配时自动切换构造器。

        catalog 的 asset 类型可能缺失或推断错误（例如 Image 被判成 ImageCollection），
        这里在探测失败且错误指向 asset 类型不匹配时，改用其余构造器重试，
        返回最终成功探测所使用的构造器名；探测失败时抛出最后一次异常。
        ee_ctor 为空表示类型未知，按全部候选依次求证。
        """
        ctor_candidates = [ctor for ctor in (ee_ctor,) if ctor]
        ctor_candidates.extend(
            ctor for ctor in self._GEE_PROBE_CTORS if ctor not in ctor_candidates
        )

        last_error: Optional[Exception] = None
        for candidate_ctor in ctor_candidates:
            try:
                gee_target = self._build_gee_probe_target(ee, candidate_ctor, ee_id)
                if candidate_ctor == "ee.Image":
                    gee_target.bandNames().getInfo()
                else:
                    gee_target.limit(1).size().getInfo()
                return candidate_ctor
            except _GEEProbeTimeoutError:
                raise
            except Exception as probe_error:  # noqa: BLE001
                last_error = probe_error
                if not self._is_gee_asset_type_mismatch(probe_error):
                    raise
        raise last_error

    def _is_gee_asset_type_mismatch(self, error: Exception) -> bool:
        """判断 GEE 异常是否由 asset 类型与构造器不匹配导致。"""
        message = str(error or "").lower()
        if not message:
            return False
        # Earth Engine 对类型不匹配有多种文案，实测最常见的是
        # "ImageCollection.load: Expected asset '<id>' to be an ImageCollection,
        #  found 'Image'."，需与 "is not an image" 系列一并识别，
        # 否则构造器回退通道永远不会触发。
        if "expected asset" in message and "found" in message:
            return True
        if "expected a" in message and ("but got" in message or "found" in message):
            return True
        return any(
            marker in message
            for marker in (
                "is not an image",
                "is not a image",
                "is not an imagecollection",
                "is not a collection",
                "is not a featurecollection",
                "is not a feature collection",
            )
        )

    async def _probe_gee(self, source: str, manifest_item: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        GEE 探测：最小认证 + 轻量 metadata 查询。

        返回格式：
        {
            "status": "pass" | "fail" | "unknown",
            "checked_at": "2024-01-01T10:00:00",
            "reason": "...",
            "latency_ms": 150
        }
        """
        import time as time_module
        from datetime import datetime

        manifest_item = manifest_item or {}
        ee_id = str(
            manifest_item.get("ee_id")
            or manifest_item.get("id")
            or self._strip_gee_source_prefix(source)
        ).strip()
        ee_ctor = self._resolve_gee_ctor(manifest_item, ee_id)
        dataset_title = str(manifest_item.get("dataset_title") or manifest_item.get("note") or ee_id).strip()
        start_time = time_module.time()

        if not ee_id:
            return {
                "status": "fail",
                "checked_at": datetime.now().isoformat(),
                "reason": "GEE probe requires a non-empty Earth Engine asset id",
                "latency_ms": int((time_module.time() - start_time) * 1000),
            }

        try:
            # 尝试导入 ee
            try:
                import ee
            except ImportError:
                return {
                    "status": "fail",
                    "checked_at": datetime.now().isoformat(),
                    "reason": "GEE Python library (earthengine-api) not installed",
                    "latency_ms": int((time_module.time() - start_time) * 1000)
                }

            if not ee_ctor:
                logger.info(
                    "GEE asset type unknown for %s; probing candidate constructors",
                    ee_id,
                )

            try:
                timeout_seconds = float(self.probe_timeout)
                if timeout_seconds <= 0:
                    raise ValueError("GEE probe timeout must be greater than zero")

                hard_timeout_armed = False
                previous_alarm_handler = None
                supports_hard_timeout = (
                    threading.current_thread() is threading.main_thread()
                    and hasattr(signal, "SIGALRM")
                    and hasattr(signal, "ITIMER_REAL")
                    and hasattr(signal, "setitimer")
                )
                if supports_hard_timeout:
                    previous_alarm = signal.getitimer(signal.ITIMER_REAL)
                    if previous_alarm[0] > 0:
                        raise RuntimeError(
                            "GEE probe cannot install a hard timeout while another real-time alarm is active"
                        )

                    def _raise_gee_probe_timeout(_signum, _frame):
                        raise _GEEProbeTimeoutError()

                    previous_alarm_handler = signal.getsignal(signal.SIGALRM)
                    signal.signal(signal.SIGALRM, _raise_gee_probe_timeout)
                    try:
                        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
                    except Exception:
                        signal.signal(signal.SIGALRM, previous_alarm_handler)
                        raise
                    hard_timeout_armed = True

                try:
                    self._initialize_gee(ee)
                    set_gee_deadline(ee, timeout_seconds * 1000)
                    ee_ctor = self._probe_gee_with_ctor_fallback(ee, ee_ctor, ee_id)
                finally:
                    if hard_timeout_armed:
                        signal.setitimer(signal.ITIMER_REAL, 0)
                        signal.signal(signal.SIGALRM, previous_alarm_handler)

                latency_ms = int((time_module.time() - start_time) * 1000)

                return {
                    "status": "pass",
                    "checked_at": datetime.now().isoformat(),
                    "reason": f"GEE asset metadata query succeeded for '{dataset_title}' ({ee_ctor})",
                    "latency_ms": latency_ms
                }

            except _GEEProbeTimeoutError:
                latency_ms = int((time_module.time() - start_time) * 1000)
                return {
                    "status": "fail",
                    "checked_at": datetime.now().isoformat(),
                    "reason": f"GEE probe timeout after {self.probe_timeout}s",
                    "latency_ms": latency_ms,
                }
            except ee.EEException as e:
                latency_ms = int((time_module.time() - start_time) * 1000)
                return {
                    "status": "fail",
                    "checked_at": datetime.now().isoformat(),
                    "reason": f"GEE error: {str(e)}",
                    "latency_ms": latency_ms
                }

        except Exception as e:
            latency_ms = int((time_module.time() - start_time) * 1000)
            error_detail = str(e).strip() or type(e).__name__
            return {
                "status": "fail",
                "checked_at": datetime.now().isoformat(),
                "reason": f"GEE probe failed: {error_detail}",
                "latency_ms": latency_ms
            }

    async def _probe_api(self, source: str) -> Dict[str, Any]:
        """
        API 探测：优先 HEAD /health，无健康端点则最小 GET 探针。

        返回格式同 _probe_gee
        """
        import time as time_module
        from datetime import datetime

        endpoint = source.replace("API:", "").strip()
        start_time = time_module.time()

        try:
            # 尝试导入 aiohttp
            try:
                import aiohttp
            except ImportError:
                return {
                    "status": "fail",
                    "checked_at": datetime.now().isoformat(),
                    "reason": "aiohttp library not installed for API probing",
                    "latency_ms": int((time_module.time() - start_time) * 1000)
                }

            timeout = aiohttp.ClientTimeout(total=self.probe_timeout)

            async with aiohttp.ClientSession(timeout=timeout) as session:
                # 尝试 HEAD /health
                health_endpoints = ["/health", "/healthz", "/status", ""]
                for health_path in health_endpoints:
                    try:
                        url = endpoint.rstrip("/") + health_path if health_path else endpoint
                        async with session.head(url) as response:
                            latency_ms = int((time_module.time() - start_time) * 1000)
                            if response.status < 400:
                                return {
                                    "status": "pass",
                                    "checked_at": datetime.now().isoformat(),
                                    "reason": f"API endpoint {url} returned {response.status}",
                                    "latency_ms": latency_ms
                                }
                    except Exception:
                        continue

                # 降级到 GET（最小请求）
                try:
                    async with session.get(endpoint) as response:
                        latency_ms = int((time_module.time() - start_time) * 1000)
                        if response.status < 500:
                            return {
                                "status": "pass",
                                "checked_at": datetime.now().isoformat(),
                                "reason": f"API endpoint returned {response.status}",
                                "latency_ms": latency_ms
                            }
                        else:
                            return {
                                "status": "fail",
                                "checked_at": datetime.now().isoformat(),
                                "reason": f"API endpoint returned {response.status} (server error)",
                                "latency_ms": latency_ms
                            }
                except Exception as e:
                    latency_ms = int((time_module.time() - start_time) * 1000)
                    return {
                        "status": "fail",
                        "checked_at": datetime.now().isoformat(),
                        "reason": f"API probe failed: {str(e)}",
                        "latency_ms": latency_ms
                    }

        except asyncio.TimeoutError:
            latency_ms = int((time_module.time() - start_time) * 1000)
            return {
                "status": "fail",
                "checked_at": datetime.now().isoformat(),
                "reason": f"API probe timeout after {self.probe_timeout}s",
                "latency_ms": latency_ms
            }
        except Exception as e:
            latency_ms = int((time_module.time() - start_time) * 1000)
            return {
                "status": "fail",
                "checked_at": datetime.now().isoformat(),
                "reason": f"API probe failed: {str(e)}",
                "latency_ms": latency_ms
            }

    def _evaluate_termination_condition_with_manifest(
        self,
        task_plan: List[Dict],
        dataset_info: Dict,
        adjustments: List[Dict],
        manifest: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Commit 1+2: 基于 manifest 的统一终止判定 + 任务-数据域一致性闸门 + 远程探测检查。

        终止条件（满足任意一条 → 终止或回 StepPlan）：
          T1: 存在 mandatory 且 unbound 的 manifest 项
          T2: 输入任务计划步骤数量为 0
          T3: 步骤仍显式引用未强绑定的数据 requirement
          T5: 任务-数据域空间不一致（目标区域与数据覆盖范围不匹配）
          T6: 存在 mandatory 远程项且探测失败（Commit 2 新增）
        """
        steps = task_plan if isinstance(task_plan, list) else []
        unresolvable = dataset_info.get("unresolvable_datasets", [])
        unresolvable_names = dataset_info.get("unresolvable_dataset_names", [])
        resolved = dataset_info.get("resolved_datasets", {})

        # T1: 检查 manifest 中是否有 mandatory 且 unbound 的项
        mandatory_unbound_all = [
            item for item in manifest
            if item.get("mandatory", False) and item.get("bind_status") != "bindable"
        ]
        t1_blocking_unbound_classes = {
            "real_unreachable",
            "semantic_rejection",
            "hard_conflict",
            "no_candidate",
        }
        mandatory_unbound = [
            item for item in mandatory_unbound_all
            if item.get("requirement_id") == "primary_dataset"
            or str(item.get("unbound_class") or "").strip() in t1_blocking_unbound_classes
        ]
        if mandatory_unbound:
            unbound_ids = [item["requirement_id"] for item in mandatory_unbound]
            if any(item.get("requirement_id") == "primary_dataset" for item in mandatory_unbound):
                return {
                    "should_terminate": True,
                    "reason": "T1: Primary dataset is not bindable (missing local path and online source).",
                    "terminal_rule": "T1_primary_data_missing",
                    "failure_class": "primary_data_missing",
                    "recovery_hint": (
                        "请为主数据集提供可访问的本地 data_path，"
                        "或在 dataset_info/base_dataset 中提供明确的在线 source。"
                    ),
                }
            unbound_groups = {
                "real_unreachable": [],
                "semantic_rejection": [],
                "hard_conflict": [],
                "no_candidate": [],
                "unknown": [],
            }
            for item in mandatory_unbound:
                unbound_class = str(item.get("unbound_class", "") or "unknown").strip()
                if unbound_class not in unbound_groups:
                    unbound_class = "unknown"
                unbound_groups[unbound_class].append(item["requirement_id"])
            class_summary = []
            if unbound_groups["real_unreachable"]:
                class_summary.append(f"真实不可达未绑定: {unbound_groups['real_unreachable']}")
            if unbound_groups["semantic_rejection"]:
                class_summary.append(f"语义拒绝导致未绑定: {unbound_groups['semantic_rejection']}")
            if unbound_groups["hard_conflict"]:
                class_summary.append(f"硬冲突导致未绑定: {unbound_groups['hard_conflict']}")
            if unbound_groups["no_candidate"]:
                class_summary.append(f"无候选导致未绑定: {unbound_groups['no_candidate']}")
            if unbound_groups["unknown"]:
                class_summary.append(f"未分类未绑定: {unbound_groups['unknown']}")
            class_summary_text = "；".join(class_summary) if class_summary else str(unbound_ids)

            nonempty_groups = [
                group_name
                for group_name, requirement_ids in unbound_groups.items()
                if requirement_ids
            ]
            if nonempty_groups == ["real_unreachable"]:
                terminal_rule = "T1_mandatory_data_real_unreachable"
                failure_class = "real_unreachable_unbound"
            elif nonempty_groups == ["semantic_rejection"]:
                terminal_rule = "T1_mandatory_data_semantic_rejection"
                failure_class = "semantic_rejection_unbound"
            elif nonempty_groups == ["hard_conflict"]:
                terminal_rule = "T1_mandatory_data_hard_conflict"
                failure_class = "hard_conflict_unbound"
            elif nonempty_groups == ["no_candidate"]:
                terminal_rule = "T1_mandatory_data_no_candidate"
                failure_class = "no_candidate_unbound"
            else:
                terminal_rule = "T1_mandatory_data_unbound"
                failure_class = "mixed"
            return {
                "should_terminate": True,
                "reason": f"T1: 存在 {len(mandatory_unbound)} 个必需数据项未绑定。{class_summary_text}",
                "terminal_rule": terminal_rule,
                "failure_class": failure_class,
                "recoverable": True,
                "replan_target": "step_plan",
                "recovery_hint": (
                    "请让 StepPlan 根据绑定诊断替换或删除以下数据 requirement，"
                    f"并同步调整显式步骤引用: {class_summary_text}"
                ),
            }

        # T6: Commit 2 新增 - 检查 mandatory 远程项的探测状态
        # 判据与 ExecutionContract.get_mandatory_remote_probe_failed_items 保持一致：
        # 凡是探测过且未通过的远程项都算，不按 source 前缀白名单筛。
        mandatory_probe_failed = []
        for item in manifest:
            if not item.get("mandatory", False):
                continue
            source = item.get("source", "")
            if not is_remote_manifest_source(source):
                continue
            probe = item.get("connectivity_probe")
            if probe and probe.get("status") not in _PROBE_OK_STATUSES:
                mandatory_probe_failed.append({
                    "requirement_id": item["requirement_id"],
                    "source": source,
                    "probe_status": probe.get("status"),
                    "probe_reason": probe.get("reason"),
                    "transient": self._is_transient_probe_failure(probe),
                })

        if mandatory_probe_failed:
            failed_ids = [f["requirement_id"] for f in mandatory_probe_failed]
            reasons = "; ".join([f"{f['requirement_id']}: {f['probe_reason']}" for f in mandatory_probe_failed])
            recovery_hints = []
            for f in mandatory_probe_failed:
                if "authentication" in f["probe_reason"].lower() or "auth" in f["probe_reason"].lower():
                    recovery_hints.append(f"- {f['source']}: 补充认证凭证（GEE 需 ee.Authenticate()）")
                elif f["transient"]:
                    recovery_hints.append(f"- {f['source']}: 网络/传输层故障，重试 {self.probe_max_retries + 1} 次仍失败，请检查网络连接")
                elif "not installed" in f["probe_reason"].lower():
                    recovery_hints.append(f"- {f['source']}: 安装必需的 Python 库")
                else:
                    recovery_hints.append(f"- {f['source']}: 改用其他数据源或调整任务区域")

            primary_probe_failed = "primary_dataset" in failed_ids
            # 全部是传输层故障时不走 replan：换 requirement 解决不了网络问题，
            # 只会消耗 2 轮无效 StepPlan 出稿。
            all_transient = all(f["transient"] for f in mandatory_probe_failed)
            termination = {
                "should_terminate": True,
                "reason": f"T6: 存在 {len(mandatory_probe_failed)} 个必需远程数据项探测失败: {failed_ids}. {reasons}",
                "terminal_rule": "T6_remote_probe_failed",
                "failure_class": "remote_probe_failed",
                "recovery_hint": "远程数据源不可达，可执行恢复建议:\n" + "\n".join(recovery_hints),
            }
            if not primary_probe_failed and not all_transient:
                termination.update({
                    "recoverable": True,
                    "replan_target": "step_plan",
                })
            return termination

        # T2: 空计划检查
        if len(steps) == 0:
            return {
                "should_terminate": True,
                "reason": "T2: 输入任务计划步骤数量为 0。",
                "terminal_rule": "T2_empty_plan",
                "failure_class": "empty_plan",
                "recoverable": True,
                "replan_target": "step_plan",
                "recovery_hint": "StepPlan 必须生成非空的可执行任务计划。",
            }

        # T3: 计划调整完整性检查
        for step in steps:
            if self._step_needs_any(
                step,
                missing_requirement_ids=unresolvable,
            ):
                unmet_ids = [
                    req_id for req_id in step_required_dataset_ids(step)
                    if req_id in set(unresolvable)
                ]
                if context.get("resource_check_mode") == "single_pass_report":
                    return {
                        "should_terminate": True,
                        "reason": (
                            f"T3: 步骤 {step.get('id')} '{step.get('step_name', '')}' "
                            f"依赖的数据需求不可达: {unmet_ids or unresolvable_names}。"
                        ),
                        "terminal_rule": "T3_referenced_dataset_unavailable",
                        "failure_class": "referenced_dataset_unavailable",
                        "recoverable": False,
                        "replan_target": "",
                        "recovery_hint": (
                            "原计划已保留；如需执行该计划，请为以下 requirement 提供可达的强绑定: "
                            f"{unmet_ids or unresolvable_names}"
                        ),
                    }
                return {
                    "should_terminate": True,
                    "reason": (
                        f"T3: 步骤 {step.get('id')} '{step.get('step_name', '')}' "
                        f"仍依赖不可达数据需求: {unmet_ids or unresolvable_names}。计划调整不完整。"
                    ),
                    "terminal_rule": "T3_plan_adjustment_incomplete",
                    "failure_class": "plan_adjustment_incomplete",
                    "recoverable": True,
                    "replan_target": "step_plan",
                    "recovery_hint": (
                        "StepPlan 需要根据解析诊断替换或删除以下 requirement，"
                        f"并同步调整步骤引用: {unmet_ids or unresolvable_names}"
                    ),
                }

        # T5: 任务-数据域空间一致性闸门
        task_spatial_extent = context.get("dataset_info", {}).get("spatial_extent") or context.get("task", {}).get("spatial_extent", {})
        if task_spatial_extent and isinstance(task_spatial_extent, dict):
            task_regions = self._extract_region_terms_from_spatial_info(task_spatial_extent)
            task_region = task_regions[0] if task_regions else None

            # 检查所有 bindable 数据的空间覆盖
            bindable_items = [item for item in manifest if item.get("bind_status") == "bindable"]
            spatial_mismatch = []

            for item in bindable_items:
                coverage = item.get("spatial_coverage")
                if coverage and isinstance(coverage, dict):
                    data_regions = self._extract_region_terms_from_spatial_info(coverage)

                    if task_region and data_regions:
                        region_match = classify_region_match(task_region, data_regions)
                        if region_match.get("status") == "conflict":
                            spatial_mismatch.append({
                                "requirement_id": item["requirement_id"],
                                "task_region": task_region,
                                "data_region": ",".join(region_match.get("conflicting_regions", data_regions)),
                            })

            if spatial_mismatch and len(spatial_mismatch) >= len(bindable_items) * 0.5:
                mismatch_ids = [item["requirement_id"] for item in spatial_mismatch]
                primary_mismatch = "primary_dataset" in mismatch_ids
                termination = {
                    "should_terminate": True,
                    "reason": (
                        f"T5: 任务目标区域（{task_region}）与数据覆盖范围不一致。"
                        f"不匹配项: {mismatch_ids}"
                    ),
                    "terminal_rule": "T5_spatial_domain_mismatch",
                    "failure_class": "spatial_domain_mismatch",
                    "recovery_hint": (
                        f"任务目标区域为 {task_region}，但多数数据覆盖其他区域。"
                        "请提供目标区域的基础数据，或由 StepPlan 替换不匹配的外部 requirement。"
                    ),
                }
                if not primary_mismatch:
                    termination.update({
                        "recoverable": True,
                        "replan_target": "step_plan",
                    })
                return termination

        # 所有检查通过
        primary_name = dataset_info.get("name") or dataset_info.get("base_dataset", {}).get("name", "primary")
        guaranteed = list(resolved.keys()) + [primary_name]

        return {
            "should_terminate": False,
            "reason": "All mandatory data requirements are bindable.",
            "terminal_rule": "",
            "failure_class": "",
            "guaranteed_datasets": guaranteed,
        }

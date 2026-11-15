"""
step_plan_agent.py — StepPlanAgent（Agent 2）

职责：基于成熟 idea 及其 revision audit 生成详细步骤，
     并执行 capability 锚定流水线。

从 plan_agent.py 拆分而来，包含：
  - Capability Index 加载 / 索引构建 / 查询
  - Schema 构建（_build_step_schema）
  - 结构校验（_validate_structured_step_contract）
  - 锚定流水线（_anchor_plan_to_capabilities）
  - 显式产物引用 DAG 校验
  - step plan prompt 构建
"""
import copy
import json
import logging
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datahypgen.mas.tools.plan_dag_validator import (
    OUTPUT_ROLES,
    SOURCE_KINDS,
    step_input_data_types,
    step_primary_output_type,
    validate_plan_dag,
)
from datahypgen.mas.tools.operation_semantics import (
    SEMANTIC_CONTRADICTED,
    SEMANTIC_UNPROVEN,
    evaluate_operation_compatibility,
    function_name_matches,
    merge_operation_constraints,
    semantic_requirement_satisfied,
    semantic_requirement_verdict,
)
from datahypgen.mas.tools.plan_step_identity import (
    MATCH_OUTPUT_ARTIFACT,
    MATCH_STEP_ID_FALLBACK,
    MATCH_STEP_NAME_EXACT,
    build_step_identity,
    relocate_step_constraints,
)
from datahypgen.mas.tools.capability_details import CapabilityDetailStore
from datahypgen.mas.tools.dataset_binding_proof import (
    requirement_core,
    requirement_fingerprint,
    validate_dataset_binding_proof,
)
from datahypgen.mas.tools.dataset_resolution_semantics import (
    FROZEN_IDENTITY_DIMENSIONS,
    FROZEN_IDENTITY_MATCH_MODES,
    extract_region_terms,
    normalize_frozen_identity_dimension,
    normalize_temporal_coverage,
    normalize_temporal_resolution,
)

from .base_agent import BaseAgent, AgentExecutionError

logger = logging.getLogger(__name__)


class StepPlanContractError(AgentExecutionError):
    """Raised when ours cannot produce a valid plan after scoped repair."""

    def __init__(
        self,
        issues: List[str],
        *,
        required_datasets: Optional[List[Dict[str, Any]]] = None,
        task_plan: Optional[List[Dict[str, Any]]] = None,
        capability_anchor_meta: Optional[Dict[str, Any]] = None,
        generation_audit: Optional[Dict[str, Any]] = None,
        coverage_gap: bool = False,
    ) -> None:
        self.issues = list(issues)
        self.coverage_gap = bool(coverage_gap)
        self.required_datasets = copy.deepcopy(required_datasets or [])
        self.task_plan = copy.deepcopy(task_plan or [])
        self.capability_anchor_meta = copy.deepcopy(capability_anchor_meta or {})
        self.generation_audit = copy.deepcopy(generation_audit or {})
        self.generation_audit["model_call_count"] = len(
            self.generation_audit.get("model_calls", [])
        )
        super().__init__(
            "StepPlan output violates structured contract: " + "; ".join(self.issues)
        )

# 这些包不是"地理空间能力"，不收录在 geo_capability_index 中，
# 但绝不应被标记为 unreachable，否则会系统性压低 tool_reachability_ratio。
_INFRASTRUCTURE_PACKAGES = frozenset({
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "sklearn",
    "matplotlib",
    "seaborn",
    "pillow",
    "gdal",
    "osgeo",
    "h5py",
    "netcdf4",
    "fiona",
    "requests",
    "tqdm",
    "joblib",
})

_PACKAGE_ALIASES = {
    "earthengine-api": "ee",
    "earthengine": "ee",
    "google-earth-engine": "ee",
}


class StepPlanAgent(BaseAgent):
    """
    步骤规划智能体 (StepPlanAgent)

    输入：idea payload（含 idea、planning_handoff、dataset_characteristics、candidate_datasets）
    输出：required_datasets + anchored task_plan（含 capability 锚定元数据）
    """

    def __init__(self, model, config: Dict[str, Any]):
        super().__init__(model, config)
        self.planning_depth = config.get("planning_depth", "detailed")
        self.planning_temperature = config.get("temperature")
        self.planning_max_tokens = config.get("max_tokens")
        self.hierarchical_generation = config.get("hierarchical_generation")
        self.graph_repair_attempts = config.get("graph_repair_attempts")
        self.detail_repair_attempts = config.get("detail_repair_attempts")

        # ── capability index ────────────────────────────────────
        global_cfg = config.get("_global_config", {})
        cap_cfg = global_cfg.get("tools", {}).get("capability_index", {})
        self._cap_index_enabled: bool = cap_cfg.get("enabled", True)
        self._cap_index_path: str = cap_cfg.get(
            "path", "datahypgen/mas/tools/data/geo_capability_index.json"
        )
        # 原始数据和内存索引
        self._cap_entries: List[Dict[str, Any]] = []
        self._idx_by_task_type: Dict[str, List[int]] = {}
        self._idx_by_package: Dict[str, List[int]] = {}
        self._idx_by_output_type: Dict[str, List[int]] = {}
        self._idx_by_input_type: Dict[str, List[int]] = {}
        self._cap_index_loaded: bool = False
        self._cap_detail_store: Optional[CapabilityDetailStore] = None

        if self._cap_index_enabled:
            self._load_geo_capability_index()
            if self._cap_entries:
                self._build_capability_indexes()
                cap_index_path = Path(self._cap_index_path)
                if not cap_index_path.is_absolute():
                    cap_index_path = Path(os.getcwd()) / cap_index_path
                self._cap_detail_store = CapabilityDetailStore(
                    cap_index_path.parent / "geo_capability_details"
                )

        # 系统提示词
        self.system_prompt = config.get("system_prompt", """
You are a senior task planning expert for data-intensive research workflows.
Your job is to convert an abstract research idea into a concrete,
step-by-step execution plan with explicit tools and dependencies.
All free-text content you generate for the task plan must be written in English.
""")

    # =====================================================================
    # Capability Index — 加载与索引构建
    # =====================================================================

    def _load_geo_capability_index(self) -> None:
        """
        从配置路径加载 geo_capability_index.json。

        加载失败时 fail-closed：记录错误并将 _cap_index_loaded 置 False，
        后续查询方法会显式返回空结果而非悄悄兜底。
        """
        path = Path(self._cap_index_path)
        if not path.is_absolute():
            path = Path(os.getcwd()) / path

        if not path.exists():
            logger.warning(
                f"capability index not found at {path}. "
                "Capability-anchored planning will be disabled for this session."
            )
            self._cap_index_loaded = False
            return

        try:
            with path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, list):
                logger.error(
                    f"capability index at {path} is not a JSON array "
                    f"(got {type(raw).__name__}). Disabling capability index."
                )
                self._cap_index_loaded = False
                return
            self._cap_entries = raw
            self._cap_index_loaded = True
            logger.info(
                f"loaded capability index: {len(raw)} entries from {path}"
            )
        except (json.JSONDecodeError, OSError) as exc:
            logger.error(
                f"failed to load capability index from {path}: {exc}. "
                "Disabling capability index."
            )
            self._cap_index_loaded = False

    def _build_capability_indexes(self) -> None:
        """
        基于 _cap_entries 构建多维倒排索引，供后续查询方法使用。

        索引维度：
          - task_type      -> entry 序号列表
          - package        -> entry 序号列表（去版本号）
          - output_data_type -> entry 序号列表
          - input_data_type  -> entry 序号列表（多值展开）
        """
        by_task: Dict[str, List[int]] = defaultdict(list)
        by_pkg: Dict[str, List[int]] = defaultdict(list)
        by_out: Dict[str, List[int]] = defaultdict(list)
        by_in: Dict[str, List[int]] = defaultdict(list)

        for i, entry in enumerate(self._cap_entries):
            tt = entry.get("task_type", "")
            if tt:
                by_task[tt].append(i)

            pkg_raw = entry.get("package", "")
            pkg = pkg_raw.split("==")[0].split(">=")[0].strip().lower() if pkg_raw else ""
            if pkg:
                by_pkg[pkg].append(i)

            output_types = entry.get("output_data_type") or []
            if isinstance(output_types, str):
                output_types = [output_types]
            for out_t in output_types:
                if out_t:
                    by_out[out_t].append(i)

            input_types = entry.get("input_data_type") or []
            if isinstance(input_types, str):
                input_types = [input_types]
            for in_t in input_types:
                if in_t:
                    by_in[in_t].append(i)

        self._idx_by_task_type = dict(by_task)
        self._idx_by_package = dict(by_pkg)
        self._idx_by_output_type = dict(by_out)
        self._idx_by_input_type = dict(by_in)

        logger.info(
            f"capability indexes built — "
            f"task_types={len(by_task)}, packages={len(by_pkg)}, "
            f"output_types={len(by_out)}, input_types={len(by_in)}"
        )

    # =====================================================================
    # Capability Index — 查询 helpers
    # =====================================================================

    def _normalize_tool_package_name(self, package: str) -> str:
        """
        归一化工具/包名，用于 capability index 检索与 reachability 判定。

        目标是消除常见安装名 / 导入名 / 文档别名之间的差异，
        避免把实际可用工具误判为 unreachable。
        """
        package_key = package.lower().split("==")[0].split(">=")[0].strip()
        return _PACKAGE_ALIASES.get(package_key, package_key)

    def _semantic_search_capabilities(
        self,
        task_type: str = "",
        input_types: Optional[List[str]] = None,
        output_type: str = "",
        package: str = "",
    ) -> List[Dict[str, Any]]:
        """
        多维语义搜索：按 task_type / input / output / package 交集筛选能力条目。

        返回全部匹配条目，不做 confidence 截断；
        目标感知排序由调用方（_select_implementation_hint）负责。
        """
        if not self._cap_index_loaded:
            logger.debug("capability index not loaded; _semantic_search returns []")
            return []

        candidate_sets: List[set] = []

        if task_type and task_type in self._idx_by_task_type:
            candidate_sets.append(set(self._idx_by_task_type[task_type]))

        if package:
            pkg_key = self._normalize_tool_package_name(package)
            if pkg_key in self._idx_by_package:
                candidate_sets.append(set(self._idx_by_package[pkg_key]))

        if output_type and output_type in self._idx_by_output_type:
            candidate_sets.append(set(self._idx_by_output_type[output_type]))

        if input_types:
            in_set: set = set()
            for it in input_types:
                in_set.update(self._idx_by_input_type.get(it, []))
            if in_set:
                candidate_sets.append(in_set)

        if not candidate_sets:
            return []

        result_ids = candidate_sets[0]
        for s in candidate_sets[1:]:
            result_ids &= s
            if not result_ids:
                return []

        results = [self._cap_entries[i] for i in result_ids]
        return results

    def _capability_entry_by_function(self, function_name: str) -> Optional[Dict[str, Any]]:
        """Return an exact capability-index entry by function name when available."""
        target = str(function_name or "").strip()
        if not target or not self._cap_index_loaded:
            return None
        for entry in self._cap_entries:
            if str(entry.get("function_name") or "").strip() == target:
                return dict(entry)
        return None

    # =====================================================================
    # Capability Index — 可达性评估 & 类型链校验
    # =====================================================================

    def _evaluate_capability_reachability(
        self, tools: List[str], task_type: str = ""
    ) -> Dict[str, Any]:
        """
        评估一组工具（库名）在 capability index 中的可达性。
        """
        if not self._cap_index_loaded:
            return {
                "reachable": [],
                "unreachable": [],
                "status": "index_unavailable",
                "reachable_count": 0,
                "total_count": len(tools),
                "details": {},
            }

        reachable: List[str] = []
        unreachable: List[str] = []
        details: Dict[str, Dict[str, Any]] = {}

        for tool in tools:
            tool_key = self._normalize_tool_package_name(tool)
            pkg_entries = self._idx_by_package.get(tool_key, [])
            in_index = len(pkg_entries) > 0
            is_infra = tool_key in _INFRASTRUCTURE_PACKAGES

            task_type_match = False
            if in_index and task_type:
                task_entries = set(self._idx_by_task_type.get(task_type, []))
                task_type_match = bool(set(pkg_entries) & task_entries)

            details[tool] = {
                "in_index": in_index,
                "is_infrastructure": is_infra,
                "entry_count": len(pkg_entries),
                "task_type_match": task_type_match,
            }

            if in_index or is_infra:
                reachable.append(tool)
            else:
                unreachable.append(tool)

        total = len(tools)
        r_count = len(reachable)
        if total == 0:
            status = "all_reachable"
        elif r_count == total:
            status = "all_reachable"
        elif r_count == 0:
            status = "none_reachable"
        else:
            status = "partial"

        return {
            "reachable": reachable,
            "unreachable": unreachable,
            "status": status,
            "reachable_count": r_count,
            "total_count": total,
            "details": details,
        }

    # =====================================================================
    # Step Schema
    # =====================================================================

    def _build_step_schema(self) -> Dict[str, Any]:
        """
        构建 LLM 输出 schema（从 _build_plan_schema 重命名而来）。

        要求 LLM 输出显式 inputs / outputs 产物引用，
        implementation_hint / implementation_candidates / tool_reachability
        由后处理流水线注入。
        """
        task_type_enum = sorted(self._idx_by_task_type.keys()) if self._cap_index_loaded else []
        data_type_enum = sorted(
            set(self._idx_by_input_type.keys()) | set(self._idx_by_output_type.keys())
        ) if self._cap_index_loaded else []

        task_type_schema: Dict[str, Any] = {
            "type": "string",
            "description": "Task type classification for this step.",
        }
        if task_type_enum:
            task_type_schema["enum"] = task_type_enum
            task_type_schema["description"] += f" Choose from: {task_type_enum}"

        data_type_schema: Dict[str, Any] = {"type": "string"}
        if data_type_enum:
            data_type_schema["enum"] = data_type_enum

        input_schema: Dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "source_kind": {
                    "type": "string",
                    "enum": sorted(SOURCE_KINDS),
                    "description": (
                        "base_dataset, required_dataset, or step_output. "
                        "The source_kind determines the source_id namespace."
                    ),
                },
                "source_id": {
                    "type": "string",
                    "description": (
                        "Use base_dataset for the provided task dataset, a required_datasets id "
                        "for required_dataset, or an exact prior outputs[].id for step_output."
                    ),
                },
                "data_type": data_type_schema,
                "role": {
                    "type": "string",
                    "description": "Scientific or operational role of this input in the step.",
                },
            },
            "required": ["source_kind", "source_id", "data_type", "role"],
        }
        output_schema: Dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Globally unique artifact id referenced by later step inputs.",
                },
                "data_type": data_type_schema,
                "role": {
                    "type": "string",
                    "enum": sorted(OUTPUT_ROLES),
                },
            },
            "required": ["id", "data_type", "role"],
        }

        # 一个 identity 维度：裸值/列表默认 any（同义或互斥备选），
        # 复合量需显式 {"values": [...], "match": "all"}。
        identity_dimension: Dict[str, Any] = {
            "anyOf": [
                {"type": "string"},
                {"type": "array", "items": {"type": "string"}},
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "values": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string"},
                        },
                        "match": {
                            "type": "string",
                            "enum": list(FROZEN_IDENTITY_MATCH_MODES),
                        },
                    },
                    "required": ["values", "match"],
                },
            ]
        }

        required_dataset_schema: Dict[str, Any] = {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": (
                        "Unique requirement ID referenced by a task_plan input whose "
                        "source_kind is required_dataset."
                    ),
                },
                "name": {
                    "type": "string",
                    "description": "Human-readable external dataset or dataset-family name.",
                },
                "canonical_requirement_name": {"type": "string"},
                "research_justification": {
                    "type": "string",
                    "description": (
                        "Why this additional dataset is needed to test the accepted idea. "
                        "Keep it to at most two short sentences."
                    ),
                },
                "execution_requirement_level": {
                    "type": "string",
                    "enum": ["required", "recommended", "optional"],
                },
                "binding_notes": {
                    "type": "string",
                    "description": "Identity, family, region, time, and signal constraints for ResourceCheck.",
                },
                "modality": {"type": "string"},
                "source_preference": {"type": "string"},
                "retrieval_hint": {"type": "string"},
                "temporal_requirement_level": {
                    "type": "string",
                    "enum": [
                        "not_required",
                        "coverage_only",
                        "coverage_and_resolution",
                    ],
                    "description": (
                        "Declare whether time is irrelevant, only temporal coverage is "
                        "required, or both coverage and cadence are required."
                    ),
                },
                "preferred_family": {"type": "string"},
                "allowed_product_families": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "aliases": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "spatial_resolution": {"type": "string"},
                "temporal_coverage": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "start": {"type": "string"},
                        "end": {"type": "string"},
                        "coverage_mode": {
                            "type": "string",
                            "enum": ["full_cover", "overlap"],
                        },
                    },
                    "required": ["start", "end", "coverage_mode"],
                },
                "temporal_resolution": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "unit": {
                            "type": "string",
                            "enum": ["hour", "day", "month", "year"],
                        },
                        "interval": {"type": "number", "exclusiveMinimum": 0},
                        "allow_finer_resolution": {"type": "boolean"},
                    },
                    "required": ["unit", "interval", "allow_finer_resolution"],
                },
                "frozen_constraints": {
                    "type": "object",
                    "properties": {
                        "entity": identity_dimension,
                        "observable": identity_dimension,
                        "measure": identity_dimension,
                        "dataset_family": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "array", "items": {"type": "string"}},
                            ]
                        },
                        "region": {"type": "string"},
                        "time": {"type": "string"},
                    },
                    "description": (
                        "Non-substitutable dataset identity constraints. Use entity, "
                        "observable, and measure whenever the scientific signal has them. "
                        "Multiple values in one dimension mean ALTERNATIVES by default: a "
                        "candidate matching any one of them satisfies that dimension. Use "
                        "this for synonyms (soil_moisture / water_content) and for mutually "
                        "exclusive sources (GRACE / GRACE-FO / GLDAS) — no single product can "
                        "be all of them at once. Only when every listed value must hold "
                        "simultaneously, because the signal is a composite quantity (climatic "
                        "water balance = precipitation minus evapotranspiration), write "
                        "{\"values\": [...], \"match\": \"all\"}. Do not use match=all to list "
                        "synonyms or candidate sources; that makes the requirement "
                        "unsatisfiable. Every frozen value must be both scientifically "
                        "non-substitutable and something catalog evidence can verify. Do not "
                        "freeze a planned output encoding or downstream representation such as "
                        "binary/fraction unless the source dataset must natively provide it."
                    ),
                },
                "soft_constraints": {"type": "object"},
            },
            "required": [
                "id",
                "name",
                "research_justification",
                "execution_requirement_level",
                "binding_notes",
                "modality",
                "source_preference",
                "retrieval_hint",
                "temporal_requirement_level",
            ],
        }

        return {
            "type": "object",
            "properties": {
                "rationale": {
                    "type": "string",
                    "description": "Brief explanation of how the step plan derives from the idea.",
                },
                "required_datasets": {
                    "type": "array",
                    "items": required_dataset_schema,
                    "description": (
                        "Additional external dataset requirements. Return an empty array when the "
                        "provided base dataset is sufficient for the accepted idea."
                    ),
                },
                "task_plan": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "id": {"type": "integer"},
                            "step_name": {"type": "string"},
                            "description": {
                                "type": "string",
                                "description": "Detailed instruction for this step.",
                            },
                            "tools": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Python libraries or tools needed.",
                            },
                            "inputs": {
                                "type": "array",
                                "minItems": 1,
                                "items": input_schema,
                                "description": "Explicit dataset or prior-artifact inputs.",
                            },
                            "outputs": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 1,
                                "items": output_schema,
                                "description": "Exactly one primary artifact produced by this step.",
                            },
                            "task_type": task_type_schema,
                        },
                        "required": [
                            "id",
                            "step_name",
                            "description",
                            "tools",
                            "inputs",
                            "outputs",
                            "task_type",
                        ],
                    },
                },
            },
            "required": ["required_datasets", "task_plan"],
        }

    def _runtime_planning_config(self) -> Tuple[float, int, int, int]:
        """Validate the explicit ours-only hierarchical planning configuration."""
        if self.hierarchical_generation is not True:
            raise AgentExecutionError(
                "step_plan.hierarchical_generation must be explicitly true for ours"
            )
        if not isinstance(self.planning_temperature, (int, float)) or isinstance(
            self.planning_temperature, bool
        ):
            raise AgentExecutionError("step_plan.temperature must be numeric")
        if not isinstance(self.planning_max_tokens, int) or isinstance(
            self.planning_max_tokens, bool
        ) or self.planning_max_tokens <= 0:
            raise AgentExecutionError("step_plan.max_tokens must be a positive integer")
        for field_name, value in (
            ("graph_repair_attempts", self.graph_repair_attempts),
            ("detail_repair_attempts", self.detail_repair_attempts),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise AgentExecutionError(
                    f"step_plan.{field_name} must be a non-negative integer"
                )
        return (
            float(self.planning_temperature),
            self.planning_max_tokens,
            self.graph_repair_attempts,
            self.detail_repair_attempts,
        )

    def _build_graph_schema(self) -> Dict[str, Any]:
        """Return the short graph-only schema used by ours phase one."""
        schema = copy.deepcopy(self._build_step_schema())
        step_schema = schema["properties"]["task_plan"]["items"]
        step_schema["properties"].pop("description")
        step_schema["properties"].pop("tools")
        step_schema["required"] = [
            field
            for field in step_schema["required"]
            if field not in {"description", "tools"}
        ]
        return schema

    def _build_requirement_draft_schema(self) -> Dict[str, Any]:
        """Return the requirement-only schema used before any final plan is generated."""
        requirement_array = copy.deepcopy(
            self._build_step_schema()["properties"]["required_datasets"]
        )
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {"required_datasets": requirement_array},
            "required": ["required_datasets"],
        }

    def _build_detail_schema(self, step_ids: List[int]) -> Dict[str, Any]:
        """Build a detail-only schema whose rows must match frozen graph step IDs."""
        if not step_ids:
            raise ValueError("detail schema requires at least one step id")
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "step_details": {
                    "type": "array",
                    "minItems": len(step_ids),
                    "maxItems": len(step_ids),
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "id": {"type": "integer", "enum": sorted(step_ids)},
                            "description": {"type": "string", "minLength": 1},
                            "tools": {
                                "type": "array",
                                "items": {"type": "string", "minLength": 1},
                            },
                            "required_runtime_tools": {
                                "type": "array",
                                "items": {"type": "string", "minLength": 1},
                                "description": (
                                    "Optional runtime dependencies beyond the selected "
                                    "implementation package that must block execution if unavailable."
                                ),
                            },
                        },
                        "required": ["id", "description", "tools"],
                    },
                }
            },
            "required": ["step_details"],
        }

    def _build_graph_patch_schema(self) -> Dict[str, Any]:
        """Build the guarded partial replacement contract for a graph repair."""
        graph_schema = self._build_graph_schema()
        requirement_schema = copy.deepcopy(
            graph_schema["properties"]["required_datasets"]["items"]
        )
        step_schema = copy.deepcopy(graph_schema["properties"]["task_plan"]["items"])
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "required_datasets": {
                    "type": "array",
                    "items": requirement_schema,
                },
                "step_replacements": {"type": "array", "items": step_schema},
                "step_additions": {"type": "array", "items": step_schema},
                "step_removals": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
                "repair_summary": {"type": "string"},
            },
            "required": [
                "required_datasets",
                "step_replacements",
                "step_additions",
                "step_removals",
                "repair_summary",
            ],
        }

    def _build_scoped_patch_schema(self, allowed_step_ids: List[int]) -> Dict[str, Any]:
        """Extend graph repair with an explicit full-replan escape and detail updates."""
        schema = self._build_graph_patch_schema()
        detail_item = copy.deepcopy(
            self._build_detail_schema(allowed_step_ids)["properties"]["step_details"]["items"]
        )
        detail_item["properties"]["id"] = {"type": "integer"}
        schema["properties"].update(
            {
                "route": {
                    "type": "string",
                    "enum": ["local_patch", "full_replan_required"],
                },
                "route_reason": {"type": "string"},
                "detail_updates": {"type": "array", "items": detail_item},
            }
        )
        schema["required"].extend(["route", "route_reason", "detail_updates"])
        return schema

    # =====================================================================
    # 结构校验
    # =====================================================================

    def _validate_structured_step_contract(self, steps: List[Dict[str, Any]]) -> List[str]:
        """对锚定后的步骤执行强结构校验，确保下游可以依赖新增字段。"""
        required_fields = [
            "task_type",
            "inputs",
            "outputs",
            "implementation_hint",
            "implementation_candidates",
            "tool_reachability",
        ]
        hint_fields = [
            "function_name",
            "package",
            "signature",
            "validation_level",
            "confidence",
            "selection_reason",
        ]

        issues: List[str] = []
        for step in steps:
            if not isinstance(step, dict):
                issues.append("non-dict step found in task_plan")
                continue
            step_id = step.get("id", "?")
            for field in required_fields:
                if field not in step:
                    issues.append(f"step {step_id} missing {field}")
            hint = step.get("implementation_hint") or {}
            if not isinstance(hint, dict):
                issues.append(f"step {step_id} implementation_hint is not an object")
            else:
                for field in hint_fields:
                    if field not in hint:
                        issues.append(f"step {step_id} implementation_hint missing {field}")
            if not isinstance(step.get("implementation_candidates", []), list):
                issues.append(f"step {step_id} implementation_candidates is not a list")
            if not isinstance(step.get("inputs"), list):
                issues.append(f"step {step_id} inputs is not a list")
            if not isinstance(step.get("outputs"), list):
                issues.append(f"step {step_id} outputs is not a list")
            if not step.get("task_type"):
                issues.append(f"step {step_id} task_type is empty")
        return issues

    def _validate_required_dataset_contract(
        self,
        required_datasets: Any,
        steps: List[Dict[str, Any]],
    ) -> List[str]:
        """Validate semantic fields of the top-level dataset requirements."""
        if not isinstance(required_datasets, list):
            return ["required_datasets is not a list"]

        required_fields = [
            "id",
            "name",
            "research_justification",
            "execution_requirement_level",
            "binding_notes",
            "modality",
            "source_preference",
            "retrieval_hint",
            "temporal_requirement_level",
        ]
        allowed_levels = {"required", "recommended", "optional"}
        allowed_temporal_levels = {
            "not_required",
            "coverage_only",
            "coverage_and_resolution",
        }
        issues: List[str] = []
        seen_ids: set[str] = set()

        for index, requirement in enumerate(required_datasets):
            if not isinstance(requirement, dict):
                issues.append(f"required_datasets[{index}] is not an object")
                continue
            missing_fields = [
                field
                for field in required_fields
                if field not in requirement or requirement.get(field) in (None, "")
            ]
            if missing_fields:
                issues.append(
                    f"required_datasets[{index}] missing non-empty fields: {missing_fields}"
                )

            requirement_id = str(requirement.get("id") or "").strip()
            if requirement_id:
                if requirement_id in seen_ids:
                    issues.append(
                        f"required_datasets contains duplicate id: {requirement_id}"
                    )
                seen_ids.add(requirement_id)

            requirement_level = str(
                requirement.get("execution_requirement_level") or ""
            ).strip().lower()
            if requirement_level and requirement_level not in allowed_levels:
                issues.append(
                    "required_datasets["
                    f"{index}] has invalid execution_requirement_level: {requirement_level}"
                )

            if requirement.get("time_range") not in (None, ""):
                issues.append(
                    f"required_datasets[{index}].time_range is not an authoritative field; "
                    "use structured temporal_coverage"
                )

            temporal_coverage = requirement.get("temporal_coverage")
            if temporal_coverage is not None:
                normalized_coverage = normalize_temporal_coverage(temporal_coverage)
                if not normalized_coverage:
                    issues.append(
                        f"required_datasets[{index}].temporal_coverage must contain ISO "
                        "start/end at date or finer granularity; end may be 'present'"
                    )
                elif normalized_coverage.get("coverage_mode") not in (
                    "full_cover",
                    "overlap",
                ):
                    # 下游 classify_temporal_coverage 只认这两个精确取值，其余一律
                    # 归为 invalid_requirement，所以缺失或拼写不符必须在此 fail closed。
                    # 不默认补值：补 overlap 会静默放宽全覆盖需求，补 full_cover 会
                    # 凭空收紧并误杀合法候选。
                    issues.append(
                        f"required_datasets[{index}].temporal_coverage.coverage_mode "
                        "must be exactly full_cover or overlap"
                    )

            temporal_resolution = requirement.get("temporal_resolution")
            if temporal_resolution is not None:
                normalized_resolution = normalize_temporal_resolution(
                    temporal_resolution
                )
                if not isinstance(temporal_resolution, dict):
                    issues.append(
                        f"required_datasets[{index}].temporal_resolution must be an object"
                    )
                elif (
                    not normalized_resolution
                    or normalized_resolution != temporal_resolution
                    or "allow_finer_resolution" not in temporal_resolution
                ):
                    issues.append(
                        f"required_datasets[{index}].temporal_resolution must contain unit, "
                        "positive interval, and boolean allow_finer_resolution"
                    )

            temporal_level = str(
                requirement.get("temporal_requirement_level") or ""
            ).strip()
            if temporal_level and temporal_level not in allowed_temporal_levels:
                issues.append(
                    f"required_datasets[{index}] has invalid temporal_requirement_level: "
                    f"{temporal_level}"
                )
            elif temporal_level == "not_required":
                # 归一化阶段已静默剥离多余的 temporal_coverage / temporal_resolution，
                # 此处不再报错。
                pass
            elif temporal_level == "coverage_only":
                # 缺 temporal_coverage 不再是 blocking issue：归一化已把这种 draft
                # 降级为 not_required。此处只保留 coverage_only 与 resolution 互斥。
                if temporal_resolution is not None:
                    issues.append(
                        f"required_datasets[{index}].temporal_requirement_level=coverage_only "
                        "forbids temporal_resolution"
                    )
            elif temporal_level == "coverage_and_resolution":
                if temporal_coverage is None:
                    issues.append(
                        f"required_datasets[{index}].temporal_requirement_level="
                        "coverage_and_resolution requires temporal_coverage"
                    )
                if temporal_resolution is None:
                    issues.append(
                        f"required_datasets[{index}].temporal_requirement_level="
                        "coverage_and_resolution requires temporal_resolution"
                    )

            frozen_constraints = requirement.get("frozen_constraints") or {}
            if isinstance(frozen_constraints, dict):
                for dimension in FROZEN_IDENTITY_DIMENSIONS:
                    raw_dimension = frozen_constraints.get(dimension)
                    if raw_dimension is None:
                        continue
                    # `match` 是模型可见的枚举合同，非法值必须显式暴露而不是
                    # 被读取侧静默当成 any——否则想表达 all 的复合量会被悄悄放宽。
                    if isinstance(raw_dimension, dict):
                        declared_match = str(raw_dimension.get("match", "") or "").strip()
                        if declared_match not in FROZEN_IDENTITY_MATCH_MODES:
                            issues.append(
                                f"required_datasets[{index}].frozen_constraints.{dimension} "
                                f"has invalid match={declared_match!r}; expected one of "
                                f"{sorted(FROZEN_IDENTITY_MATCH_MODES)}"
                            )
                    values, _ = normalize_frozen_identity_dimension(raw_dimension)
                    if dimension != "entity":
                        continue
                    for entity in values:
                        if extract_region_terms(entity):
                            issues.append(
                                f"required_datasets[{index}] has region-like value in "
                                f"frozen_constraints.entity: {entity!r}; use "
                                "frozen_constraints.region"
                            )

        return issues

    @staticmethod
    def _infer_source_preference(requirement: Dict[str, Any]) -> str:
        """按需求推导检索后端偏好。

        优先读 dataset_source（候选数据集在 scout 阶段打上的真实平台标签），
        其次靠文本证据推断。dataset_source 准确，文本推断易误判（例如 id 里
        含 / 就当 GEE，实测会把 NASA_CMR collection 误判成 GEE 资产）。
        """
        dataset_source = str(requirement.get("dataset_source") or "").strip().upper()
        if dataset_source in {"GEE_OFFICIAL", "GEE_COMMUNITY"}:
            return "GEE"
        if dataset_source in {"NASA_CMR", "PLANETARY_COMPUTER", "EARTH_SEARCH"}:
            return "STAC"
        if dataset_source == "OSM_TAGINFO":
            return "OSM"

        # 回退到文本推断。注意这里不再把 "/" 当作 GEE 的强信号——STAC collection id
        # 也常含 /（如 sentinel-2-l2a），靠它区分会误伤。
        evidence = " ".join(
            str(requirement.get(field) or "")
            for field in ("name", "binding_notes", "retrieval_hint", "preferred_family")
        ).casefold()
        if any(token in evidence for token in ("gee", "earth engine", "asset id")):
            return "GEE"
        if "osm" in evidence or "openstreetmap" in evidence:
            return "OSM"
        if any(token in evidence for token in ("stac", "catalog", "collection")):
            return "STAC"
        return ""

    def _normalize_required_dataset_draft(
        self,
        required_datasets: Any,
    ) -> Tuple[Any, List[Dict[str, Any]]]:
        """Normalize model draft noise before strict requirement validation."""
        if not isinstance(required_datasets, list):
            return required_datasets, []

        normalized: List[Dict[str, Any]] = []
        actions: List[Dict[str, Any]] = []
        for index, requirement in enumerate(required_datasets):
            if not isinstance(requirement, dict):
                normalized.append(requirement)
                continue
            item = copy.deepcopy(requirement)

            temporal_level = str(
                item.get("temporal_requirement_level") or ""
            ).strip()
            if temporal_level == "coverage_only" and "temporal_resolution" in item:
                item.pop("temporal_resolution", None)
                actions.append({
                    "index": index,
                    "action": "drop_temporal_resolution_for_coverage_only",
                })

            # 一个声明了 coverage_only 却没给出 coverage 的 draft 表达的是"没有可
            # 断言的时间约束"，不是"时间约束不可满足"。降级为 not_required 让它按无
            # 时间门禁继续走 ResourceCheck（下游 classify_temporal_coverage 对空
            # required 本来就返回 not_required），而不是让整个 StepPlan fail closed。
            # 降级只放宽本 requirement 的时间门禁，不放宽 hard conflict：真正的时间
            # 不可达仍由候选侧的 temporal_no_overlap 一类 blocking reason 拦截。
            # coverage_and_resolution 不在此降级：它显式断言 cadence 也重要，缺
            # coverage 属于自相矛盾，且降级会连带丢掉已声明的 temporal_resolution。
            if temporal_level == "coverage_only" and item.get("temporal_coverage") is None:
                item["temporal_requirement_level"] = "not_required"
                actions.append({
                    "index": index,
                    "action": "demote_coverage_only_missing_coverage",
                    "from": "coverage_only",
                    "to": "not_required",
                })
                temporal_level = "not_required"

            # not_required 表示该数据集无时间约束；LLM 可能附带多余的
            # temporal_coverage / temporal_resolution，静默剥离而非报错。
            if temporal_level == "not_required":
                if "temporal_coverage" in item:
                    item.pop("temporal_coverage", None)
                    actions.append({
                        "index": index,
                        "action": "drop_temporal_coverage_for_not_required",
                    })
                if "temporal_resolution" in item:
                    item.pop("temporal_resolution", None)
                    actions.append({
                        "index": index,
                        "action": "drop_temporal_resolution_for_not_required",
                    })

            # LLM 常把 coverage 写成带时刻和时区的 ISO datetime（例如
            # `2018-01-01T00:00:00Z`），这与写成 `2018-01-01` 表达的是同一个日粒度
            # 区间。normalize_temporal_coverage 本来就吸收 Z 后缀并降到日粒度，这里
            # 直接把归一化结果写回，让合同校验只判"能否归一化"，而不是"是否已经等于
            # 归一化形式"。归一化失败（返回空）保持原值不动，交给
            # _validate_required_dataset_contract 报错，不在这里静默丢弃时间约束。
            raw_coverage = item.get("temporal_coverage")
            if isinstance(raw_coverage, dict):
                normalized_coverage = normalize_temporal_coverage(raw_coverage)
                if normalized_coverage and normalized_coverage != raw_coverage:
                    item["temporal_coverage"] = normalized_coverage
                    actions.append({
                        "index": index,
                        "action": "normalize_temporal_coverage",
                        "from": copy.deepcopy(raw_coverage),
                        "to": copy.deepcopy(normalized_coverage),
                    })

            if item.get("source_preference") in (None, ""):
                inferred_source = self._infer_source_preference(item)
                if inferred_source:
                    item["source_preference"] = inferred_source
                    actions.append({
                        "index": index,
                        "action": "infer_source_preference",
                        "value": inferred_source,
                    })

            frozen_constraints = item.get("frozen_constraints")
            if isinstance(frozen_constraints, dict):
                constraints = copy.deepcopy(frozen_constraints)
                raw_entity = constraints.get("entity")
                values, match_mode = normalize_frozen_identity_dimension(raw_entity)
                if values:
                    identity_values: List[str] = []
                    region_values: List[str] = []
                    for value in values:
                        if extract_region_terms(value):
                            region_values.append(value)
                        else:
                            identity_values.append(value)
                    if region_values:
                        existing_region = str(constraints.get("region") or "").strip()
                        region_value = existing_region or ", ".join(region_values)
                        constraints["region"] = region_value
                        if identity_values:
                            if isinstance(raw_entity, dict):
                                constraints["entity"] = {
                                    "values": identity_values,
                                    "match": match_mode,
                                }
                            else:
                                constraints["entity"] = identity_values
                        else:
                            constraints.pop("entity", None)
                        item["frozen_constraints"] = constraints
                        actions.append({
                            "index": index,
                            "action": "move_region_like_entity_to_region",
                            "values": region_values,
                            "region": region_value,
                        })

            normalized.append(item)

        return normalized, actions

    @staticmethod
    def _required_dataset_contracts_match(
        expected: List[Dict[str, Any]],
        actual: Any,
    ) -> bool:
        if not isinstance(actual, list) or len(expected) != len(actual):
            return False
        try:
            expected_by_id = {
                str(item["id"]): requirement_fingerprint(item)
                for item in expected
                if isinstance(item, dict)
            }
            actual_by_id = {
                str(item["id"]): requirement_fingerprint(item)
                for item in actual
                if isinstance(item, dict)
            }
        except (KeyError, TypeError, ValueError):
            return False
        return len(expected_by_id) == len(expected) and expected_by_id == actual_by_id

    async def _call_planning_phase(
        self,
        *,
        phase: str,
        prompt: str,
        schema: Dict[str, Any],
        audit: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run one explicit hierarchical phase and record its observable call contract."""
        temperature, max_tokens, _, _ = self._runtime_planning_config()
        call_audit = {
            "phase": phase,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_chars": len(prompt),
            "schema_chars": len(json.dumps(schema, ensure_ascii=False)),
            "status": "started",
        }
        audit.setdefault("model_calls", []).append(call_audit)
        started_at = time.monotonic()
        try:
            result = await self._call_model(
                prompt=prompt,
                schema=schema,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            call_audit["status"] = "error"
            call_audit["error"] = str(exc)
            call_audit["elapsed_seconds"] = round(time.monotonic() - started_at, 6)
            raise
        if not isinstance(result, dict):
            call_audit["status"] = "invalid_response_type"
            call_audit["elapsed_seconds"] = round(time.monotonic() - started_at, 6)
            raise AgentExecutionError(
                f"StepPlan {phase} response must be a JSON object"
            )
        call_audit["status"] = "completed"
        call_audit["elapsed_seconds"] = round(time.monotonic() - started_at, 6)
        call_audit["output_chars"] = len(json.dumps(result, ensure_ascii=False))
        return result

    @staticmethod
    def _graph_step(step: Any) -> Dict[str, Any]:
        if not isinstance(step, dict):
            return {}
        return {
            field: copy.deepcopy(step[field])
            for field in ("id", "step_name", "task_type", "inputs", "outputs")
            if field in step
        }

    @staticmethod
    def _patch_context_step(step: Any) -> Dict[str, Any]:
        """Project an anchored step down to what a patch decision actually needs.

        `implementation_candidates` is deterministic capability lookup evidence: it is
        consumed by the local selection logic, never by the model. A single anchored
        step can carry well over a thousand candidate entries, so echoing it back into
        the SCOPED_PLAN_PATCH prompt costs six figures of tokens without informing the
        patch decision.
        """
        if not isinstance(step, dict):
            return {}
        return {
            field: value
            for field, value in step.items()
            if field != "implementation_candidates"
        }

    # 候选证据投影预算：候选清单是 non-binding 证据，只用于让模型认出数据身份、
    # 判断可替代性并写出可检索的 retrieval_hint，不用于生成代码。
    CANDIDATE_DESCRIPTION_CHARS = 400
    CANDIDATE_BANDS_TOP_N = 8
    CANDIDATE_EVIDENCE_TOP_N = 12

    @staticmethod
    def _compact_candidate_evidence(candidate: Any) -> Dict[str, Any]:
        """Project a candidate dataset card down to identity-bearing fields.

        `detail.ee_snippets` / `detail.sample_code` are Earth Engine code samples and
        `binding_decision` is ResourceCheck's own scoring trace: neither informs the
        requirement identity the model must write. Real inventories reach 101K chars
        for 31 candidates, of which `detail` is 62% and `binding_decision` 18%.
        """
        if not isinstance(candidate, dict):
            return {}
        card = {
            field: copy.deepcopy(candidate[field])
            for field in (
                "id",
                "title",
                "status",
                "product_family",
                "observable",
                "source",
                "provider",
                "type",
                "dataset_source",
                "retrieval_channel",
                "retrieval_axes",
                "naming_note",
            )
            if candidate.get(field) not in (None, "", [], {})
        }
        detail = candidate.get("detail")
        if isinstance(detail, dict):
            compact_detail = {
                field: copy.deepcopy(detail[field])
                for field in (
                    "detail_source",
                    "gsd",
                    "cadence",
                    "temporal_extent",
                    "spatial_extent",
                    "spatial_bbox",
                )
                if detail.get(field) not in (None, "", [], {})
            }
            description = str(detail.get("description") or "").strip()
            if description:
                compact_detail["description"] = description[
                    : StepPlanAgent.CANDIDATE_DESCRIPTION_CHARS
                ]
            bands = detail.get("bands")
            if isinstance(bands, list) and bands:
                compact_detail["bands"] = copy.deepcopy(
                    bands[: StepPlanAgent.CANDIDATE_BANDS_TOP_N]
                )
            if compact_detail:
                card["detail"] = compact_detail
        return card

    @classmethod
    def _candidate_evidence_block(cls, candidates: Any) -> List[Dict[str, Any]]:
        """Cap and compact the candidate evidence injected into a StepPlan prompt."""
        if not isinstance(candidates, list):
            return []
        compacted = [
            card
            for card in (cls._compact_candidate_evidence(item) for item in candidates)
            if card
        ]
        return compacted[: cls.CANDIDATE_EVIDENCE_TOP_N]

    # 单条 resolution_log 的 attempts 实测可达 40 项 / 15.4K 字符，且按 query×strategy 无界增长。
    # 修复决策只需要知道失败原因和最后几次尝试打到了什么，不需要完整检索轨迹。
    RESOLUTION_LOG_ATTEMPT_TAIL = 6

    @classmethod
    def _compact_resolution_log(cls, resolution_log: Any) -> List[Dict[str, Any]]:
        """Keep the failure semantics of each log entry, drop the search trace bulk."""
        if not isinstance(resolution_log, list):
            return []
        compacted: List[Dict[str, Any]] = []
        for entry in resolution_log:
            if not isinstance(entry, dict):
                continue
            projected = {
                field: copy.deepcopy(entry[field])
                for field in (
                    "requirement_id",
                    "dataset",
                    "status",
                    "strategy",
                    "reasons",
                    "unbound_class",
                    "confidence_flags",
                )
                if entry.get(field) not in (None, "", [], {})
            }
            top_candidates = entry.get("top_candidates")
            if isinstance(top_candidates, list) and top_candidates:
                # 每条 top_candidate 内嵌 ResourceCheck 的 binding_decision 打分轨迹，
                # 实测占该字段 87% 体积；修复只需要知道检索命中了哪些数据身份。
                projected["top_candidates"] = [
                    {
                        field: copy.deepcopy(candidate[field])
                        for field in ("id", "title", "name", "type", "status")
                        if isinstance(candidate, dict)
                        and candidate.get(field) not in (None, "", [], {})
                    }
                    for candidate in top_candidates
                    if isinstance(candidate, dict)
                ]
            attempts = entry.get("attempts")
            if isinstance(attempts, list) and attempts:
                tail = attempts[-cls.RESOLUTION_LOG_ATTEMPT_TAIL :]
                projected["attempts"] = [
                    {
                        field: copy.deepcopy(attempt[field])
                        for field in ("keyword", "strategy", "status", "found", "reasons")
                        if isinstance(attempt, dict)
                        and attempt.get(field) not in (None, "", [], {})
                    }
                    for attempt in tail
                    if isinstance(attempt, dict)
                ]
                omitted = len(attempts) - len(tail)
                if omitted > 0:
                    projected["attempts_omitted_count"] = omitted
            if projected:
                compacted.append(projected)
        return compacted

    def _graph_contract_issues(
        self,
        required_datasets: Any,
        steps: Any,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Validate the graph phase without asking for post-anchor fields."""
        validation = validate_plan_dag(required_datasets, steps)
        issues = copy.deepcopy(validation.issues)
        for message in self._validate_required_dataset_contract(required_datasets, steps):
            issues.append({"code": "required_dataset_contract_invalid", "message": message})

        task_type_enum = set(self._idx_by_task_type) if self._cap_index_loaded else set()
        if isinstance(steps, list):
            for index, step in enumerate(steps):
                if not isinstance(step, dict):
                    continue
                step_id = step.get("id", index + 1)
                if not isinstance(step.get("step_name"), str) or not step["step_name"].strip():
                    issues.append(
                        {
                            "code": "step_name_invalid",
                            "message": f"step {step_id}.step_name must be a non-empty string",
                            "step_id": step_id,
                        }
                    )
                task_type = step.get("task_type")
                if not isinstance(task_type, str) or not task_type.strip():
                    issues.append(
                        {
                            "code": "step_task_type_invalid",
                            "message": f"step {step_id}.task_type must be a non-empty string",
                            "step_id": step_id,
                        }
                    )
                elif task_type_enum and task_type not in task_type_enum:
                    issues.append(
                        {
                            "code": "step_task_type_unknown",
                            "message": f"step {step_id}.task_type is not in the capability index: {task_type}",
                            "step_id": step_id,
                        }
                    )
        meta = validation.to_dict()
        meta["plan_dag_valid"] = not issues
        meta["plan_dag_issues"] = copy.deepcopy(issues)
        return issues, meta

    @staticmethod
    def _issue_step_ids(
        issues: List[Dict[str, Any]],
        steps: List[Dict[str, Any]],
    ) -> set[int]:
        existing_ids = {
            step.get("id")
            for step in steps
            if isinstance(step, dict)
            and isinstance(step.get("id"), int)
            and not isinstance(step.get("id"), bool)
        }
        affected: set[int] = set()
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            for field in ("step_id", "from_step_id", "to_step_id"):
                value = issue.get(field)
                if isinstance(value, int) and not isinstance(value, bool):
                    affected.add(value)
            for value in issue.get("step_ids", []) or []:
                if isinstance(value, int) and not isinstance(value, bool):
                    affected.add(value)
        return (affected & existing_ids) or existing_ids

    @staticmethod
    def _requires_graph_regeneration(issues: List[Dict[str, Any]]) -> bool:
        """Return whether node identity is too broken for an ID-scoped patch."""
        identity_issue_codes = {
            "task_plan_not_nonempty_list",
            "step_not_object",
            "step_id_invalid",
            "duplicate_step_id",
        }
        return any(
            isinstance(issue, dict) and issue.get("code") in identity_issue_codes
            for issue in issues
        )

    def _apply_graph_patch(
        self,
        *,
        required_datasets: List[Dict[str, Any]],
        steps: List[Dict[str, Any]],
        patch: Dict[str, Any],
        allowed_step_ids: set[int],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Apply only explicitly allowed graph-node operations."""
        replacement_requirements = patch.get("required_datasets")
        replacements = patch.get("step_replacements")
        additions = patch.get("step_additions")
        removals = patch.get("step_removals")
        if not isinstance(replacement_requirements, list):
            raise AgentExecutionError("graph patch required_datasets must be a list")
        if not all(isinstance(value, list) for value in (replacements, additions, removals)):
            raise AgentExecutionError("graph patch operations must be lists")

        current_by_id = {
            step.get("id"): copy.deepcopy(step)
            for step in steps
            if isinstance(step, dict)
            and isinstance(step.get("id"), int)
            and not isinstance(step.get("id"), bool)
        }
        replacement_by_id: Dict[int, Dict[str, Any]] = {}
        for raw_step in replacements:
            step = self._graph_step(raw_step)
            step_id = step.get("id")
            if (
                not isinstance(step_id, int)
                or isinstance(step_id, bool)
                or step_id not in current_by_id
                or step_id not in allowed_step_ids
            ):
                raise AgentExecutionError(
                    f"graph patch cannot replace step outside repair scope: {step_id}"
                )
            if step_id in replacement_by_id:
                raise AgentExecutionError(f"graph patch replaces step {step_id} more than once")
            replacement_by_id[step_id] = step

        removal_ids: set[int] = set()
        for step_id in removals:
            if not isinstance(step_id, int) or isinstance(step_id, bool):
                raise AgentExecutionError("graph patch step_removals must contain integers")
            if step_id not in current_by_id or step_id not in allowed_step_ids:
                raise AgentExecutionError(
                    f"graph patch cannot remove step outside repair scope: {step_id}"
                )
            removal_ids.add(step_id)
        if removal_ids & set(replacement_by_id):
            raise AgentExecutionError("graph patch cannot replace and remove the same step")

        addition_by_id: Dict[int, Dict[str, Any]] = {}
        for raw_step in additions:
            step = self._graph_step(raw_step)
            step_id = step.get("id")
            if not isinstance(step_id, int) or isinstance(step_id, bool):
                raise AgentExecutionError("graph patch additions require integer ids")
            if step_id in current_by_id or step_id in addition_by_id:
                raise AgentExecutionError(f"graph patch addition has conflicting id: {step_id}")
            addition_by_id[step_id] = step

        patched_steps = []
        for step in steps:
            step_id = step.get("id") if isinstance(step, dict) else None
            if step_id in removal_ids:
                continue
            patched_steps.append(
                copy.deepcopy(replacement_by_id.get(step_id, self._graph_step(step)))
            )
        patched_steps.extend(copy.deepcopy(list(addition_by_id.values())))
        return copy.deepcopy(replacement_requirements), patched_steps

    def _validate_step_details(
        self,
        details: Any,
        expected_step_ids: set[int],
    ) -> Tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]]]:
        detail_by_id: Dict[int, Dict[str, Any]] = {}
        issues: List[Dict[str, Any]] = []
        if not isinstance(details, list):
            return {}, [{"code": "step_details_not_list", "message": "step_details must be a list"}]
        for index, detail in enumerate(details):
            if not isinstance(detail, dict):
                issues.append(
                    {
                        "code": "step_detail_not_object",
                        "message": f"step_details[{index}] must be an object",
                    }
                )
                continue
            step_id = detail.get("id")
            if (
                not isinstance(step_id, int)
                or isinstance(step_id, bool)
                or step_id not in expected_step_ids
            ):
                issues.append(
                    {
                        "code": "step_detail_id_out_of_scope",
                        "message": f"step detail id is outside the requested scope: {step_id}",
                        "step_id": step_id,
                    }
                )
                continue
            if step_id in detail_by_id:
                issues.append(
                    {
                        "code": "duplicate_step_detail",
                        "message": f"step {step_id} has duplicate detail rows",
                        "step_id": step_id,
                    }
                )
                continue
            description = detail.get("description")
            tools = detail.get("tools")
            required_runtime_tools = detail.get("required_runtime_tools", [])
            if not isinstance(description, str) or not description.strip():
                issues.append(
                    {
                        "code": "step_description_invalid",
                        "message": f"step {step_id}.description must be a non-empty string",
                        "step_id": step_id,
                    }
                )
            if not isinstance(tools, list) or not all(
                isinstance(tool, str) and tool.strip() for tool in tools
            ):
                issues.append(
                    {
                        "code": "step_tools_invalid",
                        "message": f"step {step_id}.tools must be a list of non-empty strings",
                        "step_id": step_id,
                    }
                )
            if not isinstance(required_runtime_tools, list) or not all(
                isinstance(tool, str) and tool.strip()
                for tool in required_runtime_tools
            ):
                issues.append(
                    {
                        "code": "step_required_runtime_tools_invalid",
                        "message": (
                            f"step {step_id}.required_runtime_tools must be a list "
                            "of non-empty strings"
                        ),
                        "step_id": step_id,
                    }
                )
            detail_by_id[step_id] = {
                "id": step_id,
                "description": description,
                "tools": copy.deepcopy(tools),
                "required_runtime_tools": copy.deepcopy(required_runtime_tools),
            }
        missing_ids = sorted(expected_step_ids - set(detail_by_id))
        for step_id in missing_ids:
            issues.append(
                {
                    "code": "step_detail_missing",
                    "message": f"step {step_id} is missing detail expansion",
                    "step_id": step_id,
                }
            )
        return detail_by_id, issues

    # =====================================================================
    # 锚定流水线
    # =====================================================================

    def _anchor_plan_to_capabilities(
        self,
        required_datasets: List[Dict[str, Any]],
        steps: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        能力锚定后处理流水线。

        四个阶段：
          1. 候选搜索
          2. 锚点选择
          3. 可达性标注
          4. 显式 DAG 校验

        返回 (anchored_steps, meta)。
        """
        meta: Dict[str, Any] = {
            "capability_index_loaded": self._cap_index_loaded,
            "steps_processed": 0,
            "plan_dag_valid": False,
            "plan_dag_issues": [],
            "plan_dag_edges": [],
            "plan_dag_topological_order": [],
            "critical_output_ids": [],
            "used_required_dataset_ids": [],
        }

        initial_validation = validate_plan_dag(required_datasets, steps)
        meta.update(initial_validation.to_dict())
        if not initial_validation.valid:
            return steps, meta

        enriched: List[Dict[str, Any]] = []

        for step in steps:
            step = dict(step)
            task_type = step.get("task_type", "")
            tools = step.get("tools", [])
            input_types = step_input_data_types(step)
            output_type = step_primary_output_type(step)

            # Phase 1: 候选搜索（全量池，不做 confidence 截断）
            candidates = self._search_implementation_candidates(
                task_type=task_type,
                tools=tools,
                input_types=input_types,
                output_type=output_type,
            )

            # Phase 2: 锚点选择（目标感知，从全量池中选最优）
            step["implementation_hint"] = self._select_implementation_hint(
                candidates,
                task_type,
                step=step,
            )

            # Phase 3: 存储精选候选（确定性兼容优先，控制上下文长度）
            scored = [
                (
                    candidate,
                    evaluate_operation_compatibility(
                        step,
                        self._candidate_with_capability_detail(candidate),
                    ),
                )
                for candidate in candidates
            ]
            proven = [
                self._compact_implementation_candidate(c, compat)
                for c, compat in scored
                if compat.get("compatible")
            ]
            if proven:
                step["implementation_candidates"] = proven
            else:
                # 无确定性兼容候选时保留 top-5 作为 refinement 上下文：
                # 优先取 advisory 候选（仅缺证据、无矛盾，仍可锚定），
                # 全为硬不兼容（forbidden/mismatch）时才退回全量 top-5。
                def _fallback_score(pair):
                    c, _ = pair
                    return (
                        1 if c.get("pip_verified") else 0,
                        1 if c.get("task_type") == task_type else 0,
                        self._function_text_match_score(c, step),
                        self._candidate_io_match_score(c, step),
                        c.get("confidence", 0.0),
                    )

                advisory_pool = [
                    (c, compat) for c, compat in scored if compat.get("advisory")
                ]
                top = sorted(
                    advisory_pool or scored, key=_fallback_score, reverse=True
                )[:5]
                step["implementation_candidates"] = [
                    self._compact_implementation_candidate(c, compat)
                    for c, compat in top
                ]

            # Phase 4: 工具可达性标注
            step["tool_reachability"] = self._evaluate_capability_reachability(tools, task_type)

            enriched.append(step)

        meta["steps_processed"] = len(enriched)

        # Phase 4: shared DAG validation. This stage never repairs the generated plan.
        meta.update(validate_plan_dag(required_datasets, enriched).to_dict())

        return enriched, meta

    def _search_implementation_candidates(
        self,
        task_type: str,
        tools: List[str],
        input_types: List[str],
        output_type: str,
    ) -> List[Dict[str, Any]]:
        """
        为单个步骤搜索候选实现方案。

        返回所有结构匹配的候选（不做 confidence 截断），
        目标感知排序和最终选择由 _select_implementation_hint 负责。

        搜索策略（全部执行，不提前退出）：
          1. task_type + 任一 tool package -> 交集
          2. task_type + input/output 类型约束
          3. 仅 tool packages（不限制 task_type）
        """
        if not self._cap_index_loaded:
            return []

        candidates: List[Dict[str, Any]] = []
        seen_fns: set = set()

        def _add(entries: List[Dict[str, Any]], source: str) -> None:
            for e in entries:
                fn = e.get("function_name", "")
                if fn in seen_fns:
                    continue
                seen_fns.add(fn)
                candidates.append({
                    "function_name": fn,
                    "package": e.get("package", ""),
                    "signature": e.get("signature", ""),
                    "task_type": e.get("task_type", ""),
                    "confidence": e.get("confidence", 0.0),
                    "source": source,
                    "input_data_type": e.get("input_data_type", []),
                    "output_data_type": e.get("output_data_type", ""),
                    "pip_verified": e.get("pip_verified", False),
                    "raw_description": e.get("raw_description", ""),
                })

        # Strategy 1: task_type + package 交集
        for tool in tools:
            hits = self._semantic_search_capabilities(
                task_type=task_type, package=tool
            )
            _add(hits, f"task_type+package({tool})")

        # Strategy 2: task_type + input/output 类型约束
        if task_type:
            hits = self._semantic_search_capabilities(
                task_type=task_type,
                input_types=input_types if input_types else None,
                output_type=output_type,
            )
            _add(hits, "task_type+io_types")

        # Strategy 3: 仅 package
        for tool in tools:
            hits = self._semantic_search_capabilities(package=tool)
            _add(hits, f"package_only({tool})")

        return candidates

    def _candidate_with_capability_detail(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        if self._cap_detail_store is None:
            return dict(candidate)
        detail = self._cap_detail_store.get(candidate)
        return self._cap_detail_store.merge(candidate, detail)

    @staticmethod
    def _compact_signature(signature: Any, limit: int = 180) -> str:
        text = str(signature or "").strip()
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."

    def _compact_implementation_candidate(
        self,
        candidate: Dict[str, Any],
        compatibility: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        compact = {
            "function_name": candidate.get("function_name", ""),
            "package": candidate.get("package", ""),
            "signature": self._compact_signature(candidate.get("signature", "")),
            "task_type": candidate.get("task_type", ""),
            "confidence": candidate.get("confidence", 0.0),
            "source": candidate.get("source", ""),
            "input_data_type": candidate.get("input_data_type", []),
            "output_data_type": candidate.get("output_data_type", ""),
            "pip_verified": candidate.get("pip_verified", False),
        }
        if compatibility:
            compact["operation_compatible"] = bool(compatibility.get("compatible", False))
            compact["operation_compatibility_status"] = compatibility.get("status", "")
            compact["operation_compatibility_advisory"] = bool(
                compatibility.get("advisory", False)
            )
        return compact

    @staticmethod
    def _data_type_family(value: Any) -> str:
        text = str(value or "").casefold()
        if "raster" in text or "image" in text:
            return "raster"
        if "vector" in text or "feature" in text or "geometry" in text:
            return "vector"
        if "table" in text or "dataframe" in text:
            return "table"
        if "timeseries" in text or "time_series" in text:
            return "timeseries"
        return text

    def _candidate_io_match_score(
        self,
        candidate: Dict[str, Any],
        step: Optional[Dict[str, Any]],
    ) -> int:
        if not isinstance(step, dict):
            return 0
        step_families = {
            self._data_type_family(value)
            for value in [
                *step_input_data_types(step),
                step_primary_output_type(step),
            ]
            if value
        }
        candidate_types = candidate.get("input_data_type") or []
        if isinstance(candidate_types, str):
            candidate_types = [candidate_types]
        candidate_families = {
            self._data_type_family(value)
            for value in candidate_types
            if value
        }
        return 1 if step_families.intersection(candidate_families) else 0

    @staticmethod
    def _function_text_match_score(candidate: Dict[str, Any], step: Optional[Dict[str, Any]]) -> int:
        if not isinstance(step, dict):
            return 0
        text = " ".join(
            str(step.get(field) or "")
            for field in ("step_name", "description")
        ).casefold()
        function_name = str(candidate.get("function_name") or "").casefold()
        score = 0
        for token in ("imagecollection", "featurecollection", "image"):
            if token in text and token in function_name:
                score += 1
        return score

    def _deterministic_implementation_fallbacks(
        self,
        step: Optional[Dict[str, Any]],
        task_type: str,
    ) -> List[Dict[str, Any]]:
        """Conservative local templates for common operations missing from search hits."""
        if not isinstance(step, dict):
            return []
        text = " ".join(
            str(step.get(field) or "")
            for field in ("step_name", "description", "task_type")
        ).casefold()
        tools = [
            self._normalize_tool_package_name(str(tool))
            for tool in (step.get("tools") or [])
            if str(tool).strip()
        ]
        input_types = [self._data_type_family(value) for value in step_input_data_types(step)]
        output_type = self._data_type_family(step_primary_output_type(step))

        templates: List[Dict[str, Any]] = []

        def add(
            function_name: str,
            package: str,
            signature: str,
            raw_description: str,
            confidence: float = 0.72,
            input_data_type: Optional[List[str]] = None,
            output_data_type: str = "",
        ) -> None:
            indexed = self._capability_entry_by_function(function_name) or {}
            candidate = {
                "function_name": function_name,
                "package": indexed.get("package") or package,
                "signature": indexed.get("signature") or signature,
                "task_type": indexed.get("task_type") or task_type,
                "confidence": indexed.get("confidence") or confidence,
                "source": "deterministic_operation_template",
                "input_data_type": indexed.get("input_data_type")
                or input_data_type
                or step_input_data_types(step),
                "output_data_type": indexed.get("output_data_type")
                or output_data_type
                or step_primary_output_type(step),
                "pip_verified": bool(indexed.get("pip_verified", False)),
                "raw_description": indexed.get("raw_description") or raw_description,
            }
            templates.append(candidate)

        uses_gee = bool({"ee", "earthengine"}.intersection(set(tools))) or "earth engine" in text or "gee" in text
        if uses_gee and (
            "imagecollection" in text
            or "image collection" in text
            or "raster" in input_types
            or "timeseries" in input_types
            or "timeseries" in output_type
        ):
            add(
                "ee.ImageCollection",
                "ee",
                "ee.ImageCollection(args)",
                "Load and access an Earth Engine ImageCollection for raster time-series data access and catalog retrieval.",
                input_data_type=["raster_timeseries", "raster"],
                output_data_type="raster_timeseries",
            )
        if uses_gee and ("featurecollection" in text or "feature collection" in text or "vector" in input_types):
            add(
                "ee.FeatureCollection",
                "ee",
                "ee.FeatureCollection(args)",
                "Load and access an Earth Engine FeatureCollection for vector region, polygon, or feature data access.",
                input_data_type=["vector"],
                output_data_type="vector",
            )
        if uses_gee and ("image" in text or output_type == "raster"):
            add(
                "ee.Image",
                "ee",
                "ee.Image(args)",
                "Load and access an Earth Engine Image for raster data access and retrieval.",
                input_data_type=["raster"],
                output_data_type="raster",
            )

        if "zonal" in text or (
            any(token in text for token in ("polygon aggregate", "region aggregate", "regional statistics"))
            and "raster" in input_types
        ):
            add(
                "rasterstats.zonal_stats",
                "rasterstats",
                "zonal_stats(vectors, raster, stats=None, **kwargs)",
                "Compute zonal statistics by aggregating raster values over polygon regions or zones.",
                input_data_type=["vector", "raster"],
                output_data_type="table",
            )

        if "merge" in text or "join" in text:
            add(
                "pandas.DataFrame.merge",
                "pandas",
                "DataFrame.merge(right, how='inner', on=None, **kwargs)",
                "Merge or join tabular pandas DataFrame objects by keys into a combined table.",
                input_data_type=["table", "dataframe"],
                output_data_type="table",
            )
        if any(token in text for token in ("groupby", "group by", "aggregate", "aggregation", "summarize", "summary")):
            add(
                "pandas.DataFrame.groupby",
                "pandas",
                "DataFrame.groupby(by=None, **kwargs)",
                "Group by keys and aggregate or summarize tabular pandas DataFrame values.",
                input_data_type=["table", "dataframe"],
                output_data_type="table",
            )

        if any(token in text for token in ("spearman", "rank correlation")):
            add(
                "scipy.stats.spearmanr",
                "scipy",
                "spearmanr(a, b=None, **kwargs)",
                "Compute Spearman rank correlation and association coefficient between variables.",
                input_data_type=["table", "array"],
                output_data_type="statistics",
            )
        if any(token in text for token in ("correlation", "correlate", "pearson")):
            add(
                "scipy.stats.pearsonr",
                "scipy",
                "pearsonr(x, y, **kwargs)",
                "Compute Pearson correlation and association coefficient between variables.",
                input_data_type=["table", "array"],
                output_data_type="statistics",
            )
        if "regression" in text or "regress" in text or "ols" in text:
            add(
                "statsmodels.formula.api.ols",
                "statsmodels",
                "ols(formula, data, subset=None, drop_cols=None)",
                "Fit ordinary least squares regression models for statistical regression analysis.",
                input_data_type=["table", "dataframe"],
                output_data_type="model",
            )

        seen: set[str] = set()
        unique: List[Dict[str, Any]] = []
        for candidate in templates:
            fn = str(candidate.get("function_name") or "")
            if fn and fn not in seen:
                seen.add(fn)
                unique.append(candidate)
        return unique

    def _select_implementation_hint(
        self,
        candidates: List[Dict[str, Any]],
        task_type: str,
        *,
        step: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        从候选列表中选出最佳实现锚点，生成 implementation_hint。
        """
        constraints = (step or {}).get("operation_constraints") or {}
        if not isinstance(constraints, dict):
            constraints = {}
        preferred_names = constraints.get("preferred_function_names") or []
        if isinstance(preferred_names, str):
            preferred_names = [preferred_names]

        def _prefers(candidate: Dict[str, Any]) -> int:
            return (
                1
                if preferred_names
                and function_name_matches(
                    candidate.get("function_name", ""), preferred_names
                )
                else 0
            )

        # 兼容性三级：2=确定性兼容（proven）、1=仅缺证据（advisory，可锚定
        # 但劣后于 proven）、0=硬不兼容（forbidden/mismatch，排除）。
        # advisory 候选可参选是死路修复的关键：步骤散文措辞造成全体候选
        # unproven 时，锚定不应返回空 function_name 走向
        # operation_constraint_unmatched，而应选出最佳可用函数并携带
        # advisory 标记；真正的死路只剩显式矛盾/禁止。
        def _compat_rank(compatibility: Dict[str, Any]) -> int:
            if compatibility.get("compatible"):
                return 2
            if compatibility.get("advisory"):
                return 1
            return 0

        scored_candidates: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        for candidate in candidates:
            detailed_candidate = self._candidate_with_capability_detail(candidate)
            compatibility = evaluate_operation_compatibility(
                step or {},
                detailed_candidate,
            )
            scored_candidates.append((candidate, compatibility))

        eligible_candidates = [
            (candidate, compatibility)
            for candidate, compatibility in scored_candidates
            if _compat_rank(compatibility) > 0
        ]

        if not eligible_candidates:
            fallback_candidates = self._deterministic_implementation_fallbacks(
                step,
                task_type,
            )
            fallback_scored = [
                (
                    candidate,
                    evaluate_operation_compatibility(
                        step or {},
                        self._candidate_with_capability_detail(candidate),
                    ),
                )
                for candidate in fallback_candidates
            ]
            fallback_eligible = [
                (candidate, compatibility)
                for candidate, compatibility in fallback_scored
                if _compat_rank(compatibility) > 0
            ]
            if fallback_eligible:
                best_fallback, fallback_compatibility = max(
                    fallback_eligible,
                    key=lambda pair: (
                        _compat_rank(pair[1]),
                        _prefers(pair[0]),
                        1 if pair[0].get("pip_verified") else 0,
                        1 if pair[0].get("task_type") == task_type else 0,
                        self._function_text_match_score(pair[0], step),
                        self._candidate_io_match_score(pair[0], step),
                        pair[0].get("confidence", 0.0),
                    ),
                )
                return {
                    "function_name": best_fallback.get("function_name", ""),
                    "package": best_fallback.get("package", ""),
                    "signature": self._compact_signature(
                        best_fallback.get("signature", "")
                    ),
                    "validation_level": (
                        "package_verified"
                        if best_fallback.get("pip_verified")
                        else "deterministic_fallback"
                    ),
                    "confidence": best_fallback.get("confidence", 0.0),
                    "selection_reason": (
                        "deterministic_operation_template:"
                        f"{fallback_compatibility.get('status') or 'operation_match'}"
                    ),
                    "operation_compatibility_status": fallback_compatibility.get(
                        "status", ""
                    ),
                    "operation_compatibility_advisory": bool(
                        fallback_compatibility.get("advisory", False)
                    ),
                    "raw_description": best_fallback.get("raw_description", ""),
                }
            return {
                "function_name": "",
                "package": "",
                "signature": "",
                "validation_level": "none",
                "confidence": 0.0,
                "selection_reason": (
                    "operation_constraint_unmatched"
                    if candidates
                    else "no_candidates" if self._cap_index_loaded else "index_unavailable"
                ),
                "gap_type": "coverage_gap" if self._cap_index_loaded else "index_unavailable",
            }

        def _score(pair: Tuple[Dict[str, Any], Dict[str, Any]]) -> Tuple:
            c, compatibility = pair
            return (
                _compat_rank(compatibility),
                _prefers(c),
                1 if c.get("pip_verified") else 0,
                1 if c.get("task_type") == task_type else 0,
                self._function_text_match_score(c, step),
                self._candidate_io_match_score(c, step),
                c.get("confidence", 0.0),
            )

        best, best_compatibility = max(eligible_candidates, key=_score)
        best_proven = bool(best_compatibility.get("compatible"))
        pip_ok = best.get("pip_verified", False)
        tt_match = best.get("task_type") == task_type

        compat_prefix = (
            "operation-compatible"
            if best_proven
            else "operation-unproven (advisory)"
        )
        if pip_ok and tt_match:
            level = "verified"
            reason = f"{compat_prefix} + pip_verified + task_type match"
        elif pip_ok:
            level = "package_verified"
            reason = f"{compat_prefix} + pip_verified, task_type fallback"
        elif tt_match:
            level = "task_type_match"
            reason = f"{compat_prefix} + task_type match, pip not verified"
        else:
            level = "best_effort"
            reason = f"{compat_prefix} best confidence candidate"

        return {
            "function_name": best.get("function_name", ""),
            "package": best.get("package", ""),
            "signature": self._compact_signature(best.get("signature", "")),
            "validation_level": level,
            "confidence": best.get("confidence", 0.0),
            "selection_reason": reason,
            "operation_compatibility_status": best_compatibility.get("status", ""),
            "operation_compatibility_advisory": bool(
                best_compatibility.get("advisory", False)
            ),
        }

    @staticmethod
    def _compile_refinement_operation_constraints(
        plan_refinement_feedback: Dict[str, Any],
        previous_task_plan: Optional[List[Dict[str, Any]]],
        replan_round: int,
    ) -> Dict[int, Dict[str, Any]]:
        """Compile Refinement advice into per-step constraints that anchoring cannot drop."""
        previous_hints = {
            step.get("id"): str(
                (step.get("implementation_hint") or {}).get("function_name") or ""
            ).strip()
            for step in (previous_task_plan or [])
            if isinstance(step, dict)
        }
        # 约束要跟着"步骤内容"走而不是跟着编号走：记下 Refinement 实际看到的那份
        # plan 里每个步骤的内容身份，replan 后据此把约束重定位到对应的新节点。
        previous_identities = {
            step.get("id"): build_step_identity(step)
            for step in (previous_task_plan or [])
            if isinstance(step, dict)
        }
        # Refinement 只能对它实际看到的那份 plan 提要求。旧 plan 里就不存在的
        # step_id 是 Refinement 自身的输出错误（凭空指称），必须在编译阶段归因并
        # fail closed；等到 replan 之后再报，就会被误记成 StepPlan 的契约违规。
        # previous_task_plan 为空时没有可对照的基准，不做该判定。
        known_step_ids = {
            step_id for step_id in previous_hints if step_id is not None
        }
        fabricated: List[str] = []

        def _reject_if_fabricated(step_id: int, source: str) -> bool:
            if not known_step_ids or step_id in known_step_ids:
                return False
            fabricated.append(f"{source} references step {step_id}")
            return True

        by_step: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

        for index, adjustment in enumerate(
            plan_refinement_feedback.get("step_level_adjustments", []) or [],
            1,
        ):
            if not isinstance(adjustment, dict) or adjustment.get("action") == "keep":
                continue
            step_id = StepPlanAgent._normalize_feedback_step_id(adjustment.get("step_id"))
            if step_id is None:
                continue
            if _reject_if_fabricated(step_id, f"step_level_adjustments[{index}]"):
                continue
            raw_constraints = adjustment.get("repair_constraints") or {}
            if not isinstance(raw_constraints, dict):
                raw_constraints = {}
            compiled = copy.deepcopy(raw_constraints)
            compiled["issue_id"] = str(
                adjustment.get("issue_id")
                or f"refinement-r{replan_round}-step{step_id}-adjustment{index}"
            )
            preferred = compiled.get("preferred_function_names") or []
            if isinstance(preferred, str):
                preferred = [preferred]
            suggested = str(adjustment.get("suggested_replacement") or "").strip()
            if adjustment.get("action") == "replace_function" and suggested:
                preferred.append(suggested)
            compiled["preferred_function_names"] = preferred
            forbidden = compiled.get("forbidden_function_names") or []
            if isinstance(forbidden, str):
                forbidden = [forbidden]
            previous_function = previous_hints.get(step_id, "")
            if adjustment.get("action") in {"replace_function", "replace_tool"} and previous_function:
                forbidden.append(previous_function)
            compiled["forbidden_function_names"] = forbidden
            if adjustment.get("action") == "change_task_type" and not compiled.get("required_operation"):
                compiled["required_operation"] = suggested
            # 删除请求的闭合条件与其它 action 相反：步骤消失才算照做。
            if adjustment.get("action") == "remove_step":
                compiled["removal_requested"] = True
            by_step[step_id].append(compiled)

        for index, issue in enumerate(
            plan_refinement_feedback.get("semantic_alignment_issues", []) or [],
            1,
        ):
            if not isinstance(issue, dict):
                continue
            raw_constraints = issue.get("repair_constraints") or {}
            if not isinstance(raw_constraints, dict):
                raw_constraints = {}
            for raw_step_id in issue.get("affected_step_ids", []) or []:
                step_id = StepPlanAgent._normalize_feedback_step_id(raw_step_id)
                if step_id is None:
                    continue
                if _reject_if_fabricated(step_id, f"semantic_alignment_issues[{index}]"):
                    continue
                compiled = copy.deepcopy(raw_constraints)
                compiled["issue_id"] = str(
                    issue.get("issue_id")
                    or f"refinement-r{replan_round}-semantic{index}-step{step_id}"
                )
                by_step[step_id].append(compiled)

        if fabricated:
            raise AgentExecutionError(
                "Refinement feedback references step ids absent from the reviewed plan "
                f"(known step ids: {sorted(known_step_ids)}): " + "; ".join(fabricated)
            )

        compiled_by_step: Dict[int, Dict[str, Any]] = {}
        for step_id, items in by_step.items():
            merged = merge_operation_constraints(items)
            merged["source_step_identity"] = copy.deepcopy(
                previous_identities.get(step_id) or {}
            )
            compiled_by_step[step_id] = merged
        return compiled_by_step

    @staticmethod
    def _apply_operation_constraints(
        steps: List[Dict[str, Any]],
        constraints_by_step: Dict[int, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        constrained_steps: List[Dict[str, Any]] = []
        for raw_step in steps:
            step = copy.deepcopy(raw_step)
            step_id = step.get("id")
            if step_id in constraints_by_step:
                constraints = copy.deepcopy(constraints_by_step[step_id])
                # `source_step_identity` 只是重定位的内部输入，不属于步骤的约束契约：
                # 留在 plan 里既会随产物落盘膨胀，也会让下游误以为它是可消费的语义字段。
                constraints.pop("source_step_identity", None)
                step["operation_constraints"] = constraints
            constrained_steps.append(step)
        return constrained_steps

    @staticmethod
    def _validate_refinement_constraint_closure(
        steps: List[Dict[str, Any]],
        constraints_by_step: Dict[int, Dict[str, Any]],
        *,
        unresolved_constraints: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Verify closure against relocated constraints, never against raw step ids.

        `constraints_by_step` 已由 `relocate_step_constraints` 按内容重定位，键是
        **新** plan 的 step id；`unresolved_constraints` 是在新 plan 里找不到内容
        对应节点的约束。两者的判据不同：
          - 定位得到节点 = 该步骤在新计划里仍然存在，删除指令未被执行；
          - 定位不到节点 = 该步骤的内容已从计划中消失，删除指令已被执行。
        非删除约束定位不到时是证据失效而非违约，按 advisory 放行。
        """
        if not constraints_by_step and not unresolved_constraints:
            return [], []
        step_by_id = {
            step.get("id"): step
            for step in steps
            if isinstance(step, dict)
        }
        issues: List[str] = []
        closure_records: List[Dict[str, Any]] = []

        for constraints in unresolved_constraints or []:
            source_step_id = constraints.get("source_step_id")
            if constraints.get("removal_requested"):
                # 内容在新计划里已无对应节点 —— 这正是删除被执行的证据。
                # 判据从"id 是否存在"改为"内容是否还在"：full replan 会重新编号，
                # 删掉三步后旧 id 仍然存在，按 id 判会把"照做"误判成"未照做"。
                closure_records.append({
                    "step_id": source_step_id,
                    "issue_ids": copy.deepcopy(constraints.get("issue_ids", [])),
                    "function_name": "",
                    "compatible": True,
                    "status": "step_removed_as_requested",
                    "reason": (
                        "refinement requested removal and no step in the replanned plan "
                        "matches the removed step's content"
                    ),
                    "missing_input_semantics": [],
                    "missing_output_semantics": [],
                })
                continue
            closure_records.append({
                "step_id": source_step_id,
                "issue_ids": copy.deepcopy(constraints.get("issue_ids", [])),
                "function_name": "",
                "compatible": True,
                "status": "step_content_absent_after_replan",
                "reason": (
                    "no step in the replanned plan matches this constraint's original "
                    "step content; closure is unverifiable rather than violated"
                ),
                "advisory_warnings": [
                    f"refinement constraints originally scoped to step {source_step_id} "
                    "could not be verified because the replanned plan has no "
                    "content-equivalent step"
                ],
                "missing_input_semantics": [],
                "missing_output_semantics": [],
            })

        for step_id, constraints in constraints_by_step.items():
            step = step_by_id.get(step_id)
            source_step_id = constraints.get("source_step_id", step_id)
            if not isinstance(step, dict):
                # 重定位只会指向 steps 里实际存在的节点，正常不可达；保留为审计留痕，
                # 不静默丢弃约束。
                closure_records.append({
                    "step_id": step_id,
                    "source_step_id": source_step_id,
                    "issue_ids": copy.deepcopy(constraints.get("issue_ids", [])),
                    "function_name": "",
                    "compatible": True,
                    "status": "relocated_step_unresolved",
                    "reason": (
                        "relocated target step is missing from the validated plan; "
                        "closure is unverifiable rather than violated"
                    ),
                    "missing_input_semantics": [],
                    "missing_output_semantics": [],
                })
                continue
            if constraints.get("removal_requested"):
                # 对称判据：内容仍能在新计划里定位到，才是删除指令未被执行。
                closure_records.append({
                    "step_id": step_id,
                    "source_step_id": source_step_id,
                    "relocation_match": constraints.get("relocation_match"),
                    "issue_ids": copy.deepcopy(constraints.get("issue_ids", [])),
                    "function_name": str(
                        (step.get("implementation_hint") or {}).get("function_name") or ""
                    ),
                    "compatible": False,
                    "status": "step_removal_not_applied",
                    "reason": (
                        "refinement requested removal but step "
                        f"{step_id} still carries the removed step's content"
                    ),
                    "missing_input_semantics": [],
                    "missing_output_semantics": [],
                })
                issues.append(
                    f"step {step_id} did not close refinement issues "
                    f"{constraints.get('issue_ids', [])}: removal of the step originally "
                    f"numbered {source_step_id} was requested, but step {step_id} still "
                    f"carries that step's content "
                    f"(matched by {constraints.get('relocation_match')})"
                )
                continue
            hint = step.get("implementation_hint") or {}
            candidates = step.get("implementation_candidates") or []
            selected_function = str(hint.get("function_name") or "")
            selected_entry = next(
                (
                    candidate
                    for candidate in candidates
                    if str(candidate.get("function_name") or "") == selected_function
                ),
                dict(hint),
            )
            compatibility = evaluate_operation_compatibility(step, selected_entry)
            input_semantics = constraints.get("required_input_semantics", []) or []
            output_semantics = constraints.get("required_output_semantics", []) or []
            # 语义证据除 inputs/outputs 字段外，还包含步骤描述与选中函数自身的
            # 名称/签名/描述：refinement 的 required_*_semantics 描述的是数据契约，
            # 选中函数的能力描述（如 ee.ImageCollection 的 asset id / ImageCollection）
            # 同样是该契约成立的确定性证据。
            function_evidence = " ".join(
                str(value or "")
                for value in (
                    selected_function,
                    selected_entry.get("signature"),
                    selected_entry.get("raw_description"),
                )
            )
            step_context = " ".join(
                str(value or "")
                for value in (step.get("step_name"), step.get("description"))
            )
            input_evidence = " ".join(
                (
                    json.dumps(step.get("inputs", []), ensure_ascii=False),
                    step_context,
                    function_evidence,
                )
            )
            output_evidence = " ".join(
                (
                    json.dumps(step.get("outputs", []), ensure_ascii=False),
                    step_context,
                    function_evidence,
                )
            )
            # Refinement 显式点名的函数被采用，是该 issue 已被处理的最强证据。
            # 随附的 required_*_semantics 本就是对这个函数的自然语言描述，
            # 此时再用措辞匹配去否决它，等于用弱证据推翻强证据。
            preferred_function_honored = (
                compatibility.get("status") == "preferred_function_match"
            )

            def _classify(requirements, evidence):
                contradicted, unproven = [], []
                for requirement in requirements:
                    verdict = semantic_requirement_verdict(requirement, evidence)
                    if verdict == SEMANTIC_CONTRADICTED:
                        contradicted.append(requirement)
                    elif verdict == SEMANTIC_UNPROVEN:
                        unproven.append(requirement)
                return contradicted, unproven

            contradicted_input, unproven_input = _classify(input_semantics, input_evidence)
            contradicted_output, unproven_output = _classify(output_semantics, output_evidence)
            if preferred_function_honored:
                unproven_input = unproven_input + contradicted_input
                unproven_output = unproven_output + contradicted_output
                contradicted_input, contradicted_output = [], []
            # 保留原字段名供下游审计消费：missing_* 记录所有未证实项，
            # 其中只有 contradicted_* 参与硬判定。
            missing_input_semantics = contradicted_input + unproven_input
            missing_output_semantics = contradicted_output + unproven_output
            # 与 ResourceCheck 门禁的 advisory 语义保持一致：证据缺失（unproven）
            # 不算违约，只有正向矛盾才硬拦。这条对 operation 证据和 I/O 语义同样
            # 成立——两者的证据源是同一份残缺文本（compact 候选不含
            # raw_description），把"没匹配上"当违约会形成两道门互锁的 replan 死循环。
            operation_hard_incompatible = not compatibility.get(
                "compatible"
            ) and not compatibility.get("advisory")
            # 重定位强度决定语义矛盾能否硬拦。相似度匹配只说明"这两个节点大概是
            # 同一件事"，不足以支撑"该节点违反了原步骤的数据契约"——真实失败中
            # 相似度匹配到的节点常已被 refinement 要求改变输出形态，此时 contradiction
            # 恰恰是改写生效的结果。仅 artifact id / 步骤名精确命中才允许硬判定。
            relocation_match = constraints.get("relocation_match")
            semantics_binding = relocation_match in {
                None,
                MATCH_OUTPUT_ARTIFACT,
                MATCH_STEP_NAME_EXACT,
                MATCH_STEP_ID_FALLBACK,
            }
            semantic_hard_conflict = bool(
                semantics_binding and (contradicted_input or contradicted_output)
            )
            closure_compatible = (
                not operation_hard_incompatible and not semantic_hard_conflict
            )
            is_coverage_gap = (
                not selected_function
                or hint.get("gap_type") == "coverage_gap"
                or hint.get("selection_reason")
                in {
                    "operation_constraint_unmatched",
                    "no_candidates",
                    "index_unavailable",
                }
            )
            record = {
                "step_id": step_id,
                "source_step_id": source_step_id,
                "relocation_match": relocation_match,
                "issue_ids": copy.deepcopy(constraints.get("issue_ids", [])),
                "function_name": selected_function,
                **compatibility,
                "compatible": closure_compatible,
                "missing_input_semantics": missing_input_semantics,
                "missing_output_semantics": missing_output_semantics,
                "contradicted_input_semantics": contradicted_input,
                "contradicted_output_semantics": contradicted_output,
                "unproven_input_semantics": unproven_input,
                "unproven_output_semantics": unproven_output,
            }
            warnings = list(record.get("advisory_warnings") or [])
            if closure_compatible and not compatibility.get("compatible"):
                warnings.append(
                    "operation evidence unproven; admitted as advisory under the "
                    "required_operation hard boundary (no contradiction found)"
                )
            if unproven_input or unproven_output:
                warnings.append(
                    "io semantics unproven (wording not matched, no contradiction): "
                    f"input={unproven_input}, output={unproven_output}"
                )
            if (contradicted_input or contradicted_output) and not semantics_binding:
                warnings.append(
                    "io semantics contradicted but admitted as advisory: the constraint "
                    f"was relocated onto this step by {relocation_match}, which is too "
                    "weak to prove a data-contract violation "
                    f"(input={contradicted_input}, output={contradicted_output})"
                )
            if warnings:
                record["advisory_warnings"] = warnings
            if not closure_compatible and is_coverage_gap:
                record["gap_type"] = "coverage_gap"
            closure_records.append(record)
            if not closure_compatible:
                # 只报真正参与硬判定的证据。advisory 化的 contradiction 若也写进
                # issue 文案，会把"未违约"的项伪装成失败原因，误导根因定位。
                semantic_reason = ""
                if semantic_hard_conflict:
                    semantic_reason = (
                        f"; contradicted input semantics={contradicted_input}, "
                        f"output semantics={contradicted_output}"
                    )
                relocation_note = ""
                if source_step_id != step_id:
                    relocation_note = (
                        f" (constraint originally scoped to step {source_step_id}, "
                        f"relocated by {relocation_match})"
                    )
                issues.append(
                    f"step {step_id} did not close refinement issues "
                    f"{constraints.get('issue_ids', [])}: {compatibility['reason']}"
                    f"{semantic_reason}{relocation_note}"
                )
        return issues, closure_records

    # =====================================================================
    # Prompt 构建
    # =====================================================================

    def _build_requirement_draft_prompt(
        self,
        *,
        idea_payload: Dict[str, Any],
        dataset_info: Dict[str, Any],
        previous_required_datasets: Optional[List[Dict[str, Any]]] = None,
        plan_refinement_feedback: Optional[Dict[str, Any]] = None,
        resource_check_feedback: Optional[Dict[str, Any]] = None,
        replan_round: int = 0,
    ) -> str:
        """Build the requirement-only request that precedes strict resolution."""
        if idea_payload.get("payload_type") != "idea" or not isinstance(
            idea_payload.get("idea"), dict
        ):
            raise AgentExecutionError(
                "StepPlanAgent requires a structured idea payload"
            )
        previous_core = [
            requirement_core(item)
            for item in (previous_required_datasets or [])
            if isinstance(item, dict)
        ]
        return f"""
StepPlan phase: DATASET_REQUIREMENT_DRAFT.

Accepted idea:
{json.dumps(self._idea_core(idea_payload), ensure_ascii=False, separators=(',', ':'))}

StepPlan handoff:
{json.dumps(idea_payload.get("planning_handoff") or {}, ensure_ascii=False, separators=(',', ':'))}

Base dataset characteristics:
{json.dumps(dataset_info, ensure_ascii=False, separators=(',', ':'))}

Candidate dataset evidence (non-binding):
{json.dumps(self._candidate_evidence_block(idea_payload.get("candidate_datasets")), ensure_ascii=False, separators=(',', ':'))}

Previous required datasets for replan:
{json.dumps(previous_core, ensure_ascii=False, separators=(',', ':'))}

Plan refinement feedback:
{json.dumps(plan_refinement_feedback or {}, ensure_ascii=False, separators=(',', ':'))}

ResourceCheck feedback:
{json.dumps(resource_check_feedback or {}, ensure_ascii=False, separators=(',', ':'))}

Replan round: {replan_round}

Return only the complete `required_datasets` contract needed in addition to the
provided base dataset. Do not generate task steps, tools, descriptions, inputs, or
outputs in this phase. Return an empty list when the accepted idea can be tested with
the base dataset alone. Do not add auxiliary data merely to make the idea multi-source.

Every non-empty requirement must state `id`, `name`, `research_justification`,
`execution_requirement_level`, `binding_notes`, `modality`, `source_preference`,
`retrieval_hint`, and `temporal_requirement_level`. Preserve concrete catalog IDs and
aliases from candidate evidence. The output budget is strict: keep
`research_justification` and `binding_notes` to at most two short sentences each and
`retrieval_hint` to one short sentence; do not pad any field with long prose. Use exactly one temporal contract: `not_required`
with neither temporal field; `coverage_only` with `temporal_coverage` only; or
`coverage_and_resolution` with both `temporal_coverage` and `temporal_resolution`.
Write coverage as ISO `start`/`end` plus `coverage_mode=full_cover|overlap`, and cadence
as `unit`, positive `interval`, and `allow_finer_resolution`. Static data must use
`not_required`; do not simultaneously claim that static data are acceptable and require
time-series coverage or cadence. If you cannot state a concrete coverage window, declare
`not_required` rather than `coverage_only` with no `temporal_coverage`; such a requirement
is read as having no assertable time constraint and is demoted to `not_required`.
Never emit the legacy free-text `time_range`.
Put geographic scope only in `frozen_constraints.region`; never put Global, countries,
continents, or other region names in `entity`. Put only non-substitutable scientific
entity, observable, measure, and product-family identity in the other frozen fields.
Every frozen value must be something the supplied candidate catalog evidence can verify.
Do not freeze a planned output encoding, derived statistic, or downstream representation
such as binary/fraction unless the source dataset must natively provide that quantity.
Candidate evidence remains non-binding until the
strict resolver returns `resolved_strong`; do not claim that a requirement is resolved.
For `execution_requirement_level=required`, use only datasets anchored by an explicit
catalog ID or a strong title hit in candidate evidence and with no known time, region,
type, family, or identity conflict. If the accepted idea needs a new external dataset,
write concrete product IDs/names in `retrieval_hint` so ResourceCheck can search for it.
If no strong binding can be found after that search, the requirement must be `optional`,
not `required`; optional unresolved requirements are diagnostics and cannot be referenced
by the final executable DAG.
All free-text values must be in English.
"""

    def _build_requirement_repair_prompt(
        self,
        *,
        idea_payload: Dict[str, Any],
        dataset_info: Dict[str, Any],
        current_required_datasets: List[Dict[str, Any]],
        dataset_resolution_report: Dict[str, Any],
        unresolvable_datasets: List[str],
        resolution_log: List[Dict[str, Any]],
    ) -> str:
        """Build a requirement-only repair request after strict resolution fails."""
        if idea_payload.get("payload_type") != "idea" or not isinstance(
            idea_payload.get("idea"), dict
        ):
            raise AgentExecutionError(
                "StepPlanAgent requires a structured idea payload"
            )
        return f"""
StepPlan phase: DATASET_REQUIREMENT_REPAIR.

Accepted idea:
{json.dumps(self._idea_core(idea_payload), ensure_ascii=False, separators=(',', ':'))}

StepPlan handoff:
{json.dumps(idea_payload.get("planning_handoff") or {}, ensure_ascii=False, separators=(',', ':'))}

Base dataset characteristics:
{json.dumps(dataset_info, ensure_ascii=False, separators=(',', ':'))}

Candidate dataset evidence (non-binding):
{json.dumps(self._candidate_evidence_block(idea_payload.get("candidate_datasets")), ensure_ascii=False, separators=(',', ':'))}

Current required_datasets that failed strict ResourceCheck resolution:
{json.dumps([requirement_core(item) for item in current_required_datasets], ensure_ascii=False, separators=(',', ':'))}

Unresolvable requirement ids:
{json.dumps(unresolvable_datasets, ensure_ascii=False, separators=(',', ':'))}

ResourceCheck dataset resolution report:
{json.dumps(dataset_resolution_report or {}, ensure_ascii=False, separators=(',', ':'))}

ResourceCheck resolution log:
{json.dumps(self._compact_resolution_log(resolution_log), ensure_ascii=False, separators=(',', ':'))}

Return a complete replacement `required_datasets` list only. Do not generate task
steps. Preserve requirements that ResourceCheck already resolved strongly unless
their IDs must change to remove duplication. For every failed requirement, choose
exactly one repair:
- rewrite only over-specific frozen_constraints to values directly verifiable in
  candidate evidence, when the scientific requirement is still the same;
- replace source_preference/retrieval_hint with a concrete catalog ID or strong title
  from candidate evidence when that removes the failure and no hard conflict remains;
- downgrade the requirement to `optional` when no explicit ID or strong title hit can
  satisfy it without family, temporal, region, type, or identity conflict.

Do not keep a failed requirement as `required` unless the revised requirement is
anchored by an explicit catalog ID or strong title hit and has no known hard conflict.
Do not invent candidate facts. Optional unresolved requirements are diagnostics only
and must not be referenced by the final executable DAG. Keep all requirement schema
rules from DATASET_REQUIREMENT_DRAFT: required fields, temporal contract consistency,
region only in frozen_constraints.region, and verifiable frozen identity fields.
All free-text values must be in English.
"""

    async def draft_required_datasets(
        self,
        context: Dict[str, Any],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Draft only external dataset requirements, before final plan generation."""
        self._runtime_planning_config()
        mode = str(params.get("mode") or "").strip()
        if mode not in {"plan_for_idea", "replan"}:
            raise AgentExecutionError(f"unsupported StepPlan mode: {mode}")
        previous_required_datasets = context.get("previous_required_datasets")
        if mode == "replan" and not isinstance(previous_required_datasets, list):
            raise AgentExecutionError(
                "replan requirement draft requires previous_required_datasets as a list"
            )
        replan_round = int(params.get("replan_round", 0) or 0)
        audit: Dict[str, Any] = {
            "version": "hierarchical_step_plan_v2",
            "mode": mode,
            "replan_round": replan_round,
            "model_calls": [],
            "validation_passes": [],
        }
        prompt = self._build_requirement_draft_prompt(
            idea_payload=context["idea"],
            dataset_info=context.get("dataset_info", {}),
            previous_required_datasets=previous_required_datasets,
            plan_refinement_feedback=context.get("plan_refinement_feedback") or {},
            resource_check_feedback=context.get("resource_check_feedback") or {},
            replan_round=replan_round,
        )
        response = await self._call_planning_phase(
            phase=(
                "initial_requirement_draft"
                if mode == "plan_for_idea"
                else "replan_requirement_draft"
            ),
            prompt=prompt,
            schema=self._build_requirement_draft_schema(),
            audit=audit,
        )
        requirements, normalization_actions = self._normalize_required_dataset_draft(
            response.get("required_datasets")
        )
        issues = self._validate_required_dataset_contract(requirements, [])
        audit["validation_passes"].append(
            {
                "phase": "dataset_requirement_draft",
                "valid": not issues,
                "issues": copy.deepcopy(issues),
                "normalization_actions": copy.deepcopy(normalization_actions),
            }
        )
        if issues:
            raise StepPlanContractError(
                issues,
                required_datasets=(
                    requirements if isinstance(requirements, list) else []
                ),
                generation_audit=audit,
            )
        audit["model_call_count"] = len(audit["model_calls"])
        return {
            "required_datasets": copy.deepcopy(requirements),
            "step_plan_generation_audit": audit,
        }

    async def repair_required_datasets(
        self,
        context: Dict[str, Any],
        params: Dict[str, Any],
        *,
        current_required_datasets: List[Dict[str, Any]],
        dataset_resolution_report: Dict[str, Any],
        unresolvable_datasets: List[str],
        resolution_log: Optional[List[Dict[str, Any]]] = None,
        base_audit: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Repair only required_datasets after plan-time ResourceCheck failure."""
        self._runtime_planning_config()
        if not isinstance(current_required_datasets, list):
            raise StepPlanContractError(
                ["requirement repair requires current_required_datasets as a list"],
                generation_audit=copy.deepcopy(base_audit or {}),
            )
        audit = copy.deepcopy(base_audit or {})
        audit.setdefault("version", "hierarchical_step_plan_v2")
        audit.setdefault("mode", str(params.get("mode") or "plan_for_idea"))
        audit.setdefault("replan_round", int(params.get("replan_round", 0) or 0))
        audit.setdefault("model_calls", [])
        audit.setdefault("validation_passes", [])

        prompt = self._build_requirement_repair_prompt(
            idea_payload=context["idea"],
            dataset_info=context.get("dataset_info", {}),
            current_required_datasets=current_required_datasets,
            dataset_resolution_report=dataset_resolution_report,
            unresolvable_datasets=unresolvable_datasets,
            resolution_log=resolution_log or [],
        )
        response = await self._call_planning_phase(
            phase="dataset_requirement_repair_1",
            prompt=prompt,
            schema=self._build_requirement_draft_schema(),
            audit=audit,
        )
        requirements, normalization_actions = self._normalize_required_dataset_draft(
            response.get("required_datasets")
        )
        issues = self._validate_required_dataset_contract(requirements, [])
        audit["validation_passes"].append(
            {
                "phase": "dataset_requirement_repair_1",
                "valid": not issues,
                "issues": copy.deepcopy(issues),
                "normalization_actions": copy.deepcopy(normalization_actions),
                "input_unresolvable_datasets": copy.deepcopy(unresolvable_datasets),
                "input_dataset_resolution_report": copy.deepcopy(
                    dataset_resolution_report or {}
                ),
            }
        )
        if issues:
            raise StepPlanContractError(
                issues,
                required_datasets=(
                    requirements if isinstance(requirements, list) else []
                ),
                generation_audit=audit,
            )
        audit["model_call_count"] = len(audit["model_calls"])
        return {
            "required_datasets": copy.deepcopy(requirements),
            "step_plan_generation_audit": audit,
        }

    @staticmethod
    def _idea_core(idea_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the planning-relevant idea fields shared by every StepPlan phase.

        The full payload also carries `dataset_characteristics` and `candidate_datasets`,
        which every prompt already injects separately; keeping them out of the idea block
        avoids sending the same content twice in one prompt.
        """
        idea = idea_payload.get("idea") or {}
        return {
            "idea_id": idea.get("idea_id", ""),
            "idea_type": idea.get("idea_type", ""),
            "research_question": idea.get("research_question", ""),
            "knowledge_anchor_and_motivation": idea.get("knowledge_anchor_and_motivation", {}),
            "data_synergy_and_gap": idea.get("data_synergy_and_gap", {}),
            "core_hypothesis": idea.get("core_hypothesis", {}),
            "theoretical_mechanism": idea.get("theoretical_mechanism", {}),
            "computability_and_falsifiability": idea.get("computability_and_falsifiability", {}),
            "novelty_constraints_and_value": idea.get("novelty_constraints_and_value", {}),
        }

    def _build_plan_prompt_from_idea_payload(
        self,
        idea_payload: Dict[str, Any],
        dataset_info: Dict[str, Any],
        previous_task_plan: Optional[List[Dict[str, Any]]] = None,
        previous_required_datasets: Optional[List[Dict[str, Any]]] = None,
        plan_refinement_feedback: Optional[Dict[str, Any]] = None,
        resource_check_feedback: Optional[Dict[str, Any]] = None,
        replan_round: int = 0,
        plan_repair_feedback: Optional[Dict[str, Any]] = None,
        frozen_required_datasets: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Build the StepPlan prompt for the idea-first initial path."""
        planning_handoff = idea_payload.get("planning_handoff") or {}
        payload_dataset_characteristics = idea_payload.get("dataset_characteristics") or {}
        candidate_datasets = idea_payload.get("candidate_datasets") or []
        effective_dataset_info = dataset_info or payload_dataset_characteristics

        idea_core = self._idea_core(idea_payload)
        handoff_notes = planning_handoff.get("handoff_notes_for_step_plan", [])
        unresolved_dataset_requirements = planning_handoff.get(
            "unresolved_dataset_requirements",
            [],
        )
        evidence_records = planning_handoff.get("evidence_records", [])
        handoff_context = {
            "handoff_notes_for_step_plan": handoff_notes,
            "unresolved_dataset_requirements": unresolved_dataset_requirements,
            "evidence_records": evidence_records,
        }

        replan_section = ""
        if previous_task_plan:
            # implementation_candidates 是确定性 capability 查表证据，只被本地选择逻辑消费，
            # 模型从不读它；单个 anchored step 可携带上千条候选，实测占 task_plan 字符的 97.9%。
            # 与 SCOPED_PLAN_PATCH 路径保持同一裁剪口径。
            replan_section += f"""
---
Previous Task Plan:
{self._format_context({'task_plan': [self._patch_context_step(step) for step in previous_task_plan]})}
"""
        if previous_required_datasets:
            replan_section += f"""
---
Previous Required Datasets:
{self._format_context({'required_datasets': previous_required_datasets})}
"""
        if plan_refinement_feedback:
            replan_section += f"""
---
Plan Refinement Feedback:
{self._format_context(plan_refinement_feedback)}
"""
        if resource_check_feedback:
            replan_section += f"""
---
Resource Check Failure:
{self._format_context(resource_check_feedback)}
"""
        if plan_repair_feedback:
            replan_section += f"""
---
Deterministic Replan Repair Feedback:
{self._format_context(plan_repair_feedback)}
"""
        if replan_round:
            replan_section += f"""
---
Replan Round:
{replan_round}
"""

        replan_hard_constraints = ""
        if plan_refinement_feedback or resource_check_feedback:
            replan_hard_constraints = """
- Replanning hard constraints:
    * You MUST NOT continue using any requirement expression that ResourceCheck already marked with region_conflict.
    * If the feedback says family/time/region must be more explicit, you MUST write those constraints back into structured required_datasets fields such as temporal_coverage, temporal_resolution, frozen_constraints.region, and spatial_resolution.
    * Every `repair_constraints` entry in the Plan Refinement Feedback is binding: the rewritten step MUST honor `required_operation` and `preferred_function_names`, MUST NOT reuse `forbidden_function_names`, and its `inputs`/`outputs` fields (including each item's `role` and `data_type`) MUST explicitly state the `required_input_semantics` and `required_output_semantics` terms (e.g., mention the GEE asset id and image collection in the input role, and the produced GEE image/collection in the output role).
"""

        frozen_requirement_section = ""
        if frozen_required_datasets is not None:
            frozen_requirement_section = f"""
---
Verified Required Datasets (FROZEN):
{json.dumps([requirement_core(item) for item in frozen_required_datasets], ensure_ascii=False, separators=(',', ':'))}

Final-plan binding gate:
- These requirements already passed strict ResourceCheck resolution as `resolved_strong`.
- Return exactly the same requirement IDs and core fields. Do not add, remove, rename,
  substitute, or otherwise modify any requirement.
- Build the graph only against this frozen contract. Any requirement drift causes the
  final plan to be rejected deterministically.
"""

        prompt = f"""
Accepted Idea (planning constraints; immutable):
{json.dumps(idea_core, ensure_ascii=False, separators=(',', ':'))}

StepPlan Handoff:
{json.dumps(handoff_context, ensure_ascii=False, separators=(',', ':'))}
{replan_section}
---
Dataset Characteristics (dataset_info):
{self._format_context(effective_dataset_info)}

---
Candidate Dataset Evidence (non-binding):
{json.dumps(self._candidate_evidence_block(candidate_datasets), ensure_ascii=False, separators=(',', ':'))}
{frozen_requirement_section}

---
Available Tools/Resources:
Standard Python Data Science Stack (pandas, numpy, sklearn, pytorch, xarray, geopandas, rasterio, etc.)

---
Planning Instructions:
Role: You are a geography and geospatial data science expert. Based on this idea skeleton, generate a detailed task plan that operationalizes the scientific idea into executable geospatial analysis steps.

Based on the accepted idea, narrow StepPlan handoff, and dataset characteristics, create a complete executable task_plan.

1. **Idea Constraint Alignment**:
   - Treat `research_question` as the question the plan must answer, not as decorative context.
   - Treat `knowledge_anchor_and_motivation` and `data_synergy_and_gap` as constraints on why the analysis is needed and which data relationships must be operationalized.
   - Treat `core_hypothesis` as the claim that the plan must test or falsify.
   - Treat `theoretical_mechanism` as the causal or explanatory chain that the analysis steps must make observable.
   - Treat `computability_and_falsifiability` as mandatory guidance for variables, contrasts, evaluation targets, and failure conditions.
   - Treat `novelty_constraints_and_value` as boundary conditions: do not collapse the idea into a generic descriptive analysis if the idea requires a stronger mechanism, contrast, or value claim.
2. **Methodology Alignment**:
   - Convert the idea constraints into explicit data operations, analytical comparisons, model choices, and validation checks.
   - Every major step should be traceable to at least one idea constraint or to a necessary data preparation action.
   - Ensure the steps preserve the idea's scientific claim and make the claimed mechanism empirically checkable.
3. **Logical Flow**:
   - Data access/acquisition and inspection.
   - Preprocessing and quality control.
   - Data integration and spatial/temporal scale alignment.
   - Feature engineering or transformation.
   - Analysis, modeling, comparison, or visualization.
   - Validation, evaluation, and falsification-oriented checks.
4. **Open-Point Handling**:
   - Expand every `handoff_notes_for_step_plan` item into explicit planning work.
   - Treat handoff notes and `unresolved_dataset_requirements` as unresolved planning inputs, not solved facts.
   - `evidence_records` contains relevance-filtered literature and dataset evidence already retrieved during idea maturation for these handoff items. Use it directly as planning context; StepPlan does not need to retrieve the same evidence again.
   - Evidence supports planning choices but is not an already selected dataset contract. A dataset result becomes a top-level `required_datasets` item only when the accepted idea or an open point actually requires that additional external data.
   - A `required` item must be anchored to an explicit catalog ID or a strong title hit from candidate evidence, and must have no known time, region, type, family, or identity conflict. If the idea needs a new external dataset, name concrete products/asset IDs in `retrieval_hint` for ResourceCheck search; if the search cannot produce a strong binding, downgrade that item to `optional` so it is not used by the final executable DAG.
   - If an open point has `not_fixed_by_idea_stage=true`, it means the idea stage deliberately did not fix that method choice. You MUST calibrate, compare, validate, or otherwise operationalize it inside the task_plan instead of treating it as an upstream fixed contract.
   - Use each `unresolved_dataset_requirements` item only to decide whether an additional structured dataset requirement is necessary. A retrieval query or failure note is not a resolved dataset binding.
   - **DO NOT** reproduce handoff text verbatim in any step's `description`, `task`, `method`, or `validation` fields. Resolve it through concrete planning choices so the step content reflects the resolution rather than quoting the note.
5. **StepPlan Ownership**:
   - The task_plan MUST cover data access/acquisition, preprocessing, scale alignment, threshold or window calibration when needed, analysis/modeling, and validation/evaluation.
   - StepPlan owns operational choices such as data acquisition route, preprocessing workflow, spatial/temporal scale alignment, threshold calibration, validation protocol, and evaluation criteria.
   - Do not treat upstream idea prose as an execution contract. Turn it into explicit, checkable steps and resolve underspecified operational choices inside the task_plan.
6. **Structured Dataset Contract**:
   - Return top-level `required_datasets` together with `task_plan`. `required_datasets` contains only additional external data beyond the provided base dataset.
   - If the accepted idea can be fully tested with the provided base dataset alone, return `required_datasets: []`. Do not add an auxiliary dataset merely to make the plan multi-source.
   - Candidate dataset evidence is non-binding context. Select a candidate only when the idea or an open point needs additional data; when selected, preserve its concrete title, catalog identifier, source, and relevant spatial/temporal constraints in the structured requirement fields.
   - In final-plan generation, the verified frozen requirement contract is authoritative. Do not introduce a new requirement from candidate evidence or prose.
   - Every non-empty requirement must state `id`, `name`, `research_justification`, `execution_requirement_level`, `binding_notes`, `modality`, `source_preference`, and `retrieval_hint`.
   - The output budget is strict: keep `research_justification` and `binding_notes` to at most two short sentences each, `retrieval_hint` to one short sentence, and every step `description` to at most two short sentences; never pad fields with long prose.
   - Put non-substitutable scientific identity in `frozen_constraints`: `entity`, `observable`, `measure`, and `dataset_family`. Put spatial scope only in `region`; region names are invalid entity values. A provider or catalog token never substitutes for observable identity.
   - Within one identity dimension, multiple values mean ALTERNATIVES: a candidate satisfying any one of them satisfies that dimension. List synonyms (`observable: ["soil_moisture", "water_content"]`) and mutually exclusive acceptable sources (`entity: ["GRACE", "GRACE-FO", "GLDAS"]`) this way — no single product can be all of them simultaneously.
   - Only when every listed value must hold at once, because the signal is a composite quantity, declare it explicitly: `observable: {{"values": ["precipitation", "evapotranspiration", "climatic_water_balance"], "match": "all"}}`. Never use `match: "all"` for synonyms or alternative sources; that makes the requirement impossible to satisfy and the plan will be rejected.
   - For time-varying requirements, use structured `temporal_coverage` (`start`, `end`, `coverage_mode`) and `temporal_resolution` (`unit`, `interval`, `allow_finer_resolution`). Do not emit free-text `time_range`.
   - Every step must declare a non-empty `inputs` array. Use `source_kind=base_dataset` with `source_id=base_dataset` for the provided task dataset, `source_kind=required_dataset` with an exact top-level requirement ID for external data, and `source_kind=step_output` with an exact `outputs[].id` for an upstream artifact.
   - Every step must declare exactly one `outputs` item with a globally unique artifact ID, data type, and role. Split work into separate steps when it has multiple primary outputs.
   - Declare dependencies explicitly even when the producer is not adjacent in the returned array. Do not infer dependencies from step order.
   - Producer and consumer data types must match exactly on every `step_output` edge. If conversion is required, emit an explicit conversion step and artifact; the validator will not insert one.
   - The plan must contain at least one output whose role is `hypothesis_test` or `validation`.
7. **Data-Centric Execution**:
   - Use the provided dataset characteristics as the primary data context.
   - Add steps to acquire, inspect, harmonize, and integrate additional datasets only when the idea or open points require them.
8. **Tool Allocation**: Assign specific Python libraries for each step (pandas, geopandas, rasterio, xarray, sklearn, etc.), but do NOT include internal retrieval tools here. Tools belong in the `tools` field of each step — the `description` field should articulate the analytical reasoning.
9. **Language Contract**: All free-text values in the returned task plan and required dataset contract must be written in English. Keep schema keys, enum-like values, catalog identifiers, and code/tool identifiers unchanged.
{"10. **Replan Discipline**: This is a replanning round. You MUST rewrite the full plan and full required_datasets contract, preserve valid intent, and explicitly resolve every blocking issue described in the failure and refinement feedback." if plan_refinement_feedback or resource_check_feedback else ""}
{replan_hard_constraints}
{"IMPORTANT: The new plan must remain a complete task plan, not a patch or diff. Every blocking dataset/tool issue from the feedback must be addressed." if plan_refinement_feedback or resource_check_feedback else ""}

Generate a JSON response containing top-level `required_datasets` and `task_plan`.
"""
        return prompt

    def _build_step_plan_prompt(
        self,
        idea_payload: Dict[str, Any],
        dataset_info: Dict[str, Any],
        previous_task_plan: Optional[List[Dict[str, Any]]] = None,
        previous_required_datasets: Optional[List[Dict[str, Any]]] = None,
        plan_refinement_feedback: Optional[Dict[str, Any]] = None,
        resource_check_feedback: Optional[Dict[str, Any]] = None,
        replan_round: int = 0,
        plan_repair_feedback: Optional[Dict[str, Any]] = None,
        frozen_required_datasets: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Build a StepPlan prompt from the only supported upstream payload: an idea."""
        if idea_payload.get("payload_type") != "idea" or not isinstance(idea_payload.get("idea"), dict):
            raise AgentExecutionError("StepPlanAgent requires a structured idea payload")
        return self._build_plan_prompt_from_idea_payload(
            idea_payload=idea_payload,
            dataset_info=dataset_info,
            previous_task_plan=previous_task_plan,
            previous_required_datasets=previous_required_datasets,
            plan_refinement_feedback=plan_refinement_feedback,
            resource_check_feedback=resource_check_feedback,
            replan_round=replan_round,
            plan_repair_feedback=plan_repair_feedback,
            frozen_required_datasets=frozen_required_datasets,
        )

    def _build_graph_phase_prompt(self, **prompt_kwargs: Any) -> str:
        base_prompt = self._build_step_plan_prompt(**prompt_kwargs)
        return base_prompt + """

Hierarchical StepPlan phase: GRAPH_DRAFT.
Return only the scientific dependency graph. Each task_plan item must contain exactly
id, step_name, task_type, inputs, and outputs. Do not emit description, tools, or any
capability-anchor fields in this phase. Keep the graph concise, but make every dataset
and artifact dependency explicit. The next phase will expand implementation details
without permission to change this graph.
"""

    def _build_detail_phase_prompt(
        self,
        *,
        idea_payload: Dict[str, Any],
        dataset_info: Dict[str, Any],
        required_datasets: List[Dict[str, Any]],
        graph_steps: List[Dict[str, Any]],
        requested_step_ids: List[int],
        detail_issues: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        return f"""
Hierarchical StepPlan phase: STEP_DETAIL_EXPANSION.

Accepted idea:
{json.dumps(self._idea_core(idea_payload), ensure_ascii=False, separators=(',', ':'))}

Dataset characteristics:
{json.dumps(dataset_info, ensure_ascii=False, separators=(',', ':'))}

Frozen required datasets:
{json.dumps(required_datasets, ensure_ascii=False, separators=(',', ':'))}

Frozen validated graph:
{json.dumps(graph_steps, ensure_ascii=False, separators=(',', ':'))}

Requested step IDs: {json.dumps(requested_step_ids)}
Detail validation issues:
{json.dumps(detail_issues or [], ensure_ascii=False, separators=(',', ':'))}

Return description and tools only for every requested step ID. Do not repeat or alter
step_name, task_type, inputs, outputs, required_datasets, the idea, or the graph. Each
description must state the concrete operation and validation intent in English. Tools
must contain concrete Python packages and must not contain internal retrieval tools.
"""

    async def _repair_graph(
        self,
        *,
        idea_payload: Dict[str, Any],
        dataset_info: Dict[str, Any],
        required_datasets: List[Dict[str, Any]],
        graph_steps: List[Dict[str, Any]],
        issues: List[Dict[str, Any]],
        audit: Dict[str, Any],
        phase_prefix: str,
        max_attempts: int,
        frozen_required_datasets: List[Dict[str, Any]],
        allowed_step_ids: Optional[set[int]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        current_requirements = copy.deepcopy(required_datasets)
        current_steps = copy.deepcopy(graph_steps)
        current_issues = copy.deepcopy(issues)
        current_meta: Dict[str, Any] = {}

        for attempt in range(1, max_attempts + 1):
            if self._requires_graph_regeneration(current_issues):
                prompt = f"""
Hierarchical StepPlan phase: GRAPH_REGENERATION.

Accepted idea (FROZEN):
{json.dumps(self._idea_core(idea_payload), ensure_ascii=False, separators=(',', ':'))}

Dataset characteristics:
{json.dumps(dataset_info, ensure_ascii=False, separators=(',', ':'))}

Verified required datasets (FROZEN):
{json.dumps([requirement_core(item) for item in frozen_required_datasets], ensure_ascii=False, separators=(',', ':'))}

Current invalid required datasets:
{json.dumps(current_requirements, ensure_ascii=False, separators=(',', ':'))}

Current invalid graph:
{json.dumps(current_steps, ensure_ascii=False, separators=(',', ':'))}

Deterministic validation issues:
{json.dumps(current_issues, ensure_ascii=False, separators=(',', ':'))}

Node identity is invalid, so an ID-scoped patch cannot be applied safely. Return a
complete replacement `task_plan` graph and exactly echo the verified frozen
`required_datasets`; do not add, remove, rename, or reword any requirement. Every step
must have a unique integer ID. Do not emit descriptions, tools, or capability-anchor
fields. Never modify the accepted idea.
"""
                regenerated = await self._call_planning_phase(
                    phase=f"{phase_prefix}_graph_regeneration_{attempt}",
                    prompt=prompt,
                    schema=self._build_graph_schema(),
                    audit=audit,
                )
                raw_requirements = regenerated.get("required_datasets")
                raw_steps = regenerated.get("task_plan")
                current_steps = (
                    [self._graph_step(step) for step in raw_steps]
                    if isinstance(raw_steps, list)
                    else []
                )
                current_issues, current_meta = self._graph_contract_issues(
                    frozen_required_datasets,
                    current_steps,
                )
                if not self._required_dataset_contracts_match(
                    frozen_required_datasets,
                    raw_requirements,
                ):
                    current_issues.append(
                        {
                            "code": "verified_requirement_contract_drift",
                            "message": (
                                "graph regeneration changed the verified required_datasets contract"
                            ),
                        }
                    )
                current_requirements = copy.deepcopy(frozen_required_datasets)
                audit.setdefault("validation_passes", []).append(
                    {
                        "phase": f"{phase_prefix}_graph_regeneration_{attempt}",
                        "valid": not current_issues,
                        "issues": copy.deepcopy(current_issues),
                    }
                )
                if not current_issues:
                    return current_requirements, current_steps, current_meta
                continue

            repair_scope = allowed_step_ids or self._issue_step_ids(
                current_issues,
                current_steps,
            )
            prompt = f"""
Hierarchical StepPlan phase: GRAPH_REPAIR.

Current required datasets:
{json.dumps(current_requirements, ensure_ascii=False, separators=(',', ':'))}

Current graph:
{json.dumps(current_steps, ensure_ascii=False, separators=(',', ':'))}

Deterministic validation issues:
{json.dumps(current_issues, ensure_ascii=False, separators=(',', ':'))}

Allowed existing step IDs: {json.dumps(sorted(repair_scope))}

Verified Required Datasets (FROZEN):
{json.dumps([requirement_core(item) for item in frozen_required_datasets], ensure_ascii=False, separators=(',', ':'))}

Return a minimal graph patch. `required_datasets` must exactly echo the verified frozen
contract above, character for character: do not add, remove, rename, reword, or reformat
any requirement field (including `research_justification`). Any wording drift is rejected
deterministically. Only graph nodes may change: `step_replacements` may contain only
allowed existing IDs; do not repeat unchanged steps. Use `step_additions` or
`step_removals` only when the validator issue cannot be closed by replacing an affected
node. Do not emit descriptions or tools. Never modify the accepted idea.
"""
            patch = await self._call_planning_phase(
                phase=f"{phase_prefix}_graph_repair_{attempt}",
                prompt=prompt,
                schema=self._build_graph_patch_schema(),
                audit=audit,
            )
            try:
                if not self._required_dataset_contracts_match(
                    frozen_required_datasets,
                    patch.get("required_datasets"),
                ):
                    # 回显漂移不代表步骤补丁无效：冻结契约本来就将在应用补丁前
                    # 覆盖回写，这里只记录漂移并保留步骤侧修复进度，交给下一轮带着
                    # 明确 issue 重试，而不是丢弃整个补丁直接终止。
                    current_issues.append(
                        {
                            "code": "verified_requirement_contract_drift",
                            "message": (
                                "graph repair reworded the verified required_datasets contract; "
                                "echo it exactly as provided"
                            ),
                        }
                    )
                guarded_patch = copy.deepcopy(patch)
                guarded_patch["required_datasets"] = copy.deepcopy(
                    frozen_required_datasets
                )
                current_requirements, current_steps = self._apply_graph_patch(
                    required_datasets=current_requirements,
                    steps=current_steps,
                    patch=guarded_patch,
                    allowed_step_ids=repair_scope,
                )
                current_issues, current_meta = self._graph_contract_issues(
                    current_requirements,
                    current_steps,
                )
            except AgentExecutionError as exc:
                current_issues = [
                    {"code": "graph_patch_guard_rejected", "message": str(exc)}
                ]
                current_meta = {
                    "plan_dag_valid": False,
                    "plan_dag_issues": copy.deepcopy(current_issues),
                }
            audit.setdefault("validation_passes", []).append(
                {
                    "phase": f"{phase_prefix}_graph_repair_{attempt}",
                    "valid": not current_issues,
                    "issues": copy.deepcopy(current_issues),
                }
            )
            if not current_issues:
                return current_requirements, current_steps, current_meta

        raise StepPlanContractError(
            [str(issue["message"]) for issue in current_issues],
            required_datasets=current_requirements,
            task_plan=current_steps,
            capability_anchor_meta=current_meta,
            generation_audit=audit,
        )

    async def repair_graph_contract(
        self,
        *,
        idea_payload: Dict[str, Any],
        dataset_info: Dict[str, Any],
        required_datasets: List[Dict[str, Any]],
        task_plan: List[Any],
        max_attempts: int,
        phase_prefix: str = "shared_graph_contract",
    ) -> Dict[str, Any]:
        """Repair an invalid DAG while freezing the idea and dataset contract.

        This is the public adapter boundary shared by ours and external-baseline
        planning. It deliberately exposes only structural graph repair: the
        accepted idea and the regularized ``required_datasets`` contract are
        immutable, and no ResourceCheck or refinement feedback is consumed.
        """
        if not isinstance(required_datasets, list) or any(
            not isinstance(item, dict) for item in required_datasets
        ):
            raise StepPlanContractError(
                ["graph repair requires a structured required_datasets list"],
                required_datasets=(
                    required_datasets if isinstance(required_datasets, list) else []
                ),
                task_plan=task_plan if isinstance(task_plan, list) else [],
            )
        if not isinstance(task_plan, list):
            raise StepPlanContractError(
                ["graph repair requires task_plan to be a list"],
                required_datasets=required_datasets,
            )
        if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or max_attempts <= 0:
            raise AgentExecutionError("graph repair max_attempts must be a positive integer")

        frozen_requirements = copy.deepcopy(required_datasets)
        graph_steps = [self._graph_step(step) for step in task_plan]
        issues, graph_meta = self._graph_contract_issues(
            frozen_requirements,
            graph_steps,
        )
        audit: Dict[str, Any] = {
            "mode": "shared_structural_graph_repair",
            "model_calls": [],
            "validation_passes": [],
        }

        if not issues:
            raise StepPlanContractError(
                ["planning contract failed outside the repairable DAG boundary"],
                required_datasets=frozen_requirements,
                task_plan=task_plan,
                capability_anchor_meta=graph_meta,
                generation_audit=audit,
            )

        repaired_requirements, repaired_graph, _ = await self._repair_graph(
            idea_payload=copy.deepcopy(idea_payload),
            dataset_info=copy.deepcopy(dataset_info),
            required_datasets=frozen_requirements,
            graph_steps=graph_steps,
            issues=issues,
            audit=audit,
            phase_prefix=phase_prefix,
            max_attempts=max_attempts,
            frozen_required_datasets=frozen_requirements,
        )
        detailed_steps = await self._expand_step_details(
            idea_payload=copy.deepcopy(idea_payload),
            dataset_info=copy.deepcopy(dataset_info),
            required_datasets=repaired_requirements,
            graph_steps=repaired_graph,
            audit=audit,
            phase_prefix=phase_prefix,
        )
        return self._finalize_hierarchical_plan(
            required_datasets=repaired_requirements,
            detailed_steps=detailed_steps,
            audit=audit,
        )

    async def _expand_step_details(
        self,
        *,
        idea_payload: Dict[str, Any],
        dataset_info: Dict[str, Any],
        required_datasets: List[Dict[str, Any]],
        graph_steps: List[Dict[str, Any]],
        audit: Dict[str, Any],
        phase_prefix: str,
    ) -> List[Dict[str, Any]]:
        _, _, _, detail_repair_attempts = self._runtime_planning_config()
        step_ids = [int(step["id"]) for step in graph_steps]
        prompt = self._build_detail_phase_prompt(
            idea_payload=idea_payload,
            dataset_info=dataset_info,
            required_datasets=required_datasets,
            graph_steps=graph_steps,
            requested_step_ids=step_ids,
        )
        response = await self._call_planning_phase(
            phase=f"{phase_prefix}_detail_expansion",
            prompt=prompt,
            schema=self._build_detail_schema(step_ids),
            audit=audit,
        )
        detail_by_id, issues = self._validate_step_details(
            response.get("step_details"),
            set(step_ids),
        )
        audit.setdefault("validation_passes", []).append(
            {
                "phase": f"{phase_prefix}_detail_expansion",
                "valid": not issues,
                "issues": copy.deepcopy(issues),
            }
        )

        for attempt in range(1, detail_repair_attempts + 1):
            if not issues:
                break
            repair_ids = sorted(
                {
                    issue.get("step_id")
                    for issue in issues
                    if isinstance(issue.get("step_id"), int)
                }
                & set(step_ids)
            ) or step_ids
            repair_prompt = self._build_detail_phase_prompt(
                idea_payload=idea_payload,
                dataset_info=dataset_info,
                required_datasets=required_datasets,
                graph_steps=graph_steps,
                requested_step_ids=repair_ids,
                detail_issues=issues,
            )
            repair_response = await self._call_planning_phase(
                phase=f"{phase_prefix}_detail_repair_{attempt}",
                prompt=repair_prompt,
                schema=self._build_detail_schema(repair_ids),
                audit=audit,
            )
            repaired, repair_issues = self._validate_step_details(
                repair_response.get("step_details"),
                set(repair_ids),
            )
            detail_by_id.update(repaired)
            _, issues = self._validate_step_details(
                list(detail_by_id.values()),
                set(step_ids),
            )
            if repair_issues:
                issues.extend(repair_issues)
            audit.setdefault("validation_passes", []).append(
                {
                    "phase": f"{phase_prefix}_detail_repair_{attempt}",
                    "valid": not issues,
                    "issues": copy.deepcopy(issues),
                }
            )

        if issues:
            raise StepPlanContractError(
                [str(issue["message"]) for issue in issues],
                required_datasets=required_datasets,
                task_plan=graph_steps,
                generation_audit=audit,
            )

        detailed_steps: List[Dict[str, Any]] = []
        for graph_step in graph_steps:
            step = copy.deepcopy(graph_step)
            detail = detail_by_id[step["id"]]
            step["description"] = detail["description"].strip()
            step["tools"] = [tool.strip() for tool in detail["tools"]]
            step["required_runtime_tools"] = [
                tool.strip()
                for tool in detail.get("required_runtime_tools", [])
            ]
            detailed_steps.append(step)
        return detailed_steps

    def _finalize_hierarchical_plan(
        self,
        *,
        required_datasets: List[Dict[str, Any]],
        detailed_steps: List[Dict[str, Any]],
        audit: Dict[str, Any],
        operation_constraints: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        operation_constraints = operation_constraints or {}
        # 重定位必须发生在 capability 锚定之前：约束同时驱动函数选择和闭合校验，
        # 若沿用旧编号，锚定会先在错误约束下选函数，闭合失败只是其后果。
        relocated_constraints, unresolved_constraints = relocate_step_constraints(
            detailed_steps,
            operation_constraints,
        )
        detailed_steps = self._apply_operation_constraints(
            detailed_steps,
            relocated_constraints,
        )
        graph_issues, graph_meta = self._graph_contract_issues(
            required_datasets,
            detailed_steps,
        )
        if graph_issues:
            raise StepPlanContractError(
                [str(issue["message"]) for issue in graph_issues],
                required_datasets=required_datasets,
                task_plan=detailed_steps,
                capability_anchor_meta=graph_meta,
                generation_audit=audit,
            )

        anchored_steps, anchor_meta = self._anchor_plan_to_capabilities(
            required_datasets,
            detailed_steps,
        )
        contract_issues = self._validate_structured_step_contract(anchored_steps)
        contract_issues.extend(
            self._validate_required_dataset_contract(required_datasets, anchored_steps)
        )
        closure_issues, closure_records = self._validate_refinement_constraint_closure(
            anchored_steps,
            relocated_constraints,
            unresolved_constraints=unresolved_constraints,
        )
        contract_issues.extend(closure_issues)
        if operation_constraints:
            anchor_meta["refinement_constraint_closure"] = closure_records
            anchor_meta["refinement_constraints_closed"] = not closure_issues
            anchor_meta["refinement_constraint_relocation"] = {
                str(source): {
                    "target_step_id": target,
                    "match": entry.get("relocation_match"),
                    "score": entry.get("relocation_score"),
                }
                for target, entry in relocated_constraints.items()
                for source in [entry.get("source_step_id", target)]
            }
        if not anchor_meta.get("plan_dag_valid", False):
            contract_issues.extend(
                str(issue["message"])
                for issue in anchor_meta.get("plan_dag_issues", [])
            )
        contract_issues = list(dict.fromkeys(contract_issues))
        audit.setdefault("validation_passes", []).append(
            {
                "phase": "final_anchored_contract",
                "valid": not contract_issues,
                "issues": copy.deepcopy(contract_issues),
            }
        )
        if contract_issues:
            coverage_gap = (
                bool(closure_issues)
                and set(contract_issues) == set(closure_issues)
                and all(
                    record.get("gap_type") == "coverage_gap"
                    for record in closure_records
                    if not record.get("compatible", True)
                )
            )
            raise StepPlanContractError(
                contract_issues,
                required_datasets=required_datasets,
                task_plan=anchored_steps,
                capability_anchor_meta=anchor_meta,
                generation_audit=audit,
                coverage_gap=coverage_gap,
            )

        audit["model_call_count"] = len(audit.get("model_calls", []))
        audit["final_step_count"] = len(anchored_steps)
        audit["final_required_dataset_count"] = len(required_datasets)
        return {
            "required_datasets": copy.deepcopy(required_datasets),
            "task_plan": anchored_steps,
            "capability_anchor_meta": anchor_meta,
            "step_plan_generation_audit": audit,
        }

    async def _execute_hierarchical_full_plan(
        self,
        *,
        idea_payload: Dict[str, Any],
        dataset_info: Dict[str, Any],
        previous_task_plan: Optional[List[Dict[str, Any]]],
        previous_required_datasets: Optional[List[Dict[str, Any]]],
        plan_refinement_feedback: Optional[Dict[str, Any]],
        resource_check_feedback: Optional[Dict[str, Any]],
        plan_repair_feedback: Optional[Dict[str, Any]],
        replan_round: int,
        audit: Dict[str, Any],
        phase_prefix: str,
        verified_required_datasets: List[Dict[str, Any]],
        operation_constraints: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        _, _, graph_repair_attempts, _ = self._runtime_planning_config()
        graph_prompt = self._build_graph_phase_prompt(
            idea_payload=idea_payload,
            dataset_info=dataset_info,
            previous_task_plan=previous_task_plan,
            previous_required_datasets=previous_required_datasets,
            plan_refinement_feedback=plan_refinement_feedback,
            resource_check_feedback=resource_check_feedback,
            replan_round=replan_round,
            plan_repair_feedback=plan_repair_feedback,
            frozen_required_datasets=verified_required_datasets,
        )
        graph_response = await self._call_planning_phase(
            phase=f"{phase_prefix}_graph_draft",
            prompt=graph_prompt,
            schema=self._build_graph_schema(),
            audit=audit,
        )
        raw_required_datasets = graph_response.get("required_datasets")
        required_datasets = copy.deepcopy(verified_required_datasets)
        raw_steps = graph_response.get("task_plan")
        graph_steps = (
            [self._graph_step(step) for step in raw_steps]
            if isinstance(raw_steps, list)
            else []
        )
        issues, graph_meta = self._graph_contract_issues(
            required_datasets,
            graph_steps,
        )
        if not self._required_dataset_contracts_match(
            verified_required_datasets,
            raw_required_datasets,
        ):
            issues.append(
                {
                    "code": "verified_requirement_contract_drift",
                    "message": (
                        "final graph draft changed the verified required_datasets contract"
                    ),
                }
            )
        audit.setdefault("validation_passes", []).append(
            {
                "phase": f"{phase_prefix}_graph_draft",
                "valid": not issues,
                "issues": copy.deepcopy(issues),
            }
        )
        if issues:
            required_datasets, graph_steps, graph_meta = await self._repair_graph(
                idea_payload=copy.deepcopy(idea_payload),
                dataset_info=copy.deepcopy(dataset_info),
                required_datasets=required_datasets,
                graph_steps=graph_steps,
                issues=issues,
                audit=audit,
                phase_prefix=phase_prefix,
                max_attempts=graph_repair_attempts,
                frozen_required_datasets=verified_required_datasets,
            )
        audit["graph_validation"] = copy.deepcopy(graph_meta)
        detailed_steps = await self._expand_step_details(
            idea_payload=idea_payload,
            dataset_info=dataset_info,
            required_datasets=required_datasets,
            graph_steps=graph_steps,
            audit=audit,
            phase_prefix=phase_prefix,
        )
        return self._finalize_hierarchical_plan(
            required_datasets=required_datasets,
            detailed_steps=detailed_steps,
            audit=audit,
            operation_constraints=operation_constraints,
        )

    @staticmethod
    def _normalize_feedback_step_id(value: Any) -> Optional[int]:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, str):
            match = re.fullmatch(r"(?:step[_\s-]*)?(\d+)", value.strip(), re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    def _scoped_patch_step_ids(
        self,
        *,
        previous_steps: List[Dict[str, Any]],
        plan_refinement_feedback: Dict[str, Any],
        resource_check_feedback: Dict[str, Any],
    ) -> set[int]:
        """Derive the patch boundary from existing structured feedback only."""
        affected: set[int] = set()
        for raw_value in resource_check_feedback.get("affected_steps", []) or []:
            step_id = self._normalize_feedback_step_id(raw_value)
            if step_id is not None:
                affected.add(step_id)
        weakest_step_id = self._normalize_feedback_step_id(
            resource_check_feedback.get("weakest_step_id")
        )
        if weakest_step_id is not None:
            affected.add(weakest_step_id)
        for failure in resource_check_feedback.get("step_failures", []) or []:
            if not isinstance(failure, dict):
                continue
            step_id = self._normalize_feedback_step_id(failure.get("step_id"))
            if step_id is not None:
                affected.add(step_id)
        for adjustment in plan_refinement_feedback.get("step_level_adjustments", []) or []:
            if not isinstance(adjustment, dict):
                continue
            step_id = self._normalize_feedback_step_id(adjustment.get("step_id"))
            if step_id is not None:
                affected.add(step_id)
        for issue in plan_refinement_feedback.get("semantic_alignment_issues", []) or []:
            if not isinstance(issue, dict):
                continue
            for raw_value in issue.get("affected_step_ids", []) or []:
                step_id = self._normalize_feedback_step_id(raw_value)
                if step_id is not None:
                    affected.add(step_id)

        failing_requirement_ids = {
            str(item.get("requirement_id") or "").strip()
            for item in resource_check_feedback.get("dataset_resolution_failures", []) or []
            if isinstance(item, dict) and str(item.get("requirement_id") or "").strip()
        }
        for step in previous_steps:
            if not isinstance(step, dict):
                continue
            if any(
                isinstance(item, dict)
                and item.get("source_kind") == "required_dataset"
                and str(item.get("source_id") or "").strip() in failing_requirement_ids
                for item in (step.get("inputs") or [])
            ):
                affected.add(step.get("id"))

        existing_ids = {
            step.get("id")
            for step in previous_steps
            if isinstance(step, dict)
            and isinstance(step.get("id"), int)
            and not isinstance(step.get("id"), bool)
        }
        affected &= existing_ids
        if not affected:
            return set()

        output_to_step = {
            output.get("id"): step.get("id")
            for step in previous_steps
            if isinstance(step, dict)
            for output in (step.get("outputs") or [])
            if isinstance(output, dict)
        }
        expanded = set(affected)
        for step in previous_steps:
            if not isinstance(step, dict):
                continue
            step_id = step.get("id")
            for item in step.get("inputs") or []:
                if not isinstance(item, dict) or item.get("source_kind") != "step_output":
                    continue
                producer_id = output_to_step.get(item.get("source_id"))
                if step_id in affected and producer_id in existing_ids:
                    expanded.add(producer_id)
                if producer_id in affected and step_id in existing_ids:
                    expanded.add(step_id)
        return expanded

    async def _execute_scoped_patch(
        self,
        *,
        idea_payload: Dict[str, Any],
        dataset_info: Dict[str, Any],
        previous_task_plan: List[Dict[str, Any]],
        previous_required_datasets: List[Dict[str, Any]],
        verified_required_datasets: List[Dict[str, Any]],
        plan_refinement_feedback: Dict[str, Any],
        resource_check_feedback: Dict[str, Any],
        audit: Dict[str, Any],
        operation_constraints: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        graph_steps = [self._graph_step(step) for step in previous_task_plan]
        initial_issues, _ = self._graph_contract_issues(
            previous_required_datasets,
            graph_steps,
        )
        if initial_issues:
            audit["routing"] = {
                "route": "full_replan_required",
                "reason": "previous plan graph is not a valid patch base",
                "issues": copy.deepcopy(initial_issues),
            }
            return None

        allowed_step_ids = self._scoped_patch_step_ids(
            previous_steps=previous_task_plan,
            plan_refinement_feedback=plan_refinement_feedback,
            resource_check_feedback=resource_check_feedback,
        )
        if not allowed_step_ids:
            audit["routing"] = {
                "route": "full_replan_required",
                "reason": "structured feedback did not identify a safe local step scope",
            }
            return None

        prompt = f"""
Hierarchical StepPlan phase: SCOPED_PLAN_PATCH.

Accepted idea, which is immutable:
{json.dumps(self._idea_core(idea_payload), ensure_ascii=False, separators=(',', ':'))}

Dataset characteristics:
{json.dumps(dataset_info, ensure_ascii=False, separators=(',', ':'))}

Current required datasets:
{json.dumps([requirement_core(item) for item in previous_required_datasets], ensure_ascii=False, separators=(',', ':'))}

Verified replacement required datasets (FROZEN):
{json.dumps([requirement_core(item) for item in verified_required_datasets], ensure_ascii=False, separators=(',', ':'))}

Current anchored plan:
{json.dumps([self._patch_context_step(step) for step in previous_task_plan], ensure_ascii=False, separators=(',', ':'))}

ResourceCheck feedback:
{json.dumps(resource_check_feedback, ensure_ascii=False, separators=(',', ':'))}

Refinement feedback:
{json.dumps(plan_refinement_feedback, ensure_ascii=False, separators=(',', ':'))}

Allowed existing step IDs, including their immediate graph neighbors:
{json.dumps(sorted(allowed_step_ids))}

Choose `local_patch` only when the issue can be resolved without changing any existing
step outside this scope. `required_datasets` must exactly echo the verified frozen replacement
contract; adding, removing, or modifying a requirement is forbidden. Put changed existing graph nodes in `step_replacements`, new conversion nodes in
`step_additions`, removed scoped nodes in `step_removals`, and descriptions/tools for every
replaced or added node in `detail_updates`. Do not repeat unchanged nodes. Choose
`full_replan_required` with empty operations when the accepted method or validation route
must change globally. Never modify the idea.
"""
        patch = await self._call_planning_phase(
            phase="scoped_plan_patch",
            prompt=prompt,
            schema=self._build_scoped_patch_schema(sorted(allowed_step_ids)),
            audit=audit,
        )
        route = patch.get("route")
        if route == "full_replan_required":
            audit["routing"] = {
                "route": route,
                "reason": str(patch.get("route_reason") or ""),
            }
            return None
        if route != "local_patch":
            raise AgentExecutionError(f"unsupported scoped patch route: {route}")

        if not self._required_dataset_contracts_match(
            verified_required_datasets,
            patch.get("required_datasets"),
        ):
            # 与 GRAPH_REPAIR 一致：回显漂移已被下方的冻结契约覆盖中和，不应丢弃
            # 已通过校验的步骤补丁；记录 issue 后继续应用受保护的补丁。
            audit["requirement_contract_drift"] = (
                "scoped patch reworded the verified required_datasets contract; "
                "the frozen contract was substituted before applying the patch"
            )
        guarded_patch = copy.deepcopy(patch)
        guarded_patch["required_datasets"] = copy.deepcopy(
            verified_required_datasets
        )
        patched_requirements, patched_graph = self._apply_graph_patch(
            required_datasets=previous_required_datasets,
            steps=graph_steps,
            patch=guarded_patch,
            allowed_step_ids=allowed_step_ids,
        )
        graph_issues, graph_meta = self._graph_contract_issues(
            patched_requirements,
            patched_graph,
        )
        if graph_issues:
            audit["routing"] = {
                "route": "full_replan_required",
                "reason": "local graph patch did not close the deterministic contract",
                "issues": copy.deepcopy(graph_issues),
            }
            return None

        replacement_ids = {
            step.get("id")
            for step in patch.get("step_replacements", []) or []
            if isinstance(step, dict)
        }
        addition_ids = {
            step.get("id")
            for step in patch.get("step_additions", []) or []
            if isinstance(step, dict)
        }
        required_detail_ids = replacement_ids | addition_ids
        raw_detail_updates = patch.get("detail_updates")
        if not isinstance(raw_detail_updates, list):
            raise AgentExecutionError("scoped patch detail_updates must be a list")
        detail_update_ids = {
            item.get("id")
            for item in raw_detail_updates
            if isinstance(item, dict)
            and isinstance(item.get("id"), int)
            and not isinstance(item.get("id"), bool)
        }
        if not required_detail_ids.issubset(detail_update_ids):
            missing = sorted(required_detail_ids - detail_update_ids)
            raise AgentExecutionError(
                f"scoped patch is missing details for changed steps: {missing}"
            )
        allowed_detail_ids = allowed_step_ids | addition_ids
        if not detail_update_ids.issubset(allowed_detail_ids):
            raise AgentExecutionError("scoped patch contains detail updates outside its scope")
        detail_updates, detail_issues = self._validate_step_details(
            raw_detail_updates,
            detail_update_ids,
        ) if detail_update_ids else ({}, [])
        if detail_issues:
            raise AgentExecutionError(
                "invalid scoped detail updates: "
                + "; ".join(str(issue["message"]) for issue in detail_issues)
            )

        prior_details = {
            step.get("id"): {
                "description": step.get("description"),
                "tools": copy.deepcopy(step.get("tools")),
                "required_runtime_tools": copy.deepcopy(
                    step.get("required_runtime_tools", [])
                ),
            }
            for step in previous_task_plan
            if isinstance(step, dict)
        }
        removal_ids = set(patch.get("step_removals") or [])
        detailed_steps: List[Dict[str, Any]] = []
        for graph_step in patched_graph:
            step_id = graph_step["id"]
            detail = detail_updates.get(step_id) or prior_details.get(step_id)
            if not isinstance(detail, dict):
                raise AgentExecutionError(f"scoped patch has no details for step {step_id}")
            step = copy.deepcopy(graph_step)
            step["description"] = detail.get("description")
            step["tools"] = copy.deepcopy(detail.get("tools"))
            step["required_runtime_tools"] = copy.deepcopy(
                detail.get("required_runtime_tools", [])
            )
            detailed_steps.append(step)
        if removal_ids & {step["id"] for step in detailed_steps}:
            raise AgentExecutionError("scoped patch removal was not applied")

        audit["routing"] = {
            "route": "local_patch",
            "reason": str(patch.get("route_reason") or ""),
            "allowed_step_ids": sorted(allowed_step_ids),
            "changed_step_ids": sorted(required_detail_ids),
        }
        audit["graph_validation"] = graph_meta
        return self._finalize_hierarchical_plan(
            required_datasets=patched_requirements,
            detailed_steps=detailed_steps,
            audit=audit,
            operation_constraints=operation_constraints,
        )

    # =====================================================================
    # Execute
    # =====================================================================

    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行步骤规划任务。

        Args:
            context:
                - idea: dict (含 idea、planning_handoff、dataset_characteristics、candidate_datasets)
                - dataset_info: dict
                - previous_task_plan: list (optional, replan)
                - previous_required_datasets: list (optional, replan)
                - verified_required_datasets: list with valid binding proofs
                - step_plan_requirement_audit: dict from requirement draft/resolution
                - plan_refinement_feedback: dict (optional)
                - resource_check_feedback: dict (optional)
                - plan_repair_feedback: dict (optional)
            params:
                - mode: "plan_for_idea" | "replan"
                - replan_round: int (optional)

        Returns:
            {"required_datasets": list, "task_plan": anchored_steps, "capability_anchor_meta": meta}
        """
        self._runtime_planning_config()
        idea_payload = context["idea"]
        dataset_info = context.get("dataset_info", {})
        previous_task_plan = context.get("previous_task_plan")
        previous_required_datasets = context.get("previous_required_datasets")
        verified_required_datasets = context.get("verified_required_datasets")
        requirement_audit = context.get("step_plan_requirement_audit")
        plan_refinement_feedback = context.get("plan_refinement_feedback") or {}
        resource_check_feedback = context.get("resource_check_feedback") or {}
        plan_repair_feedback = context.get("plan_repair_feedback")
        replan_round = int(params.get("replan_round", 0) or 0)
        mode = str(params.get("mode") or "").strip()
        if mode not in {"plan_for_idea", "replan"}:
            raise AgentExecutionError(f"unsupported StepPlan mode: {mode}")

        if not isinstance(verified_required_datasets, list):
            raise AgentExecutionError(
                "final StepPlan requires verified_required_datasets as a list"
            )
        if not isinstance(requirement_audit, dict):
            raise AgentExecutionError(
                "final StepPlan requires step_plan_requirement_audit"
            )
        proof_issues: List[str] = []
        for index, requirement in enumerate(verified_required_datasets):
            if not isinstance(requirement, dict):
                proof_issues.append(
                    f"verified_required_datasets[{index}] is not an object"
                )
                continue
            valid, reason, _ = validate_dataset_binding_proof(requirement)
            if not valid:
                proof_issues.append(
                    f"verified_required_datasets[{index}] {reason}"
                )
        if proof_issues:
            raise StepPlanContractError(
                proof_issues,
                required_datasets=(
                    verified_required_datasets
                    if isinstance(verified_required_datasets, list)
                    else []
                ),
                generation_audit=copy.deepcopy(requirement_audit),
            )

        audit: Dict[str, Any] = copy.deepcopy(requirement_audit)
        audit["version"] = "hierarchical_step_plan_v2"
        audit["mode"] = mode
        audit["replan_round"] = replan_round
        audit.setdefault("model_calls", [])
        audit.setdefault("validation_passes", [])
        audit["verified_required_dataset_count"] = len(
            verified_required_datasets
        )
        operation_constraints = self._compile_refinement_operation_constraints(
            plan_refinement_feedback,
            previous_task_plan,
            replan_round,
        )
        audit["compiled_operation_constraints"] = copy.deepcopy(operation_constraints)
        if mode == "replan":
            if not isinstance(previous_task_plan, list) or not previous_task_plan:
                raise AgentExecutionError("replan requires a non-empty previous_task_plan")
            if not isinstance(previous_required_datasets, list):
                raise AgentExecutionError("replan requires previous_required_datasets as a list")
            try:
                patched = await self._execute_scoped_patch(
                    idea_payload=idea_payload,
                    dataset_info=dataset_info,
                    previous_task_plan=previous_task_plan,
                    previous_required_datasets=previous_required_datasets,
                    verified_required_datasets=verified_required_datasets,
                    plan_refinement_feedback=plan_refinement_feedback,
                    resource_check_feedback=resource_check_feedback,
                    audit=audit,
                    operation_constraints=operation_constraints,
                )
            except AgentExecutionError as exc:
                audit["routing"] = {
                    "route": "full_replan_required",
                    "reason": f"scoped patch rejected: {exc}",
                }
                patched = None
            if patched is not None:
                return patched

        phase_prefix = "initial" if mode == "plan_for_idea" else "full_replan"
        return await self._execute_hierarchical_full_plan(
            idea_payload=idea_payload,
            dataset_info=dataset_info,
            previous_task_plan=previous_task_plan,
            previous_required_datasets=previous_required_datasets,
            plan_refinement_feedback=plan_refinement_feedback,
            resource_check_feedback=resource_check_feedback,
            plan_repair_feedback=plan_repair_feedback,
            replan_round=replan_round,
            audit=audit,
            phase_prefix=phase_prefix,
            verified_required_datasets=verified_required_datasets,
            operation_constraints=operation_constraints,
        )

#!/usr/bin/env python3
"""Evaluate the ours ResourceCheck axis and Alignment/Depth independently.

Run from the repository root through benchmark/run.py evaluate.
ResourceCheck summarizes generation-owned results for internagent outputs.
Default resource configuration: config/debug_config.yaml.
An explicit --resource-check-config selects a complete replacement configuration.
Source outputs remain unchanged; configuration fingerprints control cache reuse.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import yaml
from dotenv import load_dotenv

from benchmark.core.benchmark_resource_check import BenchmarkResourceChecker
from benchmark.core.benchmark_layout import (
    canonical_task_ids,
    grouped_task_dir,
    split_task_id,
    system_outputs_roots,
)

logger = logging.getLogger("axis_evaluation")

BENCHMARK_DIR = _PROJECT_ROOT / "benchmark"
TASK_INPUTS_DIR = BENCHMARK_DIR / "dataset" / "task_inputs"
SCRIPT_3B = BENCHMARK_DIR / "scripts" / "3b_alignment_scoring.py"

# 资源检查轴协议版本：口径变化时递增，旧产物自动作废重评。
# v2: 指标统一层重构——metrics.core（全系统同公式，首轮/末轮拆分）+
#     metrics.loop（仅 refinement 循环系统有定义，baseline 为 null）。
#     v1 的 ours 侧 pass_rate_at_round0 实为末轮通过率（名实不符），已废弃。
# v3: benchmark-owned 重构——baseline 改由 benchmark/scripts/benchmark_resource_check.py
#     独立检查，不再走生成侧 ResourcesCheck_Agent / PlannedContractAdapter。
# v4: 判定顺序修正——显式身份（锚点 / GEE id）先直接探测再谈目录匹配；目录侧
#     accepted_for_binding=True 不再因 status 非 resolved_strong 被判缺口；"有候选但
#     不达标"也回退一次在线。本地目录数据集补 is_dir() 分支。见 Benchmark-v3 §4.2.2。
# v5: 显式身份不再只认 GEE——锚点声明的非 GEE 平台身份（source_metadata 与
#     `NASA_CMR:`/`PLANETARY_COMPUTER:`/`API:` 等白名单前缀）同样先直接探测。此前它们
#     一律落目录语义匹配，被判 RC-L2_anchor_verification_gap。见 Benchmark-v3 §4.2.2。
RC_AXIS_VERSION = "rc_axis_v6_shared_checker"
RC_EVALUATION_FILENAME = "resource_check_evaluation.json"
DEFAULT_RESOURCE_CHECK_CONFIG = "config/debug_config.yaml"


def resolve_eval_config_path(config_path: str) -> Path:
    path = Path(config_path)
    return path if path.is_absolute() else _PROJECT_ROOT / path


def _load_config(config_path: str) -> Dict[str, Any]:
    load_dotenv(_PROJECT_ROOT / ".env", override=True)
    with open(config_path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    def expand(value):
        if isinstance(value, str):
            return re.sub(r"\$\{(\w+)\}", lambda match: os.environ[match.group(1)], value)
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        return value

    return expand(raw)

SYSTEM_CONFIG_PATHS: Dict[str, Path] = {
    "ours": _PROJECT_ROOT / "config" / "debug_config.yaml",
}

# 各系统配置文件里"资源检查通过次数"的历史键名。取到后统一折算成 agents.resources_check.count。
RC_PASS_COUNT_ALIASES: Tuple[Tuple[str, ...], ...] = (
    ("planned_boundary", "resource_check_passes_per_idea"),
    ("planning", "resource_check_passes_per_idea"),
    ("shared_planned_boundary", "resource_check", "max_passes_per_application"),
)

# 各系统声明的"不多轮、不 refinement/replan"开关，仅作产物溯源记录，不改判定。
RC_BOUNDARY_FLAG_SOURCES: Tuple[Tuple[str, ...], ...] = (
    ("planned_boundary", "enable_refinement"),
    ("planned_boundary", "enable_replan"),
    ("planning", "enable_refinement"),
    ("planning", "enable_replan"),
    ("shared_planned_boundary", "resource_check", "enable_refinement"),
    ("shared_planned_boundary", "resource_check", "enable_replan"),
)

AXIS_TOKENS = ("resource_check", "alignment")

SYSTEM_ROOTS: Dict[str, Path] = system_outputs_roots(BENCHMARK_DIR)

DATASET_TYPE_MAP = {
    "single_dataset": "single_dataset",
    "dual_dataset": "dual_dataset",
    "multi_dataset": "three_or_more_datasets",
    "three_or_more_datasets": "three_or_more_datasets",
}


# ──────────────────────────────────────────────
# 选择任务与系统
# ──────────────────────────────────────────────

def _parse_axes(raw: str) -> List[str]:
    if raw == "all":
        return list(AXIS_TOKENS)
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    invalid = [token for token in tokens if token not in AXIS_TOKENS]
    if invalid:
        raise ValueError(f"未知评估轴: {', '.join(invalid)}; 可选 {', '.join(AXIS_TOKENS)} 或 all")
    if not tokens:
        raise ValueError("--axes 不能为空")
    return tokens


def _parse_systems(raw: str) -> List[str]:
    if raw == "all":
        return list(SYSTEM_ROOTS)
    keys = [token.strip() for token in raw.split(",") if token.strip()]
    invalid = [key for key in keys if key not in SYSTEM_ROOTS]
    if invalid:
        raise ValueError(f"未知系统: {', '.join(invalid)}; 可选 {', '.join(SYSTEM_ROOTS)} 或 all")
    if not keys:
        raise ValueError("--systems 不能为空")
    return keys


def _select_task_ids(dataset_type: str, task_id: Optional[str]) -> List[str]:
    all_ids = canonical_task_ids(BENCHMARK_DIR)
    if task_id is not None:
        return [task_id] if task_id in all_ids else []
    if dataset_type == "all":
        return list(all_ids)
    normalized = DATASET_TYPE_MAP.get(dataset_type, dataset_type)
    return [tid for tid in all_ids if split_task_id(tid)[0] == normalized]


# ──────────────────────────────────────────────
# 资源检查轴
# ──────────────────────────────────────────────

def _load_output(output_path: Path) -> Dict[str, Any]:
    with open(output_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"full_output.json 顶层必须是对象: {output_path}")
    return payload


def _task_prompt_path(task_id: str) -> Path:
    group, local_id = split_task_id(task_id)
    prompt_path = TASK_INPUTS_DIR / group / local_id / "prompt.json"
    if not prompt_path.exists():
        raise FileNotFoundError(f"任务 prompt 不存在: {prompt_path}")
    return prompt_path


def _should_skip_rc(eval_path: Path, config_sha: str, *, task_id: str) -> Tuple[bool, str]:
    """资源检查轴产物是否可复用。口径（轴版本或 RC 配置指纹）变化一律重评。"""
    if not eval_path.exists():
        return False, "无既有产物"
    try:
        with open(eval_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return False, "既有产物不可解析"
    if not isinstance(payload, dict) or payload.get("evaluation_scope") != "resource_check":
        return False, "既有产物口径不匹配"
    protocol = payload.get("evaluation_protocol", {})
    if not isinstance(protocol, dict):
        return False, "既有产物缺少 protocol"
    accepted_versions = {RC_AXIS_VERSION}
    if task_id.startswith("single_dataset/"):
        accepted_versions.add("rc_axis_v2")
    if str(protocol.get("resource_check_axis_version", "")) not in accepted_versions:
        return False, "轴协议版本已过期"
    if str(protocol.get("resource_check_config_sha256", "")) != config_sha:
        return False, "resource check 配置指纹变化"
    return True, "已是当前口径"


# ──────────────────────────────────────────────
# 每个系统的资源检查器配置解析
# ──────────────────────────────────────────────

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并：override 覆盖 base；同键同为 dict 时继续下钻，其余整体替换。"""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _dig(payload: Any, path: Tuple[str, ...]) -> Any:
    node = payload
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _relative(path: Path) -> str:
    try:
        return path.relative_to(_PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _system_config_chain(system_key: str) -> List[Path]:
    if system_key not in SYSTEM_CONFIG_PATHS:
        raise ValueError(f"Unsupported system: {system_key}")
    path = SYSTEM_CONFIG_PATHS[system_key]
    if not path.is_file():
        raise FileNotFoundError(f"ResourceCheck configuration is missing: {path}")
    return [path]


def _chain_fingerprint(chain: List[Path]) -> str:
    """继承链的字节指纹：链上任一文件变化都会让既有产物失效并触发重评。"""
    digest = hashlib.sha256()
    for path in chain:
        digest.update(path.read_bytes())
        digest.update(b"\n---\n")
    return digest.hexdigest()


def resolve_system_resource_config(
    system_key: str, *, override_path: Optional[str] = None
) -> Dict[str, Any]:
    """解析单个系统实际使用的资源检查器配置。

    - 显式 override_path：整条链塌缩成那一份文件（等价于旧的全局 --resource-check-config）。
    - 否则读取 ours 的资源检查配置；文件缺失直接报错。

    返回 {config, chain, effective_path, fingerprint, source, boundary_flags}。
    """
    if system_key not in SYSTEM_CONFIG_PATHS:
        raise ValueError(f"Unsupported system: {system_key}")
    if override_path:
        chain = [resolve_eval_config_path(override_path)]
        source = "cli_override"
        own_payload: Dict[str, Any] = {}
    else:
        chain = _system_config_chain(system_key)
        if len(chain) > 1:
            source = "system_config"
            own_payload = _load_config(str(chain[-1]))
        else:
            source = "system_config"
            own_payload = {}

    merged: Dict[str, Any] = {}
    for path in chain:
        merged = _deep_merge(merged, _load_config(str(path)))

    # 使用同一份资源检查基座参数。
    base_checker = _dig(
        _load_config(str(resolve_eval_config_path(DEFAULT_RESOURCE_CHECK_CONFIG))),
        ("agents", "resources_check"),
    ) or {}
    agents = merged.setdefault("agents", {})
    checker = agents.get("resources_check")
    if not isinstance(checker, dict):
        checker = {}
        agents["resources_check"] = checker
    for key, value in base_checker.items():
        checker.setdefault(key, value)
    if not isinstance(checker.get("count"), int):
        checker["count"] = int(base_checker.get("count", 1))

    # 历史键名折算：只有本系统自己文件里的声明才算覆盖，基座里的不算。
    for aliased in RC_PASS_COUNT_ALIASES:
        declared_count = _dig(own_payload, aliased)
        if isinstance(declared_count, int) and declared_count > 0:
            checker["count"] = declared_count
            break

    boundary_flags = {
        ".".join(flag_path): _dig(own_payload, flag_path)
        for flag_path in RC_BOUNDARY_FLAG_SOURCES
        if _dig(own_payload, flag_path) is not None
    }

    return {
        "config": merged,
        "chain": chain,
        "effective_path": chain[-1],
        "fingerprint": _chain_fingerprint(chain),
        "source": source,
        "boundary_flags": boundary_flags,
    }


def _application_details(applications: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    details: List[Dict[str, Any]] = []
    for app in applications:
        if not isinstance(app, dict):
            continue
        anchor_meta = app.get("capability_anchor_meta")
        details.append(
            {
                "idea_id": app.get("idea_id", ""),
                "application_name": app.get("application_name", ""),
                "status": app.get("status", ""),
                "final_disposition": app.get("final_disposition", ""),
                "replan_round": app.get("replan_round", 0),
                "plan_dag_valid": (
                    anchor_meta.get("plan_dag_valid")
                    if isinstance(anchor_meta, dict)
                    else None
                ),
                "required_datasets": app.get("required_datasets", []),
                "resource_check": app.get("resource_check", {}),
                "resource_check_feedback": app.get("resource_check_feedback", {}),
            }
        )
    return details


# ──────────────────────────────────────────────
# 统一指标层：core（全系统可比）+ loop（仅循环系统）
# ──────────────────────────────────────────────

def _final_round_verdict(app: Dict[str, Any]) -> Dict[str, Any]:
    resource_check = app.get("resource_check") or {}
    contract = (
        resource_check.get("execution_contract", {})
        if isinstance(resource_check, dict)
        else {}
    )
    feedback = app.get("resource_check_feedback") or {}
    terminal_rule = str(feedback.get("terminal_rule") or contract.get("terminal_rule") or "")
    guaranteed = resource_check.get("guaranteed_datasets")
    return {
        "passed": contract.get("resource_check_passed") is True,
        "should_terminate": contract.get("should_terminate") is True,
        "terminal_rule": terminal_rule,
        "failure_class": str(feedback.get("failure_class", "")),
        "guaranteed_dataset_count": (
            len(guaranteed) if isinstance(guaranteed, list) else 0
        ),
        "tool_reachability_score": contract.get("tool_reachability_score"),
    }


def _first_round_verdict(
    app: Dict[str, Any], final: Dict[str, Any], *, single_pass: bool
) -> Dict[str, Any]:
    """解析首轮 ResourceCheck 判定。

    single_pass（baseline 单轮检查）→ 首轮即末轮；
    循环系统 → 从 plan_revision_history 首个 event=resource_check 的 metadata 恢复；
    无历史且 replan_round==0 → 首轮==末轮；
    其余无法恢复 → passed=None，不虚构。
    """
    if single_pass:
        return {**final, "source": "single_pass"}
    history = app.get("plan_revision_history")
    if isinstance(history, list):
        for entry in history:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("event") or "") != "resource_check":
                continue
            metadata = entry.get("metadata")
            if isinstance(metadata, dict) and "resource_check_passed" in metadata:
                feedback = entry.get("resource_check_feedback") or {}
                return {
                    "passed": metadata.get("resource_check_passed") is True,
                    "terminal_rule": str(feedback.get("terminal_rule") or ""),
                    "failure_class": str(metadata.get("failure_class") or ""),
                    "guaranteed_dataset_count": None,
                    "tool_reachability_score": metadata.get("tool_reachability_score"),
                    "source": "plan_revision_history",
                }
    if int(app.get("replan_round", 0) or 0) == 0:
        return {**final, "source": "no_replan_final_fallback"}
    return {
        "passed": None,
        "terminal_rule": "",
        "failure_class": "",
        "guaranteed_dataset_count": None,
        "tool_reachability_score": None,
        "source": "unrecoverable",
    }


def _safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _unified_metrics(
    applications: List[Dict[str, Any]], *, single_pass: bool
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """core=全系统同公式（首轮/末轮拆分）；loop=仅 refinement 循环系统，baseline 为 None。"""
    total = 0
    first_pass = first_terminal = first_non_terminal = first_unknown = 0
    final_pass = final_terminal = final_non_terminal = 0
    guaranteed_counts: List[float] = []
    reachability_scores: List[float] = []
    terminal_rule_counts: Dict[str, int] = {}
    per_app_rows: List[Dict[str, Any]] = []

    for app in applications:
        if not isinstance(app, dict):
            continue
        total += 1
        final = _final_round_verdict(app)
        first = _first_round_verdict(app, final, single_pass=single_pass)

        if final["passed"]:
            final_pass += 1
        elif final["should_terminate"] or (final["terminal_rule"] and not final["terminal_rule"].startswith("RC-L2_")):
            final_terminal += 1
        else:
            final_non_terminal += 1
        rule_key = final["terminal_rule"] or "none"
        terminal_rule_counts[rule_key] = terminal_rule_counts.get(rule_key, 0) + 1
        guaranteed_counts.append(float(final["guaranteed_dataset_count"]))
        if isinstance(final["tool_reachability_score"], (int, float)):
            reachability_scores.append(float(final["tool_reachability_score"]))

        if first["passed"] is None:
            first_unknown += 1
        elif first["passed"]:
            first_pass += 1
        elif (first["terminal_rule"] or first["failure_class"]) and not first["terminal_rule"].startswith("RC-L2_"):
            first_terminal += 1
        else:
            first_non_terminal += 1

        per_app_rows.append(
            {
                "idea_id": app.get("idea_id", ""),
                "first_rc_passed": first["passed"],
                "first_rc_source": first["source"],
                "first_rc_terminal_rule": first["terminal_rule"],
                "first_rc_failure_class": first["failure_class"],
                "final_rc_passed": final["passed"],
                "final_rc_should_terminate": final["should_terminate"],
                "final_rc_terminal_rule": final["terminal_rule"],
                "final_rc_failure_class": final["failure_class"],
                "guaranteed_dataset_count": final["guaranteed_dataset_count"],
                "tool_reachability_score": final["tool_reachability_score"],
                "replan_round": int(app.get("replan_round", 0) or 0),
                "status": str(app.get("status", "")),
            }
        )

    core: Dict[str, Any] = {
        "total_applications": total,
        "first_pass_count": first_pass,
        "first_pass_rate": round(first_pass / total, 4) if total else None,
        "first_terminal_failure_count": first_terminal,
        "first_non_terminal_failure_count": first_non_terminal,
        "first_unrecoverable_count": first_unknown,
        "final_pass_count": final_pass,
        "final_pass_rate": round(final_pass / total, 4) if total else None,
        "final_terminal_failure_count": final_terminal,
        "final_non_terminal_failure_count": final_non_terminal,
        "avg_guaranteed_dataset_count": _safe_mean(guaranteed_counts),
        "avg_tool_reachability_score": _safe_mean(reachability_scores),
        "final_terminal_rule_counts": terminal_rule_counts,
    }

    loop: Optional[Dict[str, Any]] = None
    if not single_pass:
        ready_count = sum(
            row["status"] == "ready_for_evaluation" for row in per_app_rows
        )
        failed_count = sum(
            row["status"] == "failed_before_evaluation" for row in per_app_rows
        )
        replan_rounds = [float(row["replan_round"]) for row in per_app_rows]
        replan_triggered = [round_ > 0 for round_ in replan_rounds]
        max_hits = sum(
            row["final_rc_terminal_rule"] in {
                "T7_resource_replan_limit_exceeded",
                "T7_quality_replan_limit_exceeded",
            }
            for row in per_app_rows
        )
        loop = {
            "survival_count": ready_count,
            "final_app_survival_rate": round(ready_count / total, 4) if total else None,
            "failed_before_evaluation_count": failed_count,
            "failed_before_evaluation_rate": round(failed_count / total, 4) if total else None,
            "replan_trigger_count": sum(replan_triggered),
            "replan_trigger_rate": round(sum(replan_triggered) / total, 4) if total else None,
            "mean_replan_rounds": _safe_mean(replan_rounds),
            "max_replan_rounds_hit_count": max_hits,
            "max_replan_rounds_hit_rate": round(max_hits / total, 4) if total else None,
        }

    return core, loop, per_app_rows


async def _evaluate_rc_task(
    *,
    system_key: str,
    task_id: str,
    output_path: Path,
    rc_resolution: Dict[str, Any],
    overwrite: bool,
    dry_run: bool,
) -> str:
    """执行单个任务的资源检查轴。返回 status: evaluated / skipped / failed。"""
    resource_config = rc_resolution["config"]
    config_sha = rc_resolution["fingerprint"]
    eval_path = output_path.parent / RC_EVALUATION_FILENAME
    if not overwrite:
        skip, reason = _should_skip_rc(eval_path, config_sha, task_id=task_id)
        if skip:
            logger.info(f"[{system_key}/{task_id}] 资源检查轴跳过（{reason}）")
            return "skipped"

    output_data = _load_output(output_path)
    boundary = str(output_data.get("boundary") or "").strip()
    family = str(output_data.get("family") or "").strip()
    if boundary not in {"planned", "resource_check", "refinement_terminal"}:
        raise ValueError(f"不支持的 boundary={boundary!r}: {output_path}")

    evaluator_owned = family != "internagent"
    stage = "evaluator_owned_resource_check" if evaluator_owned else "generation_owned_resource_check"

    if dry_run:
        logger.info(
            f"[{system_key}/{task_id}] dry-run: {stage}; "
            f"rc_config={_relative(rc_resolution['effective_path'])} ({rc_resolution['source']})"
        )
        return "dry_run"

    task_prompt_path = _task_prompt_path(task_id)
    with open(task_prompt_path, encoding="utf-8") as handle:
        raw_task_prompt = json.load(handle)
    task_prompt = raw_task_prompt

    applications = output_data.get("applications", [])
    if not isinstance(applications, list):
        raise ValueError(f"applications 必须是列表: {output_path}")

    if evaluator_owned:
        checker = BenchmarkResourceChecker(resource_config)
        applications, audit = await checker.evaluate(applications, task_prompt)
    else:
        # ours：生成侧已跑过 ResourceCheck，评测侧只做汇总，不重复执行。
        audit = {
            "execution_owner": "generation",
            "application_audit": [],
            "executed_count": 0,
        }

    # 统一指标层：core 全系统同公式（首轮/末轮拆分）；loop 仅循环系统，baseline 为 None。
    core, loop, per_app = _unified_metrics(applications, single_pass=evaluator_owned)
    metrics = {
        "core": core,
        "loop": loop,
        "loop_applicable": loop is not None,
        "stage_boundary": boundary,
    }

    payload = {
        "system": output_data.get("system", ""),
        "family": family,
        "matrix": output_data.get("matrix", "benchmark"),
        "boundary": boundary,
        "task": output_data.get("task", ""),
        "task_id": task_id,
        "evaluation_scope": "resource_check",
        "axis": "feasibility_resource_check",
        "evaluation_protocol": {
            "resource_check_axis_version": RC_AXIS_VERSION,
            "resource_check_config_path": _relative(rc_resolution["effective_path"]),
            "resource_check_config_chain": [
                _relative(path) for path in rc_resolution["chain"]
            ],
            "resource_check_config_source": rc_resolution["source"],
            "resource_check_config_sha256": config_sha,
            "resource_check_boundary_flags": rc_resolution["boundary_flags"],
            "resource_check_execution": audit,
            "source_output_path": _relative(output_path),
            "task_prompt_path": _relative(task_prompt_path),
        },
        "metrics": metrics,
        "per_application": per_app,
        "applications_detail": _application_details(applications),
    }
    with open(eval_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    logger.info(f"[{system_key}/{task_id}] 资源检查轴完成: {eval_path} ({stage})")
    return "evaluated"


async def run_resource_check_axis(
    *,
    systems: List[str],
    task_ids: List[str],
    override_config: Optional[str],
    overwrite: bool,
    dry_run: bool,
    limit: Optional[int],
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, Any]]]:
    stats: Dict[str, Dict[str, int]] = {}
    resolutions: Dict[str, Dict[str, Any]] = {}
    processed = 0
    for system_key in systems:
        root = SYSTEM_ROOTS[system_key]
        rc_resolution = resolve_system_resource_config(
            system_key, override_path=override_config
        )
        resolutions[system_key] = {
            "config_path": _relative(rc_resolution["effective_path"]),
            "chain": [_relative(path) for path in rc_resolution["chain"]],
            "source": rc_resolution["source"],
            "sha256": rc_resolution["fingerprint"][:12],
            "count": _dig(rc_resolution["config"], ("agents", "resources_check", "count")),
            "boundary_flags": rc_resolution["boundary_flags"],
        }
        logger.info(
            f"[{system_key}] 资源检查器配置={resolutions[system_key]['config_path']} "
            f"source={resolutions[system_key]['source']} "
            f"count={resolutions[system_key]['count']} "
            f"sha256={resolutions[system_key]['sha256']}…"
        )
        counts = {"evaluated": 0, "skipped": 0, "failed": 0, "missing": 0, "dry_run": 0}
        for task_id in task_ids:
            output_path = grouped_task_dir(root, task_id) / "full_output.json"
            if not output_path.exists():
                counts["missing"] += 1
                continue
            try:
                status = await _evaluate_rc_task(
                    system_key=system_key,
                    task_id=task_id,
                    output_path=output_path,
                    rc_resolution=rc_resolution,
                    overwrite=overwrite,
                    dry_run=dry_run,
                )
                counts[status] = counts.get(status, 0) + 1
            except Exception as exc:
                logger.error(f"[{system_key}/{task_id}] 资源检查轴失败: {exc}", exc_info=True)
                counts["failed"] += 1
            processed += 1
            if limit is not None and processed >= limit and not dry_run:
                logger.info(f"已达到 --limit {limit}，提前停止资源检查轴")
                stats[system_key] = counts
                return stats, resolutions
        stats[system_key] = counts
    return stats, resolutions


# ──────────────────────────────────────────────
# Alignment×Depth 轴（子进程编排 3b）
# ──────────────────────────────────────────────

def _subprocess_commands(
    axes: List[str], task_id: Optional[str], retrieval_config: Optional[str],
    systems: List[str], dataset_type: str, limit: Optional[int], overwrite: bool,
) -> List[Tuple[str, List[str]]]:
    commands = []
    if "alignment" in axes and retrieval_config:
        for system_key in systems:
            cmd = [
                sys.executable, str(SCRIPT_3B), "--retrieval-config", retrieval_config,
                "--system-outputs-dir", str(SYSTEM_ROOTS[system_key]),
                "--dataset-type", dataset_type,
            ]
            if task_id:
                cmd += ["--task-id", task_id]
            if limit is not None:
                cmd += ["--limit", str(limit)]
            if overwrite:
                cmd += ["--force"]
            commands.append((f"alignment_depth(3b)/{system_key}", cmd))
    summary_cmd = [sys.executable, str(BENCHMARK_DIR / "scripts" / "6_three_axis_summary.py"),
                   "--systems", ",".join(systems), "--dataset-type", dataset_type]
    if task_id:
        summary_cmd += ["--task-id", task_id]
    commands.append(("independent_axes_summary(6)", summary_cmd))
    return commands


def run_remaining_axes(
    axes: List[str], task_id: Optional[str], retrieval_config: Optional[str], dry_run: bool,
    systems: List[str], dataset_type: str, limit: Optional[int], overwrite: bool,
) -> int:
    commands = _subprocess_commands(axes, task_id, retrieval_config, systems, dataset_type, limit, overwrite)
    failed = 0
    for label, cmd in commands:
        printable = " ".join(str(part) for part in cmd)
        if dry_run:
            logger.info(f"[dry-run] {label} 将执行: {printable}")
            continue
        logger.info(f"[{label}] 启动子进程: {printable}")
        result = subprocess.run(cmd, cwd=str(_PROJECT_ROOT))
        if result.returncode != 0:
            logger.error(f"[{label}] 子进程失败，退出码 {result.returncode}")
            failed += 1
    return failed


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark 两评估轴统一入口（资源检查 / Alignment×Depth）"
    )
    parser.add_argument(
        "--axes",
        default="all",
        help="all 或逗号分隔的 resource_check,alignment（默认 all）",
    )
    parser.add_argument(
        "--systems",
        default="all",
        help=(
            "all 或逗号分隔的 "
            "ours"
            "（默认 all；作用于全部评估轴）"
        ),
    )
    parser.add_argument(
        "--dataset-type",
        default="all",
        choices=["all", "single_dataset", "dual_dataset", "multi_dataset", "three_or_more_datasets"],
        help="所有评估轴的分组过滤（默认 all）",
    )
    parser.add_argument("--task-id", default=None, help="单个规范任务 ID，如 single_dataset/task_001")
    parser.add_argument("--limit", type=int, default=None, help="每系统 Alignment/Depth 最多任务数；RC 保留总任务限流")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="强制重评 RC 和 Alignment/Depth",
    )
    parser.add_argument(
        "--resource-check-config",
        default=None,
        help=(
            "显式覆盖 ours 的 ResourceCheck 配置（默认使用 "
            "config/debug_config.yaml；"
            f"给本参数即退回旧的全局单配置口径）"
        ),
    )
    parser.add_argument(
        "--retrieval-config",
        default=str(_PROJECT_ROOT / "config" / "debug_config.yaml"),
        help="Alignment 检索配置；默认读取 config/debug_config.yaml 的 semantic_retrieval 节",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不执行")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    axes = _parse_axes(args.axes)
    systems = _parse_systems(args.systems)
    task_ids = _select_task_ids(args.dataset_type, args.task_id)
    if args.task_id and not task_ids:
        raise SystemExit(f"任务不存在: {args.task_id}")
    if "alignment" in axes and not args.retrieval_config:
        message = (
            "Alignment 轴需要有效的检索配置；默认读取 config/debug_config.yaml"
        )
        if args.dry_run:
            logger.warning(f"[dry-run] {message}；alignment 轴将被跳过")
        else:
            # fail fast：避免先跑完资源检查/质量轴后在 alignment 处中断。
            raise SystemExit(message)

    print("=" * 72)
    print(f"评估轴   : {', '.join(axes)}")
    print(f"系统     : {', '.join(systems)}")
    print(f"任务数   : {len(task_ids)}（资源检查轴）")
    print(
        "RC 配置  : "
        + (
            f"全局覆盖 {args.resource_check_config}"
            if args.resource_check_config
            else "读取 ours 资源检查配置"
        )
    )
    print("=" * 72)

    failed_total = 0

    if "resource_check" in axes:
        stats, resolutions = asyncio.run(
            run_resource_check_axis(
                systems=systems,
                task_ids=task_ids,
                override_config=args.resource_check_config,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                limit=args.limit,
            )
        )
        print("-" * 72)
        print("资源检查器配置来源:")
        for system_key in systems:
            info = resolutions.get(system_key)
            if not info:
                continue
            print(
                f"  {system_key}: {info['config_path']} "
                f"(source={info['source']}, count={info['count']}, "
                f"sha256={info['sha256']}…)"
            )
            if len(info["chain"]) > 1:
                print(f"      chain: {' -> '.join(info['chain'])}")
            if info["boundary_flags"]:
                print(f"      boundary_flags: {info['boundary_flags']}")
        print("-" * 72)
        print("资源检查轴汇总:")
        for system_key, counts in stats.items():
            parts = ", ".join(f"{key}={value}" for key, value in counts.items() if value)
            print(f"  {system_key}: {parts or 'no_output'}")
            failed_total += counts.get("failed", 0)

    remaining_axes = [axis for axis in axes if axis != "resource_check"]
    sub_failed = run_remaining_axes(remaining_axes, args.task_id, args.retrieval_config, args.dry_run,
                                    systems, args.dataset_type, args.limit, args.overwrite)
    failed_total += sub_failed

    print("=" * 72)
    if args.dry_run:
        print("dry-run 结束，未执行任何评估")
    else:
        print(f"完成。失败任务/子进程数: {failed_total}")
    raise SystemExit(1 if failed_total else 0)


if __name__ == "__main__":
    main()

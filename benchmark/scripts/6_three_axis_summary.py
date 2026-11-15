#!/usr/bin/env python3
"""Read-only independent Alignment, Depth and resource-check statistics."""

import argparse
import hashlib
import math
import statistics
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from benchmark.core.depth_evaluation import DEPTH_FINGERPRINT, RUBRIC_VERSION, SCHEMA_VERSION, PLAN_PROJECTION_VERSION
from benchmark.core.benchmark_layout import (
    canonical_task_ids,
    grouped_task_dir,
    split_task_id,
    system_outputs_roots,
)

logger = logging.getLogger("three_axis_summary")

BENCHMARK_DIR = _PROJECT_ROOT / "benchmark"
ALIGNMENT_EVALUATION_FILENAME = "alignment_depth_evaluation.json"
DEFAULT_OUTPUT_DIR = BENCHMARK_DIR / "reports" / "three_axis_summary"

# 资源检查轴当前口径，必须与 5_axis_evaluation.RC_AXIS_VERSION 一致，否则所有新产物
# 都会被本脚本判成 stale、汇总表整列显示 0。单一来源是 5_axis_evaluation.py。
RC_AXIS_VERSION = "rc_axis_v5_platform_probe"
# 缓存失效豁免：组合任务（dual / three_or_more）只接受新协议，single_dataset 仍可复用
# 旧 rc_axis_v2 产物（见 Benchmark-v3.md §4.2.2）。5_axis_evaluation._should_skip_rc
# 用同一套豁免规则，两处口径必须同步。
RC_AXIS_LEGACY_SINGLE_DATASET_VERSION = "rc_axis_v2"
RC_EVALUATION_FILENAME = "resource_check_evaluation.json"

# Depth 1-10 protocol; legacy scores must never be mixed into this report.
ALIGNMENT_SCHEMA_VERSION = SCHEMA_VERSION

SYSTEM_ROOTS: Dict[str, Path] = system_outputs_roots(BENCHMARK_DIR)

# Every selected system has its own Alignment/Depth result root.
ALIGNMENT_SYSTEMS = tuple(SYSTEM_ROOTS)

PORTFOLIO_VIEWS = ("best_of_portfolio", "median_of_portfolio", "full_distribution")

DATASET_TYPE_MAP = {
    "single_dataset": "single_dataset",
    "dual_dataset": "dual_dataset",
    "multi_dataset": "three_or_more_datasets",
    "three_or_more_datasets": "three_or_more_datasets",
}


# ──────────────────────────────────────────────
# 轴数据读取
# ──────────────────────────────────────────────

def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        logger.warning(f"无法解析 {path}: {exc}")
        return None
    return payload if isinstance(payload, dict) else None


def _load_rc(system_key: str, task_id: str) -> Dict[str, Any]:
    """读可行性轴产物。status: evaluated / stale / missing / no_output。"""
    output_root = SYSTEM_ROOTS[system_key]
    full_output = grouped_task_dir(output_root, task_id) / "full_output.json"
    if not full_output.exists():
        return {"status": "no_output"}
    rc_path = grouped_task_dir(output_root, task_id) / RC_EVALUATION_FILENAME
    payload = _read_json(rc_path) if rc_path.exists() else None
    if payload is None:
        return {"status": "missing"}
    version = str(
        (payload.get("evaluation_protocol") or {}).get("resource_check_axis_version", "")
    )
    accepted_versions = {RC_AXIS_VERSION}
    if task_id.startswith("single_dataset/"):
        accepted_versions.add(RC_AXIS_LEGACY_SINGLE_DATASET_VERSION)
    if version not in accepted_versions:
        return {"status": "stale", "axis_version": version}
    core = ((payload.get("metrics") or {}).get("core")) or {}
    if not core:
        return {"status": "missing"}
    idea_final_pass: Dict[str, bool] = {}
    for row in payload.get("per_application") or []:
        if not isinstance(row, dict):
            continue
        idea_id = row.get("idea_id")
        passed = row.get("final_rc_passed")
        if isinstance(idea_id, str) and idea_id and isinstance(passed, bool):
            idea_final_pass[idea_id] = passed
    return {
        "status": "evaluated",
        "axis_version": version,
        "final_pass_rate": core.get("final_pass_rate"),
        "final_pass_count": core.get("final_pass_count"),
        "total_applications": core.get("total_applications"),
        "final_terminal_rule_counts": core.get("final_terminal_rule_counts"),
        "idea_final_pass": idea_final_pass,
    }


def _load_alignment(task_id: str, system_key: str = "ours") -> Dict[str, Any]:
    root = grouped_task_dir(SYSTEM_ROOTS[system_key], task_id)
    hits = []
    for result_path in [root / ALIGNMENT_EVALUATION_FILENAME]:
        payload = _read_json(result_path) if result_path.is_file() else None
        if payload is None:
            continue
        retrieval_version = payload.get("frozen_retrieval_config", {}).get("alignment_model_version")
        protocol = payload.get("depth_protocol", {})
        if (payload.get("schema_version") != SCHEMA_VERSION
                or payload.get("candidate_identity_version") != "canonical-products-v3-coverage"
                or payload.get("plan_projection_version") != PLAN_PROJECTION_VERSION
                or protocol.get("fingerprint") != DEPTH_FINGERPRINT):
            hits.append({"status": "stale", "retrieval_version": retrieval_version})
            continue
        source = grouped_task_dir(SYSTEM_ROOTS[system_key], task_id) / "full_output.json"
        prompt = grouped_task_dir(BENCHMARK_DIR / "dataset" / "task_inputs", task_id) / "prompt.json"
        provenance = payload.get("provenance", {})
        if (not source.is_file() or not prompt.is_file()
                or provenance.get("full_output_sha256") != hashlib.sha256(source.read_bytes()).hexdigest()
                or provenance.get("prompt_sha256") != hashlib.sha256(prompt.read_bytes()).hexdigest()):
            hits.append({"status": "stale", "retrieval_version": retrieval_version})
            continue
        ideas = []
        for raw in payload.get("ideas", []):
            depth = raw.get("depth", {})
            score = depth.get("depth_score")
            valid = (depth.get("rubric_version") == RUBRIC_VERSION
                     and depth.get("rubric_fingerprint") == DEPTH_FINGERPRINT
                     and depth.get("review_status") == "scored"
                     and type(score) is int and 1 <= score <= 10)
            ideas.append({
                "idea_id": raw.get("idea_id"),
                "application_index": raw.get("application_index"),
                "max_alignment": raw.get("max_alignment"),
                "top3_alignment_mean": raw.get("top3_alignment_mean"),
                "existing_idea_hit": raw.get("existing_idea_hit"),
                "mapping_status": raw.get("mapping_status"),
                "depth_status": "evaluated" if valid else (
                    "insufficient_material" if depth.get("review_status") == "insufficient_material" else "missing"),
                "depth_score": score if valid else None,
            })
        hits.append({
            "status": "evaluated", "retrieval_version": retrieval_version,
            "alignment_threshold": payload.get("frozen_retrieval_config", {}).get("alignment_threshold"),
            "admitted_idea_count": len(ideas), "ideas": ideas,
            "modified": result_path.stat().st_mtime_ns,
        })
    current = [hit for hit in hits if hit["status"] == "evaluated"]
    if current:
        return max(current, key=lambda hit: hit["modified"])
    return {"status": "stale" if hits else "missing"}


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _mean(values: List[float]) -> Optional[float]:
    return round(statistics.mean(values), 4) if values else None


def _distribution(values: List[float]) -> Dict[str, Any]:
    return {
        "count": len(values), "mean": _mean(values),
        "median": statistics.median(values) if values else None,
        "min": min(values) if values else None, "max": max(values) if values else None,
        "values": values,
    }


def select_portfolio_view(ideas: List[Dict[str, Any]], view: str) -> List[Dict[str, Any]]:
    if view == "full_distribution":
        return list(ideas)
    ordered = sorted(ideas, key=lambda idea: (
        idea["max_alignment"] if _number(idea.get("max_alignment")) else -1,
        -(idea.get("application_index") if type(idea.get("application_index")) is int else 0),
    ))
    if not ordered:
        return []
    if view == "best_of_portfolio":
        return [ordered[-1]]
    if view == "median_of_portfolio":
        return [ordered[(len(ordered) - 1) // 2]]
    raise ValueError(f"Unknown portfolio view: {view}")


def _task_axes(alignment: Dict[str, Any], rc: Dict[str, Any]) -> Dict[str, Any]:
    ideas = alignment.get("ideas", []) if alignment.get("status") == "evaluated" else []
    linked = rc.get("idea_final_pass", {}) if rc.get("status") == "evaluated" else {}
    records = [{**idea, "final_rc_passed": linked.get(idea.get("idea_id"))} for idea in ideas]
    align_values = [i["max_alignment"] for i in records if _number(i.get("max_alignment"))]
    depth_values = [i["depth_score"] for i in records if i.get("depth_status") == "evaluated"]
    rc_values = [int(i["final_rc_passed"]) for i in records if isinstance(i.get("final_rc_passed"), bool)]
    common = [i for i in records if _number(i.get("max_alignment"))
              and i.get("depth_status") == "evaluated" and isinstance(i.get("final_rc_passed"), bool)]
    threshold = alignment.get("alignment_threshold")
    hit_count = sum(i.get("existing_idea_hit") is True for i in records)
    aligned_count = sum(_number(i.get("max_alignment")) and _number(threshold)
                        and i["max_alignment"] >= threshold for i in records)
    return {
        "admitted_idea_count": len(records), "ideas": records,
        "alignment": {
            **_distribution(align_values),
            "unscored_count": len(records) - len(align_values),
            "alignment_hit_count": aligned_count,
            "alignment_hit_denominator": len(records) if _number(threshold) else 0,
            "alignment_hit_rate": round(aligned_count / len(records), 4) if records and _number(threshold) else None,
            "existing_idea_hit_count": hit_count,
            "existing_idea_hit_denominator": len(records),
            "existing_idea_hit_rate": round(hit_count / len(records), 4) if records else None,
        },
        "depth": {
            **_distribution(depth_values),
            "histogram": {str(n): depth_values.count(n) for n in range(1, 11)},
            "insufficient_material_count": sum(i.get("depth_status") == "insufficient_material" for i in records),
            "missing_count": sum(i.get("depth_status") == "missing" for i in records),
        },
        "resource_check": {
            **_distribution(rc_values), "passed_count": sum(rc_values),
            "unlinked_count": len(records) - len(rc_values),
            "full_rc_population": {
                "count": rc.get("total_applications"), "passed_count": rc.get("final_pass_count"),
                "pass_rate": rc.get("final_pass_rate"), "status": rc.get("status"),
            },
        },
        "common_cohort_coordinates": {
            "count": len(common), "idea_ids": [i["idea_id"] for i in common],
            "alignment_mean": _mean([i["max_alignment"] for i in common]),
            "depth_mean": _mean([i["depth_score"] for i in common]),
            "resource_pass_rate": _mean([int(i["final_rc_passed"]) for i in common]),
        },
        "portfolio_views": {
            view: select_portfolio_view(records, view)
            for view in PORTFOLIO_VIEWS
        },
    }


def build_summary(task_ids: List[str], systems: List[str]) -> Dict[str, Any]:
    rows = []
    for task_id in task_ids:
        row = {"task_id": task_id, "feasibility": {}, "alignment": {}, "independent_axes": {}}
        for system_key in systems:
            rc = _load_rc(system_key, task_id)
            alignment = _load_alignment(task_id, system_key)
            row["feasibility"][system_key] = rc
            row["alignment"][system_key] = alignment
            row["independent_axes"][system_key] = _task_axes(alignment, rc)
        rows.append(row)
    system_summary = {}
    coverage = {"total_tasks": len(task_ids), "systems": {}}
    for system_key in systems:
        cells = [row["independent_axes"][system_key] for row in rows]
        aggregates = {}
        for axis in ("alignment", "depth", "resource_check"):
            values = [value for cell in cells for value in cell[axis]["values"]]
            task_means = [cell[axis]["mean"] for cell in cells if cell[axis]["mean"] is not None]
            aggregates[axis] = {
                "idea_distribution": _distribution(values),
                "task_macro": _distribution(task_means),
                "pending_tasks": len(task_ids) - len(task_means),
            }
            if axis == "depth":
                aggregates[axis]["histogram"] = {str(n): values.count(n) for n in range(1, 11)}
        full_rc = [row["feasibility"][system_key].get("final_pass_rate") for row in rows
                   if row["feasibility"][system_key]["status"] == "evaluated"]
        aggregates["resource_check"]["full_rc_task_macro"] = _distribution([v for v in full_rc if _number(v)])
        aggregates["common_cohort_task_count"] = sum(cell["common_cohort_coordinates"]["count"] > 0 for cell in cells)
        system_summary[system_key] = aggregates
        coverage["systems"][system_key] = {
            axis: {state: sum(row[axis][system_key]["status"] == state for row in rows)
                   for state in ("evaluated", "stale", "missing", "no_output")}
            for axis in ("feasibility", "alignment")
        }
    return {
        "report_schema": "three_independent_axes_v2",
        "depth_rubric_version": RUBRIC_VERSION, "depth_fingerprint": DEPTH_FINGERPRINT,
        "comparison_rules": {
            "no_composite_score": True,
            "alignment": "Per-idea Max-over-Set; null scores excluded from score means, retained in coverage/hit denominators.",
            "depth": "Holistic ordinal 1-10; means descriptive, histogram retained; no depth_high threshold.",
            "resource_check": "Final RC pass flag, not scientific infeasibility; linked cohort and full RC population reported separately.",
            "coordinates": "All three coordinates use the same scored and RC-linked idea cohort.",
            "macro": "Each axis is averaged within task, then equally across tasks with that axis present; counts reported.",
        },
        "system_summary": system_summary, "axis_coverage": coverage, "rows": rows,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def render_markdown(summary: Dict[str, Any], systems: List[str]) -> str:
    lines = ["# Three Independent Axes", "", "No composite score. Depth: 1-10; Alignment: 1-10; RC: final pass rate.",
             "", "## System Task Macro Means", "",
             "| System | Alignment | Tasks | Depth | Tasks | Linked RC | Tasks | Full RC | Tasks |",
             "|---|---|---|---|---|---|---|---|---|"]
    for key in systems:
        stats = summary["system_summary"][key]
        a, d, r = [stats[axis]["task_macro"] for axis in ("alignment", "depth", "resource_check")]
        full_rc = stats["resource_check"]["full_rc_task_macro"]
        lines.append(f"| {key} | {_fmt(a['mean'])} | {a['count']} | {_fmt(d['mean'])} | {d['count']} | {_fmt(r['mean'])} | {r['count']} | {_fmt(full_rc['mean'])} | {full_rc['count']} |")
    lines += ["", "## Task Common Cohort Coordinates", "",
              "| Task | System | Common ideas | Alignment | Depth | RC |",
              "|---|---|---|---|---|---|"]
    for row in summary["rows"]:
        for key in systems:
            c = row["independent_axes"][key]["common_cohort_coordinates"]
            lines.append(f"| {row['task_id']} | {key} | {c['count']} | {_fmt(c['alignment_mean'])} | {_fmt(c['depth_mean'])} | {_fmt(c['resource_pass_rate'])} |")
    lines += ["", "## Depth Histogram", "", "| System | " + " | ".join(map(str, range(1, 11))) + " |",
              "|---|" + "---|" * 10]
    for key in systems:
        hist = summary["system_summary"][key]["depth"]["histogram"]
        lines.append(f"| {key} | " + " | ".join(str(hist[str(n)]) for n in range(1, 11)) + " |")
    lines += ["", "## Coverage", "", "| System | Axis | Evaluated | Stale | Missing | No output |",
              "|---|---|---|---|---|---|"]
    for key, axes in summary["axis_coverage"]["systems"].items():
        for axis, states in axes.items():
            lines.append(f"| {key} | {axis} | {states['evaluated']} | {states['stale']} | {states['missing']} | {states['no_output']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="三轴独立统计（只读聚合，不执行评估）")
    parser.add_argument(
        "--systems",
        default="all",
        help="ours 或 all（当前仅汇总 ours）",
    )
    parser.add_argument(
        "--dataset-type",
        default="all",
        choices=["all", "single_dataset", "dual_dataset", "multi_dataset", "three_or_more_datasets"],
    )
    parser.add_argument("--task-id", default=None, help="单个规范任务 ID，如 single_dataset/task_001")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.systems == "all":
        systems = list(SYSTEM_ROOTS)
    else:
        systems = [key.strip() for key in args.systems.split(",") if key.strip()]
        invalid = [key for key in systems if key not in SYSTEM_ROOTS]
        if invalid:
            raise SystemExit(f"未知系统: {', '.join(invalid)}")

    all_ids = canonical_task_ids(BENCHMARK_DIR)
    if args.task_id is not None:
        task_ids = [args.task_id] if args.task_id in all_ids else []
        if not task_ids:
            raise SystemExit(f"任务不存在: {args.task_id}")
    elif args.dataset_type == "all":
        task_ids = list(all_ids)
    else:
        normalized = DATASET_TYPE_MAP.get(args.dataset_type, args.dataset_type)
        task_ids = [tid for tid in all_ids if split_task_id(tid)[0] == normalized]

    summary = build_summary(task_ids, systems)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "three_axis_summary.json"
    md_path = args.output_dir / "three_axis_summary.md"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    markdown = render_markdown(summary, systems)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(markdown)

    logger.info(f"汇总完成: {json_path}")
    logger.info(f"Markdown 报告: {md_path}")
    print(markdown)


if __name__ == "__main__":
    main()

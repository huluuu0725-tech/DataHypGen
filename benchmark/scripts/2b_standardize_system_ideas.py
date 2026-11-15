"""
2b_standardize_system_ideas.py - 从DataGenHyp输出中提取和标准化最佳System Idea

功能：
  1. 读取 system_outputs/<group>/task_XXX/full_output.json
  2. 从applications列表中选择最佳application（基于idea_evaluation分数）
  3. 将application标准化为与Paper Idea相同的8字段格式
  4. 保存到 system_outputs/<group>/task_XXX/best_idea.json

字段映射：
  application.idea_candidate.research_question      -> research_question
  application.idea_candidate.theoretical_mechanism  -> mechanism + hypothesis
  application.idea_candidate.novelty_constraints_and_value -> expected_contribution
  application.required_datasets                     -> data_sources
  application.task_plan                             -> analytical_approach

用法:
  py benchmark/scripts/2b_standardize_system_ideas.py
  py benchmark/scripts/2b_standardize_system_ideas.py --task-id single_dataset/task_001
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from benchmark.core.admission import (
    eligible_applications, load_admitted_idea, project_idea_fields, source_digest,
)
from benchmark.core.benchmark_layout import (
    canonical_task_ids,
    grouped_json_path,
    grouped_task_dir,
    require_canonical_task_id,
    system_outputs_root,
)

logger = logging.getLogger(__name__)


def _pick_best_application(applications: List[Dict]) -> Optional[Tuple[Dict, float]]:
    """
    从applications列表中选择最佳application。
    优先使用 idea_evaluation.scores 的综合分，其次使用 priority 字段。
    返回 (application, score) 或 None。
    """
    scored = []
    for app in applications:
        if not isinstance(app, dict):
            continue

        # 跳过明确失败的
        status = app.get("status", "")
        final  = app.get("final_disposition", "")
        if final in ("hard_conflict", "rc_failed") or status == "failed":
            continue

        # 尝试从idea_evaluation中取分
        eval_data = app.get("idea_evaluation") or {}
        score = None

        # 结构1: idea_evaluation.scores.overall
        scores = eval_data.get("scores") or {}
        if isinstance(scores, dict):
            score = scores.get("overall") or scores.get("total")

        # 结构2: idea_evaluation.overall_score
        if score is None:
            score = eval_data.get("overall_score") or eval_data.get("score")

        # 退化：使用priority（越小越优先）转换为分数
        if score is None:
            priority = app.get("priority", 999)
            score = 1000 - priority

        scored.append((app, float(score)))

    if not scored:
        return None

    # 取分最高的
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0]


def _extract_idea_fields(app: Dict) -> Dict:
    """8 字段投影的唯一实现在 admission.project_idea_fields；3b 评估全部 admitted 时复用同一函数。"""
    return project_idea_fields(app)


def standardize_single_task(task_id: str, system_outputs_dir: Path) -> bool:
    """处理单个任务，返回是否成功。"""
    task_dir    = grouped_task_dir(system_outputs_dir, task_id)
    output_file = task_dir / "full_output.json"
    best_file   = task_dir / "best_idea.json"

    if not output_file.exists():
        logger.warning(f"{task_id}: full_output.json 不存在，跳过")
        return False

    if best_file.exists():
        try:
            load_admitted_idea(best_file)
        except (ValueError, OSError) as exc:
            logger.error(f"{task_id}: 旧标准化产物不可复用，请归档后重新标准化: {exc}")
            return False
        logger.info(f"{task_id}: 已有经过准入校验的 best_idea.json，跳过")
        return True

    with open(output_file, encoding="utf-8") as f:
        full_output = json.load(f)

    try:
        applications = eligible_applications(full_output)
    except ValueError as exc:
        logger.error(f"{task_id}: 不满足 benchmark 准入要求: {exc}")
        return False
    if not applications:
        logger.warning(f"{task_id}: applications列表为空")
        return False

    result = _pick_best_application(applications)
    if result is None:
        logger.warning(f"{task_id}: 没有可用的application（全部失败或被过滤）")
        return False

    best_app, score = result
    idea_fields = _extract_idea_fields(best_app)

    # 读取paper_id（从paper_ideas对应文件）
    paper_ideas_dir = _PROJECT_ROOT / "benchmark" / "dataset" / "paper_ideas"
    paper_file = grouped_json_path(paper_ideas_dir, task_id)
    paper_id = ""
    if paper_file.exists():
        with open(paper_file, encoding="utf-8") as f:
            paper_data = json.load(f)
        paper_id = paper_data.get("paper_id", "")

    # 构造best_idea.json
    best_idea = {
        "benchmark_admission": {
            "version": 1,
            "source_sha256": source_digest(output_file),
            "application_index": full_output["applications"].index(best_app),
            "boundary": full_output["boundary"],
            "system": full_output.get("system"),
        },
        "paper_id":         paper_id,
        "task_id":          task_id,
        "system_idea":      idea_fields,
        # Keep the structured projection for inspection; evaluation still reloads
        # and hash-verifies the source application as the authority.
        "required_datasets": best_app.get("required_datasets") or [],
        "component_usage": [
            {
                "step_id": step.get("id"),
                "source_kind": item.get("source_kind"),
                "source_id": item.get("source_id"),
                "role": item.get("role"),
            }
            for step in (best_app.get("task_plan") or [])
            if isinstance(step, dict)
            for item in (step.get("inputs") or [])
            if isinstance(item, dict)
            and item.get("source_kind") in {"base_dataset", "required_dataset"}
        ],
        "source_application": best_app.get("application_name") or best_app.get("idea_id", ""),
        "selection_score":  score,
        "system_scores": {
            "novelty":       (best_app.get("idea_evaluation") or {}).get("scores", {}).get("novelty", 0),
            "realizability": (best_app.get("idea_evaluation") or {}).get("scores", {}).get("realizability", 0),
        },
        "status":           best_app.get("status", ""),
        "final_disposition": best_app.get("final_disposition", ""),
    }

    with open(best_file, "w", encoding="utf-8") as f:
        json.dump(best_idea, f, indent=2, ensure_ascii=False)

    logger.info(f"✓ {task_id}: best_idea保存完成 (score={score:.1f}, app={best_idea['source_application']})")
    return True


def main():
    parser = argparse.ArgumentParser(description="标准化System Ideas")
    parser.add_argument(
        "--task-id",
        type=str,
        help="规范任务 ID，例如 single_dataset/task_001",
    )
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--system-outputs-dir", type=Path, help="待标准化系统的独立产物目录")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    system_outputs_dir = args.system_outputs_dir or system_outputs_root(
        _PROJECT_ROOT / "benchmark", "ours"
    )

    if args.task_id:
        benchmark_dir = _PROJECT_ROOT / "benchmark"
        require_canonical_task_id(benchmark_dir, args.task_id)
        standardize_single_task(args.task_id, system_outputs_dir)
        return

    # 发现所有已有full_output的任务
    benchmark_dir = _PROJECT_ROOT / "benchmark"
    task_ids = [
        task_id
        for task_id in canonical_task_ids(benchmark_dir)
        if (grouped_task_dir(system_outputs_dir, task_id) / "full_output.json").is_file()
    ]
    if not task_ids:
        logger.warning("没有找到任何system_outputs，请先运行 launch_discovery.py --benchmark all")
        return

    success, failed = 0, []
    for task_id in task_ids:
        ok = standardize_single_task(task_id, system_outputs_dir)
        if ok:
            success += 1
        else:
            failed.append(task_id)

    logger.info(f"\n标准化完成: {success}/{len(task_ids)}")
    if failed:
        logger.warning(f"失败任务: {', '.join(failed)}")


if __name__ == "__main__":
    main()

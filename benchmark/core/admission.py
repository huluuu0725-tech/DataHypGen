"""Benchmark 生成产物准入；不执行模型或修复源产物。"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

# 语义性终止：idea 因生成侧判定“语义 / 发现质量未达标”而结束，而非资源或结构现实。
# 评估侧不接收这类候选，避免对生成侧自身的语义判断重复计分。
SEMANTIC_TERMINAL_RULES = frozenset({
    "T7_quality_replan_limit_exceeded",
})


def has_plan_content(value) -> bool:
    """只判断是否生成计划内容，不判定表示格式或计划质量。"""
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return any(has_plan_content(item) for item in value.values())
    if isinstance(value, list):
        return any(has_plan_content(item) for item in value)
    return False


def terminal_rule(app: Dict[str, Any]) -> str:
    feedback = app.get("resource_check_feedback") or {}
    rule = feedback.get("terminal_rule") if isinstance(feedback, dict) else None
    return rule if isinstance(rule, str) else ""


def eligible_applications(output: dict) -> list[dict]:
    """仅接收已到达声明边界、具有完整 StepPlan 且非语义性终止的候选。"""
    if output.get("boundary_reached") is not True:
        raise ValueError("生成流程未到达边界")
    family = output.get("family")
    boundary = output.get("boundary")
    states = output.get("executed_states") or []
    if not family:
        raise ValueError("产物缺少 family，无法确认生成边界")
    if family == "internagent":
        if boundary != "refinement_terminal" or not {"resource_check", "plan_refinement"}.issubset(states):
            raise ValueError("ours 必须完成 ResourceCheck 和 refinement")
        if output.get("final_state") != "idea_evaluation" or "idea_evaluation" in states:
            raise ValueError("ours 必须停在内部 IDEA_EVALUATION 之前")
    elif boundary not in {"planned", "resource_check", "refinement_terminal"}:
        raise ValueError("baseline 必须完成 step plan，不能使用 idea-only 产物")

    applications = []
    for app in output.get("applications", []):
        if not isinstance(app, dict):
            continue
        # Evaluation must observe the complete generated idea distribution.
        # Resource and structural failures are scored as evidence of a weak or
        # incomplete design; they must not be filtered before Alignment/Depth.
        # Only applications without any plan material, or whose termination was
        # itself a semantic/discovery-quality judgement by the generator, are
        # ineligible.
        if not has_plan_content(app.get("task_plan")):
            continue
        if terminal_rule(app) in SEMANTIC_TERMINAL_RULES:
            continue
        applications.append(app)
    if not applications:
        raise ValueError("没有具有完整 StepPlan 的可评测 application")
    return applications


def source_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_admitted_portfolio(full_output_path: Path) -> tuple[dict, list[tuple[int, dict]]]:
    """返回源产物与全部 admitted application 及其在 applications 里的下标。

    评估侧不再预选单条 idea；Benchmark-v3 §4.1 的 best-of / median-of /
    full-distribution 必须在全部 admitted idea 评完之后由汇总层计算。
    """
    output = json.loads(full_output_path.read_text(encoding="utf-8"))
    admitted = eligible_applications(output)
    applications = output["applications"]
    return output, [(applications.index(app), app) for app in admitted]


def project_idea_fields(app: Dict[str, Any]) -> Dict[str, Any]:
    """把 application 投影成与 Paper Idea 同形的 8 字段；不改变源产物。"""
    idea_cand = app.get("idea_candidate") or {}
    if not isinstance(idea_cand, dict):
        idea_cand = {}

    research_question = (
        app.get("research_question")
        or idea_cand.get("research_question")
        or app.get("application_name", "")
    )

    mech = idea_cand.get("theoretical_mechanism") or {}
    if isinstance(mech, str):
        hypothesis = mech
        mechanism = mech
    elif isinstance(mech, dict):
        hypothesis = (
            mech.get("core_hypothesis")
            or mech.get("hypothesis")
            or mech.get("mechanism_chain", "")
        )
        mechanism = (
            mech.get("mechanism_chain")
            or mech.get("causal_path")
            or mech.get("description", "")
        )
    else:
        hypothesis = ""
        mechanism = ""

    anchor = idea_cand.get("knowledge_anchor_and_motivation") or {}
    if isinstance(anchor, dict):
        target = anchor.get("target_variable") or anchor.get("dependent_variable") or ""
        predictors = anchor.get("predictors") or anchor.get("independent_variables") or []
    else:
        target = ""
        predictors = []
    if isinstance(predictors, str):
        predictors = [predictors]

    req_datasets = app.get("required_datasets") or []
    if isinstance(req_datasets, list) and req_datasets:
        data_sources = []
        for rd in req_datasets:
            if isinstance(rd, dict):
                name = rd.get("entity") or rd.get("dataset_name") or rd.get("name") or str(rd)
                data_sources.append(name)
            else:
                data_sources.append(str(rd))
    else:
        grounding = app.get("idea_dataset_grounding") or []
        data_sources = [
            (g.get("dataset_name") or g.get("name") or str(g))
            if isinstance(g, dict) else str(g)
            for g in grounding
        ]

    # 保留完整计划内容；不因字段名或表示形式丢弃 baseline 的步骤。
    task_plan = app.get("task_plan")
    if isinstance(task_plan, str):
        analytical_approach = task_plan
    elif task_plan:
        analytical_approach = json.dumps(task_plan, ensure_ascii=False)
    else:
        analytical_approach = app.get("approach_description") or app.get("application_description", "")

    value = idea_cand.get("novelty_constraints_and_value") or {}
    if isinstance(value, dict):
        expected_contribution = (
            value.get("expected_contribution")
            or value.get("system_value")
            or value.get("novelty_statement", "")
        )
    elif isinstance(value, str):
        expected_contribution = value
    else:
        expected_contribution = app.get("rationale", "")

    falsification_condition = (
        idea_cand.get("falsification_condition")
        or idea_cand.get("rejection_criteria")
        or "If the proposed relationship cannot be detected at p<0.05 in multiple independent study areas, the hypothesis would be falsified."
    )

    return {
        "research_question": str(research_question),
        "hypothesis": str(hypothesis),
        "mechanism": str(mechanism),
        "key_variables": {
            "target": str(target),
            "predictors": [str(p) for p in predictors] if predictors else [],
        },
        "data_sources": [str(d) for d in data_sources] if data_sources else [],
        "analytical_approach": str(analytical_approach),
        "expected_contribution": str(expected_contribution),
        "falsification_condition": str(falsification_condition),
    }


def load_admitted_application(path: Path) -> tuple[dict, dict, dict]:
    """重新检查源产物，旧缓存不能绕过准入。"""
    idea = json.loads(path.read_text(encoding="utf-8"))
    evidence = idea.get("benchmark_admission") or {}
    source = path.parent / "full_output.json"
    if evidence.get("version") != 1 or not source.exists():
        raise ValueError(f"{path}: 缺少准入证据，请重新标准化")
    if evidence.get("source_sha256") != source_digest(source):
        raise ValueError(f"{path}: 源产物已变化，请重新标准化")
    output = json.loads(source.read_text(encoding="utf-8"))
    apps = eligible_applications(output)
    index = evidence.get("application_index")
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(output["applications"]):
        raise ValueError("准入证据的 application_index 无效")
    application = output["applications"][index]
    if application not in apps:
        raise ValueError("选中的 application 不满足准入要求")
    return idea, output, application


def load_admitted_idea(path: Path) -> dict:
    """Reload the source output so cached projections cannot bypass admission."""
    idea, _, _ = load_admitted_application(path)
    return idea

"""DataHypGen benchmark generation and boundary-controlled session execution."""
import asyncio
import copy
import contextlib
import json
import logging
import os
import re
import sys
import yaml
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

from benchmark.core.log_hygiene import silence_noisy_third_party_loggers
from datahypgen.mas.tools.dataset_detail_enricher import DatasetDetailEnricher

load_dotenv(_PROJECT_ROOT / ".env", override=True)

logger = logging.getLogger(__name__)

DEFAULT_IDEA_COUNT = 5
DEFAULT_CONFIG = "config/debug_config.yaml"
DIRECT_LLM_RUNTIME_SYSTEM_ID = "direct_llm_plan"
RAG_LLM_RUNTIME_SYSTEM_ID = "rag_llm"


class ExperimentBoundary(str, Enum):
    IDEA = "idea"
    PLANNED = "planned"
    RESOURCE_CHECK = "resource_check"
    REFINEMENT_TERMINAL = "refinement_terminal"


@dataclass(frozen=True)
class SystemSpec:
    name: str
    family: str
    matrix: str
    boundary: ExperimentBoundary
    output_dir: Path
    config_relpath: str = DEFAULT_CONFIG
    task_dir: Optional[Path] = None




def _resolve_task_dir(task_name: str) -> Path:
    task_dir = _PROJECT_ROOT / "tasks" / task_name
    if not task_dir.is_dir():
        raise FileNotFoundError(f"Task directory not found: {task_dir}")
    return task_dir


def _resolve_task_prompt_path(task_name: str) -> Path:
    task_dir = _resolve_task_dir(task_name)
    prompt_path = task_dir / "prompt.json"
    if not prompt_path.exists():
        raise FileNotFoundError(f"prompt.json not found for task '{task_name}': {prompt_path}")
    return prompt_path


def _load_config(config_path: str) -> Dict[str, Any]:
    import yaml

    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    def _expand(obj: Any) -> Any:
        if isinstance(obj, str):
            return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), ""), obj)
        if isinstance(obj, dict):
            return {k: _expand(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_expand(v) for v in obj]
        return obj

    return _expand(raw)


def _provider_model_name(config: Dict[str, Any], provider: str) -> str:
    model_section = config.get("models", {})
    resolved_provider = provider
    if provider == "default":
        resolved_provider = model_section.get("default_provider", "openai")
    return model_section.get(resolved_provider, {}).get("model_name", "unknown")


def _agent_enabled(config: Dict[str, Any], agent_key: str) -> bool:
    agent_cfg = config.get("agents", {}).get(agent_key, {})
    if not isinstance(agent_cfg, dict):
        return True
    return bool(agent_cfg.get("enabled", True))


def _resolve_phase_models(spec: SystemSpec, config: Dict[str, Any]) -> Dict[str, str]:
    phase_models: Dict[str, str] = {}

    phases_cfg = config.get("phases", {})
    if phases_cfg.get("enable_dataset_profiling_phase", True):
        provider = config.get("agents", {}).get("dataset_profiler", {}).get("model_provider", "default")
        phase_models["dataset_profiling"] = _provider_model_name(config, provider)
    provider = config.get("agents", {}).get("survey", {}).get("model_provider", "default")
    phase_models["literature_search"] = _provider_model_name(config, provider)
    if _agent_enabled(config, "domain_knowledge"):
        provider = config.get("agents", {}).get("domain_knowledge", {}).get("model_provider", "default")
        phase_models["domain_knowledge"] = _provider_model_name(config, provider)
    idea_generation_cfg = config.get("agents", {}).get("idea_generation", {})
    step_plan_cfg = config.get("agents", {}).get("step_plan", {})
    idea_generation_provider = idea_generation_cfg.get("model_provider", "default")
    phase_models["idea_generation"] = _provider_model_name(config, idea_generation_provider)
    if spec.boundary != ExperimentBoundary.IDEA:
        step_plan_provider = step_plan_cfg.get("model_provider", "default")
        phase_models["step_planning"] = _provider_model_name(config, step_plan_provider)

    if spec.boundary in {
        ExperimentBoundary.RESOURCE_CHECK,
        ExperimentBoundary.REFINEMENT_TERMINAL,
    }:
        provider = config.get("agents", {}).get("resources_check", {}).get("model_provider", "default")
        phase_models["resource_check"] = _provider_model_name(config, provider)

    if spec.boundary == ExperimentBoundary.REFINEMENT_TERMINAL and config.get("phases", {}).get("enable_plan_refinement_phase", True):
        provider = config.get("agents", {}).get("refinement", {}).get("model_provider", "default")
        phase_models["plan_refinement"] = _provider_model_name(config, provider)

    return phase_models






def _primary_model_name(phase_models: Dict[str, str]) -> str:
    names = [name for name in phase_models.values() if name]
    if not names:
        return "unknown"
    unique_names = sorted(set(names))
    if len(unique_names) == 1:
        return unique_names[0]
    return "multiple"


def _build_output_dir(system_name: str, task_name: str, output_dir: Path) -> Path:
    """Use the complete output directory explicitly selected by the caller."""
    if not isinstance(output_dir, Path) or not output_dir.is_absolute():
        raise ValueError("The caller must provide an absolute output_dir Path")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@contextlib.contextmanager
def _task_log_capture(system_name: str, task_name: str, output_dir: Path):
    out_dir = _build_output_dir(system_name, task_name, output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"execution_{timestamp}.log"

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    try:
        logger.info(
            "[%s/%s] Writing execution log to %s",
            system_name,
            task_name,
            log_path,
        )
        yield log_path
    finally:
        handler.flush()
        root_logger.removeHandler(handler)
        handler.close()


def _valid_application_count(applications: Sequence[Dict[str, Any]]) -> int:
    valid = 0
    for app in applications:
        if not isinstance(app, dict):
            continue
        idea = app.get("idea_candidate")
        if isinstance(idea, dict) and str(idea.get("research_question") or "").strip():
            valid += 1
            continue
        if app.get("application_name") and app.get("application_description"):
            valid += 1
    return valid


def _derive_application_fields_from_idea(app: Dict[str, Any]) -> Dict[str, str]:
    idea = app.get("idea_candidate") if isinstance(app.get("idea_candidate"), dict) else {}
    motivation = idea.get("knowledge_anchor_and_motivation") if isinstance(idea.get("knowledge_anchor_and_motivation"), dict) else {}
    mechanism = idea.get("theoretical_mechanism") if isinstance(idea.get("theoretical_mechanism"), dict) else {}
    value = idea.get("novelty_constraints_and_value") if isinstance(idea.get("novelty_constraints_and_value"), dict) else {}
    return {
        "application_name": str(app.get("application_name") or idea.get("research_question") or ""),
        "application_description": str(
            app.get("application_description")
            or motivation.get("research_motivation")
            or ""
        ),
        "application_context": str(app.get("application_context") or f"idea_type={idea.get('idea_type', '')}"),
        "approach_description": str(
            app.get("approach_description")
            or app.get("proposed_method")
            or mechanism.get("mechanism_chain")
            or ""
        ),
        "rationale": str(app.get("rationale") or value.get("system_value") or ""),
    }


def _summarize_application_statuses(applications: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    statuses: List[Dict[str, Any]] = []
    for app in applications:
        if not isinstance(app, dict):
            continue
        execution_contract = ((app.get("resource_check") or {}).get("execution_contract", {}) if isinstance(app.get("resource_check"), dict) else {})
        statuses.append(
            {
                "idea_id": app.get("idea_id", ""),
                "status": app.get("status", ""),
                "final_disposition": app.get("final_disposition", ""),
                "replan_round": app.get("replan_round", 0),
                "resource_check_passed": execution_contract.get("resource_check_passed"),
                "terminal_rule": (app.get("resource_check_feedback") or {}).get("terminal_rule", ""),
            }
        )
    return statuses


def _application_record_to_output(app_record: Any) -> Dict[str, Any]:
    data = app_record.to_dict() if hasattr(app_record, "to_dict") else dict(app_record)
    return {
        "idea_id": data.get("idea_id", ""),
        "application_name": data.get("application_name", ""),
        "application_description": data.get("application_description", ""),
        # research_question 应是人类可读的科学问题，而非 binding token 汤。
        # 优先取真实生成的 research_question，其次 application_name（已是完整问题句/题名），
        # 最后才回退 application_context（含 gap/extra_data_role 等审计标记，非问题本体）。
        "research_question": (
            data.get("research_question")
            or data.get("application_name")
            or data.get("application_context", "")
        ),
        "application_context": data.get("application_context", ""),
        "approach_description": data.get("approach_description", data.get("proposed_method", "")),
        "idea_candidate": data.get("idea_candidate", {}),
        "idea_dataset_grounding": data.get("idea_dataset_grounding", []),
        "idea_revision_audit": data.get("idea_revision_audit", {}),
        "task_plan": data.get("task_plan", []),
        "required_datasets": data.get("required_datasets", []),
        "capability_anchor_meta": data.get("capability_anchor_meta", {}),
        "step_plan_generation_audit": data.get("step_plan_generation_audit", []),
        "plan_revision_history": data.get("plan_revision_history", []),
        "rationale": data.get("rationale", ""),
        "status": data.get("status", ""),
        "final_disposition": data.get("final_disposition", ""),
        "resource_check": data.get("resource_check", {}),
        "resource_check_feedback": data.get("resource_check_feedback", {}),
        "plan_refinement_result": data.get("plan_refinement_result", {}),
        "idea_evaluation": data.get("idea_evaluation", {}),
        "replan_round": data.get("replan_round", 0),
        "priority": data.get("priority", 0),
    }


def _standard_app_to_record_payload(app: Dict[str, Any], idx: int) -> Dict[str, Any]:
    derived = _derive_application_fields_from_idea(app)
    return {
        "idea_id": str(app.get("idea_id") or f"app_{idx + 1:03d}"),
        "application_name": derived["application_name"],
        "application_description": derived["application_description"],
        "application_context": derived["application_context"] or app.get("research_question", ""),
        "approach_description": derived["approach_description"],
        "priority": idx + 1,
        "status": app.get("status", "pending"),
        "final_disposition": app.get("final_disposition", ""),
        "task_plan": app.get("task_plan", []),
        "required_datasets": app.get("required_datasets", []),
        "capability_anchor_meta": app.get("capability_anchor_meta", {}),
        "plan_revision_history": app.get("plan_revision_history", []),
        "idea_candidate": app.get("idea_candidate", {}),
        "idea_dataset_grounding": app.get("idea_dataset_grounding", []),
        "idea_revision_audit": app.get("idea_revision_audit", {}),
        "datasets": app.get("datasets", []),
        "tools": app.get("tools", []),
        "rationale": derived["rationale"],
        "resource_check": app.get("resource_check", {}),
        "resource_check_feedback": app.get("resource_check_feedback", {}),
        "plan_refinement_result": app.get("plan_refinement_result", {}),
        "step_plan_generation_audit": app.get("step_plan_generation_audit", []),
        "idea_evaluation": app.get("idea_evaluation", {}),
        "replan_round": app.get("replan_round", 0),
    }


def _build_result_payload(
    spec: SystemSpec,
    task_name: str,
    phase_models: Dict[str, str],
    applications: List[Dict[str, Any]],
    requested_app_count: int,
    executed_states: List[str],
    stop_reason: str,
    final_state: str,
    boundary_reached: bool,
    planning_audit: Optional[Dict[str, Any]] = None,
    literature_review: Optional[List[Dict[str, Any]]] = None,
    dataset_portrait: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "system": spec.name,
        "family": spec.family,
        "matrix": spec.matrix,
        "boundary": spec.boundary.value,
        "task": task_name,
        "model": _primary_model_name(phase_models),
        "actual_models_by_phase": phase_models,
        "requested_app_count": requested_app_count,
        "valid_app_count": _valid_application_count(applications),
        "applications": applications,
        # idea_evaluation 的 novelty 评审要求“与现有工作显式对比”，评估器从 output.json
        # 顶层的 literature_review 取证据。缺该字段时评审只能按“无文献对比”系统性压低 novelty，
        # 因此必须把检索到的文献随 payload 落盘。
        "literature_review": literature_review or [],
        "dataset_portrait": dataset_portrait or {},
        "per_app_status": _summarize_application_statuses(applications),
        "executed_states": executed_states,
        "phases_completed": executed_states,
        "final_state": final_state,
        "stop_reason": stop_reason,
        "boundary_reached": boundary_reached,
    }
    if planning_audit:
        payload["planning_audit"] = planning_audit
    return payload


def _save_output(system_name: str, task_name: str, payload: Dict[str, Any], output_dir: Path) -> None:
    out_dir = _build_output_dir(system_name, task_name, output_dir)
    out_path = out_dir / "output.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info(f"[{system_name}] Saved output to {out_path}")


def _boundary_reached(
    boundary: ExperimentBoundary,
    current_state: Any,
) -> bool:
    from datahypgen.mas.workflow.data_type import WorkflowState

    if boundary == ExperimentBoundary.IDEA:
        return current_state == WorkflowState.RESOURCE_CHECK
    if boundary == ExperimentBoundary.PLANNED:
        raise ValueError("The planned boundary is handled by SharedPlanningBridge")
    if boundary == ExperimentBoundary.RESOURCE_CHECK:
        return current_state == WorkflowState.PLAN_REFINEMENT
    return current_state == WorkflowState.IDEA_EVALUATION


def _all_applications_failed_initial_step_plan(
    applications: Sequence[Dict[str, Any]],
) -> bool:
    if not applications:
        return False
    return all(
        isinstance(app, dict)
        and app.get("status") == "failed_before_evaluation"
        and isinstance(app.get("resource_check"), dict)
        and app["resource_check"].get("failure_stage") == "initial_step_plan"
        for app in applications
    )


def _validate_boundary(
    spec: SystemSpec,
    executed_states: Sequence[str],
    applications: Sequence[Dict[str, Any]] = (),
) -> None:
    executed = set(executed_states)

    if spec.boundary == ExperimentBoundary.IDEA:
        forbidden = {"resource_check", "plan_refinement", "idea_evaluation"}
        hit = sorted(executed & forbidden)
        if hit:
            raise RuntimeError(f"{spec.name} violated idea boundary: executed forbidden states {hit}")

    if spec.boundary == ExperimentBoundary.PLANNED:
        native_planned = {
            "native_stage_8_idea_adapter",
            "native_stage_9_plan_adapter",
        }
        shared_planned = {"shared_step_planning"}
        if native_planned <= executed:
            required = native_planned
        else:
            required = shared_planned
        missing = sorted(required - executed)
        if missing:
            raise RuntimeError(
                f"{spec.name} did not complete the shared planned boundary: {missing}"
            )
        forbidden = {
            "single_pass_resource_check",
            "resource_check",
            "plan_refinement",
            "replan",
            "idea_evaluation",
        }
        hit = sorted(executed & forbidden)
        if hit:
            raise RuntimeError(
                f"{spec.name} violated planned boundary: executed forbidden states {hit}"
            )

    if spec.boundary == ExperimentBoundary.RESOURCE_CHECK:
        if "resource_check" not in executed:
            raise RuntimeError(f"{spec.name} did not execute RESOURCE_CHECK under matrix B")
        forbidden = {"plan_refinement", "idea_evaluation"}
        hit = sorted(executed & forbidden)
        if hit:
            raise RuntimeError(f"{spec.name} violated resource-check boundary: executed forbidden states {hit}")

    if spec.boundary == ExperimentBoundary.REFINEMENT_TERMINAL:
        if (
            "resource_check" not in executed
            and not _all_applications_failed_initial_step_plan(applications)
        ):
            raise RuntimeError(f"{spec.name} did not execute RESOURCE_CHECK under matrix C")
        forbidden = {"idea_evaluation"}
        hit = sorted(executed & forbidden)
        if hit:
            raise RuntimeError(f"{spec.name} violated refinement-terminal boundary: executed forbidden states {hit}")

    if spec.name.endswith("_no_refinement") and "plan_refinement" in executed:
        raise RuntimeError(f"{spec.name} unexpectedly executed PLAN_REFINEMENT")


async def _run_session_to_boundary(
    spec: SystemSpec,
    task_name: str,
    session: Any,
    orchestration: Any,
    requested_app_count: int,
    phase_models: Dict[str, str],
    config_path: str,
    pseudo_states: Optional[List[str]] = None,
) -> Dict[str, Any]:
    from datahypgen.mas.workflow.data_type import WorkflowState

    executed_states = list(pseudo_states or [])
    stop_reason = ""
    boundary_reached = False
    resource_checked_ids: List[str] = []
    max_steps = 200
    steps_executed = 0
    maturation_saved = False  # 标记是否已保存成熟化结果

    while steps_executed < max_steps:
        current_state = session.state
        logger.info(f"[{spec.name}] Step {steps_executed + 1}: state={current_state.value}")

        if spec.boundary == ExperimentBoundary.RESOURCE_CHECK and current_state == WorkflowState.PLAN_REFINEMENT:
            current_app = orchestration._get_active_application_record(session)
            if current_app.idea_id not in resource_checked_ids:
                resource_checked_ids.append(current_app.idea_id)

            next_pending = next(
                (
                    app for app in orchestration._normalize_application_records(session)
                    if app.idea_id not in resource_checked_ids
                ),
                None,
            )
            if next_pending is None:
                stop_reason = "boundary_reached:resource_check_terminal"
                boundary_reached = True
                break

            session.active_application_id = next_pending.idea_id
            logger.info(
                f"[{spec.name}] ResourceCheck-only boundary: skip PLAN_REFINEMENT and continue with application {next_pending.idea_id}"
            )
            await orchestration.memory_manager.store_session(session)
            await orchestration._update_session_state(session, WorkflowState.RESOURCE_CHECK)
            continue

        if _boundary_reached(spec.boundary, current_state):
            if spec.boundary == ExperimentBoundary.IDEA:
                stop_reason = "boundary_reached:idea_maturation_terminal"
            elif spec.boundary == ExperimentBoundary.REFINEMENT_TERMINAL:
                stop_reason = "boundary_reached:refinement_terminal"
            boundary_reached = True
            break

        if current_state == WorkflowState.AWAITING_FEEDBACK:
            stop_reason = "workflow_paused:awaiting_feedback"
            break
        if current_state == WorkflowState.COMPLETED:
            stop_reason = "workflow_completed"
            break
        if current_state == WorkflowState.ERROR:
            raise RuntimeError("Error in MAS session")

        executed_states.append(current_state.value)
        await orchestration._execute_current_phase(session)
        await orchestration.memory_manager.store_session(session)
        steps_executed += 1

        # 成熟化完成后立即保存中间结果
        if not maturation_saved and getattr(session, "planning_audit", None):
            maturation_traces = session.planning_audit.get("idea_maturation_traces", [])
            if maturation_traces:  # 确认成熟化已执行
                logger.info(f"[{spec.name}] Idea maturation completed, saving intermediate output...")
                app_records = getattr(session, "application_records", []) or []
                applications = [_application_record_to_output(rec) for rec in app_records]
                intermediate_payload = _build_result_payload(
                    spec=spec,
                    task_name=task_name,
                    phase_models=phase_models,
                    applications=applications,
                    requested_app_count=requested_app_count,
                    executed_states=executed_states,
                    stop_reason="intermediate:idea_maturation_completed",
                    final_state=session.state.value,
                    boundary_reached=False,
                    planning_audit=session.planning_audit,
                    literature_review=getattr(session, "literature_review", []) or [],
                    dataset_portrait=getattr(session, "dataset_portrait", {}) or {},
                )
                _save_output(spec.name, task_name, intermediate_payload, spec.output_dir)
                maturation_saved = True

    if steps_executed >= max_steps:
        raise RuntimeError(f"{spec.name} exceeded max phase steps ({max_steps}) before reaching boundary")

    app_records = getattr(session, "application_records", []) or []
    applications = [_application_record_to_output(rec) for rec in app_records]
    _validate_boundary(spec, executed_states, applications)
    result = _build_result_payload(
        spec=spec,
        task_name=task_name,
        phase_models=phase_models,
        applications=applications,
        requested_app_count=requested_app_count,
        executed_states=executed_states,
        stop_reason=stop_reason,
        final_state=session.state.value,
        boundary_reached=boundary_reached,
        planning_audit=getattr(session, "planning_audit", {}),
        literature_review=getattr(session, "literature_review", []) or [],
        dataset_portrait=getattr(session, "dataset_portrait", {}) or {},
    )
    return result


async def _create_session(task_name: str, config_path: str, system_name: str = "", *, output_dir: Path, task_dir: Optional[Path] = None) -> Tuple[Any, Any, Any]:
    from datahypgen.stage import FindingGenerator

    class _Args:
        pass

    if task_dir is None:
        task_dir = _resolve_task_dir(task_name)

    args = _Args()
    args.config = config_path
    args.task_dir = str(task_dir)
    args.task_name = task_name
    args.offline_feedback = None
    args.output_dir = str(_build_output_dir(system_name, task_name, output_dir))
    generator = FindingGenerator(args, logger)
    await generator.load_task()
    orchestration = generator.interface.orchestration_agent
    session = await orchestration._get_session(generator.session_id)
    return generator, orchestration, session


async def _load_existing_session(
    task_name: str,
    config_path: str,
    session_id: str,
    system_name: str = "",
    *,
    output_dir: Path,
) -> Tuple[Any, Any, Any]:
    from datahypgen.stage import FindingGenerator

    class _Args:
        pass

    task_dir = _resolve_task_dir(task_name)
    _resolve_task_prompt_path(task_name)

    args = _Args()
    args.config = config_path
    args.task_dir = str(task_dir)
    args.task_name = task_name
    args.offline_feedback = None
    args.output_dir = str(_build_output_dir(system_name, task_name, output_dir))

    generator = FindingGenerator(args, logger)
    await generator.interface.startup()
    generator.session_id = session_id
    orchestration = generator.interface.orchestration_agent
    session = await orchestration._get_session(session_id)
    return generator, orchestration, session


def _requested_application_count(config: Dict[str, Any], family: str) -> int:
    agent_cfg = config.get("agents", {})
    idea_generation_cfg = agent_cfg.get("idea_generation", {})
    requested = idea_generation_cfg.get("application")
    if isinstance(requested, int) and requested > 0:
        return requested
    if family == "direct_llm":
        return DEFAULT_IDEA_COUNT
    return DEFAULT_IDEA_COUNT














def _validate_step_plan_resume_request(spec: SystemSpec, session_id: str) -> None:
    if spec.family != "internagent":
        raise ValueError("--resume-step-plan-session is only supported for DataGenHyp systems")
    if spec.boundary not in {
        ExperimentBoundary.RESOURCE_CHECK,
        ExperimentBoundary.REFINEMENT_TERMINAL,
    }:
        raise ValueError(
            "--resume-step-plan-session requires a resource-check or refinement-terminal boundary"
        )
    if not isinstance(session_id, str) or not re.fullmatch(
        r"session_[A-Za-z0-9_-]+",
        session_id,
    ):
        raise ValueError(
            "--resume-step-plan-session must be a plain session ID such as session_1785807741"
        )


def _validate_resource_check_resume_request(spec: SystemSpec, session_id: str) -> None:
    if spec.family != "internagent":
        raise ValueError("--resume-resource-check-session is only supported for DataGenHyp systems")
    if spec.boundary not in {
        ExperimentBoundary.RESOURCE_CHECK,
        ExperimentBoundary.REFINEMENT_TERMINAL,
    }:
        raise ValueError(
            "--resume-resource-check-session requires a resource-check or refinement-terminal boundary"
        )
    if not isinstance(session_id, str) or not re.fullmatch(
        r"session_[A-Za-z0-9_-]+",
        session_id,
    ):
        raise ValueError(
            "--resume-resource-check-session must be a plain session ID such as session_1785807741"
        )


def _extract_authoritative_matured_ideas(
    planning_audit: Dict[str, Any],
    revision_audit_by_idea_id: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    if not isinstance(planning_audit, dict):
        raise ValueError("StepPlan resume requires a planning_audit object")
    traces = planning_audit.get("idea_maturation_traces")
    if not isinstance(traces, list) or not traces:
        raise ValueError(
            "StepPlan resume requires non-empty planning_audit.idea_maturation_traces"
        )

    recovered: List[Dict[str, Any]] = []
    seen_idea_ids = set()
    for trace_index, trace in enumerate(traces):
        label = f"idea_maturation_traces[{trace_index}]"
        if not isinstance(trace, dict):
            raise ValueError(f"{label} must be an object")
        final_status = trace.get("final_status")
        if final_status not in {"matured", "preserved_original"}:
            raise ValueError(
                f"{label}.final_status must be an authoritative final status: "
                "'matured' or 'preserved_original'"
            )

        revision_audit = trace.get("revision_audit")
        if not isinstance(revision_audit, dict):
            # IdeaMaturation 用 `trace.pop("revision_audit")` 把这份 audit 搬到
            # application_plans[*].idea_revision_audit（最终落在 ApplicationRecord 上），
            # trace 自身从不携带它，因此 trace 永远过不了这里的类型校验。允许按 idea_id
            # 从同一 session 的既有 record 精确回取那一份真实 audit；取不到仍 fail closed，
            # 不合成空 audit，也不跨 idea 借用。
            fallback = (revision_audit_by_idea_id or {}).get(trace.get("idea_id"))
            if isinstance(fallback, dict) and fallback:
                revision_audit = fallback
        if not isinstance(revision_audit, dict):
            raise ValueError(f"{label}.revision_audit must be an object")

        best_selection = trace.get("best_selection")
        if not isinstance(best_selection, dict):
            raise ValueError(f"{label}.best_selection must be an object")
        best_version = best_selection.get("best_version")
        if not isinstance(best_version, str) or not best_version.strip():
            raise ValueError(f"{label}.best_selection.best_version must be a non-empty string")

        candidate_versions = trace.get("candidate_versions")
        if not isinstance(candidate_versions, list) or not candidate_versions:
            raise ValueError(f"{label}.candidate_versions must be a non-empty list")
        matches = [
            candidate
            for candidate in candidate_versions
            if isinstance(candidate, dict) and candidate.get("version") == best_version
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{label} must contain exactly one candidate_versions entry for best_version {best_version!r}"
            )

        selected = matches[0]
        idea = selected.get("idea")
        if not isinstance(idea, dict) or not idea:
            raise ValueError(f"{label} best version {best_version!r} must contain a non-empty idea")
        idea_id = idea.get("idea_id")
        idea_type = idea.get("idea_type")
        if not isinstance(idea_id, str) or not idea_id.strip():
            raise ValueError(f"{label} best version {best_version!r} has no idea_id")
        if not isinstance(idea_type, str) or not idea_type.strip():
            raise ValueError(f"{label} best version {best_version!r} has no idea_type")
        if trace.get("idea_id") != idea_id:
            raise ValueError(f"{label} idea_id does not match its selected idea")
        if trace.get("idea_type") != idea_type:
            raise ValueError(f"{label} idea_type does not match its selected idea")
        if idea_id in seen_idea_ids:
            raise ValueError(f"Duplicate recovered idea_id: {idea_id}")
        seen_idea_ids.add(idea_id)
        recovered.append(
            {
                "idea": copy.deepcopy(idea),
                "idea_id": idea_id,
                "idea_type": idea_type,
                "best_version": best_version,
                "source_round": selected.get("source_round"),
                "revision_audit": copy.deepcopy(revision_audit),
            }
        )
    return recovered


def _step_plan_resume_candidate_datasets(
    planning_audit: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Return full StepPlan candidate evidence for ERROR-session resume.

    New sessions persist `final_candidate_inventory_full` so resume can pass the
    same enriched candidate objects StepPlan saw originally. Older sessions only
    have the compact audit view; for those, recover the best local approximation
    by re-running deterministic detail enrichment over `final_candidate_inventory`.
    """
    if not isinstance(planning_audit, dict):
        raise ValueError("StepPlan resume requires a planning_audit object")

    full_inventory = planning_audit.get("final_candidate_inventory_full")
    if full_inventory is not None:
        if not isinstance(full_inventory, list) or any(
            not isinstance(candidate, dict) for candidate in full_inventory
        ):
            raise ValueError(
                "StepPlan resume requires planning_audit.final_candidate_inventory_full "
                "to be a list of objects"
            )
        return copy.deepcopy(full_inventory)

    compact_inventory = planning_audit.get("final_candidate_inventory")
    if not isinstance(compact_inventory, list) or any(
        not isinstance(candidate, dict) for candidate in compact_inventory
    ):
        raise ValueError(
            "StepPlan resume requires planning_audit.final_candidate_inventory "
            "to be a list of objects"
        )
    return DatasetDetailEnricher().enrich_batch(copy.deepcopy(compact_inventory))


async def _run_datahypgen_from_step_plan_session(
    task_name: str,
    spec: SystemSpec,
    config_path: str,
    session_id: str,
) -> Dict[str, Any]:
    from datahypgen.mas.agents.step_plan_agent import StepPlanContractError
    from datahypgen.mas.workflow.data_type import ApplicationRecord, WorkflowState

    _validate_step_plan_resume_request(spec, session_id)
    config = _load_config(config_path)
    requested = _requested_application_count(config, spec.family)
    _, orchestration, session = await _load_existing_session(
        task_name,
        config_path,
        session_id,
        spec.name,
        output_dir=spec.output_dir,
    )

    if session.id != session_id:
        raise ValueError(
            f"Loaded session ID {session.id!r} does not match requested session {session_id!r}"
        )

    existing_records = list(getattr(session, "application_records", []) or [])
    revision_audit_by_idea_id: Dict[str, Dict[str, Any]] = {}
    if not existing_records:
        # 形态 A：initial StepPlan 抛出非合同异常，session 停在 ERROR，尚未写入任何 record。
        if session.state != WorkflowState.ERROR:
            raise ValueError(
                f"StepPlan resume requires an ERROR session, got {session.state.value}"
            )
        if getattr(session, "active_application_id", None) is not None:
            raise ValueError("StepPlan resume requires active_application_id to be null")
    else:
        # 形态 B：StepPlan 合同对每个 idea 都失败，orchestration 逐条写入
        # failed_before_evaluation 后 session 正常推进到终态。这些 record 不含任何可用
        # 计划，重跑整体替换它们不会覆盖有效结果，因此不要求 ERROR 状态和空 record 列表。
        # 只要有一条不是 failed_before_evaluation（pending/已评估等），就说明存在真实结果，
        # 一律拒绝而不是静默覆盖。
        unexpected = [
            f"{getattr(app, 'idea_id', '')}={getattr(app, 'status', '')}"
            for app in existing_records
            if getattr(app, "status", "") != "failed_before_evaluation"
        ]
        if unexpected:
            raise ValueError(
                "StepPlan resume requires every existing application_record to be "
                f"failed_before_evaluation, got: {unexpected}"
            )
        for app in existing_records:
            idea_id = getattr(app, "idea_id", "")
            audit = getattr(app, "idea_revision_audit", None)
            if isinstance(idea_id, str) and idea_id and isinstance(audit, dict) and audit:
                revision_audit_by_idea_id[idea_id] = copy.deepcopy(audit)

    planning_audit = getattr(session, "planning_audit", None)
    recovered_ideas = _extract_authoritative_matured_ideas(
        planning_audit,
        revision_audit_by_idea_id,
    )
    candidate_datasets = _step_plan_resume_candidate_datasets(planning_audit)

    step_plan_agent = orchestration._get_agent("step_plan")
    if not step_plan_agent:
        raise ValueError("Step plan agent initialization failed")
    resources_agent = orchestration._get_agent("resources_check")
    if not resources_agent:
        raise ValueError(
            "Resource check agent initialization failed for StepPlan resume"
        )

    dataset_info = getattr(session.task, "spatial_extent", None)
    if not isinstance(dataset_info, dict) or not dataset_info:
        raise ValueError("StepPlan resume requires session.task.spatial_extent")

    application_plans: List[Dict[str, Any]] = []
    for priority, recovered in enumerate(recovered_ideas):
        idea = recovered["idea"]
        selection = {
            "idea": idea,
            "plan": {
                "idea_id": recovered["idea_id"],
                "idea_type": recovered["idea_type"],
                "priority": priority,
                "idea_candidate": copy.deepcopy(idea),
                "idea_revision_audit": copy.deepcopy(recovered["revision_audit"]),
            },
        }
        try:
            step_result = await orchestration._execute_step_plan_with_verified_datasets(
                step_plan_agent=step_plan_agent,
                resources_agent=resources_agent,
                step_context={
                    "idea": orchestration._build_step_plan_idea_payload(
                        idea=idea,
                        revision_audit=recovered["revision_audit"],
                        dataset_info=dataset_info,
                        candidate_datasets=candidate_datasets,
                    ),
                    "dataset_info": dataset_info,
                },
                step_params={"mode": "plan_for_idea"},
            )
            app_plan = orchestration._build_application_plan_from_selected_idea(
                selection,
                step_result,
            )
        except StepPlanContractError as exc:
            logger.error(
                "Resumed StepPlan contract failed for idea %s: %s",
                recovered["idea_id"],
                exc,
            )
            app_plan = orchestration._build_failed_application_plan_from_step_error(
                selection,
                exc,
            )
        application_plans.append(app_plan)

    application_records = [
        ApplicationRecord.from_dict(plan)
        for plan in application_plans
        if isinstance(plan, dict)
    ]
    if len(application_records) != len(recovered_ideas):
        raise ValueError("StepPlan resume did not produce one ApplicationRecord per recovered idea")

    first_pending_id = next(
        (app.idea_id for app in application_records if app.status == "pending"),
        None,
    )
    if first_pending_id is not None:
        target_state = WorkflowState.RESOURCE_CHECK
        active_application_id = first_pending_id
    else:
        first_failed_id = next(
            (
                app.idea_id
                for app in application_records
                if app.status == "failed_before_evaluation"
            ),
            None,
        )
        if first_failed_id is None:
            raise ValueError("StepPlan resume produced neither pending nor failed applications")
        target_state = WorkflowState.IDEA_EVALUATION
        active_application_id = first_failed_id

    resume_audit = {
        "mode": "in_place_from_matured_ideas",
        "source_session_id": session.id,
        # 记录真实源状态：形态 B 的 session 已推进到 completed/idea_evaluation 等终态，
        # 硬编码 ERROR 会让审计与实际恢复来源不符。
        "source_state": session.state.value,
        "source_application_record_count": len(existing_records),
        "recovered_revision_audit_from_records": sorted(revision_audit_by_idea_id),
        "target_state": target_state.value,
        "recovered_idea_count": len(recovered_ideas),
        "recovered_ideas": [
            {
                "idea_id": recovered["idea_id"],
                "idea_type": recovered["idea_type"],
                "best_version": recovered["best_version"],
                "source_round": recovered["source_round"],
            }
            for recovered in recovered_ideas
        ],
    }
    updated_planning_audit = copy.deepcopy(planning_audit)
    updated_planning_audit["step_plan_resume"] = resume_audit
    updated_planning_audit["generated_application_plan_count"] = len(application_records)
    updated_planning_audit["selected_application_count"] = len(application_records)
    updated_planning_audit["selected_idea_ids"] = [
        app.idea_id for app in application_records
    ]

    # This is a validated recovery operation, not a normal ERROR state transition.
    session.application_records = application_records
    session.active_application_id = active_application_id
    session.planning_audit = updated_planning_audit
    session.state = target_state
    await orchestration.memory_manager.store_session(session)

    phase_models = _resolve_phase_models(spec, config)
    result = await _run_session_to_boundary(
        spec=spec,
        task_name=task_name,
        session=session,
        orchestration=orchestration,
        requested_app_count=requested,
        phase_models=phase_models,
        pseudo_states=["step_planning"],
        config_path=config_path,
    )
    result["task"] = task_name
    _save_output(spec.name, task_name, result, spec.output_dir)
    return result


async def _run_datahypgen_from_resource_check_session(
    task_name: str,
    spec: SystemSpec,
    config_path: str,
    session_id: str,
) -> Dict[str, Any]:
    from datahypgen.mas.workflow.data_type import ApplicationRecord, WorkflowState

    _validate_resource_check_resume_request(spec, session_id)
    config = _load_config(config_path)
    requested = _requested_application_count(config, spec.family)
    _, orchestration, session = await _load_existing_session(
        task_name,
        config_path,
        session_id,
        spec.name,
        output_dir=spec.output_dir,
    )

    if session.id != session_id:
        raise ValueError(
            f"Loaded session ID {session.id!r} does not match requested session {session_id!r}"
        )
    if session.state not in {WorkflowState.RESOURCE_CHECK, WorkflowState.PLAN_REFINEMENT}:
        raise ValueError(
            "ResourceCheck resume requires a session currently at resource_check "
            f"or plan_refinement, got {session.state.value}"
        )

    application_records = list(getattr(session, "application_records", []) or [])
    if application_records and isinstance(application_records[0], dict):
        application_records = [
            ApplicationRecord.from_dict(item)
            for item in application_records
            if isinstance(item, dict)
        ]
        session.application_records = application_records
    if not application_records:
        raise ValueError("ResourceCheck resume requires non-empty application_records")

    active_application_id = getattr(session, "active_application_id", None)
    if not isinstance(active_application_id, str) or not active_application_id.strip():
        raise ValueError("ResourceCheck resume requires active_application_id")
    current_app = next(
        (app for app in application_records if app.idea_id == active_application_id),
        None,
    )
    if current_app is None:
        raise ValueError(
            f"Active application {active_application_id!r} not found in application_records"
        )
    if not isinstance(current_app.task_plan, list) or not current_app.task_plan:
        raise ValueError(
            f"ResourceCheck resume requires active application {active_application_id} "
            "to have a non-empty task_plan"
        )
    if not isinstance(current_app.capability_anchor_meta, dict):
        raise ValueError(
            f"ResourceCheck resume requires active application {active_application_id} "
            "to have capability_anchor_meta"
        )

    updated_planning_audit = copy.deepcopy(getattr(session, "planning_audit", {}) or {})
    updated_planning_audit["resource_check_resume"] = {
        "mode": "in_place_from_existing_step_plan",
        "source_session_id": session.id,
        "source_state": session.state.value,
        "target_state": WorkflowState.RESOURCE_CHECK.value,
        "active_application_id": active_application_id,
    }

    session.planning_audit = updated_planning_audit
    session.state = WorkflowState.RESOURCE_CHECK
    session.completed_at = None
    await orchestration.memory_manager.store_session(session)

    phase_models = _resolve_phase_models(spec, config)
    result = await _run_session_to_boundary(
        spec=spec,
        task_name=task_name,
        session=session,
        orchestration=orchestration,
        requested_app_count=requested,
        phase_models=phase_models,
        pseudo_states=["step_planning"],
        config_path=config_path,
    )
    result["task"] = task_name
    _save_output(spec.name, task_name, result, spec.output_dir)
    return result


async def _run_datahypgen_from_initial(task_name: str, spec: SystemSpec, config_path: str) -> Dict[str, Any]:
    from datahypgen.mas.workflow.data_type import WorkflowState

    config = _load_config(config_path)
    requested = _requested_application_count(config, spec.family)
    _, orchestration, session = await _create_session(task_name, config_path, spec.name, output_dir=spec.output_dir, task_dir=spec.task_dir)
    if spec.boundary == ExperimentBoundary.IDEA:
        setattr(session, "stop_after_idea_maturation", True)

    if session.state == WorkflowState.INITIAL:
        if orchestration.enable_dataset_profiling_phase:
            await orchestration._update_session_state(session, WorkflowState.DATASET_PROFILING)
        else:
            await orchestration._update_session_state(session, WorkflowState.LITERATURE_SEARCH)

    phase_models = _resolve_phase_models(spec, config)
    result = await _run_session_to_boundary(
        spec=spec,
        task_name=task_name,
        session=session,
        orchestration=orchestration,
        requested_app_count=requested,
        phase_models=phase_models,
        config_path=config_path,
    )
    result["task"] = task_name
    _save_output(spec.name, task_name, result, spec.output_dir)
    return result




async def run_system(
    spec: SystemSpec,
    task_name: str,
    config_path: str,
    *,
    resume_step_plan_session: Optional[str] = None,
    resume_resource_check_session: Optional[str] = None,
) -> Dict[str, Any]:
    if spec.family != "internagent":
        raise ValueError("This runner supports only the internagent family")
    if resume_step_plan_session is not None and resume_resource_check_session is not None:
        raise ValueError(
            "--resume-step-plan-session cannot be combined with --resume-resource-check-session"
        )
    if resume_step_plan_session is not None:
        return await _run_datahypgen_from_step_plan_session(
            task_name,
            spec,
            config_path,
            resume_step_plan_session,
        )
    if resume_resource_check_session is not None:
        return await _run_datahypgen_from_resource_check_session(
            task_name,
            spec,
            config_path,
            resume_resource_check_session,
        )
    return await _run_datahypgen_from_initial(task_name, spec, config_path)



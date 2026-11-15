"""Deterministic validation for the shared artifact-reference StepPlan DAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


SOURCE_KINDS = {"base_dataset", "required_dataset", "step_output"}
OUTPUT_ROLES = {
    "intermediate",
    "hypothesis_test",
    "validation",
    "visualization",
    "report",
}
CRITICAL_OUTPUT_ROLES = {"hypothesis_test", "validation"}
BASE_DATASET_SOURCE_ID = "base_dataset"


@dataclass
class PlanDAGValidation:
    """Serializable result of validating one planned research workflow."""

    valid: bool
    issues: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    topological_order: List[int] = field(default_factory=list)
    critical_output_ids: List[str] = field(default_factory=list)
    used_required_dataset_ids: List[str] = field(default_factory=list)

    def messages(self) -> List[str]:
        return [str(issue["message"]) for issue in self.issues]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_dag_valid": self.valid,
            "plan_dag_issues": list(self.issues),
            "plan_dag_edges": list(self.edges),
            "plan_dag_topological_order": list(self.topological_order),
            "critical_output_ids": list(self.critical_output_ids),
            "used_required_dataset_ids": list(self.used_required_dataset_ids),
        }


def _issue(code: str, message: str, **context: Any) -> Dict[str, Any]:
    return {"code": code, "message": message, **context}


def step_input_data_types(step: Dict[str, Any]) -> List[str]:
    """Read input types from the authoritative inputs contract."""
    inputs = step["inputs"]
    return [str(item["data_type"]).strip() for item in inputs]


def step_primary_output_type(step: Dict[str, Any]) -> str:
    """Read the sole primary output type from the authoritative outputs contract."""
    return str(step["outputs"][0]["data_type"]).strip()


def step_required_dataset_ids(step: Dict[str, Any]) -> List[str]:
    """Derive external dataset dependencies from explicit DAG input references."""
    return [
        str(item["source_id"]).strip()
        for item in step["inputs"]
        if item["source_kind"] == "required_dataset"
    ]


def validate_plan_dag(
    required_datasets: Any,
    task_plan: Any,
) -> PlanDAGValidation:
    """Validate the shared plan contract without repairing or reordering the plan."""
    issues: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    if not isinstance(required_datasets, list):
        issues.append(_issue("required_datasets_not_list", "required_datasets must be a list"))
        required_datasets = []
    if not isinstance(task_plan, list) or not task_plan:
        issues.append(_issue("task_plan_not_nonempty_list", "task_plan must be a non-empty list"))
        return PlanDAGValidation(valid=False, issues=issues)

    declared_dataset_ids: set[str] = set()
    for index, requirement in enumerate(required_datasets):
        if not isinstance(requirement, dict):
            issues.append(_issue(
                "required_dataset_not_object",
                f"required_datasets[{index}] must be an object",
                requirement_index=index,
            ))
            continue
        requirement_id = requirement.get("id")
        if not isinstance(requirement_id, str) or not requirement_id.strip():
            issues.append(_issue(
                "required_dataset_id_invalid",
                f"required_datasets[{index}].id must be a non-empty string",
                requirement_index=index,
            ))
            continue
        normalized_id = requirement_id.strip()
        if normalized_id in declared_dataset_ids:
            issues.append(_issue(
                "duplicate_required_dataset_id",
                f"required_datasets contains duplicate id: {normalized_id}",
                requirement_id=normalized_id,
            ))
        declared_dataset_ids.add(normalized_id)

    step_ids: set[int] = set()
    output_producers: Dict[str, Dict[str, Any]] = {}
    critical_output_ids: List[str] = []

    for index, step in enumerate(task_plan):
        if not isinstance(step, dict):
            issues.append(_issue(
                "step_not_object",
                f"task_plan[{index}] must be an object",
                step_index=index,
            ))
            continue
        step_id = step.get("id")
        if not isinstance(step_id, int) or isinstance(step_id, bool):
            issues.append(_issue(
                "step_id_invalid",
                f"task_plan[{index}].id must be an integer",
                step_index=index,
            ))
            continue
        if step_id in step_ids:
            issues.append(_issue(
                "duplicate_step_id",
                f"task_plan contains duplicate step id: {step_id}",
                step_id=step_id,
            ))
        step_ids.add(step_id)
        inputs = step.get("inputs")
        if not isinstance(inputs, list) or not inputs:
            issues.append(_issue(
                "step_inputs_invalid",
                f"step {step_id}.inputs must be a non-empty list",
                step_id=step_id,
            ))

        outputs = step.get("outputs")
        if not isinstance(outputs, list) or len(outputs) != 1:
            issues.append(_issue(
                "step_outputs_invalid",
                f"step {step_id}.outputs must contain exactly one output",
                step_id=step_id,
            ))
            continue
        output = outputs[0]
        if not isinstance(output, dict):
            issues.append(_issue(
                "step_output_not_object",
                f"step {step_id}.outputs[0] must be an object",
                step_id=step_id,
            ))
            continue
        output_id = output.get("id")
        output_type = output.get("data_type")
        output_role = output.get("role")
        if not isinstance(output_id, str) or not output_id.strip():
            issues.append(_issue(
                "step_output_id_invalid",
                f"step {step_id}.outputs[0].id must be a non-empty string",
                step_id=step_id,
            ))
        else:
            normalized_output_id = output_id.strip()
            if normalized_output_id in output_producers:
                issues.append(_issue(
                    "duplicate_output_id",
                    f"task_plan contains duplicate output id: {normalized_output_id}",
                    step_id=step_id,
                    output_id=normalized_output_id,
                ))
            output_producers[normalized_output_id] = {
                "step_id": step_id,
                "data_type": output_type,
            }
            if isinstance(output_role, str) and output_role in CRITICAL_OUTPUT_ROLES:
                critical_output_ids.append(normalized_output_id)
        if not isinstance(output_type, str) or not output_type.strip():
            issues.append(_issue(
                "step_output_type_invalid",
                f"step {step_id}.outputs[0].data_type must be a non-empty string",
                step_id=step_id,
            ))
        if not isinstance(output_role, str) or output_role not in OUTPUT_ROLES:
            issues.append(_issue(
                "step_output_role_invalid",
                f"step {step_id}.outputs[0].role must be one of {sorted(OUTPUT_ROLES)}",
                step_id=step_id,
            ))

    if not critical_output_ids:
        issues.append(_issue(
            "critical_output_missing",
            "task_plan must produce at least one hypothesis_test or validation output",
        ))

    adjacency: Dict[int, set[int]] = {step_id: set() for step_id in step_ids}
    indegree: Dict[int, int] = {step_id: 0 for step_id in step_ids}
    used_required_dataset_ids: set[str] = set()

    for step in task_plan:
        if not isinstance(step, dict):
            continue
        step_id = step.get("id")
        if not isinstance(step_id, int) or isinstance(step_id, bool):
            continue
        if step_id not in step_ids:
            continue
        inputs = step.get("inputs")
        if not isinstance(inputs, list):
            continue
        seen_inputs: set[tuple[str, str, str]] = set()
        for input_index, item in enumerate(inputs):
            if not isinstance(item, dict):
                issues.append(_issue(
                    "step_input_not_object",
                    f"step {step_id}.inputs[{input_index}] must be an object",
                    step_id=step_id,
                    input_index=input_index,
                ))
                continue
            source_kind = item.get("source_kind")
            source_id = item.get("source_id")
            data_type = item.get("data_type")
            role = item.get("role")
            if not isinstance(source_kind, str) or source_kind not in SOURCE_KINDS:
                issues.append(_issue(
                    "input_source_kind_invalid",
                    f"step {step_id}.inputs[{input_index}].source_kind is invalid",
                    step_id=step_id,
                    input_index=input_index,
                ))
                continue
            if not isinstance(source_id, str) or not source_id.strip():
                issues.append(_issue(
                    "input_source_id_invalid",
                    f"step {step_id}.inputs[{input_index}].source_id must be a non-empty string",
                    step_id=step_id,
                    input_index=input_index,
                ))
                continue
            normalized_source_id = source_id.strip()
            if not isinstance(data_type, str) or not data_type.strip():
                issues.append(_issue(
                    "input_data_type_invalid",
                    f"step {step_id}.inputs[{input_index}].data_type must be a non-empty string",
                    step_id=step_id,
                    input_index=input_index,
                ))
                continue
            if not isinstance(role, str) or not role.strip():
                issues.append(_issue(
                    "input_role_invalid",
                    f"step {step_id}.inputs[{input_index}].role must be a non-empty string",
                    step_id=step_id,
                    input_index=input_index,
                ))
                continue

            input_key = (source_kind, normalized_source_id, role.strip())
            if input_key in seen_inputs:
                issues.append(_issue(
                    "duplicate_step_input",
                    f"step {step_id} contains a duplicate input reference: {input_key}",
                    step_id=step_id,
                    input_index=input_index,
                ))
            seen_inputs.add(input_key)

            if source_kind == "base_dataset":
                if normalized_source_id != BASE_DATASET_SOURCE_ID:
                    issues.append(_issue(
                        "base_dataset_source_id_invalid",
                        f"step {step_id} base_dataset input must use source_id=base_dataset",
                        step_id=step_id,
                        source_id=normalized_source_id,
                    ))
                edges.append({
                    "source_kind": source_kind,
                    "source_id": normalized_source_id,
                    "to_step_id": step_id,
                    "data_type": data_type.strip(),
                    "role": role.strip(),
                })
                continue

            if source_kind == "required_dataset":
                if normalized_source_id not in declared_dataset_ids:
                    issues.append(_issue(
                        "undeclared_required_dataset_reference",
                        f"step {step_id} references undeclared required dataset id: {normalized_source_id}",
                        step_id=step_id,
                        source_id=normalized_source_id,
                    ))
                else:
                    used_required_dataset_ids.add(normalized_source_id)
                edges.append({
                    "source_kind": source_kind,
                    "source_id": normalized_source_id,
                    "to_step_id": step_id,
                    "data_type": data_type.strip(),
                    "role": role.strip(),
                })
                continue

            producer = output_producers.get(normalized_source_id)
            if producer is None:
                issues.append(_issue(
                    "unknown_step_output_reference",
                    f"step {step_id} references unknown step output: {normalized_source_id}",
                    step_id=step_id,
                    source_id=normalized_source_id,
                ))
                continue
            producer_step_id = producer["step_id"]
            producer_type = producer["data_type"]
            edges.append({
                "source_kind": source_kind,
                "source_id": normalized_source_id,
                "from_step_id": producer_step_id,
                "to_step_id": step_id,
                "data_type": data_type.strip(),
                "role": role.strip(),
            })
            if isinstance(producer_type, str) and producer_type.strip() != data_type.strip():
                issues.append(_issue(
                    "edge_type_mismatch",
                    f"output {normalized_source_id} has type {producer_type.strip()} but step {step_id} consumes it as {data_type.strip()}",
                    from_step_id=producer_step_id,
                    to_step_id=step_id,
                    source_id=normalized_source_id,
                    producer_type=producer_type.strip(),
                    consumer_type=data_type.strip(),
                ))
            if step_id not in adjacency[producer_step_id]:
                adjacency[producer_step_id].add(step_id)
                indegree[step_id] += 1

    unused_dataset_ids = sorted(declared_dataset_ids - used_required_dataset_ids)
    if unused_dataset_ids:
        issues.append(_issue(
            "unused_required_datasets",
            "required_datasets contains requirements unused by task_plan inputs: "
            + ", ".join(unused_dataset_ids),
            requirement_ids=unused_dataset_ids,
        ))

    ready = sorted(step_id for step_id, count in indegree.items() if count == 0)
    topological_order: List[int] = []
    while ready:
        current = ready.pop(0)
        topological_order.append(current)
        for successor in sorted(adjacency[current]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort()

    if len(topological_order) != len(step_ids):
        cyclic_step_ids = sorted(step_id for step_id, count in indegree.items() if count > 0)
        issues.append(_issue(
            "plan_cycle_detected",
            f"task_plan contains a dependency cycle involving steps: {cyclic_step_ids}",
            step_ids=cyclic_step_ids,
        ))

    return PlanDAGValidation(
        valid=not issues,
        issues=issues,
        edges=edges,
        topological_order=topological_order,
        critical_output_ids=critical_output_ids,
        used_required_dataset_ids=sorted(used_required_dataset_ids),
    )

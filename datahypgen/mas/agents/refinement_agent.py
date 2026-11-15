"""
Refinement Agent for DataGenHyp

Commit 5: 消费 Commit 4 结构化失败诊断（step_failures / failure_class / weakest_step_id），
         输出针对弱步骤的定点重规划反馈，不直接改写计划。

输入：goal + idea + idea_revision_audit + dataset_info + required_datasets + task_plan
       + (optional) literature_review
       + (optional) resource_check_feedback（来自 RESOURCE_CHECK 失败回流）
       + (optional) execution_contract（来自 RESOURCE_CHECK 契约，含 step_tool_validations）
输出：PlanRefinementResult.to_dict()
失败语义：输出模式非法时表示失败，编排器应直接进入 ERROR。
"""

import json
import logging
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent, AgentExecutionError

logger = logging.getLogger(__name__)


class RefinementAgent(BaseAgent):
    """
    对任务规划进行反思与精化的 Agent（仅 plan_refinement 模式）。

    在 RESOURCE_CHECK 之后运行，检查根据resource检查结果中的数据可行性、
    工具依赖是否合理，并输出结构化审查结果 (PlanRefinementResult)。
    若带 resource_check_feedback，则只输出给 TASK_PLANNING 的重规划反馈，不直接改计划。
    """

    def __init__(self, model, config: Dict[str, Any]):
        super().__init__(model, config)
        self.refinement_max_tokens = config.get("max_tokens")

    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        精化任务计划。

        Args:
            context (Dict): 必填：goal, idea, idea_revision_audit, task_plan,
                            required_datasets；可选：dataset_info,
                            literature_review, resource_check_feedback, execution_contract,
                            replan_round
            params (Dict): 暂未使用

        Returns:
            PlanRefinementResult.to_dict()
            mode 非法时表示失败。
        """
        from ..workflow.data_type import PlanRefinementResult

        goal: str = context.get("goal", "")
        dataset_info: Dict = context.get("dataset_info", {})
        task_plan: List[Dict] = context.get("task_plan", [])
        required_datasets = context.get("required_datasets")
        literature_review: List[Dict] = context.get("literature_review", [])
        resource_check_feedback: Optional[Dict] = context.get("resource_check_feedback")
        execution_contract: Optional[Dict] = context.get("execution_contract")
        replan_round = context.get("replan_round")
        idea = context.get("idea")
        idea_revision_audit = context.get("idea_revision_audit")

        if not goal:
            raise AgentExecutionError("goal is required for plan refinement")
        if not task_plan:
            raise AgentExecutionError("task_plan is required for plan refinement")
        if not isinstance(required_datasets, list):
            raise AgentExecutionError("required_datasets must be an explicit list for plan refinement")
        if not isinstance(idea, dict) or not idea:
            raise AgentExecutionError("idea is required for plan refinement")
        if not isinstance(idea_revision_audit, dict):
            raise AgentExecutionError("idea_revision_audit must be an explicit object for plan refinement")
        if isinstance(replan_round, bool) or not isinstance(replan_round, int) or replan_round < 0:
            raise AgentExecutionError("replan_round must be an explicit non-negative integer")
        if (
            not isinstance(self.refinement_max_tokens, int)
            or isinstance(self.refinement_max_tokens, bool)
            or self.refinement_max_tokens <= 0
        ):
            raise AgentExecutionError("refinement.max_tokens must be a positive integer")

        repair_constraint_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "required_operation": {"type": "string"},
                "preferred_function_names": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "forbidden_function_names": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "required_input_semantics": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "required_output_semantics": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
            },
            "required": [
                "required_operation",
                "preferred_function_names",
                "forbidden_function_names",
                "required_input_semantics",
                "required_output_semantics",
            ],
        }

        output_schema = {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["final_review", "replan_feedback"]
                },
                "should_replan": {
                    "type": "boolean"
                },
                "feedback_summary": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Summary of the final review or the replan guidance for TASK_PLANNING."
                },
                "step_level_adjustments": {
                    "type": "array",
                    "description": (
                        "Targeted per-step fix instructions for TASK_PLANNING. "
                        "Each item should identify a specific step and the fix action."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "issue_id": {
                                "type": "string",
                                "minLength": 1,
                                "description": "Stable identifier for closure tracking.",
                            },
                            "step_id": {"type": "integer", "description": "ID of the step to fix."},
                            "action": {
                                "type": "string",
                                "enum": ["replace_tool", "replace_function", "change_task_type",
                                         "add_conversion_step", "remove_step", "keep"],
                                "description": "What TASK_PLANNING should do with this step."
                            },
                            "reason": {
                                "type": "string",
                                "minLength": 1,
                                "description": "Why this fix is needed.",
                            },
                            "suggested_replacement": {
                                "type": "string",
                                "description": "Suggested replacement tool/function/task_type (if applicable)."
                            },
                            "repair_constraints": repair_constraint_schema,
                        },
                        "required": [
                            "step_id",
                            "action",
                            "reason",
                            "repair_constraints",
                        ],
                    },
                },
                "dataset_actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "issue_id": {
                                "type": "string",
                                "minLength": 1,
                            },
                            "action": {"type": "string", "minLength": 1},
                            "reason": {"type": "string", "minLength": 1},
                        },
                        "required": ["action", "reason"],
                    },
                },
                "tool_actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "minLength": 1},
                            "reason": {"type": "string", "minLength": 1},
                        },
                        "required": ["action", "reason"],
                    },
                },
                "must_preserve": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1}
                },
                "semantic_alignment_issues": {
                    "type": "array",
                    "description": (
                        "Plan-side semantic alignment issues relative to the accepted idea. "
                        "Only populate when task_plan fails to operationalize, contradicts, "
                        "or exceeds the accepted idea. "
                        "Do NOT use for missing datasets, unavailable tools, or step failures — "
                        "those belong in step_level_adjustments / dataset_actions."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "issue_type": {
                                "type": "string",
                                "enum": [
                                    "plan_mechanism_mismatch",
                                    "missing_idea_operationalization",
                                    "plan_scope_overreach",
                                ],
                                "description": (
                                    "Classifies how the task_plan must be corrected to remain "
                                    "faithful to the accepted idea."
                                ),
                            },
                            "affected_step_ids": {
                                "type": "array",
                                "minItems": 1,
                                "items": {
                                    "anyOf": [
                                        {"type": "integer"},
                                        {"type": "string", "minLength": 1},
                                    ]
                                },
                                "description": "Task-plan step IDs that require correction.",
                            },
                            "severity": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                            },
                            "reason": {
                                "type": "string",
                                "minLength": 1,
                                "description": "Why this is a semantic alignment issue.",
                            },
                            "evidence": {
                                "type": "string",
                                "minLength": 1,
                                "description": (
                                    "Specific evidence from the accepted idea and task_plan."
                                ),
                            },
                            "plan_fix": {
                                "type": "string",
                                "minLength": 1,
                                "description": (
                                    "Concrete guidance for TASK_PLANNING to adjust steps, "
                                    "required datasets, or tools without modifying the idea."
                                ),
                            },
                            "repair_constraints": repair_constraint_schema,
                        },
                        "required": [
                            "issue_type",
                            "affected_step_ids",
                            "severity",
                            "reason",
                            "evidence",
                            "plan_fix",
                            "repair_constraints",
                        ],
                    },
                },
            },
            "required": [
                "mode",
                "should_replan",
                "feedback_summary",
                "step_level_adjustments",
                "dataset_actions",
                "tool_actions",
                "must_preserve",
                "semantic_alignment_issues",
            ]
        }

        prompt = self._build_plan_refinement_prompt(
            goal=goal,
            dataset_info=dataset_info,
            task_plan=task_plan,
            required_datasets=required_datasets,
            literature_review=literature_review,
            resource_check_feedback=resource_check_feedback,
            execution_contract=execution_contract,
            idea=idea,
            idea_revision_audit=idea_revision_audit,
        )
        system_prompt = self._build_plan_refinement_system_prompt()

        try:
            response = await self._call_model(
                prompt=prompt,
                system_prompt=system_prompt,
                schema=output_schema,
                max_tokens=self.refinement_max_tokens,
            )

            if not isinstance(response, dict):
                raise AgentExecutionError("Refinement model output must be an object")
            response = self._assign_stable_issue_ids(response, replan_round)
            response = self._reconcile_removal_dependents(response, task_plan)
            refinement_result = PlanRefinementResult.from_dict({
                **response,
                "replan_round": replan_round,
            })

            # 合法布尔/模式组合中的语义问题强制回流 TASK_PLANNING；畸形类型不做修复。
            if (
                isinstance(refinement_result.semantic_alignment_issues, list)
                and refinement_result.semantic_alignment_issues
                and isinstance(refinement_result.should_replan, bool)
                and refinement_result.mode in {"final_review", "replan_feedback"}
            ):
                refinement_result.mode = "replan_feedback"
                refinement_result.should_replan = True
                logger.info(
                    f"[RefinementAgent] {len(refinement_result.semantic_alignment_issues)} "
                    "semantic_alignment_issues "
                    "detected; forcing mode=replan_feedback, should_replan=True"
                )

            return refinement_result.to_dict()

        except Exception as e:
            logger.error(f"[RefinementAgent] plan_refinement failed (round {replan_round}): {e}")
            return PlanRefinementResult(
                mode="invalid",
                should_replan=False,
                feedback_summary=f"Refinement failed: {e}",
                step_level_adjustments=[],
                dataset_actions=[],
                tool_actions=[],
                must_preserve=[],
                replan_round=replan_round,
            ).to_dict()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _assign_stable_issue_ids(
        response: Dict[str, Any],
        replan_round: int,
    ) -> Dict[str, Any]:
        """Assign deterministic IDs so issue closure does not depend on model wording."""
        normalized = dict(response)
        adjustments = []
        for index, raw_item in enumerate(response.get("step_level_adjustments", []) or [], 1):
            item = dict(raw_item) if isinstance(raw_item, dict) else raw_item
            if isinstance(item, dict):
                step_id = item.get("step_id", "unknown")
                action = str(item.get("action") or "adjust").strip().replace("_", "-")
                item["issue_id"] = (
                    f"refinement-r{replan_round}-step-{step_id}-{action}-{index}"
                )
                RefinementAgent._coerce_suggested_replacement(item)
                RefinementAgent._normalize_repair_constraints(item)
            adjustments.append(item)
        normalized["step_level_adjustments"] = adjustments

        semantic_issues = []
        for index, raw_item in enumerate(response.get("semantic_alignment_issues", []) or [], 1):
            item = dict(raw_item) if isinstance(raw_item, dict) else raw_item
            if isinstance(item, dict):
                issue_type = str(item.get("issue_type") or "semantic").replace("_", "-")
                step_ids = "-".join(
                    str(step_id)
                    for step_id in (item.get("affected_step_ids") or [])
                ) or "unknown"
                item["issue_id"] = (
                    f"refinement-r{replan_round}-{issue_type}-steps-{step_ids}-{index}"
                )
                RefinementAgent._normalize_repair_constraints(item)
            semantic_issues.append(item)
        normalized["semantic_alignment_issues"] = semantic_issues
        return normalized

    @staticmethod
    def _reconcile_removal_dependents(
        response: Dict[str, Any],
        task_plan: Any,
    ) -> Dict[str, Any]:
        """Flag `keep` verdicts that depend on a step this review asked to remove.

        `remove_step` 只描述被删节点，不描述其下游。实测中 refinement 同时要求删除
        step 3/5 又要求 keep step 6/7，而 6/7 正消费 3/5 的产出：删生产者、留消费者，
        local patch 必然过不了 DAG 契约，只能升级 full replan —— 而 full replan 会
        重新编号，把 id-keyed 的约束推入错位。归因应留在这里，而不是等下游报错。

        本方法不改删除意图，只把矛盾的 `keep` 降级为 `add_conversion_step` 并写明
        断掉的依赖，让 StepPlan 明确知道该节点的输入必须重新接。无法判定依赖关系时
        不做改动。
        """
        adjustments = response.get("step_level_adjustments")
        if not isinstance(adjustments, list) or not isinstance(task_plan, list):
            return response

        removal_ids = {
            item.get("step_id")
            for item in adjustments
            if isinstance(item, dict) and item.get("action") == "remove_step"
        }
        removal_ids.discard(None)
        if not removal_ids:
            return response

        # 产出 artifact -> 生产步骤，用于把消费引用解析回被删的生产者。
        producer_by_output: Dict[str, Any] = {}
        for step in task_plan:
            if not isinstance(step, dict):
                continue
            for output in step.get("outputs") or []:
                if isinstance(output, dict) and str(output.get("id") or "").strip():
                    producer_by_output[str(output["id"]).strip()] = step.get("id")

        broken_deps: Dict[Any, List[str]] = {}
        for step in task_plan:
            if not isinstance(step, dict) or step.get("id") in removal_ids:
                continue
            for item in step.get("inputs") or []:
                if not isinstance(item, dict) or item.get("source_kind") != "step_output":
                    continue
                source_id = str(item.get("source_id") or "").strip()
                if producer_by_output.get(source_id) in removal_ids:
                    broken_deps.setdefault(step.get("id"), []).append(source_id)

        if not broken_deps:
            return response

        normalized = dict(response)
        reconciled = []
        for item in adjustments:
            if (
                not isinstance(item, dict)
                or item.get("action") != "keep"
                or item.get("step_id") not in broken_deps
            ):
                reconciled.append(item)
                continue
            step_id = item.get("step_id")
            lost = sorted(set(broken_deps[step_id]))
            item = dict(item)
            item["action"] = "add_conversion_step"
            item["reason"] = (
                f"{item.get('reason') or ''} "
                f"[deterministic reconciliation] This step was reviewed as `keep`, but it "
                f"consumes {lost} produced by step(s) "
                f"{sorted(pid for pid in removal_ids if pid is not None)} that this same "
                "review asked to remove. Keeping it unchanged would leave a dangling "
                "dependency, so its inputs must be re-sourced from surviving steps."
            ).strip()
            # action 已从 keep 变为 add_conversion_step，约束契约要求至少一维非空
            # （keep 允许全空）。必须按新 action 重新归一化，否则 data_type 校验会
            # 报 "all constraint dimensions are empty"。
            RefinementAgent._normalize_repair_constraints(item)
            logger.info(
                f"[RefinementAgent] step {step_id} kept but depends on removed "
                f"output(s) {lost}; downgraded keep -> add_conversion_step"
            )
            reconciled.append(item)
        normalized["step_level_adjustments"] = reconciled
        return normalized

    @staticmethod
    def _coerce_suggested_replacement(item: Dict[str, Any]) -> None:
        """Coerce the optional suggested_replacement hint to a string in place.

        The output contract declares suggested_replacement as a string, but the
        model occasionally emits a scalar, list, object, or null. This field is an
        advisory hint (compiled into preferred_function_names downstream), so a
        malformed type must not hard-fail the whole refinement. Scalars and string
        lists are coerced; unusable shapes (dict/null/bool) drop the key so the
        strict contract validation on critical fields stays intact.
        """
        if "suggested_replacement" not in item:
            return
        value = item.get("suggested_replacement")
        if isinstance(value, str):
            return
        if isinstance(value, bool):
            item.pop("suggested_replacement", None)
            return
        if isinstance(value, (int, float)):
            item["suggested_replacement"] = str(value)
            return
        if isinstance(value, list):
            parts = [
                str(part).strip()
                for part in value
                if isinstance(part, (str, int, float))
                and not isinstance(part, bool)
                and str(part).strip()
            ]
            if parts:
                item["suggested_replacement"] = ", ".join(parts)
            else:
                item.pop("suggested_replacement", None)
            return
        item.pop("suggested_replacement", None)

    _REPAIR_CONSTRAINT_LIST_FIELDS = (
        "preferred_function_names",
        "forbidden_function_names",
        "required_input_semantics",
        "required_output_semantics",
    )

    @staticmethod
    def _normalize_repair_constraints(item: Dict[str, Any]) -> None:
        """Repair partially-shaped repair_constraints in place.

        约束的五个维度是 TASK_PLANNING 侧闭合校验的机器可读输入，契约要求键在验证时必须存在
        且为合法 dict（data_type.validate_repair_constraints）。以下情形需确定性补全：
          - 模型整体省略该键（None / 缺失）
          - 模型输出了非 dict 类型（字符串、列表等）
          - 模型输出了部分维度（缺字段 / 字段类型错误）
          - 模型输出了所有维度均为空的 dict（仅对非 keep 步骤触发验证错误）

        策略：repair_constraints 始终以合法 dict 写回 item，**从不 pop**。
        对于非 keep 步骤在所有维度归一化后仍为空的情况，用 suggested_replacement 或
        action 名称填入 required_operation，保证至少一个维度非空，满足验证契约。
        这样做不伪造实质约束，仅提供最小描述；TASK_PLANNING 仍可凭 action / reason /
        suggested_replacement 字段理解修复意图；闭合校验也能正常运行。
        """
        action = str(item.get("action") or "").strip()
        value = item.get("repair_constraints")

        if not isinstance(value, dict):
            # 模型输出了 None / 非 dict 类型，补全为最小合法结构。
            if action == "keep":
                value = {"required_operation": ""}
            elif action == "remove_step":
                # remove_step 惯例：required_operation 填 "none" 以通过非空检查。
                value = {"required_operation": "none"}
            else:
                # 其他 action：尝试从 suggested_replacement 提取操作描述。
                suggested = str(item.get("suggested_replacement") or "").strip()
                value = {"required_operation": suggested or action}

        normalized = dict(value)
        required_operation = normalized.get("required_operation")
        if isinstance(required_operation, bool) or not isinstance(
            required_operation, (str, int, float)
        ):
            normalized["required_operation"] = ""
        else:
            normalized["required_operation"] = str(required_operation)

        for field_name in RefinementAgent._REPAIR_CONSTRAINT_LIST_FIELDS:
            raw = normalized.get(field_name)
            if isinstance(raw, str):
                raw = [raw]
            elif not isinstance(raw, list):
                raw = []
            normalized[field_name] = [
                str(entry).strip()
                for entry in raw
                if isinstance(entry, (str, int, float))
                and not isinstance(entry, bool)
                and str(entry).strip()
            ]

        # 非 keep 步骤在所有维度归一化后仍为空时，填入最小 required_operation，
        # 避免触发 "all constraint dimensions are empty" 验证错误。
        # 不 pop：validate_repair_constraints 对缺失键同样报 "must be an object"。
        if action != "keep" and not bool(
            normalized["required_operation"].strip()
            or any(normalized[f] for f in RefinementAgent._REPAIR_CONSTRAINT_LIST_FIELDS)
        ):
            suggested = str(item.get("suggested_replacement") or "").strip()
            normalized["required_operation"] = suggested or action

        item["repair_constraints"] = normalized

    @staticmethod
    def _trim_idea_revision_audit(idea_revision_audit: Dict[str, Any]) -> Dict[str, Any]:
        """只保留 plan 审查真正消费的 audit 字段。

        `questioning_findings` / `revision_package` / `applied_revisions` /
        `semantic_writeback_checks` 是 idea 成熟化阶段的内部修订过程态。
        本 agent 不允许改写 idea，prompt 也从不引用这些字段，因此不注入。
        """
        allowed_fields = (
            "handoff_notes_for_step_plan",
            "unresolved_findings",
            "evidence_records",
            "maturation_rationale",
        )
        return {
            field: idea_revision_audit[field]
            for field in allowed_fields
            if idea_revision_audit.get(field) not in (None, "", [], {})
        }

    def _build_plan_refinement_prompt(
        self,
        goal: str,
        dataset_info: Dict[str, Any],
        task_plan: List[Dict[str, Any]],
        required_datasets: List[Dict[str, Any]],
        literature_review: List[Dict[str, Any]],
        resource_check_feedback: Optional[Dict[str, Any]],
        idea: Dict[str, Any],
        idea_revision_audit: Dict[str, Any],
        execution_contract: Optional[Dict[str, Any]] = None,
    ) -> str:

        prompt = f"# Research Goal\n{goal}\n\n"

        prompt += "# Accepted Idea (semantic authority; do not modify)\n"
        prompt += json.dumps(idea, ensure_ascii=False, separators=(',', ':')) + "\n\n"

        prompt += "# Idea Maturation Audit (independent review context)\n"
        prompt += json.dumps(
            self._trim_idea_revision_audit(idea_revision_audit),
            ensure_ascii=False,
            separators=(',', ':'),
        ) + "\n\n"

        if dataset_info:
            # required_datasets 已由下面的 StepPlan 契约段单独给出，这里不重复注入。
            base_dataset_info = {
                key: value
                for key, value in dataset_info.items()
                if key != "required_datasets"
            }
            if base_dataset_info:
                prompt += f"# Base Dataset and Resource Information\n{json.dumps(base_dataset_info, ensure_ascii=False, separators=(',', ':'))}\n\n"

        prompt += "# Required External Datasets (StepPlan contract)\n"
        prompt += json.dumps(required_datasets, ensure_ascii=False, separators=(',', ':')) + "\n\n"

        prompt += "# Current Task Plan\n"
        for i, step in enumerate(task_plan, 1):
            # implementation_candidates 是确定性查表证据，由 StepPlan 的本地选择逻辑消费；
            # 单个 step 可携带上千条候选函数，注入 prompt 只会挤占上下文而不影响 refinement 判断。
            step_view = (
                {
                    key: value
                    for key, value in step.items()
                    if key != "implementation_candidates"
                }
                if isinstance(step, dict)
                else step
            )
            prompt += f"## Step {i}\n{json.dumps(step_view, ensure_ascii=False, separators=(',', ':'))}\n\n"

        if literature_review:
            prompt += "# Relevant Literature\n"
            for i, item in enumerate(literature_review[:5], 1):
                if isinstance(item, dict):
                    title = item.get("title", "")
                    content = item.get("abstract", item.get("content", ""))
                    prompt += f"## [{i}] {title}\n{content}\n\n"

        if resource_check_feedback:
            prompt += "# Resource Check Feedback (fix required before re-check)\n"
            prompt += f"Reason: {resource_check_feedback.get('reason', '')}\n"
            prompt += f"Recovery hint: {resource_check_feedback.get('recovery_hint', '')}\n"
            prompt += f"Recoverable: {resource_check_feedback.get('recoverable', False)}\n"
            missing = resource_check_feedback.get("missing_datasets", [])
            if missing:
                prompt += f"Missing datasets: {missing}\n"
            tools = resource_check_feedback.get("unavailable_tools", [])
            if tools:
                prompt += f"Unavailable tools: {tools}\n"
            steps = resource_check_feedback.get("affected_steps", [])
            if steps:
                prompt += f"Affected steps: {steps}\n"

            # Commit 5: 消费 Commit 4 结构化诊断
            step_failures = resource_check_feedback.get("step_failures", [])
            failure_class = resource_check_feedback.get("failure_class", "")
            weakest_step_id = resource_check_feedback.get("weakest_step_id")
            suggested_tt = resource_check_feedback.get("suggested_task_type", "")
            suggested_conv = resource_check_feedback.get("suggested_conversion_task", "")

            if step_failures:
                prompt += "\n## Step-Level Failure Diagnostics\n"
                prompt += f"Failure class: {failure_class}\n"
                if weakest_step_id is not None:
                    prompt += f"Weakest step ID: {weakest_step_id}\n"
                prompt += "\nFailed steps:\n"
                for sf in step_failures:
                    prompt += (
                        f"- Step {sf.get('step_id', '?')} ({sf.get('step_name', '')}): "
                        f"{sf.get('failure_type', '')} — {sf.get('detail', '')}\n"
                    )
                    gap = sf.get("gap_type", "")
                    if gap:
                        prompt += f"  gap_type: {gap}\n"
                    stt = sf.get("suggested_task_type", "")
                    if stt:
                        prompt += f"  suggested_task_type: {stt}\n"
                prompt += "\n"

            dataset_resolution_failures = resource_check_feedback.get("dataset_resolution_failures", [])
            if dataset_resolution_failures:
                prompt += "## Dataset Resolution Failures\n"
                prompt += (
                    "For each failed dataset requirement, you MUST reason from these exact fields: "
                    "binding_status, blocking_reasons, best_candidate, search_stage, online_attempted.\n"
                )
                for failure in dataset_resolution_failures:
                    prompt += (
                        f"- {failure.get('requirement_id', 'unknown')} ({failure.get('dataset_name', '')})\n"
                        f"  binding_status: {failure.get('binding_status', '')}\n"
                        f"  blocking_reasons: {failure.get('blocking_reasons', [])}\n"
                        f"  best_candidate: {json.dumps(failure.get('best_candidate'), ensure_ascii=False)}\n"
                        f"  search_stage: {failure.get('search_stage', '')}\n"
                        f"  online_attempted: {failure.get('online_attempted', False)}\n"
                    )
                prompt += "\n"

            if suggested_tt:
                prompt += f"Suggested alternative task_type: {suggested_tt}\n"
            if suggested_conv:
                prompt += f"Suggested conversion: {suggested_conv}\n"
            prompt += "\n"

        # Commit 5: 展示 execution_contract 的结构化验证结果（无论成功或失败路径）
        if execution_contract:
            validations = execution_contract.get("step_tool_validations", [])
            score = execution_contract.get("tool_reachability_score", 1.0)
            compatibility_passed = execution_contract.get(
                "implementation_compatibility_passed",
                False,
            )
            weakest_step_id = execution_contract.get("weakest_step_id")
            plan_dag_valid = execution_contract["plan_dag_valid"]
            plan_dag_issues = execution_contract["plan_dag_issues"]
            manifest = execution_contract.get("data_access_manifest", [])

            prompt += "# Resource Check Validation Summary\n"
            prompt += f"Plan-level tool reachability score (min of all steps): {score}\n"
            prompt += f"Selected implementations operation-compatible: {compatibility_passed}\n"
            if weakest_step_id is not None:
                prompt += f"Weakest step ID: {weakest_step_id}\n"
            prompt += f"Plan DAG valid: {plan_dag_valid}\n\n"

            if validations:
                prompt += "## Step Tool Validation Scores\n"
                for v in validations:
                    prompt += (
                        f"- Step {v.get('step_id', '?')}: "
                        f"score={v.get('score', '?')}, "
                        f"level={v.get('validation_level', '?')}, "
                        f"fn={v.get('function_name', 'N/A')}, "
                        f"pkg={v.get('package', 'N/A')}"
                    )
                    gap = v.get("gap_type", "")
                    if gap:
                        prompt += f", gap={gap}"
                    prompt += (
                        f", operation_compatible={v.get('operation_compatible', False)}, "
                        f"compatibility_status={v.get('operation_compatibility_status', '')}"
                    )
                    prompt += "\n"
                prompt += "\n"

            if plan_dag_issues:
                prompt += "## Plan DAG Issues\n"
                for issue in plan_dag_issues:
                    prompt += f"- {json.dumps(issue, ensure_ascii=False)}\n"
                prompt += "\n"

            if manifest:
                mandatory_unbound = [
                    item for item in manifest
                    if item.get("mandatory", False) and item.get("bind_status") != "bindable"
                ]
                bindable_items = [
                    item for item in manifest
                    if item.get("bind_status") == "bindable"
                ]

                if mandatory_unbound:
                    prompt += "## Missing Mandatory Bindings\n"
                    for item in mandatory_unbound:
                        prompt += (
                            f"- {item.get('requirement_id', 'unknown')}: "
                            f"source={item.get('source', 'N/A')}, "
                            f"title={item.get('dataset_title', item.get('name', 'N/A'))}\n"
                        )
                    prompt += "\n"

                if bindable_items:
                    prompt += "## Already Bound Datasets\n"
                    for item in bindable_items:
                        prompt += (
                            f"- {item.get('requirement_id', 'unknown')}: "
                            f"source={item.get('source', 'N/A')}, "
                            f"title={item.get('dataset_title', item.get('name', 'N/A'))}\n"
                        )
                    prompt += "\n"

        prompt += """# Idea-Authoritative Data Sufficiency Review
First classify the accepted idea from its own semantics:
- Base-data-only: the provided base dataset is sufficient to test the core hypothesis,
  mechanism-relevant comparison, and falsification condition.
- External-signal-dependent: the accepted idea explicitly depends on one or more external
  signals beyond the provided base dataset.

Apply these rules:
1. `required_datasets: []` is valid for a base-data-only idea when the task plan can test the
   core hypothesis, required comparison, and falsification condition with the base dataset.
2. Never require driver, exposure, impact, validation, or any other auxiliary dataset merely
   to increase the number of data sources or make the plan look more discovery-oriented.
3. For an external-signal-dependent idea, check that each external signal explicitly required
   by the accepted idea, or logically necessary to operationalize its fixed hypothesis,
   mechanism, comparison, observable, or falsification condition, appears in
   `required_datasets` and is operationalized by task-plan steps through `inputs` entries
   whose `source_kind` is `required_dataset`.
4. Candidate datasets and evidence records in the independent audit are non-binding evidence.
   They are not `required_datasets` unless the accepted idea explicitly depends on them.
5. A plan may add operational bridge variables, preprocessing datasets, controls, validation
   references, or intermediate indicators when they are necessary to make the accepted idea
   executable and do not change the idea's scientific claim, mechanism, study object,
   target observable, comparison, or falsification condition.
6. An external requirement that is neither explicitly idea-dependent nor an allowed
   operational bridge is a plan-contract issue; ask TASK_PLANNING to remove or correct it.
   A missing idea-dependent or logically necessary bridge signal is also a plan issue; ask
   TASK_PLANNING to add and operationalize the requirement.

Only an actual plan/data-contract mismatch should trigger `mode=replan_feedback` and
`should_replan=true`. Reuse `feedback_summary`, `step_level_adjustments`, and
`dataset_actions`; do not invent new fields and do not propose changes to the accepted idea.

"""

        prompt += """# Semantic Alignment Check
After confirming executability and data sufficiency, check whether task_plan faithfully
operationalizes the accepted idea. The idea is the semantic authority; the plan is the only
object that may be changed by this review.

Populate `semantic_alignment_issues` ONLY when ALL of the following are true:
1. The task_plan is already executable (resource check passed or resource issues are separately handled).
2. The task_plan omits, contradicts, or exceeds semantics already fixed by the accepted idea.
3. The issue is NOT a missing dataset, unavailable tool, or step execution failure
   (those belong in step_level_adjustments / dataset_actions).

Three issue types:
- `plan_mechanism_mismatch`: task_plan uses a mechanism or analytical path that contradicts
  the mechanism fixed by the accepted idea.
- `missing_idea_operationalization`: task_plan omits a hypothesis, comparison, observable,
  or falsification condition fixed by the accepted idea.
- `plan_scope_overreach`: task_plan introduces claims, mechanisms, study objects, target
  observables, comparisons, or falsification conditions beyond the accepted idea.

Do NOT mark a plan as unfaithful merely because it adds operational bridge steps or datasets
needed to execute a top-level idea. Treat these additions as allowed when they only connect,
preprocess, control, validate, or derive intermediate variables for the accepted idea and do
not introduce a new scientific claim or change the idea semantics listed above.

For each issue, specify:
- `affected_step_ids`: task-plan step IDs that require correction
- `severity`: high | medium | low
- `reason`: why the plan is not faithful to the accepted idea
- `evidence`: cite the accepted idea semantics and specific task-plan steps
- `plan_fix`: tell TASK_PLANNING how to adjust steps, data requirements, or tools
- `repair_constraints`: the machine-checkable required operation, preferred/forbidden
  function names, and required input/output semantics that the repaired plan must satisfy

If no semantic alignment issues exist, output an empty array `[]` for `semantic_alignment_issues`.
Never request an idea-field edit or make the idea conform to the current plan.

"""

        if resource_check_feedback:
            prompt += """# Task
Review the current task plan and produce a **targeted** replan feedback report for TASK_PLANNING.

Requirements:
1. Do NOT rewrite the task plan directly — TASK_PLANNING is the sole plan author.
2. Prioritize fixing the missing mandatory bindings first. Use the provided missing mandatory dataset list as the primary replan target.
3. Preserve already bound datasets whenever possible. Do not replace a bound dataset unless there is a clear blocking reason.
4. For each failed step (score < 0.6 OR operation_compatible=false), output a specific `step_level_adjustments` entry with:
   - `step_id`: the failing step
   - `action`: one of replace_tool, replace_function, change_task_type, add_conversion_step, remove_step
   - `reason`: why this specific fix is needed
   - `suggested_replacement`: concrete alternative (tool name, function, or task_type)
   - `repair_constraints`: REQUIRED for every non-keep step. `required_operation` MUST be a
     short non-empty phrase describing the key operation the repaired step must perform
     (e.g. `zonal statistics`, `point sampling`, `Pearson correlation`). Even if you have no
     specific preferred/forbidden functions or semantic requirements, you MUST still fill in
     `required_operation`. Use empty arrays only for list dimensions that genuinely do not apply.
5. For steps that passed both checks (score >= 0.6 AND operation_compatible=true), use action=keep — do NOT suggest changes.
6. Distinguish gap types:
   - coverage_gap (task_type valid but function not found): suggest replace_function or replace_tool
   - capability_gap (task_type not in closed set): suggest change_task_type
7. If the plan is executable but violates the idea-authoritative sufficiency rules above, still output `replan_feedback` with `should_replan=true`, and use `dataset_actions` to identify missing, unjustified, or incorrectly operationalized external requirements.
8. Preserve the original research goal and all valid steps.
9. Set mode to `replan_feedback` and should_replan to true.
10. When dataset resolution failed, explicitly use binding_status, blocking_reasons, best_candidate, search_stage, and online_attempted to explain how TASK_PLANNING should rewrite the affected requirement.
11. In recoverable dataset semantic rejection cases (for example `failure_class=semantic_rejection_unbound`), do NOT terminate or modify the idea. Use `dataset_actions` to explain the required correction for TASK_PLANNING.
"""
        else:
            prompt += """# Task
Review the current task plan after resource checking has passed.

Requirements:
1. Confirm executability, idea fidelity, and data sufficiency under the idea-authoritative rules above.
2. If the task plan or data contract does not operationalize the accepted idea, output `mode=replan_feedback` and `should_replan=true` even though resource checking passed.
3. Use `dataset_actions` only for external signals the idea explicitly requires, logically necessary bridge signals, or requirements the plan added without idea or bridge justification.
4. For each step, output a step_level_adjustments entry with action=keep if it passed and does not need structural reframing.
5. Output `mode=final_review` and `should_replan=false` when the plan is executable and faithful to the idea. This includes sufficient base-data-only plans with `required_datasets: []`.
"""
        prompt += """
# Structured Action Contract
An empty `dataset_actions` or `tool_actions` list is valid when no corresponding issue exists.
Every item that is present in either list must contain:
- `action`: a non-empty instruction describing what TASK_PLANNING should change.
- `reason`: a non-empty explanation grounded in the accepted idea, current plan, or resource diagnostics.
Optional identifiers such as dataset_id, affected_step_ids, step_id, tool, or suggested_replacement
may be included when they help TASK_PLANNING locate the problem.

Every `step_level_adjustments` and `semantic_alignment_issues` item must include `repair_constraints`.
For every non-keep step, `required_operation` MUST be a short non-empty phrase describing what the
repaired step must do (e.g. `independent group comparison`, `Pearson correlation`, `zonal statistics`,
`point sampling`). A replacement suggestion is advisory prose unless it is also copied into
`preferred_function_names`; the currently rejected function belongs in `forbidden_function_names`.
Use empty arrays for list dimensions that genuinely do not apply, but `required_operation` must
never be empty for non-keep steps.
`issue_id` is assigned deterministically by the runtime after generation; do not spend output
tokens inventing one.
"""
        return prompt

    def _build_plan_refinement_system_prompt(self) -> str:
        return (
            "You are a geospatial analysis plan reviewer. "
            "Your job is to critically examine a task execution plan and produce either a final review "
            "or a structured replan feedback report for TASK_PLANNING. "
            "You receive step-level tool validation scores and failure diagnostics. "
            "When producing replan_feedback, focus on TARGETED per-step fixes — "
            "do not suggest rewriting the entire plan. Only flag steps that actually failed "
            "reachability (score < 0.6) or operation compatibility, and leave steps that pass "
            "both checks untouched (action=keep). "
            "Distinguish coverage_gap (function not found for valid task_type) from "
            "capability_gap (task_type itself not in the closed set). "
            "Treat the accepted idea as immutable semantic authority. After executability checks, "
            "verify that the plan operationalizes the idea; when it does not, populate plan-side "
            "semantic_alignment_issues for TASK_PLANNING. Keep execution issues (missing datasets, "
            "unavailable tools, step failures) strictly out of semantic_alignment_issues. "
            "A sufficient base-data-only plan is valid and must not be forced to add external data. "
            "Return valid JSON matching the required schema. "
            "Never rewrite the task plan directly — TASK_PLANNING is the sole plan author."
        )

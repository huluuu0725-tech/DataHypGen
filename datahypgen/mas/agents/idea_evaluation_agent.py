"""Evidence-grounded seven-dimension idea and planned-quality evaluation."""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datahypgen.prompts import (
    IDEA_QUALITY_DIMENSION_RUBRICS_LEGACY,
    IDEA_QUALITY_EVALUATION_SYSTEM_PROMPT_LEGACY,
    IDEA_QUALITY_SUBCRITERION_ANCHOR_OVERRIDES,
    build_idea_quality_dimension_prompt_legacy,
    build_idea_quality_stage_guidance,
)

from ..workflow.data_type import (
    IDEA_QUALITY_CORE_DIMENSIONS,
    IDEA_QUALITY_DIMENSIONS,
    IDEA_QUALITY_EVALUATION_VERSION,
    IdeaQualityAggregates,
    IdeaQualityDimensionResult,
    IdeaQualityEvaluationResult,
)
from .base_agent import AgentExecutionError, BaseAgent


logger = logging.getLogger(__name__)

# Skill 文件在进程生命周期内不变：按 skill_name 缓存解析后的 system prompt，
# 文件缺失/解析失败回退 legacy prompt 的结果同样缓存，warning 只打一次。
_SKILL_PROMPT_CACHE: Dict[str, str] = {}

# 只剥离文件开头的 YAML frontmatter。正文中的 Markdown 表格分隔行（|---|---|）同样
# 含有 "---" 子串，按 split("---") 取 parts[2] 会在第一个表格处截断 system prompt。
_SKILL_FRONTMATTER_RE = re.compile(r"\A﻿?---[ \t]*\r?\n.*?\r?\n---[ \t]*\r?\n", re.DOTALL)

IDEA_QUALITY_SCORE_ANCHORS = {0, 25, 50, 75, 100}
IDEA_QUALITY_CORE_WEIGHTS = {
    "novelty": 0.25,
    "logical_coherence": 0.25,
    "data_grounding": 0.20,
    "realizability": 0.20,
    "significance": 0.10,
}
IDEA_QUALITY_SUBCRITERION_WEIGHTS: Dict[str, Dict[str, float]] = {
    "realizability": {
        "conceptual_consistency": 0.30,
        "observable_in_principle": 0.25,
        "technical_path_plausibility": 0.30,
        "fatal_blocker_absence": 0.15,
    },
    "novelty": {
        "closest_work_difference": 0.45,
        "non_incremental_contribution": 0.35,
        "non_obviousness": 0.20,
    },
    "data_grounding": {
        "target_observability": 0.25,
        "spatiotemporal_scope_fit": 0.25,
        "source_role_adequacy": 0.40,
        "claim_calibration": 0.10,
    },
    "logical_coherence": {
        "motivation_question_link": 0.10,
        "question_hypothesis_link": 0.30,
        "mechanism_observable_expectation_link": 0.15,
        "internal_consistency": 0.45,
    },
    "significance": {
        "problem_consequence": 0.35,
        "knowledge_gain": 0.35,
        "field_or_practice_impact": 0.30,
    },
    "validation_design_quality": {
        "hypothesis_alignment": 0.25,
        "diagnostic_contrast": 0.30,
        "metric_claim_fit": 0.25,
        "falsification_design": 0.20,
    },
    "reproducibility_readiness": {
        "validation_workflow_completeness": 0.25,
        "data_reproducibility": 0.25,
        "method_environment_reproducibility": 0.20,
        "evaluation_protocol_reproducibility": 0.30,
    },
}
IDEA_QUALITY_SOURCE_RELIABILITY = {
    "idea_field": 0.80,
    "dataset_evidence": 1.00,
    "literature_record": 0.90,
    "historical_idea": 0.85,
    "step_plan": 0.90,
    "resource_check": 1.00,
}
IDEA_QUALITY_SOURCE_TYPES = tuple(IDEA_QUALITY_SOURCE_RELIABILITY)
IDEA_QUALITY_MODEL_IDEA_FIELDS = (
    "idea_type",
    "component_usage",
    "knowledge_anchor_and_motivation",
    "data_synergy_and_gap",
    "research_question",
    "core_hypothesis",
    "theoretical_mechanism",
    "computability_and_falsifiability",
    "novelty_constraints_and_value",
)
IDEA_QUALITY_BLINDED_PROVENANCE_KEYS = {
    "actual_models_by_phase",
    "application_id",
    "cache_provenance",
    "cache_source_stage",
    "family",
    "generator",
    "generator_system",
    "idea_id",
    "model_name",
    "model_provider",
    "source_stage",
    "source_system",
    "system",
}


def _strip_blinded_provenance(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_blinded_provenance(item)
            for key, item in value.items()
            if key not in IDEA_QUALITY_BLINDED_PROVENANCE_KEYS
        }
    if isinstance(value, list):
        return [_strip_blinded_provenance(item) for item in value]
    return value


def _validate_weight_contract() -> None:
    if set(IDEA_QUALITY_CORE_WEIGHTS) != set(IDEA_QUALITY_CORE_DIMENSIONS):
        raise RuntimeError("Core weights do not match the idea quality dimensions")
    if abs(sum(IDEA_QUALITY_CORE_WEIGHTS.values()) - 1.0) > 1e-9:
        raise RuntimeError("Core weights must sum to 1.0")
    if set(IDEA_QUALITY_SUBCRITERION_WEIGHTS) != set(IDEA_QUALITY_DIMENSIONS):
        raise RuntimeError("Idea quality dimension weights do not match the result contract")
    for dimension, weights in IDEA_QUALITY_SUBCRITERION_WEIGHTS.items():
        if set(weights) != set(IDEA_QUALITY_DIMENSION_RUBRICS_LEGACY[dimension]["subcriteria"]):
            raise RuntimeError(f"Idea quality subcriteria disagree for {dimension}")
        if abs(sum(weights.values()) - 1.0) > 1e-9:
            raise RuntimeError(f"Idea quality weights must sum to 1.0 for {dimension}")


_validate_weight_contract()


class IdeaQualityCalculator:
    """Deterministically validates and aggregates LLM atomic judgments."""

    @classmethod
    def calculate_dimension_score(cls, dimension: str, subscores: Dict[str, Any]) -> float:
        weights = IDEA_QUALITY_SUBCRITERION_WEIGHTS.get(dimension)
        if weights is None:
            raise ValueError(f"Unsupported idea quality dimension: {dimension}")
        if set(subscores) != set(weights):
            raise ValueError(f"Subscores do not match the {dimension} contract")

        total = 0.0
        for key, weight in weights.items():
            value = subscores[key]
            if isinstance(value, bool) or value not in IDEA_QUALITY_SCORE_ANCHORS:
                raise ValueError(f"{dimension}.{key} must use one of {sorted(IDEA_QUALITY_SCORE_ANCHORS)}")
            total += float(value) * weight
        return round(total, 2)

    @classmethod
    def calculate_assessment_confidence(
        cls,
        dimension: str,
        subscores: Dict[str, Any],
        supporting_evidence: List[Dict[str, Any]],
        counterevidence_or_gaps: List[str],
    ) -> float:
        subcriteria = set(IDEA_QUALITY_SUBCRITERION_WEIGHTS[dimension])
        covered = {
            str(item.get("subcriterion", ""))
            for item in supporting_evidence
            if isinstance(item, dict) and str(item.get("subcriterion", "")) in subcriteria
        }
        evidence_coverage = len(covered) / len(subcriteria)

        reliabilities = [
            IDEA_QUALITY_SOURCE_RELIABILITY.get(str(item.get("source_type", "")), 0.0)
            for item in supporting_evidence
            if isinstance(item, dict)
        ]
        source_reliability = sum(reliabilities) / len(reliabilities) if reliabilities else 0.0

        high_score_with_unresolved_gaps = bool(counterevidence_or_gaps) and min(
            float(value) for value in subscores.values()
        ) >= 75.0
        cross_check_consistency = 0.5 if high_score_with_unresolved_gaps else 1.0

        confidence = (
            0.40 * evidence_coverage
            + 0.30 * source_reliability
            + 0.30 * cross_check_consistency
        )
        return round(max(0.0, min(1.0, confidence)), 4)

    @classmethod
    def build_dimension_result(
        cls,
        dimension: str,
        judgment: Dict[str, Any],
    ) -> IdeaQualityDimensionResult:
        status = str(judgment.get("status", "evaluated"))
        if status != "evaluated":
            if status not in {"insufficient_evidence", "not_applicable", "error"}:
                raise ValueError(f"Unsupported dimension status for {dimension}: {status}")
            return IdeaQualityDimensionResult(
                status=status,
                score=None,
                assessment_confidence=None,
                subscores={},
                supporting_evidence=[],
                counterevidence_or_gaps=[
                    str(item) for item in (judgment.get("counterevidence_or_gaps", []) or [])
                ],
                rationale=str(judgment.get("rationale", "")),
            )

        subscores = dict(judgment.get("subscores", {}) or {})
        evidence = [
            dict(item) for item in (judgment.get("supporting_evidence", []) or [])
            if isinstance(item, dict)
        ]
        gaps = [str(item) for item in (judgment.get("counterevidence_or_gaps", []) or [])]
        return IdeaQualityDimensionResult(
            status="evaluated",
            score=cls.calculate_dimension_score(dimension, subscores),
            assessment_confidence=cls.calculate_assessment_confidence(
                dimension,
                subscores,
                evidence,
                gaps,
            ),
            subscores={key: float(value) for key, value in subscores.items()},
            supporting_evidence=evidence,
            counterevidence_or_gaps=gaps,
            rationale=str(judgment.get("rationale", "")),
        )

    @classmethod
    def build_result(
        cls,
        *,
        idea_id: str,
        evaluation_stage: str,
        dimension_judgments: Dict[str, Dict[str, Any]],
    ) -> IdeaQualityEvaluationResult:
        if evaluation_stage not in {"idea_only", "planned"}:
            raise ValueError(f"Unsupported idea quality evaluation stage: {evaluation_stage}")
        if set(dimension_judgments) != set(IDEA_QUALITY_DIMENSIONS):
            raise ValueError("Idea quality result must contain exactly six dimensions")

        dimensions = {
            dimension: cls.build_dimension_result(dimension, dimension_judgments[dimension])
            for dimension in IDEA_QUALITY_DIMENSIONS
        }
        core_scores = [
            dimensions[key].score
            for key in IDEA_QUALITY_CORE_DIMENSIONS
            if dimensions[key].status == "evaluated" and dimensions[key].score is not None
        ]
        core_complete = len(core_scores) == len(IDEA_QUALITY_CORE_DIMENSIONS)
        idea_quality_score = round(sum(
            dimensions[key].score * weight
            for key, weight in IDEA_QUALITY_CORE_WEIGHTS.items()
        ), 2) if core_complete else None

        validity_scores = [
            dimensions[key].score
            for key in ("realizability", "data_grounding", "logical_coherence")
            if dimensions[key].status == "evaluated" and dimensions[key].score is not None
        ]
        minimum_core_score = min(validity_scores) if len(validity_scores) == 3 else None

        validation_design = dimensions["validation_design_quality"]
        reproducibility = dimensions["reproducibility_readiness"]
        planned_validation_score = None
        if (
            evaluation_stage == "planned"
            and validation_design.status == "evaluated"
            and validation_design.score is not None
        ):
            planned_validation_score = validation_design.score

        extended_quality_score = None
        if (
            evaluation_stage == "planned"
            and idea_quality_score is not None
            and validation_design.status == "evaluated"
            and validation_design.score is not None
            and reproducibility.status == "evaluated"
            and reproducibility.score is not None
        ):
            extended_quality_score = round(
                0.80 * idea_quality_score
                + 0.10 * validation_design.score
                + 0.10 * reproducibility.score,
                2,
            )

        planned_complete = (
            evaluation_stage == "idea_only"
            or (
                validation_design.status == "evaluated"
                and reproducibility.status == "evaluated"
            )
        )
        evaluation_status = "complete" if core_complete and planned_complete else "partial"
        evaluated_scores = {
            key: result.score
            for key, result in dimensions.items()
            if result.status == "evaluated" and result.score is not None
        }
        dominant_strengths = [
            key for key, value in evaluated_scores.items() if value >= 75.0
        ]
        dominant_weaknesses = [
            key for key, result in dimensions.items()
            if result.status == "insufficient_evidence"
            or (result.status == "evaluated" and result.score is not None and result.score <= 50.0)
        ]
        if idea_quality_score is None:
            summary = "Idea quality evaluation is partial because at least one core dimension lacks sufficient evidence."
        else:
            summary = f"Idea quality score: {idea_quality_score:.2f}."
            if extended_quality_score is not None:
                summary += f" Planned-stage extended quality score: {extended_quality_score:.2f}."

        return IdeaQualityEvaluationResult(
            idea_id=idea_id,
            evaluation_stage=evaluation_stage,
            evaluation_status=evaluation_status,
            dimensions=dimensions,
            aggregates=IdeaQualityAggregates(
                idea_quality_score=idea_quality_score,
                planned_validation_score=planned_validation_score,
                extended_quality_score=extended_quality_score,
                minimum_core_score=minimum_core_score,
            ),
            dominant_strengths=dominant_strengths,
            dominant_weaknesses=dominant_weaknesses,
            summary=summary,
        )


class IdeaEvaluationAgent(BaseAgent):
    """Produce isolated, evidence-grounded atomic judgments for idea and planned dimensions."""

    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        application = context.get("application")
        if not isinstance(application, dict) or not application:
            raise AgentExecutionError("application is required for idea quality evaluation")

        source_idea_snapshot = self._extract_idea_candidate(application)
        if not source_idea_snapshot:
            raise AgentExecutionError("application.idea_candidate is required for idea quality evaluation")
        idea_snapshot = self._blind_idea_snapshot(source_idea_snapshot)

        evidence_pack = self._build_evidence_pack(context, idea_snapshot)
        evaluation_stage = evidence_pack["evaluation_stage"]

        judgments: Dict[str, Dict[str, Any]] = {}
        for dimension in IDEA_QUALITY_DIMENSIONS:
            assessable, reason = self._is_dimension_assessable(
                dimension,
                idea_snapshot,
                evidence_pack,
            )
            if not assessable:
                status = "insufficient_evidence"
                if dimension in {"validation_design_quality", "reproducibility_readiness"}:
                    if evaluation_stage == "idea_only":
                        status = "not_applicable"
                judgments[dimension] = {
                    "status": status,
                    "counterevidence_or_gaps": [reason],
                    "rationale": reason,
                }
                continue

            # Skill-based evaluation: build isolated evidence pack
            skill_evidence = self._build_evidence_for_skill(dimension, idea_snapshot, evidence_pack)
            response = await self._call_model(
                prompt=self._build_skill_dimension_prompt(dimension, skill_evidence),
                system_prompt=self._load_skill_system_prompt(dimension),
                schema=self._build_dimension_output_schema(dimension),
                temperature=0.1,
            )

            if not isinstance(response, dict):
                raise AgentExecutionError(f"{dimension} evaluator must return a JSON object")

            self._normalize_evidence_source_types(dimension, response)
            self._validate_dimension_response(dimension, response)
            judgments[dimension] = {"status": "evaluated", **response}

        return {
            "evaluation_version": IDEA_QUALITY_EVALUATION_VERSION,
            "idea_id": str(source_idea_snapshot.get("idea_id") or application.get("idea_id") or ""),
            "evaluation_stage": evaluation_stage,
            "dimension_judgments": judgments,
        }

    @staticmethod
    def _extract_idea_candidate(application: Dict[str, Any]) -> Dict[str, Any]:
        idea = application.get("idea_candidate")
        if isinstance(idea, dict) and idea:
            return idea
        required = {
            "data_synergy_and_gap",
            "research_question",
            "core_hypothesis",
            "theoretical_mechanism",
            "computability_and_falsifiability",
        }
        if required <= set(application):
            return application
        return {}

    @staticmethod
    def _blind_idea_snapshot(idea_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: idea_snapshot[key]
            for key in IDEA_QUALITY_MODEL_IDEA_FIELDS
            if idea_snapshot.get(key) not in (None, "", [], {})
        }

    @staticmethod
    def _compact_dataset_description(dataset_info: Any) -> Dict[str, Any]:
        if not isinstance(dataset_info, dict):
            return {}
        payload = dataset_info
        base_dataset = payload.get("base_dataset") if isinstance(payload.get("base_dataset"), dict) else {}
        is_composite = base_dataset.get("type") == "composite_dataset"
        if isinstance(payload.get("base_dataset"), dict):
            payload = payload["base_dataset"]
        if isinstance(payload.get("data_description"), dict):
            payload = payload["data_description"]
        if not isinstance(payload, dict):
            return {}

        result = {
            key: payload[key]
            for key in (
                "dataset_name",
                "description",
                "data_type",
                "spatial_resolution",
                "temporal_coverage",
                "temporal_unit",
                "spatial_coverage",
                "analytical_capabilities",
                "unit_of_measurement",
                "important_note",
            )
            if payload.get(key) not in (None, "", [], {})
        }
        raster_content = payload.get("raster_content")
        if isinstance(raster_content, dict):
            result["raster_content"] = {
                key: raster_content[key]
                for key in (
                    "band_names",
                    "first_band_names",
                    "band_semantics",
                    "collection_size",
                    "gee_asset_id",
                    "thematic_group",
                )
                if raster_content.get(key) not in (None, "", [], {})
            }
        if is_composite:
            result["anchor_asset_type"] = "composite_observation"
            result["components"] = [
                {
                    "role": component.get("role"),
                    "dataset_name": component.get("name"),
                    "source": component.get("source") or component.get("data_path"),
                    "data_description": component.get("data_description", {}),
                }
                for component in (base_dataset.get("dataset_components") or [])
                if isinstance(component, dict)
            ]
            contract = base_dataset.get("combination_contract")
            if isinstance(contract, dict) and contract:
                result["combination_contract"] = contract
        return result

    @staticmethod
    def _compact_dataset_portrait(dataset_portrait: Any) -> Dict[str, Any]:
        """只保留 portrait 中评估维度真正依赖的部分。

        完整 portrait 序列化后约 23KB（历史 indent=2 格式测得），注入 realizability 与 data_grounding
        两个维度，而它与同一个 dataset_evidence 里的 `task_dataset_description` 大量重复：
        `source_description` 就是原始 data_description 的完整副本，`enriched_data_description`
        又是它的超集，于是同一份 data_description 在一个 prompt 里出现三次。

        保留项的依据：`data_grounding` 的 `target_observability` 子标准判断"目标变量、代理、
        标签是否可观测或可信地构造"，`dataset_features` 的 {feature, evidence_field,
        availability} 结构正是数据集可观测能力的权威声明；`candidate_question_axes` 与
        `suggested_research_hooks` 是画像对可研究方向的判断，与 realizability 相关且体量很小。
        `spatiotemporal_scope_fit` 需要的覆盖范围、分辨率、时间语义由
        `_compact_dataset_description` 提供，不必再从 portrait 取一份。

        排除项：`source_description` / `enriched_data_description`（与 compact 描述重复，
        后者内部还嵌套一份 dataset_features）、`analysis`（下游零消费，信息已被 portrait_text
        摘要）、`profile_quality`（除 status/quality_tier 外全是 resolved_path_count 一类
        执行计量）、`portrait_text`（把结构化字段重述成散文，逐字包含 dataset_features 全部条目）。
        """
        if not isinstance(dataset_portrait, dict):
            return {}
        return {
            key: dataset_portrait[key]
            for key in (
                "dataset_type",
                "dataset_features",
                "candidate_question_axes",
                "suggested_research_hooks",
            )
            if dataset_portrait.get(key) not in (None, "", [], {})
        }

    @staticmethod
    def _summarize_literature(items: Any, max_items: int = 8) -> List[Dict[str, str]]:
        if not isinstance(items, list):
            return []
        result: List[Dict[str, str]] = []
        for item in items[:max_items]:
            if not isinstance(item, dict):
                continue
            result.append(
                {
                    "id": str(item.get("id") or item.get("doi") or item.get("url") or "")[:240],
                    "title": str(item.get("title") or item.get("paper_title") or "")[:400],
                    "summary": str(
                        item.get("summary")
                        or item.get("abstract")
                        or item.get("key_content")
                        or item.get("snippet")
                        or ""
                    )[:1600],
                    "source": str(item.get("source") or item.get("venue") or "")[:200],
                }
            )
        return result

    @staticmethod
    def _summarize_step_plan(step_plan: Any, max_steps: int = 12) -> List[Dict[str, Any]]:
        if not isinstance(step_plan, list):
            return []
        fields = {
            "id",
            "description",
            "task",
            "inputs",
            "outputs",
            "method",
            "tool",
            "parameters",
            "validation",
            "evaluation",
            "success_criteria",
        }
        summarized = [
            {key: value for key, value in step.items() if key in fields}
            for step in step_plan[:max_steps]
            if isinstance(step, dict)
        ]
        return [step for step in summarized if step]

    @staticmethod
    def _summarize_resource_check(resource_check: Any) -> Dict[str, Any]:
        if not isinstance(resource_check, dict):
            return {}
        execution_contract = resource_check.get("execution_contract", {})
        if not isinstance(execution_contract, dict):
            execution_contract = {}
        # 只保留资源可达性事实（数据集/工具是否实际可达），去掉上游模块的过程裁决字段
        # （resource_check_passed / should_terminate / terminal_rule 是 ResourceCheck 的总结论，
        # 不应注入评估证据——评估方应基于 idea 和 plan 内容自行得出结论）
        summary = {
            "data_access_manifest": list(execution_contract.get("data_access_manifest", []) or [])[:12],
            "step_tool_validations": list(execution_contract.get("step_tool_validations", []) or [])[:12],
            "guaranteed_datasets": list(resource_check.get("guaranteed_datasets", []) or [])[:12],
            "missing_datasets": list(resource_check.get("missing_datasets", []) or [])[:12],
            "unavailable_tools": list(execution_contract.get("unavailable_tools", []) or [])[:12],
        }
        dataset_bindings = IdeaEvaluationAgent._summarize_resource_check_dataset_bindings(
            resource_check
        )
        if dataset_bindings is not None:
            summary["dataset_bindings"] = dataset_bindings
        if "plan_dag_valid" in execution_contract:
            summary["plan_dag_valid"] = execution_contract["plan_dag_valid"]
            summary["plan_dag_issues"] = list(execution_contract["plan_dag_issues"])[:12]
        if not any(value not in (None, "", [], {}) for value in summary.values()):
            return {}
        return summary

    @staticmethod
    def _extract_confirmed_auxiliary_datasets(
        resource_check: Any,
    ) -> List[Dict[str, Any]]:
        """Extract descriptions of auxiliary datasets that were successfully resolved.

        For data_grounding assessment: if the idea needs precipitation, soil moisture, etc.,
        and these were confirmed available in resource_check, include them so
        Dr. Garcia knows they're available for the analysis.
        """
        if not isinstance(resource_check, dict):
            return []

        dataset_info = resource_check.get("dataset_info")
        if not isinstance(dataset_info, dict):
            return []

        required_datasets = dataset_info.get("required_datasets")
        if not isinstance(required_datasets, list):
            return []

        confirmed: List[Dict[str, Any]] = []
        for req in required_datasets:
            if not isinstance(req, dict):
                continue

            # Only include if successfully resolved
            if req.get("resolution_status") not in ("resolved_strong", "resolved_weak"):
                continue

            resolved_dataset = req.get("resolved_dataset")
            if not isinstance(resolved_dataset, dict):
                continue

            # Build a compact description for data_grounding evidence
            auxiliary_desc = {
                "requirement_name": req.get("canonical_requirement_name") or req.get("name"),
                "research_justification": req.get("research_justification"),
                "dataset_title": resolved_dataset.get("title"),
                "dataset_name": resolved_dataset.get("name"),
                "type": resolved_dataset.get("type"),
                "source": resolved_dataset.get("source"),
                "provider": resolved_dataset.get("provider"),
                "spatial_resolution": req.get("spatial_resolution"),
                "temporal_resolution": req.get("temporal_resolution"),
                "time_range": req.get("time_range"),
            }
            # Filter out None/empty values
            auxiliary_desc = {
                k: v for k, v in auxiliary_desc.items()
                if v not in (None, "", [], {})
            }

            if auxiliary_desc:
                confirmed.append(auxiliary_desc)

        return confirmed

    @staticmethod
    def _summarize_declared_dataset_requirements(
        application: Any,
    ) -> Optional[List[Dict[str, Any]]]:
        """Extract scientific data roles without ResourceCheck availability verdicts."""
        if not isinstance(application, dict):
            return None
        required_datasets = application.get("required_datasets")
        if not isinstance(required_datasets, list):
            return None

        requirements: List[Dict[str, Any]] = []
        for requirement in required_datasets[:12]:
            if not isinstance(requirement, dict):
                continue
            requirements.append({
                key: requirement[key]
                for key in (
                    "id",
                    "name",
                    "canonical_requirement_name",
                    "research_justification",
                    "execution_requirement_level",
                    "modality",
                    "source_preference",
                    "preferred_family",
                    "allowed_product_families",
                    "spatial_resolution",
                    "temporal_resolution",
                    "temporal_requirement_level",
                    "temporal_coverage",
                    "time_range",
                )
                if requirement.get(key) not in (None, "", [], {})
            })
        return requirements

    @staticmethod
    def _summarize_resource_check_dataset_bindings(
        resource_check: Any,
    ) -> Optional[List[Dict[str, Any]]]:
        """Extract the current application's final auxiliary-dataset binding verdicts."""
        if not isinstance(resource_check, dict):
            return None
        dataset_info = resource_check.get("dataset_info")
        if not isinstance(dataset_info, dict):
            return None
        required_datasets = dataset_info.get("required_datasets")
        if not isinstance(required_datasets, list):
            return None

        bindings: List[Dict[str, Any]] = []
        for requirement in required_datasets[:12]:
            if not isinstance(requirement, dict):
                continue
            binding = {
                key: requirement[key]
                for key in (
                    "id",
                    "name",
                    "canonical_requirement_name",
                    "research_justification",
                    "execution_requirement_level",
                    "modality",
                    "spatial_resolution",
                    "temporal_resolution",
                    "time_range",
                    "resolution_status",
                )
                if requirement.get(key) not in (None, "", [], {})
            }
            if "resolution_status" not in binding:
                binding["resolution_status"] = "missing"

            resolved_dataset = requirement.get("resolved_dataset")
            if isinstance(resolved_dataset, dict):
                compact_resolved = {
                    key: resolved_dataset[key]
                    for key in (
                        "id",
                        "title",
                        "name",
                        "type",
                        "source",
                        "provider",
                        "binding_status",
                        "bind_status_reason",
                        "resolution_source",
                        "constraint_report",
                    )
                    if resolved_dataset.get(key) not in (None, "", [], {})
                }
                if compact_resolved:
                    binding["resolved_dataset"] = compact_resolved

            failure = requirement.get("dataset_resolution_failure")
            if isinstance(failure, dict):
                compact_failure = {
                    key: failure[key]
                    for key in (
                        "binding_status",
                        "bind_status_reason",
                        "resolution_source",
                        "blocking_reasons",
                        "failure_status",
                        "unbound_class",
                        "best_candidate",
                    )
                    if failure.get(key) not in (None, "", [], {})
                }
                if compact_failure:
                    binding["binding_failure"] = compact_failure
            bindings.append(binding)
        return bindings

    def _build_evidence_pack(
        self,
        context: Dict[str, Any],
        idea_snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        raw_step_plan = context.get("task_plan")
        raw_resource_check = context.get("resource_check")
        step_present = raw_step_plan not in (None, [], {})
        resource_present = raw_resource_check not in (None, [], {})
        evaluation_stage = "planned" if step_present and resource_present else "idea_only"

        task_dataset_description = context.get("task_dataset_description")
        if isinstance(task_dataset_description, dict) and task_dataset_description:
            dataset_evidence = {"task_dataset_description": task_dataset_description}
        else:
            task_dataset_description = self._compact_dataset_description(context.get("dataset_info", {}))
            dataset_evidence = {
                "task_dataset_description": task_dataset_description,
                "dataset_portrait": self._compact_dataset_portrait(
                    context.get("dataset_portrait")
                ),
            }
        data_grounding_evidence = dict(dataset_evidence)
        declared_requirements = self._summarize_declared_dataset_requirements(
            context.get("application")
        )
        if evaluation_stage != "planned":
            declared_requirements = None
        elif declared_requirements is not None:
            data_grounding_evidence["declared_dataset_requirements"] = declared_requirements

        # data_grounding 只接收科学需求，不接收 ResourceCheck 绑定或访问结论。

        shared_novelty_evidence = context.get("novelty_evidence", {})
        if not isinstance(shared_novelty_evidence, dict):
            shared_novelty_evidence = {}
        literature_neighbors = self._summarize_literature(context.get("literature_review", []))
        historical_neighbors = self._summarize_literature(context.get("historical_idea_neighbors", []))
        raw_literature = context.get("literature_review", [])
        queries = sorted(
            {
                str(item.get("query") or item.get("search_query") or "").strip()
                for item in raw_literature
                if isinstance(item, dict) and str(item.get("query") or item.get("search_query") or "").strip()
            }
        ) if isinstance(raw_literature, list) else []
        retrieved_at = ""
        if isinstance(raw_literature, list):
            retrieved_at = max(
                (
                    str(item.get("retrieved_at") or item.get("timestamp") or "")
                    for item in raw_literature
                    if isinstance(item, dict)
                ),
                default="",
            )
        falsifiability = idea_snapshot.get("computability_and_falsifiability", {})
        return {
            "evaluation_stage": evaluation_stage,
            "dataset_evidence": dataset_evidence,
            "data_grounding_evidence": data_grounding_evidence,
            "declared_dataset_requirements": declared_requirements,
            "novelty_evidence": self._normalize_shared_novelty_evidence(
                shared_novelty_evidence=shared_novelty_evidence,
                fallback_queries=queries,
                fallback_retrieved_at=retrieved_at,
                fallback_literature_neighbors=literature_neighbors,
                fallback_historical_neighbors=historical_neighbors,
            ),
            "structural_facts": {
                "present_idea_fields": sorted(key for key, value in idea_snapshot.items() if value not in (None, "", [], {})),
                "falsifiability_fields": sorted(falsifiability) if isinstance(falsifiability, dict) else [],
            },
            "planned_evidence": {
                "step_plan": self._summarize_step_plan(raw_step_plan),
                "resource_check": self._summarize_resource_check(raw_resource_check),
                "raw_types_valid": isinstance(raw_step_plan, list) and isinstance(raw_resource_check, dict),
            },
        }

    def _build_evidence_for_skill(
        self,
        dimension: str,
        idea_snapshot: Dict[str, Any],
        evidence_pack: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build dimension-specific evidence pack with physical isolation.

        Each skill only receives the evidence types needed for its assessment,
        preventing cross-dimension contamination.
        """
        evaluation_stage = evidence_pack["evaluation_stage"]

        # Core idea fields - all skills see these
        skill_evidence = {
            "idea": idea_snapshot,
            "evaluation_stage": evaluation_stage,
        }

        # Dimension-specific evidence allocation
        if dimension == "novelty":
            # Dr. Sarah Chen: literature review specialist
            skill_evidence["novelty_evidence"] = evidence_pack["novelty_evidence"]
            # Does NOT see: dataset specs, planning, resource checks

        elif dimension == "realizability":
            # Dr. James Liu: technical feasibility specialist
            skill_evidence["dataset_evidence"] = evidence_pack["dataset_evidence"]
            skill_evidence["structural_facts"] = evidence_pack["structural_facts"]
            if evaluation_stage == "planned":
                skill_evidence["declared_dataset_requirements"] = evidence_pack.get("declared_dataset_requirements")
            # Does NOT see: literature reviews (already used by novelty)

        elif dimension == "significance":
            # Dr. Maria Rodriguez: scientific impact analyst
            skill_evidence["novelty_evidence"] = {
                "queries": evidence_pack["novelty_evidence"].get("queries", []),
                "literature_neighbors": evidence_pack["novelty_evidence"].get("literature_neighbors", []),
            }
            # Minimal literature context for motivation, NOT full novelty assessment
            # Does NOT see: dataset specs, technical implementation

        elif dimension == "data_grounding":
            # Dr. Elena Garcia: data-hypothesis alignment specialist
            skill_evidence["data_grounding_evidence"] = evidence_pack["data_grounding_evidence"]
            if evaluation_stage == "planned":
                # Can see declared requirements only at planned stage
                skill_evidence["declared_dataset_requirements"] = evidence_pack.get("declared_dataset_requirements")
            # Does NOT see: literature reviews, novelty assessments

        elif dimension == "logical_coherence":
            # Dr. Robert Kim: research design logic specialist
            skill_evidence["structural_facts"] = evidence_pack["structural_facts"]
            # ONLY sees idea structure, NO external evidence at all

        elif dimension == "validation_design_quality":
            # Dr. Amara Okafor: experimental design specialist (planned stage only)
            if evaluation_stage == "planned":
                skill_evidence["planned_evidence"] = {
                    "step_plan": evidence_pack["planned_evidence"]["step_plan"],
                }
                skill_evidence["structural_facts"] = evidence_pack["structural_facts"]
            # Does NOT see: literature (not available), dataset feasibility (already assessed)

        elif dimension == "reproducibility_readiness":
            # Dr. Yuki Tanaka: computational reproducibility specialist (planned stage only)
            if evaluation_stage == "planned":
                skill_evidence["planned_evidence"] = evidence_pack["planned_evidence"]
                skill_evidence["dataset_evidence"] = evidence_pack["dataset_evidence"]
            # Does NOT see: literature, novelty evidence

        return skill_evidence

    def _load_skill_system_prompt(self, dimension: str) -> str:
        """Load system prompt from skill SKILL.md file.

        Returns the legacy prompt if skill file not found (graceful degradation).
        """
        dimension_to_skill = {
            "novelty": "novelty-reviewer",
            "realizability": "realizability-reviewer",
            "significance": "significance-reviewer",
            "data_grounding": "data-grounding-reviewer",
            "logical_coherence": "logical-coherence-reviewer",
            "validation_design_quality": "validation-design-reviewer",
            "reproducibility_readiness": "reproducibility-reviewer",
        }
        skill_name = dimension_to_skill.get(dimension)
        if not skill_name:
            logger.warning(f"No skill mapping for dimension {dimension}, using legacy prompt")
            return IDEA_QUALITY_EVALUATION_SYSTEM_PROMPT

        cached = _SKILL_PROMPT_CACHE.get(skill_name)
        if cached is not None:
            return cached

        # Try to locate skill file relative to this agent file
        agent_dir = Path(__file__).parent
        skill_path = agent_dir / "skills" / skill_name / "SKILL.md"

        if not skill_path.exists():
            logger.warning(f"Skill file not found: {skill_path}, using legacy prompt")
            _SKILL_PROMPT_CACHE[skill_name] = IDEA_QUALITY_EVALUATION_SYSTEM_PROMPT_LEGACY
            return IDEA_QUALITY_EVALUATION_SYSTEM_PROMPT_LEGACY

        try:
            content = skill_path.read_text(encoding="utf-8")
            # 只剥离开头的 YAML frontmatter；没有 frontmatter 时按全文使用。
            match = _SKILL_FRONTMATTER_RE.match(content)
            system_prompt = content[match.end():].strip() if match else content.strip()
            if not system_prompt:
                raise ValueError(f"Skill file {skill_path} has no body after frontmatter")
        except Exception as e:
            logger.error(f"Failed to load skill {skill_name}: {e}, using legacy prompt")
            _SKILL_PROMPT_CACHE[skill_name] = IDEA_QUALITY_EVALUATION_SYSTEM_PROMPT_LEGACY
            return IDEA_QUALITY_EVALUATION_SYSTEM_PROMPT_LEGACY
        # skill 正文是模型看到的打分说明，输出 schema 由代码生成；两者子准则名一旦漂移，
        # 模型会在 system prompt 与 schema 之间摇摆，纠错重试也无法收敛，所以这里 fail closed。
        missing = [
            key for key in IDEA_QUALITY_SUBCRITERION_WEIGHTS[dimension]
            if not re.search(rf"\b{re.escape(key)}\b", system_prompt)
        ]
        if missing:
            raise RuntimeError(
                f"Skill {skill_name} does not define subcriteria {missing} required by the "
                f"{dimension} output schema"
            )
        _SKILL_PROMPT_CACHE[skill_name] = system_prompt
        return system_prompt

    def _build_skill_dimension_prompt(
        self,
        dimension: str,
        skill_evidence: Dict[str, Any],
    ) -> str:
        """Build dimension prompt using skill-specific evidence format.

        The skill evidence pack is already filtered by _build_evidence_for_skill(),
        so we just need to serialize it into a readable format.
        """
        idea = skill_evidence.get("idea", {})
        evaluation_stage = skill_evidence.get("evaluation_stage", "idea_only")

        # 紧凑序列化：缩进与分隔空格只消耗 token，不改变内容
        def dump(value: Any) -> str:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

        prompt_parts = ["# Research Idea to Evaluate\n"]

        # Serialize idea fields
        for key, value in idea.items():
            if value not in (None, "", [], {}):
                prompt_parts.append(f"## {key.replace('_', ' ').title()}\n{dump(value)}\n")

        prompt_parts.append(f"\n# Evaluation Stage\n{evaluation_stage}\n")
        prompt_parts.append(build_idea_quality_stage_guidance(dimension, evaluation_stage))

        # Serialize dimension-specific evidence
        prompt_parts.append("\n# Evidence Available to You\n")

        if "novelty_evidence" in skill_evidence:
            prompt_parts.append(f"## Novelty Evidence\n{dump(skill_evidence['novelty_evidence'])}\n")

        if "dataset_evidence" in skill_evidence:
            prompt_parts.append(f"## Dataset Evidence\n{dump(skill_evidence['dataset_evidence'])}\n")

        if "data_grounding_evidence" in skill_evidence:
            prompt_parts.append(f"## Data Grounding Evidence\n{dump(skill_evidence['data_grounding_evidence'])}\n")

        if "structural_facts" in skill_evidence:
            prompt_parts.append(f"## Structural Facts\n{dump(skill_evidence['structural_facts'])}\n")

        if "planned_evidence" in skill_evidence:
            prompt_parts.append(f"## Planned Evidence\n{dump(skill_evidence['planned_evidence'])}\n")

        if "declared_dataset_requirements" in skill_evidence and skill_evidence["declared_dataset_requirements"]:
            prompt_parts.append(
                f"## Declared Dataset Requirements\n{dump(skill_evidence['declared_dataset_requirements'])}\n"
            )

        # 强调必需字段；子准则名、锚点和长度上限以代码合同为准，避免模型沿用 skill 正文里的措辞变体
        subcriteria = list(IDEA_QUALITY_SUBCRITERION_WEIGHTS[dimension])
        anchors = sorted(IDEA_QUALITY_SCORE_ANCHORS)
        props = self._build_dimension_output_schema(dimension)["properties"]
        evidence_max_items = props["supporting_evidence"]["maxItems"]
        evidence_max_len = props["supporting_evidence"]["items"]["properties"]["evidence"]["maxLength"]
        gaps_max_items = props["counterevidence_or_gaps"]["maxItems"]
        gaps_max_len = props["counterevidence_or_gaps"]["items"]["maxLength"]
        rationale_max_len = props["rationale"]["maxLength"]
        prompt_parts.append(
            "\n\n# IMPORTANT: Required Output Fields\n"
            "Your JSON response MUST include ALL of the following fields. Do not omit any:\n"
            f"1. subscores: object with exactly these keys and no others: {json.dumps(subcriteria)}. "
            f"Each value must be one of the integers {anchors}; never an intermediate number, "
            "never a string.\n"
            f"2. supporting_evidence: non-empty array (at most {evidence_max_items} items); each item has "
            "`subcriterion` (one of the keys above), "
            f"`source_type` (one of {json.dumps(list(IDEA_QUALITY_SOURCE_TYPES))}), `source_field`, "
            f"`evidence` (string, at most {evidence_max_len} characters)\n"
            f"3. counterevidence_or_gaps: array of at most {gaps_max_items} plain strings, each at most "
            f"{gaps_max_len} characters; never objects (can be empty [] but field must exist)\n"
            f"4. rationale: brief overall assessment string, at most {rationale_max_len} characters "
            "(cannot be empty or null)\n"
            "\nIf any of these fields are missing, the evaluation will fail. "
            "Always provide a rationale and include counterevidence_or_gaps field even if empty. "
            "Keep every string within its limit so the whole object fits in the response budget."
        )

        return "\n".join(prompt_parts)

    @staticmethod
    def _normalize_shared_novelty_evidence(
        *,
        shared_novelty_evidence: Dict[str, Any],
        fallback_queries: List[str],
        fallback_retrieved_at: str,
        fallback_literature_neighbors: List[Dict[str, Any]],
        fallback_historical_neighbors: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if shared_novelty_evidence:
            literature_neighbors = shared_novelty_evidence.get("literature_neighbors", [])
            historical_neighbors = shared_novelty_evidence.get("historical_idea_neighbors", [])
            result: Dict[str, Any] = {
                "queries": shared_novelty_evidence.get("queries", []),
                "retrieved_at": shared_novelty_evidence.get("retrieved_at", ""),
                "literature_neighbors": literature_neighbors if isinstance(literature_neighbors, list) else [],
                "coverage_status": shared_novelty_evidence.get("coverage_status", "insufficient"),
                "retrieval_source": shared_novelty_evidence.get("retrieval_source", "shared_evaluator"),
                "sources": shared_novelty_evidence.get("sources", []),
                "errors": shared_novelty_evidence.get("errors", []),
            }
            # Only expose historical_idea_neighbors when non-empty so the LLM
            # evaluator does not treat an architecture-level absence as a
            # novelty coverage gap.
            hist = historical_neighbors if isinstance(historical_neighbors, list) else []
            if hist:
                result["historical_idea_neighbors"] = hist
            return _strip_blinded_provenance(result)

        result = {
            "queries": fallback_queries,
            "retrieved_at": fallback_retrieved_at,
            "literature_neighbors": fallback_literature_neighbors,
            "coverage_status": (
                "sufficient"
                if fallback_literature_neighbors or fallback_historical_neighbors
                else "insufficient"
            ),
            "retrieval_source": "system_output_fallback",
        }
        if fallback_historical_neighbors:
            result["historical_idea_neighbors"] = fallback_historical_neighbors
        return _strip_blinded_provenance(result)

    @staticmethod
    def _is_dimension_assessable(
        dimension: str,
        idea_snapshot: Dict[str, Any],
        evidence_pack: Dict[str, Any],
    ) -> Tuple[bool, str]:
        if dimension == "novelty":
            evidence = evidence_pack["novelty_evidence"]
            coverage_status = str(evidence.get("coverage_status", ""))
            if coverage_status == "insufficient_idea_fields":
                return False, "The idea lacks enough fields to construct novelty retrieval queries."
            if coverage_status == "insufficient_evidence":
                return False, "Shared novelty evidence retrieval returned too few domain-relevant literature neighbors."
            if not evidence["literature_neighbors"] and not evidence.get("historical_idea_neighbors"):
                if coverage_status in {"sufficient_no_close_neighbor", "retrieval_failed"}:
                    return True, ""
                return False, "No literature or historical-idea neighbor evidence is available for novelty assessment."
        elif dimension == "data_grounding":
            evidence = evidence_pack["dataset_evidence"]
            if not evidence.get("task_dataset_description") and not evidence.get("dataset_portrait"):
                return False, "No task dataset description or dataset portrait is available for data-grounding assessment."
        elif dimension == "realizability":
            required = ("research_question", "core_hypothesis", "computability_and_falsifiability")
            if any(not idea_snapshot.get(key) for key in required):
                return False, "The idea lacks a research question, core hypothesis, or computability statement."
        elif dimension == "logical_coherence":
            required = (
                "knowledge_anchor_and_motivation",
                "research_question",
                "core_hypothesis",
                "theoretical_mechanism",
                "computability_and_falsifiability",
            )
            if any(not idea_snapshot.get(key) for key in required):
                return False, "The idea lacks at least one required argument-chain field."
        elif dimension == "significance":
            if not idea_snapshot.get("knowledge_anchor_and_motivation") or not idea_snapshot.get("research_question"):
                return False, "The idea lacks motivation or a research question for significance assessment."
        elif dimension == "validation_design_quality":
            if evidence_pack["evaluation_stage"] == "idea_only":
                return False, "StepPlan is not available."
            planned = evidence_pack["planned_evidence"]
            if not planned["raw_types_valid"] or not planned["step_plan"]:
                return False, "StepPlan exists but cannot be parsed into validation-design evidence."
        elif dimension == "reproducibility_readiness":
            if evidence_pack["evaluation_stage"] == "idea_only":
                return False, "StepPlan and ResourceCheck are not both available."
            planned = evidence_pack["planned_evidence"]
            if not planned["raw_types_valid"] or not planned["step_plan"] or not planned["resource_check"]:
                return False, "StepPlan or ResourceCheck exists but cannot be parsed into reproducibility evidence."
        return True, ""



    @staticmethod
    def _build_dimension_output_schema(dimension: str) -> Dict[str, Any]:
        subcriteria = list(IDEA_QUALITY_SUBCRITERION_WEIGHTS[dimension])
        anchor_overrides = IDEA_QUALITY_SUBCRITERION_ANCHOR_OVERRIDES.get(dimension, {})
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "subscores": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        key: {
                            "type": "integer",
                            "enum": sorted(IDEA_QUALITY_SCORE_ANCHORS),
                            **(
                                {"description": anchor_overrides[key]}
                                if key in anchor_overrides
                                else {}
                            ),
                        }
                        for key in subcriteria
                    },
                    "required": subcriteria,
                },
                "supporting_evidence": {
                    "type": "array",
                    "maxItems": 16,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "subcriterion": {"type": "string", "enum": subcriteria},
                            "source_type": {"type": "string", "enum": list(IDEA_QUALITY_SOURCE_TYPES)},
                            "source_field": {"type": "string"},
                            "evidence": {"type": "string", "maxLength": 240},
                        },
                        "required": ["subcriterion", "source_type", "source_field", "evidence"],
                    },
                },
                "counterevidence_or_gaps": {
                    "type": "array",
                    "maxItems": 8,
                    "items": {"type": "string", "maxLength": 200},
                },
                "rationale": {"type": "string", "maxLength": 300},
            },
            "required": ["subscores", "supporting_evidence", "counterevidence_or_gaps", "rationale"],
        }

    @staticmethod
    def _normalize_evidence_source_types(
        dimension: str,
        response: Dict[str, Any],
    ) -> None:
        """Drop evidence items whose source_type is outside the allowed vocabulary.

        The model is only softly constrained by the JSON schema, so an unmapped
        evidence channel can surface as an invented source_type. Such an item is
        unusable for reliability weighting, but it must not invalidate the whole
        idea record: the observation is preserved as a gap instead.
        """
        evidence = response.get("supporting_evidence")
        if not isinstance(evidence, list):
            return
        kept: List[Any] = []
        rejected: List[str] = []
        for item in evidence:
            if isinstance(item, dict) and item.get("source_type") not in IDEA_QUALITY_SOURCE_TYPES:
                source_type = str(item.get("source_type", ""))
                source_field = str(item.get("source_field", "")).strip() or "unknown field"
                rejected.append(
                    f"Discarded one {dimension} evidence item citing unsupported source_type "
                    f"'{source_type}' on '{source_field}'; it was not counted toward evidence "
                    f"coverage or source reliability."
                )
                continue
            kept.append(item)
        if not rejected:
            return
        logger.warning(
            "[idea_evaluation] Dropped %s %s evidence item(s) with unsupported source_type",
            len(rejected),
            dimension,
        )
        response["supporting_evidence"] = kept
        gaps = response.get("counterevidence_or_gaps")
        response["counterevidence_or_gaps"] = [
            *(gaps if isinstance(gaps, list) else []),
            *rejected,
        ]

    @staticmethod
    def _validate_dimension_response(dimension: str, response: Dict[str, Any]) -> None:
        expected = set(IDEA_QUALITY_SUBCRITERION_WEIGHTS[dimension])
        subscores = response.get("subscores")
        if not isinstance(subscores, dict) or set(subscores) != expected:
            raise AgentExecutionError(f"{dimension} evaluator returned invalid subcriteria")
        for key, value in subscores.items():
            if isinstance(value, bool) or value not in IDEA_QUALITY_SCORE_ANCHORS:
                raise AgentExecutionError(f"{dimension}.{key} must use an anchored score")
        evidence = response.get("supporting_evidence")
        gaps = response.get("counterevidence_or_gaps")
        if not isinstance(evidence, list) or not isinstance(gaps, list):
            logger.error(
                f"{dimension} evaluator returned invalid evidence lists structure. "
                f"Response keys: {set(response.keys())}. "
                f"supporting_evidence type: {type(evidence).__name__}. "
                f"counterevidence_or_gaps type: {type(gaps).__name__}."
            )
            raise AgentExecutionError(f"{dimension} evaluator returned invalid evidence lists")
        for item in evidence:
            if not isinstance(item, dict):
                raise AgentExecutionError(f"{dimension} evaluator returned a non-object evidence item")
            if item.get("subcriterion") not in expected:
                raise AgentExecutionError(f"{dimension} evaluator cited an unknown evidence subcriterion")
            if item.get("source_type") not in IDEA_QUALITY_SOURCE_TYPES:
                raise AgentExecutionError(f"{dimension} evaluator cited an unsupported evidence source type")
            if not str(item.get("source_field", "")).strip() or not str(item.get("evidence", "")).strip():
                raise AgentExecutionError(f"{dimension} evaluator returned empty evidence provenance")
        if any(not isinstance(item, str) or not item.strip() for item in gaps):
            logger.error(
                f"{dimension} evaluator returned an invalid gap description. "
                f"gaps type: {type(gaps).__name__}, "
                f"item types: {sorted({type(item).__name__ for item in gaps})}."
            )
            raise AgentExecutionError(f"{dimension} evaluator returned an invalid gap description")
        if not str(response.get("rationale", "")).strip():
            logger.error(
                f"{dimension} evaluator must return a rationale. "
                f"rationale type: {type(response.get('rationale')).__name__}."
            )
            raise AgentExecutionError(f"{dimension} evaluator must return a rationale")

"""Versioned, evidence-grounded Depth evaluation for complete research schemes."""

import hashlib
import json
from pathlib import Path
from typing import Any


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
RUBRIC_PATH = BENCHMARK_DIR / "protocol" / "depth_rubric_v1.json"
PROMPT_PATH = BENCHMARK_DIR / "protocol" / "depth_review_prompt_v1.md"
RUBRIC = json.loads(RUBRIC_PATH.read_text(encoding="utf-8"))
RUBRIC_VERSION = RUBRIC["rubric_version"]
SCHEMA_VERSION = "4.0-semantic-reference-retrieval-depth10"
DEPTH_FINGERPRINT = hashlib.sha256(RUBRIC_PATH.read_bytes() + PROMPT_PATH.read_bytes()).hexdigest()
REVIEW_FIELDS = ("reasoning_review", "alternative_review", "boundary_review")
PLAN_PROJECTION_VERSION = "scientific_plan_variables_v3_evidence_quote"
EXCLUDED_FIELDS = frozenset({
    "resource_check", "resource_check_feedback", "execution_contract",
    "implementation_candidates", "tool_reachability", "resolved_dataset_bindings",
    "plan_refinement_result", "idea_revision_audit", "capability_anchor_meta",
    "priority", "status", "final_disposition", "replan_round", "alignment",
    "max_alignment", "depth", "system", "idea_id", "family",
})


def _clean(value: Any) -> Any:
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items() if key not in EXCLUDED_FIELDS}
    return value


def project_research_plan(value: Any) -> Any:
    projected = _clean(value)
    if not isinstance(projected, list):
        return projected
    # Keep explanatory text separate from the immutable text used as evidence.
    for step in projected:
        if isinstance(step, dict) and isinstance(step.get("description"), str):
            step.setdefault("evidence_quote", step["description"])
    return projected


def build_research_scheme(prompt: dict, application: dict) -> dict:
    candidate = application.get("idea_candidate")
    if candidate is not None and not isinstance(candidate, dict):
        raise ValueError("idea_candidate must be an object when present")
    context_fields = (
        "region_name", "data_description", "data_path", "datasets", "data_descriptions",
        "data_paths", "anchor_asset", "anchor_asset_id", "anchor_asset_type",
        "anchor_components", "hard_constraints",
    )
    question = candidate.get("research_question") if candidate is not None else None
    return _clean({
        "task_context": {key: prompt[key] for key in context_fields if key in prompt},
        "title": application.get("application_name"),
        "research_question": question if question is not None else application.get("research_question"),
        "scientific_proposal": candidate,
        "application_description": application.get("application_description"),
        "application_context": application.get("application_context"),
        "approach_description": application.get("approach_description"),
        "required_datasets": application.get("required_datasets"),
        "research_plan": project_research_plan(application.get("task_plan")),
        "rationale": application.get("rationale"),
    })


def build_depth_prompt(research_scheme: dict) -> str:
    _, separator, body = PROMPT_PATH.read_text(encoding="utf-8").partition("## 提示词正文")
    if not separator or body.count("{{research_scheme}}") != 1:
        raise ValueError("Depth prompt must contain one research_scheme placeholder")
    scheme_json = json.dumps(research_scheme, ensure_ascii=False, indent=2)
    valid_paths = sorted(_text_fields(research_scheme))
    path_guidance = (
        "\n\n程序生成的合法证据 field_path 清单（只能从此清单中选择，禁止改写、猜测或跨方案借用路径）：\n"
        + json.dumps(valid_paths, ensure_ascii=False, indent=2)
    )
    return body.replace("{{research_scheme}}", scheme_json) + path_guidance


def _text_fields(value: Any, prefix: str = "research_scheme") -> dict[str, str]:
    if isinstance(value, str):
        return {prefix: value}
    fields = {}
    if isinstance(value, dict):
        for key, item in value.items():
            # Structured source fields can themselves be cited by the judge.
            # Index their deterministic JSON representation as well as their
            # individual leaves so object/list paths remain verifiable.
            fields[f"{prefix}.{key}"] = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
            fields.update(_text_fields(item, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            fields[f"{prefix}[{index}]"] = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
            fields.update(_text_fields(item, f"{prefix}[{index}]"))
    return fields


def normalize_depth_response(parsed: dict, research_scheme: dict) -> dict:
    missing = set(RUBRIC["output_contract"]["required_fields"]) - set(parsed)
    if missing:
        raise ValueError(f"Depth response missing fields: {sorted(missing)}")
    if parsed["rubric_version"] != RUBRIC_VERSION:
        raise ValueError("Depth rubric version mismatch")
    status, score = parsed["review_status"], parsed["score"]
    if status not in {"scored", "insufficient_material"}:
        raise ValueError("Invalid Depth review_status")
    if status == "scored" and (type(score) is not int or not 1 <= score <= 10):
        raise ValueError("Depth score must be an integer in 1..10")
    if status == "insufficient_material" and score is not None:
        raise ValueError("Insufficient material cannot receive a score")
    if parsed["confidence"] not in {"high", "medium", "low"}:
        raise ValueError("Invalid Depth confidence")
    for key in ("research_type", "central_claim", "score_rationale"):
        if not isinstance(parsed[key], str) or not parsed[key].strip():
            raise ValueError(f"Depth {key} must be nonempty text")
    if parsed["research_type"] not in {"causal", "prediction", "descriptive", "measurement", "other", "mixed"}:
        raise ValueError("Invalid Depth research_type")
    if not isinstance(parsed["material_issues"], list):
        raise ValueError("Depth material_issues must be a list")
    gap = parsed["next_level_gap"]
    if status == "scored" and score < 10:
        if not isinstance(gap, str) or not gap.strip():
            raise ValueError("Depth needs a sample-specific next_level_gap")
    elif gap is not None:
        raise ValueError("Depth next_level_gap must be null for score 10 or insufficient material")
    fields = _text_fields(research_scheme)
    evidence_count = 0
    for axis in REVIEW_FIELDS:
        review = parsed[axis]
        if not isinstance(review, dict) or not isinstance(review.get("assessment"), str):
            raise ValueError(f"Invalid Depth {axis}")
        if not isinstance(review.get("evidence"), list) or not isinstance(review.get("gaps"), list):
            raise ValueError(f"Depth {axis} must contain evidence and gaps lists")
        for evidence in review["evidence"]:
            if not isinstance(evidence, dict):
                raise ValueError("Depth evidence must be an object")
            path, quote = evidence.get("field_path"), evidence.get("quote")
            if not isinstance(path, str) or not isinstance(quote, str) or not quote.strip():
                raise ValueError("Depth evidence requires a field path and nonempty quote")
            # Permit either payload-relative or fully qualified field paths.
            qualified = path if path.startswith("research_scheme.") else f"research_scheme.{path}"
            if qualified not in fields:
                raise ValueError(f"Depth evidence field path does not exist: {path}")
            evidence_count += 1
    if status == "scored" and evidence_count == 0:
        raise ValueError("Scored Depth requires source evidence")
    return {
        **{key: parsed[key] for key in RUBRIC["output_contract"]["required_fields"]},
        "depth_score": score,
        "rubric_fingerprint": DEPTH_FINGERPRINT,
    }

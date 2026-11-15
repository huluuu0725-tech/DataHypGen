#!/usr/bin/env python3
"""Semantic reference retrieval followed by gated four-dimensional Alignment and Depth.

Every admitted application of a task is evaluated (Benchmark-v3 §3.1 / §4.1):
the evaluator reads ``full_output.json`` directly, applies the benchmark
admission filter and scores each surviving idea independently.  No single
"best" idea is pre-selected here; best-of / median-of / full-distribution views
are computed afterwards by ``6_three_axis_summary.py``.

Per idea, the evaluator never sends every reference direction directly to the
Alignment judge.  It first maps the prompt anchor plus the application's
supplemental dataset requirements to complete ``PaperResearchInstance`` nodes.
For single/dual inputs, complete single-instance candidates whose status is
``exact_instance_match``, ``equivalent_combination_match`` or
``approximate_substitution_match`` may enter Alignment scoring;
dual tasks additionally need at least one input anchor to keep its observation
identity. Three-or-more-anchor tasks also admit partial research directions;
missing other task anchors does not disqualify them.

Depth (Benchmark-v3 §4.5) is judged on every admitted idea regardless of the
mapping outcome: a calibrated 1-10 review of the complete research scheme,
with exact source evidence. Depth is stored next to Alignment in the same
per-idea record but never merged into a single score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from dataclasses import replace


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=True)

from benchmark.core.admission import (
    load_admitted_portfolio,
    project_idea_fields,
)
from benchmark.core.depth_evaluation import (
    PLAN_PROJECTION_VERSION, project_research_plan,
    DEPTH_FINGERPRINT, RUBRIC_VERSION, SCHEMA_VERSION,
    build_depth_prompt, build_research_scheme, normalize_depth_response,
)
from benchmark.core.benchmark_layout import (
    canonical_task_ids,
    grouped_task_dir,
    require_canonical_task_id,
    system_outputs_root,
)
from benchmark.core.semantic_reference_retrieval import (
    RetrievalConfig,
    apply_alignment,
    batched,
    build_query_components,
    collect_embedding_texts,
    embed_texts,
    evaluate_mapping,
    initial_candidate_pairs,
    load_retrieval_config,
    propose_expansion_pairs,
    scope_graph_for_task,
    validate_graph_freshness,
    validate_pair_reviews,
)


BENCHMARK_DIR = ROOT / "benchmark"
SYSTEM_OUTPUTS_DIR = system_outputs_root(BENCHMARK_DIR, "ours")
DEFAULT_SYSTEM_OUTPUTS_DIR = SYSTEM_OUTPUTS_DIR
TASK_INPUTS_DIR = BENCHMARK_DIR / "dataset" / "task_inputs"
DEFAULT_GRAPH = BENCHMARK_DIR / "reference_sets" / "knowledge_graph" / "grouped_reference_graph_v2.json"
DEFAULT_OUTPUT_DIR = None
ALIGNMENT_EVALUATION_FILENAME = "alignment_depth_evaluation.json"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_BASE_URL_ENV = "OPENAI_API_BASE_URL"
# Embeddings may be served from a different endpoint than the chat judges.
DEFAULT_EMBEDDING_API_KEY_ENV = DEFAULT_API_KEY_ENV
DEFAULT_EMBEDDING_BASE_URL_ENV = DEFAULT_BASE_URL_ENV

DIMENSIONS = (
    "research_question",
    "hypothesis_mechanism",
    "variables_usage",
    "method_paradigm",
)


logger = logging.getLogger("semantic_alignment_scoring")


def model_endpoint(client: Any) -> str:
    url = urlsplit(str(getattr(client, "base_url", "")))
    return f"{url.scheme}://{url.hostname}{':' + str(url.port) if url.port else ''}{url.path}"


def exception_chain_types(exc: Exception) -> str:
    names: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        names.append(type(current).__name__)
        current = current.__cause__ if current.__cause__ is not None else current.__context__
    return " -> ".join(names)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _join_values(value: Any) -> str:
    if isinstance(value, list):
        return " | ".join(_as_text(item) for item in value)
    return _as_text(value)


def extract_idea_fields(document: dict[str, Any]) -> dict[str, str]:
    idea = document.get("idea")
    if not isinstance(idea, dict):
        idea = document.get("system_idea")
    if not isinstance(idea, dict):
        idea = document
    key_variables = idea.get("key_variables")
    if not isinstance(key_variables, dict):
        key_variables = {}
    return {
        "research_question": _as_text(idea.get("research_question")),
        "hypothesis_mechanism": " | ".join(
            item for item in (_as_text(idea.get("hypothesis")), _as_text(idea.get("mechanism"))) if item
        ),
        "variables_usage": " | ".join(
            item
            for item in (
                f"target={_as_text(key_variables.get('target'))}" if key_variables.get("target") is not None else "",
                f"predictors={_join_values(key_variables.get('predictors'))}" if key_variables.get("predictors") is not None else "",
                f"contrast={_as_text(key_variables.get('contrast'))}" if key_variables.get("contrast") else "",
                f"proxy_claim={_as_text(key_variables.get('proxy_claim'))}" if key_variables.get("proxy_claim") else "",
                f"data_sources={_join_values(idea.get('data_sources'))}" if idea.get("data_sources") is not None else "",
            )
            if item
        ),
        "method_paradigm": " | ".join(
            item for item in (_as_text(idea.get("analytical_approach")), _as_text(idea.get("method_paradigm"))) if item
        ),
    }


def project_alignment_idea(application: dict[str, Any]) -> dict[str, Any]:
    projected = dict(application)
    projected["task_plan"] = project_research_plan(application.get("task_plan"))
    result = project_idea_fields(projected)
    candidate = application.get("idea_candidate")
    if isinstance(candidate, dict):
        computability = candidate.get("computability_and_falsifiability")
        hypothesis = candidate.get("core_hypothesis")
        if isinstance(computability, dict):
            if computability.get("observable_variable"):
                result["key_variables"]["target"] = _as_text(computability["observable_variable"])
            result["key_variables"]["contrast"] = _as_text(computability.get("mechanism_relevant_contrast"))
            if computability.get("falsification_condition"):
                result["falsification_condition"] = _as_text(computability["falsification_condition"])
        if isinstance(hypothesis, dict):
            result["hypothesis"] = _as_text(hypothesis.get("directional_statement"))
            result["key_variables"]["proxy_claim"] = _as_text(hypothesis.get("proxy_or_inference_claim"))
    return result




def load_task_portfolio(task_id: str) -> tuple[dict[str, Any], list[tuple[int, dict[str, Any]]], Path]:
    """全部 admitted application（含原始下标）与源产物路径；不经 best_idea.json。"""
    path = grouped_task_dir(SYSTEM_OUTPUTS_DIR, task_id) / "full_output.json"
    if not path.exists():
        raise FileNotFoundError(f"{task_id}: {path} not found")
    output, admitted = load_admitted_portfolio(path)
    return output, admitted, path


def list_task_ids() -> list[str]:
    return [
        task_id
        for task_id in canonical_task_ids(BENCHMARK_DIR)
        if (grouped_task_dir(SYSTEM_OUTPUTS_DIR, task_id) / "full_output.json").is_file()
    ]


def make_client(api_key_env: str, base_url_env: str):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The openai package is required for non-dry semantic evaluation") from exc
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing required API key environment variable: {api_key_env}")
    kwargs: dict[str, Any] = {"api_key": api_key}
    base_url = os.environ.get(base_url_env)
    if base_url:
        kwargs["base_url"] = base_url
    # 503 超时重试扩展到 20 分钟，避免上游短暂服务中断时批量放弃
    kwargs["timeout"] = 1200.0
    kwargs["max_retries"] = 0
    client = OpenAI(**kwargs)
    logger.info("Model client endpoint=%s key_env=%s timeout=%s sdk_max_retries=%s",
                model_endpoint(client), api_key_env, client.timeout, client.max_retries)
    return client


def parse_json_response(raw: str) -> dict[str, Any]:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("model response does not contain a JSON object")
    payload = text[start : end + 1]
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        # 模型偶尔在字符串值里直接输出换行/制表等控制字符；strict=False 仅放宽这一点，不改变合法 JSON 的解析结果
        if "control character" not in str(exc):
            raise
        value = json.loads(payload, strict=False)
    if not isinstance(value, dict):
        raise ValueError("model response JSON root must be an object")
    return value


def model_retry_delay(exc: Exception, attempt: int) -> float:
    # 503 服务不可用时延长等待，避免快速重试耗尽配额
    if getattr(exc, "status_code", None) == 503:
        base_seconds = 60.0
    elif getattr(exc, "status_code", None) == 524:
        base_seconds = 120.0
    else:
        base_seconds = 5.0
    hints: list[Any] = []
    response = getattr(exc, "response", None)
    if response is not None:
        hints.append(response.headers.get("retry-after"))
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        hints.append(body.get("retry_after"))
    for hint in hints:
        if isinstance(hint, bool):
            continue
        try:
            seconds = float(hint)
        except (TypeError, ValueError):
            continue
        if 0 < seconds < float("inf"):
            base_seconds = max(base_seconds, seconds)
    return base_seconds * 2 ** (attempt - 1)


def call_chat(client: Any, model: str, prompt: str, retries: int, validator=None,
              failure_context=None, failure_dir: Path | None = None) -> tuple[Any, int]:
    client = client.with_options(max_retries=0)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        started = time.monotonic()
        request_id = None
        content = None
        logger.info("Model request start endpoint=%s model=%s attempt=%d/%d prompt_chars=%d stream=False",
                    model_endpoint(client), model, attempt, retries, len(prompt))
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                stream=False,
            )
            request_id = getattr(response, "_request_id", None)
            if not response.choices:
                raise ValueError("model returned no choices")
            choice = response.choices[0]
            content = choice.message.content
            logger.info("Model response model=%s returned_model=%s attempt=%d elapsed=%.3fs "
                        "finish_reason=%s request_id=%s response_chars=%d refusal_present=%s",
                        model, getattr(response, "model", None), attempt, time.monotonic() - started,
                        getattr(choice, "finish_reason", None), request_id,
                        len(content) if isinstance(content, str) else 0,
                        bool(getattr(choice.message, "refusal", None)))
            if content is None or not content.strip():
                raise ValueError("model returned an empty response")
            parsed = parse_json_response(content)
            if validator is not None:
                parsed = validator(parsed)
            logger.info("Model request success model=%s attempt=%d elapsed=%.3fs response_chars=%d",
                        model, attempt, time.monotonic() - started, len(content))
            return parsed, attempt
        except Exception as exc:
            last_error = exc
            logger.error("Model request failed endpoint=%s model=%s attempt=%d/%d elapsed=%.3fs "
                         "exception_chain=%s status=%s request_id=%s",
                         model_endpoint(client), model, attempt, retries, time.monotonic() - started,
                         exception_chain_types(exc), getattr(exc, "status_code", None),
                         getattr(exc, "request_id", request_id))
            if isinstance(exc, ValueError):
                logger.error("Model response validation failed: %s", exc)
            if failure_context is not None:
                logger.error("Model failure context=%s error=%s", failure_context, exc)
                if failure_dir is not None:
                    failure_path = failure_dir / f"attempt_{attempt}_{time.time_ns()}.json"
                    write_atomic_json(failure_path, {
                        **failure_context, "attempt": attempt, "model": model,
                        **getattr(exc, "pair_diagnostics", {}),
                        "request_id": request_id, "error": str(exc), "failed_response": content,
                    })
                    logger.error("Failed response saved: %s", failure_path)
            if attempt < retries:
                delay = model_retry_delay(exc, attempt)
                logger.info("Model request retry in %.1fs (exponential backoff)", delay)
                time.sleep(delay)
    raise RuntimeError(f"model request failed after {retries} attempts: {last_error}")


class EmbeddingCache:
    def __init__(self, path: Path | None, model: str) -> None:
        self.path = path
        self.model = model
        self.vectors: dict[str, list[float]] = {}
        self._lock = threading.Lock()
        if path is None or not path.exists():
            return
        payload = load_json(path)
        if payload.get("schema_version") != "1.0" or payload.get("model") != model:
            raise ValueError(f"embedding cache model/schema mismatch: {path}")
        vectors = payload.get("vectors")
        if not isinstance(vectors, dict):
            raise ValueError(f"embedding cache vectors must be an object: {path}")
        self.vectors = vectors

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def resolve(
        self,
        texts: list[str],
        batch_size: int,
        embed_batch,
    ) -> tuple[dict[str, list[float]], int]:
        # Keep the read/embed/update transaction atomic when ideas run in parallel.
        with self._lock:
            missing = [text for text in texts if self.key(text) not in self.vectors]
            if missing:
                embedded = embed_texts(missing, batch_size, embed_batch)
                for text, vector in embedded.items():
                    self.vectors[self.key(text)] = vector
            return {text: self.vectors[self.key(text)] for text in texts}, len(missing)

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": "1.0", "model": self.model, "vectors": self.vectors}
        self.path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


PAIR_REQUEST_METADATA = (
    "verify_origin",
    "graph_relation_hint",
    "graph_relation_path",
    "graph_fallback_depth",
)


def build_pair_review_prompt(
    pairs: list[dict[str, Any]],
    components: dict[str, dict[str, Any]],
    assets: dict[str, dict[str, Any]],
) -> str:
    payload = []
    for pair in pairs:
        entry = {
            "component_id": pair["component_id"],
            "asset_id": pair["asset_id"],
            "retrieval_score": pair["retrieval_score"],
            "query_component": components[pair["component_id"]],
            "candidate_asset": {key: value for key, value in assets[pair["asset_id"]].items() if key != "members"},
        }
        if pair.get("verify_origin"):
            entry["candidate_origin"] = {
                "source": pair["verify_origin"],
                "graph_relation_hint": pair.get("graph_relation_hint"),
                "graph_relation_path": pair.get("graph_relation_path"),
            }
        payload.append(entry)
    return (
        "Act as a dataset-identity cross-encoder. Judge every pair independently from the "
        "structured fields. Observation identity is decided only by three things: the physical "
        "object observed (observable), the source lineage (sensor/mission/model/production "
        "chain), and the core measurement semantics actually used. Spatial/temporal scale may "
        "refine the judgement. analysis_role describes how a study uses a product, NOT what the "
        "product is: a different analysis role must never lower semantic_similarity, flip "
        "observable_consistent, or raise material_conflict. Report role_consistent for "
        "diagnostics only. A topical resemblance is insufficient. Return semantic_similarity in "
        "[0,1], booleans observable_consistent / role_consistent / material_conflict, and exactly "
        "one relation_type: EXACT = the same product or compatible lineage; HAS_EQUIVALENT = a "
        "different product that is an observation-equivalent variant (same physical object, "
        "lineage family and measurement semantics; version, grid, release or aggregation "
        "differences only); HAS_UPSTREAM = a parent/upstream product of the same observation; "
        "HAS_ANALOGUE = an approximate substitute with the same observable but a different "
        "lineage; HAS_COMPLEMENT = additive data for a different observable that cannot replace "
        "the component; NONE otherwise. Never relabel upstream, analogue or complement as EXACT "
        "or HAS_EQUIVALENT. When a pair carries candidate_origin, the graph_relation_hint only "
        "explains how the candidate was discovered; it is not evidence and must not bias your "
        "relation_type. Return exactly one review for every input pair, including NONE pairs. "
        "Copy component_id and asset_id verbatim from that input pair; never rename, normalize, "
        "omit, duplicate, merge or add pairs. Return JSON only as "
        '{"pair_reviews":[{"component_id":"...","asset_id":"...",'
        '"semantic_similarity":0.0,"observable_consistent":false,"role_consistent":false,'
        '"material_conflict":false,"relation_type":"NONE","rationale":"..."}]}\n\n'
        f"Pairs:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def run_pair_reviews(
    client: Any,
    model: str,
    pairs: list[dict[str, Any]],
    components: list[dict[str, Any]],
    assets: list[dict[str, Any]],
    batch_size: int,
    retries: int,
    application_index: int | None = None,
    task_id: str | None = None,
    failure_dir: Path | None = None,
    review_round: str = "initial",
) -> tuple[list[dict[str, Any]], int]:
    component_map = {item["component_id"]: item for item in components}
    asset_map = {item["product_id"]: item for item in assets}
    reviews = []
    attempts = 0
    def validate_response(batch, parsed):
        rows = parsed.get("pair_reviews")
        if not isinstance(rows, list):
            raise ValueError("cross-encoder response has no pair_reviews list")
        return validate_pair_reviews(batch, rows)

    for batch_index, batch in enumerate(batched(pairs, batch_size), start=1):
        rows, request_attempts = call_chat(
            client, model, build_pair_review_prompt(batch, component_map, asset_map), retries,
            validator=lambda parsed: validate_response(batch, parsed),
            failure_context={"task_id": task_id, "application_index": application_index,
                             "review_round": review_round, "batch_index": batch_index,
                             "requested_pairs": [{"component_id": p["component_id"], "asset_id": p["asset_id"]} for p in batch]},
            failure_dir=failure_dir)
        metadata = {
            (pair["component_id"], pair["asset_id"]): {
                key: pair[key] for key in PAIR_REQUEST_METADATA if key in pair
            }
            for pair in batch
        }
        for item in rows:
            reviews.append({**item, **metadata[(item["component_id"], item["asset_id"])]})
        attempts += request_attempts
    return reviews, attempts


def build_alignment_prompt(
    system_fields: dict[str, str],
    candidates: list[dict[str, Any]],
) -> str:
    reference_instances = []
    for candidate in candidates:
        instance = candidate["reference_instance"]
        reference_instances.append(
            {
                "instance_id": candidate["instance_id"],
                "paper_id": candidate["paper_id"],
                "dataset_mapping_status": candidate.get("mapping_status"),
                "essential_coverage": candidate.get("essential_coverage"),
                "missing_components": candidate.get("missing_components", []),
                "covered_components": candidate.get("covered_components", []),
                "reference_direction_policy": candidate.get("reference_direction_policy"),
                "research_question": instance.get("research_question", ""),
                "hypothesis_mechanism": instance.get("hypothesis_mechanism", ""),
                "variables_usage": instance.get("variables_usage", ""),
                "method_paradigm": instance.get("method_paradigm", ""),
                "evidence": instance.get("evidence", {}),
            }
        )
    payload = {"system_idea": system_fields, "eligible_reference_instances": reference_instances}
    return (
        "You are an independent research-methods judge. Reference-direction eligibility has "
        "been established. For three-or-more-anchor tasks a reference may substantively use only "
        "one or two anchors. Missing other task anchors or missing three-dataset joint-use evidence "
        "is not a reason to skip that direction or mechanically lower its scores. Compare the "
        "research question, mechanism, variable relationships and method on the evidenced shared "
        "scope; actual scientific differences still matter. Do not fabricate unobserved data use, "
        "mechanisms or joint research instances. The generated scheme's task-input compliance is "
        "assessed separately and must not be inferred from reference coverage. Do not assume full "
        "dataset-combination equivalence. Score each real research instance on four "
        "dimensions from 1 to 10: research_question, hypothesis_mechanism, variables_usage, "
        "and method_paradigm. Judge semantic alignment, not writing quality. Empty or "
        "insufficient reference-instance fields cannot receive a high score. Do not produce a "
        "hit label or threshold decision. Return JSON only as "
        '{"instance_scores":[{"instance_id":"...",'
        '"research_question":{"score":1,"rationale":"..."},'
        '"hypothesis_mechanism":{"score":1,"rationale":"..."},'
        '"variables_usage":{"score":1,"rationale":"..."},'
        '"method_paradigm":{"score":1,"rationale":"..."}}]}\n\n'
        f"Input:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def normalize_alignment_response(
    candidates: list[dict[str, Any]],
    parsed: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = parsed.get("instance_scores")
    if not isinstance(rows, list):
        raise ValueError("alignment response has no instance_scores list")
    expected = {item["instance_id"] for item in candidates}
    received = {item.get("instance_id") for item in rows if isinstance(item, dict)}
    if received != expected or len(rows) != len(expected):
        raise ValueError("alignment response does not exactly cover requested instances: "
                         f"missing={sorted(expected - received)} extra={sorted(str(item) for item in received - expected)} "
                         f"expected_count={len(expected)} returned_count={len(rows)}")
    normalized = []
    for row in rows:
        scores = {}
        rationales = {}
        for dimension in DIMENSIONS:
            field = row.get(dimension)
            if not isinstance(field, dict):
                raise ValueError(f"{row.get('instance_id')}: missing {dimension} score object")
            scores[dimension] = field.get("score")
            rationales[dimension] = _as_text(field.get("rationale"))
        normalized.append(
            {"instance_id": row["instance_id"], "scores": scores, "rationales": rationales}
        )
    return normalized


def run_alignment(
    client: Any,
    model: str,
    system_fields: dict[str, str],
    candidates: list[dict[str, Any]],
    batch_size: int,
    retries: int,
) -> tuple[list[dict[str, Any]], int]:
    rows = []
    attempts = 0
    for batch in batched(candidates, batch_size):
        normalized, request_attempts = call_chat(
            client, model, build_alignment_prompt(system_fields, batch), retries,
            validator=lambda parsed: normalize_alignment_response(batch, parsed))
        rows.extend(normalized)
        attempts += request_attempts
    return rows, attempts




def run_depth_judge(
    client: Any,
    model: str,
    idea_payload: dict[str, Any],
    retries: int,
) -> tuple[dict[str, Any], int]:
    return call_chat(client, model, build_depth_prompt(idea_payload), retries,
                     validator=lambda parsed: normalize_depth_response(parsed, idea_payload))


def run_dataset_matching(
    *,
    application_index: int,
    application: dict[str, Any],
    prompt: dict[str, Any],
    scoped_graph: dict[str, Any],
    assets: list[dict[str, Any]],
    config: RetrievalConfig,
    retries: int,
    client: Any,
    embedding_cache: EmbeddingCache,
    embedding_client: Any | None = None,
    failure_dir: Path | None = None,
) -> dict[str, Any]:
    """单个 admitted application：检索 → 门控 Alignment → Depth。返回 per-idea 记录。"""
    idea_fields = project_alignment_idea(application)
    components = build_query_components(prompt, application)
    texts = collect_embedding_texts(components, assets)
    embedder = embedding_client or client

    def embed_batch(batch: list[str]) -> list[list[float]]:
        response = embedder.embeddings.create(model=config.embedding_model_version, input=batch)
        ordered = sorted(response.data, key=lambda item: item.index)
        return [list(item.embedding) for item in ordered]

    embeddings, embedded_text_count = embedding_cache.resolve(texts, config.embedding_batch_size, embed_batch)
    pairs = initial_candidate_pairs(components, assets, embeddings, config)
    pair_reviews, pair_attempts = run_pair_reviews(
        client,
        config.cross_encoder_version,
        pairs,
        components,
        assets,
        config.cross_encoder_batch_size,
        retries,
        application_index=application_index, task_id=application.get("task_id"), failure_dir=failure_dir,
    )
    mapping = evaluate_mapping(scoped_graph, components, pair_reviews, config)
    def audit_reviews(reviews):
        for review in reviews:
            reasons = []
            if review["material_conflict"]:
                reasons.append("material_conflict")
            if not review["observable_consistent"]:
                reasons.append("observable_inconsistent")
            if review["semantic_similarity"] < config.entity_similarity_threshold:
                reasons.append("below_similarity_threshold")
            if review["relation_type"] in {"NONE", "HAS_COMPLEMENT"}:
                reasons.append("non_substituting_relation")
            review["admitted"] = not reasons
            review["rejection_reasons"] = reasons
    audit_reviews(pair_reviews)
    verification_pairs: list[dict[str, Any]] = []
    verification_reviews: list[dict[str, Any]] = []
    verification_attempts = 0
    if not any(item["essential_coverage"] == 1.0 for item in mapping["candidate_instance_matches"]):
        verification_pairs = propose_expansion_pairs(components, scoped_graph, mapping, pair_reviews, config)
        if verification_pairs:
            verification_reviews, verification_attempts = run_pair_reviews(
                client,
                config.cross_encoder_version,
                verification_pairs,
                components,
                assets,
                config.cross_encoder_batch_size,
                retries,
                application_index=application_index, task_id=application.get("task_id"),
                failure_dir=failure_dir, review_round="graph_expansion",
            )
            mapping = evaluate_mapping(
                scoped_graph, components, [*pair_reviews, *verification_reviews], config
            )
            audit_reviews(verification_reviews)
    return {
        "components": components, "mapping": mapping,
        "embedded_text_count": embedded_text_count, "pairs": pairs,
        "pair_reviews": pair_reviews,
        "pair_attempts": pair_attempts, "verification_pairs": verification_pairs,
        "verification_reviews": verification_reviews, "verification_attempts": verification_attempts,
    }


def write_atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".writing")
    try:
        with staging.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        staging.replace(path)
    finally:
        if staging.exists():
            staging.unlink()


def score_idea(
    *,
    application_index: int,
    application: dict[str, Any],
    prompt: dict[str, Any],
    scoped_graph: dict[str, Any],
    assets: list[dict[str, Any]],
    config: RetrievalConfig,
    retries: int,
    client: Any,
    embedding_cache: EmbeddingCache,
    embedding_client: Any | None = None,
    checkpoint_dir: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    fingerprint = hashlib.sha256(json.dumps({
        "contract": "semantic-stages-partial-anchor-directions-v3", "application_index": application_index,
        "application": application, "prompt": prompt, "graph": scoped_graph,
        "assets": assets, "config": config.to_dict(),
    }, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    if force and checkpoint_dir is not None:
        for name in ("dataset_matching", "alignment", "depth"):
            path = checkpoint_dir / f"{name}.json"
            if path.exists():
                path.unlink()
        logger.info(
            "%s idea[%d] force: cleared prior stage checkpoints checkpoint_dir=%s",
            application.get("task_id", "unknown_task"), application_index, checkpoint_dir,
        )

    def stage(name: str, compute):
        path = checkpoint_dir / f"{name}.json" if checkpoint_dir is not None else None
        stage_fingerprint = fingerprint
        if name == "depth":
            stage_fingerprint = hashlib.sha256((fingerprint + DEPTH_FINGERPRINT).encode()).hexdigest()
        elif name == "alignment":
            stage_fingerprint = hashlib.sha256((fingerprint + PLAN_PROJECTION_VERSION).encode()).hexdigest()
        if path is not None and path.exists() and not force:
            saved = load_json(path)
            current = saved.get("schema_version") == "semantic-stages-v1" and saved.get("fingerprint") == stage_fingerprint and saved.get("stage") == name
            if not current:
                logger.warning(
                    "%s idea[%d] stage=%s checkpoint input/config mismatch; recompute checkpoint=%s",
                    application.get("task_id", "unknown_task"), application_index, name, path,
                )
                # An invalid checkpoint must never block a task that has no
                # completed alignment_depth_evaluation.json yet. Recompute and
                # atomically replace it below; --force still clears all stages
                # before starting when explicitly requested.
                saved = None
            if current:
                logger.info(
                    "%s idea[%d] stage=%s resume checkpoint=%s checkpoint_dir=%s",
                    application.get("task_id", "unknown_task"), application_index, name, path, checkpoint_dir,
                )
                return saved["result"]
        logger.info(
            "%s idea[%d] stage=%s start checkpoint_dir=%s",
            application.get("task_id", "unknown_task"), application_index, name, checkpoint_dir,
        )
        result = compute()
        if path is not None:
            write_atomic_json(path, {"schema_version": "semantic-stages-v1",
                                    "fingerprint": stage_fingerprint, "stage": name, "result": result})
            logger.info(
                "%s idea[%d] stage=%s saved checkpoint=%s checkpoint_dir=%s",
                application.get("task_id", "unknown_task"), application_index, name, path, checkpoint_dir,
            )
        return result

    matched = stage("dataset_matching", lambda: run_dataset_matching(
        application_index=application_index, application=application, prompt=prompt,
        scoped_graph=scoped_graph, assets=assets, config=config, retries=retries,
        client=client, embedding_cache=embedding_cache, embedding_client=embedding_client,
        failure_dir=checkpoint_dir / "failures" if checkpoint_dir is not None else None,
    ))
    components = matched["components"]
    mapping = matched["mapping"]
    embedded_text_count = matched["embedded_text_count"]
    pairs = matched["pairs"]
    pair_attempts = matched["pair_attempts"]
    verification_pairs = matched["verification_pairs"]
    verification_reviews = matched["verification_reviews"]
    verification_attempts = matched["verification_attempts"]
    idea_fields = project_alignment_idea(application)

    def alignment_stage():
        if not mapping["candidate_instance_matches"]:
            return {"rows": [], "attempts": 0, "status": "skipped_no_candidates"}
        rows, attempts = run_alignment(client, config.alignment_model_version,
                                       extract_idea_fields(idea_fields), mapping["candidate_instance_matches"],
                                       config.alignment_batch_size, retries)
        return {"rows": rows, "attempts": attempts, "status": "evaluated"}

    aligned = stage("alignment", alignment_stage)
    def depth_stage():
        depth, attempts = run_depth_judge(
            client, config.alignment_model_version, build_research_scheme(prompt, application), retries)
        return {"depth": depth, "attempts": attempts}

    depth_result = stage("depth", depth_stage)
    alignment_rows = aligned["rows"]
    alignment_attempts = aligned["attempts"]
    depth = depth_result["depth"]
    depth_attempts = depth_result["attempts"]
    result = apply_alignment(mapping, alignment_rows, config)
    return {
        "idea_id": application.get("idea_id"),
        "application_index": application_index,
        "application_name": application.get("application_name", ""),
        "idea_type": application.get("idea_type", ""),
        "priority": application.get("priority"),
        "status": application.get("status", ""),
        "idea_fields": idea_fields,
        "query_components": components,
        "execution_counts": {
            "new_embedding_texts": embedded_text_count,
            "candidate_pairs": len(pairs),
            "cross_encoder_attempts": pair_attempts,
            "expansion_verification_pairs": len(verification_pairs),
            "expansion_verification_attempts": verification_attempts,
            "expansion_verified_admitted": sum(
                1
                for item in result["dataset_match_paths"]
                if item.get("expansion") == "graph_verified"
            ),
            "eligible_alignment_instances": len(mapping["candidate_instance_matches"]),
            "alignment_attempts": alignment_attempts,
            "depth_attempts": depth_attempts,
        },
        "expansion_verification_reviews": verification_reviews,
        "dataset_matching_reviews": matched.get("pair_reviews", []),
        "retrieval_candidates": pairs,
        "research_scheme": build_research_scheme(prompt, application),
        "depth": depth,
        **result,
    }


def score_task(
    task_id: str,
    graph: dict[str, Any],
    config: RetrievalConfig,
    output_dir: Path | None,
    retries: int,
    dry_run: bool,
    force: bool,
    client: Any | None,
    embedding_cache: EmbeddingCache,
    graph_path: Path,
    config_path: Path,
    embedding_client: Any | None = None,
    idea_workers: int = 1,
    prepare_only: bool = False,
) -> Any:
    _, admitted, source_path = load_task_portfolio(task_id)
    prompt_path = grouped_task_dir(TASK_INPUTS_DIR, task_id) / "prompt.json"
    if not prompt_path.is_file():
        raise FileNotFoundError(f"{task_id}: prompt not found: {prompt_path}")
    prompt = load_json(prompt_path)
    scoped_graph, seed_exclusion_doi = scope_graph_for_task(graph, task_id)
    assets = scoped_graph.get("products")
    if not isinstance(assets, list):
        raise ValueError("semantic graph products must be a list")
    task_dir = source_path.parent if output_dir is None else grouped_task_dir(output_dir, task_id)
    output_path = task_dir / ALIGNMENT_EVALUATION_FILENAME
    # The task result is the completion marker.  Once the final evaluation file
    # exists, do not descend into stale stage checkpoints unless --force is used.
    # This prevents an old intermediate checkpoint from blocking a completed task
    # with an input/config mismatch.
    if output_path.exists() and not force:
        saved_result = load_json(output_path)
        if saved_result.get("status") == "evaluated" and saved_result.get("reference_direction_policy") == "partial_anchor_research_v1":
            logger.info("%s: %s exists, skip recomputation", task_id, output_path)
            return True

    if dry_run:
        fan_out = [len(value) for value in scoped_graph.get("product_to_instances", {}).values()]
        component_counts = [
            len(build_query_components(prompt, application)) for _, application in admitted
        ]
        planned_pairs = sum(
            count * min(config.initial_candidate_limit, len(assets)) for count in component_counts
        )
        logger.info(
            "%s: admitted_ideas=%d, components_per_idea=%s, instances=%d, products=%d "
            "(multi-instance=%d, max_fan_out=%d), graph_relations=%d, cross_encoder_pairs<=%d "
            "(+ conditional graph-expansion verification round per idea), depth_judge_calls=%d, output=%s",
            task_id,
            len(admitted),
            component_counts,
            len(scoped_graph.get("research_instances", [])),
            len(assets),
            sum(1 for value in fan_out if value > 1),
            max(fan_out, default=0),
            len(scoped_graph.get("relations", [])),
            planned_pairs,
            len(admitted),
            output_path,
        )
        return True
    if client is None:
        raise RuntimeError("non-dry semantic scoring requires a model client")

    def score_one(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        index, application = item
        idea_path = task_dir / "alignment_depth_ideas" / f"idea_{index:04d}.json"
        try:
            idea = score_idea(
                application_index=index,
                application={**application, "task_id": task_id},
                prompt=prompt,
                scoped_graph=scoped_graph,
                assets=assets,
                config=config,
                retries=retries,
                client=client,
                embedding_cache=embedding_cache,
                embedding_client=embedding_client,
                checkpoint_dir=task_dir / "alignment_depth_checkpoints" / f"idea_{index:04d}",
                force=force,
            )
        except Exception as exc:
            write_atomic_json(idea_path, {**document, "status": "failed", "application_index": index,
                                         "admitted_idea_count": 1,
                                         "idea_id": application["idea_id"], "error": str(exc), "ideas": []})
            logger.error("%s idea[%d] (%s) failed: %s result=%s",
                         task_id, index, application["idea_id"], exc, idea_path)
            raise
        write_atomic_json(idea_path, {**document, "application_index": index,
                                     "admitted_idea_count": 1, "idea_id": idea["idea_id"], "ideas": [idea]})
        logger.info("%s idea[%d] (%s): mapping=%s hit=%s depth=%s result=%s",
                    task_id, index, idea["idea_id"], idea["mapping_status"],
                    idea["existing_idea_hit"], idea["depth"]["depth_score"], idea_path)
        return idea

    document = {
        "candidate_identity_version": "canonical-products-v3-coverage",
        "schema_version": SCHEMA_VERSION,
        "reference_direction_policy": "partial_anchor_research_v1",
        "plan_projection_version": PLAN_PROJECTION_VERSION,
        "depth_protocol": {"rubric_version": RUBRIC_VERSION, "fingerprint": DEPTH_FINGERPRINT},
        "task_id": task_id,
        "status": "evaluated",
        "frozen_retrieval_config": config.to_dict(),
        "provenance": {
            "graph_snapshot": str(graph_path),
            "graph_sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
            "retrieval_config": str(config_path),
            "retrieval_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "full_output": str(source_path),
            "full_output_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "prompt": str(prompt_path),
            "prompt_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
        },
        "seed_exclusion_doi": seed_exclusion_doi,
        "admitted_idea_count": len(admitted),
    }

    def finalize(results, errors):
        write_atomic_json(output_path, {**document,
            "status": "partial_failure" if errors else "evaluated",
            "ideas": [results[index] for index, _ in admitted if index in results],
            "idea_failures": [{"application_index": index, "error": errors[index]}
                              for index, _ in admitted if index in errors],
        })
        logger.info("%s: ideas completed=%d failed=%d -> %s", task_id, len(results), len(errors), output_path)

    if prepare_only:
        return score_one, admitted, finalize
    results, errors = {}, {}
    with ThreadPoolExecutor(max_workers=idea_workers) as pool:
        futures = {pool.submit(score_one, item): item[0] for item in admitted}
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                index = futures.pop(future)
                try:
                    results[index] = future.result()
                except Exception as exc:
                    errors[index] = str(exc)
    finalize(results, errors)
    return not errors


def run_idea_pool(task_ids, prepare_task, workers: int) -> int:
    # Lazily prepare tasks and keep only workers idea futures in flight.
    failures = 0
    states = {}

    def jobs():
        nonlocal failures
        for task_id in task_ids:
            try:
                prepared = prepare_task(task_id)
            except Exception as exc:
                failures += 1
                logger.error("%s preparation failed: %s", task_id, exc)
                continue
            if prepared is True:
                continue
            score_one, admitted, finalize = prepared
            if not admitted:
                finalize({}, {})
                continue
            states[task_id] = {"remaining": len(admitted), "results": {}, "errors": {}, "finalize": finalize}
            for item in admitted:
                yield task_id, score_one, item

    iterator = iter(jobs())
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="alignment-idea") as pool:
        pending = {}
        exhausted = False
        while pending or not exhausted:
            while len(pending) < workers and not exhausted:
                job = next(iterator, None)
                if job is None:
                    exhausted = True
                    break
                task_id, score_one, item = job
                pending[pool.submit(score_one, item)] = (task_id, item[0])
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                task_id, index = pending.pop(future)
                state = states[task_id]
                try:
                    state["results"][index] = future.result()
                except Exception as exc:
                    state["errors"][index] = str(exc)
                state["remaining"] -= 1
                if state["remaining"] == 0:
                    try:
                        state["finalize"](state["results"], state["errors"])
                    except Exception as exc:
                        state["errors"]["finalize"] = str(exc)
                        logger.error("%s aggregate write failed: %s", task_id, exc)
                    failures += bool(state["errors"])
                    del states[task_id]
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve complete dataset-combination instances before four-dimensional Alignment"
    )
    parser.add_argument("--retrieval-config", type=Path, default=ROOT / "config" / "debug_config.yaml", help="Retrieval config; defaults to semantic_retrieval in config/debug_config.yaml")
    parser.add_argument("--graph-snapshot", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--task-id", help="One canonical task ID, for example dual_dataset/task_001")
    parser.add_argument("--dataset-type", choices=("all", "single_dataset", "dual_dataset", "multi_dataset", "three_or_more_datasets"), default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Optional output task-tree root; defaults to each full_output.json directory")
    parser.add_argument("--system-outputs-dir", type=Path, default=SYSTEM_OUTPUTS_DIR)
    parser.add_argument("--embedding-cache", type=Path)
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--base-url-env", default=DEFAULT_BASE_URL_ENV)
    parser.add_argument("--embedding-api-key-env", default=DEFAULT_EMBEDDING_API_KEY_ENV, help="Env var holding the API key for the embedding endpoint (defaults to the chat key)")
    parser.add_argument("--embedding-base-url-env", default=DEFAULT_EMBEDDING_BASE_URL_ENV, help="Env var holding the base URL of the embedding endpoint (defaults to the chat base URL)")
    parser.add_argument("--embedding-model", help="Explicit embedding model id; overrides embedding_model_version from --retrieval-config")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--idea-workers", type=int, default=4,
                        help="Global idea slots shared across tasks (default: 4)")
    parser.add_argument("--dry-run", action="store_true", help="Validate provenance and report planned calls without API access")
    parser.add_argument("--force", action="store_true", help="Overwrite existing result files")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args()


def main() -> int:
    global SYSTEM_OUTPUTS_DIR
    args = parse_args()
    SYSTEM_OUTPUTS_DIR = args.system_outputs_dir.resolve()
    if args.retries < 1:
        raise ValueError("--retries must be >= 1")
    if args.idea_workers < 1:
        raise ValueError("--idea-workers must be >= 1")
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(message)s")
    config_path = args.retrieval_config.resolve()
    graph_path = args.graph_snapshot.resolve()
    config = load_retrieval_config(config_path)
    if args.embedding_model:
        config = replace(config, embedding_model_version=args.embedding_model)
    graph = load_json(graph_path)
    validate_graph_freshness(graph, BENCHMARK_DIR)
    if args.task_id:
        require_canonical_task_id(BENCHMARK_DIR, args.task_id)
        task_ids = [args.task_id]
    else:
        task_ids = list_task_ids()
    if args.dataset_type != "all":
        group = "three_or_more_datasets" if args.dataset_type == "multi_dataset" else args.dataset_type
        task_ids = [task_id for task_id in task_ids if task_id.startswith(group + "/")]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be >= 1")
        task_ids = task_ids[:args.limit]
    if not task_ids:
        logger.warning("No admitted grouped system ideas found")
        return 0

    client = None if args.dry_run else make_client(args.api_key_env, args.base_url_env)
    embedding_client = None
    if client is not None:
        same_endpoint = (args.embedding_api_key_env, args.embedding_base_url_env) == (args.api_key_env, args.base_url_env)
        embedding_client = client if same_endpoint else make_client(args.embedding_api_key_env, args.embedding_base_url_env)
        logger.info(
            "chat judges via %s/%s; embeddings via %s/%s (%s)",
            args.api_key_env,
            args.base_url_env,
            args.embedding_api_key_env,
            args.embedding_base_url_env,
            "same client" if same_endpoint else "separate client",
        )
    cache = EmbeddingCache(args.embedding_cache.resolve() if args.embedding_cache else None, config.embedding_model_version)
    def prepare_task(task_id):
        return score_task(
                task_id=task_id,
                graph=graph,
                config=config,
                output_dir=args.output_dir.resolve() if args.output_dir is not None else None,
                retries=args.retries,
                dry_run=args.dry_run,
                force=args.force,
                client=client,
                embedding_cache=cache,
                graph_path=graph_path,
                config_path=config_path,
                embedding_client=embedding_client,
                idea_workers=args.idea_workers,
                prepare_only=True,
            )
    failures = run_idea_pool(task_ids, prepare_task, args.idea_workers)
    if not args.dry_run:
        cache.save()
    if failures:
        logger.error("Semantic Alignment failed for %d/%d tasks", failures, len(task_ids))
        return 1
    logger.info("Semantic Alignment completed for %d tasks", len(task_ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())

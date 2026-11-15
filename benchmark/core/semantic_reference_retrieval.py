"""Pure retrieval and mapping logic for the semantic reference-space evaluator."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


# Fields that define observation identity within a product.  ``analysis_role``
# is deliberately excluded: it describes how a study *uses* a dataset, not what
# the dataset *is*, so it must never enter retrieval ranking or pair gating.
# ``dataset_name`` is ranked because string-only product mentions carry no other
# identity text.
SEMANTIC_FIELDS = (
    "dataset_name",
    "observable",
    "measurement_semantics",
    "source_lineage",
    "spatial_scale",
    "temporal_scale",
)
GRAPH_SCHEMA_VERSION = "2.0-grouped-semantic-reference-graph"
RECORDED_ONLY_FIELDS = ("analysis_role",)
FULL_MATCH_STATUSES = {
    "exact_instance_match",
    "equivalent_combination_match",
    "approximate_substitution_match",
}
ALIGNMENT_ELIGIBLE_STATUSES = FULL_MATCH_STATUSES | {"anchor_only_match", "partial_reference_direction_match"}
MAPPING_STATUSES = FULL_MATCH_STATUSES | {
    "anchor_only_match",
    "partial_reference_direction_match",
    "partial_instance_match",
    "distributed_compositional_support",
    "unsupported_component",
    "anchor_identity_not_retained",
    "none",
}
RELATION_TYPES = {
    "EXACT",
    "HAS_EQUIVALENT",
    "HAS_UPSTREAM",
    "HAS_COMPLEMENT",
    "HAS_ANALOGUE",
    "NONE",
}
# Only an explicit equivalence relation proves that a different product can stand
# in for another observation identity.  Upstream / complement / analogue edges may
# surface candidates during expansion, but the relation alone cannot be admitted
# as a substitution.
SUBSTITUTION_VERIFYING_RELATIONS = frozenset({"HAS_EQUIVALENT"})
# A component keeps its original observation identity when it is matched to the
# same product or an observation-equivalent variant.  An approximate substitute
# does not retain it.
IDENTITY_RETAINING_RELATIONS = frozenset({"EXACT", "HAS_EQUIVALENT"})
# An additive complement fills a *different* observable slot, so it can never
# substitute for a missing required product.
NON_SUBSTITUTING_RELATIONS = frozenset({"HAS_COMPLEMENT"})
MATCH_KIND_EXACT = "exact"
MATCH_KIND_EQUIVALENT = "observation_equivalent"
MATCH_KIND_APPROXIMATE = "approximate_substitution"
_STATUS_RANK = {
    "exact_instance_match": 0,
    "equivalent_combination_match": 1,
    "approximate_substitution_match": 2,
}


def _require_number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"retrieval config {key} must be a finite number")
    return float(value)


def _require_positive_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"retrieval config {key} must be an integer >= 1")
    return value


@dataclass(frozen=True)
class RetrievalConfig:
    embedding_model_version: str
    cross_encoder_version: str
    alignment_model_version: str
    entity_similarity_threshold: float
    role_weight_version: str
    role_weights: dict[str, float]
    field_weights: dict[str, float]
    allowed_relation_types: tuple[str, ...]
    relation_costs: dict[str, float]
    maximum_fallback_depth: int
    maximum_evidence_size: int
    unmatched_penalty: float
    alignment_threshold: float
    initial_candidate_limit: int
    embedding_batch_size: int
    cross_encoder_batch_size: int
    alignment_batch_size: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RetrievalConfig":
        required_text = (
            "embedding_model_version",
            "cross_encoder_version",
            "alignment_model_version",
            "role_weight_version",
        )
        text_values = {}
        for key in required_text:
            value = payload.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"retrieval config {key} must be a non-empty string")
            if value.strip().startswith("REPLACE_WITH_"):
                raise ValueError(f"retrieval config {key} still contains a placeholder")
            text_values[key] = value.strip()

        threshold = _require_number(payload, "entity_similarity_threshold")
        unmatched_penalty = _require_number(payload, "unmatched_penalty")
        alignment_threshold = _require_number(payload, "alignment_threshold")
        if not 0 < threshold <= 1:
            raise ValueError("entity_similarity_threshold must be in (0, 1]")
        if not 0 < alignment_threshold <= 10:
            raise ValueError("alignment_threshold must be in (0, 10]")

        field_weights = payload.get("field_weights")
        if not isinstance(field_weights, dict) or set(field_weights) != set(SEMANTIC_FIELDS):
            raise ValueError(f"field_weights must define exactly {list(SEMANTIC_FIELDS)}")
        normalized_field_weights = {}
        for key, value in field_weights.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0:
                raise ValueError(f"field_weights.{key} must be positive")
            normalized_field_weights[key] = float(value)

        role_weights = payload.get("role_weights")
        if not isinstance(role_weights, dict) or "*" not in role_weights:
            raise ValueError("role_weights must be an object containing a '*' weight")
        normalized_role_weights = {}
        for key, value in role_weights.items():
            if not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("role_weights entries must map strings to numbers")
            if float(value) <= 0:
                raise ValueError("role weights must be positive")
            normalized_role_weights[key] = float(value)

        allowed = payload.get("allowed_relation_types")
        if not isinstance(allowed, list) or not allowed or any(item not in RELATION_TYPES - {"EXACT", "NONE"} for item in allowed):
            raise ValueError("allowed_relation_types contains an unsupported relation")
        costs = payload.get("relation_costs")
        if not isinstance(costs, dict) or any(item not in costs for item in allowed):
            raise ValueError("relation_costs must define every allowed relation")
        normalized_costs = {}
        for relation in allowed:
            value = costs[relation]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0:
                raise ValueError(f"relation_costs.{relation} must be positive")
            normalized_costs[relation] = float(value)

        return cls(
            **text_values,
            entity_similarity_threshold=threshold,
            role_weights=normalized_role_weights,
            field_weights=normalized_field_weights,
            allowed_relation_types=tuple(allowed),
            relation_costs=normalized_costs,
            maximum_fallback_depth=_require_positive_int(payload, "maximum_fallback_depth"),
            maximum_evidence_size=_require_positive_int(payload, "maximum_evidence_size"),
            unmatched_penalty=unmatched_penalty,
            alignment_threshold=alignment_threshold,
            initial_candidate_limit=_require_positive_int(payload, "initial_candidate_limit"),
            embedding_batch_size=_require_positive_int(payload, "embedding_batch_size"),
            cross_encoder_batch_size=_require_positive_int(payload, "cross_encoder_batch_size"),
            alignment_batch_size=_require_positive_int(payload, "alignment_batch_size"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "embedding_model_version": self.embedding_model_version,
            "cross_encoder_version": self.cross_encoder_version,
            "alignment_model_version": self.alignment_model_version,
            "entity_similarity_threshold": self.entity_similarity_threshold,
            "role_weight_version": self.role_weight_version,
            "role_weights": self.role_weights,
            "field_weights": self.field_weights,
            "allowed_relation_types": list(self.allowed_relation_types),
            "relation_costs": self.relation_costs,
            "maximum_fallback_depth": self.maximum_fallback_depth,
            "maximum_evidence_size": self.maximum_evidence_size,
            "unmatched_penalty": self.unmatched_penalty,
            "alignment_threshold": self.alignment_threshold,
            "initial_candidate_limit": self.initial_candidate_limit,
            "embedding_batch_size": self.embedding_batch_size,
            "cross_encoder_batch_size": self.cross_encoder_batch_size,
            "alignment_batch_size": self.alignment_batch_size,
        }


def load_retrieval_config(path: Path) -> RetrievalConfig:
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or "semantic_retrieval" not in document:
            raise ValueError(f"configuration must define semantic_retrieval: {path}")
        value = document["semantic_retrieval"]
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"retrieval config root must be an object: {path}")
    return RetrievalConfig.from_dict(value)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _join(*values: Any) -> str:
    return " | ".join(text for value in values if (text := _as_text(value)))


def _identity_values(value: Any) -> str:
    if isinstance(value, dict):
        return _as_text(value.get("values"))
    return _as_text(value)


def _normalize_description_component(
    raw: dict[str, Any],
    *,
    component_id: str,
    role: str,
    essential: bool,
    origin: str,
) -> dict[str, Any]:
    description = raw.get("data_description")
    if not isinstance(description, dict):
        description = raw
    unit = description.get("unit_of_measurement")
    return {
        "component_id": component_id,
        "dataset_name": _as_text(description.get("dataset_name") or raw.get("dataset_name") or raw.get("name")),
        "observable": _join(
            unit.get("primary_measurement") if isinstance(unit, dict) else None,
            description.get("data_type"),
            description.get("analytical_capabilities"),
        ),
        "source_lineage": _join(
            description.get("data_source"),
            description.get("source_metadata"),
            description.get("gee_asset"),
            raw.get("data_path"),
            description.get("data_path"),
        ),
        "measurement_semantics": _join(description.get("data_format"), unit),
        "analysis_role": role,
        "spatial_scale": _join(description.get("spatial_resolution"), description.get("spatial_coverage")),
        "temporal_scale": _join(description.get("temporal_unit"), description.get("temporal_coverage")),
        "essential": essential,
        "origin": origin,
    }


def _base_components(prompt: dict[str, Any]) -> list[dict[str, Any]]:
    anchor = prompt.get("anchor_asset")
    components = anchor.get("components") if isinstance(anchor, dict) else None
    if not isinstance(components, list) or not components:
        components = prompt.get("datasets")
    if isinstance(components, list) and components:
        result = []
        for index, item in enumerate(components, start=1):
            if not isinstance(item, dict):
                continue
            role = _as_text(item.get("role") or item.get("member_id")) or f"base_dataset_{index}"
            result.append(
                _normalize_description_component(
                    item,
                    component_id=f"anchor:{role}",
                    role=role,
                    essential=True,
                    origin="prompt.anchor_component",
                )
            )
        if result:
            return result
    description = prompt.get("data_description")
    if isinstance(description, dict):
        return [
            _normalize_description_component(
                {"data_description": description, "data_path": prompt.get("data_path")},
                component_id="anchor:base_dataset",
                role="base_dataset",
                essential=True,
                origin="prompt.base_dataset",
            )
        ]
    raise ValueError("prompt has no structured base dataset or anchor components")


def _task_plan_roles(application: dict[str, Any]) -> dict[str, list[str]]:
    roles: dict[str, set[str]] = {}
    task_plan = application.get("task_plan")
    if not isinstance(task_plan, list):
        return {}
    for step in task_plan:
        if not isinstance(step, dict):
            continue
        for item in step.get("inputs", []):
            if not isinstance(item, dict) or item.get("source_kind") != "required_dataset":
                continue
            source_id = _as_text(item.get("source_id"))
            role = _as_text(item.get("role"))
            if source_id and role:
                roles.setdefault(source_id, set()).add(role)
    return {key: sorted(values) for key, values in roles.items()}


def _required_component(raw: dict[str, Any], roles: dict[str, list[str]], index: int) -> dict[str, Any]:
    frozen = raw.get("frozen_constraints")
    if not isinstance(frozen, dict):
        frozen = {}
    requirement_id = _as_text(raw.get("id")) or f"required_{index}"
    role = _join(roles.get(requirement_id), raw.get("research_justification")) or requirement_id
    temporal = _join(raw.get("temporal_coverage"), raw.get("temporal_resolution"))
    level = _as_text(raw.get("execution_requirement_level"))
    if level not in {"required", "recommended", "optional"}:
        raise ValueError(f"required_datasets[{index}] has invalid execution_requirement_level")
    return {
        "component_id": f"supplemental:{requirement_id}",
        "dataset_name": _as_text(raw.get("canonical_requirement_name") or raw.get("name")),
        "observable": _join(_identity_values(frozen.get("observable")), raw.get("modality")),
        "source_lineage": _join(
            _identity_values(frozen.get("entity")),
            frozen.get("dataset_family"),
            raw.get("preferred_family"),
            raw.get("allowed_product_families"),
            raw.get("source_preference"),
        ),
        "measurement_semantics": _join(
            _identity_values(frozen.get("measure")),
            raw.get("binding_notes"),
            raw.get("retrieval_hint"),
        ),
        "analysis_role": role,
        "spatial_scale": _as_text(raw.get("spatial_resolution")),
        "temporal_scale": temporal,
        "essential": level == "required",
        "origin": "application.required_datasets",
        "execution_requirement_level": level,
    }


def build_query_components(prompt: dict[str, Any], application: dict[str, Any]) -> list[dict[str, Any]]:
    components = _base_components(prompt)
    roles = _task_plan_roles(application)
    required = application.get("required_datasets")
    if required is None:
        required = []
    if not isinstance(required, list):
        raise ValueError("selected application required_datasets must be a list")
    for index, item in enumerate(required, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"required_datasets[{index}] must be an object")
        components.append(_required_component(item, roles, index))
    component_ids = [item["component_id"] for item in components]
    if len(component_ids) != len(set(component_ids)):
        raise ValueError("query component IDs are not unique")
    return components


def _validate_source_entries(entries: Any, benchmark_dir: Path, label: str) -> None:
    if not isinstance(entries, list):
        raise ValueError(f"semantic graph has no {label} provenance")
    for source in entries:
        if not isinstance(source, dict):
            raise ValueError(f"invalid semantic graph {label} provenance entry")
        path = benchmark_dir / _as_text(source.get("path"))
        if source.get("status") == "missing":
            if path.exists():
                raise ValueError(f"semantic graph is stale: previously missing source now exists: {path}")
            continue
        if not path.is_file():
            raise ValueError(f"semantic graph source is missing: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != source.get("sha256"):
            raise ValueError(f"semantic graph is stale: source hash changed: {path}")


def validate_graph_freshness(graph: dict[str, Any], benchmark_dir: Path) -> None:
    if graph.get("schema_version") != GRAPH_SCHEMA_VERSION:
        raise ValueError("unsupported semantic graph schema")
    manifest = graph.get("task_manifest")
    if not isinstance(manifest, dict):
        raise ValueError("semantic graph has no task_manifest provenance")
    manifest_path = benchmark_dir / _as_text(manifest.get("path"))
    if not manifest_path.is_file():
        raise ValueError(f"semantic graph task manifest is missing: {manifest_path}")
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != manifest.get("sha256"):
        raise ValueError("semantic graph is stale: task_groups.json hash changed")
    _validate_source_entries(graph.get("source_reference_sets"), benchmark_dir, "source_reference_sets")
    _validate_source_entries(graph.get("source_relation_tables"), benchmark_dir, "source_relation_tables")
    seed_dois = graph.get("task_seed_dois")
    if not isinstance(seed_dois, dict) or not seed_dois:
        raise ValueError("semantic graph has no per-task seed exclusion map")


def canonical_product_identity(asset: dict[str, Any]) -> str:
    explicit = asset.get("explicit_ids", [])
    if len(explicit) == 1:
        return "id:" + str(explicit[0]).strip().casefold()
    name = unicodedata.normalize("NFKC", str(asset.get("dataset_name", ""))).casefold()
    # Only explicit version/resolution identities are merged; SRTM3 alone can
    # mean 3 arc seconds and must not be interpreted as version 3.
    srtm_name = name.replace("shuttle radar topography mission", "srtm")
    srtm_patterns = (
        r"srtmgl1\s*(?:v(?:ersion)?\.?\s*)?3(?:\.0)?",
        r"srtm\s+v(?:ersion)?\s*3(?:\.0)?\s+global\s+1\s+arc\s+second\s*\(srtmgl1\)",
        r"srtm\s+global\s+1\s+arc\s+second\s+v0*3\s+(?:digital elevation model|dem)",
    )
    if not explicit and any(re.fullmatch(pattern, srtm_name) for pattern in srtm_patterns):
        return "known:srtmgl1-v3"
    worldcover = re.fullmatch(r"(?:esa[ /])?worldcover(?:\s+(?:land cover classification|land cover))?\s*(?:10\s*m\s*)?(?:2020\s*)?(?:\(?version\s*)?v?(100|200)\)?", name)
    if worldcover and not explicit:
        return "known:esa-worldcover-v" + worldcover[1]
    # Parentheses may contain meaningful product qualifiers. Do not discard
    # derived/composite/tile identities or conflicting version statements.
    if re.search(r"gap.fill|derived|extracted|corrected|merged|\b[nsew]\d{2,3}\b|\bwith\b|\band\b", name):
        return "node:" + asset["product_id"]
    name = name.replace("advanced spaceborne thermal emission and reflection radiometer", "aster")
    name = name.replace("advanced spaceborne thermal elevation model", "aster")
    name = name.replace("global digital elevation model", "gdem")
    name = re.sub(r"\(gdem\)", "", name)
    name = re.sub(r"\((?:gdem[, ]*)?(?:version\s*|v)0*([123])(?:,\s*\d{4})?\)", r"v\1", name)
    name = re.sub(r"\((?:aster\s*)?(?:gdem|astgtm(?:\.0*3)?|ast30|meti\s*&\s*nasa,\s*\d{4})\)", "", name)
    name = re.sub(r"\bversion\s*", "v", name)
    name = re.sub(r"\belevation data\b", "", name)
    name = re.sub(r"\b30\s*m\b", "", name)
    compact = re.sub(r"[\s,.-]+", "", name)
    match = re.fullmatch(r"astergdemv?0*([123])", compact)
    if match and not explicit:
        return "known:aster-gdem-v" + match[1]
    return "node:" + asset["product_id"]


def deduplicate_product_graph(graph: dict[str, Any]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for asset in graph["products"]:
        groups.setdefault(canonical_product_identity(asset), []).append(asset)
    remap = {}
    products = []
    product_to_instances = {}
    for identity, members in groups.items():
        members = sorted(members, key=lambda item: item["product_id"])
        representative = dict(members[0])
        product_id = representative["product_id"]
        aliases = [item["product_id"] for item in members]
        remap.update({alias: product_id for alias in aliases})
        instances = sorted({iid for alias in aliases for iid in graph["product_to_instances"].get(alias, [])})
        representative.update(canonical_identity=identity, alias_product_ids=aliases,
                              alias_names=[item["dataset_name"] for item in members],
                              instance_count=len(instances))
        representative["members"] = [entry for item in members for entry in item.get("members", [])]
        representative["member_count"] = len(representative["members"])
        products.append(representative)
        product_to_instances[product_id] = instances
    instances = [{**item, "product_ids": sorted({remap[pid] for pid in item.get("product_ids", [])})}
                 for item in graph["research_instances"]]
    relations = [{**item, "source_asset_id": remap[item["source_asset_id"]],
                  "target_asset_id": remap[item["target_asset_id"]]} for item in graph["relations"]]
    instance_map = {item["instance_id"]: item for item in instances}
    for product in products:
        names = product["alias_names"]
        mentions = {name.casefold() for name in names if len(name) >= 4}
        mentions.update(token.casefold() for name in names
                        for token in re.findall(r"\(([A-Z][A-Z0-9-]{3,})\)", name))
        evidence = []
        for iid in product_to_instances[product["product_id"]]:
            instance = instance_map[iid]
            quotes = instance.get("evidence", {}).get("joint_use_evidence_quotes", [])
            sentences = re.split(r"[。\n]|(?<=\.)\s+", instance.get("variables_usage", ""))
            for quote in [*quotes, *sentences]:
                if isinstance(quote, str) and len(quote) <= 1200 and any(
                        re.search(r"(?<!\w)" + re.escape(mention) + r"(?!\w)", quote.casefold())
                        for mention in mentions):
                    entry = {"instance_id": iid, "quote": quote}
                    if entry not in evidence:
                        evidence.append(entry)
        product["identity_evidence"] = evidence[:8]
    return {**graph, "products": products, "product_to_instances": product_to_instances,
            "research_instances": instances, "relations": relations}


def scope_graph_for_task(graph: dict[str, Any], task_id: str) -> tuple[dict[str, Any], str]:
    seed_dois = graph.get("task_seed_dois")
    if not isinstance(seed_dois, dict) or not isinstance(seed_dois.get(task_id), str):
        raise ValueError(f"semantic graph has no seed DOI for {task_id}")
    seed_doi = seed_dois[task_id]
    instances = [
        item
        for item in graph.get("research_instances", [])
        if isinstance(item, dict) and item.get("paper_id") != seed_doi
    ]
    instance_ids = {item["instance_id"] for item in instances}
    product_to_instances = {
        product_id: kept
        for product_id, instance_list in graph.get("product_to_instances", {}).items()
        if (kept := [instance_id for instance_id in instance_list if instance_id in instance_ids])
    }
    products = [
        item
        for item in graph.get("products", [])
        if isinstance(item, dict) and item.get("product_id") in product_to_instances
    ]
    relations = [
        item
        for item in graph.get("relations", [])
        if isinstance(item, dict)
        and item.get("source_asset_id") in product_to_instances
        and item.get("target_asset_id") in product_to_instances
    ]
    scoped = {
        **graph,
        "research_instances": instances,
        "products": products,
        "relations": relations,
        "product_to_instances": product_to_instances,
    }
    return deduplicate_product_graph(scoped), seed_doi


def retrieval_field_text(item: dict[str, Any], field: str) -> str:
    text = _as_text(item.get(field))
    if field == "dataset_name" and item.get("identity_evidence"):
        text = " | ".join([text, *[entry["quote"] for entry in item["identity_evidence"]]])
    return text


def collect_embedding_texts(
    components: list[dict[str, Any]],
    assets: list[dict[str, Any]],
) -> list[str]:
    texts = {
        retrieval_field_text(item, field)
        for item in [*components, *assets]
        for field in SEMANTIC_FIELDS
        if retrieval_field_text(item, field)
    }
    return sorted(texts)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def initial_candidate_pairs(
    components: list[dict[str, Any]],
    assets: list[dict[str, Any]],
    embeddings: dict[str, list[float]],
    config: RetrievalConfig,
) -> list[dict[str, Any]]:
    pairs = []
    for component in components:
        scored = []
        for asset in assets:
            weighted_sum = 0.0
            total_weight = 0.0
            field_scores = {}
            for field in SEMANTIC_FIELDS:
                weight = config.field_weights[field]
                query_text = _as_text(component.get(field))
                asset_text = retrieval_field_text(asset, field)
                if query_text and asset_text:
                    if query_text not in embeddings or asset_text not in embeddings:
                        raise ValueError("embedding map does not cover all semantic field texts")
                    similarity = (_cosine(np.asarray(embeddings[query_text]), np.asarray(embeddings[asset_text])) + 1.0) / 2.0
                else:
                    similarity = config.unmatched_penalty
                field_scores[field] = similarity
                weighted_sum += weight * similarity
                total_weight += weight
            score = weighted_sum / total_weight
            scored.append((score, _as_text(asset.get("product_id")), field_scores))
        scored.sort(key=lambda item: (-item[0], item[1]))
        asset_map = {item["product_id"]: item for item in assets}
        for score, asset_id, field_scores in scored[: config.initial_candidate_limit]:
            pairs.append(
                {
                    "component_id": component["component_id"],
                    "asset_id": asset_id,
                    "retrieval_score": score,
                    "field_scores": field_scores,
                    "selection_rank": len([item for item in pairs if item["component_id"] == component["component_id"]]) + 1,
                    "alias_product_ids": asset_map[asset_id].get("alias_product_ids", [asset_id]),
                    "candidate_pool_size": len(scored),
                    "first_excluded_candidate": ({"asset_id": scored[config.initial_candidate_limit][1],
                                                  "retrieval_score": scored[config.initial_candidate_limit][0]}
                                                 if len(scored) > config.initial_candidate_limit else None),
                }
            )
    return pairs


def validate_pair_reviews(
    requested_pairs: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expected = {(item["component_id"], item["asset_id"]) for item in requested_pairs}
    received = {
        (item.get("component_id"), item.get("asset_id"))
        for item in reviews
        if isinstance(item, dict)
    }
    if received != expected or len(reviews) != len(expected):
        counts = {}
        for item in reviews:
            if isinstance(item, dict):
                key = (item.get("component_id"), item.get("asset_id"))
                counts[key] = counts.get(key, 0) + 1
        details = {
            "missing_pairs": sorted(expected - received, key=repr),
            "unexpected_pairs": sorted(received - expected, key=repr),
            "duplicate_pairs": [{"pair": key, "count": count} for key, count in counts.items() if count > 1],
            "expected_count": len(expected), "returned_count": len(reviews),
        }
        error = ValueError("cross-encoder reviews do not exactly cover requested component/asset pairs: "
                           + json.dumps(details, ensure_ascii=False))
        error.pair_diagnostics = details
        raise error
    normalized = []
    for item in reviews:
        similarity = item.get("semantic_similarity")
        if isinstance(similarity, bool) or not isinstance(similarity, (int, float)) or not 0 <= float(similarity) <= 1:
            raise ValueError("cross-encoder semantic_similarity must be in [0, 1]")
        relation = item.get("relation_type")
        if relation not in RELATION_TYPES:
            raise ValueError(f"unsupported cross-encoder relation_type: {relation!r}")
        for key in ("observable_consistent", "role_consistent", "material_conflict"):
            if not isinstance(item.get(key), bool):
                raise ValueError(f"cross-encoder {key} must be boolean")
        normalized.append({**item, "semantic_similarity": float(similarity)})
    return normalized


def _hungarian_minimize(cost: list[list[float]]) -> list[int]:
    if not cost:
        return []
    row_count = len(cost)
    column_count = len(cost[0])
    if row_count > column_count:
        raise ValueError("Hungarian solver requires rows <= columns")
    u = [0.0] * (row_count + 1)
    v = [0.0] * (column_count + 1)
    p = [0] * (column_count + 1)
    way = [0] * (column_count + 1)
    for row in range(1, row_count + 1):
        p[0] = row
        column0 = 0
        minimum = [float("inf")] * (column_count + 1)
        used = [False] * (column_count + 1)
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = float("inf")
            column1 = 0
            for column in range(1, column_count + 1):
                if used[column]:
                    continue
                current = cost[row0 - 1][column - 1] - u[row0] - v[column]
                if current < minimum[column]:
                    minimum[column] = current
                    way[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    column1 = column
            for column in range(column_count + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    assignment = [-1] * row_count
    for column in range(1, column_count + 1):
        if p[column]:
            assignment[p[column] - 1] = column - 1
    return assignment


def maximum_weight_assignment(
    row_ids: list[str],
    column_ids: list[str],
    weights: dict[tuple[str, str], float],
    unmatched_penalty: float,
) -> list[tuple[str, str | None, float]]:
    if not row_ids:
        return []
    augmented_columns = [*column_ids, *[f"__unmatched_{index}" for index in range(len(row_ids))]]
    unavailable = unmatched_penalty - 1_000_000.0
    matrix = []
    maximum = max([unmatched_penalty, *weights.values()])
    for row_id in row_ids:
        row_weights = []
        for column_id in augmented_columns:
            if column_id.startswith("__unmatched_"):
                row_weights.append(unmatched_penalty)
            else:
                row_weights.append(weights.get((row_id, column_id), unavailable))
        matrix.append([maximum - value for value in row_weights])
    assignment = _hungarian_minimize(matrix)
    result = []
    for row_index, column_index in enumerate(assignment):
        column_id = augmented_columns[column_index]
        if column_id.startswith("__unmatched_"):
            result.append((row_ids[row_index], None, unmatched_penalty))
        else:
            result.append((row_ids[row_index], column_id, weights[(row_ids[row_index], column_id)]))
    return result


def _role_weight(component: dict[str, Any], config: RetrievalConfig) -> float:
    return config.role_weights.get(_as_text(component.get("analysis_role")), config.role_weights["*"])


def _is_anchor_component(component: dict[str, Any]) -> bool:
    return _as_text(component.get("origin")).startswith("prompt.")


def _classify_match_kind(relation_type: str) -> str:
    if relation_type == "EXACT":
        return MATCH_KIND_EXACT
    if relation_type == "HAS_EQUIVALENT":
        return MATCH_KIND_EQUIVALENT
    return MATCH_KIND_APPROXIMATE


GRAPH_EXPANSION_ORIGIN = "graph_expansion"


def _match_from_review(review: dict[str, Any], config: RetrievalConfig) -> dict[str, Any] | None:
    # role_consistent is recorded for traceability only; research usage is not
    # part of product identity and must not gate the pairing.
    if review["material_conflict"] or not review["observable_consistent"]:
        return None
    if review["semantic_similarity"] < config.entity_similarity_threshold:
        return None
    relation = review["relation_type"]
    if relation == "NONE" or relation in NON_SUBSTITUTING_RELATIONS:
        return None
    graph_proposed = review.get("verify_origin") == GRAPH_EXPANSION_ORIGIN
    match = {
        "component_id": review["component_id"],
        "asset_id": review["asset_id"],
        "score": review["semantic_similarity"],
        "relation_type": relation,
        "match_kind": _classify_match_kind(relation),
        "identity_retained": relation in IDENTITY_RETAINING_RELATIONS,
        "expansion": "graph_verified" if graph_proposed else "direct",
        "admission_basis": (
            "graph_proposed_cross_encoder_verified" if graph_proposed else "cross_encoder_identity_verified"
        ),
        "role_consistent": review.get("role_consistent"),
        "fallback_depth": 0,
        "relation_path": [],
        "rationale": _as_text(review.get("rationale")),
        "evidence_origin": "cross_encoder",
    }
    if graph_proposed:
        depth = review.get("graph_fallback_depth")
        match["fallback_depth"] = depth if isinstance(depth, int) and not isinstance(depth, bool) else 0
        match["relation_path"] = list(review.get("graph_relation_path") or [])
        match["graph_relation_hint"] = review.get("graph_relation_hint")
    return match


def _relation_adjacency(
    graph: dict[str, Any],
    config: RetrievalConfig,
    relations: frozenset[str] = SUBSTITUTION_VERIFYING_RELATIONS,
) -> dict[str, list[tuple[str, str]]]:
    traversable = relations & set(config.allowed_relation_types)
    adjacency: dict[str, list[tuple[str, str]]] = {}
    for relation in graph.get("relations", []):
        if not isinstance(relation, dict):
            continue
        relation_type = relation.get("relation_type")
        if relation_type not in traversable:
            continue
        source = _as_text(relation.get("source_asset_id"))
        target = _as_text(relation.get("target_asset_id"))
        if source and target:
            adjacency.setdefault(source, []).append((target, relation_type))
            if relation_type == "HAS_EQUIVALENT":
                adjacency.setdefault(target, []).append((source, relation_type))
    return adjacency


def _best_first_expansion(
    seed_matches: list[dict[str, Any]],
    graph: dict[str, Any],
    config: RetrievalConfig,
    relations: frozenset[str],
) -> list[dict[str, Any]]:
    adjacency = _relation_adjacency(graph, config, relations)
    queue: list[tuple[float, int, str, float, tuple[str, ...]]] = []
    best_cost: dict[str, float] = {}
    for match in seed_matches:
        asset_id = match["asset_id"]
        heapq.heappush(queue, (0.0, 0, asset_id, match["score"], ()))
        best_cost[asset_id] = 0.0
    discovered = []
    while queue:
        cost, depth, asset_id, semantic_score, path = heapq.heappop(queue)
        if depth >= config.maximum_fallback_depth:
            continue
        for target, relation_type in adjacency.get(asset_id, []):
            next_depth = depth + 1
            next_cost = cost + config.relation_costs[relation_type]
            if next_cost >= best_cost.get(target, float("inf")):
                continue
            best_cost[target] = next_cost
            next_path = (*path, relation_type)
            discovered.append(
                {
                    "asset_id": target,
                    "relation_type": relation_type,
                    "relation_path": list(next_path),
                    "fallback_depth": next_depth,
                    "score": semantic_score / (1.0 + next_cost),
                }
            )
            heapq.heappush(queue, (next_cost, next_depth, target, semantic_score, next_path))
    return discovered


def _graph_fallback_matches(
    component_id: str,
    seed_matches: list[dict[str, Any]],
    graph: dict[str, Any],
    config: RetrievalConfig,
) -> list[dict[str, Any]]:
    results = []
    for item in _best_first_expansion(seed_matches, graph, config, SUBSTITUTION_VERIFYING_RELATIONS):
        if item["score"] < config.entity_similarity_threshold:
            continue
        results.append(
            {
                "component_id": component_id,
                "asset_id": item["asset_id"],
                "score": item["score"],
                "relation_type": item["relation_type"],
                "match_kind": MATCH_KIND_EQUIVALENT,
                "identity_retained": True,
                "expansion": "graph_fallback",
                "admission_basis": "equivalence_relation_chain",
                "role_consistent": None,
                "fallback_depth": item["fallback_depth"],
                "relation_path": item["relation_path"],
                "rationale": "explicit typed graph fallback",
                "evidence_origin": "knowledge_graph_relation",
            }
        )
    return results


def propose_expansion_pairs(
    components: list[dict[str, Any]],
    graph: dict[str, Any],
    mapping: dict[str, Any],
    pair_reviews: list[dict[str, Any]],
    config: RetrievalConfig,
) -> list[dict[str, Any]]:
    """Graph-discovered endpoints for still-missing essential components.

    The graph relation only surfaces a candidate; it is never admitted here.
    Every proposal must go back through the Cross-Encoder, whose verdict alone
    decides relation_type, match_kind and identity_retained.
    """
    if any(math.isclose(item["essential_coverage"], 1.0)
           for item in mapping.get("candidate_instance_matches", [])):
        return []
    essential = {item["component_id"] for item in components if item["essential"]}
    # Gaps are measured against the representative single instance, not the
    # distributed union: the goal is to complete one real research instance.
    covered_single = {item["component_id"] for item in mapping.get("dataset_match_paths", [])}
    targets = sorted(essential - covered_single)
    if not targets:
        return []
    reviewed = {(item.get("component_id"), item.get("asset_id")) for item in pair_reviews}
    seeds_by_component: dict[str, list[dict[str, Any]]] = {}
    for review in pair_reviews:
        match = _match_from_review(review, config)
        if match is not None and match["component_id"] in targets:
            seeds_by_component.setdefault(match["component_id"], []).append(match)
    traversable = frozenset(config.allowed_relation_types)
    proposals = []
    seen: set[tuple[str, str]] = set()
    for component_id in targets:
        seeds = seeds_by_component.get(component_id, [])
        if not seeds:
            continue
        for item in _best_first_expansion(seeds, graph, config, traversable):
            key = (component_id, item["asset_id"])
            if key in reviewed or key in seen:
                continue
            seen.add(key)
            proposals.append(
                {
                    "component_id": component_id,
                    "asset_id": item["asset_id"],
                    "retrieval_score": item["score"],
                    "verify_origin": GRAPH_EXPANSION_ORIGIN,
                    "graph_relation_hint": item["relation_type"],
                    "graph_relation_path": item["relation_path"],
                    "graph_fallback_depth": item["fallback_depth"],
                }
            )
    return proposals


def _instance_assignments(
    graph: dict[str, Any],
    components: list[dict[str, Any]],
    matches: list[dict[str, Any]],
    config: RetrievalConfig,
) -> list[dict[str, Any]]:
    products = {item["product_id"]: item for item in graph.get("products", [])}
    instances = {item["instance_id"]: item for item in graph.get("research_instances", [])}
    product_to_instances = graph.get("product_to_instances", {})
    matches_by_instance: dict[str, list[dict[str, Any]]] = {}
    # A product is a global node; one verified match fans out to every research
    # instance that actually used that product.
    for match in matches:
        if match["asset_id"] not in products:
            continue
        for instance_id in product_to_instances.get(match["asset_id"], []):
            if instance_id in instances:
                matches_by_instance.setdefault(instance_id, []).append(match)
    component_map = {item["component_id"]: item for item in components}
    component_ids = list(component_map)
    anchor_ids = [cid for cid in component_ids if _is_anchor_component(component_map[cid])]
    # Dual tasks keep their existing identity-retention gate. Three-or-more
    # anchor tasks may retrieve a substantive local reference direction through
    # a verified compatible product without claiming a complete combination.
    partial_direction_policy = len(anchor_ids) >= 3
    retention_required = len(anchor_ids) == 2
    results = []
    for instance_id, instance_matches in matches_by_instance.items():
        instance = instances[instance_id]
        asset_ids = list(instance.get("product_ids", []))
        best_pair: dict[tuple[str, str], dict[str, Any]] = {}
        for match in instance_matches:
            key = (match["component_id"], match["asset_id"])
            if key not in best_pair or match["score"] > best_pair[key]["score"]:
                best_pair[key] = match
        weights = {key: value["score"] for key, value in best_pair.items()}
        assignment = maximum_weight_assignment(component_ids, asset_ids, weights, config.unmatched_penalty)
        assigned = []
        for component_id, asset_id, score in assignment:
            if asset_id is None or (component_id, asset_id) not in best_pair:
                continue
            match = best_pair[(component_id, asset_id)]
            assigned.append(
                {
                    **match,
                    # members live in the graph snapshot; keep match records compact
                    "dataset_asset": {key: value for key, value in products[asset_id].items() if key != "members"},
                    "is_anchor_component": component_id in anchor_ids,
                }
            )
        covered_map = {item["component_id"]: item for item in assigned}
        covered = set(covered_map)
        total_weight = sum(_role_weight(item, config) for item in components)
        essential_total = sum(_role_weight(item, config) for item in components if item["essential"])
        covered_weight = sum(_role_weight(component_map[item], config) for item in covered)
        essential_covered = sum(
            _role_weight(component_map[item], config)
            for item in covered
            if component_map[item]["essential"]
        )
        essential_coverage = essential_covered / essential_total if essential_total else 1.0
        overall_coverage = covered_weight / total_weight if total_weight else 1.0
        essential_assigned = [item for item in assigned if component_map[item["component_id"]]["essential"]]
        kinds = {item["match_kind"] for item in essential_assigned}
        retained_anchors = sorted(cid for cid in anchor_ids if cid in covered_map and covered_map[cid]["identity_retained"])
        replaced_anchors = sorted(cid for cid in anchor_ids if cid in covered_map and not covered_map[cid]["identity_retained"])
        anchor_retention_ok = (not retention_required) or bool(retained_anchors)
        # Alignment eligibility is gated on the task anchors only: a reference
        # instance's product list may omit auxiliary data the paper used, so a
        # missing supplemental component lowers coverage but does not disqualify.
        # Without any anchor component the gate falls back to every essential one.
        gate_ids = anchor_ids or [cid for cid in component_ids if component_map[cid]["essential"]]
        anchors_covered = all(cid in covered_map for cid in gate_ids)
        mapping_status = None
        if anchors_covered:
            if not anchor_retention_ok:
                mapping_status = "anchor_identity_not_retained"
            elif kinds and kinds == {MATCH_KIND_EXACT}:
                mapping_status = "exact_instance_match"
            elif MATCH_KIND_APPROXIMATE in kinds:
                mapping_status = "approximate_substitution_match"
            else:
                mapping_status = "equivalent_combination_match"
            if anchor_retention_ok and not math.isclose(essential_coverage, 1.0):
                mapping_status = "anchor_only_match"
        elif partial_direction_policy and any(cid in covered_map for cid in anchor_ids):
            # Missing task inputs describe reference scope, not direction irrelevance.
            mapping_status = "partial_reference_direction_match"
        results.append(
            {
                "instance_id": instance_id,
                "paper_id": instance.get("paper_id"),
                "task_id": instance.get("task_id"),
                "direction_id": instance.get("direction_id"),
                "mapping_status": mapping_status,
                "entity_match_confidence": sum(item["score"] for item in assigned) / len(assigned) if assigned else 0.0,
                "essential_coverage": essential_coverage,
                "overall_coverage": overall_coverage,
                "covered_components": sorted(covered),
                "missing_components": [item for item in component_ids if item not in covered],
                "match_kinds": {item["component_id"]: item["match_kind"] for item in essential_assigned},
                "approximate_substitution_components": sorted(
                    item["component_id"] for item in essential_assigned if item["match_kind"] == MATCH_KIND_APPROXIMATE
                ),
                "anchor_components": list(anchor_ids),
                "retained_anchor_components": retained_anchors,
                "replaced_anchor_components": replaced_anchors,
                "anchor_retention_required": retention_required,
                "anchor_retention_ok": anchor_retention_ok,
                "reference_direction_policy": "partial_anchor_research_v1" if partial_direction_policy else "complete_anchor_combination",
                "dataset_match_paths": assigned,
                "fallback_depth": max((item["fallback_depth"] for item in assigned), default=0),
                "conflicts": [],
                "reference_instance": instance,
            }
        )
    return sorted(
        results,
        key=lambda item: (-item["essential_coverage"], -item["overall_coverage"], -item["entity_match_confidence"], item["instance_id"]),
    )


def _minimum_evidence_bundle(
    assignments: list[dict[str, Any]],
    components: list[dict[str, Any]],
    config: RetrievalConfig,
) -> tuple[list[dict[str, Any]], set[str]]:
    essential_ids = [item["component_id"] for item in components if item["essential"]]
    bit_for = {component_id: 1 << index for index, component_id in enumerate(essential_ids)}
    target = (1 << len(essential_ids)) - 1
    candidates = []
    for assignment in assignments:
        mask = 0
        for component_id in assignment["covered_components"]:
            mask |= bit_for.get(component_id, 0)
        if mask:
            candidates.append((mask, assignment))
    states: dict[int, tuple[list[dict[str, Any]], float]] = {0: ([], 0.0)}
    for _ in range(config.maximum_evidence_size):
        next_states = dict(states)
        for mask, (bundle, confidence) in states.items():
            used = {item["instance_id"] for item in bundle}
            for candidate_mask, candidate in candidates:
                if candidate["instance_id"] in used:
                    continue
                merged = mask | candidate_mask
                proposed = ([*bundle, candidate], confidence + candidate["entity_match_confidence"])
                current = next_states.get(merged)
                if current is None or len(proposed[0]) < len(current[0]) or (
                    len(proposed[0]) == len(current[0]) and proposed[1] > current[1]
                ):
                    next_states[merged] = proposed
        states = next_states
        if target in states:
            break
    if target in states:
        bundle = states[target][0]
    else:
        best_mask, (bundle, _) = max(
            states.items(),
            key=lambda item: (item[0].bit_count(), -len(item[1][0]), item[1][1]),
        )
        target = best_mask
    covered = {
        component_id
        for component_id, bit in bit_for.items()
        if target & bit
    }
    return bundle, covered


def evaluate_mapping(
    graph: dict[str, Any],
    components: list[dict[str, Any]],
    pair_reviews: list[dict[str, Any]],
    config: RetrievalConfig,
) -> dict[str, Any]:
    direct_matches = []
    fallback_matches = []
    typed_fallback_by_component: dict[str, list[dict[str, Any]]] = {}
    review_seeds: dict[str, list[dict[str, Any]]] = {}
    component_ids = [item["component_id"] for item in components]
    for review in pair_reviews:
        match = _match_from_review(review, config)
        if match is None:
            continue
        review_seeds.setdefault(match["component_id"], []).append(match)
        # Graph-verified matches carry a non-zero fallback_depth for provenance,
        # but they were judged by the Cross-Encoder and are usable immediately.
        if match["expansion"] == "graph_fallback":
            typed_fallback_by_component.setdefault(match["component_id"], []).append(match)
        else:
            direct_matches.append(match)

    assignments = _instance_assignments(graph, components, direct_matches, config)
    full = [item for item in assignments if item["mapping_status"] in ALIGNMENT_ELIGIBLE_STATUSES]
    lock_source = next(
        (item for item in assignments if item["mapping_status"] != "anchor_identity_not_retained"),
        None,
    )
    locked_components = set(lock_source["covered_components"]) if lock_source else set()
    # Graph expansion still runs until some eligible instance covers every
    # essential component; anchor-only eligibility must not shrink coverage.
    if not any(math.isclose(item["essential_coverage"], 1.0) for item in full):
        missing = [item for item in component_ids if item not in locked_components]
        for component_id in missing:
            seeds = review_seeds.get(component_id, [])
            fallback_matches.extend(typed_fallback_by_component.get(component_id, []))
            fallback_matches.extend(_graph_fallback_matches(component_id, seeds, graph, config))
        assignments = _instance_assignments(graph, components, [*direct_matches, *fallback_matches], config)
        full = [item for item in assignments if item["mapping_status"] in ALIGNMENT_ELIGIBLE_STATUSES]

    if full:
        representative = full[0]
        covered = set(representative["covered_components"])
        missing = list(representative["missing_components"])
        mapping_status = representative["mapping_status"]
        evidence_bundle = [representative["instance_id"]]
    else:
        identity_blocked = next(
            (item for item in assignments if item["mapping_status"] == "anchor_identity_not_retained"),
            None,
        )
        bundle, distributed_covered = _minimum_evidence_bundle(assignments, components, config)
        essential = {item["component_id"] for item in components if item["essential"]}
        best = assignments[0] if assignments else None
        globally_supported = {
            item["component_id"]
            for item in [*direct_matches, *fallback_matches]
        }
        if identity_blocked is not None:
            mapping_status = "anchor_identity_not_retained"
            covered = set(identity_blocked["covered_components"])
            evidence_bundle = [identity_blocked["instance_id"]]
            representative = identity_blocked
        elif bundle and essential.issubset(distributed_covered) and len(bundle) > 1:
            mapping_status = "distributed_compositional_support"
            covered = set().union(*(set(item["covered_components"]) for item in bundle))
            evidence_bundle = [item["instance_id"] for item in bundle]
            representative = best
        elif best and best["covered_components"]:
            mapping_status = "partial_instance_match"
            covered = set(best["covered_components"])
            evidence_bundle = [best["instance_id"]]
            representative = best
        elif any(item not in globally_supported for item in essential):
            mapping_status = "unsupported_component"
            covered = globally_supported
            evidence_bundle = []
            representative = None
        else:
            mapping_status = "none"
            covered = set()
            evidence_bundle = []
            representative = None
        missing = [item for item in component_ids if item not in covered]

    component_map = {item["component_id"]: item for item in components}
    essential_total = sum(_role_weight(item, config) for item in components if item["essential"])
    total = sum(_role_weight(item, config) for item in components)
    essential_covered = sum(
        _role_weight(component_map[item], config)
        for item in covered
        if component_map[item]["essential"]
    )
    overall_covered = sum(_role_weight(component_map[item], config) for item in covered)
    return {
        "mapping_status": mapping_status,
        "matched_instance_id": None,
        "matched_paper_id": None,
        "candidate_instance_matches": full,
        "covered_components": sorted(covered),
        "missing_components": missing,
        "essential_coverage": essential_covered / essential_total if essential_total else 1.0,
        "overall_coverage": overall_covered / total if total else 1.0,
        "dataset_match_paths": representative["dataset_match_paths"] if representative else [],
        "fallback_depth": representative["fallback_depth"] if representative else 0,
        "evidence_bundle": evidence_bundle,
        **_identity_trace(representative, components),
        "max_alignment": None,
        "top3_instance_ids": [],
        "top3_alignment_mean": None,
        "topk_alignment_mean": None,
        "top3_available": False,
        "existing_idea_hit": False,
        "complete_combination_hit": False,
    }


def _identity_trace(
    representative: dict[str, Any] | None,
    components: list[dict[str, Any]],
) -> dict[str, Any]:
    anchor_ids = [item["component_id"] for item in components if _is_anchor_component(item)]
    if representative is None:
        return {
            "match_kinds": {},
            "approximate_substitution_components": [],
            "anchor_retention": {
                "required": len(anchor_ids) == 2,
                "ok": None,
                "anchor_components": anchor_ids,
                "retained": [],
                "replaced": [],
            },
        }
    return {
        "match_kinds": dict(representative.get("match_kinds", {})),
        "approximate_substitution_components": list(representative.get("approximate_substitution_components", [])),
        "anchor_retention": {
            "required": bool(representative.get("anchor_retention_required")),
            "ok": bool(representative.get("anchor_retention_ok")),
            "anchor_components": list(representative.get("anchor_components", anchor_ids)),
            "retained": list(representative.get("retained_anchor_components", [])),
            "replaced": list(representative.get("replaced_anchor_components", [])),
        },
    }


def validate_alignment_rows(
    candidates: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    dimensions = ("research_question", "hypothesis_mechanism", "variables_usage", "method_paradigm")
    expected = {item["instance_id"] for item in candidates}
    received = {item.get("instance_id") for item in rows if isinstance(item, dict)}
    if received != expected or len(rows) != len(expected):
        raise ValueError("alignment rows do not exactly cover eligible candidate instances")
    normalized = []
    for row in rows:
        scores = row.get("scores")
        if not isinstance(scores, dict) or set(scores) != set(dimensions):
            raise ValueError("alignment scores must define exactly four dimensions")
        normalized_scores = {}
        for key, value in scores.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 1 <= float(value) <= 10:
                raise ValueError(f"alignment {key} must be in [1, 10]")
            normalized_scores[key] = float(value)
        normalized.append(
            {
                "instance_id": row["instance_id"],
                "scores": normalized_scores,
                "rationales": row.get("rationales") if isinstance(row.get("rationales"), dict) else {},
                "alignment": min(normalized_scores.values()),
                "dimension_mean": sum(normalized_scores.values()) / len(normalized_scores),
            }
        )
    return normalized


def apply_alignment(
    mapping: dict[str, Any],
    rows: list[dict[str, Any]],
    config: RetrievalConfig,
) -> dict[str, Any]:
    candidates = mapping["candidate_instance_matches"]
    if not candidates:
        if rows:
            raise ValueError("alignment rows were provided for an ineligible mapping")
        return mapping
    normalized = validate_alignment_rows(candidates, rows)
    alignment_by_id = {item["instance_id"]: item for item in normalized}
    ranked = []
    for candidate in candidates:
        row = alignment_by_id[candidate["instance_id"]]
        ranked.append({**candidate, **row})
    ranked.sort(
        key=lambda item: (
            -item["alignment"],
            _STATUS_RANK.get(item["mapping_status"], len(_STATUS_RANK)),
            -item["dimension_mean"],
            -item["entity_match_confidence"],
            item["instance_id"],
        )
    )
    best = ranked[0]
    top = ranked[:3]
    topk_mean = sum(item["alignment"] for item in top) / len(top)
    top3_available = len(ranked) >= 3
    result = {
        **mapping,
        "mapping_status": best["mapping_status"],
        "matched_instance_id": best["instance_id"],
        "matched_paper_id": best["paper_id"],
        "candidate_instance_matches": ranked,
        "covered_components": best["covered_components"],
        "missing_components": best["missing_components"],
        "essential_coverage": best["essential_coverage"],
        "overall_coverage": best["overall_coverage"],
        "dataset_match_paths": best["dataset_match_paths"],
        "fallback_depth": best["fallback_depth"],
        "evidence_bundle": [best["instance_id"]],
        **_identity_trace(best, []),
        "max_alignment": best["alignment"],
        "top3_instance_ids": [item["instance_id"] for item in top],
        "top3_alignment_mean": topk_mean if top3_available else None,
        "topk_alignment_mean": topk_mean,
        "top3_available": top3_available,
        "existing_idea_hit": (
            best["mapping_status"] in FULL_MATCH_STATUSES | {"partial_reference_direction_match"}
            and not best["conflicts"]
            and best["alignment"] >= config.alignment_threshold
        ),
        "complete_combination_hit": (
            best["mapping_status"] in FULL_MATCH_STATUSES
            and not best["conflicts"]
            and best["alignment"] >= config.alignment_threshold
        ),
    }
    if result["mapping_status"] not in MAPPING_STATUSES:
        raise AssertionError("invalid mapping status")
    return result


def batched(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def embed_texts(
    texts: list[str],
    batch_size: int,
    embed_batch: Callable[[list[str]], list[list[float]]],
) -> dict[str, list[float]]:
    result = {}
    for batch in batched(texts, batch_size):
        vectors = embed_batch(batch)
        if len(vectors) != len(batch):
            raise ValueError("embedding provider returned the wrong vector count")
        for text, vector in zip(batch, vectors):
            if not isinstance(vector, list) or not vector:
                raise ValueError("embedding vector must be a non-empty list")
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in vector):
                raise ValueError("embedding vector entries must be numeric")
            result[text] = [float(value) for value in vector]
    dimensions = {len(vector) for vector in result.values()}
    if len(dimensions) > 1:
        raise ValueError("embedding vectors have inconsistent dimensions")
    return result

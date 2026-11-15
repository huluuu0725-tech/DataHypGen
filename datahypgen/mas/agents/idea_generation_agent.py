"""Generate fixed-type research ideas for the initial planning path."""
import asyncio
import copy
import json
import logging
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from .base_agent import AgentExecutionError, BaseAgent
from .plan_utils import build_idea_dataset_context
from datahypgen.prompts import IDEA_GENERATION_SYSTEM_PROMPT, build_idea_generation_prompt


logger = logging.getLogger(__name__)

IDEA_TYPES = (
    {
        "idea_type": "state_reconstruction",
        "type_question": "What hidden state can the data help reconstruct?",
        "applicability": "Use complementary signals to infer an otherwise unobserved state or process.",
    },
    {
        "idea_type": "behavior_pattern",
        "type_question": "What spatiotemporal behavior pattern can the data reveal?",
        "applicability": "Study movement, clustering, transition, periodicity, disruption, or regime change.",
    },
    {
        "idea_type": "relationship_mechanism",
        "type_question": "What variable relationship or mechanism can the data explain?",
        "applicability": "Form a relationship, lag, mediation, or causal-mechanism hypothesis.",
    },
    {
        "idea_type": "anomaly_discrepancy",
        "type_question": "What anomaly or discrepancy can the data expose?",
        "applicability": "Use inconsistent observations as the scientific phenomenon itself.",
    },
    {
        "idea_type": "impact_assessment",
        "type_question": "What consequence or external impact can the data assess?",
        "applicability": "Link observed change to environmental, social, economic, safety, or policy outcomes.",
    },
)

LITERATURE_CONTEXT_MAX_CHARS = 14000
LITERATURE_CONTEXT_MAX_PAPERS = 16
LITERATURE_ABSTRACT_MAX_CHARS = 500
LITERATURE_DETAIL_MAX_CHARS = 320
LITERATURE_DATASET_MAX_ITEMS = 6
LITERATURE_ANCHOR_COUNT = 2
LITERATURE_DOMINANT_COUNT = 3
LITERATURE_UNDERREPRESENTED_COUNT = 2
LITERATURE_BRIDGE_COUNT = 1
LITERATURE_NEAR_DUPLICATE_JACCARD = 0.80
LITERATURE_MIN_RELEVANCE = 0.40
LITERATURE_DATASET_ANCHOR_HINT = (
    "dataset validation accuracy uncertainty resolution measurement benchmark calibration "
    "quality reliability error bias reference"
)
LITERATURE_DATASET_ANCHOR_TITLE_TERMS = frozenset({
    "accuracy", "benchmark", "calibration", "consistency", "dem", "error",
    "evaluation", "geomorphometry", "geomorphometric", "geomorpho90m", "quality", "reliability",
    "resolution", "uncertainty", "validation",
})
IDEA_TYPE_LITERATURE_HINTS = {
    "state_reconstruction": (
        "hidden latent state infer inference reconstruct reconstruction proxy estimate estimation"
    ),
    "behavior_pattern": (
        "spatial temporal distribution pattern cluster clustering transition trend dynamics regime"
    ),
    "relationship_mechanism": (
        "mechanism driver cause causal forcing mediation interaction anthropogenic climatic "
        "precipitation disturbance"
    ),
    "anomaly_discrepancy": (
        "anomaly discrepancy unexpected contrast divergence mismatch outlier natural anthropogenic"
    ),
    "impact_assessment": (
        "impact consequence hazard risk exposure vulnerability damage fatal fatalities casualty "
        "casualties mortality social economic safety policy"
    ),
}
IDEA_CANDIDATE_CONTEXT_MIN = 8
IDEA_CANDIDATE_CONTEXT_MAX = 12
GROUNDING_SOURCE_ROLES = (
    "base_dataset",
    "observable_component",
    "driver",
    "human_activity",
    "spatial_stratifier",
    "validation_reference",
)
IDEA_TYPE_AXIS_PREFERENCES = {
    "state_reconstruction": (
        "observable_component",
        "validation_reference",
        "driver",
        "spatial_stratifier",
        "base_dataset",
        "human_activity",
    ),
    "behavior_pattern": (
        "base_dataset",
        "spatial_stratifier",
        "driver",
        "observable_component",
        "human_activity",
        "validation_reference",
    ),
    "relationship_mechanism": (
        "driver",
        "human_activity",
        "observable_component",
        "spatial_stratifier",
        "validation_reference",
        "base_dataset",
    ),
    "anomaly_discrepancy": (
        "validation_reference",
        "base_dataset",
        "spatial_stratifier",
        "observable_component",
        "driver",
        "human_activity",
    ),
    "impact_assessment": (
        "human_activity",
        "spatial_stratifier",
        "driver",
        "observable_component",
        "validation_reference",
        "base_dataset",
    ),
}
LITERATURE_STOPWORDS = frozenset({
    "about", "after", "again", "against", "also", "among", "and", "are", "based",
    "before", "being", "between", "both", "can", "could", "data", "dataset", "datasets",
    "does", "each", "for", "from", "global", "has", "have", "into", "its", "may",
    "method", "model", "more", "most", "not", "our", "paper", "results", "study", "than",
    "that", "the", "their", "these", "this", "through", "using", "was", "were", "which",
    "with", "within", "would",
})


class IdeaGenerationAgent(BaseAgent):
    """The sole idea producer for the initial batch flow."""

    def __init__(self, model: Any, config: Dict[str, Any]):
        super().__init__(model, config)
        planning = (config.get("_global_config") or {}).get("planning") or {}
        self.ideas_per_type = int(planning.get("ideas_per_type", 3))
        self.max_tokens = int(config.get("idea_generation_max_tokens", 4096))

    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        mode = params.get("mode", "initial_batch")
        if mode != "initial_batch":
            raise AgentExecutionError(f"IdeaGenerationAgent does not support legacy mode: {mode}")

        dataset_info = context.get("dataset_info") or {}
        candidate_datasets = context.get("candidate_datasets") or []
        composite_component_roles = self._composite_component_roles(dataset_info)
        # 跨领域方向直接进入 idea 生成上下文：仅靠 suggested_research_hooks 间接传导
        # 会被下游的 hooks[:5] 截断挤掉，导致多元化信号在本阶段丢失。
        cross_domain_axes = self._cross_domain_axes(context)
        candidate_allocations, candidate_allocation_audit = self._allocate_candidate_contexts(
            candidate_datasets
        )
        batches = await asyncio.gather(*[
            self._generate_type(
                idea_type,
                dataset_info,
                context.get("literature_review") or [],
                candidate_allocations[idea_type["idea_type"]],
                cross_domain_axes,
                composite_component_roles,
            )
            for idea_type in IDEA_TYPES
        ])
        candidates: List[Dict[str, Any]] = []
        groundings_by_idea_id: Dict[str, List[Dict[str, str]]] = {}
        audit: List[Dict[str, Any]] = []
        for definition, batch in zip(IDEA_TYPES, batches):
            ideas, literature_selection, groundings, grounding_rejections = batch
            if len(ideas) != self.ideas_per_type:
                raise AgentExecutionError(
                    f"idea_type={definition['idea_type']} generated {len(ideas)} ideas; expected {self.ideas_per_type}"
                )
            candidates.extend(ideas)
            groundings_by_idea_id.update(groundings)
            assigned_candidates = candidate_allocations[definition["idea_type"]]
            audit.append({
                "idea_type": definition["idea_type"],
                "generated_idea_count": len(ideas),
                "idea_ids": [idea["idea_id"] for idea in ideas],
                "literature_context_selection": literature_selection,
                "candidate_context_selection": {
                    "selected_count": len(assigned_candidates),
                    "candidate_ids": [
                        candidate["id"] for candidate in assigned_candidates
                    ],
                },
                "idea_dataset_grounding": [
                    {
                        "idea_id": idea["idea_id"],
                        "sources": copy.deepcopy(
                            groundings_by_idea_id.get(idea["idea_id"], [])
                        ),
                    }
                    for idea in ideas
                ],
                "rejected_idea_dataset_grounding": copy.deepcopy(grounding_rejections),
            })

        application_plans = [
            self._application_entry(
                idea,
                index,
                groundings_by_idea_id.get(idea["idea_id"], []),
            )
            for index, idea in enumerate(candidates)
        ]
        return {
            "idea_candidates": candidates,
            "idea_dataset_grounding": [
                {
                    "idea_id": idea["idea_id"],
                    "sources": copy.deepcopy(
                        groundings_by_idea_id.get(idea["idea_id"], [])
                    ),
                }
                for idea in candidates
            ],
            "application_plans": application_plans,
            "planning_audit": {
                "fixed_idea_types": copy.deepcopy(list(IDEA_TYPES)),
                "ideas_per_type": self.ideas_per_type,
                "generated_idea_count": len(candidates),
                "idea_generation_audit": audit,
                "candidate_context_allocation": candidate_allocation_audit,
            },
            "metadata": {
                "idea_type_count": len(IDEA_TYPES),
                "ideas_per_type": self.ideas_per_type,
                "application_count": len(application_plans),
            },
        }

    async def _generate_type(
        self,
        definition: Dict[str, str],
        dataset_info: Dict[str, Any],
        literature_review: List[Any],
        candidate_datasets: Optional[List[Dict[str, Any]]] = None,
        cross_domain_axes: Optional[List[Dict[str, Any]]] = None,
        composite_component_roles: Optional[List[Dict[str, str]]] = None,
    ) -> tuple[
        List[Dict[str, Any]],
        Dict[str, Any],
        Dict[str, List[Dict[str, str]]],
        List[Dict[str, str]],
    ]:
        schema = self._schema(composite_component_roles)
        literature_context, literature_selection = self._select_literature_context(
            literature_review,
            definition,
            dataset_info,
        )
        result = await self._call_model(
            prompt=build_idea_generation_prompt(
                idea_type=definition["idea_type"],
                type_question=definition["type_question"],
                applicability=definition["applicability"],
                ideas_per_type=self.ideas_per_type,
                dataset_context_json=self._json(self._dataset_context(dataset_info)),
                literature_context_json=self._json(literature_context),
                candidate_datasets_json=self._json([
                    self._compact_candidate_card(candidate)
                    for candidate in (candidate_datasets or [])
                ]),
                cross_domain_axes_json=self._json(cross_domain_axes or []),
                composite_component_roles_json=self._json(composite_component_roles or []),
            ),
            system_prompt=IDEA_GENERATION_SYSTEM_PROMPT,
            schema=schema,
            temperature=0.35,
            max_tokens=self.max_tokens,
        )
        raw_ideas = result.get("ideas", []) if isinstance(result, dict) else []
        ideas: List[Dict[str, Any]] = []
        groundings: Dict[str, List[Dict[str, str]]] = {}
        grounding_rejections: List[Dict[str, str]] = []
        for index, raw_idea in enumerate(raw_ideas):
            idea = self._normalize_idea(
                raw_idea,
                definition["idea_type"],
                index + 1,
                composite_component_roles,
            )
            if not idea:
                continue
            ideas.append(idea)
            raw_grounding = (
                raw_idea.get("idea_dataset_grounding", [])
                if isinstance(raw_idea, dict)
                else []
            )
            accepted, rejected = self._normalize_grounding(
                raw_grounding,
                candidate_datasets or [],
                idea_id=idea["idea_id"],
            )
            groundings[idea["idea_id"]] = accepted
            grounding_rejections.extend(rejected)
        if grounding_rejections:
            logger.warning(
                "idea_type=%s dropped %d invalid grounding references",
                definition["idea_type"],
                len(grounding_rejections),
            )
        return ideas, literature_selection, groundings, grounding_rejections

    @classmethod
    def _select_literature_context(
        cls,
        literature_review: List[Any],
        definition: Dict[str, str],
        dataset_info: Dict[str, Any],
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        entries, duplicate_drops = cls._deduplicate_literature(literature_review)
        if not entries:
            return [], {
                "source_count": len(literature_review),
                "deduplicated_count": 0,
                "selected_count": 0,
                "context_chars": 2,
                "char_budget": LITERATURE_CONTEXT_MAX_CHARS,
                "selected_papers": [],
                "duplicate_drops": duplicate_drops,
            }

        document_tokens = [
            cls._literature_tokens(cls._paper_selection_text(entry["paper"]))
            for entry in entries
        ]
        idf = cls._idf(document_tokens)
        document_vectors = [cls._tfidf_vector(tokens, idf) for tokens in document_tokens]
        dataset_vector = cls._tfidf_vector(
            cls._literature_tokens(cls._flatten_text(dataset_info)),
            idf,
        )
        anchor_vector = cls._tfidf_vector(
            cls._literature_tokens(
                f"{cls._flatten_text(dataset_info)} {LITERATURE_DATASET_ANCHOR_HINT}"
            ),
            idf,
        )
        type_hint = IDEA_TYPE_LITERATURE_HINTS.get(str(definition.get("idea_type") or ""), "")
        type_vector = cls._tfidf_vector(
            cls._literature_tokens(f"{cls._flatten_text(definition)} {type_hint}"),
            idf,
        )
        dataset_similarities = [
            cls._cosine(vector, dataset_vector)
            for vector in document_vectors
        ]
        type_similarities = [
            cls._cosine(vector, type_vector)
            for vector in document_vectors
        ]
        dataset_relevance = cls._normalize_scores(dataset_similarities)
        anchor_relevance = cls._normalize_scores([
            cls._cosine(vector, anchor_vector)
            for vector in document_vectors
        ])
        type_relevance = cls._normalize_scores(type_similarities)

        pairwise = [
            [cls._cosine(left, right) for right in document_vectors]
            for left in document_vectors
        ]
        densities: List[float] = []
        for index, similarities in enumerate(pairwise):
            neighbors = sorted(
                (value for other, value in enumerate(similarities) if other != index),
                reverse=True,
            )[:4]
            densities.append(sum(neighbors) / len(neighbors) if neighbors else 0.0)
        commonness = cls._normalize_scores(densities)

        relevance: List[float] = []
        for index, entry in enumerate(entries):
            survey_score = cls._normalized_paper_score(entry["paper"].get("score"))
            relevance.append(
                0.45 * survey_score
                + 0.30 * dataset_relevance[index]
                + 0.25 * type_relevance[index]
            )

        eligible = [
            index
            for index, score in enumerate(relevance)
            if score >= LITERATURE_MIN_RELEVANCE
        ]
        if not eligible:
            eligible = list(range(len(entries)))

        selected: List[int] = []
        roles: Dict[int, str] = {}

        def current_cards() -> List[Dict[str, Any]]:
            return [cls._compact_paper_card(entries[index]["paper"], roles[index]) for index in selected]

        def add_ranked(
            count: int,
            role: str,
            base_scores: List[float],
            allowed: Optional[List[int]] = None,
        ) -> None:
            for _ in range(count):
                pool = eligible if allowed is None else allowed
                remaining = [index for index in pool if index not in roles]
                while remaining:
                    best = max(
                        remaining,
                        key=lambda index: (
                            base_scores[index]
                            - 0.20 * max(
                                (pairwise[index][chosen] for chosen in selected),
                                default=0.0,
                            ),
                            relevance[index],
                            -entries[index]["source_rank"],
                        ),
                    )
                    candidate_card = cls._compact_paper_card(entries[best]["paper"], role)
                    candidate_cards = [*current_cards(), candidate_card]
                    if (
                        len(candidate_cards) <= LITERATURE_CONTEXT_MAX_PAPERS
                        and len(cls._json(candidate_cards)) <= LITERATURE_CONTEXT_MAX_CHARS
                    ):
                        selected.append(best)
                        roles[best] = role
                        break
                    remaining.remove(best)
                else:
                    return

        anchor_scores = [
            0.75 * anchor_relevance[index] + 0.25 * relevance[index]
            for index in range(len(entries))
        ]
        dominant_scores = [
            0.70 * relevance[index] + 0.30 * commonness[index]
            for index in range(len(entries))
        ]
        underrepresented_scores = [
            0.40 * relevance[index]
            + 0.30 * (1.0 - commonness[index])
            + 0.30 * type_relevance[index]
            for index in range(len(entries))
        ]
        bridge_scores = [
            0.40 * relevance[index]
            + 0.50 * type_relevance[index]
            + 0.10 * (1.0 - commonness[index])
            for index in range(len(entries))
        ]

        anchor_candidates = [
            index
            for index in eligible
            if set(cls._literature_tokens(entries[index]["paper"].get("title")))
            & LITERATURE_DATASET_ANCHOR_TITLE_TERMS
        ]
        add_ranked(
            LITERATURE_ANCHOR_COUNT,
            "dataset_anchor",
            anchor_scores,
            allowed=anchor_candidates,
        )
        dominant_candidates = [
            index for index in eligible if commonness[index] >= 0.5
        ]
        underrepresented_candidates = [
            index for index in eligible if commonness[index] < 0.5
        ]
        add_ranked(
            LITERATURE_DOMINANT_COUNT,
            "dominant_scope",
            dominant_scores,
            allowed=dominant_candidates,
        )
        add_ranked(
            LITERATURE_UNDERREPRESENTED_COUNT,
            "underrepresented_scope",
            underrepresented_scores,
            allowed=underrepresented_candidates,
        )
        add_ranked(LITERATURE_BRIDGE_COUNT, "type_bridge", bridge_scores)
        add_ranked(LITERATURE_CONTEXT_MAX_PAPERS, "diversity_fill", relevance)

        cards = current_cards()
        selected_papers = [
            {
                "id": cls._paper_identity(entries[index]["paper"], entries[index]["source_rank"]),
                "title": str(entries[index]["paper"].get("title") or ""),
                "source_rank": entries[index]["source_rank"],
                "survey_score": entries[index]["paper"].get("score"),
                "coverage_role": roles[index],
                "relevance": round(relevance[index], 4),
                "dataset_relevance": round(dataset_relevance[index], 4),
                "type_relevance": round(type_relevance[index], 4),
                "commonness": round(commonness[index], 4),
            }
            for index in selected
        ]
        return cards, {
            "source_count": len(literature_review),
            "deduplicated_count": len(entries),
            "selected_count": len(cards),
            "context_chars": len(cls._json(cards)),
            "char_budget": LITERATURE_CONTEXT_MAX_CHARS,
            "max_papers": LITERATURE_CONTEXT_MAX_PAPERS,
            "selected_papers": selected_papers,
            "duplicate_drops": duplicate_drops,
        }

    @classmethod
    def _deduplicate_literature(
        cls,
        literature_review: List[Any],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        entries: List[Dict[str, Any]] = []
        drops: List[Dict[str, Any]] = []
        for source_index, raw in enumerate(literature_review):
            if not isinstance(raw, dict) or not str(raw.get("title") or "").strip():
                continue
            paper = copy.deepcopy(raw)
            title_tokens = set(cls._literature_tokens(paper.get("title")))
            doi = str(paper.get("doi") or "").strip().lower()
            duplicate = None
            for kept in entries:
                kept_paper = kept["paper"]
                kept_doi = str(kept_paper.get("doi") or "").strip().lower()
                kept_tokens = kept["title_tokens"]
                title_union = title_tokens | kept_tokens
                title_jaccard = (
                    len(title_tokens & kept_tokens) / len(title_union)
                    if title_union
                    else 0.0
                )
                if (doi and kept_doi and doi == kept_doi) or (
                    min(len(title_tokens), len(kept_tokens)) >= 5
                    and title_jaccard >= LITERATURE_NEAR_DUPLICATE_JACCARD
                ):
                    duplicate = kept
                    break
            if duplicate is not None:
                drops.append({
                    "id": cls._paper_identity(paper, source_index + 1),
                    "title": str(paper.get("title") or ""),
                    "duplicate_of": cls._paper_identity(
                        duplicate["paper"],
                        duplicate["source_rank"],
                    ),
                })
                continue
            entries.append({
                "paper": paper,
                "source_rank": source_index + 1,
                "title_tokens": title_tokens,
            })
        return entries, drops

    @staticmethod
    def _paper_identity(paper: Dict[str, Any], source_rank: int) -> str:
        return str(
            paper.get("doi")
            or paper.get("id")
            or paper.get("title")
            or f"paper_{source_rank}"
        )

    @classmethod
    def _paper_selection_text(cls, paper: Dict[str, Any]) -> str:
        return cls._flatten_text({
            "title": paper.get("title"),
            "abstract": paper.get("abstract"),
            "keywords": paper.get("keywords"),
            "methods": paper.get("methods"),
            "applications": paper.get("applications"),
            "datasets_used": paper.get("datasets_used"),
        })

    @classmethod
    def _compact_paper_card(cls, paper: Dict[str, Any], coverage_role: str) -> Dict[str, Any]:
        card: Dict[str, Any] = {
            "id": cls._paper_identity(paper, 0),
            "title": str(paper.get("title") or "").strip(),
            "coverage_role": coverage_role,
        }
        for field in ("year", "journal", "score"):
            if paper.get(field) not in (None, ""):
                card[field] = copy.deepcopy(paper[field])
        abstract = str(paper.get("abstract") or "").strip()
        if abstract:
            card["abstract"] = abstract[:LITERATURE_ABSTRACT_MAX_CHARS]
        for field in ("methods", "applications"):
            value = cls._flatten_text(paper.get(field)).strip()
            if value:
                card[field] = value[:LITERATURE_DETAIL_MAX_CHARS]
        datasets_used = paper.get("datasets_used")
        if isinstance(datasets_used, list):
            values = [str(item).strip() for item in datasets_used if str(item).strip()]
            if values:
                card["datasets_used"] = values[:LITERATURE_DATASET_MAX_ITEMS]
        return card

    @staticmethod
    def _flatten_text(value: Any) -> str:
        parts: List[str] = []

        def collect(item: Any) -> None:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
            elif isinstance(item, dict):
                for nested in item.values():
                    collect(nested)
            elif isinstance(item, list):
                for nested in item:
                    collect(nested)

        collect(value)
        return " ".join(parts)

    @staticmethod
    def _literature_tokens(value: Any) -> List[str]:
        text = str(value or "").lower().replace("_", " ").replace("-", " ")
        tokens = re.findall(r"[a-z0-9][a-z0-9]*|[\u4e00-\u9fff]{2,}", text)
        return [
            token
            for token in tokens
            if token not in LITERATURE_STOPWORDS
            and (not token.isascii() or len(token) >= 3)
        ]

    @staticmethod
    def _idf(documents: List[List[str]]) -> Dict[str, float]:
        document_frequency: Counter[str] = Counter()
        for tokens in documents:
            document_frequency.update(set(tokens))
        count = max(1, len(documents))
        return {
            token: math.log((count + 1) / (frequency + 1)) + 1.0
            for token, frequency in document_frequency.items()
        }

    @staticmethod
    def _tfidf_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
        counts = Counter(token for token in tokens if token in idf)
        if not counts:
            return {}
        total = sum(counts.values())
        vector = {
            token: (frequency / total) * idf[token]
            for token, frequency in counts.items()
        }
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if norm == 0:
            return {}
        return {token: value / norm for token, value in vector.items()}

    @staticmethod
    def _cosine(left: Dict[str, float], right: Dict[str, float]) -> float:
        if len(left) > len(right):
            left, right = right, left
        return sum(value * right.get(token, 0.0) for token, value in left.items())

    @staticmethod
    def _normalize_scores(values: List[float]) -> List[float]:
        if not values:
            return []
        minimum = min(values)
        maximum = max(values)
        if math.isclose(minimum, maximum):
            return [0.5 for _ in values]
        return [(value - minimum) / (maximum - minimum) for value in values]

    @staticmethod
    def _normalized_paper_score(value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.5
        return min(1.0, max(0.0, score / 10.0))

    @staticmethod
    def _candidate_final_score(candidate: Dict[str, Any]) -> float:
        decision = candidate.get("binding_decision") or {}
        try:
            return float(decision.get("final_score", 0.0))
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _allocate_candidate_contexts(
        cls,
        candidate_datasets: List[Dict[str, Any]],
    ) -> tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
        """Allocate 8-12 strong candidates per idea type while covering the batch pool."""
        strong_pool: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()
        for candidate in candidate_datasets:
            if not isinstance(candidate, dict):
                continue
            candidate_id = str(candidate.get("id") or "").strip()
            if (
                not candidate_id
                or candidate_id in seen_ids
                or str(candidate.get("status") or "") != "resolved_strong"
            ):
                continue
            seen_ids.add(candidate_id)
            strong_pool.append(copy.deepcopy(candidate))

        type_names = [definition["idea_type"] for definition in IDEA_TYPES]
        allocations: Dict[str, List[Dict[str, Any]]] = {
            idea_type: [] for idea_type in type_names
        }
        if not strong_pool:
            return allocations, {
                "pool_size": 0,
                "covered_candidate_count": 0,
                "per_type": {idea_type: [] for idea_type in type_names},
            }

        for index, candidate in enumerate(strong_pool):
            allocations[type_names[index % len(type_names)]].append(candidate)

        target_size = min(
            IDEA_CANDIDATE_CONTEXT_MAX,
            len(strong_pool),
            max(IDEA_CANDIDATE_CONTEXT_MIN, math.ceil(len(strong_pool) / len(type_names))),
        )
        for idea_type in type_names:
            preferences = IDEA_TYPE_AXIS_PREFERENCES[idea_type]
            preference_rank = {axis: index for index, axis in enumerate(preferences)}

            def rank(candidate: Dict[str, Any]) -> tuple[float, float, str]:
                axes = candidate.get("retrieval_axes") or [candidate.get("retrieval_axis")]
                axis_rank = min(
                    (preference_rank.get(str(axis), len(preferences)) for axis in axes if axis),
                    default=len(preferences),
                )
                return (
                    -float(axis_rank),
                    cls._candidate_final_score(candidate),
                    str(candidate.get("id") or ""),
                )

            selected_ids = {candidate["id"] for candidate in allocations[idea_type]}
            for candidate in sorted(strong_pool, key=rank, reverse=True):
                if len(allocations[idea_type]) >= target_size:
                    break
                if candidate["id"] in selected_ids:
                    continue
                allocations[idea_type].append(copy.deepcopy(candidate))
                selected_ids.add(candidate["id"])

        covered_ids = {
            candidate["id"]
            for candidates in allocations.values()
            for candidate in candidates
        }
        return allocations, {
            "pool_size": len(strong_pool),
            "covered_candidate_count": len(covered_ids),
            "per_type": {
                idea_type: [candidate["id"] for candidate in candidates]
                for idea_type, candidates in allocations.items()
            },
        }

    @staticmethod
    def _compact_candidate_card(candidate: Dict[str, Any]) -> Dict[str, Any]:
        card = {
            field: copy.deepcopy(candidate[field])
            for field in (
                "id",
                "title",
                "status",
                "retrieval_axes",
                "product_family",
                "observable",
                "source",
                "retrieval_channel",
                "dataset_source",
                "doi",
                "parameters",
                "data_url",
                "metadata_url",
                "is_collection",
                "source_preference",
                "naming_note",
                "provider",
                "type",
            )
            if candidate.get(field) not in (None, "", [], {})
        }
        detail = candidate.get("detail")
        if candidate.get("dataset_source") == "PANGAEA":
            card["relations"] = copy.deepcopy(candidate.get("relations", [])[:12])
            card["relation_count"] = len(candidate.get("relations", []))
        if isinstance(detail, dict):
            compact_detail = {
                field: copy.deepcopy(detail[field])
                for field in (
                    "detail_source",
                    "gsd",
                    "cadence",
                    "temporal_extent",
                    "spatial_extent",
                )
                if detail.get(field) not in (None, "", [], {})
            }
            description = str(detail.get("description") or "").strip()
            if description:
                compact_detail["description"] = description[:600]
            bands = detail.get("bands")
            if isinstance(bands, list) and bands:
                compact_detail["bands"] = copy.deepcopy(bands[:8])
            if compact_detail:
                card["detail"] = compact_detail
        return card

    @staticmethod
    def _normalize_grounding(
        raw_grounding: Any,
        assigned_candidates: List[Dict[str, Any]],
        *,
        idea_id: str,
    ) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """校验 grounding 引用，返回 (已接受, 已拒绝)。

        grounding 是 provenance 侧车，不是执行合同：StepPlan 的 `required_datasets`
        才是。一条幻觉的 candidate_id 不应该连坐整批生成——这个函数在 `_call_model`
        之后运行，抛错拿不到模型层重试，会让五类共 `5 * ideas_per_type` 条 idea 全部
        作废。因此非法引用逐条丢弃并记入审计，idea 语义本身保持不变。
        """
        assigned_by_id = {
            str(candidate.get("id") or ""): candidate
            for candidate in assigned_candidates
            if isinstance(candidate, dict) and str(candidate.get("id") or "")
        }
        normalized: List[Dict[str, str]] = []
        rejected: List[Dict[str, str]] = []
        seen_ids: set[str] = set()

        def reject(candidate_id: str, source_role: str, reason: str) -> None:
            rejected.append({
                "idea_id": idea_id,
                "candidate_id": candidate_id,
                "source_role": source_role,
                "reason": reason,
            })

        if not isinstance(raw_grounding, list):
            reject("", "", "grounding_not_a_list")
            return normalized, rejected

        for item in raw_grounding:
            if not isinstance(item, dict):
                reject("", "", "entry_not_an_object")
                continue
            candidate_id = str(item.get("candidate_id") or "").strip()
            source_role = str(item.get("source_role") or "").strip()
            if candidate_id not in assigned_by_id:
                reject(candidate_id, source_role, "candidate_not_assigned_to_idea_type")
                continue
            if str(assigned_by_id[candidate_id].get("status") or "") != "resolved_strong":
                reject(candidate_id, source_role, "candidate_not_resolved_strong")
                continue
            if source_role not in GROUNDING_SOURCE_ROLES:
                reject(candidate_id, source_role, "invalid_source_role")
                continue
            if candidate_id in seen_ids:
                continue
            seen_ids.add(candidate_id)
            normalized.append({
                "candidate_id": candidate_id,
                "source_role": source_role,
            })
        return normalized, rejected

    def _schema(self, composite_component_roles: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        text = {"type": "string", "maxLength": 400}
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "ideas": {
                    "type": "array",
                    "minItems": self.ideas_per_type,
                    "maxItems": self.ideas_per_type,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "idea_id": {"type": "string", "maxLength": 200},
                            "idea_type": {"type": "string", "maxLength": 200},
                            "idea_dataset_grounding": {
                                "type": "array",
                                "maxItems": 6,
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "candidate_id": {"type": "string", "maxLength": 300},
                                        "source_role": {
                                            "type": "string",
                                            "enum": list(GROUNDING_SOURCE_ROLES),
                                        },
                                    },
                                    "required": ["candidate_id", "source_role"],
                                },
                            },
                            "knowledge_anchor_and_motivation": {
                                "type": "object", "additionalProperties": False,
                                "properties": {"macro_phenomenon": text, "research_motivation": text},
                                "required": ["macro_phenomenon", "research_motivation"],
                            },
                            "data_synergy_and_gap": {
                                "type": "object", "additionalProperties": False,
                                "properties": {
                                    "single_source_limits": {
                                        "type": "array", "maxItems": 3,
                                        "items": {
                                            "type": "object", "additionalProperties": False,
                                            "properties": {"data_source": text, "limitation": text},
                                            "required": ["data_source", "limitation"],
                                        },
                                    },
                                    "synergy_opportunity": text,
                                },
                                "required": ["single_source_limits", "synergy_opportunity"],
                            },
                            "research_question": text,
                            "core_hypothesis": {
                                "type": "object", "additionalProperties": False,
                                "properties": {"directional_statement": text, "proxy_or_inference_claim": text},
                                "required": ["directional_statement", "proxy_or_inference_claim"],
                            },
                            "theoretical_mechanism": {
                                "type": "object", "additionalProperties": False,
                                "properties": {
                                    "mechanism_chain": text,
                                    "assumptions": {"type": "array", "maxItems": 3, "items": text},
                                },
                                "required": ["mechanism_chain", "assumptions"],
                            },
                            "computability_and_falsifiability": {
                                "type": "object", "additionalProperties": False,
                                "properties": {
                                    "observable_variable": text,
                                    "mechanism_relevant_contrast": text,
                                    "falsification_condition": text,
                                },
                                "required": [
                                    "observable_variable",
                                    "mechanism_relevant_contrast",
                                    "falsification_condition",
                                ],
                            },
                            "novelty_constraints_and_value": {
                                "type": "object", "additionalProperties": False,
                                "properties": {
                                    "system_value": text,
                                    "applicability_boundaries": {"type": "array", "maxItems": 3, "items": text},
                                },
                                "required": ["system_value", "applicability_boundaries"],
                            },
                        },
                        "required": [
                            "idea_id", "idea_type", "knowledge_anchor_and_motivation", "data_synergy_and_gap",
                            "research_question", "core_hypothesis", "theoretical_mechanism",
                            "computability_and_falsifiability", "novelty_constraints_and_value",
                            "idea_dataset_grounding",
                        ],
                    },
                },
            },
            "required": ["ideas"],
        }
        roles = [
            str(item.get("role") or "").strip()
            for item in (composite_component_roles or [])
            if isinstance(item, dict) and str(item.get("role") or "").strip()
        ]
        if roles:
            schema["properties"]["ideas"]["items"]["properties"]["component_usage"] = {
                "type": "array",
                "minItems": len(roles),
                "maxItems": len(roles),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "role": {"type": "string", "enum": roles},
                        "contribution": {"type": "string", "maxLength": 400},
                        "necessity": {"type": "string", "enum": ["required"]},
                    },
                    "required": ["role", "contribution", "necessity"],
                },
            }
            schema["properties"]["ideas"]["items"]["required"].append("component_usage")
        return schema

    @staticmethod
    def _normalize_idea(
        raw: Any,
        idea_type: str,
        ordinal: int,
        composite_component_roles: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        if not isinstance(raw, dict) or not str(raw.get("research_question") or "").strip():
            return {}
        idea = copy.deepcopy(raw)
        idea.pop("idea_dataset_grounding", None)
        required_roles = {
            str(item.get("role") or "").strip()
            for item in (composite_component_roles or [])
            if isinstance(item, dict) and str(item.get("role") or "").strip()
        }
        if required_roles:
            usage = idea.get("component_usage")
            if not isinstance(usage, list):
                return {}
            usage_roles = [
                str(item.get("role") or "").strip()
                for item in usage
                if isinstance(item, dict)
            ]
            if set(usage_roles) != required_roles or len(usage_roles) != len(required_roles):
                return {}
        idea["idea_id"] = f"{idea_type}_idea_{ordinal}"
        idea["idea_type"] = idea_type
        return idea

    @staticmethod
    def _application_entry(
        idea: Dict[str, Any],
        index: int,
        grounding: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        return {
            "idea_id": idea["idea_id"],
            "idea_type": idea["idea_type"],
            "priority": index + 1,
            "idea_candidate": copy.deepcopy(idea),
            "idea_candidates": [copy.deepcopy(idea)],
            "idea_dataset_grounding": copy.deepcopy(grounding or []),
        }

    @staticmethod
    def _dataset_context(dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        # 与 IdeaMaturationAgent 共用同一份实现，避免两侧看到不同的数据集事实。
        return build_idea_dataset_context(dataset_info)

    @staticmethod
    def _composite_component_roles(dataset_info: Dict[str, Any]) -> List[Dict[str, str]]:
        base_dataset = dataset_info.get("base_dataset") if isinstance(dataset_info, dict) else {}
        if not isinstance(base_dataset, dict) or base_dataset.get("type") != "composite_dataset":
            return []
        contract = base_dataset.get("combination_contract") or {}
        required_roles = contract.get("required_roles") if isinstance(contract, dict) else []
        required_roles = {
            str(role).strip() for role in (required_roles or []) if str(role).strip()
        }
        result = []
        for component in base_dataset.get("dataset_components") or []:
            if not isinstance(component, dict):
                continue
            role = str(component.get("role") or "").strip()
            if role and (not required_roles or role in required_roles):
                result.append({
                    "role": role,
                    "dataset_name": str(component.get("name") or "").strip(),
                    "data_path": str(component.get("source") or component.get("data_path") or "").strip(),
                })
        return result

    # 单条方向的字段上限：控制 prompt 体积，跨领域段只需方向与角色，不需检索词。
    _CROSS_DOMAIN_AXIS_MAX = 8
    _CROSS_DOMAIN_TEXT_MAX = 120

    @classmethod
    def _cross_domain_axes(cls, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract compact cross-domain direction cards from the dataset portrait.

        画像可能整体缺失（技能被关闭或推演失败），此时返回空列表，prompt 侧不渲染
        该段，行为与接线前一致。
        """
        portrait = context.get("dataset_portrait")
        if not isinstance(portrait, dict):
            return []
        raw_axes = portrait.get("cross_domain_axes")
        if not isinstance(raw_axes, list):
            return []

        cards: List[Dict[str, Any]] = []
        for axis_record in raw_axes:
            if not isinstance(axis_record, dict):
                continue
            domain_name = str(axis_record.get("domain_name") or "").strip()
            if not domain_name:
                continue
            card = {"domain_name": domain_name[: cls._CROSS_DOMAIN_TEXT_MAX]}
            role = str(axis_record.get("role") or "").strip()
            if role:
                card["role"] = role[: cls._CROSS_DOMAIN_TEXT_MAX]
            cards.append(card)
            if len(cards) >= cls._CROSS_DOMAIN_AXIS_MAX:
                break
        return cards

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

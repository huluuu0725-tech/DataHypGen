"""Build a diverse, provenance-preserving dataset inventory before idea generation."""

import copy
import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .dataset_resolution_semantics import infer_product_families
from .geo_data_search import GeoDataSearch

# GEE 本地 catalog 里有两类来源：GEE 官方与社区贡献。
GEE_CATALOG_SOURCES = {"GEE_OFFICIAL", "GEE_COMMUNITY"}


logger = logging.getLogger(__name__)

DEFAULT_TOP_N = 30
# 召回池必须显著大于每路输出上限，否则文本粗排就等于终选，
# `_evaluate_candidate_set` / `_rank_candidate_set` 的绑定判据没有重排空间。
# 50 沿用 `_build_local_resolution_intent` 的历史 search_k 默认值。
PER_QUERY_RECALL_POOL_K = 50
PER_QUERY_LOCAL_TOP_K = 10
PER_QUERY_ONLINE_TOP_K = 10
PER_QUERY_OSM_TOP_K = 10
PER_PRODUCT_FAMILY_LIMIT = 2
PER_OBSERVABLE_LIMIT = 2
TARGET_STRONG_PER_AXIS = 3
MAX_QUERIES_PER_AXIS = 4
# cross_domain 是唯一一条"多个领域共用一条轴"的轴：expander 推演出的每个领域
# 各带 1-2 个检索词，全部塞进同一个桶。用 4 的通用配额去截，等于让领域之间
# 抢名额——而 LLM 按熟悉度排列领域，先到的必然是最正典的那两个，真正的远域
# 被静默丢弃。实测 Geomorpho90m 推演 5 域 10 词只发出前 4 条，恰好是
# Hydrology/Geomorphology，而被饿死的 Agriculture 轴恰恰产出了该任务最高分的
# idea。这条轴因此单独给更高配额，并在下面按域轮转发放。
MAX_QUERIES_CROSS_DOMAIN = 8

BINDABLE_STATUSES = frozenset({"resolved_strong"})
NEAR_MISS_STATUSES = frozenset({"resolved_weak"})
EXISTENCE_EVIDENCE_STATUSES = frozenset({
    "resolved_strong",
    "resolved_weak",
    "unresolved_semantic",
})

RETRIEVAL_AXES = (
    "base_dataset",
    "observable_component",
    "driver",
    "human_activity",
    "spatial_stratifier",
    "validation_reference",
    # 跨领域推演技能产出的可结合方向，检索词来自 portrait.cross_domain_axes
    "cross_domain",
)

HYDROLOGY_MARKERS = frozenset({
    "grace",
    "groundwater",
    "hydrolog",
    "runoff",
    "terrestrial water",
    "water storage",
})
HYDROLOGY_COMPONENTS = (
    "soil moisture",
    "snow water equivalent",
    "surface water",
    "groundwater",
)
HYDROLOGY_DRIVERS = ("precipitation", "evapotranspiration", "runoff")
HYDROLOGY_HUMAN_ACTIVITY = (
    "irrigation",
    "groundwater pumping",
    "reservoir storage",
)
SPATIAL_STRATIFIERS = (
    "koppen geiger climate classification",
    "land cover",
    "river basin boundaries",
)

GENERAL_DRIVER_TERMS = (
    "precipitation",
    "evapotranspiration",
    "runoff",
    "temperature",
    "drought",
    "wind",
    "fire",
    "elevation",
    "slope",
)
GENERAL_HUMAN_TERMS = (
    "irrigation",
    "pumping",
    "reservoir",
    "withdrawal",
    "mining",
    "urbanization",
    "settlement",
    "roads",
    "protected areas",
)


class CandidateDatasetScout:
    """Retrieve per-query top-K candidates and build a diverse strong inventory."""

    def __init__(
        self,
        geo_search: Optional[GeoDataSearch] = None,
        enable_online_scouting: bool = True,
    ):
        self.geo_search = geo_search or GeoDataSearch(cache_enabled=True)
        self.enable_online_scouting = enable_online_scouting

    @staticmethod
    def _normalize_query(query: Any) -> str:
        return re.sub(r"\s+", " ", str(query or "").strip())

    @classmethod
    def _flatten_text(cls, value: Any) -> str:
        parts: List[str] = []

        def collect(item: Any) -> None:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
            elif isinstance(item, dict):
                for nested in item.values():
                    collect(nested)
            elif isinstance(item, (list, tuple)):
                for nested in item:
                    collect(nested)

        collect(value)
        return cls._normalize_query(" ".join(parts))

    @staticmethod
    def _data_description(base_dataset: Dict[str, Any]) -> Dict[str, Any]:
        description = base_dataset.get("data_description")
        return description if isinstance(description, dict) else {}

    @classmethod
    def _dataset_name(cls, base_dataset: Dict[str, Any]) -> str:
        description = cls._data_description(base_dataset)
        return cls._normalize_query(
            description.get("dataset_name") or base_dataset.get("dataset_name")
        )

    @classmethod
    def _variable_queries(cls, description: Dict[str, Any]) -> List[str]:
        variables = description.get("variables")
        if not isinstance(variables, list):
            return []
        queries: List[str] = []
        for variable in variables:
            text = cls._normalize_query(variable)
            name = text.split(":", 1)[0].strip().replace("_", " ")
            if name and len(name) >= 3 and name.lower() not in {item.lower() for item in queries}:
                queries.append(name)
        return queries[:MAX_QUERIES_PER_AXIS]

    @classmethod
    def build_query_axes(
        cls,
        base_dataset: Dict[str, Any],
        task_keywords: Any = None,
        cross_domain_axes: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, str]]:
        """Build six semantic retrieval axes from trusted dataset-profile fields."""
        base_dataset = base_dataset if isinstance(base_dataset, dict) else {}
        description = cls._data_description(base_dataset)
        dataset_name = cls._dataset_name(base_dataset)
        profile_text = cls._flatten_text({
            "base_dataset": base_dataset,
            "task_keywords": task_keywords,
        })
        if not profile_text:
            return []
        profile_text_lower = profile_text.lower()
        domain = cls._normalize_query(
            base_dataset.get("domain") or description.get("domain") or task_keywords
        )
        hydrology_context = any(marker in profile_text_lower for marker in HYDROLOGY_MARKERS)

        by_axis: Dict[str, List[str]] = {axis: [] for axis in RETRIEVAL_AXES}

        def axis_cap(axis: str) -> int:
            return MAX_QUERIES_CROSS_DOMAIN if axis == "cross_domain" else MAX_QUERIES_PER_AXIS

        def add(axis: str, query: Any) -> None:
            normalized = cls._normalize_query(query)
            if not normalized:
                return
            existing = {item.casefold() for item in by_axis[axis]}
            if normalized.casefold() not in existing and len(by_axis[axis]) < axis_cap(axis):
                by_axis[axis].append(normalized)

        # 多数据输入：合成容器名是 stage 拼出来的，在数据集目录里不存在——
        # task_052 实测 base_dataset / observable_component 两条轴 0 strong，
        # 候选池因此只剩 driver 等泛化轴拉来的资产，与输入组件同族或互补的
        # 数据一条没进池（5 个 idea 的 grounding 全部 0 条）。多数据下各成员
        # 才是真实资产：成员名进 base_dataset 轴，成员描述的变量进
        # observable_component 轴，与单数据路径的语义逐成员对齐。
        components = [
            component for component in (base_dataset.get("dataset_components") or [])
            if isinstance(component, dict)
        ]
        if base_dataset.get("type") == "composite_dataset" and len(components) >= 2:
            for component in components:
                add("base_dataset", cls._dataset_name(component))
                for query in cls._variable_queries(cls._data_description(component)):
                    add("observable_component", query)
        else:
            add("base_dataset", dataset_name or domain)

        if hydrology_context:
            for query in HYDROLOGY_COMPONENTS:
                add("observable_component", query)
            for query in HYDROLOGY_DRIVERS:
                add("driver", query)
            for query in HYDROLOGY_HUMAN_ACTIVITY:
                add("human_activity", query)
        else:
            for query in cls._variable_queries(description):
                add("observable_component", query)
            for term in GENERAL_DRIVER_TERMS:
                if term in profile_text_lower:
                    add("driver", term)
            for term in GENERAL_HUMAN_TERMS:
                if term in profile_text_lower:
                    add("human_activity", term)

        if not by_axis["observable_component"]:
            add("observable_component", f"{dataset_name or domain} observable")
        if not by_axis["driver"]:
            add("driver", f"{domain or dataset_name} environmental drivers")
        if not by_axis["human_activity"]:
            add("human_activity", f"{domain or dataset_name} human activity")

        for query in SPATIAL_STRATIFIERS:
            add("spatial_stratifier", query)

        if hydrology_context:
            add("validation_reference", "terrestrial water storage validation")
            add("validation_reference", "groundwater monitoring")
        add("validation_reference", f"{dataset_name or domain} validation reference")

        # 跨领域推演技能产出的可结合方向：检索词由 LLM 基于数据集语义动态生成，
        # 保证候选池能越过画像/文献锚定的单一领域。
        #
        # 按域轮转发放，而不是按域顺序灌入：先给每个域发第 1 个词，再回头发第 2 个。
        # 顺序灌入时配额会被排在前面的领域吃光，而 LLM 倾向于先列最熟悉（即最正典）
        # 的领域，于是远域被系统性饿死——这与"跨域"的目的正好相反。轮转保证配额
        # 不足时每个域至少各有一次机会，被牺牲的是各域的次要词而非整个域。
        cross_domain_term_lists = [
            [
                term
                for term in (axis_record.get("dataset_search_terms") or [])
                if cls._normalize_query(term)
            ]
            for axis_record in (cross_domain_axes or [])
            if isinstance(axis_record, dict)
        ]
        for round_index in range(max((len(terms) for terms in cross_domain_term_lists), default=0)):
            for terms in cross_domain_term_lists:
                if round_index < len(terms):
                    add("cross_domain", terms[round_index])

        return [
            {"axis": axis, "query": query}
            for axis in RETRIEVAL_AXES
            for query in by_axis[axis]
        ]

    async def _resolve_query(
        self,
        query_record: Dict[str, str],
    ) -> Tuple[Dict[str, str], List[Tuple[str, Dict[str, Any]]]]:
        query = query_record["query"]
        resolution_intent = {
            "query_text": query,
            "search_k": PER_QUERY_RECALL_POOL_K,
        }
        local_results = self.geo_search.rank_local_datasets(
            resolution_intent,
            limit=PER_QUERY_LOCAL_TOP_K,
        )
        osm_results = self.geo_search._search_osm_by_keyword(
            query,
            limit=PER_QUERY_OSM_TOP_K,
        )
        online_results = await self.geo_search.rank_online_datasets(
            resolution_intent,
            limit=PER_QUERY_ONLINE_TOP_K,
        )
        results: List[Tuple[str, Dict[str, Any]]] = [
            ("local", result) for result in (local_results or [])
        ]
        results.extend(("online", result) for result in (online_results or []))
        results.extend(
            (
                "osm",
                {
                    "status": "resolved_strong",
                    "candidate": feature,
                    "binding_decision": {
                        "accepted_for_binding": True,
                        "binding_grade": "resolved_strong",
                        "final_score": self._osm_relevance_score(feature),
                    },
                },
            )
            for feature in (osm_results or [])
        )
        return query_record, results

    @staticmethod
    def _osm_relevance_score(feature: Any) -> float:
        """OSM 命中沿用检索器算出的关键词相关度。

        写死 1.0 会让 OSM 无条件排在所有 GEE 强绑定候选之前（strong 门槛只有
        0.65），而 family/observable 限额拦不住它们：OSM 的 feature_id 不含 `/`，
        推出的 product_family 各不相同。`_filter_features_by_keyword` 已按
        `min_relevance_score` 过滤并写回真实分数，直接复用即可保持量纲可比。
        """
        properties = getattr(feature, "properties", None) or {}
        try:
            return float(properties.get("relevance_score", 0.0))
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _candidate_entry(
        cls,
        *,
        query_record: Dict[str, str],
        source_label: str,
        result: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        candidate = result.get("candidate")
        if candidate is None:
            return None
        props = getattr(candidate, "properties", None) or {}
        candidate_id = str(
            props.get("id")
            or props.get("ee_id")
            or getattr(candidate, "feature_id", "")
            or ""
        ).strip()
        if not candidate_id:
            return None
        title = str(props.get("title") or getattr(candidate, "name", "") or candidate_id).strip()
        candidate_text = " ".join(
            str(value or "").strip()
            for value in (
                title,
                candidate_id,
                props.get("provider"),
                " ".join(str(item) for item in (props.get("tags") or [])),
            )
            if str(value or "").strip()
        )
        inferred_families = sorted(infer_product_families(candidate_text))
        product_family = cls._normalize_query(
            props.get("product_family")
            or (inferred_families[0] if inferred_families else "")
            or props.get("provider")
            or candidate_id.split("/", 1)[0]
        ).lower()
        observable = cls._normalize_query(query_record["query"]).lower()
        status = str(result.get("status") or "").strip()
        # `source` 记的是检索通道（local / online / osm），`dataset_source` 才是
        # 数据集所属平台（GEE_OFFICIAL / NASA_CMR / PLANETARY_COMPUTER / ...）。
        # 两者必须都留：前者用于审计"这条是哪路召回的"，后者是下游判断该走哪个
        # 客户端、能不能用 ee API 的唯一依据。此前只留前者，于是 GEE 资产与
        # NASA_CMR collection 在 prompt 里都显示 source="local"，LLM 只能靠 id
        # 形态猜平台，实测会把 STAC collection 当 GEE asset 处理。
        dataset_source = str(props.get("source") or "").strip().upper()
        entry: Dict[str, Any] = {
            "query": query_record["query"],
            "retrieval_axis": query_record["axis"],
            "retrieval_axes": [query_record["axis"]],
            "retrieval_queries": [query_record["query"]],
            "source": source_label,
            "retrieval_channel": source_label,
            "dataset_source": dataset_source,
            "status": status,
            "title": title,
            "id": candidate_id,
            "type": props.get("type") or getattr(candidate, "feature_type", ""),
            "provider": props.get("provider"),
            "docs": props.get("docs"),
            "product_family": product_family,
            "observable": observable,
            "binding_decision": copy.deepcopy(result.get("binding_decision", {})),
        }
        inferred_preference = cls._infer_source_preference_from_dataset_source(
            dataset_source,
            source_label,
        )
        if dataset_source == "PANGAEA":
            for key in ("doi", "relations", "parameters", "metadata_url", "data_url", "is_collection", "access_note", "distributions", "conditions_of_access", "is_accessible_for_free", "collected_start", "collected_end"):
                entry[key] = copy.deepcopy(props.get(key))
            entry["detail"] = {
                "detail_source": "PANGAEA DOI metadata",
                "description": props.get("description", ""),
                "spatial_extent": props.get("bbox"),
            }
        if inferred_preference:
            entry["source_preference"] = inferred_preference
        if dataset_source == "PANGAEA":
            entry["naming_note"] = (
                "Use the PANGAEA DOI and source_preference='pangaea'. Access via its DOI landing page; "
                "typed relations describe provenance, supplements and dataset hierarchy, not proven joint empirical use. "
                "Do not construct Earth Engine or STAC calls for this dataset."
            )
        elif source_label == "osm" or dataset_source == "OSM_TAGINFO":
            entry["naming_note"] = (
                "Use a descriptive OpenStreetMap dataset name and source_preference='osm'; "
                "do not use the feature id as a dataset name."
            )
        elif dataset_source and dataset_source not in GEE_CATALOG_SOURCES:
            entry["naming_note"] = (
                f"This dataset lives in {dataset_source}, not Google Earth Engine. "
                "Access it through its own STAC/API client; do not write ee.* calls against it."
            )
        return entry

    # 平台 -> source_preference。source_preference 只用来路由检索后端，取值必须
    # 落在 ResourcesCheck._resolve_search_backend 认得的集合里。
    _DATASET_SOURCE_TO_PREFERENCE = {
        "PANGAEA": "pangaea",
        "GEE_OFFICIAL": "gee",
        "GEE_COMMUNITY": "gee",
        "NASA_CMR": "stac",
        "PLANETARY_COMPUTER": "stac",
        "EARTH_SEARCH": "stac",
        "OSM_TAGINFO": "osm",
    }

    @classmethod
    def _infer_source_preference_from_dataset_source(
        cls,
        dataset_source: str,
        source_label: str,
    ) -> str:
        """按真实平台推导检索后端偏好。"""
        preference = cls._DATASET_SOURCE_TO_PREFERENCE.get(dataset_source, "")
        if preference:
            return preference
        # 在线 GEE catalog 回退通道没有 catalog source 字段，但它确实是 GEE 资产。
        if source_label in {"osm"}:
            return "osm"
        if source_label == "online":
            return "gee"
        return ""

    @staticmethod
    def _merge_entry(existing: Dict[str, Any], incoming: Dict[str, Any]) -> None:
        for field in ("retrieval_axes", "retrieval_queries"):
            merged = list(existing.get(field) or [])
            for value in incoming.get(field) or []:
                if value not in merged:
                    merged.append(value)
            existing[field] = merged
        existing.setdefault("retrieval_hits", []).append({
            "axis": incoming.get("retrieval_axis", ""),
            "query": incoming.get("query", ""),
            "source": incoming.get("source", ""),
            "status": incoming.get("status", ""),
        })

    @staticmethod
    def _candidate_score(entry: Dict[str, Any]) -> float:
        decision = entry.get("binding_decision") or {}
        try:
            return float(decision.get("final_score", 0.0))
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _select_diverse_strong_candidates(
        cls,
        candidates: List[Dict[str, Any]],
        *,
        top_n: int,
    ) -> List[Dict[str, Any]]:
        ranked = sorted(
            candidates,
            key=lambda item: (
                cls._candidate_score(item),
                item.get("id", ""),
            ),
            reverse=True,
        )
        selected: List[Dict[str, Any]] = []
        selected_ids: set[str] = set()
        family_counts: Counter[str] = Counter()
        observable_counts: Counter[str] = Counter()

        def add(entry: Dict[str, Any]) -> bool:
            candidate_id = str(entry.get("id") or "")
            family = str(entry.get("product_family") or candidate_id).casefold()
            observable = str(entry.get("observable") or candidate_id).casefold()
            if candidate_id in selected_ids:
                return False
            if family_counts[family] >= PER_PRODUCT_FAMILY_LIMIT:
                return False
            if observable_counts[observable] >= PER_OBSERVABLE_LIMIT:
                return False
            selected.append(entry)
            selected_ids.add(candidate_id)
            family_counts[family] += 1
            observable_counts[observable] += 1
            return True

        # 轮转分配而非逐轴填满：顺序填充会让靠后的轴在 top_n 用尽时饿死
        # （7 轴 × TARGET_STRONG_PER_AXIS=3 需要 21 个名额，top_n<=18 时末位的
        # cross_domain 恒为 0）。轮转保证每轴拿到第 n 个之前，所有轴都已拿到第 n-1 个。
        axis_queues = {
            axis: [
                item for item in ranked if axis in (item.get("retrieval_axes") or [])
            ]
            for axis in RETRIEVAL_AXES
        }
        axis_cursors = {axis: 0 for axis in RETRIEVAL_AXES}

        def take_one(axis: str) -> bool:
            """Select the next still-admissible candidate for one axis.

            add() 的三个拒绝条件（已选中、产品族超限、观测量超限）都只会随选择推进而
            更加严格，因此被拒的条目不可能在后续轮次转为可选，游标直接前移即可。
            """
            queue = axis_queues[axis]
            index = axis_cursors[axis]
            while index < len(queue):
                entry = queue[index]
                index += 1
                if add(entry):
                    axis_cursors[axis] = index
                    return True
            axis_cursors[axis] = index
            return False

        for _ in range(TARGET_STRONG_PER_AXIS):
            if len(selected) >= top_n:
                break
            for axis in RETRIEVAL_AXES:
                if len(selected) >= top_n:
                    break
                take_one(axis)

        for entry in ranked:
            if len(selected) >= top_n:
                break
            add(entry)
        return selected

    async def scout(
        self,
        base_dataset: Dict[str, Any],
        task_keywords: Any = None,
        top_n: int = DEFAULT_TOP_N,
        accepted_statuses: Optional[frozenset] = None,
        cross_domain_axes: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Return strong inventory, weak near misses, and per-query retrieval audit."""
        empty = {
            "candidate_datasets": [],
            "near_miss_candidates": [],
            "query_audit": [],
            "axis_coverage": {},
        }
        if not self.enable_online_scouting:
            return empty

        query_axes = self.build_query_axes(
            base_dataset, task_keywords, cross_domain_axes=cross_domain_axes
        )
        if not query_axes:
            return empty
        batches = [
            await self._resolve_query(query_record)
            for query_record in query_axes
        ]

        strong_by_id: Dict[str, Dict[str, Any]] = {}
        weak_by_id: Dict[str, Dict[str, Any]] = {}
        unresolved_by_id: Dict[str, Dict[str, Any]] = {}
        query_audit: List[Dict[str, Any]] = []
        include_unresolved = bool(
            accepted_statuses and "unresolved_semantic" in accepted_statuses
        )

        for query_record, results in batches:
            counts: Counter[str] = Counter()
            source_counts: Counter[str] = Counter()
            for source_label, result in results:
                status = str(result.get("status") or "").strip()
                counts[status] += 1
                source_counts[source_label] += 1
                entry = self._candidate_entry(
                    query_record=query_record,
                    source_label=source_label,
                    result=result,
                )
                if entry is None:
                    continue
                candidate_id = entry["id"]
                target: Optional[Dict[str, Dict[str, Any]]] = None
                if status == "resolved_strong":
                    target = strong_by_id
                    weak_by_id.pop(candidate_id, None)
                    unresolved_by_id.pop(candidate_id, None)
                elif status == "resolved_weak" and candidate_id not in strong_by_id:
                    target = weak_by_id
                elif (
                    status == "unresolved_semantic"
                    and include_unresolved
                    and candidate_id not in strong_by_id
                    and candidate_id not in weak_by_id
                ):
                    target = unresolved_by_id
                if target is None:
                    continue
                if candidate_id in target:
                    self._merge_entry(target[candidate_id], entry)
                else:
                    entry["retrieval_hits"] = [{
                        "axis": query_record["axis"],
                        "query": query_record["query"],
                        "source": source_label,
                        "status": status,
                    }]
                    target[candidate_id] = entry
            query_audit.append({
                "axis": query_record["axis"],
                "query": query_record["query"],
                "local_top_k": PER_QUERY_LOCAL_TOP_K,
                "online_top_k": PER_QUERY_ONLINE_TOP_K,
                "source_result_counts": dict(source_counts),
                "status_counts": dict(counts),
            })

        strong_candidates = self._select_diverse_strong_candidates(
            list(strong_by_id.values()),
            top_n=max(1, int(top_n)),
        )
        strong_ids = {item["id"] for item in strong_candidates}
        axis_coverage = {
            axis: {
                "available_strong_count": sum(
                    axis in (candidate.get("retrieval_axes") or [])
                    for candidate in strong_by_id.values()
                ),
                "selected_strong_count": sum(
                    axis in (candidate.get("retrieval_axes") or [])
                    for candidate in strong_candidates
                ),
                "minimum_required": 2,
                "target": TARGET_STRONG_PER_AXIS,
            }
            for axis in RETRIEVAL_AXES
        }
        for coverage in axis_coverage.values():
            coverage["meets_minimum"] = coverage["selected_strong_count"] >= 2
        near_misses = sorted(
            (
                item for candidate_id, item in weak_by_id.items()
                if candidate_id not in strong_ids
            ),
            key=self._candidate_score,
            reverse=True,
        )[: max(1, int(top_n))]
        result = {
            "candidate_datasets": strong_candidates,
            "near_miss_candidates": near_misses,
            "query_audit": query_audit,
            "axis_coverage": axis_coverage,
        }
        if include_unresolved:
            result["unresolved_semantic_candidates"] = list(unresolved_by_id.values())[
                : max(1, int(top_n))
            ]
        return result

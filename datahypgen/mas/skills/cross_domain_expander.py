"""跨领域组合方向推演技能（Cross-Domain Expansion Skill）。

定位：数据集画像阶段用于丰富数据集自身语义；本技能在画像之后，
基于已丰富的语义进一步推演该数据集变量在地球系统中的可结合方向，
产出结构化的跨领域检索轴（cross_domain_axes），供文献检索、
候选数据集侦察与 idea 生成消费。

技能契约：
- 输入：dataset_portrait（优先，画像丰富语义）或 data_description（降级，原始描述）
- 输出：List[Dict]，每项含 domain_name / role / dataset_search_terms / literature_query
- 模型：单次 LLM 调用（generate_json），解析失败时返回空列表，不阻断 pipeline
"""

import logging
import re
from typing import Any, Dict, List, Optional

from ..workflow.data_type import (
    CROSS_DOMAIN_AXIS_MAX_SEARCH_TERMS,
    CROSS_DOMAIN_AXIS_ROLES,
    normalize_cross_domain_axes,
)

logger = logging.getLogger(__name__)

MAX_AXES = 8
MIN_AXES = 3
# 单轴检索词上限由契约层定义，避免技能与契约各持一份数字。
MAX_SEARCH_TERMS_PER_AXIS = CROSS_DOMAIN_AXIS_MAX_SEARCH_TERMS

# 提示词自身出现过的领域词。门检据此衡量产出是否只是复读示例：历史故障中
# 提示词用 slope 举例并列出四个领域，Geomorpho90m 产出的五个方向有四个与
# 该示例逐一对应。改动后示例只剩两条且明示"仅示范措辞"，此集合供门检核验。
PROMPT_EXAMPLE_DOMAIN_TOKENS = frozenset(
    {"crop", "irrigation", "nighttime", "lights", "economic"}
)

# 提取基变量词时忽略的通用词，避免把 "global"/"data" 当成变量名去过滤检索词。
_BASE_VARIABLE_STOPWORDS = frozenset({
    "global", "data", "dataset", "datasets", "map", "maps", "layer", "layers",
    "annual", "monthly", "daily", "version", "collection", "product", "products",
    "high", "resolution", "image", "imagery", "world", "the", "and", "for",
    "first", "second", "order", "derivatives", "index", "indices", "meter",
    "metre", "degree", "degrees",
})

_CROSS_DOMAIN_SYSTEM_PROMPT = (
    "You are a geospatial data scientist specialized in Earth system science. "
    "Your job is to reason about which other Earth system domains a given "
    "dataset's variable can meaningfully combine with, based on its physical "
    "semantics. Respond with JSON only."
)

_CROSS_DOMAIN_PROMPT = """Analyze the dataset profile below and identify {min_axes}-{max_axes} Earth system domains where this variable plays a scientifically meaningful role ({roles}).

DATASET PROFILE:
{profile_block}

RULES:
- Derive domains from the variable's PHYSICAL MEANING, not from words already present in the profile, and not from any example appearing in these instructions.
- Domains MUST span at least 3 distinct Earth system spheres (atmosphere, biosphere, hydrosphere, lithosphere, anthroposphere).
- Exclude the dataset's own primary domain (do not propose combinations that merely stay inside the same research field).
- `dataset_search_terms` name the OTHER domain's own data product — the thing you would have to obtain in order to study that domain. They MUST NOT contain this dataset's core variable term. A catalog query carrying this dataset's own variable name retrieves this dataset's field again, which is the one thing this axis is not for.
- Write each term the way that data product is titled in a catalog, not the way a research question is phrased: two to four words, the product type only. No research verbs (assess, control, validate, isolate), no qualifiers of resolution or period (high-resolution, annual, global, time-series), no words about what the data is for. Catalogs match on title and description tokens, so every extra word dilutes the signal. The transformation looks like:
    "annual crop water use to control for irrigation" -> "crop water use"
    "high-resolution nighttime lights as a proxy for economic activity" -> "nighttime lights"
  Those two lines illustrate PHRASING ONLY. Do not treat the domains they mention as a menu of domains to choose.
- Never give a named index or standard composite metric as a whole search term; name the measured quantity the index is built from instead.
- `literature_query`: one plain keyword query (4-10 words) for an academic search engine. Unlike the search terms, this one SHOULD carry the dataset's core variable alongside the target domain, because its purpose is to find work that already connects the two.

Output schema:
{{
  "cross_domain_axes": [
    {{
      "domain_name": "short domain label",
      "role": "{role_enum}",
      "dataset_search_terms": ["term1", "term2"],
      "literature_query": "one academic search query"
    }}
  ]
}}"""


class CrossDomainExpander:
    """跨领域推演技能：数据集语义 → 地球系统可结合方向。"""

    def __init__(self, model: Any):
        """model: 任意实现了 generate_json(prompt, schema, ...) 的模型客户端。"""
        self.model = model
        # expand() 每次调用都会重写；未跑过时保持 attempted=False，
        # 使"没跑"与"跑了没产出"在审计里可区分。
        self.last_run_audit: Dict[str, Any] = {
            "attempted": False,
            "status": "not_run",
            "axis_count": 0,
        }

    # ── 语义提取 ────────────────────────────────────────────────────────────

    @classmethod
    def _flatten(cls, value: Any, limit: int = 400) -> str:
        text = str(value or "").strip()
        return text[:limit]

    @classmethod
    def build_profile_block(
        cls,
        dataset_portrait: Optional[Dict[str, Any]],
        data_description: Dict[str, Any],
    ) -> str:
        """从画像（优先）或原始描述中提取语义特征，构造推演 prompt 输入。"""
        desc = data_description if isinstance(data_description, dict) else {}
        portrait = dataset_portrait if isinstance(dataset_portrait, dict) else {}

        raster_content = desc.get("raster_content") or {}
        band_semantics = raster_content.get("band_semantics") or desc.get("band_semantics") or {}
        band_text = "; ".join(
            f"{band}: {cls._flatten(semantic, 200)}"
            for band, semantic in list(band_semantics.items())[:3]
        )
        unit = desc.get("unit_of_measurement") or {}
        capabilities = [
            cls._flatten(item, 120)
            for item in (desc.get("analytical_capabilities") or [])[:6]
        ]

        lines = [
            f"- Dataset name: {cls._flatten(desc.get('dataset_name'))}",
            f"- Band semantics: {band_text or 'unknown'}",
            f"- Primary measurement: {cls._flatten(unit.get('primary_measurement'), 240)}",
            f"- Data type: {cls._flatten(desc.get('data_type'))} / {cls._flatten(desc.get('data_format'))}",
            f"- Temporal coverage: {cls._flatten(desc.get('temporal_coverage'), 240)}",
            f"- Spatial resolution: {cls._flatten(desc.get('spatial_resolution'))}",
            f"- Spatial coverage: {cls._flatten(desc.get('spatial_coverage'), 200)}",
            f"- Thematic group: {cls._flatten(raster_content.get('thematic_group'))}",
        ]
        if capabilities:
            lines.append("- Declared analytical capabilities: " + " | ".join(capabilities))

        # 画像增量：实测统计与画像摘要进一步约束推演
        portrait_text = cls._flatten(portrait.get("portrait_text"), 800)
        if portrait_text:
            lines.append(f"- Portrait summary: {portrait_text}")
        value_distribution = (portrait.get("analysis") or {}).get("value_distribution")
        if isinstance(value_distribution, dict) and value_distribution:
            lines.append(f"- Observed value distribution: {value_distribution}")
        axes = portrait.get("candidate_question_axes") or []
        if axes:
            axis_summaries = [
                cls._flatten((axis or {}).get("summary"), 160)
                for axis in axes[:4]
                if isinstance(axis, dict)
            ]
            lines.append("- Data-driven question axes: " + " | ".join(axis_summaries))

        return "\n".join(lines)

    # ── 结果校验 ────────────────────────────────────────────────────────────

    @staticmethod
    def base_variable_tokens(data_description: Optional[Dict[str, Any]]) -> set:
        """从数据集名与主测量量提取"基变量词"，用于确定性后检。

        提示词已要求 dataset_search_terms 不得携带本数据集的核心变量，但提示词
        是软约束。历史故障正是软约束失守：自锚检索词实测 0 strong / 2 weak，
        最佳命中是基数据集自己。这里做一道不依赖模型服从性的硬检查。
        """
        if not isinstance(data_description, dict):
            return set()
        unit = data_description.get("unit_and_measurement") or {}
        text = " ".join(
            str(value or "")
            for value in (
                data_description.get("dataset_name"),
                unit.get("primary_measurement") if isinstance(unit, dict) else "",
            )
        ).lower()
        return {
            token
            for token in re.findall(r"[a-z]{3,}", text)
            if token not in _BASE_VARIABLE_STOPWORDS
        }

    @classmethod
    def _strip_self_anchored_terms(
        cls,
        axes: List[Dict[str, Any]],
        base_tokens: set,
    ) -> tuple:
        """删除携带基变量词的检索词，返回 (净化后轴, 违规记录)。

        整条轴的检索词都违规时该轴被丢弃——留一条空词轴会在下游变成静默失败，
        而审计里能看到丢弃计数。literature_query 不参与过滤：按新规则它本就
        应当同时携带基变量与目标域。
        """
        if not base_tokens:
            return axes, []
        kept: List[Dict[str, Any]] = []
        violations: List[Dict[str, Any]] = []
        for axis in axes:
            terms = axis.get("dataset_search_terms") or []
            clean, bad = [], []
            for term in terms:
                hit = sorted(base_tokens & set(re.findall(r"[a-z]{3,}", str(term).lower())))
                (bad if hit else clean).append(term if not hit else {"term": term, "tokens": hit})
            if bad:
                violations.append({
                    "domain_name": axis.get("domain_name", ""),
                    "rejected_terms": bad,
                    "axis_dropped": not clean,
                })
            if clean:
                axis = dict(axis)
                axis["dataset_search_terms"] = clean
                kept.append(axis)
        return kept, violations

    @classmethod
    def _sanitize_axes(cls, raw: Any) -> List[Dict[str, Any]]:
        """Normalize model output through the CrossDomainAxis contract, then cap length."""
        if isinstance(raw, dict):
            raw = raw.get("cross_domain_axes")
        # 去重、字段归一、非法项丢弃由契约层统一负责，本技能只再施加数量上限。
        return normalize_cross_domain_axes(raw)[:MAX_AXES]

    # ── 技能入口 ────────────────────────────────────────────────────────────

    _OUTPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "cross_domain_axes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "domain_name": {"type": "string", "minLength": 1},
                        "role": {
                            "type": "string",
                            "enum": list(CROSS_DOMAIN_AXIS_ROLES),
                        },
                        "dataset_search_terms": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1},
                        },
                        "literature_query": {"type": "string", "minLength": 1},
                    },
                    "required": [
                        "domain_name",
                        "role",
                        "dataset_search_terms",
                        "literature_query",
                    ],
                },
            }
        },
        "required": ["cross_domain_axes"],
    }

    async def expand(
        self,
        dataset_portrait: Optional[Dict[str, Any]] = None,
        data_description: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """推演跨领域组合方向；失败时返回空列表，不阻断调用方。

        每次调用都会写 `self.last_run_audit`：本技能的失败此前只留一行 warning，
        产物里零痕迹，事后无法区分"没产出轴"与"根本没跑过"（实测有会话的画像
        里干脆没有 cross_domain_axes 键）。审计随画像持久化，让空轴可见可诊断。
        """
        self.last_run_audit = {
            "attempted": True,
            "status": "unknown",
            "axis_count": 0,
            "domains": [],
            "self_anchored_violations": [],
            "error": "",
        }
        if not isinstance(data_description, dict) or not data_description:
            logger.warning("CrossDomainExpander: empty data_description, skip expansion")
            self.last_run_audit.update(
                status="skipped_empty_data_description", attempted=False
            )
            return []

        prompt = _CROSS_DOMAIN_PROMPT.format(
            min_axes=MIN_AXES,
            max_axes=MAX_AXES,
            roles=", ".join(CROSS_DOMAIN_AXIS_ROLES),
            role_enum=" | ".join(CROSS_DOMAIN_AXIS_ROLES),
            profile_block=self.build_profile_block(dataset_portrait, data_description),
        )
        try:
            response = await self.model.generate_json(
                prompt=prompt,
                schema=self._OUTPUT_SCHEMA,
                system_prompt=_CROSS_DOMAIN_SYSTEM_PROMPT,
                default={"cross_domain_axes": []},
            )
        except Exception as exc:
            logger.warning("CrossDomainExpander model call failed: %s", exc)
            self.last_run_audit.update(status="model_call_failed", error=str(exc))
            return []

        axes = self._sanitize_axes(response)
        raw_axis_count = len(axes)
        base_tokens = self.base_variable_tokens(data_description)
        axes, violations = self._strip_self_anchored_terms(axes, base_tokens)
        if violations:
            logger.info(
                "CrossDomainExpander: dropped self-anchored search terms in %d/%d axes",
                len(violations),
                raw_axis_count,
            )

        self.last_run_audit.update(
            status="ok" if axes else "empty_after_sanitize",
            axis_count=len(axes),
            raw_axis_count=raw_axis_count,
            domains=[str(axis.get("domain_name") or "") for axis in axes],
            roles=[str(axis.get("role") or "") for axis in axes],
            self_anchored_violations=violations,
            base_variable_tokens=sorted(base_tokens),
        )
        logger.info(
            "CrossDomainExpander: %d cross-domain axes for %s",
            len(axes),
            str(data_description.get("dataset_name") or "unknown dataset"),
        )
        return axes

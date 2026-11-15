"""Idea maturation between type-level triage and detailed task planning."""
import asyncio
import copy
import json
import logging
import math
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from .base_agent import BaseAgent
from .plan_utils import build_idea_dataset_context
from .socratic_idea_dual_tree import (
    AXIS_BASIS_SCORE_MAP,
    BRANCH_BUDGET_PER_ROUND,
    PersistentIdeaSocraticTree,
    REVISION_ACTIONABLE_VERDICTS,
    SocraticIdeaDualTree,
)
from datahypgen.mas.tools.literature_query import StructuredLiteratureQuery
from datahypgen.mas.tools.directional_stat_probe import DirectionalStatProbe
from datahypgen.mas.tools.candidate_dataset_scout import (
    EXISTENCE_EVIDENCE_STATUSES,
    CandidateDatasetScout,
)
from datahypgen.mas.tools.dataset_detail_enricher import DatasetDetailEnricher
from datahypgen.mas.tools.filtered_literature_cache import FilteredLiteratureCache
from datahypgen.mas.tools.literature_search import (
    LiteratureSearch,
    PaperMetadata,
    publication_year_is_allowed,
    validate_max_publication_year,
)
from datahypgen.prompts import (
    IDEA_EVIDENCE_CHANNEL_SYSTEM_PROMPT,
    IDEA_EVIDENCE_RELEVANCE_SYSTEM_PROMPT,
    IDEA_MATURATION_REVISION_SYSTEM_PROMPT,
    IDEA_MATURATION_ACCEPTANCE_SYSTEM_PROMPT,
    IDEA_SELECTION_FV_SYSTEM_PROMPT,
    build_idea_evidence_channel_prompt,
    build_idea_evidence_relevance_prompt,
    build_idea_maturation_revision_prompt,
    build_idea_maturation_acceptance_prompt,
    build_idea_selection_fv_prompt,
)


logger = logging.getLogger(__name__)


IDEA_FIELDS = (
    "knowledge_anchor_and_motivation",
    "data_synergy_and_gap",
    "research_question",
    "core_hypothesis",
    "theoretical_mechanism",
    "computability_and_falsifiability",
    "novelty_constraints_and_value",
)
IDEA_IDENTITY_FIELDS = ("idea_id", "idea_type", "component_usage")

FALSIFIABILITY_FIELD = "computability_and_falsifiability"
# 判别性对照所在的子字段。混淆型 finding 要求这一项真的变化，改 observable 或
# falsification 措辞都不算——只有对照本身能分开竞争解释。
DISCRIMINATING_CONTRAST_SUBFIELD = "mechanism_relevant_contrast"
CLAIM_SEMANTIC_FIELDS = {
    "research_question",
    "core_hypothesis",
    "theoretical_mechanism",
    FALSIFIABILITY_FIELD,
}
# 这两个字段承载同一个方向性主张：问题问"X 升高时 Y 是否升高"，假设必须回答同一方向。
# 可共同授权用于语义对齐；是否一致由最终内容判断，不要求两者同时改写。
DIRECTIONAL_CLAIM_PAIR = {"research_question", "core_hypothesis"}
# 语义验收判 idea_fields_coherent 时要求问题、假设、机制三者断言同一方向，所以改写授权
# 必须覆盖整条链：只放开问题与假设却冻结机制，任何真正改变方向的改写都会与旧机制矛盾，
# 撞上 idea_incoherent 而永远落不了地。
DIRECTIONAL_CLAIM_CHAIN = {*DIRECTIONAL_CLAIM_PAIR, "theoretical_mechanism"}
# 一次改写最多放开这么多非主张类字段（data_synergy_and_gap / novelty_constraints_and_value
# 等），避免把单点缺陷的修订扩成整个 idea 重写。
MAX_SUPPORTING_REVISION_FIELDS = 2
# 被驳回的 attempt 携带超过这么多失败码时不再重试。实测（post-patch 口径 1707 轮）第二次
# 恢复率随码数单调崩塌：1 码 32.9%、2 码 25.4%、3 码 20.5%、4 码 12.9%，>5 码只剩 6.7%
# （2/30），而一次重试要付 _revise_idea + _accept_revision 两次完整调用。码多意味着 prompt
# 要求同时清掉的约束多，模型只能每个字段都动一点，既救不回来还会制造跨字段语义不一致。
# 阈值取 5 而非更低：4 码仍有 12.9% 恢复率，砍掉它会牺牲 13 个本可落地的改写。
# 注意不能按"码的类型"跳出——纯表征码（target_unresolved / revision_package_mismatch /
# unsupported_evidence_dependency）的 attempt 实测 203 例仍恢复 42 例（20.7%），
# 可操作诊断由验收 reason 文本承载，不体现在码上。
MAX_RETRY_FAILURE_CODES = 5
# reason / rationale 字段的字符上限。实验数据显示360字符会系统性地在词中截断每条判断的
# 尾部，导致关键诊断信息丢失；提升到1200字符保留完整语义，不显著增加存储。
MATURATION_REASON_MAX_CHARS = 1200

# 苏格拉底树的 evidence_context 上限。树对每个节点发 3 次模型调用（提问/辩护/裁决），
# 每次都带全量 evidence_json，所以这里按"够判定 evidence_needed"裁剪，不搬全文。
EVIDENCE_PAPER_COUNT = 6
EVIDENCE_ABSTRACT_CHARS = 700
EVIDENCE_PAPER_FIELDS = (
    "source",
    "title",
    "year",
    "journal",
    "doi",
    "methods",
    "applications",
    "datasets_used",
    "retrieval_source",
    "retrieval_fallback",
    "cache_match_score",
)
EVIDENCE_CANDIDATE_FIELDS = (
    "doi",
    "parameters",
    "data_url",
    "metadata_url",
    "is_collection",
    "title",
    "id",
    "type",
    "provider",
    "docs",
    "source",
    "retrieval_channel",
    "dataset_source",
    "source_preference",
    "naming_note",
    "status",
    "coverage",
    "spatial_extent",
    "temporal_extent",
    "gsd",
    "resolution",
    "cadence",
)
EVIDENCE_CANDIDATE_DETAIL_CHARS = 900
# 未被 idea 引用的候选在树 prompt 里只留一句摘要。树的 evidence_json 实测由
# candidate_datasets 主导（Mining 26 条占 86%、8,641 token，literature 6 篇才 1,429），
# 而它在每次树调用里全量重发、单 task 上百次。同时实测每个 idea 只引用其中 1~2 条，
# 其余仅用于证明"这类数据存在"以免裁决误判 evidence_needed——那只需要身份与硬事实，
# 不需要 900 字符散文。
EVIDENCE_DEMOTED_DETAIL_CHARS = 150
# 降级视图保留的 detail 键：时空、分辨率、波段都是 combination_essentiality 判 alignment
# 的硬事实，合计只占体积的 ~41%，砍掉省不了多少却直接损伤判定。丢的只有长散文。
EVIDENCE_DEMOTED_DETAIL_KEYS = ("bands", "gsd", "cadence", "temporal_extent")
# title 分词匹配时忽略的通用词：数据集标题里几乎人人都有，命中它们不构成"被引用"。
# `land` / `cover` / `change` 是实测踩到的坑：它们在地理数据集标题里超高频，靠它们命中会把
# 三个不相关的 land-cover 产品误升成完整视图。
EVIDENCE_TITLE_STOPWORDS = frozenset({
    "data", "dataset", "datasets", "global", "map", "maps", "layer", "layers",
    "product", "products", "collection", "version", "annual", "monthly", "daily",
    "yearly", "resolution", "satellite", "image", "imagery", "raster", "vector",
    "grid", "gridded", "time", "series", "index", "level", "national", "world",
    "open", "public", "from", "with", "over", "based", "using", "high", "low",
    "land", "cover", "change", "area", "areas", "region", "regions",
})
# 单个显著词就足以判定被引用的长度门槛。取 7 而非 8：`aerosol` 这类 7 字母专业词特异性
# 已经足够，实测卡在 8 会把 PM2.5 研究的核心气溶胶数据集漏判成未引用。
EVIDENCE_TITLE_SPECIFIC_LEN = 7
EVIDENCE_QUERY_MAX_CHARS = 420
EVIDENCE_NEED_QUERY_TERMS = 12
EVIDENCE_MODIFICATION_QUERY_TERMS = 12
EVIDENCE_QUESTION_QUERY_TERMS = 5
EVIDENCE_ANCHOR_QUERY_TERMS = 10
EVIDENCE_RESULTS_PER_SOURCE = 3
EVIDENCE_DATASET_TOP_N = 3
# catalog 的 token-overlap 召回在 query 超过 4 个词后迅速退化（实测 12 词全部零命中，
# 4 词能召回 irrigation extent / evapotranspiration 等条目）。
DATASET_QUERY_MAX_TERMS = 4
# 通道判定只需读 evidence_need 本身，截断上限按实测最长需求（约 300 字符）留足余量。
EVIDENCE_CHANNEL_NEED_MAX_CHARS = 600
# near_miss 候选只用于人工诊断假阴性，描述截到够判断主题即可。
NEAR_MISS_DESCRIPTION_CHARS = 300
# 只用于清洗**模型给出的** catalog 检索词：prompt 已明确禁止这些限定语，但模型偶尔仍会
# 带上，而 4 词上限会因此把真正的产品名挤出去（实测 "high-resolution time-series
# vegetation index ndvi" 截断后丢掉了 ndvi）。
#
# 注意这张表不能并入 DATASET_QUERY_METHODOLOGY_TERMS：那张表作用于从 evidence_need
# 抽取的词袋，而 time-series / annual 在词袋路径上是**有效产品名**（实测出现在成功召回
# 的 query 里）。同一个词在两条路径上角色不同——模型输出里它是被禁的修饰语，词袋里它
# 可能就是数据集名的一部分。
DATASET_SEARCH_TERM_QUALIFIERS = frozenset({
    "annual", "coarse", "daily", "decadal", "external", "fine", "high-resolution",
    "highres", "hourly", "monthly", "multi-year", "regional", "sub-annual",
    "time-series", "timeseries", "weekly",
})
EVIDENCE_LITERATURE_SOURCES = ("openalex", "arxiv")
EVIDENCE_NEED_LITERATURE_MARKERS = (
    "literature",
    "empirical evidence",
    "empirical validation",
    "published evidence",
    "methodological evidence",
    "methodological framework",
    "empirical precedent",
    "methodological precedent",
)
EVIDENCE_NEED_DATASET_MARKERS = (
    "dataset",
    "data on",
    "data for",
    "data linking",
    "available dataset",
    "usable dataset",
    "dataset providing",
    "dataset with",
    "data layer",
    "event records",
    "historical records",
    "inventory",
    "inventories",
    "ground-truth labels",
    "ground truth labels",
    "proxy data",
    "observations of",
    "measurements of",
    "catalog entry",
)
# 只用于构造数据集检索 query：evidence_need 里描述"拿到这份数据要做什么"的方法学词汇
# （control for、in the regression model、to distinguish…）对 catalog 的 token-overlap
# 召回是纯噪声。实测保留它们时 39 次数据集检索全部零命中，剥离后同样的 evidence_need
# 能召回 irrigation extent / evapotranspiration / precipitation climatology 等条目。
# 不并入 DATASET_CAPABILITY_GENERIC_TERMS：那个集合服务于候选校验的 gap-term 命中计数，
# 在那里多剥词会让 minimum_hits 失去判别力。
DATASET_QUERY_METHODOLOGY_TERMS = frozenset({
    "account", "actual", "analysis", "assess", "assumed", "calculate", "compare",
    "comparative", "compute", "confirm", "confounding", "constrain", "control",
    "controlling", "decouple", "derive", "determine", "differentiate", "disentangle",
    "distinguish", "effect", "effects", "estimate", "evaluate", "explain", "filter",
    "identify", "impact", "individual", "isolate", "level", "levels", "magnitude",
    "measure", "mediates", "model", "quantify", "quantifying", "rate", "ratio",
    "regression", "relationship", "separate", "significantly", "stratify", "test",
    "testing", "validate", "validating", "variance", "verify", "whether",
    # 以下来自五个任务 172 条真实 dataset query 的实测：每个词都只出现在零召回的
    # query 里、成功召回的 query 中一次都没有出现过。它们描述的是"要一份说明/方法/
    # 定义"，对应的 evidence_need 实际索要文档或查找表（如 "GMW v3 documentation
    # specifying the semantic meaning of change class codes"、"A mapping table or
    # study linking LANDFIRE EVT classes to root depth"、"Metadata specifying the
    # unit of the AREA attribute"），而 catalog 按 token overlap 打分，这些词在数据集
    # 标题里不出现，只会稀释真正的产品名词。
    #
    # 注意不要顺手加入 time-series / biomass / irrigation / landfire / dem 这类词：
    # 它们在失败 query 里也高频，但同时出现在成功召回的 query 中，是有效的产品名。
    "accepted", "capable", "clarification", "definition", "defining", "documentation",
    "distinguishing", "feature", "handling", "method", "methodological", "methodology",
    "resolving", "specification", "specifying", "standard", "study", "table", "unit",
})
DATASET_CAPABILITY_GENERIC_TERMS = frozenset({
    "available", "availability", "coverage", "data", "dataset", "datasets", "empirical",
    "event", "events", "global", "ground", "historical", "inventory", "inventories",
    "labels", "layer", "layers", "linking", "local", "measurement", "measurements",
    "named", "needed", "observations", "observed", "proxies", "proxy", "provides",
    "providing", "records", "required", "source", "spatial", "specific", "temporal",
    "truth", "usable", "validation", "variable", "variables",
})
EVIDENCE_QUERY_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "because", "before", "between",
    "by", "can", "could", "current", "defender", "does", "evidence", "fails", "for",
    "from", "given", "how", "idea", "if", "in", "into", "is", "it", "its", "must",
    "need", "of", "on", "or", "our", "provide", "question", "rather", "resolve",
    "resolved", "should", "specific", "than", "that", "the", "their", "this", "to",
    "using", "verify", "what", "when", "where", "which", "while", "why", "with",
    "without", "would",
})
# 批评性元话语：这些词描述"质询双方在争什么"，不描述研究对象本身。finding 文本
# （question / modification_direction）里大量出现，如果进入检索 query 或相关性
# focus terms，会把检索拉向错误学科（实测 "convergence ... contradiction" 召回了
# 数论论文《Conjectures on Convergence and Scalar Curvature》），也会让词面相关性
# 判定靠 "study/claims/states" 这类词凑满命中数。与 EVIDENCE_QUERY_STOPWORDS 并用，
# 同时净化 query 构造、focus terms 与相关性匹配三条路径。
CRITIQUE_META_TERMS = frozenset({
    "answer", "answers", "argument", "asserts", "assertion", "category", "cannot",
    "claim", "claimed", "claims", "concedes", "contains", "contradiction",
    "contradictions", "contradicts", "correct", "critique", "defenders", "defends",
    "error", "finding", "findings", "fundamental", "hypotheses", "hypothesis",
    "incoherent", "incorrect", "invalid", "justify", "layer", "parent", "rebuttal",
    "revision", "semantic", "states", "study", "valid", "verdict",
})
REVISION_ROUTES = frozenset({"idea_field_update", "step_plan_open_point", "unresolved"})
# revision package item 的证据要求。`none_required` 表示通道判定**肯定地**得出这条
# finding 不需要任何外部证据（定义性/内部一致性修正），它引用的检索记录必然不是
# grounded；验收侧必须据此豁免 grounded 要求，否则同一条记录会被路由和验收反着判两次。
EVIDENCE_REQUIREMENT_GROUNDED = "grounded_records"
EVIDENCE_REQUIREMENT_NONE = "none_required"
STEP_PLAN_OPEN_POINT_MARKERS = (
    "exact threshold",
    "numeric threshold",
    "parameter value",
    "preprocessing step",
    "resampling method",
    "reprojection method",
    "train/test split",
    "validation protocol",
    "download procedure",
    "tool pipeline",
)

FV_FEASIBILITY_AXES = (
    "idea_operational_coherence",
    "mechanism_soundness",
    "claim_defensibility",
)
FV_VALUE_AXES = (
    "novelty_frontier",
    "discovery_leverage",
    "combination_essentiality",
)
FV_AXES = (*FV_FEASIBILITY_AXES, *FV_VALUE_AXES)
FV_RUBRIC_VERSION = "idea_fv_v2_basis_codes"
# 权重在各组内归一化为 1，保留 feasibility/value 两个可解释的 0~1 聚合量，
# 再由二者相乘得到 fv_product。第一版权重依据 ablation_full 的结果：data-fit/
# 可执行性是主要短板，因此提高 operational coherence；value 组保留 novelty 的
# 基础权重，同时让 discovery leverage 与 combination essentiality 各占更大权重。
FV_DEFAULT_AXIS_WEIGHTS = {
    "idea_operational_coherence": 0.40,
    "mechanism_soundness": 0.30,
    "claim_defensibility": 0.30,
    "novelty_frontier": 0.30,
    "discovery_leverage": 0.35,
    "combination_essentiality": 0.35,
}
# 轴锚点表移到 socratic_idea_dual_tree：树的裁决改用 basis code 后与 selection FV 用同一套
# 六轴刻度，两份逐字节相同的副本会在任一侧单独修改时让同一个 code 拿到不同分数。旧名保留
# 为别名，FV 侧的 rubric 版本与判据文本不受影响。
FV_BASIS_SCORE_MAP = AXIS_BASIS_SCORE_MAP


class IdeaMaturationAgent(BaseAgent):
    """Improve selected ideas before detailed execution planning."""

    def __init__(self, model: Any, config: Dict[str, Any]):
        super().__init__(model, config)
        global_config = config.get("_global_config") or {}
        planning = global_config.get("planning") or {}
        maturation = planning.get("idea_maturation") or {}
        literature_config = (global_config.get("tools") or {}).get("literature_search") or {}
        self.max_publication_year = validate_max_publication_year(
            literature_config.get("max_publication_year")
        )
        self.max_rounds = max(1, int(maturation.get("max_iteration_rounds", 2)))
        self.max_tree_depth = max(2, int(maturation.get("max_tree_depth", self.max_rounds * 2)))
        self.max_concurrent_ideas = max(1, int(maturation.get("max_concurrent_ideas", 2)))
        self.fv_target_threshold = float(maturation.get("fv_target_threshold", 0.625))
        self.fv_min_feasibility = float(maturation.get("fv_min_feasibility", 0.75))
        self.fv_min_value = float(maturation.get("fv_min_value", 0.75))
        configured_weights = maturation.get("fv_axis_weights")
        if configured_weights is None:
            configured_weights = FV_DEFAULT_AXIS_WEIGHTS
        self.fv_axis_weights = self._validate_fv_axis_weights(configured_weights)
        # 方向性主张微型实证探针：懒加载（首次需要时按 base_dataset.data_path 建表缓存）。
        # 结果只挂在当轮 finding 的 micro_empirical_check 上，不进 evidence_context。
        self._stat_probe: Optional[DirectionalStatProbe] = None
        self._stat_probe_df_cache: Optional[Tuple[str, Any]] = None
        self.fv_plateau_eps = float(maturation.get("fv_plateau_eps", 0.01))
        self.fv_plateau_rounds = max(1, int(maturation.get("fv_plateau_rounds", 2)))
        self.fv_plateau_min_product = float(
            maturation.get("fv_plateau_min_product", self.fv_target_threshold)
        )
        self.fv_tie_eps = float(maturation.get("fv_tie_eps", self.fv_plateau_eps))
        self.max_accepted_versions = max(1, int(maturation.get("max_accepted_versions", 5)))
        self.questioning_max_tokens = int(maturation.get("questioning_max_tokens", 700))
        # 一次修订最多同时改整条主张链（research_question / core_hypothesis /
        # theoretical_mechanism / computability_and_falsifiability）加最多
        # MAX_SUPPORTING_REVISION_FIELDS 个支撑字段，按 _idea_patch_schema 的 maxLength
        # 合计 4000 字符。调低会让 patch 在字段中途截断，产出空改写。
        self.revision_max_tokens = int(maturation.get("revision_max_tokens", 2600))
        self.acceptance_max_tokens = int(maturation.get("acceptance_max_tokens", 700))
        # 通道判定输出两个 bool 加两条检索式（dataset + literature），220 给足余量。
        self.evidence_channel_max_tokens = int(
            maturation.get("evidence_channel_max_tokens", 220)
        )
        # evidence_need 非空但字面标记识别不出通道时，用一次短调用判定它要的是数据产品
        # 还是已发表结论。关闭时退回保守判定（按需要数据集处理）——语义仍然安全，只是会
        # 把"其实只需文献"的需求也拦成未满足，因此实验配置里应保持开启。
        self.enable_evidence_channel_classifier = bool(
            maturation.get("enable_evidence_channel_classifier", True)
        )
        # 词面预筛之后由一次短调用对候选证据做终裁：词面重合判不出"泛词凑数"的
        # 假阳性（数学论文靠 slope/convergence 混入）也判不出措辞不同的假阴性。
        # 关闭时显式回退词面结果，回退状态写入 relevance_filter.judge 供审计。
        self.enable_evidence_relevance_judge = bool(
            maturation.get("enable_evidence_relevance_judge", True)
        )
        self.evidence_relevance_max_tokens = int(
            maturation.get("evidence_relevance_max_tokens", 300)
        )
        self.evidence_results_per_source = max(
            1,
            int(maturation.get("evidence_results_per_source", EVIDENCE_RESULTS_PER_SOURCE)),
        )
        self.evidence_dataset_top_n = max(
            1,
            int(maturation.get("evidence_dataset_top_n", EVIDENCE_DATASET_TOP_N)),
        )
        self.enable_literature_cache_fallback = (
            maturation.get("enable_literature_cache_fallback") is True
        )
        literature_cache_path = maturation.get("literature_cache_path")
        if self.enable_literature_cache_fallback and not (
            isinstance(literature_cache_path, str) and literature_cache_path.strip()
        ):
            raise ValueError(
                "planning.idea_maturation.literature_cache_path is required when "
                "literature cache fallback is enabled"
            )
        self._filtered_literature_cache = (
            FilteredLiteratureCache(literature_cache_path)
            if self.enable_literature_cache_fallback
            else None
        )
        # 语义验收驳回时在同一深度对内带着 failure_codes 重写。重试不占深度预算：
        # 否则一次被驳回的修订就消耗掉一次树生长机会，同一批 finding 再无落地机会。
        self.revision_max_attempts = max(1, int(maturation.get("revision_max_attempts", 2)))
        # 连续这么多轮"有可改写 finding 但一条都没落地"就停掉该 idea 的成熟化：
        # 再往下钻只是把树问深，idea 不会变，剩余深度预算全部浪费。
        # 取 1 而非 2 的依据：570 个 idea / 2185 轮的实测里，revision_stalled 是第一大
        # 终止原因（37.7%），且末尾连续 rejected 轮数高度集中在 2（200/215 例），
        # 合计 446 轮 = 全部轮次的 20.4%。tail 恰好为 2 意味着这些 idea 的第二轮同样
        # 没能落地——真翻盘的轮次会 reset 计数并且不会计入 stalled，所以那一轮纯属
        # 空烧约 24 次树调用。降到 1 回收约 9% 的总调用量，语义损失接近零。
        self.revision_stall_rounds = max(1, int(maturation.get("revision_stall_rounds", 1)))
        # 一轮内所有 finding 都被路由走（无 idea_field_update）不再立即终止：树下一轮
        # 会带着累积的 evidence_context 长出不同的 finding，检索也可能换一批需求命中。
        # 只有连续 routed_stall_rounds 轮都无可行动 finding 才判定"确实没有可修的点"。
        self.routed_stall_rounds = max(1, int(maturation.get("routed_stall_rounds", 2)))
        # 分支休眠兜底：连续这么多轮 axis_score 完全不动的分支停止生长。axis_score 现在是
        # {0,.25,.5,.75,1} 的离散档，任何真实档位变化都 >= 0.25，所以默认 eps 只会在同档
        # 重复时触发。与 revision_stall_rounds 是两件事：那个停整个 idea，这个只停单个分支。
        self.branch_stall_rounds = max(1, int(maturation.get("branch_stall_rounds", 2)))
        self.branch_stall_eps = float(maturation.get("branch_stall_eps", 0.05))
        # 第 ≥2 轮每轮放行的分支名额。休眠判据只能停掉"判断不再变化"的分支，停不掉
        # "判断在同一档反复横跳"的分支——实测三个分支的休眠率只有 7.3%，第 ≥2 轮平均
        # 仍生长 4.64 个分支，而那一轮的 FV 环比增量均值只有 +0.0108（43.6% 为 0、
        # 24.2% 为负）。名额与 BALANCER_MAX_FINDINGS 同值：单轮最多用得上三条 finding。
        self.branch_budget_per_round = max(
            1,
            int(maturation.get("branch_budget_per_round", BRANCH_BUDGET_PER_ROUND)),
        )
        self.dual_tree = SocraticIdeaDualTree(
            self._call_dual_tree_model,
            self.questioning_max_tokens,
            branch_stall_rounds=self.branch_stall_rounds,
            branch_stall_eps=self.branch_stall_eps,
            branch_budget_per_round=self.branch_budget_per_round,
        )
        # 由 execute 从 context 注入；随 dataset_context 一起进入 revision/acceptance prompt
        self._candidate_datasets: List[Dict[str, Any]] = []
        # 由 execute 从 context 注入；进入固定 evidence_context，供树与 selection FV 共用。
        self._literature_review: List[Dict[str, Any]] = []
        self._literature_search: Optional[LiteratureSearch] = None
        self._candidate_scout: Optional[CandidateDatasetScout] = None
        self._candidate_enricher: Optional[DatasetDetailEnricher] = None

    def _dataset_context_payload(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """dataset_context 基础上并入候选数据集（含 detail），供 prompt 使用。"""
        payload = self._dataset_context(dataset_info)
        component_roles = self._composite_component_roles(dataset_info)
        if component_roles:
            payload["required_composite_components"] = component_roles
        if self._candidate_datasets:
            payload["candidate_datasets"] = self._candidate_datasets
        return payload

    def _stat_probe_dataframe(self, dataset_info: Dict[str, Any]) -> Optional[Any]:
        """按 base_dataset.data_path 懒加载探针用表（每路径缓存一份）。"""
        info = dataset_info or {}
        base = info.get("base_dataset") or {}
        if not isinstance(base, dict):
            base = {}
        # 任务描述符把 data_path 放顶层、base_dataset 内各有一份；两处都试。
        path = str(base.get("data_path") or info.get("data_path") or "").strip()
        if not path:
            return None
        if self._stat_probe is None:
            self._stat_probe = DirectionalStatProbe(model_call=self._call_model)
        cached = self._stat_probe_df_cache
        if cached is not None and cached[0] == path:
            return cached[1]
        df = self._stat_probe.load_dataframe(path)
        self._stat_probe_df_cache = (path, df)
        return df

    async def _attach_micro_empirical_checks(
        self,
        findings: List[Dict[str, Any]],
        idea: Dict[str, Any],
        dataset_info: Dict[str, Any],
    ) -> None:
        """给本轮被裁决为可改写的方向性 finding 挂一行实测统计。

        作用域：结果只写进 finding 的 `micro_empirical_check`，随 findings_json 进入
        本轮修订 prompt 与轮次审计；findings 每轮由树重建，因此不跨轮、不进
        evidence_context、不进 selection FV。探针失败静默跳过（不得中断成熟化）。
        """
        if not findings:
            return
        df = self._stat_probe_dataframe(dataset_info)
        if df is None or self._stat_probe is None:
            return
        hypothesis = idea.get("core_hypothesis")
        if isinstance(hypothesis, dict):
            directional = str(hypothesis.get("directional_statement") or "")
        else:
            directional = str(hypothesis or "")
        if not directional:
            return
        columns = [str(column) for column in df.columns]
        for finding in findings:
            # 不按 verdict 过滤：prepared findings 不携带 verdict 字段（裁决语义在
            # source_verdicts/issue_type 里，形态不稳定）；是否可算由 spec 抽取自判
            # （extractable=false 即无指令），代价是每 finding 一次短调用。
            try:
                probe = await self._stat_probe.probe_finding(
                    df=df,
                    question=str(finding.get("question") or ""),
                    failure_hypothesis=str(finding.get("failure_hypothesis") or ""),
                    directional_statement=directional,
                    columns=columns,
                )
            except Exception as exc:
                logger.warning(
                    "micro empirical probe failed for %s: %s",
                    finding.get("finding_id"), exc,
                )
                continue
            contrast = (probe or {}).get("contrast")
            if isinstance(contrast, dict) and contrast.get("status") == "ok":
                finding["micro_empirical_check"] = probe

    @staticmethod
    def _composite_component_roles(dataset_info: Dict[str, Any]) -> List[Dict[str, str]]:
        base_dataset = dataset_info.get("base_dataset") if isinstance(dataset_info, dict) else {}
        if not isinstance(base_dataset, dict) or base_dataset.get("type") != "composite_dataset":
            return []
        contract = base_dataset.get("combination_contract") or {}
        required_roles = {
            str(role).strip()
            for role in (contract.get("required_roles") or [])
            if str(role).strip()
        } if isinstance(contract, dict) else set()
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

    def _evidence_context(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """构造苏格拉底树与 selection FV 使用的当前证据集合。

        树在裁决时用 `evidence_needed` 声明"缺一份具名证据"。若只给 base_dataset，
        文献里已有的算法验证结论、以及已侦察到的候选数据集都会被判成缺证据 —— 历史
        数据里约 18% 的 evidence_need 属于这一类。这些证据在 planning context 里
        已经存在，此处只做裁剪后透传，不发起请求。质询后的强制检索由
        `_run_idea_evidence_grounding` 执行，再由 `_merge_grounded_evidence_context` 合并。
        """
        context_manager = getattr(self, "research_context", None)
        if context_manager is not None and context_manager.memory.initialized:
            return context_manager.memory.evidence_context()
        context: Dict[str, Any] = {
            "base_dataset": self._dataset_context(dataset_info).get("base_dataset", {}),
        }
        if self._literature_review:
            context["literature"] = [
                self._evidence_paper(paper)
                for paper in self._literature_review[:EVIDENCE_PAPER_COUNT]
                if isinstance(paper, dict)
            ]
        if self._candidate_datasets:
            context["candidate_datasets"] = [
                self._evidence_candidate(candidate)
                for candidate in self._candidate_datasets
                if isinstance(candidate, dict)
            ]
        return context

    @staticmethod
    def _evidence_paper(paper: Dict[str, Any]) -> Dict[str, Any]:
        """保留标题/出处与结构化抽取，摘要截断到判定所需长度。"""
        trimmed = {
            field: paper[field]
            for field in EVIDENCE_PAPER_FIELDS
            if paper.get(field) not in (None, "", [], {})
        }
        abstract = paper.get("abstract")
        if isinstance(abstract, str) and abstract.strip():
            trimmed["abstract"] = abstract[:EVIDENCE_ABSTRACT_CHARS]
        return trimmed

    @classmethod
    def _near_miss_candidate(cls, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """被 status 门槛丢弃的候选：只留可诊断假阴性所需的最小身份。

        纯审计用途，不进模型输入（`_model_evidence_records` 不含 `near_miss_candidates`），
        因此不能影响任何判定。留 gap-term 命中情况是为了回答关键问题：这一档里有多少是
        真该采纳的（命中定性词，如 population/density 对应人口密度需求），多少是词面
        碰巧凑够两个通用词的假阳性（如 heat/flux 对应城市人为热通量需求）。
        """
        trimmed = {
            field: candidate[field]
            for field in ("id", "title", "status", "source")
            if candidate.get(field) not in (None, "", [], {})
        }
        detail = candidate.get("detail")
        if isinstance(detail, dict):
            description = detail.get("description")
            if isinstance(description, str) and description.strip():
                trimmed["description"] = description[:NEAR_MISS_DESCRIPTION_CHARS]
        return trimmed

    @staticmethod
    def _evidence_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
        """保留数据集身份与波段/覆盖信息；丢掉 query 与 ee_snippets 等实现细节。"""
        trimmed = {
            field: candidate[field]
            for field in EVIDENCE_CANDIDATE_FIELDS
            if candidate.get(field) not in (None, "", [], {})
        }
        if candidate.get("dataset_source") == "PANGAEA":
            trimmed["relations"] = candidate.get("relations", [])[:12]
            trimmed["relation_count"] = len(candidate.get("relations", []))
        detail = candidate.get("detail")
        if isinstance(detail, dict):
            description = detail.get("description")
            trimmed["detail"] = {
                key: detail[key]
                for key in ("detail_source", "bands", "gsd", "cadence", "temporal_extent")
                if detail.get(key) not in (None, "", [], {})
            }
            if isinstance(description, str) and description.strip():
                trimmed["detail"]["description"] = description[:EVIDENCE_CANDIDATE_DETAIL_CHARS]
        return trimmed

    @staticmethod
    def _demoted_evidence_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
        """未被 idea 引用的候选：留身份与硬事实，长描述压成一句摘要。

        这类候选在树里的唯一作用是证明"这类数据存在"，从而不让裁决把已有数据判成
        `evidence_needed`。列表里一条不少，只是详略分层——所以存在性零损失。
        """
        trimmed = {
            field: candidate[field]
            for field in EVIDENCE_CANDIDATE_FIELDS
            if candidate.get(field) not in (None, "", [], {})
        }
        detail = candidate.get("detail")
        if isinstance(detail, dict):
            kept = {
                key: detail[key]
                for key in EVIDENCE_DEMOTED_DETAIL_KEYS
                if detail.get(key) not in (None, "", [], {})
            }
            description = detail.get("description")
            if isinstance(description, str) and description.strip():
                kept["description"] = description[:EVIDENCE_DEMOTED_DETAIL_CHARS]
            if kept:
                trimmed["detail"] = kept
        return trimmed

    @staticmethod
    def _candidate_title_tokens(title: str) -> set:
        """标题里可用于匹配的显著词：去通用词、长度 >= 4。"""
        return {
            token
            for token in re.findall(r"[a-z0-9]+", str(title).lower())
            if len(token) >= 4 and token not in EVIDENCE_TITLE_STOPWORDS
        }

    @classmethod
    def _is_referenced_candidate(
        cls,
        candidate: Dict[str, Any],
        idea_blob: str,
        base_dataset_ids: set,
    ) -> bool:
        """判断候选是否被当前 idea 引用。判据宁滥勿缺。

        漏判会让 idea 正在使用的数据集被降级，`combination_essentiality` 可能据此判成
        `undeclared_or_unavailable_dependency`，等于惩罚正确的修订；多判几条只是少省一点
        体积（实测 referenced 从 2 涨到 8 也只多花 13 个百分点）。所以这里刻意放宽。
        """
        candidate_id = str(candidate.get("id") or "").strip().lower()
        title = str(candidate.get("title") or "").strip()
        if candidate_id and candidate_id in base_dataset_ids:
            return True
        if candidate_id and candidate_id in idea_blob:
            return True
        if title and title.lower() in idea_blob:
            return True
        hits = {token for token in cls._candidate_title_tokens(title) if token in idea_blob}
        # 两个显著词命中，或单个足够特异的长词命中（如 `aerosol` / `geomorpho90m`）。
        return len(hits) >= 2 or any(len(token) >= EVIDENCE_TITLE_SPECIFIC_LEN for token in hits)

    def _tiered_evidence_context(
        self,
        evidence_context: Dict[str, Any],
        idea: Dict[str, Any],
    ) -> Dict[str, Any]:
        """给树用的 evidence_context：候选按"是否被 idea 引用"分两档。

        只作用于树。revision prompt 与 selection FV 继续拿完整视图——它们要写具体引用，
        且调用量比树低两个数量级（树单 task 实测 435~520 次）。

        每轮重算：idea 被修订后新引用的数据集，下一轮自动升回完整视图。
        """
        context_manager = getattr(self, "research_context", None)
        if context_manager is not None:
            return context_manager.memory.evidence_context()
        candidates = evidence_context.get("candidate_datasets")
        if not isinstance(candidates, list) or not candidates:
            return evidence_context

        idea_blob = json.dumps(idea, ensure_ascii=False).lower()
        base_dataset = evidence_context.get("base_dataset")
        base_ids = set()
        if isinstance(base_dataset, dict):
            for key in ("id", "asset_id", "dataset_id"):
                value = str(base_dataset.get(key) or "").strip().lower()
                if value:
                    base_ids.add(value)

        tiered = dict(evidence_context)
        tiered["candidate_datasets"] = [
            candidate
            if self._is_referenced_candidate(candidate, idea_blob, base_ids)
            else self._demoted_evidence_candidate(candidate)
            for candidate in candidates
            if isinstance(candidate, dict)
        ]
        return tiered

    async def _call_model(self, prompt, system_prompt=None, schema=None, temperature=None, **kwargs):
        from datahypgen.mas.tools.context_manager import preserve_content_schema, model_stage
        started = time.monotonic()
        try:
            return await super()._call_model(prompt=prompt, system_prompt=system_prompt,
                schema=preserve_content_schema(schema), temperature=temperature, **kwargs)
        finally:
            memory = getattr(self, "research_context", None)
            if memory is not None:
                memory.memory.append_event({"kind": "model_stage", "stage": model_stage(schema),
                    "elapsed_seconds": time.monotonic() - started,
                    "prompt_characters": len(prompt), "timing_scope": "inclusive"})

    async def _call_dual_tree_model(self, **kwargs: Any) -> Dict[str, Any]:
        return await self._call_model(**kwargs)

    @classmethod
    def _prepare_questioning_findings(
        cls,
        findings: List[Dict[str, Any]],
        active_leaves: List[Dict[str, Any]],
        round_number: int,
    ) -> List[Dict[str, Any]]:
        """Attach traceable IDs and include leaves explicitly blocked on evidence.

        This is post-processing only. The dual-tree Balancer and its growth mechanics remain
        unchanged; evidence-blocked leaves are surfaced here so the mandatory retrieval stage can
        act on them instead of stopping the idea with no search attempt.
        """
        leaves_by_field: Dict[str, List[Dict[str, Any]]] = {}
        for leaf in active_leaves:
            if not isinstance(leaf, dict):
                continue
            field = str(leaf.get("target_field") or "").strip()
            if field in IDEA_FIELDS:
                leaves_by_field.setdefault(field, []).append(leaf)

        ordered: List[Dict[str, Any]] = []
        seen_fields: set[str] = set()
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            field = str(finding.get("target_field") or "").strip()
            if field not in IDEA_FIELDS or field in seen_fields:
                continue
            ordered.append(copy.deepcopy(finding))
            seen_fields.add(field)

        for leaf in active_leaves:
            if not isinstance(leaf, dict) or leaf.get("verdict") != "evidence_needed":
                continue
            field = str(leaf.get("target_field") or "").strip()
            if field not in IDEA_FIELDS or field in seen_fields:
                continue
            ordered.append({
                "target_field": field,
                "question": str(leaf.get("question") or ""),
                "resolution_hint": str(leaf.get("adjudication") or ""),
            })
            seen_fields.add(field)

        prepared: List[Dict[str, Any]] = []
        for index, finding in enumerate(ordered, start=1):
            field = str(finding.get("target_field") or "")
            source_leaves = leaves_by_field.get(field, [])
            evidence_needs = list(dict.fromkeys(
                str(leaf.get("evidence_need") or "").strip()
                for leaf in source_leaves
                if str(leaf.get("evidence_need") or "").strip()
            ))
            verdicts = list(dict.fromkeys(
                str(leaf.get("verdict") or "").strip()
                for leaf in source_leaves
                if str(leaf.get("verdict") or "").strip()
            ))
            prepared.append({
                **finding,
                "finding_id": f"r{round_number:02d}_f{index:02d}_{field}",
                "issue_type": (
                    "evidence_gap"
                    if verdicts and set(verdicts) == {"evidence_needed"}
                    else "idea_gap"
                ),
                "is_core_contradiction": bool(finding.get("is_core_contradiction", False)),
                # 显式带下来：`**finding` 只覆盖 Balancer 产出的 finding，而下面由
                # evidence_needed 叶子新造的 finding dict 不含这两个键。
                "requires_discriminating_contrast": bool(
                    finding.get("requires_discriminating_contrast", False)
                ),
                "required_contrast": str(finding.get("required_contrast") or ""),
                "modification_direction": str(
                    finding.get("modification_direction")
                    or finding.get("resolution_hint")
                    or ""
                ),
                "evidence_need": "; ".join(evidence_needs),
                "source_verdicts": verdicts,
                "source_node_ids": [
                    str(leaf.get("node_id") or "")
                    for leaf in source_leaves
                    if str(leaf.get("node_id") or "")
                ],
            })
        return prepared

    @staticmethod
    def _evidence_query_text(value: Any) -> str:
        fragments: List[str] = []

        def collect(item: Any) -> None:
            if isinstance(item, str) and item.strip():
                fragments.append(item.strip())
            elif isinstance(item, dict):
                for nested in item.values():
                    collect(nested)
            elif isinstance(item, list):
                for nested in item:
                    collect(nested)

        collect(value)
        return re.sub(r"\s+", " ", " ".join(fragments)).strip()

    @classmethod
    def _required_evidence_channels(cls, finding: Dict[str, Any]) -> List[str]:
        """按字面标记识别所需证据通道；识别不出时返回空表。

        这是快速路径，只在措辞恰好命中标记表时给出答案。标记表是白名单式字面匹配，
        换一批措辞就漏（实测 "External time-series of regional climatic variables"、
        "land-cover masks"、"policy implementation dates" 全部漏判），因此**不能**把
        返回空表当作"这条需求对通道无要求"——那会让文献命中冒充数据集证据。空表的
        语义是"未能识别"，由 `_resolve_required_evidence_channels` 交给模型判定。
        """
        text = cls._evidence_query_text(finding.get("evidence_need")).lower()
        channels: List[str] = []
        if any(marker in text for marker in EVIDENCE_NEED_LITERATURE_MARKERS):
            channels.append("literature")
        if any(marker in text for marker in EVIDENCE_NEED_DATASET_MARKERS):
            channels.append("dataset")
        return channels

    async def _resolve_required_evidence_channels(
        self,
        finding: Dict[str, Any],
    ) -> Tuple[List[str], Dict[str, Any]]:
        """确定这条 finding 需要哪些证据通道，并返回判定来源审计。

        字面标记命中时通道以标记为准，但**仍然调用模型**去拿 catalog 检索词：标记只能
        判断"要不要数据集"，翻译不出"数据集会怎么命名自己"。此前 marker 命中就直接返回、
        省掉这次调用，代价是这批 finding 只能用词袋检索——实测 `Global_Mining_Polygons`
        一次运行里 19 条走 marker 快速路径的记录全部回退到 `gap_terms`，召回率 5%，而走
        模型检索词的 16 条召回率 37%。省下一次 120 token 的短调用，损失约 6 条证据，
        这个取舍不成立。因此现在只在明确不需要数据集时才跳过调用。

        标记完全命中不到且 `evidence_need` 非空时，由模型判断该需求要的是数据产品还是
        已发表结论。这一步不能省：`required_channels` 为空会让
        `_run_idea_evidence_grounding` 走"任一通道有结果即 grounded"的宽松分支，于是
        "需要 VPD 时间序列"这类需求被三篇相关论文判成已满足，路由成 `idea_field_update`，
        修订被迫把数据集没有的变量写进主张，再被验收的可得性边界拦下——形成重试无法
        逃脱的死锁（实测 7 个失败轮次中 5 个是这个模式）。
        """
        marker_channels = self._required_evidence_channels(finding)
        evidence_need = self._evidence_query_text(finding.get("evidence_need")).strip()

        if not evidence_need:
            # 没有声明证据需求，本就无通道要求。
            return [], {"source": "no_evidence_need", "channels": []}

        if not self.enable_evidence_channel_classifier:
            # 分类器关闭时：标记结论优先，标记也没有则保守按需要数据集处理。
            channels = list(marker_channels) or ["dataset"]
            return channels, {
                "source": "marker_classifier_disabled" if marker_channels else "classifier_disabled",
                "channels": channels,
            }

        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "needs_dataset": {"type": "boolean"},
                "needs_literature": {"type": "boolean"},
                # catalog 检索词由模型给出，因为从 evidence_need 抽词袋这个方法本身
                # 产不出 catalog 能匹配的形式：实测 6 组配对里词袋只 1 组命中，而模型
                # 式的产品名 6 组全中（"time-series vegetation indices ndvi" 零命中，
                # "vegetation index ndvi" 命中 AVHRR NDVI）。扩停用词表已到极限，
                # IDF 加权也验证过方向相反。
                "dataset_search_terms": {"type": "string", "maxLength": 120},
                # 文献检索式同样由模型翻译：词袋抽取会把批评元话语（contradiction/
                # defender/claim）和截断残词带进 query，实测把检索拉向数论、医学等
                # 错误学科。领域锚点 + 名词短语的翻译式 query 是 6.1 节 dataset 侧
                # 已验证有效的同一做法。
                "literature_search_query": {"type": "string", "maxLength": 160},
            },
            "required": [
                "needs_dataset",
                "needs_literature",
                "dataset_search_terms",
                "literature_search_query",
            ],
        }
        try:
            result = await self._call_model(
                prompt=build_idea_evidence_channel_prompt(
                    evidence_need=evidence_need[:EVIDENCE_CHANNEL_NEED_MAX_CHARS],
                ),
                system_prompt=IDEA_EVIDENCE_CHANNEL_SYSTEM_PROMPT,
                schema=schema,
                temperature=0.0,
                max_tokens=self.evidence_channel_max_tokens,
            )
        except Exception as exc:
            logger.warning("Evidence channel classification failed: %s", exc)
            # 判定不可用时保守要求数据集通道：宁可多拦一次，也不让未验证的数据需求
            # 冒充已满足。marker 已给出的通道结论仍然优先，别因为翻译失败就丢掉它。
            channels = list(marker_channels) or ["dataset"]
            return channels, {
                "source": "marker_translation_unavailable" if marker_channels else "classifier_unavailable",
                "channels": channels,
            }

        if not isinstance(result, dict):
            channels = list(marker_channels) or ["dataset"]
            return channels, {
                "source": "marker_translation_invalid" if marker_channels else "classifier_invalid",
                "channels": channels,
            }
        # 通道以 marker 为准（它是确定性的），模型只在 marker 未命中时决定通道；
        # 无论哪种情况，检索词都取模型的翻译结果。
        if marker_channels:
            channels = list(marker_channels)
            source = "marker"
        else:
            channels = []
            if result.get("needs_literature"):
                channels.append("literature")
            if result.get("needs_dataset"):
                channels.append("dataset")
            source = "classifier"
        decision = {
            "source": source,
            "channels": list(channels),
            "needs_dataset": bool(result.get("needs_dataset")),
            "needs_literature": bool(result.get("needs_literature")),
        }
        # 检索词随判定一起带出，供 _retrieve_datasets_for_finding 优先使用。放进
        # decision 而不是改返回签名，是为了不波及其他调用点。
        search_terms = self._bounded_query(
            self._dataset_search_term_words(result.get("dataset_search_terms"))
        )
        if search_terms:
            decision["dataset_search_terms"] = search_terms
        literature_query = self._literature_search_query_words(
            result.get("literature_search_query")
        )
        if literature_query:
            decision["literature_search_query"] = self._bounded_query(literature_query)
        return channels, decision

    @classmethod
    def _literature_search_query_words(cls, value: Any) -> List[str]:
        """清洗模型给出的文献检索式：保序、去元话语，不做词袋去重截断。

        prompt 已禁止元话语，这里再过一遍 CRITIQUE_META_TERMS 兜底，防止
        "hypothesis"/"contradiction" 类词把检索拉出学科。"""
        if not isinstance(value, str) or not value.strip():
            return []
        tokens = re.findall(r"[a-z0-9][a-z0-9_-]*", value.lower())
        return [
            token
            for token in tokens
            if token.strip("_-")
            and token.strip("_-") not in CRITIQUE_META_TERMS
        ]

    @classmethod
    def _dataset_search_term_words(cls, value: Any) -> List[str]:
        """清洗模型给出的 catalog 检索词。

        模型偶尔会带上被明确禁止的修饰语，这里再过一遍同一张方法学词表兜底，并按
        DATASET_QUERY_MAX_TERMS 截断——实测 catalog 召回在 4 词以上迅速退化。
        """
        if not isinstance(value, str) or not value.strip():
            return []
        return [
            term
            for term in cls._evidence_query_terms(value, DATASET_QUERY_MAX_TERMS * 3)
            if term not in DATASET_QUERY_METHODOLOGY_TERMS
            and term not in DATASET_CAPABILITY_GENERIC_TERMS
            and term not in DATASET_SEARCH_TERM_QUALIFIERS
        ][:DATASET_QUERY_MAX_TERMS]

    @classmethod
    def _evidence_query_terms(cls, value: Any, max_terms: int) -> List[str]:
        """Extract compact ordered terms while retaining both ends of long critiques."""
        text = cls._evidence_query_text(value).lower()
        tokens = re.findall(r"[a-z0-9][a-z0-9_-]*|[\u4e00-\u9fff]{2,}", text)
        unique: List[str] = []
        seen: set[str] = set()
        for token in tokens:
            normalized = token.strip("_-")
            if (
                not normalized
                or normalized in EVIDENCE_QUERY_STOPWORDS
                or normalized in CRITIQUE_META_TERMS
                or (normalized.isascii() and len(normalized) < 3)
                or normalized in seen
            ):
                continue
            seen.add(normalized)
            unique.append(normalized)
        if len(unique) <= max_terms:
            return unique
        head_count = (max_terms + 1) // 2
        tail_count = max_terms - head_count
        return [*unique[:head_count], *unique[-tail_count:]] if tail_count else unique[:head_count]

    @staticmethod
    def _append_unique_terms(target: List[str], terms: List[str]) -> None:
        seen = set(target)
        for term in terms:
            if term not in seen:
                target.append(term)
                seen.add(term)

    @classmethod
    def _evidence_query_anchors(
        cls,
        idea: Dict[str, Any],
        dataset_info: Dict[str, Any],
    ) -> List[str]:
        base_dataset = dataset_info.get("base_dataset", {}) if isinstance(dataset_info, dict) else {}
        data_description = (
            base_dataset.get("data_description", {})
            if isinstance(base_dataset, dict)
            else {}
        )
        dataset_name = (
            data_description.get("dataset_name")
            if isinstance(data_description, dict)
            else ""
        )
        component_names = []
        if isinstance(base_dataset, dict) and base_dataset.get("type") == "composite_dataset":
            component_names = [
                str(component.get("name") or "").strip()
                for component in (base_dataset.get("dataset_components") or [])
                if isinstance(component, dict) and str(component.get("name") or "").strip()
            ]
        if not dataset_name and isinstance(base_dataset, dict):
            dataset_name = base_dataset.get("dataset_name")
        explicit_domain = dataset_info.get("domain") if isinstance(dataset_info, dict) else ""
        knowledge_anchor = idea.get("knowledge_anchor_and_motivation")
        macro_phenomenon = (
            knowledge_anchor.get("macro_phenomenon")
            if isinstance(knowledge_anchor, dict)
            else knowledge_anchor
        )
        anchors: List[str] = []
        for value, budget in (
            (explicit_domain, 3),
            (macro_phenomenon, 3),
            (idea.get("research_question"), 3),
            (dataset_name, 3),
            (" ".join(component_names), 6),
            (base_dataset.get("region_name") if isinstance(base_dataset, dict) else "", 2),
        ):
            cls._append_unique_terms(anchors, cls._evidence_query_terms(value, budget))
            if len(anchors) >= EVIDENCE_ANCHOR_QUERY_TERMS:
                break
        return anchors[:EVIDENCE_ANCHOR_QUERY_TERMS]

    @classmethod
    def _evidence_focus_terms(cls, finding: Dict[str, Any]) -> List[str]:
        terms: List[str] = []
        cls._append_unique_terms(
            terms,
            cls._evidence_query_terms(
                finding.get("evidence_need"),
                EVIDENCE_NEED_QUERY_TERMS,
            ),
        )
        cls._append_unique_terms(
            terms,
            cls._evidence_query_terms(
                finding.get("modification_direction"),
                EVIDENCE_MODIFICATION_QUERY_TERMS,
            ),
        )
        if not terms:
            cls._append_unique_terms(
                terms,
                cls._evidence_query_terms(
                    finding.get("question"),
                    EVIDENCE_QUESTION_QUERY_TERMS,
                ),
            )
        return terms

    @classmethod
    def _dataset_gap_terms(cls, evidence_need: Any) -> List[str]:
        return [
            term
            for term in cls._evidence_query_terms(
                evidence_need,
                EVIDENCE_NEED_QUERY_TERMS,
            )
            if term not in DATASET_CAPABILITY_GENERIC_TERMS
        ]

    @staticmethod
    async def _skipped_channel_outcome() -> Dict[str, Any]:
        """通道判定未要求该通道时的占位结果，不发起任何检索。"""
        return {"query": "", "results": [], "errors": [], "skipped": True}

    @staticmethod
    def _channel_diagnosis(
        *,
        raw_count: int,
        relevant_count: int,
        query: str,
        near_miss_count: int = 0,
        skipped: bool = False,
    ) -> str:
        """区分一个检索通道为何没有产出可用证据。

        只有 `recalled_but_irrelevant` 才支持"这份数据确实不存在/不适用"的判断；
        `no_recall` 与 `near_miss_only` 都是检索侧未定，不能用来否定 idea。
        `not_required` 是第三类：通道判定说这条 finding 不需要该通道，压根没检索，
        与"检索了但没结果"必须分开，否则诊断计数会把浪费的检索算成检索失效。
        """
        if skipped:
            return "not_required"
        if relevant_count:
            return "grounded"
        if not query:
            return "no_query"
        if raw_count:
            return "recalled_but_irrelevant"
        if near_miss_count:
            # catalog 有主题匹配的条目，只是绑定分不足：存在性未定，不是不存在。
            return "near_miss_only"
        return "no_recall"

    @classmethod
    def _dataset_query_terms(cls, evidence_need: Any) -> List[str]:
        """数据集检索 query 用的词：gap terms 再剥掉方法学动词，并限制长度。

        catalog 召回按 token overlap 打分，query 越长噪声词越多，命中越差。
        `_dataset_gap_terms` 的完整结果仍用于候选校验的 gap-term 命中计数。
        """
        return [
            term
            for term in cls._dataset_gap_terms(evidence_need)
            if term not in DATASET_QUERY_METHODOLOGY_TERMS
        ][:DATASET_QUERY_MAX_TERMS]

    @staticmethod
    def _bounded_query(terms: List[str]) -> str:
        query = " ".join(terms)
        if len(query) <= EVIDENCE_QUERY_MAX_CHARS:
            return query
        bounded = query[:EVIDENCE_QUERY_MAX_CHARS].rsplit(" ", 1)[0].strip()
        return bounded

    @classmethod
    def _build_evidence_queries(
        cls,
        finding: Dict[str, Any],
        idea: Dict[str, Any],
        dataset_info: Dict[str, Any],
    ) -> Dict[str, str]:
        focus_terms = cls._evidence_focus_terms(finding)
        cls._append_unique_terms(
            focus_terms,
            cls._evidence_query_terms(
                finding.get("question"),
                EVIDENCE_QUESTION_QUERY_TERMS,
            ),
        )
        literature_terms = list(focus_terms)
        cls._append_unique_terms(
            literature_terms,
            cls._evidence_query_anchors(idea, dataset_info),
        )
        dataset_terms = cls._dataset_query_terms(finding.get("evidence_need"))
        return {
            "literature": cls._bounded_query(literature_terms),
            "dataset": cls._bounded_query(dataset_terms),
        }

    @classmethod
    def _evidence_result_relevance(
        cls,
        result: Dict[str, Any],
        finding: Dict[str, Any],
        idea: Dict[str, Any],
        dataset_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        focus_terms = cls._evidence_focus_terms(finding)
        anchor_terms = cls._evidence_query_anchors(idea, dataset_info)
        result_terms = set(cls._evidence_query_terms(result, max_terms=200))
        matched_focus = [term for term in focus_terms if term in result_terms]
        matched_anchors = [term for term in anchor_terms if term in result_terms]
        # 保持宽松：这一层是**预筛**，精度由 _judge_evidence_relevance 的语义终裁负责。
        # 曾试过追加"必须命中 anchor"，在 Geomorpho90m_Slope 实测 126 条通过的文献里
        # 只有 9 条无 anchor 命中，其中 7 条（SMAP 降尺度、DEM 超分、土壤水力保持）
        # 确实相关，噪声仅 2 条——收紧会为剔 2 条误杀 7 条，净损失。真正的噪声源是
        # query 把检索拉出学科（由 classifier 检索式解决），不是这条规则太松。
        relevant = len(matched_focus) >= 3 or (
            bool(matched_anchors)
            and len(matched_focus) >= min(2, len(focus_terms))
            and bool(focus_terms)
        )
        if not focus_terms:
            reason = "no_modification_focus_terms"
        elif relevant:
            reason = "modification_focus_supported"
        else:
            reason = "insufficient_modification_focus_overlap"
        return {
            "relevant": relevant,
            "reason": reason,
            "matched_focus_terms": matched_focus,
            "matched_anchor_terms": matched_anchors,
            "required_rule": (
                "focus_hits>=3 or focus_hits>=min(2, available_focus_terms) "
                "with anchor_hit>=1"
            ),
        }

    @classmethod
    def _filter_relevant_evidence(
        cls,
        results: List[Dict[str, Any]],
        finding: Dict[str, Any],
        idea: Dict[str, Any],
        dataset_info: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        relevant_results: List[Dict[str, Any]] = []
        decisions: List[Dict[str, Any]] = []
        for index, result in enumerate(results):
            if not isinstance(result, dict):
                continue
            decision = cls._evidence_result_relevance(result, finding, idea, dataset_info)
            identity = str(result.get("doi") or result.get("id") or result.get("title") or index)
            decisions.append({"result_id": identity, **decision})
            if decision["relevant"]:
                relevant_results.append({
                    **copy.deepcopy(result),
                    "relevance": decision,
                })
        return relevant_results, decisions

    async def _judge_evidence_relevance(
        self,
        finding: Dict[str, Any],
        literature_results: List[Dict[str, Any]],
        dataset_results: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """对词面预筛幸存的候选做一次批量 LLM 终裁。

        词面重合是预筛：便宜，但对"泛词凑数"的假阳性与"措辞不同"的假阴性都无判别力，
        而 grounded 状态直接决定 finding 能否授权修订。终裁一条 finding 只发一次短调用，
        失败或关闭时显式回退词面结果并在审计里标明，不做静默兜底。
        """
        if not self.enable_evidence_relevance_judge:
            return literature_results, dataset_results, {"judge": "disabled"}
        if not literature_results and not dataset_results:
            return literature_results, dataset_results, {"judge": "skipped_empty"}
        literature_payload = [
            {
                "title": item.get("title"),
                "year": item.get("year"),
                "journal": item.get("journal"),
                "abstract": str(item.get("abstract") or "")[:300],
            }
            for item in literature_results
        ]
        dataset_payload = [
            {
                "title": item.get("title"),
                "id": item.get("id"),
                "description": str(
                    item.get("description")
                    or (item.get("detail") or {}).get("description")
                    or ""
                )[:300],
            }
            for item in dataset_results
        ]
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "literature": {
                    "type": "array",
                    "maxItems": max(1, len(literature_payload)),
                    "items": {"type": "boolean"},
                },
                "dataset": {
                    "type": "array",
                    "maxItems": max(1, len(dataset_payload)),
                    "items": {"type": "boolean"},
                },
            },
            "required": ["literature", "dataset"],
        }
        try:
            result = await self._call_model(
                prompt=build_idea_evidence_relevance_prompt(
                    evidence_need=self._evidence_query_text(
                        finding.get("evidence_need")
                    )[:EVIDENCE_CHANNEL_NEED_MAX_CHARS],
                    modification_direction=str(
                        finding.get("modification_direction") or ""
                    )[:EVIDENCE_CHANNEL_NEED_MAX_CHARS],
                    literature_json=self._json(literature_payload),
                    dataset_json=self._json(dataset_payload),
                ),
                system_prompt=IDEA_EVIDENCE_RELEVANCE_SYSTEM_PROMPT,
                schema=schema,
                temperature=0.0,
                max_tokens=self.evidence_relevance_max_tokens,
            )
        except Exception as exc:
            logger.warning("Evidence relevance judge failed: %s", exc)
            return literature_results, dataset_results, {"judge": "failed"}
        literature_flags = result.get("literature") if isinstance(result, dict) else None
        dataset_flags = result.get("dataset") if isinstance(result, dict) else None
        if (
            not isinstance(literature_flags, list)
            or len(literature_flags) != len(literature_results)
            or not isinstance(dataset_flags, list)
            or len(dataset_flags) != len(dataset_results)
        ):
            # 长度不齐说明模型没有逐条对应，判定不可用；不猜测对齐关系。
            return literature_results, dataset_results, {"judge": "invalid_output"}
        kept_literature = [
            item
            for item, keep in zip(literature_results, literature_flags)
            if bool(keep)
        ]
        kept_datasets = [
            item for item, keep in zip(dataset_results, dataset_flags) if bool(keep)
        ]
        audit = {
            "judge": "applied",
            "literature_flags": [bool(flag) for flag in literature_flags],
            "dataset_flags": [bool(flag) for flag in dataset_flags],
            "literature_rejected_count": len(literature_results) - len(kept_literature),
            "dataset_rejected_count": len(dataset_results) - len(kept_datasets),
        }
        return kept_literature, kept_datasets, audit

    @staticmethod
    def _dataset_candidate_capability_text(candidate: Dict[str, Any]) -> Dict[str, Any]:
        detail = candidate.get("detail") if isinstance(candidate.get("detail"), dict) else {}
        return {
            "title": candidate.get("title"),
            "id": candidate.get("id"),
            "description": candidate.get("description") or detail.get("description"),
            "bands": candidate.get("bands") or detail.get("bands"),
        }

    @classmethod
    def _validate_dataset_candidates(
        cls,
        candidates: List[Dict[str, Any]],
        evidence_need: Any,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        gap_terms = cls._dataset_gap_terms(evidence_need)
        accepted: List[Dict[str, Any]] = []
        decisions: List[Dict[str, Any]] = []

        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                continue
            candidate_terms = set(cls._evidence_query_terms(
                cls._dataset_candidate_capability_text(candidate),
                max_terms=200,
            ))
            matched_gap_terms = [term for term in gap_terms if term in candidate_terms]
            minimum_hits = 1 if len(gap_terms) == 1 else 2
            relevant = bool(gap_terms) and len(matched_gap_terms) >= minimum_hits
            decision = {
                "relevant": relevant,
                "matched_gap_terms": matched_gap_terms,
                "gap_terms": gap_terms,
                "reason": (
                    "candidate metadata names the missing data capability"
                    if relevant
                    else "candidate metadata does not name enough missing data capability terms"
                ),
                "required_rule": "one hit for a one-term gap, otherwise at least two hits",
            }
            identity = str(candidate.get("id") or candidate.get("title") or index)
            decisions.append({"result_id": identity, **decision})
            if relevant:
                accepted.append(copy.deepcopy(candidate))
        return accepted, decisions

    @staticmethod
    def _model_evidence_records(
        evidence_records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Exclude raw rejected recalls from revision and acceptance model inputs.

        `queries` 与 `evidence_summary` 也被排除：
        - `queries` 是检索用的词袋，acceptance prompt 自己写明 "Search queries are not
          evidence; only concrete returned literature and dataset records are evidence."
          既然规则禁止模型据此判断，就不该传。
        - `evidence_summary` 是 "Confirmed 3/5 literature ...: <title>, <title>" 形式的
          自然语言复述，其中的 title 逐字来自同一条记录的 `literature_results`，模型会
          收到两份同样的信息。计数信息由 `literature_results` / `dataset_results` 的长度
          直接可得。
        两者仍完整保留在 output.json 的审计记录里，只是不进模型输入。
        """
        model_fields = (
            "evidence_record_id",
            "finding_id",
            "target_field",
            "modification_direction",
            "required_channels",
            "literature_results",
            "dataset_results",
            "retrieval_errors",
            "status",
            "evidence_limits",
        )
        return [
            {
                field: copy.deepcopy(record[field])
                for field in model_fields
                if field in record
            }
            for record in evidence_records
            if isinstance(record, dict)
        ]

    def _supplement_literature_from_cache(
        self,
        *,
        finding: Dict[str, Any],
        idea: Dict[str, Any],
        dataset_info: Dict[str, Any],
        online_results: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        target_count = self.evidence_results_per_source
        audit: Dict[str, Any] = {
            "enabled": self._filtered_literature_cache is not None,
            "attempted": False,
            "target_count": target_count,
            "online_relevant_count": len(online_results),
            "max_publication_year": self.max_publication_year,
            "cache_candidate_count": 0,
            "cache_relevant_count": 0,
            "cache_selected_count": 0,
            "cache_write_count": 0,
        }
        if getattr(self, "research_context", None) is not None:
            audit.update(enabled=True, source="unified_research_memory", attempted=False)
            return [], [], audit
        if self._filtered_literature_cache is None or len(online_results) >= target_count:
            return [], [], audit

        audit["attempted"] = True
        audit["cache_path"] = str(self._filtered_literature_cache.path)
        cached_papers = self._filtered_literature_cache.load()
        cached_papers = [
            paper
            for paper in cached_papers
            if publication_year_is_allowed(
                paper.get("year"),
                self.max_publication_year,
            )
        ]
        audit["cache_candidate_count"] = len(cached_papers)
        existing_keys = {
            FilteredLiteratureCache.paper_identity_key(paper)
            for paper in online_results
            if FilteredLiteratureCache.paper_identity_key(paper)
        }
        cached_keys = set()
        ranked: List[Tuple[int, int, str, Dict[str, Any], Dict[str, Any]]] = []
        for index, paper in enumerate(cached_papers):
            key = FilteredLiteratureCache.paper_identity_key(paper)
            if not key or key in existing_keys or key in cached_keys:
                continue
            cached_keys.add(key)
            decision = self._evidence_result_relevance(paper, finding, idea, dataset_info)
            if not decision["relevant"]:
                continue
            evidence_paper = self._evidence_paper(paper)
            evidence_paper["retrieval_source"] = "filtered_literature_cache"
            evidence_paper["retrieval_fallback"] = "filtered_literature_cache"
            match_score = (
                2 * len(decision["matched_focus_terms"])
                + len(decision["matched_anchor_terms"])
            )
            evidence_paper["cache_match_score"] = match_score
            evidence_paper["relevance"] = copy.deepcopy(decision)
            year = paper.get("year")
            year_sort = int(year) if isinstance(year, int) else 0
            title = str(paper.get("title") or "").lower()
            ranked.append((match_score, year_sort, title, evidence_paper, decision))

        ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        audit["cache_relevant_count"] = len(ranked)
        missing_count = target_count - len(online_results)
        selected_rows = ranked[:missing_count]
        selected = [copy.deepcopy(row[3]) for row in selected_rows]
        decisions = [
            {
                "result_id": str(
                    paper.get("doi") or paper.get("title") or f"cache_{index}"
                ),
                "retrieval_fallback": "filtered_literature_cache",
                **copy.deepcopy(decision),
            }
            for index, (_, _, _, paper, decision) in enumerate(selected_rows)
        ]
        audit["cache_selected_count"] = len(selected)
        return selected, decisions, audit

    def _write_relevant_literature_to_cache(
        self,
        papers: List[Dict[str, Any]],
    ) -> int:
        context_manager = getattr(self, "research_context", None)
        if context_manager is not None:
            before = len(context_manager.memory.documents)
            context_manager.ingest({"context": {"literature": papers}})
            return len(context_manager.memory.documents) - before
        if self._filtered_literature_cache is None or not papers:
            return 0
        papers = [
            paper
            for paper in papers
            if publication_year_is_allowed(
                paper.get("year"),
                self.max_publication_year,
            )
        ]
        if not papers:
            return 0
        task_name = self.config.get("task_name")
        return self._filtered_literature_cache.add_papers(
            papers,
            source_stage="idea_maturation_grounding",
            task_name=str(task_name) if task_name is not None else "",
        )

    async def _retrieve_literature_for_finding(
        self,
        finding: Dict[str, Any],
        idea: Dict[str, Any],
        dataset_info: Dict[str, Any],
        *,
        search_query: str = "",
    ) -> Dict[str, Any]:
        # 优先用通道判定翻译出的文献检索式（领域锚点 + 名词短语）；缺失时回退
        # 词袋抽取或结构化查询。回退路径必须在产物里可分辨（query_source），否则召回率差异
        # 无法归因。
        if search_query.strip():
            # 路径 1: 通道判定给出了明确的检索式
            query_source = "classifier_search_query"
            query = self._anchor_literature_query(search_query.strip(), dataset_info)
            topic_names = None
        else:
            # 路径 2: 回退到结构化查询构造
            query_source = "structured_query"
            structured = StructuredLiteratureQuery.build_structured_query(
                finding, idea, dataset_info
            )
            query = structured["primary_query"]
            topic_names = structured.get("topic_names") or None
            # 如果结构化查询也失败，回退到旧的词袋路径
            if not query:
                query_source = "gap_terms"
                query = self._build_evidence_queries(finding, idea, dataset_info)["literature"]
                topic_names = None

        if not query:
            return {
                "query": "",
                "query_source": query_source,
                "max_publication_year": self.max_publication_year,
                "results": [],
                "errors": ["empty_literature_query"],
            }
        if self._literature_search is None:
            tools_config = (self.config.get("_global_config") or {}).get("tools") or {}
            literature_config = tools_config.get("literature_search") or {}
            api_keys = dict(literature_config.get("api_keys") or {})
            if literature_config.get("openalex_key") and not api_keys.get("openalex"):
                api_keys["openalex"] = literature_config["openalex_key"]
            if (
                literature_config.get("springer_nature_key")
                and not api_keys.get("springer_nature")
            ):
                api_keys["springer_nature"] = literature_config["springer_nature_key"]
            self._literature_search = LiteratureSearch(
                api_keys=api_keys,
                max_publication_year=self.max_publication_year,
            )

        jobs = [
            self._literature_search.search_openalex(
                query,
                max_results=self.evidence_results_per_source,
                topic_names=topic_names,
            ),
            self._literature_search.search_arxiv(
                query,
                max_results=self.evidence_results_per_source,
            ),
        ]
        outcomes = await asyncio.gather(*jobs, return_exceptions=True)
        results: List[Dict[str, Any]] = []
        errors: List[str] = []
        seen: set[str] = set()
        for source, outcome in zip(EVIDENCE_LITERATURE_SOURCES, outcomes):
            if isinstance(outcome, BaseException):
                errors.append(f"{source}:{type(outcome).__name__}:{outcome}")
                continue
            for paper in outcome or []:
                payload = paper.to_dict() if isinstance(paper, PaperMetadata) else dict(paper)
                if not publication_year_is_allowed(
                    payload.get("year"),
                    self.max_publication_year,
                ):
                    continue
                title = str(payload.get("title") or "").strip()
                key = str(payload.get("doi") or title).strip().lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                payload["retrieval_source"] = source
                evidence_paper = self._evidence_paper(payload)
                if str(payload.get("url") or "").strip():
                    evidence_paper["url"] = str(payload["url"]).strip()
                results.append(evidence_paper)
        return {
            "query": query,
            "query_source": query_source,
            "sources": list(EVIDENCE_LITERATURE_SOURCES),
            "max_publication_year": self.max_publication_year,
            "results": results,
            "errors": errors,
        }

    @classmethod
    def _anchor_literature_query(
        cls,
        query: str,
        dataset_info: Dict[str, Any],
    ) -> str:
        """检索式缺领域锚点时确定性补上，给歧义词指定解释空间（对齐 Survey 侧做法）。"""
        domain = str(
            dataset_info.get("domain") if isinstance(dataset_info, dict) else ""
        ).strip()
        if not domain:
            return query
        domain_terms = cls._evidence_query_terms(domain, 3)
        query_tokens = set(re.findall(r"[a-z0-9][a-z0-9_-]*", query.lower()))
        if not domain_terms or query_tokens & set(domain_terms):
            return query
        return cls._bounded_query([*domain_terms, *query.split()])

    async def _retrieve_datasets_for_finding(
        self,
        finding: Dict[str, Any],
        idea: Dict[str, Any],
        dataset_info: Dict[str, Any],
        *,
        search_terms: str = "",
    ) -> Dict[str, Any]:
        # 优先用通道判定给出的 catalog 检索词：它把"研究缺口描述"翻译成"数据集会怎么
        # 命名自己"，而 _build_evidence_queries 只能从 evidence_need 抽词袋。实测 6 组
        # 配对中词袋 1 组命中、模型检索词 6 组全中。
        query = search_terms.strip() or self._build_evidence_queries(
            finding, idea, dataset_info
        )["dataset"]
        query_source = "classifier_search_terms" if search_terms.strip() else "gap_terms"
        if not query:
            return {"query": "", "query_source": "none", "results": [], "errors": []}
        # base_dataset 故意传空：这里找的是 idea 当前缺的那份数据，不是 base 的近邻。
        # scout 会把 base 的 dataset_name/region_name 拼成额外 query，实测那只会把
        # base_dataset 自己召回来，再被 _validate_dataset_candidates 的 gap-term 检查
        # 正确地滤掉，纯属浪费一次检索。召回质量由 _dataset_query_terms 负责。
        if self._candidate_scout is None:
            self._candidate_scout = CandidateDatasetScout(enable_online_scouting=True)
        if self._candidate_enricher is None:
            self._candidate_enricher = DatasetDetailEnricher()
        try:
            # 一次检索取到放宽档，再按 status 分流：可绑定的进 results 当证据，
            # unresolved_semantic 只计数。这一档（catalog 主题/类型/族匹配但绑定分低于
            # 阈值）不能当证据——实测把它放进 results 后，gap-term 词面检查会放过
            # "Ocean Heat Fluxes" 回答城市人为热通量、"Surface Urban Heat Islands"
            # 回答灌溉强度这类假阳性，让 idea 拿着错数据通过验收，比零召回更糟。
            # 它的作用只是阻止下游把零命中读成"该数据不存在"。
            scout_result = await self._candidate_scout.scout(
                {},
                query,
                top_n=self.evidence_dataset_top_n,
                accepted_statuses=EXISTENCE_EVIDENCE_STATUSES,
            )
            candidates = list(scout_result.get("candidate_datasets") or [])
            near_miss_candidates = list(
                scout_result.get("near_miss_candidates") or []
            )
            unresolved_candidates = list(
                scout_result.get("unresolved_semantic_candidates") or []
            )
            enriched = self._candidate_enricher.enrich_batch(candidates)
            return {
                "query": query,
                "query_source": query_source,
                "sources": ["local_gee", "online_gee", "nasa_cmr", "planetary_computer", "pangaea", "osm"],
                "results": [
                    self._evidence_candidate(candidate)
                    for candidate in enriched
                    if isinstance(candidate, dict)
                ],
                "near_miss_count": len(near_miss_candidates) + len(unresolved_candidates),
                # 被 status 门槛丢弃的候选必须留下身份，否则假阴性不可诊断。此前只记
                # near_miss_count，实测五个任务 103 条 near_miss_only 记录的
                # raw_dataset_results 与 dataset_decisions 全为空，无法从产物看出丢了
                # 什么——"GPWv411: Population Density"（正是 evidence_need 点名要的
                # 全球人口密度栅格）被 unresolved_semantic 挡掉这件事，只能靠重跑
                # scout 才发现。这里只做记录，不改变判定。
                "near_miss_candidates": [
                    self._near_miss_candidate(candidate)
                    for candidate in [*near_miss_candidates, *unresolved_candidates]
                    if isinstance(candidate, dict)
                ],
                "errors": [],
            }
        except Exception as exc:
            return {
                "query": query,
                "query_source": query_source,
                "sources": ["local_gee", "online_gee", "nasa_cmr", "planetary_computer", "pangaea", "osm"],
                "results": [],
                "errors": [f"dataset:{type(exc).__name__}:{exc}"],
            }

    async def _lookup_then_retrieve(self, channel, finding, idea, dataset_info, **kwargs):
        memory = getattr(self, "research_context", None)
        if memory is not None:
            cached = await memory.lookup(channel, finding, self._call_dual_tree_model)
            if cached is not None:
                return cached
        method = self._retrieve_literature_for_finding if channel == "literature" else self._retrieve_datasets_for_finding
        started = time.monotonic()
        try:
            outcome = await method(finding, idea, dataset_info, **kwargs)
        finally:
            if memory is not None:
                memory.memory.append_event({"kind": "retrieval_timing", "channel": channel,
                    "elapsed_seconds": time.monotonic() - started})
        if memory is not None:
            key = "literature_results" if channel == "literature" else "dataset_results"
            memory.ingest({"records": [{key: outcome.get("results", [])}]})
        return outcome

    async def _run_idea_evidence_grounding(
        self,
        findings: List[Dict[str, Any]],
        idea: Dict[str, Any],
        dataset_info: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Ground declared external needs; internal repairs need no retrieval."""
        records: List[Dict[str, Any]] = []
        for finding in findings:
            required_channels, channel_decision = (
                await self._resolve_required_evidence_channels(finding)
            )
            if self._writeback_needs_no_external_evidence(
                {"required_channel_decision": channel_decision}
            ):
                records.append({
                    "evidence_record_id": f"evidence_{finding['finding_id']}",
                    "finding_id": finding["finding_id"],
                    "target_field": finding.get("target_field", ""),
                    "modification_direction": finding.get("modification_direction", ""),
                    "required_channels": [],
                    "required_channel_decision": channel_decision,
                    "status": "not_required",
                    "queries": {},
                    "literature_results": [], "dataset_results": [],
                    "raw_literature_results": [], "raw_dataset_results": [],
                    "retrieval_errors": [], "unsatisfied_required_channels": [],
                    "channel_diagnosis": {"literature": "not_required", "dataset": "not_required"},
                    "evidence_summary": "Internal design repair; no external evidence requested or retrieved.",
                    "evidence_limits": "Authorizes internal reasoning only, not new empirical facts or completed validation.",
                })
                continue
            # 只跳过判定明确不需要的**数据集**检索：literature-only 的需求（要方法、
            # 定义、文档、查找表）拿去查数据集 catalog 是纯浪费，还会污染诊断——实测
            # 五个任务 103 条 near_miss_only 里混着大量"本来就不该查数据集"的记录，
            # 让"该查但查不到"这个信号读不出来。
            #
            # 文献检索**不跳过**，即使判定只要 dataset。两个原因：一是相关性过滤后的
            # 文献会并入共享文献池和 evidence_context 供树与修订使用，实测 132 条
            # required_channels==['dataset'] 的记录里 132 条都有文献命中，跳过是净损失；
            # 二是 raw_literature_results 承载"召回了但不相关"这个唯一能否定 idea 的
            # 信号，跳过会让它无法审计。
            #
            # 肯定不需要外部证据的修订已在上方直接记录；分类失败不走该路径。
            skip_dataset = bool(required_channels) and "dataset" not in required_channels
            dataset_search_terms = str(channel_decision.get("dataset_search_terms") or "")
            literature_search_query = str(
                channel_decision.get("literature_search_query") or ""
            )
            literature_outcome, dataset_outcome = await asyncio.gather(
                self._lookup_then_retrieve(
                    "literature", finding, idea, dataset_info, search_query=literature_search_query
                ),
                self._skipped_channel_outcome()
                if skip_dataset
                else self._lookup_then_retrieve(
                    "dataset", finding, idea, dataset_info, search_terms=dataset_search_terms
                ),
                return_exceptions=True,
            )
            literature = (
                {
                    "query": self._build_evidence_queries(finding, idea, dataset_info)["literature"],
                    "results": [],
                    "errors": [
                        f"literature:{type(literature_outcome).__name__}:{literature_outcome}"
                    ],
                }
                if isinstance(literature_outcome, BaseException)
                else literature_outcome
            )
            datasets = (
                {
                    "query": self._build_evidence_queries(finding, idea, dataset_info)["dataset"],
                    "results": [],
                    "errors": [f"dataset:{type(dataset_outcome).__name__}:{dataset_outcome}"],
                }
                if isinstance(dataset_outcome, BaseException)
                else dataset_outcome
            )
            online_raw_literature_results = [
                paper
                for paper in list(literature.get("results") or [])
                if isinstance(paper, dict)
                and publication_year_is_allowed(
                    paper.get("year"),
                    self.max_publication_year,
                )
            ]
            raw_dataset_results = list(datasets.get("results") or [])
            online_literature_results, literature_relevance = self._filter_relevant_evidence(
                online_raw_literature_results,
                finding,
                idea,
                dataset_info,
            )
            cached_literature, cached_relevance, literature_cache_audit = (
                self._supplement_literature_from_cache(
                    finding=finding,
                    idea=idea,
                    dataset_info=dataset_info,
                    online_results=online_literature_results,
                )
            )
            literature_cache_audit["cache_write_count"] = (
                self._write_relevant_literature_to_cache(online_literature_results)
            )
            raw_literature_results = [
                *online_raw_literature_results,
                *copy.deepcopy(cached_literature),
            ]
            literature_results = [
                *online_literature_results,
                *cached_literature,
            ]
            literature_relevance.extend(cached_relevance)
            dataset_results, dataset_relevance = self._validate_dataset_candidates(
                raw_dataset_results,
                finding.get("evidence_need"),
            )
            # 词面预筛的幸存者交批量 LLM 终裁；judge 状态（applied/failed/disabled…）
            # 随 relevance_filter 落盘，回退路径可审计。
            literature_results, dataset_results, relevance_judge = (
                await self._judge_evidence_relevance(
                    finding, literature_results, dataset_results
                )
            )
            # 对被 status 门槛丢弃的候选，另跑一次同样的 gap-term 校验但**只记录不采纳**。
            # 这是分流决策所需的唯一数据：如果这一档里多数是命中定性词的真阳性（人口密度
            # 需求命中 population/density），说明门槛把有用数据挡在外面；如果多数是凑够
            # 两个通用词的假阳性（城市人为热通量需求命中 heat/flux），说明门槛是必要的。
            near_miss_candidates = list(datasets.get("near_miss_candidates") or [])
            near_miss_audit: List[Dict[str, Any]] = []
            if near_miss_candidates:
                _, near_miss_decisions = self._validate_dataset_candidates(
                    near_miss_candidates,
                    finding.get("evidence_need"),
                )
                by_identity = {
                    str(decision.get("result_id") or ""): decision
                    for decision in near_miss_decisions
                }
                for candidate in near_miss_candidates:
                    identity = str(candidate.get("id") or candidate.get("title") or "")
                    decision = by_identity.get(identity) or {}
                    near_miss_audit.append({
                        **candidate,
                        "would_pass_gap_term_check": bool(decision.get("relevant")),
                        "matched_gap_terms": decision.get("matched_gap_terms") or [],
                    })
            errors = [
                str(error)
                for error in [*(literature.get("errors") or []), *(datasets.get("errors") or [])]
                if str(error).strip()
            ]
            channel_satisfied = {
                "literature": bool(literature_results),
                "dataset": bool(dataset_results),
            }
            # 未召回和"召回了但相关性不足"是两回事：只有后者能支持"该数据确实不存在"的判断，
            # 也只有后者是拒绝 idea 的正当理由。前者是检索侧失效，必须在产物里可分辨，
            # 否则 unresolved 率里会混入检索缺陷，看起来像 idea 的科学缺陷。
            channel_diagnosis = {
                "literature": self._channel_diagnosis(
                    raw_count=len(raw_literature_results),
                    relevant_count=len(literature_results),
                    query=str(literature.get("query") or ""),
                ),
                "dataset": self._channel_diagnosis(
                    raw_count=len(raw_dataset_results),
                    relevant_count=len(dataset_results),
                    query=str(datasets.get("query") or ""),
                    near_miss_count=int(datasets.get("near_miss_count") or 0),
                    skipped=bool(datasets.get("skipped")),
                ),
            }
            unsatisfied_channels = [
                channel
                for channel in (required_channels or [])
                if not channel_satisfied[channel]
            ]
            if required_channels and all(channel_satisfied[channel] for channel in required_channels):
                status = "grounded"
            elif not required_channels and (literature_results or dataset_results):
                status = "grounded"
            elif errors:
                status = "retrieval_failed"
            elif any(
                channel_diagnosis[channel] in {"no_recall", "near_miss_only", "no_query"}
                for channel in unsatisfied_channels
            ):
                # 检索器没给出可判定的候选：不能据此断言该数据不存在，
                # 因此这条 finding 属于"未验证"，不是"已被证据驳回"。
                status = "retrieval_no_recall"
            else:
                status = "insufficient"
            titles = [
                str(item.get("title") or item.get("id") or "").strip()
                for item in [*literature_results, *dataset_results]
                if isinstance(item, dict) and str(item.get("title") or item.get("id") or "").strip()
            ]
            finding_id = str(finding.get("finding_id") or "")
            records.append({
                "evidence_record_id": f"evidence_{finding_id}",
                "finding_id": finding_id,
                "target_field": str(finding.get("target_field") or ""),
                "modification_direction": str(finding.get("modification_direction") or ""),
                "required_channels": required_channels,
                "required_channel_decision": channel_decision,
                "queries": {
                    "literature": str(literature.get("query") or ""),
                    "dataset": str(datasets.get("query") or ""),
                    # classifier_search_terms 表示用的是模型翻译的产品名，gap_terms
                    # 表示回退到了词袋。两者召回率差距很大，需要能分开统计。
                    "dataset_query_source": str(datasets.get("query_source") or ""),
                    # 同理：classifier_search_query = 模型翻译的文献检索式，
                    # gap_terms = 词袋回退。
                    "literature_query_source": str(literature.get("query_source") or ""),
                },
                "literature_results": literature_results,
                "dataset_results": dataset_results,
                "raw_literature_results": raw_literature_results,
                "raw_dataset_results": raw_dataset_results,
                "literature_cache": literature_cache_audit,
                "relevance_filter": {
                    "online_raw_literature_count": len(online_raw_literature_results),
                    "online_relevant_literature_count": len(online_literature_results),
                    "raw_literature_count": len(raw_literature_results),
                    "relevant_literature_count": len(literature_results),
                    "raw_dataset_count": len(raw_dataset_results),
                    "relevant_dataset_count": len(dataset_results),
                    "literature_decisions": literature_relevance,
                    "dataset_decisions": dataset_relevance,
                    "judge": relevance_judge,
                },
                "retrieval_errors": errors,
                "status": status,
                "channel_diagnosis": channel_diagnosis,
                "unsatisfied_required_channels": unsatisfied_channels,
                # 纯审计：被 status 门槛丢弃的候选及其在现有 gap-term 规则下的假设判定。
                # 不进模型输入，不参与任何路由或状态判定。
                "near_miss_dataset_candidates": near_miss_audit,
                "evidence_summary": (
                    f"Confirmed {len(literature_results)}/{len(raw_literature_results)} "
                    f"literature and {len(dataset_results)}/{len(raw_dataset_results)} "
                    "dataset records as relevant"
                    + (f": {', '.join(titles[:5])}" if titles else "")
                ),
                "evidence_limits": (
                    "Retrieved records support idea-level revision only; execution parameters "
                    "and validation protocols remain StepPlan responsibilities."
                    if status == "grounded"
                    else (
                        "Retrieval returned no candidate for this need, so the need is "
                        "unverified; absence of a retrieved record is not evidence that the "
                        "data does not exist."
                        if status == "retrieval_no_recall"
                        else "No retrieved record can support an idea-field writeback."
                    )
                ),
            })
        return records

    @classmethod
    def _build_idea_revision_package(
        cls,
        findings: List[Dict[str, Any]],
        evidence_records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        evidence_by_finding = {
            str(record.get("finding_id") or ""): record
            for record in evidence_records
            if isinstance(record, dict)
        }
        package: List[Dict[str, Any]] = []
        for finding in findings:
            finding_id = str(finding.get("finding_id") or "")
            evidence = evidence_by_finding.get(finding_id, {})
            evidence_supports_writeback, evidence_route_reason = (
                cls._evidence_supports_idea_writeback(finding, evidence)
            )
            # 未声明的 observable 依赖是 idea 内部的声明自洽缺陷：修复动作是把测量源声明进
            # data_synergy_and_gap，或把 observable 收窄到已声明数据支撑的范围——两者都靠
            # idea 自身改写完成，不需要外部检索。它必须在 evidence 门槛之前短路，否则会因
            # 分类器判 needs_dataset、检索又拿不到 grounded 而落 unresolved，observable 永远
            # 不被退守（正是 3DEP/HAND desync 的成因）。core_contradiction 仍优先：那是改
            # 字段消不掉的逻辑矛盾，短路只让声明缺口走改写，不放行真正的内部矛盾。
            undeclared_observable = bool(finding.get("undeclared_observable_dependency"))
            if finding.get("is_core_contradiction"):
                route = "unresolved"
                route_reason = "the finding is a core contradiction"
            elif undeclared_observable:
                route = "idea_field_update"
                route_reason = (
                    "the observable variable lacks a declared measuring source; the repair "
                    "declares that source or retreats the observable using the idea's own "
                    "reasoning, so no external retrieval is required"
                )
            elif not evidence_supports_writeback:
                route = "unresolved"
                route_reason = evidence_route_reason
            elif cls._requires_step_plan_handoff(finding):
                route = "step_plan_open_point"
                route_reason = (
                    "the evidence is available, but the requested decision belongs to "
                    "detailed execution planning"
                )
            else:
                route = "idea_field_update"
                route_reason = evidence_route_reason
            authorized_fields = (
                cls._revision_fields([finding]) if route == "idea_field_update" else []
            )
            evidence_record_id = str(evidence.get("evidence_record_id") or "")
            # 路由已判定这条 finding 靠 idea 自身推理即可修时，它引用的检索记录必然不是
            # grounded。把这个前提显式落到 package 上，验收侧据此豁免；记录 id 仍然保留，
            # 溯源不丢。undeclared_observable 短路同属"靠自身推理即可修"：它是声明自洽缺陷，
            # 与检索是否 grounded 无关，因此一并豁免证据依赖，避免撞 unsupported_evidence_dependency。
            evidence_requirement = (
                EVIDENCE_REQUIREMENT_NONE
                if route == "idea_field_update"
                and (
                    undeclared_observable
                    or cls._writeback_needs_no_external_evidence(evidence)
                )
                else EVIDENCE_REQUIREMENT_GROUNDED
            )
            package.append({
                "revision_package_item_id": f"revision_{finding_id}",
                "finding_id": finding_id,
                "target_field": str(finding.get("target_field") or ""),
                "resolution_route": route,
                "writeback_intent": str(
                    finding.get("modification_direction")
                    or finding.get("evidence_need")
                    or finding.get("question")
                    or ""
                ),
                "authorized_fields": authorized_fields,
                "evidence_record_ids": [evidence_record_id] if evidence_record_id else [],
                "evidence_requirement": evidence_requirement,
                "route_reason": route_reason,
            })
        cls._co_authorize_consistency_fields(package)
        return package

    @staticmethod
    def _co_authorize_consistency_fields(package: List[Dict[str, Any]]) -> None:
        """把同轮被锁、却与已授权改写直接冲突的字段放开做一致性写回。

        某条 finding 因证据未 grounded 被路由 unresolved 时它的目标字段就被锁住，而同轮
        另一条 finding 的改写往往必须动到同一批字段才能自洽：改了 falsification 要求控制
        某个混淆，mechanism 却仍写着该混淆不存在，验收判 idea_incoherent。修订因此没有
        能收敛的路径——它被要求只改一半。

        这里只做**一致性授权**：被锁 item 的 route 不变（finding 仍完整进 unresolved /
        step_plan 审计），仅允许修订把这些字段对齐到改写后的主张，不授权引入新的证据
        依赖或把该 finding 断言成已解决——前者由验收的 evidence_dependencies_supported
        与打分的 combination_essentiality 把关，后者由 revision_package_consistent 与
        deferred_findings_preserved 把关。这类字段记入 `consistency_authorized_fields`，
        prompt 据此区分"可对齐"与"可解决"两种授权。
        """
        synergy_field = "data_synergy_and_gap"
        claim_chain_items = [
            item
            for item in package
            if item.get("resolution_route") == "idea_field_update"
            and set(item.get("authorized_fields") or []) & CLAIM_SEMANTIC_FIELDS
        ]
        if not claim_chain_items:
            return
        # 只放开同轮真的被 finding 指到、却没拿到写回授权的字段；不是把主张链无条件全开。
        locked_fields = {
            str(item.get("target_field") or "")
            for item in package
            if item.get("resolution_route") != "idea_field_update"
        }
        consistency_fields = locked_fields & (CLAIM_SEMANTIC_FIELDS | {synergy_field})
        if not consistency_fields:
            return
        # 允许主张链中有语义依赖的字段一起对齐；授权不代表这些字段必须同时改写。
        if DIRECTIONAL_CLAIM_CHAIN & consistency_fields:
            consistency_fields |= DIRECTIONAL_CLAIM_CHAIN
        # 本轮被授权去**解决**的 finding 的目标字段不能标成"只可对齐"：那正是这轮要
        # 修的缺陷落点，标窄了会与 writeback_intent 冲突，并撞 revision_target_resolved。
        resolvable_targets = {
            str(item.get("target_field") or "")
            for item in package
            if item.get("resolution_route") == "idea_field_update"
        }
        consistency_marked = consistency_fields - resolvable_targets
        for item in claim_chain_items:
            authorized = set(item.get("authorized_fields") or [])
            added = consistency_fields - authorized
            if added:
                item["authorized_fields"] = [
                    field for field in IDEA_FIELDS if field in (authorized | added)
                ]
                item["route_reason"] = (
                    str(item.get("route_reason") or "")
                    + "; "
                    + ", ".join(field for field in IDEA_FIELDS if field in added)
                    + " co-authorized for consistency writeback"
                )
            # 标记范围比 added 大：`_revision_fields` 的 DIRECTIONAL_CLAIM_CHAIN 整链扩展
            # 会把被 defer 的 finding 的目标字段顺带放进 authorized_fields。那些字段同样
            # 处在"字段可改 + finding 未解决"的状态，只是不必再扩充授权。不标记它们，
            # package 给模型的就是一句自相矛盾的输入：一边授权改，一边说未解决。
            marked = consistency_marked & (authorized | added)
            if marked:
                item["consistency_authorized_fields"] = [
                    field for field in IDEA_FIELDS if field in marked
                ]

    @staticmethod
    def _writeback_needs_no_external_evidence(evidence: Dict[str, Any]) -> bool:
        """通道判定是否**肯定地**得出"这条 finding 不需要任何外部证据"。

        只认两种肯定结论：evidence_need 为空（source=no_evidence_need），或分类器判定
        needs_dataset/needs_literature 双 false（prompt 对这个组合的定义就是"可由 idea
        自身推理解决的定义性/内部一致性修正"）。classifier_disabled / _unavailable /
        _invalid 这类保守回退**不算**肯定结论，避免"翻译失败"被读成"不需要证据"。

        路由（`_evidence_supports_idea_writeback`）与 package 标记
        （`evidence_requirement`）必须共用这一个判定，否则两侧口径会漂移。
        """
        decision = evidence.get("required_channel_decision") or {}
        source = str(decision.get("source") or "")
        if source == "no_evidence_need":
            return True
        return bool(
            source == "classifier"
            and not decision.get("needs_dataset")
            and not decision.get("needs_literature")
        )

    @classmethod
    def _evidence_supports_idea_writeback(
        cls,
        finding: Dict[str, Any],
        evidence: Dict[str, Any],
    ) -> Tuple[bool, str]:
        if cls._writeback_needs_no_external_evidence(evidence):
            return (
                True,
                "channel classification found no external evidence requirement, so the "
                "revision is authorized on the idea's own reasoning",
            )
        if evidence.get("status") == "retrieval_no_recall":
            # 检索器零召回不等于该数据不存在，路由理由必须说明这是检索侧失效，
            # 避免下游把它读成"证据表明这个 idea 站不住"。
            return (
                False,
                "retrieval returned no candidate at all for the declared evidence need, "
                "so the need is unverified rather than refuted",
            )
        if evidence.get("status") != "grounded":
            return False, "evidence retrieval did not produce a relevant grounded result"

        # required_channels 由 _resolve_required_evidence_channels 判定后写入 record，
        # 这里直接采用。不再回退到 _required_evidence_channels：那是字面标记快速路径，
        # 对识别不出的措辞返回空表，会把"需要外部数据"重新读成"无通道要求"。
        required_channels = list(evidence.get("required_channels") or [])

        diagnosis = evidence.get("channel_diagnosis") or {}
        missing_channels = [
            channel
            for channel in required_channels
            if not evidence.get(f"{channel}_results")
        ]
        if missing_channels:
            required = " and ".join(required_channels)
            missing = " and ".join(missing_channels)
            detail = " and ".join(
                f"{channel}:{diagnosis.get(channel, 'unknown')}"
                for channel in missing_channels
            )
            return (
                False,
                f"declared evidence need requires relevant {required}, but no {missing} "
                f"result was grounded ({detail})",
            )
        if required_channels:
            required = " and ".join(required_channels)
            return (
                True,
                f"grounded evidence satisfies the declared {required} need for an "
                "idea-level writeback",
            )
        return True, "retrieved relevant evidence is available for an idea-level writeback"

    @classmethod
    def _requires_step_plan_handoff(cls, finding: Dict[str, Any]) -> bool:
        text = cls._evidence_query_text([
            finding.get("question"),
            finding.get("modification_direction"),
            finding.get("evidence_need"),
        ]).lower()
        return any(marker in text for marker in STEP_PLAN_OPEN_POINT_MARKERS)

    @staticmethod
    def _revision_findings(
        findings: List[Dict[str, Any]],
        revision_package: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        authorized_ids = {
            str(item.get("finding_id") or "")
            for item in revision_package
            if item.get("resolution_route") == "idea_field_update"
        }
        return [
            finding
            for finding in findings
            if str(finding.get("finding_id") or "") in authorized_ids
        ]

    @staticmethod
    def _handoff_notes_for_step_plan(
        revision_package: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return [
            {
                "finding_id": str(item.get("finding_id") or ""),
                "target_field": str(item.get("target_field") or ""),
                "open_point": str(item.get("writeback_intent") or ""),
                "evidence_record_ids": list(item.get("evidence_record_ids") or []),
            }
            for item in revision_package
            if item.get("resolution_route") == "step_plan_open_point"
        ]

    @staticmethod
    def _unresolved_from_package(
        revision_package: List[Dict[str, Any]],
        evidence_records: List[Dict[str, Any]],
        round_number: int,
    ) -> List[Dict[str, Any]]:
        evidence_by_id = {
            str(item.get("evidence_record_id") or ""): item
            for item in evidence_records
            if isinstance(item, dict)
        }
        unresolved: List[Dict[str, Any]] = []
        for item in revision_package:
            if item.get("resolution_route") != "unresolved":
                continue
            evidence_ids = list(item.get("evidence_record_ids") or [])
            evidence = next(
                (
                    evidence_by_id[evidence_id]
                    for evidence_id in evidence_ids
                    if evidence_id in evidence_by_id
                ),
                {},
            )
            statuses = [
                str(evidence_by_id[evidence_id].get("status") or "")
                for evidence_id in evidence_ids
                if evidence_id in evidence_by_id
            ]
            required_channels = {
                str(channel).strip().lower()
                for channel in (evidence.get("required_channels", []) or [])
                if str(channel).strip()
            }
            dataset_requirement_unresolved = (
                "dataset" in required_channels
                and not list(evidence.get("dataset_results", []) or [])
            )
            unresolved_item = {
                "round": round_number,
                "kind": (
                    "dataset_requirement_unresolved"
                    if dataset_requirement_unresolved
                    else "evidence_unresolved"
                ),
                "finding_id": str(item.get("finding_id") or ""),
                "target_field": str(item.get("target_field") or ""),
                "evidence_status": statuses[0] if statuses else "missing",
                "reason": str(item.get("route_reason") or ""),
            }
            if dataset_requirement_unresolved:
                queries = evidence.get("queries") if isinstance(evidence.get("queries"), dict) else {}
                unresolved_item["dataset_query"] = str(queries.get("dataset") or "").strip()
                retrieval_errors = [
                    str(error).strip()
                    for error in (evidence.get("retrieval_errors", []) or [])
                    if str(error).strip()
                ]
                if retrieval_errors:
                    unresolved_item["retrieval_errors"] = retrieval_errors
            unresolved.append(unresolved_item)
        return unresolved

    @classmethod
    def _check_observable_data_binding(
        cls,
        idea: Dict[str, Any],
    ) -> Dict[str, Any]:
        """检查 idea 的 observable 与 data_synergy_and_gap 的绑定状态。

        返回结构化状态供 FV 评分器判断 idea_operational_coherence：
        - observable_source_declared: data_synergy_and_gap 是否声明了能测量 observable 的数据源
        - observable_body_mentioned: observable 本身（非派生统计量）的数据来源是否在任何字段中明确提及
        - unresolved_dependency: 检测到的未声明依赖列表（observable 需要但 data_synergy 未提及的数据）

        这是 idea-internal 的声明一致性检查，不判断数据是否真实可得（后者是 ResourceCheck 职责）。
        """
        result = {
            "observable_source_declared": False,
            "observable_body_mentioned": False,
            "unresolved_dependency": [],
        }

        falsifiability = idea.get(FALSIFIABILITY_FIELD)
        if not isinstance(falsifiability, dict):
            return result

        observable = falsifiability.get("observable", "")
        if not isinstance(observable, str) or not observable.strip():
            return result

        observable_lower = observable.lower()
        data_synergy = idea.get("data_synergy_and_gap", "")
        if not isinstance(data_synergy, str):
            data_synergy = ""
        data_synergy_lower = data_synergy.lower()

        # 检测常见的需要专门数据源的 observable 模式（非从 base_dataset 直接读取的情况）
        # 这些模式通常需要在 data_synergy_and_gap 中明确声明测量来源
        specialized_observable_patterns = [
            # 时间序列派生指标
            ("recession constant", ["streamflow", "gauge", "discharge", "hydrograph"]),
            ("response time", ["event", "observation", "measurement", "record"]),
            ("trend coefficient", ["time series", "temporal", "historical"]),
            # 标签和分类数据
            ("susceptibility", ["event", "inventory", "label", "occurrence", "inundation"]),
            ("vulnerability", ["damage", "loss", "impact", "claim"]),
            ("classification", ["label", "category", "class", "training"]),
            # 验证和对比数据
            ("validation", ["reference", "ground truth", "in-situ", "field"]),
            ("bias", ["reference", "benchmark", "observation"]),
            ("anomaly", ["baseline", "climatology", "expected", "historical"]),
        ]

        # 检查 observable 是否需要专门数据源
        requires_specialized_source = False
        required_source_keywords = []
        for pattern, keywords in specialized_observable_patterns:
            if pattern in observable_lower:
                requires_specialized_source = True
                required_source_keywords.extend(keywords)

        # 如果需要专门数据源，检查 data_synergy_and_gap 是否声明了
        if requires_specialized_source and required_source_keywords:
            has_source_declaration = any(
                kw in data_synergy_lower for kw in required_source_keywords
            )
            result["observable_source_declared"] = has_source_declaration

            # 如果没有声明，记录为 unresolved_dependency
            if not has_source_declaration:
                # 提取 observable 中明确提到的数据类型（如 "streamflow gauge"）
                for pattern, keywords in specialized_observable_patterns:
                    if pattern in observable_lower:
                        result["unresolved_dependency"].append(
                            f"Observable mentions '{pattern}' but data source for it is not declared in data_synergy_and_gap"
                        )
                        break
        else:
            # 不需要专门数据源（如直接从 base_dataset 计算的指标），默认认为声明充分
            result["observable_source_declared"] = True

        # 检查 observable body 是否在任何字段中被提及
        # 这是一个较宽松的检查：observable 本身（或其核心测量对象）应该在 idea 的某处被讨论
        observable_tokens = set(observable_lower.split())
        idea_text_fields = [
            "knowledge_anchor_and_motivation",
            "data_synergy_and_gap",
            "research_question",
            "core_hypothesis",
            "theoretical_mechanism",
        ]
        idea_text = " ".join(
            str(idea.get(field, "")).lower()
            for field in idea_text_fields
        )

        # 如果 observable 的主要词汇（排除 "coefficient", "correlation" 等统计术语）在 idea 中多次出现，
        # 认为 observable body 被充分讨论
        statistical_terms = {"coefficient", "correlation", "ratio", "index", "value", "score"}
        core_observable_tokens = observable_tokens - statistical_terms
        if core_observable_tokens:
            mentions = sum(1 for token in core_observable_tokens if token in idea_text and len(token) > 3)
            result["observable_body_mentioned"] = mentions >= 2
        else:
            result["observable_body_mentioned"] = True

        return result

    @classmethod
    def _scoring_context(
        cls,
        evidence_context: Dict[str, Any],
        idea: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """从当前 evidence_context 派生 FV 打分上下文。

        `literature` 取初始快照并在整个成熟化过程中冻结：新增文献只改变评分者对
        新颖性与机制的主观印象，不构成"某个辅助源是否可得"这类事实，却会让同一个
        idea 版本在不同轮次拿到不同分数。`candidate_datasets` 相反，必须跟随最新
        证据——`combination_essentiality` 按它判断 idea 声明的外部辅助源是否真的存在。

        新增 `idea_data_contract_status`：检查 idea 内部的 observable-data 绑定声明一致性，
        供 FV 评分器判断 `idea_operational_coherence` 时使用。这是 idea-level 的声明检查，
        不涉及数据的真实可达性（后者由 ResourceCheck 负责）。
        """
        context = copy.deepcopy(evidence_context)
        if idea is not None:
            context["idea_data_contract_status"] = cls._check_observable_data_binding(idea)
        return context

    @classmethod
    def _refresh_scoring_datasets(
        cls,
        scoring_context: Dict[str, Any],
        evidence_context: Dict[str, Any],
        idea: Dict[str, Any],
    ) -> Dict[str, Any]:
        """把累积后的候选数据集同步进打分上下文，literature 保持冻结。

        每轮 grounded 检索后调用。返回新对象而不是原地改，让"打分上下文只在这一处
        变化、且只有 candidate_datasets 会变"这件事在调用点可见。

        同时刷新 idea_data_contract_status：每轮修订后 idea 的 observable/data_synergy 可能变化，
        需要重新检查绑定状态供 FV 评分器使用。
        """
        refreshed = copy.deepcopy(scoring_context)
        datasets = evidence_context.get("candidate_datasets")
        if datasets:
            refreshed["candidate_datasets"] = copy.deepcopy(datasets)
        # 刷新 observable-data 绑定状态检查
        refreshed["idea_data_contract_status"] = cls._check_observable_data_binding(idea)
        return refreshed

    @classmethod
    def _merge_grounded_evidence_context(
        cls,
        evidence_context: Dict[str, Any],
        evidence_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        merged = copy.deepcopy(evidence_context)
        literature = list(merged.get("literature") or [])
        datasets = list(merged.get("candidate_datasets") or [])
        seen_papers = {
            str(item.get("doi") or item.get("title") or "").strip().lower()
            for item in literature
            if isinstance(item, dict)
        }
        seen_datasets = {
            str(item.get("id") or item.get("title") or "").strip().lower()
            for item in datasets
            if isinstance(item, dict)
        }
        for record in evidence_records:
            if record.get("status") == "grounded":
                for paper in record.get("literature_results") or []:
                    key = str(paper.get("doi") or paper.get("title") or "").strip().lower()
                    if key and key not in seen_papers:
                        seen_papers.add(key)
                        literature.append(copy.deepcopy(paper))
            for dataset in record.get("dataset_results") or []:
                key = str(dataset.get("id") or dataset.get("title") or "").strip().lower()
                if key and key not in seen_datasets:
                    seen_datasets.add(key)
                    datasets.append(copy.deepcopy(dataset))
        if literature:
            merged["literature"] = literature
        if datasets:
            merged["candidate_datasets"] = datasets
        return merged

    def _append_dataset_results_to_candidates(
        self,
        evidence_records: List[Dict[str, Any]],
    ) -> None:
        context_manager = getattr(self, "research_context", None)
        if context_manager is not None:
            context_manager.ingest({"records": evidence_records})
            self._candidate_datasets = context_manager.memory.evidence_context()["candidate_datasets"]
            return
        seen = {
            str(item.get("id") or item.get("title") or "").strip().lower()
            for item in self._candidate_datasets
            if isinstance(item, dict)
        }
        for record in evidence_records:
            for dataset in record.get("dataset_results") or []:
                if not isinstance(dataset, dict):
                    continue
                key = str(dataset.get("id") or dataset.get("title") or "").strip().lower()
                if not key or key in seen:
                    continue
                self._candidate_datasets.append(copy.deepcopy(dataset))
                seen.add(key)

    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        idea_result = copy.deepcopy(context.get("idea_result") or {})
        self._candidate_datasets = list(context.get("candidate_datasets") or [])
        literature_review = list(context.get("literature_review") or [])
        if self.max_publication_year is not None:
            literature_review = [
                paper
                for paper in literature_review
                if isinstance(paper, dict)
                and publication_year_is_allowed(
                    paper.get("year"),
                    self.max_publication_year,
                )
            ]
        self._literature_review = literature_review
        plans = idea_result.get("application_plans") or []
        if not isinstance(plans, list):
            return {
                "idea_result": idea_result,
                "idea_maturation_traces": [],
                "candidate_datasets": copy.deepcopy(self._candidate_datasets),
            }

        # 使用信号量控制并发度，避免同时检索文献导致限流
        semaphore = asyncio.Semaphore(self.max_concurrent_ideas)

        async def mature_one_plan(plan: Dict[str, Any]) -> tuple:
            """成熟化单个 plan，受信号量控制"""
            if not isinstance(plan, dict):
                return plan, None
            idea = self._extract_idea(plan)
            if not idea:
                return plan, None

            async with semaphore:
                matured, trace = await self._mature_idea(
                    idea=idea,
                    dataset_info=context.get("dataset_info") or {},
                )

            updated_plan = copy.deepcopy(plan)
            updated_plan["idea_candidate"] = matured
            updated_plan["idea_candidates"] = [copy.deepcopy(matured)]
            updated_plan["idea_revision_audit"] = trace.pop("revision_audit")
            return updated_plan, trace

        # 并发处理所有 plan，但通过信号量限制同时运行的数量
        results = await asyncio.gather(
            *[mature_one_plan(plan) for plan in plans],
            return_exceptions=True,
        )

        traces: List[Dict[str, Any]] = []
        matured_plans: List[Dict[str, Any]] = []
        for result in results:
            if isinstance(result, BaseException):
                logger.warning("Idea maturation failed: %s", result)
                continue
            matured_plan, trace = result
            matured_plans.append(matured_plan)
            if trace is not None:
                traces.append(trace)

        idea_result["application_plans"] = matured_plans
        idea_result["idea_candidates"] = [
            copy.deepcopy(plan["idea_candidate"])
            for plan in matured_plans
            if isinstance(plan.get("idea_candidate"), dict)
        ]
        return {
            "idea_result": idea_result,
            "idea_maturation_traces": traces,
            "candidate_datasets": copy.deepcopy(self._candidate_datasets),
        }

    @staticmethod
    def _extract_idea(plan: Dict[str, Any]) -> Dict[str, Any]:
        candidate = plan.get("idea_candidate")
        if isinstance(candidate, dict):
            return copy.deepcopy(candidate)
        candidates = plan.get("idea_candidates")
        if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
            return copy.deepcopy(candidates[0])
        return {}

    @staticmethod
    def _semantic_idea(idea: Dict[str, Any]) -> Dict[str, Any]:
        allowed_fields = (*IDEA_IDENTITY_FIELDS, *IDEA_FIELDS)
        return {
            field: copy.deepcopy(idea[field])
            for field in allowed_fields
            if field in idea
        }

    @staticmethod
    def _selection_idea(idea: Dict[str, Any]) -> Dict[str, Any]:
        """Return only source-blind semantic content for FV selection."""
        allowed_fields = ("idea_type", "component_usage", *IDEA_FIELDS)
        return {
            field: copy.deepcopy(idea[field])
            for field in allowed_fields
            if field in idea
        }

    async def _mature_idea(
        self,
        *,
        idea: Dict[str, Any],
        dataset_info: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        memory = getattr(self, "research_context", None)
        if memory is not None and not memory.memory.initialized:
            memory.ingest({"context": self._evidence_context(dataset_info)})
            memory.memory.initialized = True
            memory.memory.persist()
        current = self._semantic_idea(idea)
        rounds: List[Dict[str, Any]] = []
        tree = PersistentIdeaSocraticTree()
        latest_questioning_findings: List[Dict[str, Any]] = []
        latest_evidence_records: List[Dict[str, Any]] = []
        latest_revision_package: List[Dict[str, Any]] = []
        latest_applied_revisions: List[Dict[str, Any]] = []
        latest_handoff_notes: List[Dict[str, Any]] = []
        latest_semantic_writeback_checks: Dict[str, Any] = {}
        maturation_rationale = ""
        unresolved_findings: List[Dict[str, Any]] = []
        consecutive_unapplied_rounds = 0
        consecutive_fv_no_improvement = 0
        # 全路由轮（revision package 非空但没有任何 idea_field_update）的连续计数。
        consecutive_routed_rounds = 0
        # evidence_context 会随每轮 grounded 检索增长，供质询树与修订使用。
        # 打分用独立的 scoring_context：FV 打分的变量应当是 idea 本身，而不是上下文长度。
        # 文献不进打分上下文——它只影响评分者对新颖性/机制的主观印象，实测文献增长会让
        # 同一个 idea 版本拿到不同分数，使 consecutive_fv_no_improvement 变成上下文噪声。
        # 数据集证据必须进：`combination_essentiality` 明确按
        # `candidate_datasets` 判断 idea 声明的外部辅助源是否真的可得，缺了它，
        # 引用新检索数据集的改写会被判成 undeclared_or_unavailable_dependency，
        # 等于惩罚正确的修订。
        evidence_context = self._evidence_context(dataset_info)
        scoring_context = self._scoring_context(evidence_context, current)
        original_audit = {
            "questioning_findings": [],
            "evidence_records": [],
            "revision_package": [],
            "applied_revisions": [],
            "handoff_notes_for_step_plan": [],
            "semantic_writeback_checks": {},
            "maturation_rationale": "",
            "unresolved_findings": [],
        }
        initial_selection_fv = await self._score_selection_fv(current, scoring_context)
        initial_selection_fv_axes = self._selection_fv_axes(initial_selection_fv)
        initial_candidate = {
            "version": "V0",
            "source_round": 0,
            "idea": copy.deepcopy(current),
            "revision_audit": original_audit,
            "selection_fv": initial_selection_fv,
        }
        candidate_factory = getattr(self, "_scored_candidate_record", None)
        if candidate_factory is not None:
            initial_candidate = candidate_factory(current, initial_selection_fv, 0, original_audit)
        candidates: List[Dict[str, Any]] = [initial_candidate]
        online_best_fv: Optional[float] = (
            initial_selection_fv["fv_product"] if initial_selection_fv["valid"] else None
        )
        online_best_version: Optional[str] = "V0" if initial_selection_fv["valid"] else None
        online_best_selection_fv: Optional[Dict[str, Any]] = (
            copy.deepcopy(initial_selection_fv) if initial_selection_fv["valid"] else None
        )
        online_stop_reason = ""
        if self._selection_target_reached(initial_selection_fv):
            online_stop_reason = "selection_fv_target_reached"

        for round_number in range(1, self.max_tree_depth // 2 + 1):
            if online_stop_reason:
                break
            if hasattr(self, "research_consistency_gate"):
                self.research_consistency_gate.round_number = round_number
            raw_findings, dual_tree_trace, active_leaves = await self.dual_tree.grow_depth_pair(
                tree=tree,
                idea=current,
                # 树按"候选是否被当前 idea 引用"分两档；revision 与 selection FV 仍用完整
                # evidence_context。树是唯一把这份上下文重发上百次的消费者。
                evidence_context=self._tiered_evidence_context(evidence_context, current),
                stage=round_number,
            )
            findings = self._prepare_questioning_findings(
                raw_findings,
                active_leaves,
                round_number,
            )
            # 微型实证：只给本轮被裁决可改写的方向性 finding 附加一行组间统计，
            # 供修订 prompt 的 Directional Claim Restatement Rule 与验收门使用。
            await self._attach_micro_empirical_checks(findings, current, dataset_info)
            evidence_records: List[Dict[str, Any]] = []
            revision_package: List[Dict[str, Any]] = []
            revision_findings: List[Dict[str, Any]] = []
            handoff_notes: List[Dict[str, Any]] = []
            if findings:
                evidence_records = await self._run_idea_evidence_grounding(
                    findings,
                    current,
                    dataset_info,
                )
                revision_package = self._build_idea_revision_package(
                    findings,
                    evidence_records,
                )
                revision_findings = self._revision_findings(findings, revision_package)
                handoff_notes = self._handoff_notes_for_step_plan(revision_package)
                self._append_dataset_results_to_candidates(evidence_records)
                evidence_context = self._merge_grounded_evidence_context(
                    evidence_context,
                    evidence_records,
                )
                # 新检索到的数据集要进打分上下文（否则引用它们的改写会被
                # combination_essentiality 误判为"辅助源不可得"），新文献不进。
                # 同时刷新 idea_data_contract_status，因为 current 的 observable/data_synergy 可能已变化。
                scoring_context = self._refresh_scoring_datasets(
                    scoring_context,
                    evidence_context,
                    current,
                )
                package_unresolved = self._unresolved_from_package(
                    revision_package,
                    evidence_records,
                    round_number,
                )
                unresolved_keys = {
                    (item.get("round"), item.get("finding_id"), item.get("kind"))
                    for item in unresolved_findings
                }
                unresolved_findings.extend(
                    item
                    for item in package_unresolved
                    if (item.get("round"), item.get("finding_id"), item.get("kind"))
                    not in unresolved_keys
                )

            # 只有 revision package 明确授权的 idea_field_update 才能进入 executor。
            if revision_findings:
                candidate, rationale, acceptance, rejected_attempts, last_patch_empty = await self._revise_until_accepted(
                    current=current,
                    findings=revision_findings,
                    evidence_records=evidence_records,
                    revision_package=revision_package,
                    dataset_info=dataset_info,
                )
            else:
                candidate, rationale, rejected_attempts, last_patch_empty = current, "", [], False
                acceptance = {
                    "passed": False,
                    "failure_codes": [],
                    "reason": (
                        "no evidence-grounded idea-field update in this depth pair"
                        if findings
                        else "no revision-actionable finding in this depth pair"
                    ),
                }
            accepted = bool(acceptance.get("passed"))
            if revision_package and not revision_findings:
                consecutive_routed_rounds += 1
            elif revision_findings:
                consecutive_routed_rounds = 0
            if accepted:
                consecutive_unapplied_rounds = 0
            elif revision_findings:
                # 只数 package 已授权 idea 写回却没能落地的轮次；纯 StepPlan/unresolved
                # 路由不是修订失败，不计入 revision stall。
                consecutive_unapplied_rounds += 1
            round_trace = {
                "round": round_number,
                "questioning_findings": findings,
                "evidence_records": evidence_records,
                "revision_package": revision_package,
                "handoff_notes_for_step_plan": handoff_notes,
                "socratic_dual_tree": dual_tree_trace,
                "accepted": accepted,
                "acceptance": acceptance,
                "revision_attempts": rejected_attempts,
                "consecutive_unapplied_rounds": consecutive_unapplied_rounds,
                "consecutive_routed_rounds": consecutive_routed_rounds,
                "applied_idea_field_updates": self._changed_fields(current, candidate) if accepted else [],
                # 这是修订前树叶的诊断快照，不是版本 FV，也不参与终止或选版。
                "pressure_audit": self._pressure_audit(active_leaves),
                "selection_fv": None,
                "selection_fv_axes": {},
            }
            rounds.append(round_trace)
            if accepted:
                current = candidate
                tree.stage_diffs.append(round_trace["applied_idea_field_updates"])
                latest_questioning_findings = copy.deepcopy(findings)
                latest_evidence_records = copy.deepcopy(evidence_records)
                latest_revision_package = copy.deepcopy(revision_package)
                latest_applied_revisions = self._applied_revisions_with_provenance(
                    round_trace["applied_idea_field_updates"],
                    revision_package,
                )
                round_trace["applied_idea_field_updates"] = copy.deepcopy(
                    latest_applied_revisions
                )
                latest_handoff_notes = copy.deepcopy(handoff_notes)
                latest_semantic_writeback_checks = copy.deepcopy(acceptance)
                if not rationale:
                    rationale = "semantic acceptance passed"
                maturation_rationale = rationale[:MATURATION_REASON_MAX_CHARS]
            else:
                tree.stage_diffs.append([])
                if revision_findings:
                    # 区分两类失败：模型输出空 patch（模型/schema 问题）vs
                    # patch 存在但语义验收拒绝（科学质量问题）。两者在产物里用不同
                    # kind 记录，让质量率统计可信且不同混淆原因。
                    all_attempts_empty = all(
                        a.get("patch_empty", False) for a in rejected_attempts
                    ) if rejected_attempts else last_patch_empty
                    failure_kind = "empty_patch" if all_attempts_empty else "revision_rejected"
                    attempt_failure_codes = list(dict.fromkeys(
                        code
                        for a in rejected_attempts
                        for code in (a.get("failure_codes") or [])
                    ))
                    unresolved_findings.append({
                        "round": round_number,
                        "kind": failure_kind,
                        "finding_fields": [item.get("target_field", "") for item in revision_findings],
                        "revision_attempts": len(rejected_attempts),
                        "last_patch_empty": last_patch_empty,
                        "attempt_failure_codes": attempt_failure_codes,
                        "reason": str(acceptance.get("reason") or "semantic acceptance did not pass")[:MATURATION_REASON_MAX_CHARS],
                    })
                    # 被拒绝的尝试没有创建新版本；该 unresolved finding 属于当前候选。
            if findings and not accepted:
                latest_questioning_findings = copy.deepcopy(findings)
                latest_evidence_records = copy.deepcopy(evidence_records)
                latest_revision_package = copy.deepcopy(revision_package)
                latest_handoff_notes = copy.deepcopy(handoff_notes)
                candidates[-1]["revision_audit"].update({
                    "questioning_findings": copy.deepcopy(latest_questioning_findings),
                    "evidence_records": copy.deepcopy(latest_evidence_records),
                    "revision_package": copy.deepcopy(latest_revision_package),
                    "handoff_notes_for_step_plan": copy.deepcopy(latest_handoff_notes),
                    "unresolved_findings": copy.deepcopy(unresolved_findings),
                })

            should_stop = False
            if accepted:
                selection_fv = await self._score_selection_fv(current, scoring_context)
                round_trace["selection_fv"] = selection_fv
                round_trace["selection_fv_axes"] = self._selection_fv_axes(selection_fv)
                version = f"V{len(candidates)}"
                candidate_record = {
                    "version": version,
                    "source_round": round_number,
                    "idea": copy.deepcopy(current),
                    "revision_audit": {
                        "questioning_findings": copy.deepcopy(latest_questioning_findings),
                        "evidence_records": copy.deepcopy(latest_evidence_records),
                        "revision_package": copy.deepcopy(latest_revision_package),
                        "applied_revisions": copy.deepcopy(latest_applied_revisions),
                        "handoff_notes_for_step_plan": copy.deepcopy(latest_handoff_notes),
                        "semantic_writeback_checks": copy.deepcopy(latest_semantic_writeback_checks),
                        "maturation_rationale": maturation_rationale,
                        "unresolved_findings": copy.deepcopy(unresolved_findings),
                    },
                    "selection_fv": copy.deepcopy(selection_fv),
                }
                if candidate_factory is not None:
                    candidate_record = candidate_factory(current, selection_fv, round_number, candidate_record["revision_audit"])
                candidates.append(candidate_record)

                best_before_round = online_best_fv
                version_before_round = online_best_version
                updated_best = False
                meaningful_improvement: Optional[bool] = None
                if selection_fv["valid"]:
                    current_fv = selection_fv["fv_product"]
                    improvement_delta = (
                        None if best_before_round is None else current_fv - best_before_round
                    )
                    meaningful_improvement = (
                        best_before_round is None
                        or (
                            improvement_delta is not None
                            and improvement_delta > 0
                            and improvement_delta >= self.fv_plateau_eps
                        )
                    )
                    if meaningful_improvement:
                        consecutive_fv_no_improvement = 0
                    else:
                        # 下降、持平和小于 eps 的微升都属于没有改善。
                        consecutive_fv_no_improvement += 1
                    # 精确同分时让最新 accepted 版本成为在线 best；这不算 meaningful
                    # improvement，plateau 计数仍按上面的 eps 规则累计。
                    if online_best_fv is None or current_fv >= online_best_fv:
                        online_best_fv = current_fv
                        online_best_version = version
                        online_best_selection_fv = copy.deepcopy(selection_fv)
                        updated_best = True
                if candidate_factory is not None:
                    authoritative_best = self._select_final_candidate(candidates)
                    updated_best = authoritative_best["version"] != version_before_round or (
                        authoritative_best["selection_fv"]["fv_product"] != best_before_round)
                    online_best_version = authoritative_best["version"]
                    online_best_selection_fv = copy.deepcopy(authoritative_best["selection_fv"])
                    online_best_fv = online_best_selection_fv["fv_product"]
                round_trace["online_selection"] = {
                    "version": version,
                    "previous_best_fv_product": best_before_round,
                    "current_fv_product": selection_fv["fv_product"] if selection_fv["valid"] else None,
                    "meaningful_improvement": meaningful_improvement,
                    "consecutive_no_improvement": consecutive_fv_no_improvement,
                    "updated_best": updated_best,
                    "best_version": online_best_version,
                    "best_fv_product": online_best_fv,
                    "best_feasibility": (
                        online_best_selection_fv or {}
                    ).get("feasibility"),
                    "best_value": (online_best_selection_fv or {}).get("value"),
                }
                stop_score = online_best_selection_fv if candidate_factory is not None else selection_fv
                if self._selection_target_reached(stop_score):
                    round_trace["stop_reason"] = "selection_fv_target_reached"
                    should_stop = True
                elif (
                    consecutive_fv_no_improvement >= self.fv_plateau_rounds
                    and self._selection_plateau_can_stop(online_best_selection_fv)
                ):
                    round_trace["stop_reason"] = "selection_fv_plateau"
                    should_stop = True
                elif len(candidates) - 1 >= self.max_accepted_versions:
                    round_trace["stop_reason"] = "accepted_version_limit"
                    should_stop = True

            # 收敛的判据是"所有分支都已问透"，不是"本轮没有可改写的 finding"：
            # `continue` 表示该分支还有可施压的点，此时继续下钻才是设计意图。休眠分支
            # 本轮起不再生长，与 resolved 同样算该分支已收口。
            dormant_dimensions = {
                str(entry.get("dimension", ""))
                for entry in dual_tree_trace.get("dormant_branches", [])
                if entry.get("dimension")
            }
            if self.dual_tree.is_converged(active_leaves, dormant_dimensions):
                round_trace.setdefault("stop_reason", "dual_convergence")
                should_stop = True
            elif not accepted and active_leaves and all(
                leaf.get("verdict") in {"resolved", "evidence_needed"}
                for leaf in active_leaves
            ):
                round_trace.setdefault("stop_reason", "evidence_only_remaining")
                should_stop = True
            elif (
                revision_package
                and not revision_findings
                and consecutive_routed_rounds >= self.routed_stall_rounds
            ):
                # 连续 routed_stall_rounds 轮每条 finding 都被路由到 StepPlan 或
                # unresolved：换了树深度和证据上下文仍产不出 idea-field writeback，
                # 才判定继续长树不会有新的可修点。单轮全路由不再立即终止——那会把
                # "这一轮检索没搜到词"直接放大成"这个 idea 不用再修了"。
                round_trace.setdefault("stop_reason", "revision_package_routed")
                should_stop = True
            # 修订连续停滞时收口：finding 一直提得出来但一条都改不动，继续下钻只会把
            # 剩余深度预算烧在不会改变的 idea 上。输出语义不变，仍取历史最佳版本。
            if consecutive_unapplied_rounds >= self.revision_stall_rounds:
                round_trace.setdefault("stop_reason", "revision_stalled")
                should_stop = True
            if memory is not None:
                memory.complete_round(round_trace)
            logger.info(
                "Idea maturation round audit: %s",
                self._json(self._round_log_summary(current.get("idea_id", ""), round_trace)),
            )
            if should_stop:
                break

        # 每个候选在首次形成时只评分一次；该 selection_fv 同时用于在线终止和最终选版。
        # 后续证据增长不得追溯重评旧候选，否则模型调用波动会覆盖真实的版本差异。
        selected = self._select_final_candidate(candidates)
        final_idea = copy.deepcopy(selected["idea"])
        selected_audit = copy.deepcopy(selected["revision_audit"])
        selected_fv = selected["selection_fv"]
        # 终止前强制实证（详见 _final_directional_check）：零探针退出的 idea 在这里
        # 补一次方向性检验，矛盾则强制修复并替换最终 idea。
        final_check = await self._final_directional_check(final_idea, dataset_info, rounds)
        if final_check is not None and final_check.get("repair_accepted"):
            final_idea = final_check.pop("repaired_idea")
        candidate_versions = [
            {
                "version": item["version"],
                "source_round": item["source_round"],
                "idea": copy.deepcopy(item["idea"]),
                "selection_fv": copy.deepcopy(item["selection_fv"]),
            }
            for item in candidates
        ]
        return final_idea, {
            "idea_id": final_idea.get("idea_id", ""),
            "idea_type": final_idea.get("idea_type", ""),
            "initial_selection_fv": copy.deepcopy(initial_selection_fv),
            "initial_selection_fv_axes": initial_selection_fv_axes,
            "rounds": rounds,
            "candidate_versions": candidate_versions,
            "online_stop_reason": online_stop_reason or (
                rounds[-1].get("stop_reason", "max_tree_depth") if rounds else "max_tree_depth"
            ),
            "final_status": "preserved_original" if selected["source_round"] == 0 else "matured",
            "final_empirical_check": final_check,
            "best_selection": {
                "best_round": selected["source_round"],
                "best_version": selected["version"],
                "best_fv_product": selected_fv["fv_product"] if selected_fv["valid"] else None,
                "selected_by": "authoritative_selection_fv",
                "fv_tie_eps": self.fv_tie_eps,
            },
            # execute() 把这份 audit 搬到 application_plans[*].idea_revision_audit 供 StepPlan
            # 消费；trace 自身不再保留副本，避免 output.json 里同一份内容存两遍。
            "revision_audit": selected_audit,
        }

    async def _final_directional_check(
        self,
        idea: Dict[str, Any],
        dataset_info: Dict[str, Any],
        rounds: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """终止前强制实证：对最终存续的方向性陈述无条件再测一次。

        轮内探针的覆盖是采样性的（可能挂在其他 finding 上，存续主张本身未必被测），
        终检在终止点无条件执行——同一对比已检过时同数据同温度得到同一结果，只多一次
        短调用，换来"每条离开管线的方向性主张都至少被测量一次"的闭环保证。统计与
        主张方向矛盾时强制一轮修复修订（翻转或撤销方向），修复失败保留原版并把矛盾
        记入 trace。
        """
        record: Dict[str, Any] = {"trigger": "terminal_directional_check", "probe_attached": False}
        df = self._stat_probe_dataframe(dataset_info)
        if df is None or self._stat_probe is None:
            record["reason"] = "no dataset table available for probe"
            return record
        hypothesis = idea.get("core_hypothesis")
        directional = (
            str(hypothesis.get("directional_statement") or "")
            if isinstance(hypothesis, dict) else str(hypothesis or "")
        )
        if not directional:
            record["reason"] = "no directional statement"
            return record
        finding = {
            "finding_id": f"final_probe_{idea.get('idea_id', 'idea')}",
            "target_field": "core_hypothesis",
            "question": (
                "Does the surviving directional statement agree with the computed group "
                "contrast on the task dataset?"
            ),
            "failure_hypothesis": (
                "The surviving directional assertion contradicts the computed statistic."
            ),
        }
        try:
            probe = await self._stat_probe.probe_finding(
                df=df,
                question=finding["question"],
                failure_hypothesis=finding["failure_hypothesis"],
                directional_statement=directional,
                columns=[str(c) for c in df.columns],
            )
        except Exception as exc:
            logger.warning("final empirical probe failed for %s: %s", idea.get("idea_id"), exc)
            probe = None
        if not probe or (probe.get("contrast") or {}).get("status") != "ok":
            record["reason"] = "claim not extractable as a data-decidable contrast"
            return record
        finding["micro_empirical_check"] = probe
        directive = probe.get("directive") or {}
        record.update({
            "probe_attached": True,
            "comparison": directive.get("comparison"),
            "observed_sign": directive.get("observed_sign"),
            "claimed_sign": (probe.get("spec") or {}).get("claimed_sign"),
            "sign_agreement": directive.get("sign_agreement"),
            "observed_fragment": directive.get("observed_fragment"),
            "contradicted": directive.get("sign_agreement") is False,
            "pre_repair_directional": directional[:400],
        })
        if not record["contradicted"]:
            return record
        # 方向核所有权归统计：directional_statement 由代码从统计量构造（模型不再渲染
        # 方向句，"主语互换保命题"的换皮在结构上不可能），模型的修复任务是把其余字段
        # （机制/代理主张/问题/可证伪条件）调整到与该方向一致。
        outcome = str((probe.get("contrast") or {}).get("outcome_column") or "")
        a_key = str((probe.get("contrast") or {}).get("group_a") or "")
        b_key = str((probe.get("contrast") or {}).get("group_b") or "")
        if directive.get("observed_sign") in ("a_gt_b", "a_lt_b"):
            directional_core = str(directive.get("observed_fragment") or "")
        else:
            directional_core = (
                f"No significant difference in {outcome} between {a_key} and {b_key} "
                f"(MWU p={(probe.get('contrast') or {}).get('mwu_p_two_sided')}); "
                "the originally asserted direction is withdrawn."
            )
        fields = self._revision_fields([finding])
        package = [{
            "revision_package_item_id": "revision_final_probe",
            "finding_id": finding["finding_id"],
            "target_field": "core_hypothesis",
            "resolution_route": "idea_field_update",
            "writeback_intent": (
                f"The empirical check OWNS the direction. The final "
                f"core_hypothesis.directional_statement will be set verbatim to: "
                f"\"{directional_core}\" — do not write it, and do not assert any other "
                f"direction anywhere. Your task: revise research_question, "
                f"proxy_or_inference_claim, theoretical_mechanism, and "
                f"computability_and_falsifiability so they are scientifically consistent with "
                f"this fixed direction (mechanism may explain it; scope may bound it; the "
                f"direction itself is not negotiable)."
            ),
            "authorized_fields": fields,
            "evidence_record_ids": [],
            "evidence_requirement": EVIDENCE_REQUIREMENT_NONE,
            "route_reason": (
                "final empirical check: directional statement contradicts the computed "
                "statistic; the direction is replaced by the stat-owned core and the "
                "surrounding fields must be brought in line with it"
            ),
        }]
        try:
            repaired, rationale, acceptance, rejected, _empty = await self._revise_until_accepted(
                current=copy.deepcopy(idea),
                findings=[finding],
                evidence_records=[],
                revision_package=package,
                dataset_info=dataset_info,
            )
        except Exception as exc:
            logger.warning("final probe repair failed for %s: %s", idea.get("idea_id"), exc)
            record["repair_attempted"] = False
            repaired = copy.deepcopy(idea)
            acceptance = {"passed": False, "failure_codes": ["repair_exception"]}
        record["repair_attempted"] = True
        record["repair_accepted"] = bool(acceptance.get("passed"))
        record["repair_failure_codes"] = acceptance.get("failure_codes") or []
        record["repair_restatement"] = acceptance.get("directional_restatement")
        # 无论修复是否通过验收，方向核一律用代码构造的统计句覆盖——离开管线的方向
        # 不可能错；支持字段未对齐时如实记录（不删方向、不假装连贯）。
        final_idea_out = repaired if repaired else copy.deepcopy(idea)
        hypothesis = final_idea_out.get("core_hypothesis")
        if isinstance(hypothesis, dict):
            hypothesis["directional_statement"] = directional_core
        else:
            final_idea_out["core_hypothesis"] = {"directional_statement": directional_core}
        record["direction_source"] = "statistical_core"
        record["directional_core"] = directional_core
        record["post_repair_directional"] = directional_core[:400]
        if record["repair_accepted"]:
            record["repaired_idea"] = final_idea_out
        else:
            # 验收未通过（支持字段仍与方向不一致）：方向核依然正确，不匹配的字段
            # 如实入档供人工复核，而不是撤销方向或假装连贯。
            record["fields_alignment"] = "not_verified_after_attempts"
            record["repaired_idea"] = final_idea_out
        return record

    @staticmethod
    def _round_log_summary(idea_id: str, round_trace: Dict[str, Any]) -> Dict[str, Any]:
        selection_fv = round_trace.get("selection_fv") or {}
        fv_product = selection_fv.get("fv_product") if selection_fv.get("valid") else None
        threshold = selection_fv.get("fv_target_threshold")
        plateau_eps = selection_fv.get("fv_plateau_eps")
        plateau_min_product = selection_fv.get("fv_plateau_min_product")
        selection_fv_axes = round_trace.get("selection_fv_axes")
        if not selection_fv_axes and selection_fv:
            selection_fv_axes = IdeaMaturationAgent._selection_fv_axes(selection_fv)
        applied_updates = round_trace.get("applied_idea_field_updates") or []
        changed_fields = [
            item.get("field", "")
            for item in applied_updates
            if isinstance(item, dict) and item.get("field")
        ]
        findings = [
            {
                "finding_id": item.get("finding_id", ""),
                "target_field": item.get("target_field", ""),
                "verdict": item.get("verdict", ""),
                "summary": str(
                    item.get("adjudication")
                    or item.get("question")
                    or item.get("answer")
                    or ""
                )[:180],
            }
            for item in (round_trace.get("questioning_findings") or [])[:5]
            if isinstance(item, dict)
        ]
        stop_reason = round_trace.get("stop_reason")
        online_selection = round_trace.get("online_selection") or {}
        pressure_audit = round_trace.get("pressure_audit") or {}
        evidence_status_counts: Dict[str, int] = {}
        dataset_channel_counts: Dict[str, int] = {}
        for record in round_trace.get("evidence_records") or []:
            status = str(record.get("status") or "unknown")
            evidence_status_counts[status] = evidence_status_counts.get(status, 0) + 1
            if "dataset" in (record.get("required_channels") or []):
                diagnosis = str(
                    (record.get("channel_diagnosis") or {}).get("dataset") or "unknown"
                )
                dataset_channel_counts[diagnosis] = (
                    dataset_channel_counts.get(diagnosis, 0) + 1
                )
        resolution_routes = [
            str(item.get("resolution_route") or "")
            for item in (round_trace.get("revision_package") or [])
            if str(item.get("resolution_route") or "")
        ]
        return {
            "idea_id": idea_id,
            "round": round_trace.get("round"),
            "questioning_findings": findings,
            "evidence_status_counts": evidence_status_counts,
            "dataset_channel_diagnosis_counts": dataset_channel_counts,
            "resolution_routes": resolution_routes,
            "accepted": bool(round_trace.get("accepted")),
            "changed_fields": changed_fields,
            "selection_fv_product": fv_product,
            "selection_fv_axes": selection_fv_axes or {},
            "fv_target_threshold": threshold,
            "fv_plateau_eps": plateau_eps,
            "fv_plateau_min_product": plateau_min_product,
            "selection_fv_target_reached": stop_reason == "selection_fv_target_reached",
            "pressure_coverage": pressure_audit.get("coverage", 0.0),
            "stopped": bool(stop_reason),
            "stop_reason": stop_reason or "continue",
            "acceptance_reason": str((round_trace.get("acceptance") or {}).get("reason", ""))[:180],
            "revision_attempts": len(round_trace.get("revision_attempts") or []),
            "consecutive_unapplied_rounds": round_trace.get("consecutive_unapplied_rounds", 0),
            "consecutive_fv_no_improvement": online_selection.get("consecutive_no_improvement", 0),
            "online_best_updated": bool(online_selection.get("updated_best")),
            "online_best_version": online_selection.get("best_version"),
            "online_best_fv_product": online_selection.get("best_fv_product"),
        }

    @staticmethod
    def _pressure_audit(active_leaves: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Expose current tree pressure without turning it into a competing FV score."""
        by_axis = {
            str(leaf.get("dimension")): leaf
            for leaf in active_leaves
            if isinstance(leaf, dict) and leaf.get("dimension") in FV_AXES
        }
        axes = {
            axis: (
                {
                    "axis_score": float(by_axis[axis].get("axis_score", 0.0) or 0.0),
                    "verdict": str(by_axis[axis].get("verdict", "")),
                    "node_id": str(by_axis[axis].get("node_id", "")),
                    "evidence_need": str(by_axis[axis].get("evidence_need", "")),
                }
                if axis in by_axis
                else None
            )
            for axis in FV_AXES
        }
        covered_axes = [axis for axis in FV_AXES if axes[axis] is not None]
        return {
            "axes": axes,
            "covered_axes": covered_axes,
            "coverage": len(covered_axes) / len(FV_AXES),
        }

    @staticmethod
    def _selection_fv_axes(selection_fv: Dict[str, Any]) -> Dict[str, Optional[float]]:
        """Expose the six official selection FV axis scores without derived totals."""
        if not selection_fv.get("valid"):
            return {axis: None for axis in FV_AXES}
        return {
            axis: (
                float(selection_fv[axis])
                if type(selection_fv.get(axis)) in (int, float)
                else None
            )
            for axis in FV_AXES
        }

    @staticmethod
    def _validate_fv_axis_weights(weights: Any) -> Dict[str, float]:
        if not isinstance(weights, dict) or set(weights) != set(FV_AXES):
            raise ValueError(
                "planning.idea_maturation.fv_axis_weights must define exactly all six FV axes"
            )
        parsed = {}
        for axis in FV_AXES:
            value = weights[axis]
            if type(value) not in (int, float) or not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"FV axis weight must be a finite non-negative number: {axis}")
            parsed[axis] = float(value)
        for group_name, group in (("feasibility", FV_FEASIBILITY_AXES), ("value", FV_VALUE_AXES)):
            total = sum(parsed[axis] for axis in group)
            if not math.isclose(total, 1.0, abs_tol=1e-9):
                raise ValueError(f"FV {group_name} axis weights must sum to 1.0, got {total}")
        return parsed

    async def _score_selection_fv(
        self,
        idea: Dict[str, Any],
        evidence_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        memory = getattr(self, "research_context", None)
        if memory is not None and "selected_evidence" not in evidence_context:
            original_context = evidence_context
            evidence_context = memory.evidence_view(original_context, {"idea": idea}, stage="score")
            if "idea_data_contract_status" in original_context:
                evidence_context["idea_data_contract_status"] = copy.deepcopy(original_context["idea_data_contract_status"])
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                axis: {"type": "string", "enum": list(FV_BASIS_SCORE_MAP[axis])}
                for axis in FV_AXES
            },
            "required": list(FV_AXES),
        }
        try:
            result = await self._call_model(
                prompt=build_idea_selection_fv_prompt(
                    idea_json=self._json(self._selection_idea(idea)),
                    evidence_context_json=self._json(evidence_context),
                ),
                system_prompt=IDEA_SELECTION_FV_SYSTEM_PROMPT,
                schema=schema,
                temperature=0.0,
                max_tokens=self.questioning_max_tokens,
            )
        except Exception as exc:
            logger.warning("selection_fv scoring failed: %s", exc)
            result = {}
        valid = isinstance(result, dict) and all(
            isinstance(result.get(axis), str)
            and result[axis] in FV_BASIS_SCORE_MAP[axis]
            for axis in FV_AXES
        )
        basis_codes = {
            axis: result[axis] if valid else None
            for axis in FV_AXES
        }
        scores = {
            axis: FV_BASIS_SCORE_MAP[axis][basis_codes[axis]] if valid else 0.0
            for axis in FV_AXES
        }
        feasibility = sum(
            scores[axis] * self.fv_axis_weights[axis] for axis in FV_FEASIBILITY_AXES
        )
        value = sum(scores[axis] * self.fv_axis_weights[axis] for axis in FV_VALUE_AXES)
        return {
            **scores,
            "feasibility": feasibility,
            "value": value,
            "fv_product": feasibility * value,
            "valid": valid,
            "fv_rubric_version": FV_RUBRIC_VERSION,
            "fv_axis_weights": copy.deepcopy(self.fv_axis_weights),
            "basis_codes": basis_codes,
            "fv_target_threshold": self.fv_target_threshold,
            "fv_min_feasibility": self.fv_min_feasibility,
            "fv_min_value": self.fv_min_value,
            "fv_plateau_eps": self.fv_plateau_eps,
            "fv_plateau_min_product": self.fv_plateau_min_product,
        }

    def _selection_target_reached(self, selection_fv: Dict[str, Any]) -> bool:
        return bool(
            selection_fv.get("valid")
            and selection_fv.get("fv_product", 0.0) >= self.fv_target_threshold
            and selection_fv.get("feasibility", 0.0) >= self.fv_min_feasibility
            and selection_fv.get("value", 0.0) >= self.fv_min_value
        )

    def _selection_plateau_can_stop(
        self,
        selection_fv: Optional[Dict[str, Any]],
    ) -> bool:
        return bool(
            selection_fv
            and selection_fv.get("valid")
            and selection_fv.get("fv_product", 0.0) >= self.fv_plateau_min_product
            and selection_fv.get("feasibility", 0.0) >= self.fv_min_feasibility
            and selection_fv.get("value", 0.0) >= self.fv_min_value
        )

    def _select_final_candidate(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select by each candidate's authoritative FV, preferring the latest exact tie."""
        selected = candidates[0]
        selected_score = selected.get("selection_fv") or {}
        for candidate in candidates[1:]:
            score = candidate.get("selection_fv") or {}
            if not score.get("valid"):
                continue
            if not selected_score.get("valid") or (
                score["fv_product"] > selected_score["fv_product"] + self.fv_tie_eps
            ) or (
                score["fv_product"] == selected_score["fv_product"]
            ):
                selected = candidate
                selected_score = score
        return selected

    async def _revise_until_accepted(
        self,
        *,
        current: Dict[str, Any],
        findings: List[Dict[str, Any]],
        evidence_records: List[Dict[str, Any]],
        revision_package: List[Dict[str, Any]],
        dataset_info: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], str, Dict[str, Any], List[Dict[str, Any]], bool]:
        """在同一深度对内重试修订，直到语义验收通过或用尽尝试次数。

        每次重试都从未改动的 `current` 出发，只把上一轮的 failure_codes / reason 回灌
        给修订模型：被驳回的候选不能当下一次的基线，否则错误会累积。重试发生在轮内，
        因此不消耗树的深度预算。
        """
        editable_fields = self._revision_fields_from_package(revision_package)
        rejected_attempts: List[Dict[str, Any]] = []
        candidate: Dict[str, Any] = current
        rationale = ""
        acceptance: Dict[str, Any] = {}
        last_patch_empty = False
        for attempt in range(1, self.revision_max_attempts + 1):
            patch, rationale, restatement = await self._revise_idea(
                current,
                findings,
                dataset_info,
                evidence_records=evidence_records,
                revision_package=revision_package,
                rejected_attempts=rejected_attempts,
            )
            candidate = self._apply_patch(current, patch)
            # 记录本次 attempt 是否产出了空 patch（模型输出问题 vs 语义被驳回是两类原因，
            # 必须在产物里区分，否则 unresolved_findings 的失败率无法可信测量）
            last_patch_empty = not bool(self._changed_fields(current, candidate))
            acceptance = await self._accept_revision(
                before=current,
                after=candidate,
                findings=findings,
                evidence_records=evidence_records,
                revision_package=revision_package,
                dataset_info=dataset_info,
                directional_restatement=restatement,
            )
            if acceptance.get("passed"):
                break
            rejected_attempts.append({
                "attempt": attempt,
                "rejected_patch": patch,
                "patch_empty": last_patch_empty,
                "failure_codes": list(acceptance.get("failure_codes") or []),
                "directional_restatement": restatement,
                "reason": str(acceptance.get("reason") or "")[:MATURATION_REASON_MAX_CHARS],
            })
            # 没有可编辑字段时补写只会拿回同一个空 patch，重试没有信息增益。
            if not editable_fields:
                break
            # 失败码过多时重试的期望产出接近零，见 MAX_RETRY_FAILURE_CODES。跳出只结束
            # 本轮重试：candidate 退回未改动的 current，idea 内容不受损，树照常进入下一轮。
            if len(acceptance.get("failure_codes") or []) > MAX_RETRY_FAILURE_CODES:
                break
        return candidate, rationale, acceptance, rejected_attempts, last_patch_empty

    async def _revise_idea(
        self,
        idea: Dict[str, Any],
        findings: List[Dict[str, str]],
        dataset_info: Dict[str, Any],
        *,
        evidence_records: Optional[List[Dict[str, Any]]] = None,
        revision_package: Optional[List[Dict[str, Any]]] = None,
        rejected_attempts: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, Any], str]:
        fields = (
            self._revision_fields_from_package(revision_package)
            if revision_package is not None
            else self._revision_fields(findings)
        )
        if not fields:
            return {}, "", None
        schema_properties: Dict[str, Any] = {
            "revised_idea": self._idea_patch_schema(fields),
            "revision_rationale": {"type": "string", "maxLength": 1200},
        }
        schema_required = ["revised_idea", "revision_rationale"]
        # 生成端正确性：绑定约束存在时，方向承诺是结构化必填字段（逐条 finding_id 对齐），
        # 验收门在语义评审前就能确定性地核对符号；无绑定时退回自由句式（仅审计用）。
        binding = self._binding_constraints(findings)
        if "core_hypothesis" in fields:
            if binding:
                schema_properties["directional_restatement"] = {
                    "type": "array",
                    "minItems": 1,
                    "description": (
                        "One entry per finding that carries micro_empirical_check. The committed "
                        "sign must equal that finding's directive.observed_sign, or be 'withdrawn' "
                        "(the direction is dropped for that comparison)."
                    ),
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "finding_id": {"type": "string", "maxLength": 120},
                            "surviving_sign": {
                                "type": "string",
                                "enum": ["a_gt_b", "a_lt_b", "no_difference", "withdrawn"],
                                "description": (
                                    "must equal micro_empirical_check.directive.observed_sign of "
                                    "the matching finding, or 'withdrawn'"
                                ),
                            },
                            "statement": {
                                "type": "string",
                                "maxLength": 300,
                                "description": (
                                    "the restated sentence; adopting directive.observed_fragment "
                                    "wording is the intended fast path"
                                ),
                            },
                        },
                        "required": ["finding_id", "surviving_sign", "statement"],
                    },
                }
            else:
                schema_properties["directional_restatement"] = {
                    "type": "string",
                    "maxLength": 400,
                    "description": (
                        "one sentence per surviving direction: the sign (which side is higher/"
                        "lower) and the observable comparison that decides it"
                    ),
                }
            schema_required.append("directional_restatement")
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": schema_properties,
            "required": schema_required,
        }
        model_findings = findings
        model_records = self._model_evidence_records(evidence_records or [])
        model_package = revision_package or []
        model_context = self._dataset_context_payload(dataset_info)
        memory = getattr(self, "research_context", None)
        if memory is not None:
            memory.ingest({"context": self._evidence_context(dataset_info), "records": model_records})
            stage_context = memory.for_stage("revision", {"idea": idea, "findings": findings})
            selected = stage_context["selected_evidence"]
            model_records = [{"evidence_ids": list(selected), "evidence_location": "Task Dataset Context.selected_evidence"}]
            model_context = stage_context
            model_package = [{key: value for key, value in item.items() if key not in ("evidence_records", "literature_results", "dataset_results")} for item in model_package]
        try:
            revision_call = self._call_model
            if memory is not None:
                async def revision_call(**model_kwargs):
                    return await memory.decide_with_sources(self._call_model, cards=selected,
                        history_ids=[row["episode_id"] for row in stage_context["working_context"].get("history_index", [])],
                        **model_kwargs)
            result = await revision_call(
                prompt=build_idea_maturation_revision_prompt(
                    idea_json=self._json(idea),
                    findings_json=self._json(model_findings),
                    evidence_records_json=self._json(model_records),
                    revision_package_json=self._json(model_package),
                    editable_fields_json=self._json(fields),
                    dataset_context_json=self._json(model_context),
                    claim_chain_derivation_context_json=self._json(
                        self._claim_chain_derivation_context(idea)
                    ) if self._edits_claim_chain(fields) else "",
                    rejected_attempts_json=self._json(rejected_attempts) if rejected_attempts else "",
                ),
                system_prompt=IDEA_MATURATION_REVISION_SYSTEM_PROMPT,
                schema=schema,
                temperature=0.15,
                max_tokens=self.revision_max_tokens,
            )
        except Exception as exc:
            from datahypgen.mas.tools.context_manager import ContextReadUnavailable
            if isinstance(exc, ContextReadUnavailable):
                raise
            logger.warning("Idea revision failed for %s: %s", idea.get("idea_id"), exc)
            return {}, "", None
        patch = result.get("revised_idea", {}) if isinstance(result, dict) else {}
        rationale = str(result.get("revision_rationale", ""))
        restatement = result.get("directional_restatement")
        if isinstance(restatement, list):
            rationale = f"{rationale}\nDirectional restatement: {self._json(restatement)}"
        elif restatement:
            rationale = f"{rationale}\nDirectional restatement: {restatement}"
        return self._sanitize_patch(patch, fields), rationale[:MATURATION_REASON_MAX_CHARS], restatement if isinstance(restatement, (list, dict)) else None

    @staticmethod
    def _binding_constraints(findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """从 findings 收集确定性方向指令（生成端约束，仅含已挂探针结果的 finding）。"""
        items: List[Dict[str, Any]] = []
        for finding in findings or []:
            if not isinstance(finding, dict):
                continue
            directive = (finding.get("micro_empirical_check") or {}).get("directive")
            if isinstance(directive, dict):
                items.append({"finding_id": str(finding.get("finding_id") or ""), **directive})
        return items

    @staticmethod
    def _edits_claim_chain(fields: List[str]) -> bool:
        return bool(
            set(fields)
            & (DIRECTIONAL_CLAIM_CHAIN | {FALSIFIABILITY_FIELD})
        )

    @staticmethod
    def _claim_chain_derivation_context(idea: Dict[str, Any]) -> Dict[str, Any]:
        return {
            field: copy.deepcopy(idea[field])
            for field in ("knowledge_anchor_and_motivation", "data_synergy_and_gap")
            if field in idea
        }

    @staticmethod
    def _contrast_subfield_changed(
        before: Dict[str, Any],
        after: Dict[str, Any],
    ) -> bool:
        """判断判别性对照本身是否真的变了。

        只比 `mechanism_relevant_contrast`：改 observable 的措辞或 falsification 的
        表述都不构成"能分开竞争解释"，把整个 computability_and_falsifiability 是否
        在 diff 里当判据会让换措辞重新通过。
        """
        def contrast(idea: Dict[str, Any]) -> str:
            signal = idea.get(FALSIFIABILITY_FIELD)
            if not isinstance(signal, dict):
                return ""
            value = signal.get(DISCRIMINATING_CONTRAST_SUBFIELD)
            return value.strip() if isinstance(value, str) else ""

        return contrast(before) != contrast(after)

    @staticmethod
    def _directional_sign_consistent(
        after: Dict[str, Any],
        findings: List[Dict[str, Any]],
        restatement: Optional[Any] = None,
    ) -> bool:
        """改写后的方向性主张不得与当轮探针统计的符号矛盾。

        两层判定：
        1) 结构化承诺优先——修订输出里按 finding_id 对齐的 directional_restatement
           条目，其 surviving_sign 必须等于该 finding 指令的 observed_sign 或为
           withdrawn；no_difference 的对照只允许 no_difference/withdrawn。
        2) 短语残留兜底——没有结构化条目（或条目缺失）时，若主张原文的方向词组
           （claimed_phrase）仍出现在改写后的 directional_statement 且统计方向相反，
           判不一致。限定词（normalize/control for）不影响判定。
        """
        entries_by_finding: Dict[str, Dict[str, Any]] = {}
        if isinstance(restatement, list):
            for entry in restatement:
                if isinstance(entry, dict) and entry.get("finding_id"):
                    entries_by_finding[str(entry["finding_id"])] = entry
        elif isinstance(restatement, dict) and restatement.get("finding_id"):
            entries_by_finding[str(restatement["finding_id"])] = restatement

        hypothesis = after.get("core_hypothesis") if isinstance(after, dict) else None
        if isinstance(hypothesis, dict):
            text = str(hypothesis.get("directional_statement") or "")
        elif isinstance(hypothesis, str):
            text = hypothesis
        else:
            text = ""
        normalized = " ".join(text.lower().split())

        for finding in findings:
            if not isinstance(finding, dict):
                continue
            check = finding.get("micro_empirical_check")
            if not isinstance(check, dict):
                continue
            directive = check.get("directive")
            contrast = check.get("contrast") or {}
            if (not isinstance(directive, dict)) or contrast.get("status") != "ok":
                continue
            observed = directive.get("observed_sign")
            fid = str(finding.get("finding_id") or "")
            entry = entries_by_finding.get(fid)
            if entry is not None:
                sign = entry.get("surviving_sign")
                if sign == "withdrawn":
                    continue
                if observed == "no_difference":
                    if sign in ("a_gt_b", "a_lt_b"):
                        return False
                elif sign is not None and sign != observed:
                    return False
                continue
            # 兜底：无该 finding 的结构化条目时退回短语残留检查
            spec = check.get("spec") or {}
            claimed = spec.get("claimed_sign")
            if observed in (None, "no_difference") or observed == claimed:
                continue
            phrase = " ".join(str(spec.get("claimed_phrase") or "").lower().split())
            if phrase and phrase in normalized:
                return False
        return True

    async def _accept_revision(self, **kwargs):
        result = await self._evaluate_revision(**kwargs)
        memory = getattr(self, "research_context", None)
        if memory is not None:
            memory.memory.save_episode({"kind": "revision_review", "round": len(memory.memory.rounds) + 1,
                "before_key": memory.idea_key(kwargs["before"]), "candidate_key": memory.idea_key(kwargs["after"]),
                "issue_ids": sorted(memory.issue_ids(kwargs["findings"])), "findings": copy.deepcopy(kwargs["findings"]),
                "before": copy.deepcopy(kwargs["before"]), "candidate": copy.deepcopy(kwargs["after"]),
                "accepted": result["passed"], "review": copy.deepcopy(result)})
        return result

    async def _evaluate_revision(
        self,
        *,
        before: Dict[str, Any],
        after: Dict[str, Any],
        findings: List[Dict[str, str]],
        dataset_info: Dict[str, Any],
        evidence_records: Optional[List[Dict[str, Any]]] = None,
        revision_package: Optional[List[Dict[str, Any]]] = None,
        directional_restatement: Optional[Any] = None,
    ) -> Dict[str, Any]:
        changed_fields = self._changed_fields(before, after)
        if not changed_fields:
            return {
                "passed": False,
                "revision_target_resolved": False,
                "substantive_change_landed": False,
                "research_object_preserved": False,
                "claim_scope_valid": False,
                "falsifiable_signal_added_or_clarified": False,
                "observable_within_data_affordance": False,
                "descriptive_retreat_avoided": False,
                "base_dataset_remains_relevant": False,
                "idea_fields_coherent": False,
                "revision_package_consistent": False,
                "evidence_dependencies_supported": False,
                "deferred_findings_preserved": False,
                "discriminating_contrast_updated": False,
                "failure_codes": ["no_substantive_change"],
                "reason": "revision produced no idea-field change",
            }
        semantic_writeback = self._validate_idea_semantic_writeback(
            changed_fields=changed_fields,
            findings=findings,
            evidence_records=evidence_records,
            revision_package=revision_package,
        )
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "revision_target_resolved": {"type": "boolean"},
                "substantive_change_landed": {"type": "boolean"},
                "research_object_preserved": {"type": "boolean"},
                "claim_scope_valid": {"type": "boolean"},
                "falsifiable_signal_added_or_clarified": {"type": "boolean"},
                "observable_within_data_affordance": {"type": "boolean"},
                "descriptive_retreat_avoided": {"type": "boolean"},
                "base_dataset_remains_relevant": {"type": "boolean"},
                "idea_fields_coherent": {"type": "boolean"},
                "revision_package_consistent": {"type": "boolean"},
                "evidence_dependencies_supported": {"type": "boolean"},
                "deferred_findings_preserved": {"type": "boolean"},
                # 仅用于混淆型 finding 的第二条合法出路：确实无法构造判别对照时，把
                # 主张收窄到现有对照真正支持的强度，也是正当修复。语义窄到可审计，
                # 不是"这次改写是否回应质询"那种模型系统性放行的宽问题。
                "discriminating_claim_withdrawn": {"type": "boolean"},
                "failure_codes": {
                    "type": "array",
                    "maxItems": 10,
                    "items": {
                        "type": "string",
                        "enum": [
                            "target_unresolved",
                            "no_substantive_change",
                            "research_object_lost",
                            "claim_scope_invalid",
                            "falsifiable_signal_missing",
                            "observable_outside_data_affordance",
                            "descriptive_retreat",
                            "base_dataset_irrelevant",
                            "idea_incoherent",
                            "revision_package_mismatch",
                            "unsupported_evidence_dependency",
                            "discriminating_contrast_unchanged",
                            "directional_sign_contradicts_stat",
                        ],
                    },
                },
                "reason": {"type": "string", "maxLength": 1200},
            },
            "required": [
                "revision_target_resolved",
                "substantive_change_landed",
                "research_object_preserved",
                "claim_scope_valid",
                "falsifiable_signal_added_or_clarified",
                "observable_within_data_affordance",
                "descriptive_retreat_avoided",
                "base_dataset_remains_relevant",
                "idea_fields_coherent",
                "revision_package_consistent",
                "evidence_dependencies_supported",
                "deferred_findings_preserved",
                "discriminating_claim_withdrawn",
                "failure_codes",
                "reason",
            ],
        }
        model_records = self._model_evidence_records(evidence_records or [])
        model_package = revision_package or []
        model_context = self._dataset_context_payload(dataset_info)
        memory = getattr(self, "research_context", None)
        review_call = self._call_model
        if memory is not None:
            memory.ingest({"context": self._evidence_context(dataset_info), "records": model_records})
            model_context = memory.for_stage("review", {"idea": after, "findings": findings})
            selected = model_context["selected_evidence"]
            model_records = [{"evidence_ids": list(selected), "evidence_location": "Task Dataset Context.selected_evidence"}]
            model_package = [{k: v for k, v in item.items() if k not in (
                "evidence_records", "literature_results", "dataset_results")} for item in model_package]
            async def review_call(**model_kwargs):
                return await memory.decide_with_sources(self._call_model, cards=selected,
                    history_ids=[row["episode_id"] for row in model_context["working_context"].get("history_index", [])],
                    **model_kwargs)
        try:
            result = await review_call(
                prompt=build_idea_maturation_acceptance_prompt(
                    idea_before_json=self._json(before),
                    idea_after_json=self._json(after),
                    findings_json=self._json(findings),
                    evidence_records_json=self._json(model_records),
                    revision_package_json=self._json(model_package),
                    dataset_context_json=self._json(model_context),
                ),
                system_prompt=IDEA_MATURATION_ACCEPTANCE_SYSTEM_PROMPT,
                schema=schema,
                temperature=0.0,
                max_tokens=self.acceptance_max_tokens,
            )
        except Exception as exc:
            logger.warning("Idea semantic acceptance failed for %s: %s", before.get("idea_id"), exc)
            return {
                "passed": False,
                "failure_codes": ["target_unresolved"],
                "reason": "semantic acceptance unavailable",
            }
        if not isinstance(result, dict):
            return {"passed": False, "failure_codes": ["target_unresolved"], "reason": "invalid acceptance output"}
        changed_field_names = {item["field"] for item in changed_fields}
        falsifiable_signal_complete = self._has_falsifiable_signal(after)
        # 只有改写前的信号本身不完备时，才确定性地要求这一轮把它补上。改写前已经完备的
        # 信号是否仍然匹配改写后的主张，是语义判断，交给验收模型的
        # falsifiable_signal_added_or_clarified；这里再叠加"字段必须出现在 diff 里"会把
        # "信号已正确所以无需改动"读成"信号缺失"，而 _revise_until_accepted 每次都从未改动的
        # current 重新出发，于是重试永远撞同一面墙，只能以 revision_stalled 收场。
        requires_falsifiable_update = not self._has_falsifiable_signal(before) and any(
            item.get("target_field") in CLAIM_SEMANTIC_FIELDS
            for item in findings
        )
        # 混淆型 finding（竞争解释产生相同观测）只有判别性对照能消除。不加这条门禁时，
        # 修订可以一直靠改主张措辞通过验收：实测 behavior_pattern_idea_1 四轮被采纳的
        # 修订全部只重命名同一个统计量的含义，5 个版本的
        # computability_and_falsifiability 只有 1 个取值，V1-V4 六轴分数完全相同。
        # 之所以能一直蒙混，是因为 requires_falsifiable_update 只在"改写前信号不完备"
        # 时才要求动这个字段，而 _has_falsifiable_signal 只检查三个子字段非空。
        contrast_findings = [
            item for item in findings if item.get("requires_discriminating_contrast")
        ]
        # 两条合法出路，缺一不可：
        #   1) 换一个能分开竞争解释的对照（确定性判定，主路径）；
        #   2) 确实构造不出这样的对照时，把主张收窄到现有对照真正支持的强度。
        # 第 2 条必须由模型认定，但确定性前提是 core_hypothesis 真的变了 —— 否则
        # 一个布尔就能把"什么都没做"重新放行。该布尔单独落盘，滥用率可被统计；
        # 若审计显示它被用来放行加强判别语气的改写，就收紧回只认第 1 条。
        claim_withdrawn = (
            "core_hypothesis" in changed_field_names
            and bool(result.get("discriminating_claim_withdrawn"))
        )
        contrast_changed = self._contrast_subfield_changed(before, after)
        # 仅对本轮实际针对并解决的混淆型 finding 强制要求判别对照。
        # 若本轮有非混淆型 finding 得到有效解决（如声明基准 DEM），而混淆型 finding 仍处于延后状态
        # （由 deferred_findings_preserved 保证未冒领解决），则不能因混淆型 finding 尚未解决而否决当前的非混淆改进。
        non_contrast_findings = [
            item for item in findings if not item.get("requires_discriminating_contrast")
        ]
        non_contrast_target_resolved = (
            any(item.get("target_field") in changed_field_names for item in non_contrast_findings)
            and bool(result.get("revision_target_resolved"))
            and bool(result.get("deferred_findings_preserved", True))
        )
        discriminating_contrast_satisfied = (
            not contrast_findings
            or contrast_changed
            or claim_withdrawn
            or non_contrast_target_resolved
        )
        checks = {
            # 确定性前提：至少一个 finding 的目标字段真的在改写集里。模型系统性地对
            # "改了非目标字段"的改写也判 revision_target_resolved=True。
            "revision_target_resolved": (
                any(item.get("target_field") in changed_field_names for item in findings)
                and bool(result.get("revision_target_resolved"))
            ),
            # 确定性前提：changed_fields 非空意味着确实有字段改变；模型的 True
            # 判断作为补充（防止 LLM 对实质空洞的改写虚报 True）。
            "substantive_change_landed": (
                bool(changed_fields)
                and bool(result.get("substantive_change_landed"))
            ),
            "research_object_preserved": bool(result.get("research_object_preserved")),
            "claim_scope_valid": bool(result.get("claim_scope_valid")),
            # 改写后的信号必须完备（确定性），且验收模型认可它仍然服务于改写后的主张。
            # 只有信号原本就不完备时才额外要求这一轮真的写了这个字段。
            "falsifiable_signal_added_or_clarified": (
                falsifiable_signal_complete
                and bool(result.get("falsifiable_signal_added_or_clarified"))
                and (
                    not requires_falsifiable_update
                    or FALSIFIABILITY_FIELD in changed_field_names
                )
            ),
            # 只在改写触及主张链时判定：未动 claim 字段的改写沿用改写前的可观测边界，
            # 让验收模型对它另起一轮判断只会引入噪声。
            "observable_within_data_affordance": (
                not (CLAIM_SEMANTIC_FIELDS & changed_field_names)
                or bool(result.get("observable_within_data_affordance"))
            ),
            "descriptive_retreat_avoided": bool(result.get("descriptive_retreat_avoided")),
            "base_dataset_remains_relevant": bool(result.get("base_dataset_remains_relevant")),
            "composite_component_contract_preserved": (
                self._composite_component_usage_valid(before, dataset_info)
                and self._composite_component_usage_valid(after, dataset_info)
            ),
            # 根据最终字段的科学含义审核；字段是否同时改写不能证明一致或矛盾。
            "idea_fields_coherent": bool(result.get("idea_fields_coherent")),
            "revision_package_consistent": (
                bool(semantic_writeback["revision_package_consistent"])
                and bool(result.get("revision_package_consistent", True))
            ),
            "evidence_dependencies_supported": (
                bool(semantic_writeback["evidence_dependencies_supported"])
                and bool(result.get("evidence_dependencies_supported", True))
            ),
            "deferred_findings_preserved": (
                bool(semantic_writeback["deferred_findings_preserved"])
                and bool(result.get("deferred_findings_preserved", True))
            ),
            # 纯确定性：不问验收模型。它判断的是"这次改写有没有回应质询"，而混淆型
            # 缺陷的改写在措辞层面看起来总是回应了的——这正是它系统性放行的那一类。
            "discriminating_contrast_updated": discriminating_contrast_satisfied,
            # 纯确定性：方向性主张与探针统计的符号一致性。实测教训（WFD02 behavior_pattern
            # 2026-11-01）：裁决证伪了主张的一半，改写保留未过审的另一半并靠限定词抬分，
            # 而当轮已挂的 micro_empirical_check 统计方向恰好相反——这类符号残留由代码判定。
            # 结构化承诺（directional_restatement.surviving_sign）优先；短语残留兜底。
            "directional_sign_consistent": self._directional_sign_consistent(
                after, findings, directional_restatement,
            ),
        }
        failure_code_by_check = {
            "revision_target_resolved": "target_unresolved",
            "substantive_change_landed": "no_substantive_change",
            "research_object_preserved": "research_object_lost",
            "claim_scope_valid": "claim_scope_invalid",
            "falsifiable_signal_added_or_clarified": "falsifiable_signal_missing",
            "observable_within_data_affordance": "observable_outside_data_affordance",
            "descriptive_retreat_avoided": "descriptive_retreat",
            "base_dataset_remains_relevant": "base_dataset_irrelevant",
            "composite_component_contract_preserved": "composite_component_contract_lost",
            "idea_fields_coherent": "idea_incoherent",
            "revision_package_consistent": "revision_package_mismatch",
            "evidence_dependencies_supported": "unsupported_evidence_dependency",
            "deferred_findings_preserved": "revision_package_mismatch",
            "discriminating_contrast_updated": "discriminating_contrast_unchanged",
            "directional_sign_consistent": "directional_sign_contradicts_stat",
        }
        allowed_failure_codes = set(failure_code_by_check.values())
        failure_codes = [
            str(code)
            for code in result.get("failure_codes", [])
            if str(code) in allowed_failure_codes
        ]
        for check, passed in checks.items():
            code = failure_code_by_check[check]
            if not passed and code not in failure_codes:
                failure_codes.append(code)
        failure_codes = failure_codes[:10]
        reason_text = str(result.get("reason", ""))
        return {
            "passed": all(checks.values()) and not failure_codes,
            **checks,
            "semantic_writeback_validation": semantic_writeback,
            # 结构化方向承诺随验收记录入审计（accepted 与 rejected 两条路径都可溯源）
            "directional_restatement": directional_restatement,
            # 混淆型门禁走了哪条出路：contrast_updated（换对照）/ claim_withdrawn
            # （收窄主张）/ not_required（本轮无混淆型 finding）。用于统计出路 2
            # 是否被当成新的蒙混通道。
            "discriminating_contrast_resolution": (
                "not_required" if not contrast_findings
                else "contrast_updated" if contrast_changed
                else "claim_withdrawn" if claim_withdrawn
                else "unresolved"
            ),
            "failure_codes": failure_codes,
            "reason": reason_text[:MATURATION_REASON_MAX_CHARS],
        }

    @staticmethod
    def _validate_idea_semantic_writeback(
        *,
        changed_fields: List[Dict[str, Any]],
        findings: List[Dict[str, Any]],
        evidence_records: Optional[List[Dict[str, Any]]],
        revision_package: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Deterministically enforce package scope and evidence traceability."""
        if revision_package is None:
            return {
                "revision_package_consistent": True,
                "evidence_dependencies_supported": True,
                "deferred_findings_preserved": True,
                "authorized_fields": [],
                "changed_fields": [str(item.get("field") or "") for item in changed_fields],
                "errors": [],
            }

        records_by_id = {
            str(record.get("evidence_record_id") or ""): record
            for record in (evidence_records or [])
            if isinstance(record, dict)
        }
        finding_ids = {
            str(finding.get("finding_id") or "")
            for finding in findings
            if str(finding.get("finding_id") or "")
        }
        package_items = [item for item in revision_package if isinstance(item, dict)]
        idea_updates = [
            item for item in package_items
            if item.get("resolution_route") == "idea_field_update"
        ]
        deferred = [
            item for item in package_items
            if item.get("resolution_route") in {"step_plan_open_point", "unresolved"}
        ]
        authorized_fields = {
            str(field)
            for item in idea_updates
            for field in (item.get("authorized_fields") or [])
            if str(field) in IDEA_FIELDS
        }
        changed_names = {
            str(item.get("field") or "")
            for item in changed_fields
            if str(item.get("field") or "")
        }
        package_finding_ids = {
            str(item.get("finding_id") or "")
            for item in idea_updates
            if str(item.get("finding_id") or "")
        }
        errors: List[str] = []
        routes_valid = all(item.get("resolution_route") in REVISION_ROUTES for item in package_items)
        if not routes_valid:
            errors.append("invalid_resolution_route")
        if not changed_names.issubset(authorized_fields):
            errors.append("changed_field_outside_package_scope")
        if not package_finding_ids.issubset(finding_ids):
            errors.append("package_finding_not_in_revision_input")

        evidence_supported = True
        for item in idea_updates:
            # 路由侧已肯定地判定这条 finding 不需要外部证据（`evidence_requirement` =
            # none_required），它引用的检索记录必然是 insufficient。这里再要求 grounded
            # 会把同一条记录反着判两次：route 说"授权凭 idea 自身推理修"，验收说"证据
            # 未 grounded"，该轮无论修订模型改什么都必败，重试也只会撞同一面墙。写回
            # 是否引入了未被证据支持的新外部实体，仍由验收模型的
            # evidence_dependencies_supported 把关。
            if str(item.get("evidence_requirement") or "") == EVIDENCE_REQUIREMENT_NONE:
                continue
            evidence_ids = [str(value) for value in (item.get("evidence_record_ids") or [])]
            if not evidence_ids or any(
                evidence_id not in records_by_id
                or records_by_id[evidence_id].get("status") != "grounded"
                for evidence_id in evidence_ids
            ):
                evidence_supported = False
                errors.append(
                    f"ungrounded_package_item:{item.get('revision_package_item_id', '')}"
                )

        deferred_target_fields = {
            str(item.get("target_field") or "")
            for item in deferred
            if str(item.get("target_field") or "")
        }
        deferred_preserved = not bool(
            (deferred_target_fields - authorized_fields) & changed_names
        )
        if not deferred_preserved:
            errors.append("deferred_finding_written_back")

        return {
            "revision_package_consistent": routes_valid and changed_names.issubset(authorized_fields),
            "evidence_dependencies_supported": evidence_supported,
            "deferred_findings_preserved": deferred_preserved,
            "authorized_fields": [field for field in IDEA_FIELDS if field in authorized_fields],
            "changed_fields": [field for field in IDEA_FIELDS if field in changed_names],
            "deferred_finding_ids": [
                str(item.get("finding_id") or "") for item in deferred
            ],
            "grounded_evidence_record_ids": [
                evidence_id
                for evidence_id, record in records_by_id.items()
                if record.get("status") == "grounded"
            ],
            "errors": errors,
        }

    @staticmethod
    def _idea_patch_schema(fields: List[str]) -> Dict[str, Any]:
        text = {"type": "string", "maxLength": 320}
        field_schemas: Dict[str, Dict[str, Any]] = {
            "knowledge_anchor_and_motivation": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "macro_phenomenon": text,
                    "research_motivation": text,
                },
                "required": ["macro_phenomenon", "research_motivation"],
            },
            "data_synergy_and_gap": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "single_source_limits": {
                        "type": "array",
                        "maxItems": 2,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "data_source": {"type": "string", "maxLength": 120},
                                "limitation": {"type": "string", "maxLength": 220},
                            },
                            "required": ["data_source", "limitation"],
                        },
                    },
                    "synergy_opportunity": {"type": "string", "maxLength": 320},
                },
                "required": ["single_source_limits", "synergy_opportunity"],
            },
            "research_question": {"type": "string", "maxLength": 420},
            "core_hypothesis": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "directional_statement": text,
                    "proxy_or_inference_claim": text,
                },
                "required": ["directional_statement", "proxy_or_inference_claim"],
            },
            "theoretical_mechanism": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "mechanism_chain": {"type": "string", "maxLength": 360},
                    "assumptions": {"type": "array", "maxItems": 3, "items": {"type": "string", "maxLength": 360}},
                },
                "required": ["mechanism_chain", "assumptions"],
            },
            "computability_and_falsifiability": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "observable_variable": text,
                    "mechanism_relevant_contrast": {"type": "string", "maxLength": 360},
                    "falsification_condition": {"type": "string", "maxLength": 360},
                },
                "required": [
                    "observable_variable",
                    "mechanism_relevant_contrast",
                    "falsification_condition",
                ],
            },
            "novelty_constraints_and_value": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "system_value": text,
                    "applicability_boundaries": {
                        "type": "array",
                        "maxItems": 3,
                        "items": {"type": "string", "maxLength": 180},
                    },
                },
                "required": ["system_value", "applicability_boundaries"],
            },
        }
        schema_fields = [field for field in fields if field in field_schemas]
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                field: field_schemas[field]
                for field in schema_fields
            },
            "required": schema_fields,
        }

    def _sanitize_patch(self, patch: Any, fields: List[str]) -> Dict[str, Any]:
        if not isinstance(patch, dict):
            return {}
        sanitized: Dict[str, Any] = {}
        for field in fields:
            if field not in patch or patch[field] in (None, "", [], {}):
                continue
            sanitized[field] = copy.deepcopy(patch[field])
        return sanitized

    @staticmethod
    def _revision_fields(findings: List[Dict[str, str]]) -> List[str]:
        targets = list(dict.fromkeys(
            item["target_field"]
            for item in findings
            if item.get("target_field") in IDEA_FIELDS
        ))
        if not targets:
            return []
        authorized = {field for field in targets if field in CLAIM_SEMANTIC_FIELDS}
        # research_question / core_hypothesis / theoretical_mechanism 必须方向一致：
        # 只改其中一部分会让未改的字段继续指向改写前的方向，产出符号矛盾的 idea，并被
        # 语义验收的 idea_fields_coherent 挡住。因此任一被命中就把整条链一起放开。
        if DIRECTIONAL_CLAIM_CHAIN & authorized:
            authorized |= DIRECTIONAL_CLAIM_CHAIN
        if authorized:
            authorized.add(FALSIFIABILITY_FIELD)
        # 混淆型 finding 的验收门禁要求 mechanism_relevant_contrast 真的变化，因此这个
        # 字段必须可写——哪怕 finding 的目标字段是 data_synergy_and_gap 这类非主张字段。
        # 少了这一行，门禁会变成不可满足条件：修订拿不到字段，重试每次撞同一面墙，
        # 只能以 revision_stalled 收场。
        if any(item.get("requires_discriminating_contrast") for item in findings):
            authorized.add(FALSIFIABILITY_FIELD)
        # 未声明的 observable 依赖有两条修复出路：把测量源声明进 data_synergy_and_gap
        # （target_field 本身，已由 supporting_targets 放开），或把 observable 收窄到已
        # 声明数据支撑的范围（改 computability_and_falsifiability.observable_variable）。
        # 后者要求 FALSIFIABILITY_FIELD 可写，否则"退守"这条出路拿不到字段，只剩"声明"
        # 一条路可走。
        if any(item.get("undeclared_observable_dependency") for item in findings):
            authorized.add(FALSIFIABILITY_FIELD)
        # 支撑类 finding（数据能否支撑该主张、新颖性边界等）与主张链一同放开：
        # 命中这些字段却只授权主张链，缺陷落点本身不可写，改写只能在主张里换措辞，
        # 会稳定触发 target_unresolved / no_substantive_change。
        supporting_targets = [
            field
            for field in targets
            if field not in CLAIM_SEMANTIC_FIELDS
        ][:MAX_SUPPORTING_REVISION_FIELDS]
        authorized |= set(supporting_targets)
        # 按 IDEA_FIELDS 声明顺序排列，让 data_synergy_and_gap -> research_question ->
        # core_hypothesis -> theoretical_mechanism 的推理链在 patch schema 里顺序稳定。
        return [field for field in IDEA_FIELDS if field in authorized]

    @staticmethod
    def _revision_fields_from_package(
        revision_package: Optional[List[Dict[str, Any]]],
    ) -> List[str]:
        authorized = {
            str(field)
            for item in (revision_package or [])
            if isinstance(item, dict) and item.get("resolution_route") == "idea_field_update"
            for field in (item.get("authorized_fields") or [])
            if str(field) in IDEA_FIELDS
        }
        if DIRECTIONAL_CLAIM_PAIR & authorized:
            authorized |= DIRECTIONAL_CLAIM_PAIR
        return [field for field in IDEA_FIELDS if field in authorized]

    @staticmethod
    def _has_falsifiable_signal(idea: Dict[str, Any]) -> bool:
        signal = idea.get(FALSIFIABILITY_FIELD)
        if not isinstance(signal, dict):
            return False
        return all(
            isinstance(signal.get(field), str) and bool(signal[field].strip())
            for field in (
                "observable_variable",
                "mechanism_relevant_contrast",
                "falsification_condition",
            )
        )

    @staticmethod
    def _composite_component_usage_valid(
        idea: Dict[str, Any],
        dataset_info: Dict[str, Any],
    ) -> bool:
        base_dataset = dataset_info.get("base_dataset") if isinstance(dataset_info, dict) else {}
        if not isinstance(base_dataset, dict) or base_dataset.get("type") != "composite_dataset":
            return True
        contract = base_dataset.get("combination_contract") or {}
        required_roles = {
            str(role).strip()
            for role in (contract.get("required_roles") or [])
            if str(role).strip()
        } if isinstance(contract, dict) else set()
        if not required_roles:
            required_roles = {
                str(component.get("role") or "").strip()
                for component in (base_dataset.get("dataset_components") or [])
                if isinstance(component, dict) and str(component.get("role") or "").strip()
            }
        if not required_roles:
            return True
        usage = idea.get("component_usage")
        if not isinstance(usage, list) or len(usage) != len(required_roles):
            return False
        usage_roles = {
            str(item.get("role") or "").strip()
            for item in usage
            if isinstance(item, dict) and item.get("necessity") == "required"
        }
        return usage_roles == required_roles

    @staticmethod
    def _apply_patch(idea: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
        merged = copy.deepcopy(idea)
        for field, value in patch.items():
            if field in IDEA_FIELDS:
                merged[field] = value
        return merged

    @staticmethod
    def _changed_fields(before: Dict[str, Any], after: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            {"field": field, "before": before.get(field), "after": after.get(field)}
            for field in IDEA_FIELDS
            if before.get(field) != after.get(field)
        ]

    @staticmethod
    def _applied_revisions_with_provenance(
        changed_fields: List[Dict[str, Any]],
        revision_package: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        applied: List[Dict[str, Any]] = []
        for change in changed_fields:
            field = str(change.get("field") or "")
            matching = [
                item
                for item in revision_package
                if item.get("resolution_route") == "idea_field_update"
                and field in (item.get("authorized_fields") or [])
            ]
            applied.append({
                **copy.deepcopy(change),
                "finding_ids": list(dict.fromkeys(
                    str(item.get("finding_id") or "")
                    for item in matching
                    if str(item.get("finding_id") or "")
                )),
                "revision_package_item_ids": list(dict.fromkeys(
                    str(item.get("revision_package_item_id") or "")
                    for item in matching
                    if str(item.get("revision_package_item_id") or "")
                )),
                "evidence_record_ids": list(dict.fromkeys(
                    str(evidence_id)
                    for item in matching
                    for evidence_id in (item.get("evidence_record_ids") or [])
                    if str(evidence_id)
                )),
            })
        return applied

    @staticmethod
    def _dataset_context(dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        # 与 IdeaGenerationAgent 共用同一份实现，避免两侧看到不同的数据集事实。
        return build_idea_dataset_context(dataset_info)

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

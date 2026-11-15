"""Persistent idea-native Weakness x Value Socratic tree."""
import asyncio
import json
import copy
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from datahypgen.prompts import (
    IDEA_SOCRATIC_ADJUDICATOR_SYSTEM_PROMPT,
    IDEA_SOCRATIC_BRANCH_SYSTEM_PROMPT,
    IDEA_SOCRATIC_RESPONSE_SYSTEM_PROMPT,
    SOCRATIC_ADJUDICATION_CHAR_BUDGET,
    SOCRATIC_ANSWER_CHAR_BUDGET,
    SOCRATIC_QUESTION_CHAR_BUDGET,
    build_idea_socratic_adjudication_prompt,
    build_idea_socratic_branch_prompt,
    build_idea_socratic_response_prompt,
)

# schema 的 maxLength、prompt 里明说的字符预算、以及写回时的切片必须同源，否则模型按
# prompt 的预算写、schema 按另一个数判违规，会让模型层反复触发无意义的纠错重试。
# 上限 = 预算 + 20 字符余量，容忍模型对自身输出长度的估算误差。
_QUESTION_MAX_CHARS = SOCRATIC_QUESTION_CHAR_BUDGET + 20
_ANSWER_MAX_CHARS = SOCRATIC_ANSWER_CHAR_BUDGET + 20
_ADJUDICATION_MAX_CHARS = SOCRATIC_ADJUDICATION_CHAR_BUDGET + 20


# 裁决 verdict 的语义（见 build_idea_socratic_adjudication_prompt）：只有这两个
# 声明了"存在可靠改写 idea 字段修掉的缺陷"，才允许转成修订要求。`continue` 只表示
# 该分支还有可施压的点，`evidence_needed` 表示缺少具名证据，二者都不是改写字段能
# 消除的，硬转成修订要求只会得到语义等价的空改写。两侧共用同一组 verdict：value
# 侧实际也用 needs_revision 表达可改写缺陷，不按 side 分别配对。
REVISION_ACTIONABLE_VERDICTS = frozenset({"needs_revision", "value_opportunity"})
RESOLVED_VERDICT = "resolved"
PRESSURE_VERDICT = "continue"
EVIDENCE_BLOCKED_VERDICT = "evidence_needed"
# axis_score <= 该阈值表示裁决认为该分支存在实质缺陷，与 continue/resolved 互斥。
DEFICIENT_AXIS_SCORE = 0.5

# 轴锚点表。六个轴与 selection FV 的 FV_AXES 同名同序，因此两处共用同一张表：分开维护
# 会让裁决与 FV 对同一个 basis code 给出不同分数。定义在这里而不是 idea_maturation_agent，
# 是因为该模块被 agent import，反向 import 会成环。
#
# 裁决此前让模型直接给 axis_score 数字，而 prompt 只规定了"<=0.5 与 continue/resolved
# 不相容"这条约束、从未定义各档含义。实测 102 个裁决节点里 101 个落在 0.2~0.4，全部
# <= DEFICIENT_AXIS_SCORE，于是 _reconcile_verdict 把它们一律归成 needs_revision，
# resolved 出现 0 次，skip 与 is_converged 双双永久不可达。改为选 code 后由代码映射，
# 模型不再需要自己发明分数刻度。
AXIS_BASIS_SCORE_MAP: Dict[str, Dict[str, float]] = {
    "idea_operational_coherence": {
        "absent_or_contradictory_design": 0.0,
        "major_answerability_defect": 0.25,
        "material_design_gap": 0.5,
        "answerable_with_minor_uncertainty": 0.75,
        "explicit_answerable_design": 1.0,
    },
    "mechanism_soundness": {
        "absent_or_contradictory_mechanism": 0.0,
        "major_proxy_or_logic_failure": 0.25,
        "material_mechanism_gap": 0.5,
        "sound_with_minor_uncertainty": 0.75,
        "explicit_evidence_supported_mechanism": 1.0,
    },
    "claim_defensibility": {
        "absent_or_indefensible_claim": 0.0,
        "major_scope_overreach": 0.25,
        "material_claim_support_gap": 0.5,
        "defensible_with_minor_uncertainty": 0.75,
        "explicit_well_calibrated_claim": 1.0,
    },
    "novelty_frontier": {
        "absent_or_contradicted_novelty": 0.0,
        "generic_value_only": 0.25,
        "comparator_vague_or_unsupported": 0.5,
        "specific_defensible_contrast": 0.75,
        "evidence_supported_frontier": 1.0,
    },
    "discovery_leverage": {
        "absent_or_invalid_research_question": 0.0,
        "descriptive_restatement": 0.25,
        "latent_target_not_discriminated": 0.5,
        "discriminating_hidden_target": 0.75,
        "competing_explanations_resolved": 1.0,
    },
    "combination_essentiality": {
        "contradictory_source_contract": 0.0,
        "undeclared_or_unavailable_dependency": 0.25,
        "material_source_role_or_alignment_gap": 0.5,
        "coherent_source_design": 0.75,
        "explicit_sufficiency_or_essentiality": 1.0,
    },
}


def axis_basis_codes(dimension: str) -> Tuple[str, ...]:
    """按分数升序返回该轴的 basis code；未登记的 dimension 返回空元组。"""
    ladder = AXIS_BASIS_SCORE_MAP.get(dimension)
    if not ladder:
        return ()
    return tuple(sorted(ladder, key=lambda code: ladder[code]))


# 裁决语境的档位判据。code 名按 selection FV 的绝对评估语境命名，裁决模型在对抗性质询
# 后按字面读它们，会把"质询提到过 gap"当成"gap 存活"：实测 36 个裁决节点 75% 落在 0.5
# 档、supported 辩护 28 例里 27 例仍给 0.5。判据把 0.5/0.75 的分界改成"辩护是否给出了
# 消除缺口的理由"：被正确反驳的质询不留下 gap，无法命名文本具体错误的剩余疑虑属于
# 0.75 档的 minor uncertainty。
#
# 2026-11-01 再收紧：0.75 档的"给出理由"对方向性主张（ asserted sign/ordering ）不够——
# WFD02 behavior_pattern 轮实测，裁决正确证伪了主张的火龄半句，改写却保留未过审的
# "lower intensity"半句并靠归一化限定词把 novelty 轴从 0.0 抬回 0.75（分支转 resolved），
# 而数据的实测方向恰好相反（2019 触发雨强显著更高）。0.75 档因此对方向性主张加一条：
# 理由必须点名可计算的对照（哪些观测量、按什么分组比较）；叙事重述或追加限定词不算理由。
_DIRECTIONAL_CLAIM_CLAUSE = (
    "; for a claim asserting a direction or ordering, the reason must name the observable "
    "comparison that decides it (which measured variable, compared between which groups) — "
    "restating the narrative or adding qualifiers is not a reason"
)
ADJUDICATION_BASIS_HINTS: Dict[str, Dict[str, str]] = {
    "idea_operational_coherence": {
        "absent_or_contradictory_design": "the operational design is missing or self-contradictory in the idea text",
        "major_answerability_defect": "the exchange exposed a defect that breaks answerability, and the defender conceded or could not rebut it",
        "material_design_gap": "a specific design gap in the idea text survived the exchange; the defender gave no reason that removes it",
        "answerable_with_minor_uncertainty": "the defender gave a reason the design is answerable; remaining doubt does not name a concrete error in the idea text" + _DIRECTIONAL_CLAIM_CLAUSE,
        "explicit_answerable_design": "the idea text states the answerable design explicitly and available evidence supports it",
    },
    "mechanism_soundness": {
        "absent_or_contradictory_mechanism": "the mechanism is missing or self-contradictory in the idea text",
        "major_proxy_or_logic_failure": "the exchange exposed a proxy or logic failure that breaks the mechanism, and the defender conceded or could not rebut it",
        "material_mechanism_gap": "a specific mechanism gap in the idea text survived the exchange; the defender gave no reason that removes it",
        "sound_with_minor_uncertainty": "the defender gave a reason the mechanism is sound; remaining doubt does not name a concrete error in the idea text" + _DIRECTIONAL_CLAIM_CLAUSE,
        "explicit_evidence_supported_mechanism": "the idea text states the mechanism explicitly and available evidence supports it",
    },
    "claim_defensibility": {
        "absent_or_indefensible_claim": "the claim is missing or indefensible in the idea text",
        "major_scope_overreach": "the exchange exposed a scope overreach that breaks the claim, and the defender conceded or could not rebut it",
        "material_claim_support_gap": "a specific support gap for the claim in the idea text survived the exchange; the defender gave no reason that removes it",
        "defensible_with_minor_uncertainty": "the defender gave a reason the claim is defensible; remaining doubt does not name a concrete error in the idea text" + _DIRECTIONAL_CLAIM_CLAUSE,
        "explicit_well_calibrated_claim": "the idea text states the claim explicitly with well-calibrated scope",
    },
    "novelty_frontier": {
        "absent_or_contradicted_novelty": "novelty is missing or contradicted in the idea text",
        "generic_value_only": "the exchange exposed only generic value claims, and the defender conceded or could not rebut it",
        "comparator_vague_or_unsupported": "the idea text names a vague or unsupported comparator that survived the exchange; the defender gave no reason that removes it",
        "specific_defensible_contrast": "the defender gave a reason the contrast is specific and defensible; remaining doubt does not name a concrete error in the idea text" + _DIRECTIONAL_CLAIM_CLAUSE,
        "evidence_supported_frontier": "the idea text states the novelty frontier explicitly and available evidence supports it",
    },
    "discovery_leverage": {
        "absent_or_invalid_research_question": "the research question is missing or invalid in the idea text",
        "descriptive_restatement": "the exchange exposed that the question is a descriptive restatement, and the defender conceded or could not rebut it",
        "latent_target_not_discriminated": "the idea text does not discriminate the latent target, and the defender gave no reason that removes the gap",
        "discriminating_hidden_target": "the defender gave a reason the question discriminates the hidden target; remaining doubt does not name a concrete error in the idea text" + _DIRECTIONAL_CLAIM_CLAUSE,
        "competing_explanations_resolved": "the idea text explicitly resolves competing explanations",
    },
    "combination_essentiality": {
        "contradictory_source_contract": "the source contract is self-contradictory in the idea text",
        "undeclared_or_unavailable_dependency": "the exchange exposed an undeclared or unavailable dependency, and the defender conceded or could not rebut it",
        "material_source_role_or_alignment_gap": "a specific role or alignment gap for a source in the idea text survived the exchange; the defender gave no reason that removes it",
        "coherent_source_design": "the defender gave a reason the source design is coherent; remaining doubt does not name a concrete error in the idea text" + _DIRECTIONAL_CLAIM_CLAUSE,
        "explicit_sufficiency_or_essentiality": "the idea text states sufficiency or essentiality explicitly and available evidence supports it",
    },
}


def axis_basis_menu(dimension: str) -> str:
    """裁决 prompt 用的 code 清单，每个 code 带一句裁决语境的档位判据。

    code 名按 selection FV 的绝对评估语境命名，裸名单会让裁决模型按字面把"质询提到过
    gap"读成"gap 存活"。判据表约 500 词，裁决每轮 3~6 次调用，相比 FV 判据的 1000 字符
    整段注入仍然更省。
    """
    codes = axis_basis_codes(dimension)
    if not codes:
        return ""
    hints = ADJUDICATION_BASIS_HINTS.get(dimension) or {}
    lines = "\n".join(
        f"- `{code}` — {hints[code]}" if code in hints else f"- `{code}`"
        for code in codes
    )
    return (
        f"Ordered ladder for `{dimension}` (most severe first):\n{lines}\n"
        "Pick the code by this test: does the exchange expose a concrete error in the "
        "current idea text that the defender failed to remove? The lower three codes "
        "require such an error; the top two mean the axis held."
    )
# Balancer 单轮汇总出的修订 finding 上限。放开到六条不会显著扩大改写范围（主张链本来
# 就整条放开，六条只多授权 novelty_constraints_and_value 一个字段），但六条互相冲突的
# 要求塞进一次改写容易彼此抵消，因此保留三条并改为按 axis_score 选最弱的。
BALANCER_MAX_FINDINGS = 3

# 第 ≥2 轮每轮放行的分支名额上限，与 BALANCER_MAX_FINDINGS 同值：单轮最多只用得上三条
# finding，多长的分支不可能改变本轮改写范围，只会把预算烧在拿不到名额的字段上。
#
# 依据（1810 个 idea / 5566 轮的实测）：
# - 六个分支里三个（mechanism_soundness / claim_defensibility / discovery_leverage）的
#   休眠率只有 7.3%，第 ≥2 轮平均仍生长 4.64 个分支，即每轮 23 次模型调用；
# - 第 ≥2 轮的 FV 环比增量均值只有 +0.0108，43.6% 为 0、24.2% 为负；
# - 生长 3 个分支与生长 6 个分支的 FV 增量差异（+0.0040 vs +0.0139）落在噪声内，而
#   生长 4/5/6 个分支相对 3 个只多约 +0.01（量程 0.46~0.48，约 2%）。
# 第 1 轮不受此限：它是唯一的覆盖轮，实测接受率 71.7%，也是唯一有实质 FV 增益的轮次。
BRANCH_BUDGET_PER_ROUND = 3


# 每个分支只质询一个 target_field，但判断连贯性需要主张链作锚点。这里按"目标字段 +
# 该分支指令实际引用的字段 + 方向性主张锚点"给出白名单，其余字段不进该分支的 prompt。
#
# research_question 与 core_hypothesis 对所有分支保留：裁决要判 verdict 与 axis_score
# 是否与"问题和假设是否自相矛盾"一致，缺了它们无法判定。
# knowledge_anchor_and_motivation 对所有分支都不保留：它不是任何分支的 target_field，
# 也没有任何分支指令或三个 prompt 正文引用它（实测 3678 个真实节点的 target_field
# 从未取到该字段）。
BRANCH_IDEA_FIELDS = {
    "idea_operational_coherence": (
        "data_synergy_and_gap", "research_question", "core_hypothesis",
        "computability_and_falsifiability",
    ),
    "mechanism_soundness": (
        "theoretical_mechanism", "research_question", "core_hypothesis",
        "computability_and_falsifiability",
    ),
    "claim_defensibility": (
        "core_hypothesis", "research_question", "computability_and_falsifiability",
    ),
    "novelty_frontier": (
        "novelty_constraints_and_value", "research_question", "core_hypothesis",
    ),
    "discovery_leverage": (
        "research_question", "core_hypothesis", "theoretical_mechanism",
    ),
    "combination_essentiality": (
        "data_synergy_and_gap", "research_question", "core_hypothesis",
        "theoretical_mechanism", "computability_and_falsifiability",
    ),
}
# idea_id / idea_type 不进树 prompt：selection FV 的 _selection_idea 专门剥掉 idea_id
# 以保持来源盲化，树没有理由比它更宽松；且 idea_id 本身带 idea_type 前缀，等于重复告知。
BRANCH_IDEA_IDENTITY_FIELDS = ("idea_type",)


def branch_idea_view(idea: Dict[str, Any], dimension: str) -> Dict[str, Any]:
    """按分支裁剪 idea，未登记的 dimension 回退为全量语义字段。"""
    allowed = BRANCH_IDEA_FIELDS.get(dimension)
    if allowed is None:
        return {
            key: value for key, value in idea.items()
            if key != "idea_id"
        }
    keep = (*BRANCH_IDEA_IDENTITY_FIELDS, *allowed)
    return {key: idea[key] for key in keep if key in idea}


BRANCHES = (
    ("weakness", "idea_operational_coherence", "data_synergy_and_gap", "Test whether the declared data can produce every variable the question needs, the observable first. In order: (1) name the observable variable and the predictor variables from research_question, core_hypothesis and computability_and_falsifiability; (2) identify which declared source or role in data_synergy_and_gap measures the observable variable itself - if the observable is an independently measured quantity (outcome, event, label, gauge/instrument reading) that the base or predictor data cannot construct, and no declared source carries that measuring role, this undeclared-observable dependency is the primary defect; a routine derivative of the base data such as slope, ruggedness or zonal statistics is constructible and is not this gap; the legitimate repair, in order of preference, is: first construct a proxy-mechanism observable from the base and declared data when one lands inside their temporal, spatial and label affordances; else declare the measuring source as an explicit data role when an independent measurement is genuinely required; else retreat the observable to a neighbouring question the declared data can decide — never defer the observable's own data to retrieval; (3) only then test whether the remaining declared features, synergy signal and falsification condition make the rest of the question answerable. Treat profiler-emitted dataset features as existing dataset traits."),
    ("weakness", "mechanism_soundness", "theoretical_mechanism", "Trace every link from observable signals to the claimed process. Expose hidden assumptions, proxy leaps, or a mechanism that does not entail the directional hypothesis."),
    ("weakness", "claim_defensibility", "core_hypothesis", "Test whether the hypothesis claims causation, attribution, or a hidden state beyond what its proposed variables and falsification condition can defend."),
    ("value", "novelty_frontier", "novelty_constraints_and_value", "Locate the precise contrast with existing work that could make the idea non-trivially new, then test whether that contrast is actually stated and defensible."),
    ("value", "discovery_leverage", "research_question", "Identify the hidden state, mechanism, counterfactual, discrepancy, or phenomenon that the question could reveal and that a descriptive restatement would miss."),
    ("value", "combination_essentiality", "data_synergy_and_gap", "First identify whether the idea is genuinely single-source or multi-source. Treat multiple views, bands, labels, or time slices of the same dataset as single-source; disguising this as multi-source is a source-role defect. For a single-source idea, test whether that source is sufficient for the core question and whether the claim has any hidden dependency on an additional source; judge only multi-source necessity here and leave whether the observable variable itself is measurable to idea_operational_coherence; do not demand an auxiliary source merely to create a combination. For a declared external auxiliary source, check that data_synergy_and_gap explains the spatial/temporal/alignment relationship and that fixed evidence candidate_datasets contains a corresponding available dataset. For a multi-source idea, run the semantic counterfactual: whether the base signal alone answers materially the same question, and require the added sources to change the observable mechanism, hidden state, or falsification signal rather than merely add context."),
)


@dataclass
class IdeaSocraticNode:
    node_id: str
    side: str
    dimension: str
    target_field: str
    depth: int
    parent_node_id: str = ""
    question: str = ""
    failure_hypothesis: str = ""
    answer: str = ""
    answer_status: str = ""
    adjudication: str = ""
    verdict: str = ""
    # 模型原始返回的 verdict，_reconcile_verdict 归正之前的值。只留档、不参与任何判据：
    # 实测需要还原"模型本来判了 resolved/continue 却被 axis_score 归正成 needs_revision"
    # 这类翻转，分开"模型自判要修"与"代码归正造活"两种失败。
    raw_verdict: str = ""
    axis_score: float = 0.0
    # 留档模型实际选的 code，而不只是映射后的分数：重跑对比时要能分辨"模型给了高档但
    # verdict 仍偏保守"与"模型本来就只给低档"，只看分数无法区分这两种失败。
    axis_basis_code: str = ""
    evidence_need: str = ""
    next_focus: str = ""
    # 裁决判定该缺陷源于"竞争解释产生相同观测"，只有判别性对照能消除它。
    requires_discriminating_contrast: bool = False
    required_contrast: str = ""
    # 裁决判定该缺陷是"被观测量本体缺已声明测量源"（未声明的 observable 依赖）。
    # 只由 idea_operational_coherence 分支置真：它是 idea 内部的声明自洽缺陷，改写靠
    # 声明测量源或收窄 observable 即可消除，不需外部检索。下游据此短路证据通道路由。
    undeclared_observable_dependency: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class PersistentIdeaSocraticTree:
    tree_id: str = field(default_factory=lambda: f"idea_tree_{uuid.uuid4().hex[:12]}")
    nodes: Dict[str, IdeaSocraticNode] = field(default_factory=dict)
    active_leaf_ids: Dict[str, str] = field(default_factory=dict)
    branch_depths: Dict[str, int] = field(default_factory=dict)
    stage_diffs: List[List[Dict[str, Any]]] = field(default_factory=list)
    # 每个分支逐轮的 axis_score，用于判定分支是否已经问不动了。只存分数不存节点，
    # 跨轮不重复序列化叶子全文。
    branch_score_history: Dict[str, List[float]] = field(default_factory=dict)


class SocraticIdeaDualTree:
    """Persistent idea tree with one adjudication per two-layer branch exchange."""

    def __init__(
        self,
        call_model: Callable[..., Awaitable[Dict[str, Any]]],
        max_tokens: int,
        *,
        branch_stall_rounds: int = 2,
        branch_stall_eps: float = 0.05,
        branch_budget_per_round: int = BRANCH_BUDGET_PER_ROUND,
    ) -> None:
        self._call_model = call_model
        self.context_manager = None
        self._max_tokens = max_tokens
        # 分支休眠兜底。即使裁决恢复了区分度，也不能假设模型一定会给出 resolved；实测旧
        # 实现里 resolved 出现 0 次，六个分支每轮全额生长直到 max_tree_depth。这里用不依赖
        # verdict 语义的确定性判据止损：连续若干轮 axis_score 完全不动的分支停止生长。
        self._branch_stall_rounds = max(1, int(branch_stall_rounds))
        self._branch_stall_eps = float(branch_stall_eps)
        # 第 ≥2 轮的分支名额。休眠判据只能停掉"判断不再变化"的分支，停不掉"判断每轮都在
        # 同一档反复横跳"的分支，所以还需要一层与判断内容无关的预算闸。
        self._branch_budget = max(1, int(branch_budget_per_round))

    def _is_branch_stalled(
        self,
        tree: PersistentIdeaSocraticTree,
        branch: Tuple[str, str, str, str],
        changed_fields: set,
    ) -> bool:
        """分支是否已问不动。

        判据取窗口内的极差而不是"相对起点的改善"：分数下降说明质询仍在改变判断，
        那不是停滞。axis_score 现在是 {0,.25,.5,.75,1} 的离散档，任何真实档位变化都
        >= 0.25，远大于默认 eps，所以这条实际只在连续同档时触发。

        target_field 刚被改写过的分支一律不休眠：idea 变了就必须重新质询，否则休眠会
        锁住一个已经不存在的旧判断。
        """
        _, dimension, target_field, _ = branch
        if target_field in changed_fields:
            return False
        history = tree.branch_score_history.get(dimension, [])
        if len(history) < self._branch_stall_rounds:
            return False
        window = history[-self._branch_stall_rounds:]
        return (max(window) - min(window)) < self._branch_stall_eps

    def _apply_branch_budget(
        self,
        candidates: List[Tuple[Tuple[str, str, str, str], Optional[IdeaSocraticNode]]],
        stage: int,
    ) -> Tuple[
        List[Tuple[Tuple[str, str, str, str], Optional[IdeaSocraticNode]]],
        List[Dict[str, str]],
    ]:
        """按本轮名额裁剪候选分支，返回 (放行的分支, 被裁掉的分支及原因)。

        只在候选数超过名额时才动手。名额不紧张时保留同字段的第二条分支，因为 `_balance`
        虽然只取一个 lead，却会把同字段所有节点的 `requires_discriminating_contrast` 合并
        进 finding；无条件去重会白白丢掉 value 侧的这个标记。名额紧张时才让它让位——那时
        它占的是一个本可以给其它字段的名额，而它自己的 question 与 adjudication 反正进不了
        finding。

        两道闸，顺序不能换：

        1. target_field 去重。BRANCHES 里 idea_operational_coherence 与
           combination_essentiality 共用 data_synergy_and_gap，而 `_balance` 按字段合并、
           `lead` 只取一个节点（weakness 优先），第二个分支的 question 与 adjudication
           本来就进不了 finding，只有两个布尔标记能留下。这里保留 weakness 侧，与
           `_balance` 的 lead 规则同口径。
        2. 名额轮转。去重后仍多于名额时按 stage 旋转取前 N 个。用轮转而不是按
           axis_score 排序：实测轴分数高度饱和（0.5 档占 55%~89%），排序在平票时退回
           插入顺序、结构性地偏向 weakness 侧——那正是 `_balance` 当初引入 stage 轮转
           要消除的偏置。

        第 1 轮（stage <= 1）不裁：它是唯一的覆盖轮，也是唯一有实质 FV 增益的轮次。

        被裁掉的分支与休眠分支分开记账：预算是成本决策，不是"这个分支问透了"的语义
        判断，因此它不参与 `is_converged` 的收口判据，也不写进 branch_score_history。
        """
        if stage <= 1 or len(candidates) <= self._branch_budget:
            return list(candidates), []

        by_field: Dict[str, Tuple[Tuple[str, str, str, str], Optional[IdeaSocraticNode]]] = {}
        pruned: List[Dict[str, str]] = []
        for branch, parent in candidates:
            field = branch[2]
            kept = by_field.get(field)
            if kept is None:
                by_field[field] = (branch, parent)
            elif branch[0] == "weakness" and kept[0][0] != "weakness":
                pruned.append({"dimension": kept[0][1], "reason": "duplicate_target_field"})
                by_field[field] = (branch, parent)
            else:
                pruned.append({"dimension": branch[1], "reason": "duplicate_target_field"})

        unique = list(by_field.values())
        if len(unique) <= self._branch_budget:
            return unique, pruned
        offset = stage % len(unique)
        rotated = unique[offset:] + unique[:offset]
        pruned.extend(
            {"dimension": branch[1], "reason": "round_budget"}
            for branch, _ in rotated[self._branch_budget:]
        )
        return rotated[: self._branch_budget], pruned

    async def grow_depth_pair(
        self,
        *,
        tree: PersistentIdeaSocraticTree,
        idea: Dict[str, Any],
        evidence_context: Dict[str, Any],
        stage: int,
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any], List[Dict[str, Any]]]:
        changed_fields = {item.get("field") for item in (tree.stage_diffs[-1] if tree.stage_diffs else [])}
        dormant_branches: List[Dict[str, str]] = []
        candidates = []
        for branch in BRANCHES:
            parent_id = tree.active_leaf_ids.get(branch[1])
            parent = tree.nodes.get(parent_id) if parent_id else None
            if parent is not None and parent.verdict == RESOLVED_VERDICT and parent.target_field not in changed_fields:
                dormant_branches.append({"dimension": branch[1], "reason": "resolved"})
                continue
            if self._is_branch_stalled(tree, branch, changed_fields):
                dormant_branches.append({"dimension": branch[1], "reason": "score_stalled"})
                continue
            candidates.append((branch, parent))

        selected, budget_pruned_branches = self._apply_branch_budget(candidates, stage)
        first_jobs = [
            (
                branch,
                parent,
                self._run_question_answer(
                    idea,
                    evidence_context,
                    branch,
                    stage * 2 - 1,
                    pair_layer=1,
                    parent=parent,
                ),
            )
            for branch, parent in selected
        ]
        first_layer = await asyncio.gather(*[job for _, _, job in first_jobs])

        second_jobs = [
            (
                branch,
                parent,
                first_node,
                self._run_question_answer(
                    idea,
                    evidence_context,
                    branch,
                    stage * 2,
                    pair_layer=2,
                    parent=first_node,
                ),
            )
            for (branch, parent, _), first_node in zip(first_jobs, first_layer)
            if first_node is not None
        ]
        second_layer = await asyncio.gather(*[job for _, _, _, job in second_jobs])

        adjudication_jobs = [
            (
                branch,
                parent,
                first_node,
                second_node,
                self._adjudicate_pair(
                    idea,
                    evidence_context,
                    branch,
                    parent,
                    first_node,
                    second_node,
                ),
            )
            for (branch, parent, first_node, _), second_node in zip(second_jobs, second_layer)
            if second_node is not None
        ]
        adjudicated_second_layer = await asyncio.gather(
            *[job for _, _, _, _, job in adjudication_jobs]
        )

        completed_pairs = [
            (branch, first_node, judged_second)
            for (branch, _, first_node, _, _), judged_second in zip(
                adjudication_jobs,
                adjudicated_second_layer,
            )
            if judged_second is not None
        ]
        first_nodes = [first_node for _, first_node, _ in completed_pairs]
        current_pair_leaves = [second_node for _, _, second_node in completed_pairs]
        for branch, first_node, second_node in completed_pairs:
            tree.nodes[first_node.node_id] = first_node
            tree.nodes[second_node.node_id] = second_node
            tree.active_leaf_ids[branch[1]] = second_node.node_id
            tree.branch_depths[branch[1]] = second_node.depth
            tree.branch_score_history.setdefault(branch[1], []).append(second_node.axis_score)

        new_nodes = [*first_nodes, *current_pair_leaves]
        current_leaves = self._current_leaves(tree)
        findings, balancing = self._balance(current_leaves, stage=stage)
        # active_leaf_ids 与 balancing 一样只存节点引用：两层问答已经在它们被创建的那一轮
        # 写入 new_nodes，pair 裁决只落在第二层叶子，跨轮不再重复序列化叶子全文。
        trace = {
            "tree_id": tree.tree_id,
            "growth_stage": stage,
            "tree_depth": max(tree.branch_depths.values(), default=0),
            "new_nodes": [node.to_dict() for node in new_nodes],
            "active_leaf_ids": [node.node_id for node in current_leaves],
            "balancing": balancing,
            # 本轮未生长的分支及原因，用于把"树问透了"和"分支被止损"区分开。
            "dormant_branches": dormant_branches,
            # 本轮被预算裁掉的分支。与 dormant 分开记：预算是成本决策，不是"该分支问透了"，
            # 重跑统计时不能把它算进休眠率，否则会把成本削减误读成收敛提前。
            "budget_pruned_branches": budget_pruned_branches,
        }
        return findings, trace, [node.to_dict() for node in current_leaves]

    def _stage_context(self, evidence, idea, branch, stage):
        if self.context_manager is None:
            return evidence
        return self.context_manager.evidence_view(evidence,
            {"idea": idea, "dimension": branch[1], "target_field": branch[2], "instruction": branch[3]},
            stage=stage)

    async def _context_call(self, evidence, **kwargs):
        if self.context_manager is None:
            return await self._call_model(**kwargs)
        return await self.context_manager.decide_with_sources(self._call_model,
            cards=copy.deepcopy(evidence["selected_evidence"]),
            history_ids=[row["episode_id"] for row in evidence["working_context"].get("history_index", [])], **kwargs)

    async def _run_question_answer(
        self,
        idea: Dict[str, Any],
        evidence_context: Dict[str, Any],
        branch: Tuple[str, str, str, str],
        depth: int,
        pair_layer: int,
        parent: Optional[IdeaSocraticNode] = None,
    ) -> Optional[IdeaSocraticNode]:
        side, dimension, target_field, instruction = branch
        parent_summary = self._parent_summary(parent)
        evidence_context = self._stage_context(evidence_context, idea, branch, "question")
        # 紧凑分隔符 + 按分支裁剪：同一深度对的五次调用共享这段大前缀。
        common = {
            "idea_json": json.dumps(
                branch_idea_view(idea, dimension),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "evidence_json": json.dumps(
                evidence_context, ensure_ascii=False, separators=(",", ":")
            ),
            "dimension": dimension,
            "target_field": target_field,
            "branch_instruction": instruction,
            "depth": depth,
            "pair_layer": pair_layer,
            "parent_summary": parent_summary,
        }
        try:
            question_result = await self._context_call(evidence_context,
                prompt=build_idea_socratic_branch_prompt(**common),
                system_prompt=IDEA_SOCRATIC_BRANCH_SYSTEM_PROMPT,
                schema={"type": "object", "properties": {"question": {"type": "string", "maxLength": _QUESTION_MAX_CHARS}, "failure_hypothesis": {"type": "string", "maxLength": _QUESTION_MAX_CHARS}}, "required": ["question", "failure_hypothesis"], "additionalProperties": False},
                temperature=0.1, max_tokens=self._max_tokens,
            )
            answer_result = await self._context_call(evidence_context,
                prompt=build_idea_socratic_response_prompt(
                    **common, question=str(question_result.get("question", "")), failure_hypothesis=str(question_result.get("failure_hypothesis", ""))
                ),
                system_prompt=IDEA_SOCRATIC_RESPONSE_SYSTEM_PROMPT,
                schema={"type": "object", "properties": {"answer": {"type": "string", "maxLength": _ANSWER_MAX_CHARS}, "answer_status": {"type": "string", "enum": ["supported", "partial", "unsupported"]}}, "required": ["answer", "answer_status"], "additionalProperties": False},
                temperature=0.1, max_tokens=self._max_tokens,
            )
        except Exception:
            return None
        answer_status = str(answer_result.get("answer_status", ""))
        return IdeaSocraticNode(
            node_id=f"{dimension}_{uuid.uuid4().hex[:10]}", side=side, dimension=dimension,
            target_field=target_field, depth=depth, parent_node_id=parent.node_id if parent else "",
            question=str(question_result.get("question", ""))[:_QUESTION_MAX_CHARS],
            failure_hypothesis=str(question_result.get("failure_hypothesis", ""))[:_QUESTION_MAX_CHARS],
            answer=str(answer_result.get("answer", ""))[:_ANSWER_MAX_CHARS],
            answer_status=answer_status,
        )

    async def _adjudicate_pair(
        self,
        idea: Dict[str, Any],
        evidence_context: Dict[str, Any],
        branch: Tuple[str, str, str, str],
        parent: Optional[IdeaSocraticNode],
        first_node: IdeaSocraticNode,
        second_node: IdeaSocraticNode,
    ) -> Optional[IdeaSocraticNode]:
        side, dimension, target_field, instruction = branch
        evidence_context = self._stage_context(evidence_context, idea, branch, "adjudication")
        try:
            judge_result = await self._context_call(evidence_context,
                prompt=build_idea_socratic_adjudication_prompt(
                    idea_json=json.dumps(
                        branch_idea_view(idea, dimension),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    evidence_json=json.dumps(
                        evidence_context, ensure_ascii=False, separators=(",", ":")
                    ),
                    dimension=dimension,
                    target_field=target_field,
                    branch_instruction=instruction,
                    depth=second_node.depth,
                    parent_summary=self._parent_summary(parent),
                    first_question=first_node.question,
                    first_failure_hypothesis=first_node.failure_hypothesis,
                    first_answer=first_node.answer,
                    first_answer_status=first_node.answer_status,
                    second_question=second_node.question,
                    second_failure_hypothesis=second_node.failure_hypothesis,
                    second_answer=second_node.answer,
                    second_answer_status=second_node.answer_status,
                    side=side,
                    axis_basis_menu=axis_basis_menu(dimension),
                ),
                system_prompt=IDEA_SOCRATIC_ADJUDICATOR_SYSTEM_PROMPT,
                schema={
                    "type": "object",
                    "properties": {
                        "verdict": {"type": "string", "enum": ["resolved", "continue", "needs_revision", "value_opportunity", "evidence_needed"]},
                        "adjudication": {"type": "string", "maxLength": _ADJUDICATION_MAX_CHARS},
                        "axis_basis_code": {"type": "string", "enum": list(axis_basis_codes(dimension))},
                        "evidence_need": {"type": "string", "maxLength": 240},
                        "next_focus": {"type": "string", "maxLength": 240},
                        # 混淆型缺陷标记：竞争解释会产生与本 idea 机制相同的观测。
                        # 这类缺陷改主张措辞消不掉，只有能分开两种解释的对照才行。
                        "requires_discriminating_contrast": {"type": "boolean"},
                        "required_contrast": {"type": "string", "maxLength": 240},
                        # 未声明的 observable 依赖标记：被观测量本体缺已声明测量源。
                        # 只对 idea_operational_coherence 分支有意义，其它分支填 false 即弃。
                        "undeclared_observable_dependency": {"type": "boolean"},
                    },
                    "required": [
                        "verdict", "adjudication", "axis_basis_code",
                        "evidence_need",
                        "next_focus", "requires_discriminating_contrast",
                        "required_contrast",
                        "undeclared_observable_dependency",
                    ],
                    "additionalProperties": False,
                },
                temperature=0.0,
                max_tokens=self._max_tokens,
            )
        except Exception:
            return None

        # basis code 缺失或不属于该轴时整次裁决作废，不静默兜底成某个默认分。落回默认分
        # 会让一次无效裁决伪装成"该轴得了 X 分"，既污染 _balance 的选取，也可能凭空满足
        # 休眠或收敛判据。与 selection FV 对未知 code 判 valid=false 是同一口径。
        ladder = AXIS_BASIS_SCORE_MAP.get(dimension, {})
        basis_code = str(judge_result.get("axis_basis_code", "") or "")
        if basis_code not in ladder:
            return None
        axis_score = float(ladder[basis_code])

        raw_verdict = str(judge_result.get("verdict", "") or PRESSURE_VERDICT)
        verdict = self._reconcile_verdict(
            verdict=raw_verdict,
            axis_score=axis_score,
            answer_status=self._pair_answer_status(first_node, second_node),
        )
        second_node.adjudication = str(judge_result.get("adjudication", ""))[:_ADJUDICATION_MAX_CHARS]
        second_node.verdict = verdict
        second_node.raw_verdict = raw_verdict
        second_node.axis_score = axis_score
        second_node.axis_basis_code = basis_code
        second_node.evidence_need = str(judge_result.get("evidence_need", ""))[:240]
        second_node.next_focus = str(judge_result.get("next_focus", ""))[:240]
        # 只有可改写缺陷才谈得上"该用什么对照修"；continue/resolved/evidence_needed
        # 不进修订，带上标记只会在 Balancer 里造成噪声。
        if verdict in REVISION_ACTIONABLE_VERDICTS:
            second_node.requires_discriminating_contrast = bool(
                judge_result.get("requires_discriminating_contrast")
            )
            second_node.required_contrast = str(
                judge_result.get("required_contrast", "")
            )[:240]
            # 未声明的 observable 依赖只认 idea_operational_coherence 分支的判定：
            # 该缺陷的修复动作是改 data_synergy_and_gap（声明源）或 observable（收窄），
            # 都落在这个分支的 target_field / 授权链上。其它分支即便置真也弃用——
            # 它们的 target_field 不同，短路路由会把授权指到错误字段。
            if dimension == "idea_operational_coherence":
                second_node.undeclared_observable_dependency = bool(
                    judge_result.get("undeclared_observable_dependency")
                )
        return second_node

    @staticmethod
    def _pair_answer_status(
        first_node: IdeaSocraticNode,
        second_node: IdeaSocraticNode,
    ) -> str:
        statuses = {first_node.answer_status, second_node.answer_status}
        if "unsupported" in statuses:
            return "unsupported"
        if "partial" in statuses:
            return "partial"
        return "supported"

    @staticmethod
    def _parent_summary(parent: Optional[IdeaSocraticNode]) -> str:
        if parent is None:
            return "ROOT: start with the branch-specific first pressure test."
        return json.dumps({"question": parent.question, "failure_hypothesis": parent.failure_hypothesis, "answer": parent.answer, "answer_status": parent.answer_status, "adjudication": parent.adjudication, "verdict": parent.verdict, "next_focus": parent.next_focus}, ensure_ascii=False)

    @staticmethod
    def _current_leaves(tree: PersistentIdeaSocraticTree) -> List[IdeaSocraticNode]:
        return [tree.nodes[node_id] for node_id in tree.active_leaf_ids.values() if node_id in tree.nodes]

    @staticmethod
    def _reconcile_verdict(*, verdict: str, axis_score: float, answer_status: str) -> str:
        """把与自身评分/辩护结果矛盾的 verdict 归正到最保守的可行档。

        裁决常把"机制不成立""存在语义缺口"写进 adjudication，却仍给 continue，
        使这些缺陷永远进不了修订。低分或辩护 unsupported 都表示该分支未被守住，
        此时 continue/resolved 不成立；已声明缺证据的保持 evidence_needed。

        反方向的矛盾（说不出缺陷却要求修订）曾用一个 `surviving_defect` 字段拦截，
        实测两次运行分别为 0/87 与 0/104 触发，即裁决确实指得出具体缺陷，该字段纯属
        空转，已连同 prompt 指令一并移除。
        """
        if verdict not in {
            RESOLVED_VERDICT, PRESSURE_VERDICT, EVIDENCE_BLOCKED_VERDICT, *REVISION_ACTIONABLE_VERDICTS
        }:
            verdict = PRESSURE_VERDICT
        if verdict in REVISION_ACTIONABLE_VERDICTS or verdict == EVIDENCE_BLOCKED_VERDICT:
            return verdict
        if axis_score <= DEFICIENT_AXIS_SCORE or answer_status == "unsupported":
            # 两侧统一归到 needs_revision：value 侧实际也用它表达可改写缺陷。
            return "needs_revision"
        return verdict

    @staticmethod
    def is_converged(
        active_leaves: List[Dict[str, Any]],
        dormant_dimensions: Optional[set] = None,
    ) -> bool:
        """两侧都问透才算收敛：每个活跃分支的叶子都是 resolved 或已休眠。

        休眠分支必须与 resolved 同等算作已收口。它这一轮起就不再生长，其叶子会永远停在
        休眠前那次裁决的 verdict 上；若仍要求它变成 resolved，加了休眠反而让收敛判据比
        原来更不可能成立，树只能继续跑到 max_tree_depth。

        空叶子集合不算收敛（树还没长出来），交由 max_tree_depth 收口。
        """
        if not active_leaves:
            return False
        dormant = dormant_dimensions or set()
        return all(
            leaf.get("verdict") == RESOLVED_VERDICT or leaf.get("dimension") in dormant
            for leaf in active_leaves
        )

    @staticmethod
    def _is_revision_actionable(node: IdeaSocraticNode) -> bool:
        """只有声明了可改写缺陷的 verdict 才转成修订要求。

        `continue` 只表示分支还有可施压的点，`evidence_needed` 表示缺具名证据，
        `resolved` 表示已问透；三者都不是改写 idea 字段能消除的。
        """
        return node.verdict in REVISION_ACTIONABLE_VERDICTS

    @staticmethod
    def _balance(
        leaves: List[IdeaSocraticNode],
        *,
        stage: int = 0,
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
        """把六个已裁决叶子汇总成最多 BALANCER_MAX_FINDINGS 条修订 finding。

        选取按 `axis_score` 升序——裁决认为最弱的字段先修。此前实现是对
        `by_field.items()` 直接 `[:3]`，即按字典插入顺序截断，而 BRANCHES 把三个
        weakness 分支排在前面，于是 `novelty_constraints_and_value` 与
        `research_question` 结构性地永远落选：实测 19 轮里这两个字段入选 0 次，value
        侧三个分支的 114 个节点、342 次模型调用产出全部被这一行丢掉。这与设计意图
        （"汇总六个已裁决叶子"）不符，也让修订打不到真正的短板。

        `stage` 用于平票轮转。实测 axis_score 高度聚集（单轮内常见五个叶子同为 0.3），
        纯稳定排序在平票时仍退回插入顺序、继续偏向 weakness 侧；按 stage 轮转起点可以
        让同分字段在不同轮次都有机会，且保持确定性（同一 stage 结果可复现）。
        """
        by_field: Dict[str, List[IdeaSocraticNode]] = {}
        for node in leaves:
            if SocraticIdeaDualTree._is_revision_actionable(node):
                by_field.setdefault(node.target_field, []).append(node)
        if not by_field:
            return [], []

        ordered_fields = list(by_field)
        # 平票轮转：按 stage 旋转字段顺序，再做稳定排序。
        if ordered_fields:
            offset = stage % len(ordered_fields)
            ordered_fields = ordered_fields[offset:] + ordered_fields[:offset]
        ordered_fields.sort(
            key=lambda field: min(node.axis_score for node in by_field[field])
        )

        findings, balancing = [], []
        for field in ordered_fields[:BALANCER_MAX_FINDINGS]:
            nodes = by_field[field]
            weakness = [node for node in nodes if node.side == "weakness"]
            value = [node for node in nodes if node.side == "value"]
            lead = weakness[0] if weakness else value[0]
            finding_kind = "resolve" if weakness else "clarify"
            # 该字段下任一叶子判定需要判别性对照，finding 就带上标记：混淆缺陷
            # 只要还在，改主张措辞就消不掉它。required_contrast 取第一条非空说明。
            contrast_nodes = [
                node for node in nodes if node.requires_discriminating_contrast
            ]
            required_contrast = next(
                (
                    node.required_contrast
                    for node in contrast_nodes
                    if node.required_contrast.strip()
                ),
                "",
            )
            # 该字段下任一叶子判定为"未声明的 observable 依赖"，finding 就带上标记：
            # 下游据此把这条 finding 短路到 idea_field_update（声明源或收窄 observable
            # 是 idea 内部改写，不需外部检索），避免被证据通道误判成 needs_dataset。
            undeclared_observable_nodes = [
                node for node in nodes if node.undeclared_observable_dependency
            ]
            findings.append({
                "target_field": field,
                "dimension": lead.dimension,
                "severity": finding_kind,
                "question": lead.question,
                "resolution_hint": lead.adjudication,
                "requires_discriminating_contrast": bool(contrast_nodes),
                "required_contrast": required_contrast,
                "undeclared_observable_dependency": bool(undeclared_observable_nodes),
            })
            balancing.append({"target_field": field, "weakness_nodes": [node.node_id for node in weakness], "value_nodes": [node.node_id for node in value], "decision": "repair_without_losing_value" if weakness and value else "repair" if weakness else "make_value_explicit", "evidence_needs": [node.evidence_need for node in nodes if node.evidence_need], "axis_score": min(node.axis_score for node in nodes)})
        return findings, balancing

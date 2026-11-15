"""Content-based relocation of per-step refinement constraints across replan rounds.

step id 是位置序号而非稳定标识。full replan 会整份重写 plan 并从 1 重新编号：
删掉三步再连续编号后，旧 id 2/3/5 依然存在，但指向的已是完全不同的操作。把
refinement 的 per-step 约束按 id 贴回新 plan 因此有两个后果：

  1. `remove_step` 的闭合条件在结构上无法满足——只要新 plan 步数不少于旧 id，
     该 id 必然存在，"照做"反而被判成"未照做"；
  2. 非删除约束被贴到语义无关的节点上，既让 capability 锚定在错误约束下选函数，
     又让 I/O 语义校验产生错位的 contradiction。

本模块按内容重新定位约束：产出 artifact id 精确命中优先（artifact id 全局唯一且
语义化，是跨轮最稳的锚），其次步骤名完全相同，最后按名称 token 重叠加
task_type / 稳定输入引用 / 选中函数的一致性加权。定位是一对一的贪心分配：
一个旧约束最多落到一个新节点，一个新节点最多承接一个旧约束。

定位不到不是违约：对 `remove_step` 是删除已执行的证据，对其它 action 是证据失效。
两者的判定都交给调用方，本模块只负责给出确定性的匹配结果与匹配依据。

所有产出字段都是 JSON 可序列化的（identity 里的集合一律落成 sorted list），
因为编译后的约束会随 `step_plan_generation_audit` 落盘。
"""

import copy
from typing import Any, Dict, List, Optional, Set, Tuple

from datahypgen.mas.tools.operation_semantics import (
    content_tokens,
    normalize_function_name,
)


# step_output 引用会随整份重写漂移（artifact 改名即失配），不作为身份证据；
# base_dataset 与 required_dataset 引用锚定在外部数据契约上，跨轮稳定。
_STABLE_SOURCE_KINDS = frozenset({"base_dataset", "required_dataset"})

MATCH_OUTPUT_ARTIFACT = "output_artifact_id"
MATCH_STEP_NAME_EXACT = "step_name_exact"
MATCH_CONTENT_SIMILARITY = "content_similarity"
MATCH_STEP_ID_FALLBACK = "step_id_fallback"
MATCH_NONE = "unmatched"

_TIER_BY_MATCH = {
    MATCH_OUTPUT_ARTIFACT: 3,
    MATCH_STEP_NAME_EXACT: 2,
    MATCH_CONTENT_SIMILARITY: 1,
}

# 名称 token Jaccard 的准入下界。0.5 的校准依据：真实同义改写通常保留多数名词
# （"Summarize Zonal Statistics Table" vs "... Table Output" 为 0.8），而不同操作
# 之间即使句式相同也会跌破该线（"Access Base Climate Forcing" vs "Access Base
# VIIRS ET Dataset" 仅 0.4——共有的 access/base 是句式噪声，区分信息在
# climate/forcing vs viirs 上）。宁可判定位失败走 advisory，也不要把约束贴到
# 语义无关的节点上。
_SIMILARITY_MIN = 0.5

# 一致性加分只参与同分候选之间的排序，不参与准入。数据访问类步骤天然共享
# task_type 和 base_dataset 引用，若允许加分抬过准入线，这些结构噪声就会盖过
# 名称里真正的区分信息——实测正是它把两个不同数据源的访问步骤判成了同一步。
_TASK_TYPE_BONUS = 0.15
_INPUT_REF_BONUS = 0.15
_FUNCTION_BONUS = 0.1

# 允许支撑"删除未执行"这一硬失败判定的匹配等级。删除判定的两类错误代价不对称：
# 误判"仍存在"会硬拦一份正确计划（本次 session_1786760303 的失败形态），
# 误判"已删除"只是放行并留 advisory。因此删除只认精确锚，不认相似度。
_REMOVAL_GRADE_MATCHES = frozenset({MATCH_OUTPUT_ARTIFACT, MATCH_STEP_NAME_EXACT})


def build_step_identity(step: Any) -> Dict[str, Any]:
    """Extract one plan step's JSON-serializable content identity."""
    if not isinstance(step, dict):
        return {}
    name = str(step.get("step_name") or "").strip()
    output_ids = sorted({
        str(item.get("id") or "").strip().casefold()
        for item in (step.get("outputs") or [])
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    })
    input_refs = sorted({
        "{}:{}".format(
            str(item.get("source_kind") or "").strip().casefold(),
            str(item.get("source_id") or "").strip().casefold(),
        )
        for item in (step.get("inputs") or [])
        if isinstance(item, dict)
        and str(item.get("source_kind") or "").strip() in _STABLE_SOURCE_KINDS
        and str(item.get("source_id") or "").strip()
    })
    identity = {
        "step_id": step.get("id"),
        "step_name": name,
        "name_tokens": sorted(content_tokens(name)),
        "output_ids": output_ids,
        "input_refs": input_refs,
        "task_type": str(step.get("task_type") or "").strip().casefold(),
        "function_name": normalize_function_name(
            (step.get("implementation_hint") or {}).get("function_name")
        ),
    }
    # 全空身份无法参与内容匹配，视为没有可用锚。
    if not (
        identity["step_name"]
        or identity["output_ids"]
        or identity["input_refs"]
        or identity["task_type"]
    ):
        return {}
    return identity


def _jaccard(left: Set[str], right: Set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _content_similarity(
    anchor: Dict[str, Any],
    candidate: Dict[str, Any],
) -> Tuple[float, float]:
    """Return `(admission_score, ranking_score)` for a similarity pairing.

    准入只看名称 token 重叠；task_type / 输入引用 / 函数一致性作为加分只影响
    同分候选之间的排序。两者分开是必要的：数据访问步骤普遍共享 task_type 与
    base_dataset 引用，把加分计入准入会让结构噪声盖过名称里的区分信息。
    """
    admission = _jaccard(
        set(anchor.get("name_tokens") or []),
        set(candidate.get("name_tokens") or []),
    )
    ranking = admission
    anchor_task_type = anchor.get("task_type") or ""
    if anchor_task_type and anchor_task_type == (candidate.get("task_type") or ""):
        ranking += _TASK_TYPE_BONUS
    if set(anchor.get("input_refs") or []) & set(candidate.get("input_refs") or []):
        ranking += _INPUT_REF_BONUS
    anchor_function = anchor.get("function_name") or ""
    if anchor_function and anchor_function == (candidate.get("function_name") or ""):
        ranking += _FUNCTION_BONUS
    return admission, min(1.0, ranking)


def _score_pairing(
    anchor: Dict[str, Any],
    candidate: Dict[str, Any],
) -> Optional[Tuple[int, float, str]]:
    """Return (tier, score, match_kind) for a plausible pairing, else None."""
    anchor_outputs = set(anchor.get("output_ids") or [])
    if anchor_outputs and anchor_outputs & set(candidate.get("output_ids") or []):
        return _TIER_BY_MATCH[MATCH_OUTPUT_ARTIFACT], 1.0, MATCH_OUTPUT_ARTIFACT
    anchor_name = (anchor.get("step_name") or "").strip().casefold()
    candidate_name = (candidate.get("step_name") or "").strip().casefold()
    if anchor_name and anchor_name == candidate_name:
        return _TIER_BY_MATCH[MATCH_STEP_NAME_EXACT], 1.0, MATCH_STEP_NAME_EXACT
    admission, ranking = _content_similarity(anchor, candidate)
    if admission >= _SIMILARITY_MIN:
        return (
            _TIER_BY_MATCH[MATCH_CONTENT_SIMILARITY],
            ranking,
            MATCH_CONTENT_SIMILARITY,
        )
    return None


def relocate_step_constraints(
    steps: Any,
    constraints_by_step: Dict[int, Dict[str, Any]],
) -> Tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]]]:
    """Map refinement constraints from their original step ids onto the new plan.

    Returns `(relocated, unresolved)`:
      - `relocated` is keyed by the **new** plan step id and carries
        `source_step_id` / `relocation_match` / `relocation_score` for audit;
      - `unresolved` lists constraints with no content match, each carrying
        `source_step_id` and `relocation_match=unmatched`.
    """
    if not constraints_by_step:
        return {}, []

    identities: Dict[int, Dict[str, Any]] = {}
    for step in steps if isinstance(steps, list) else []:
        if not isinstance(step, dict):
            continue
        step_id = step.get("id")
        if not isinstance(step_id, int) or isinstance(step_id, bool):
            continue
        identities[step_id] = build_step_identity(step)

    pairings: List[Tuple[int, float, int, int, str]] = []
    for source_step_id, constraints in constraints_by_step.items():
        anchor = constraints.get("source_step_identity") or {}
        if not anchor:
            continue
        removal_requested = bool(constraints.get("removal_requested"))
        for candidate_id, candidate in identities.items():
            scored = _score_pairing(anchor, candidate)
            if scored is None:
                continue
            tier, score, match_kind = scored
            if removal_requested and match_kind not in _REMOVAL_GRADE_MATCHES:
                # 相似度匹配撑不起"删除未执行"的硬失败：留作未定位（= 删除已执行），
                # 代价是放过个别未真正删除的步骤，但不会硬拦正确计划。
                continue
            pairings.append((tier, score, source_step_id, candidate_id, match_kind))

    # 高 tier、高分优先；同分按 id 稳定排序保证确定性。
    pairings.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))

    relocated: Dict[int, Dict[str, Any]] = {}
    assigned_sources: Set[int] = set()
    assigned_targets: Set[int] = set()
    for _tier, score, source_step_id, candidate_id, match_kind in pairings:
        if source_step_id in assigned_sources or candidate_id in assigned_targets:
            continue
        assigned_sources.add(source_step_id)
        assigned_targets.add(candidate_id)
        entry = copy.deepcopy(constraints_by_step[source_step_id])
        entry["source_step_id"] = source_step_id
        entry["relocation_match"] = match_kind
        entry["relocation_score"] = round(float(score), 3)
        relocated[candidate_id] = entry

    unresolved: List[Dict[str, Any]] = []
    for source_step_id, constraints in constraints_by_step.items():
        if source_step_id in assigned_sources:
            continue
        entry = copy.deepcopy(constraints)
        entry["source_step_id"] = source_step_id
        if not (constraints.get("source_step_identity") or {}):
            # 没有可对照的旧 plan（refinement 反馈先于任何计划快照到达）时，
            # id 是唯一可用线索，退回按 id 直配以保留既有行为。
            if source_step_id in identities and source_step_id not in assigned_targets:
                entry["relocation_match"] = MATCH_STEP_ID_FALLBACK
                entry["relocation_score"] = 0.0
                relocated[source_step_id] = entry
                assigned_targets.add(source_step_id)
                continue
        entry["relocation_match"] = MATCH_NONE
        entry["relocation_score"] = 0.0
        unresolved.append(entry)

    return relocated, unresolved


__all__ = [
    "MATCH_CONTENT_SIMILARITY",
    "MATCH_NONE",
    "MATCH_OUTPUT_ARTIFACT",
    "MATCH_STEP_ID_FALLBACK",
    "MATCH_STEP_NAME_EXACT",
    "build_step_identity",
    "relocate_step_constraints",
]

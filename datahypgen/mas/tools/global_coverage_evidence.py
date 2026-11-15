"""Tiered evidence for proving a candidate's global spatial coverage.

此前全球覆盖只靠一条拼接文本（title/id/source/provider/thematic_group/tags）做区域词
提取，实测在 10821 条 catalog 记录上只能证明 26.9%。`description` 与 `keywords` 明明
已在归一化时保留，却从未参与判定；要求全球覆盖的 requirement 因此系统性拿不到证据。

但这些字段的证据强度并不相同，不能简单拼成一个字符串：实测 `description` 声称 global
的 3761 条记录里，有 468 条（12.4%）被自己的 bbox 证否——Allen Coral Atlas 的
description 写着 global，bbox 却是 `[-180, -33, 180, 33]`（仅热带珊瑚带）。把长文本与
结构化范围等价对待，等于用弱证据覆盖强证据。

因此按强度分层，并给 bbox 一票否决权：

  strong  bbox 覆盖近全经度与主要陆地纬度      结构化、可计算、不可争辩
  medium  title / id / keywords 含全球声明      短字段，出现即是断言
  weak    description 含全球声明                长文本，可能出现在其它语境

bbox 存在且明确非全球时，medium / weak 证据一律作废。无 bbox 可校验时才允许
medium / weak 独立成立。

本模块只做确定性查表与计算，不注入模型 prompt。
"""

from typing import Any, Dict, List, Optional, Sequence

from datahypgen.mas.tools.dataset_resolution_semantics import extract_region_terms


EVIDENCE_BBOX = "bbox_global_extent"
EVIDENCE_SHORT_FIELD = "short_field_global_claim"
EVIDENCE_DESCRIPTION = "description_global_claim"

# bbox 判据与 GeoDataSearch._bbox_implies_global_land_coverage 保持同一组阈值。
# 北界取 60 而非 75：SRTM 族受航天飞机轨道倾角限制天然截止在 60°N/56°S，
# 而它们是事实上的全球陆地 DEM 标准产品（判据沿革见 开发过程.md 8.4）。
_GLOBAL_BBOX_MIN_LON = -170.0
_GLOBAL_BBOX_MAX_LON = 170.0
_GLOBAL_BBOX_MIN_LAT = -50.0
_GLOBAL_BBOX_MAX_LAT = 60.0


def _parse_bbox(value: Any) -> Optional[List[float]]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        if len(parts) != 4:
            return None
        value = parts
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def bbox_implies_global(value: Any) -> bool:
    """Whether a bbox spans nearly all longitudes and the main land latitudes."""
    parsed = _parse_bbox(value)
    if parsed is None:
        return False
    min_lon, min_lat, max_lon, max_lat = parsed
    return (
        min_lon <= _GLOBAL_BBOX_MIN_LON
        and max_lon >= _GLOBAL_BBOX_MAX_LON
        and min_lat <= _GLOBAL_BBOX_MIN_LAT
        and max_lat >= _GLOBAL_BBOX_MAX_LAT
    )


def bbox_contradicts_global(value: Any) -> bool:
    """Whether a *present and parseable* bbox positively refutes global coverage.

    缺失或畸形的 bbox 不构成反驳——那是证据缺失，不是证否。只有 bbox 真实存在
    且未达到全球范围时，才足以推翻文本里的全球声明。
    """
    parsed = _parse_bbox(value)
    if parsed is None:
        return False
    return not bbox_implies_global(parsed)


def _claims_global(*values: Any) -> bool:
    text = " ".join(str(value or "") for value in values if value)
    if not text.strip():
        return False
    return "global" in extract_region_terms(text, explicit_region=None)


def assess_global_coverage_evidence(
    *,
    bbox: Any = None,
    short_fields: Optional[Sequence[Any]] = None,
    description: Any = None,
) -> Dict[str, Any]:
    """Assess whether the candidate's own metadata proves global coverage.

    Args:
        bbox: 候选的空间范围，强证据也是唯一的否决者。
        short_fields: title / id / keywords / tags 等短字段，中等强度证据。
        description: 长文本描述，弱证据。

    Returns:
        `{"proves_global": bool, "bbox_refutes": bool, "evidence": [...],
          "suppressed_evidence": [...]}`
        `suppressed_evidence` 记录被 bbox 否决掉的文本证据，供审计解释
        "description 说了 global 却仍未判定为全球" 这类结果。
    """
    evidence: List[str] = []
    suppressed: List[str] = []

    bbox_proves = bbox_implies_global(bbox)
    bbox_refutes = bbox_contradicts_global(bbox)
    if bbox_proves:
        evidence.append(EVIDENCE_BBOX)

    short_claim = _claims_global(*(short_fields or []))
    description_claim = _claims_global(description)

    if short_claim:
        (suppressed if bbox_refutes else evidence).append(EVIDENCE_SHORT_FIELD)
    if description_claim:
        (suppressed if bbox_refutes else evidence).append(EVIDENCE_DESCRIPTION)

    return {
        "proves_global": bool(evidence),
        "bbox_refutes": bbox_refutes,
        "evidence": evidence,
        "suppressed_evidence": suppressed,
    }


__all__ = [
    "EVIDENCE_BBOX",
    "EVIDENCE_DESCRIPTION",
    "EVIDENCE_SHORT_FIELD",
    "assess_global_coverage_evidence",
    "bbox_contradicts_global",
    "bbox_implies_global",
]

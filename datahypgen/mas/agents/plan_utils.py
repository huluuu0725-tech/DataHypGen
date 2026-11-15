"""
plan_utils.py — 无状态纯函数，从 PlanAgent 提取。
所有函数不依赖 self，可独立测试和复用。
"""
import copy
import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..tools.dataset_resolution_semantics import infer_product_families

logger = logging.getLogger(__name__)

# idea 层 prompt 用不到的 base_dataset 字段。剔除依据是全仓库无任何读取点（profiler
# 自己的类型探测不算下游消费），因此删掉它们不改变任何判断，只减少 prompt 体积。
# 这份上下文进 idea 生成、成熟化 revision/acceptance，以及经 _evidence_context 进
# Socratic 树的每次调用（实测单次会话约 470 次携带），所以单字段的体积会被放大。
#
# 只列零消费者。执行层字段（data_path / geometry / crs / time_range）和画像增量字段
# （profiled_statistics / profile_quality / portrait_summary_text）暂不剔除：它们
# grep 上也少有消费点，但模型是否在隐式使用无法靠静态检查确认，分开一轮验证。
DATASET_CONTEXT_DROP_TOP_LEVEL = ("local_arrow_unavailable_fields",)
DATASET_CONTEXT_DROP_DESCRIPTION = ("local_arrow_unavailable_fields",)
DATASET_CONTEXT_DROP_RASTER = (
    "first_band_count",
    # 仅 dataset_profiler_agent:650 的类型探测读取，下游 prompt 不需要。
    "band_data_types",
    "first_band_data_types",
)
# sample_properties / sample_runtime_properties 不整块删除：它们混装了 GEE runtime
# dump 和数据集的实质说明。实测 system:footprint 一项就占 4050 字符（坐标环），而
# SoilGrids 的 sample_properties.description（"Bulk density in cg/cm3 at 6 standard
# depths..."）是 data_grounding 判断用得上的语义。所以按键剔除 runtime 噪声，保留说明性字段。
DATASET_CONTEXT_DROP_SAMPLE_PROPERTY_KEYS = (
    "system:footprint",
    "system:index",
    "system:asset_size",
    "system:time_start",
    "system:time_end",
    "xsize",
    "ysize",
    "width",
    "height",
    "dtype",
    "declared_feature_dtype",
    "nodata",
    "num_bands",
)
DATASET_CONTEXT_SAMPLE_PROPERTY_FIELDS = (
    "sample_properties",
    "sample_runtime_properties",
)
# vector_content 不剔除 property_semantics_hint / property_keys：静态检查显示它们没有
# 代码消费者，但内容是向量属性的语义说明（如 "fatality_c: 事件报告的死亡人数"），
# data_grounding 的 target_observability 子标准正需要它来判断属性能否支撑主张。零代码消费
# 不等于零模型消费。
DATASET_CONTEXT_DROP_VECTOR: Tuple[str, ...] = ()
DATASET_CONTEXT_DROP_GEE_ASSET = ("thumbnail",)


def _drop_keys(payload: Any, keys: Tuple[str, ...]) -> None:
    if isinstance(payload, dict):
        for key in keys:
            payload.pop(key, None)


def build_idea_dataset_context(dataset_info: Dict[str, Any]) -> Dict[str, Any]:
    """构造 idea 层 prompt 使用的 base_dataset 视图。

    IdeaGenerationAgent 与 IdeaMaturationAgent 共用这一份实现：两者此前是逐字节相同的
    副本，任何一侧单独修改都会让 idea 生成与成熟化看到不同的数据集事实。
    """
    base_dataset = dataset_info.get("base_dataset", {}) if isinstance(dataset_info, dict) else {}
    if not isinstance(base_dataset, dict):
        return {"base_dataset": {}}
    sanitized = copy.deepcopy(base_dataset)
    _drop_keys(sanitized, DATASET_CONTEXT_DROP_TOP_LEVEL)
    data_description = sanitized.get("data_description")
    if isinstance(data_description, dict):
        _drop_keys(data_description, DATASET_CONTEXT_DROP_DESCRIPTION)
        raster_content = data_description.get("raster_content")
        _drop_keys(raster_content, DATASET_CONTEXT_DROP_RASTER)
        if isinstance(raster_content, dict):
            for field in DATASET_CONTEXT_SAMPLE_PROPERTY_FIELDS:
                _drop_keys(
                    raster_content.get(field),
                    DATASET_CONTEXT_DROP_SAMPLE_PROPERTY_KEYS,
                )
                # 只剩噪声键时整个字段就没有信息量了。
                if raster_content.get(field) == {}:
                    raster_content.pop(field, None)
        _drop_keys(data_description.get("vector_content"), DATASET_CONTEXT_DROP_VECTOR)
        _drop_keys(data_description.get("gee_asset"), DATASET_CONTEXT_DROP_GEE_ASSET)
        sanitized["data_description"] = data_description
    return {"base_dataset": sanitized}


# =====================================================================
# Capability Family Summary — 工具族摘要生成
# =====================================================================

# 用于详细规划 prompt 的人类可读标签
_TASK_TYPE_LABELS: Dict[str, str] = {
    "spatial_regression": "Spatial regression models",
    "spatial_autocorrelation": "Spatial autocorrelation analysis",
    "raster_band_math": "Raster band algebra and index computation",
    "raster_resample": "Raster resampling and regridding",
    "raster_clip": "Raster clipping and subsetting",
    "raster_reproject": "Raster reprojection",
    "raster_mosaic": "Raster mosaicking",
    "raster_io": "Raster read/write operations",
    "temporal_aggregation": "Time-series aggregation and resampling",
    "zonal_statistics": "Zonal statistics extraction",
    "spatial_join": "Spatial join and merge operations",
    "overlay_analysis": "Vector/raster overlay analysis",
    "point_interpolation": "Point data interpolation",
    "geometry_operation": "Geometry manipulation and measurement",
    "buffer_analysis": "Buffer and proximity analysis",
    "network_analysis": "Network and graph analysis",
    "catalog_search": "Data catalog discovery and access",
    "crs_transform": "Coordinate reference system transformation",
    "data_format_conversion": "Data format conversion",
    "data_quality_check": "Data quality and consistency checks",
    "dem_analysis": "Digital elevation model analysis",
    "visualization": "Map and chart rendering",
    "statistical_analysis": "Traditional statistical tests",
    "clustering": "Clustering and segmentation",
    "data_preprocessing": "Data preprocessing and normalization",
    "hydrology": "Hydrological analysis",
    "spectral_index": "Spectral index computation",
    "data_simulation": "Spatial data simulation",
    "general": "General-purpose geospatial utilities",
}

# 对 idea 生成阶段最重要的分析族（排除纯 I/O 和元数据操作族）
_ANALYSIS_RELEVANT_TASK_TYPES = frozenset({
    "spatial_regression",
    "spatial_autocorrelation",
    "raster_band_math",
    "raster_resample",
    "temporal_aggregation",
    "zonal_statistics",
    "spatial_join",
    "overlay_analysis",
    "point_interpolation",
    "buffer_analysis",
    "network_analysis",
    "dem_analysis",
    "statistical_analysis",
    "clustering",
    "data_preprocessing",
    "hydrology",
    "spectral_index",
    "data_simulation",
    "visualization",
})


def _load_capability_entries(cap_index_path: str) -> List[Dict[str, Any]]:
    """
    加载 geo_capability_index.json，返回 entry 列表。
    失败时返回空列表并记录警告。内部公用函数。
    """
    path = Path(cap_index_path)
    if not path.is_absolute():
        path = Path(os.getcwd()) / path

    if not path.exists():
        logger.warning("capability index not found at %s; skipping", path)
        return []

    try:
        with path.open("r", encoding="utf-8") as f:
            entries = json.load(f)
        return entries if isinstance(entries, list) else []
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("failed to load capability index: %s", exc)
        return []


def _aggregate_capability_entries(
    entries: List[Dict[str, Any]],
) -> Tuple[Dict[str, int], Dict[str, set]]:
    """
    按 task_type 聚合 entries，只保留分析相关族。
    返回 (type_counts, type_pkgs)。内部公用函数。
    """
    type_pkgs: Dict[str, set] = defaultdict(set)
    type_counts: Dict[str, int] = defaultdict(int)
    for entry in entries:
        tt = str(entry.get("task_type", "")).strip()
        if not tt or tt not in _ANALYSIS_RELEVANT_TASK_TYPES:
            continue
        type_counts[tt] += 1
        pkg = str(entry.get("package", "")).split("==")[0].split(">=")[0].strip().lower()
        if pkg:
            type_pkgs[tt].add(pkg)
    return dict(type_counts), dict(type_pkgs)


def build_capability_family_index(cap_index_path: str) -> Dict[str, List[str]]:
    """
    从 geo_capability_index.json 中按 task_type 聚合，返回结构化能力索引。

    返回值：{task_type: [pkg1, pkg2, ...]} ，只含分析相关族。
    可供程序化 reachability 判断使用（不依赖 LLM 自述）。
    失败时返回空字典。
    """
    entries = _load_capability_entries(cap_index_path)
    if not entries:
        return {}
    type_counts, type_pkgs = _aggregate_capability_entries(entries)
    return {tt: sorted(type_pkgs.get(tt, set())) for tt in type_counts}


def build_capability_family_summary(cap_index_path: str) -> str:
    """
    从 geo_capability_index.json 中按 task_type 聚合，输出紧凑的工具族摘要。

    只保留与分析相关的族（排除 catalog_search、crs_transform 等纯 I/O 操作），
    避免 prompt 过长。返回空字符串表示加载失败或无可用工具。
    """
    entries = _load_capability_entries(cap_index_path)
    if not entries:
        return ""

    type_counts, type_pkgs = _aggregate_capability_entries(entries)
    if not type_counts:
        return ""

    # 按能力条目数降序排列
    sorted_types = sorted(type_counts.keys(), key=lambda t: type_counts[t], reverse=True)

    lines = ["Available Analysis Toolkit (verified local capabilities):"]
    for tt in sorted_types:
        label = _TASK_TYPE_LABELS.get(tt, tt.replace("_", " ").title())
        pkgs = sorted(type_pkgs.get(tt, set()))
        pkg_str = ", ".join(pkgs) if pkgs else "built-in"
        lines.append(f"  - {tt}: {label} ({pkg_str}) — {type_counts[tt]} functions")

    # 附加通用基础设施包（不在索引中但确定可用）
    lines.append("")
    lines.append(
        "Also available as general infrastructure (not indexed above): "
        "numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, h5py, fiona, gdal"
    )

    return "\n".join(lines)


# ── 类常量 ──────────────────────────────────────────────────────────────
FUZZY_REQUIREMENT_PATTERNS = (
    r"\bor similar\b",
    r"\bor equivalent\b",
    r"\bor other\b",
    r"\bor proxy\b",
)

NOTES_FUZZY_SENTENCE_PATTERNS = (
    r"acceptable alternatives include[^.]*\.",
    r"alternatives include[^.]*\.",
    r"or other spatially continuous[^.]*\.",
)


# ── 文本归一化 ──────────────────────────────────────────────────────────

def normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    normalized = value.strip().lower()
    normalized = re.sub(r"[_\-/]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def clean_requirement_name_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if not text:
        return ""
    for pattern in FUZZY_REQUIREMENT_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*[,;:/-]?\s*\)", "", text)
    text = re.sub(r"\(\s*or [^)]+\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,;)])", r"\1", text)
    return text.strip(" ,;-")


def clean_requirement_notes_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if not text:
        return ""
    for pattern in NOTES_FUZZY_SENTENCE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# ── 分辨率差检测 ─────────────────────────────────────────────────────────

# 将常见分辨率描述字符串归一化为米（无法解析返回 None）
_RESOLUTION_UNIT_TO_METERS: Dict[str, float] = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "km": 1000.0,
    "kilometer": 1000.0,
    "kilometers": 1000.0,
}


def _parse_spatial_resolution_meters(value: Any) -> Optional[float]:
    """
    把 spatial_resolution 字符串解析为米数。
    支持 "30m", "9km", "0.5km", "500 m" 等格式。
    无法解析时返回 None。
    """
    text = str(value or "").strip().lower()
    if not text:
        return None
    # 匹配数字 + 可选空格 + 单位
    m = re.match(r"^([\d.]+)\s*(m|meter|meters|km|kilometer|kilometers)\b", text)
    if not m:
        return None
    try:
        val = float(m.group(1))
        unit = m.group(2)
        return val * _RESOLUTION_UNIT_TO_METERS[unit]
    except (ValueError, KeyError):
        return None


def detect_resolution_gap(
    datasets: List[Dict[str, Any]],
    threshold_ratio: float = 10.0,
) -> List[Dict[str, Any]]:
    """
    检测 required_datasets 列表中空间分辨率差异超过 threshold_ratio 倍的数据集对。

    返回需要桥接的记录列表，每条记录包含：
    {
        "dataset_a": str,  # 分辨率更细的数据集名称
        "dataset_b": str,  # 分辨率更粗的数据集名称
        "res_a_m": float,  # dataset_a 分辨率（米）
        "res_b_m": float,  # dataset_b 分辨率（米）
        "ratio": float,    # 粗 / 细的倍数
    }

    无法解析分辨率的数据集被跳过（不产生误报）。
    """
    parsed: List[Tuple[str, float]] = []
    for ds in datasets if isinstance(datasets, list) else []:
        if not isinstance(ds, dict):
            continue
        name = str(ds.get("name") or ds.get("id") or "").strip()
        res_m = _parse_spatial_resolution_meters(ds.get("spatial_resolution"))
        if name and res_m is not None:
            parsed.append((name, res_m))

    gaps: List[Dict[str, Any]] = []
    for i in range(len(parsed)):
        for j in range(i + 1, len(parsed)):
            name_a, res_a = parsed[i]
            name_b, res_b = parsed[j]
            if res_a <= res_b:
                fine_name, fine_res, coarse_name, coarse_res = name_a, res_a, name_b, res_b
            else:
                fine_name, fine_res, coarse_name, coarse_res = name_b, res_b, name_a, res_a
            ratio = coarse_res / fine_res if fine_res > 0 else 0.0
            if ratio >= threshold_ratio:
                gaps.append({
                    "dataset_a": fine_name,
                    "dataset_b": coarse_name,
                    "res_a_m": fine_res,
                    "res_b_m": coarse_res,
                    "ratio": ratio,
                })
    return gaps


# ── 算法可达性 deterministic 检测 ────────────────────────────────────────

# 自定义物理模型和不可复现方法的黑名单词汇（正则，词边界匹配）
_UNREACHABLE_METHOD_PATTERNS: Tuple[str, ...] = (
    r"\bcustom\s+(fire[\s-]spread|flood[\s-]inundation|wildfire[\s-]propagation)\s+model\b",
    r"\bbespoke\s+(solver|model|algorithm|framework)\b",
    r"\bproprietary\s+(model|algorithm|tool|software)\b",
    r"\bhand[\s-]coded\s+(PDE|ODE|solver|simulation)\b",
    r"\bcustom\s+physics[\s-]based\s+model\b",
    r"\bnumerical\s+(PDE|ODE)\s+solver\b",
    r"\bfinite[\s-](element|difference|volume)\s+(method|solver|simulation)\b",
    r"\bcellular\s+automaton\s+(fire|flood|spread)\b",
)


def detect_unreachable_method_claims(
    text: str,
    capability_index: Optional[Dict[str, List[str]]] = None,
) -> List[str]:
    """
    对 approach_description 等文本做最小 deterministic 匹配，
    检测声称使用了标准 Python 工具族无法实现的方法。

    返回命中的黑名单模式列表（空列表表示未检测到风险）。
    capability_index 当前未使用（保留接口，供后续扩展正向白名单匹配）。
    """
    if not text:
        return []
    text_low = text.lower()
    hits: List[str] = []
    for pattern in _UNREACHABLE_METHOD_PATTERNS:
        if re.search(pattern, text_low):
            hits.append(pattern)
    return hits


# ── 合约归一化 ──────────────────────────────────────────────────────────

def normalize_execution_requirement_level(value: Any) -> str:
    """
    缺失或非法值不归一为 required（避免静默当成硬门槛），默认 recommended，
    仅显式 required/optional/recommended 生效。
    """
    level = str(value or "").strip().lower()
    if level in {"required", "recommended", "optional"}:
        return level
    return "recommended"


def normalize_binding_notes(value: Any) -> str:
    return clean_requirement_notes_text(value)


def normalize_research_justification(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def upgrade_legacy_required_dataset_contract(entry: Any) -> Dict[str, Any]:
    """
    单一边界兼容：notes -> binding_notes 等，并删除已废弃的数据角色字段。
    非法或缺失的执行层级别在此置空，由 normalize_execution_requirement_level 单独归一（默认 recommended）。
    """
    if not isinstance(entry, dict):
        return {}
    upgraded = dict(entry)
    level = str(upgraded.get("execution_requirement_level") or "").strip().lower()
    if level not in {"required", "recommended", "optional"}:
        upgraded["execution_requirement_level"] = ""
    if upgraded.get("binding_notes") is None:
        upgraded["binding_notes"] = ""
    if not upgraded.get("binding_notes") and upgraded.get("notes"):
        upgraded["binding_notes"] = upgraded.get("notes")
    if upgraded.get("research_justification") is None:
        upgraded["research_justification"] = ""
    upgraded.pop("research_role", None)
    upgraded.pop("question_role", None)
    frozen_constraints = upgraded.get("frozen_constraints")
    if isinstance(frozen_constraints, dict):
        frozen_constraints = dict(frozen_constraints)
        frozen_constraints.pop("research_role", None)
        frozen_constraints.pop("question_role", None)
        upgraded["frozen_constraints"] = frozen_constraints
    return upgraded


def normalize_required_dataset_contract(entry: Dict[str, Any]) -> Dict[str, Any]:
    normalized = upgrade_legacy_required_dataset_contract(entry or {})
    normalized["execution_requirement_level"] = normalize_execution_requirement_level(
        normalized.get("execution_requirement_level"),
    )
    normalized["binding_notes"] = normalize_binding_notes(normalized.get("binding_notes"))
    normalized["research_justification"] = normalize_research_justification(
        normalized.get("research_justification")
    )
    return normalized


def derive_requirement_aliases(entry: Dict[str, Any]) -> List[str]:
    aliases = entry.get("aliases") or []
    if isinstance(aliases, str):
        aliases = [aliases]
    derived_aliases: List[str] = [str(alias).strip() for alias in aliases if str(alias).strip()]
    notes = str(entry.get("binding_notes") or "")
    for pattern in [
        r"\bNLCD\b.*?\bNational Land Cover Database\b",
        r"\bNational Land Cover Database\b",
        r"\bUSGS NLCD\b",
        r"\bESA WorldCover\b",
        r"\bTerraClimate\b",
        r"\bgridMET\b",
        r"\bPRISM\b",
        r"\bMODIS\b",
        r"\bNASADEM\b",
        r"\bSRTM\b",
        r"\bCopernicus DEM\b",
        r"\bGLO-30\b",
        r"\bGLO30\b",
        r"\bCopernicus GLO30\b",
        r"\bCopernicus DEM GLO-30\b",
        r"\bDAYMET\b",
        r"\bDAYMET V4\b",
        r"\bUSGS Ecoregions\b",
        r"\bEPA Ecoregions\b",
        r"\bEcoregions\b",
    ]:
        for match in re.findall(pattern, notes, flags=re.IGNORECASE):
            cleaned = re.sub(r"\s+", " ", str(match).strip())
            if cleaned and cleaned not in derived_aliases:
                derived_aliases.append(cleaned)
    return derived_aliases


def derive_frozen_constraints(entry: Dict[str, Any]) -> Dict[str, Any]:
    frozen = dict(entry.get("frozen_constraints", {}) or {})
    frozen.pop("research_role", None)
    frozen.pop("question_role", None)
    modality = str(entry.get("modality", "") or "").strip()
    region = str(
        frozen.get("region")
        or entry.get("region")
        or entry.get("region_name")
        or ""
    ).strip()
    if region:
        frozen["region"] = region
    if modality:
        frozen["modality"] = modality
    if "candidate_type" not in frozen:
        if modality in {"remote_sensing", "gridded_climate"}:
            frozen["candidate_type"] = ["image", "image_collection"]
        elif modality == "vector":
            frozen["candidate_type"] = ["table"]
        elif modality == "tabular":
            frozen["candidate_type"] = ["table"]
    return frozen


def derive_soft_constraints(entry: Dict[str, Any]) -> Dict[str, Any]:
    soft = dict(entry.get("soft_constraints", {}) or {})
    if entry.get("spatial_resolution") and "spatial_resolution" not in soft:
        soft["spatial_resolution"] = entry.get("spatial_resolution")
    if entry.get("temporal_resolution") and "temporal_alignment" not in soft:
        soft["temporal_alignment"] = entry.get("temporal_resolution")
    if entry.get("temporal_coverage") and "temporal_coverage" not in soft:
        soft["temporal_coverage"] = entry.get("temporal_coverage")
    return soft


def normalize_dataset_identity(value: Any) -> str:
    """规范化数据集身份标识：去 GEE: 前缀、去首尾引号/空白、小写。

    供 base canonical_id 派生与 base 识别共用，保证两侧规范化规则一致。
    """
    normalized = str(value or "").strip()
    if not normalized:
        return ""
    normalized = re.sub(r"^GEE\s*:\s*", "", normalized, flags=re.IGNORECASE)
    normalized = normalized.strip().strip("\"'")
    return normalized.lower()


def derive_base_canonical_id(base_dataset: Dict[str, Any]) -> str:
    """从 base_dataset 派生稳定的 canonical_id，作为 base 身份的唯一真相源。

    优先用 GEE 资产 id（gee_id/ee_id），其次本地 data_path 的 basename，
    最后回退 source 字段。该 id 由代码从 prompt.json 派生，不受 LLM 生成的
    简称或伪造身份影响。返回带来源前缀的稳定标识，如
    'gee:copernicus/s2_sr_harmonized' / 'local:cabuar_dataset'；
    无法派生时返回空串。
    """
    if not isinstance(base_dataset, dict):
        return ""
    normalized_gee = normalize_dataset_identity(
        base_dataset.get("gee_id") or base_dataset.get("ee_id")
    )
    if normalized_gee:
        return f"gee:{normalized_gee}"
    data_path = str(base_dataset.get("data_path") or "").strip()
    if data_path:
        basename = os.path.basename(data_path.rstrip("/\\"))
        normalized_path = normalize_dataset_identity(basename)
        if normalized_path:
            return f"local:{normalized_path}"
    return normalize_dataset_identity(base_dataset.get("source"))


def is_base_dataset_requirement(
    entry: Dict[str, Any],
    dataset_info: Dict[str, Any],
) -> bool:
    def _collect_identities(payload: Any) -> set[str]:
        identities: set[str] = set()
        if not isinstance(payload, dict):
            return identities

        direct_candidates = [
            payload.get("ee_id"),
            payload.get("gee_id"),
            payload.get("source"),
            payload.get("data_path"),
            payload.get("path"),
            payload.get("id"),
            payload.get("asset_id"),
            payload.get("canonical_id"),
        ]
        for value in direct_candidates:
            normalized = normalize_dataset_identity(value)
            if normalized:
                identities.add(normalized)

        data_description = payload.get("data_description")
        if isinstance(data_description, dict):
            raster_content = data_description.get("raster_content")
            if isinstance(raster_content, dict):
                normalized = normalize_dataset_identity(raster_content.get("gee_asset_id"))
                if normalized:
                    identities.add(normalized)
            gee_asset = data_description.get("gee_asset")
            if isinstance(gee_asset, dict):
                normalized = normalize_dataset_identity(gee_asset.get("id"))
                if normalized:
                    identities.add(normalized)
        return identities

    if not isinstance(entry, dict):
        return False

    base_dataset = dataset_info.get("base_dataset") or {}

    # 判据①（最强）：canonical_id 精确匹配。base_canonical 由代码派生（唯一真相），
    # entry 侧 canonical 标记来自代码盖章（strip_base_from_required_datasets），
    # 不受 LLM 伪造身份影响。
    base_canonical = str(
        base_dataset.get("canonical_id") or derive_base_canonical_id(base_dataset) or ""
    ).strip().lower()
    if base_canonical:
        entry_canonicals = {
            str(entry.get("canonical_id") or "").strip().lower(),
            str(entry.get("base_canonical_id") or "").strip().lower(),
        }
        entry_canonicals.discard("")
        if base_canonical in entry_canonicals:
            return True
        base_canonical_bare = base_canonical.split(":", 1)[-1]
        if base_canonical_bare and base_canonical_bare in _collect_identities(entry):
            return True

    # 判据②：显式 binding_notes 标记
    notes = str(entry.get("binding_notes") or "").lower()
    if "base dataset already available" in notes or "base dataset already available locally" in notes:
        return True

    # 判据③：identity 交集（对携带真实身份的 entry 有效）
    base_identities = _collect_identities(base_dataset)
    base_identities.update(_collect_identities(dataset_info))
    entry_identities = _collect_identities(entry)
    if base_identities and entry_identities and base_identities & entry_identities:
        return True

    # 判据④：name token 兜底。识别 LLM 用简称 + 伪造身份写回的 base 条目。
    GENERIC_TOKENS = frozenset({
        "digital", "elevation", "model", "data", "dataset", "global",
        "version", "collection", "product", "imagery", "satellite",
        "resolution", "level", "analysis", "ready", "surface", "terrain",
    })

    base_name = str(
        base_dataset.get("name")
        or dataset_info.get("name")
        or ""
    ).strip()
    entry_name = str(entry.get("name") or "").strip()
    if not base_name or not entry_name:
        return False

    base_tokens = {
        token for token in re.findall(r"[a-z0-9]+", base_name.lower())
        if token and token not in GENERIC_TOKENS
    }
    entry_tokens = {
        token for token in re.findall(r"[a-z0-9]+", entry_name.lower())
        if token and token not in GENERIC_TOKENS
    }
    if not base_tokens or not entry_tokens:
        return False
    overlap = len(base_tokens & entry_tokens)

    # 严格判据：token 重叠 ≥3，无需附加条件
    if overlap >= min(3, len(entry_tokens)) and overlap >= min(3, len(base_tokens)):
        return True

    return False


def strip_base_from_required_datasets(
    required_datasets: Any,
    dataset_info: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """从 required_datasets 剥离 base dataset 条目，返回 (kept, removed)。

    base 真相独立存在于 dataset_info.base_dataset；required_datasets 契约上只装
    外部数据。此函数用统一的 is_base_dataset_requirement 识别并剔除泄漏/重复的
    base 条目，供 idea 生成与成熟化 writeback 两处复用（贯穿契约）。
    """
    kept: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []
    for item in (required_datasets if isinstance(required_datasets, list) else []):
        if isinstance(item, dict) and is_base_dataset_requirement(item, dataset_info):
            removed.append(item)
        else:
            kept.append(item)
    return kept, removed


# ── dataset ID / plan entry ─────────────────────────────────────────────

def get_required_dataset_id(item: Any) -> str:
    if not isinstance(item, dict):
        return str(item or "").strip()
    dataset_id = str(item.get("id", "") or "").strip()
    if dataset_id:
        return dataset_id
    return str(item.get("name", "") or "").strip()


def build_application_idea_id(name: str, priority: int) -> str:
    normalized = normalize_text(name)
    slug = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    slug = slug or f"application_{priority}"
    return f"app_{priority:02d}_{slug}"


def build_application_plan_entry(
    scenario: Dict[str, Any],
    idx: int,
    plan_result: Dict[str, Any],
    anchored_steps: List[Dict[str, Any]],
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    priority = int(scenario.get("priority", idx + 1))
    application_name = (
        plan_result.get("application_name")
        or scenario.get("name")
        or f"Application {idx + 1}"
    )
    entry = {
        "idea_id": build_application_idea_id(application_name, priority),
        "application_id": idx + 1,
        "application_name": application_name,
        "application_description": (
            plan_result.get("application_description")
            or scenario.get("description", "")
        ),
        "application_context": scenario.get("context", ""),
        "approach_description": (
            plan_result.get("approach_description", "")
            or scenario.get("approach_description", "")
            or scenario.get("proposed_method", "")
        ),
        "priority": priority,
        "datasets": plan_result.get("datasets", []),
        "required_datasets": plan_result.get("required_datasets", []),
        "task_plan": anchored_steps,
        "rationale": plan_result.get("rationale", ""),
    }
    if extra_fields:
        entry.update(extra_fields)
    return entry


# ── 时间范围 / 数据描述 ──────────────────────────────────────────────────

def format_time_range_for_prompt(time_range: Any) -> str:
    if time_range is None:
        return ""
    if isinstance(time_range, str):
        return time_range.strip()
    if isinstance(time_range, dict):
        start = time_range.get("start") or ""
        end = time_range.get("end") or ""
        granularity = time_range.get("granularity") or ""
        parts = []
        if start:
            parts.append(f"start={start}")
        if end:
            parts.append(f"end={end}")
        if granularity:
            parts.append(f"granularity={granularity}")
        return "; ".join(parts)
    return str(time_range).strip()


def stringify_data_description(data_description: Any) -> str:
    if isinstance(data_description, str):
        return data_description.strip()
    if isinstance(data_description, (dict, list)):
        return json.dumps(data_description, ensure_ascii=False)
    return str(data_description or "").strip()


def extract_data_description_payload(dataset_info: Dict[str, Any]) -> Any:
    if not isinstance(dataset_info, dict):
        return ""
    base_dataset = dataset_info.get("base_dataset", {})
    if isinstance(base_dataset, dict) and base_dataset.get("type") == "composite_dataset":
        description = copy.deepcopy(base_dataset.get("data_description") or {})
        if not isinstance(description, dict):
            description = {"description": str(description)}
        description["dataset_name"] = base_dataset.get("name") or description.get("dataset_name") or "Composite observation"
        description["data_type"] = "Composite multi-dataset observation"
        description["anchor_asset_type"] = "composite_observation"
        components = []
        for component in base_dataset.get("dataset_components") or []:
            if not isinstance(component, dict):
                continue
            components.append({
                "role": component.get("role"),
                "dataset_name": component.get("name"),
                "data_path": component.get("source") or component.get("data_path"),
                "data_description": copy.deepcopy(component.get("data_description") or {}),
            })
        if components:
            description["components"] = components
        contract = base_dataset.get("combination_contract")
        if isinstance(contract, dict) and contract:
            description["combination_contract"] = copy.deepcopy(contract)
        return description
    if isinstance(base_dataset, dict) and base_dataset.get("data_description") not in (None, ""):
        return base_dataset.get("data_description")
    return dataset_info.get("data_description", "")


def build_compact_dataset_description(dataset_info: Dict[str, Any]) -> str:
    """
    仅保留原始数据事实层字段，避免把画像结果重复注入到 Dataset Description。
    """
    if not isinstance(dataset_info, dict):
        return ""

    base_dataset = dataset_info.get("base_dataset", {})
    if not isinstance(base_dataset, dict):
        base_dataset = {}

    data_description = extract_data_description_payload(dataset_info)
    if isinstance(data_description, str):
        return data_description.strip()
    if not isinstance(data_description, dict):
        return stringify_data_description(data_description)

    spatial_coverage = data_description.get("spatial_coverage", {}) or {}
    raster_content = data_description.get("raster_content", {}) or {}
    pixel_values = raster_content.get("pixel_values", {}) or {}

    allowed_keys = (
        "dataset_name",
        "source_dataset_capability",
        "dataset_features",
        "dataset_feature_contract",
        "idea_generation_constraint",
        "temporal_coverage",
        "data_type",
        "data_format",
        "spatial_resolution",
        "coordinate_system",
        "temporal_unit",
        "total_years",
        "records",
        "data_type_numeric",
        "total_samples",
        "total_size",
        "data_source",
        "reference",
        "anchor_asset_type",
        "components",
        "combination_contract",
    )
    compact: Dict[str, Any] = {
        key: data_description.get(key)
        for key in allowed_keys
        if data_description.get(key) not in (None, "", [], {})
    }

    region_name = spatial_coverage.get("region_name") or base_dataset.get("region_name")
    if region_name:
        compact["region_name"] = region_name
    if spatial_coverage:
        compact["spatial_coverage"] = {
            key: spatial_coverage.get(key)
            for key in ("region_name", "lon", "lat")
            if spatial_coverage.get(key) not in (None, "")
        }
    if raster_content.get("classification_type"):
        compact["classification_type"] = raster_content.get("classification_type")
    if pixel_values:
        compact["pixel_values"] = pixel_values

    return json.dumps(compact, ensure_ascii=False)



def resolve_dataset_portrait(
    context: Dict[str, Any],
    dataset_info: Dict[str, Any],
) -> Dict[str, Any]:
    portrait = context.get("dataset_portrait", {}) if isinstance(context, dict) else {}
    if isinstance(portrait, dict) and portrait:
        return portrait
    data_description = extract_data_description_payload(dataset_info)
    if isinstance(data_description, dict):
        portrait_text = str(data_description.get("portrait_summary_text") or "").strip()
        if portrait_text:
            return {"portrait_text": portrait_text}
    return {}


def resolve_enriched_data_description(
    dataset_info: Dict[str, Any],
    dataset_portrait: Dict[str, Any],
) -> Any:
    if isinstance(dataset_portrait, dict) and dataset_portrait.get("enriched_data_description") not in (None, ""):
        return dataset_portrait.get("enriched_data_description")
    return extract_data_description_payload(dataset_info)


def normalize_dataset_info_with_enriched_description(
    dataset_info: Dict[str, Any],
    enriched_data_description: Any,
) -> Dict[str, Any]:
    if not isinstance(dataset_info, dict):
        return {}
    normalized = dict(dataset_info)
    if enriched_data_description not in (None, ""):
        normalized["dataset_portrait_enriched_description"] = enriched_data_description
        base_dataset = normalized.get("base_dataset")
        if isinstance(base_dataset, dict):
            normalized_base_dataset = dict(base_dataset)
            normalized_base_dataset["data_description"] = enriched_data_description
            normalized["base_dataset"] = normalized_base_dataset
        else:
            normalized["data_description"] = enriched_data_description
    return normalized


def build_dataset_portrait_prompt_block(dataset_portrait: Dict[str, Any]) -> str:
    if not isinstance(dataset_portrait, dict) or not dataset_portrait:
        return ""

    axes = dataset_portrait.get("candidate_question_axes", []) or []
    hooks = dataset_portrait.get("suggested_research_hooks", []) or []
    portrait_text = str(dataset_portrait.get("portrait_text", "") or "").strip()

    parts: List[str] = []
    if portrait_text:
        parts.append(f"portrait_summary: {portrait_text}")
    if axes:
        parts.append(f"candidate_question_axes: {axes[:5]}")
    if hooks:
        parts.append(f"data_driven_research_hooks: {hooks[:5]}")
    return "\n".join(parts).strip()


def build_compact_dataset_portrait_prompt_block(dataset_portrait: Dict[str, Any]) -> str:
    """
    仅保留画像中的问题轴和启发层信息，不重复注入长篇 portrait_summary。
    """
    if not isinstance(dataset_portrait, dict) or not dataset_portrait:
        return ""

    axes = dataset_portrait.get("candidate_question_axes", []) or []
    hooks = dataset_portrait.get("suggested_research_hooks", []) or []

    parts: List[str] = []
    if axes:
        parts.append(f"candidate_question_axes: {axes[:3]}")
    if hooks:
        parts.append(f"data_driven_research_hooks: {hooks[:3]}")
    return "\n".join(parts).strip()




def build_dataset_observability_context(dataset_info: Dict[str, Any]) -> str:
    base_dataset = dataset_info.get("base_dataset", {}) if isinstance(dataset_info, dict) else {}
    if not isinstance(base_dataset, dict):
        base_dataset = {}

    data_description = extract_data_description_payload(dataset_info)
    dataset_portrait = resolve_dataset_portrait({}, dataset_info)
    portrait_text_block = build_dataset_portrait_prompt_block(dataset_portrait)

    region_name = str(
        (base_dataset.get("region_name") or "")
        or ((data_description.get("spatial_coverage") or {}).get("region_name") if isinstance(data_description, dict) else "")
    ).strip()
    time_range = format_time_range_for_prompt(
        (data_description.get("temporal_coverage") if isinstance(data_description, dict) else None)
        or base_dataset.get("time_range")
        or base_dataset.get("temporal_coverage")
    )
    modality = str(base_dataset.get("modality") or "").strip()
    data_desc_text = stringify_data_description(data_description)

    parts = [
        f"region_name: {region_name or 'Unknown'}",
        f"time_range: {time_range or 'Unknown'}",
        f"modality: {modality or 'Unknown'}",
        f"observability_hint (from data_description): {data_desc_text or 'Unknown'}",
    ]
    if portrait_text_block:
        parts.append(portrait_text_block)
    return "\n".join(parts).strip()


# ── forbidden title ─────────────────────────────────────────────────────

def collect_forbidden_title_fragments(domain_knowledge: Dict[str, Any]) -> List[str]:
    forbidden: List[str] = []
    gaps = domain_knowledge.get("gaps", []) if isinstance(domain_knowledge, dict) else []
    if isinstance(gaps, list):
        for g in gaps:
            if isinstance(g, dict):
                mc = str(g.get("missing_combination") or "").strip()
                if mc:
                    forbidden.append(mc)

    clusters = (domain_knowledge.get("clusters") or {}) if isinstance(domain_knowledge, dict) else {}
    applications = clusters.get("applications", {}) if isinstance(clusters, dict) else {}
    if isinstance(applications, dict):
        for theme_key in applications.keys():
            if isinstance(theme_key, str) and theme_key.strip():
                forbidden.append(theme_key)

    suggestions = domain_knowledge.get("suggestions", []) if isinstance(domain_knowledge, dict) else []
    if isinstance(suggestions, list):
        for s in suggestions:
            if isinstance(s, dict):
                rq = str(s.get("research_question") or "").strip()
                if rq:
                    forbidden.append(rq)

    # 去重（按归一化）
    seen = set()
    out: List[str] = []
    for t in forbidden:
        t = str(t or "").strip()
        if not t:
            continue
        k = normalize_text(t)
        if k and k not in seen:
            seen.add(k)
            out.append(t)
    return out


def violates_forbidden_title(title: str, forbidden_fragments: List[str]) -> bool:
    title_norm = normalize_text(title)
    if not title_norm:
        return True
    for frag in forbidden_fragments:
        frag_norm = normalize_text(frag)
        if not frag_norm:
            continue
        # 精确归一化相等（最严格）
        if title_norm == frag_norm:
            return True
        # 若片段很长，允许用子串检测避免"几乎原句"
        if len(frag_norm) >= 15 and frag_norm in title_norm:
            return True
    return False


# ── combination essential signal texts ───────────────────────────────────

def collect_combination_essential_signal_texts(
    dataset_portrait: Optional[Dict[str, Any]],
) -> List[str]:
    if not isinstance(dataset_portrait, dict):
        return []

    collected: List[str] = []
    for key in ("candidate_question_axes", "suggested_research_hooks", "data_driven_research_hooks"):
        values = dataset_portrait.get(key, []) or []
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, str):
                text_value = value.strip()
            elif isinstance(value, dict):
                text_value = str(
                    value.get("name")
                    or value.get("title")
                    or value.get("question")
                    or value.get("hook")
                    or value.get("description")
                    or ""
                ).strip()
            else:
                text_value = str(value or "").strip()
            if text_value:
                collected.append(text_value)
    return collected


# ── region extraction / scoring ──────────────────────────────────────────

def extract_region_tokens(region_name: str) -> List[str]:
    """
    从 region_name 中提取可匹配的 token，支持中英文地名。
    """
    if not region_name:
        return []

    # 常见地名的中英文映射
    region_mapping = {
        "texas": ["texas", "得克萨斯", "德克萨斯", "德州"],
        "california": ["california", "加利福尼亚", "加州"],
        "florida": ["florida", "佛罗里达", "佛州"],
        "new york": ["new york", "纽约"],
        "montana": ["montana", "蒙大拿"],
        "idaho": ["idaho", "爱达荷"],
        "washington": ["washington", "华盛顿"],
        "oregon": ["oregon", "俄勒冈"],
        "colorado": ["colorado", "科罗拉多"],
        "arizona": ["arizona", "亚利桑那"],
    }

    tokens = []
    region_lower = region_name.lower()

    # 检查是否匹配已知地名
    for eng_name, variants in region_mapping.items():
        if eng_name in region_lower:
            tokens.extend(variants)
            break

    # 如果没有匹配到，使用原始 region_name 的 token
    if not tokens:
        # 分割并清理
        parts = region_name.lower().replace(",", " ").split()
        tokens = [p.strip() for p in parts if p.strip() and len(p.strip()) >= 3]

    return tokens


def extract_target_region_signals(
    dataset_info: Optional[Dict[str, Any]],
    context: Optional[Dict[str, Any]],
) -> List[str]:
    dataset_info = dataset_info if isinstance(dataset_info, dict) else {}
    context = context if isinstance(context, dict) else {}

    base_dataset = dataset_info.get("base_dataset", {})
    geometry = base_dataset.get("geometry", {}) if isinstance(base_dataset, dict) else {}
    raw_values = [
        dataset_info.get("region_name"),
        base_dataset.get("region_name") if isinstance(base_dataset, dict) else None,
        geometry.get("region_name") if isinstance(geometry, dict) else None,
        base_dataset.get("name") if isinstance(base_dataset, dict) else None,
        context.get("task_name"),
        context.get("goal"),
    ]

    generic_tokens = {
        "mtbs", "burn", "severity", "mosaics", "mosaic", "dataset", "discovery",
        "geospatial", "analysis", "wildfire", "fire", "prediction", "model",
        "casual", "usa", "state", "task",
    }
    signals: List[str] = []
    seen: set = set()
    for raw in raw_values:
        normalized = normalize_text(raw)
        if not normalized:
            continue
        candidates = [normalized]
        candidates.extend(
            part.strip() for part in re.split(r"[(),]", normalized)
            if isinstance(part, str) and part.strip()
        )
        candidates.extend(
            token.strip() for token in normalized.split()
            if token.strip() and token.strip() not in generic_tokens and len(token.strip()) >= 4
        )
        for candidate in candidates:
            if candidate and candidate not in seen:
                seen.add(candidate)
                signals.append(candidate)
    return signals


def score_region_match(text: str, target_region_signals: List[str]) -> int:
    normalized_text = normalize_text(text)
    if not normalized_text or not target_region_signals:
        return 0

    exact_matches = [
        signal for signal in target_region_signals
        if len(signal) >= 4 and signal in normalized_text
    ]
    if exact_matches:
        return 3

    text_tokens = set(normalized_text.split())
    for signal in target_region_signals:
        signal_tokens = {
            token for token in signal.split()
            if len(token) >= 4 and token not in {"north", "south", "east", "west"}
        }
        if signal_tokens and signal_tokens.issubset(text_tokens):
            return 2
    return 0


# ── scenario builders ───────────────────────────────────────────────────

def build_gap_scenario(
    gap: Dict[str, Any],
    gap_index: int,
    target_region_signals: List[str],
) -> Optional[Dict[str, Any]]:
    if not isinstance(gap, dict):
        return None

    description = str(gap.get("description") or "").strip()
    missing_combination = str(gap.get("missing_combination") or "").strip()
    potential_impact = str(gap.get("potential_impact") or "").strip()
    cited_papers = gap.get("cited_papers", []) or []
    if not description and not missing_combination:
        return None

    name = missing_combination or description.split("，")[0].split(".")[0].strip()
    candidate_text = " ".join(
        part for part in [name, description, missing_combination, potential_impact] if part
    )
    region_match_score = score_region_match(candidate_text, target_region_signals)
    context_lines = [
        f"Scenario source: research gap #{gap_index + 1}.",
        f"Gap description: {description or missing_combination}",
    ]
    if missing_combination:
        context_lines.append(f"Missing combination: {missing_combination}")
    if potential_impact:
        context_lines.append(f"Potential impact: {potential_impact}")
    if cited_papers:
        context_lines.append(f"Supporting papers: {len(cited_papers)}")
    if region_match_score > 0:
        context_lines.append("Target-region match detected in the gap statement.")

    gap_id = str(gap.get("id") or "").strip() or f"gap_{gap_index + 1}"

    return {
        "name": name,
        "description": description or missing_combination,
        "context": " ".join(context_lines),
        "candidate_source": "gap",
        "source_rank": 0,
        "support_size": len(cited_papers),
        "region_match_score": region_match_score,
        "gap_id": gap_id,
        "priority": 0,
    }


def build_suggestion_scenario(
    suggestion: Dict[str, Any],
    suggestion_index: int,
    linked_gap: Optional[Dict[str, Any]],
    linked_gap_id: str,
    target_region_signals: List[str],
) -> Optional[Dict[str, Any]]:
    if not isinstance(suggestion, dict):
        return None

    research_question = str(suggestion.get("research_question") or "").strip()
    hypothesis = str(suggestion.get("hypothesis") or "").strip()
    proposed_method = str(suggestion.get("proposed_method") or "").strip()
    rationale = str(suggestion.get("rationale") or "").strip()
    data_requirements = str(suggestion.get("data_requirements") or "").strip()
    if not research_question and not proposed_method:
        return None

    linked_gap_description = ""
    linked_gap_support = 0
    linked_gap_id = str(linked_gap_id or "").strip()
    if isinstance(linked_gap, dict):
        linked_gap_description = str(
            linked_gap.get("description") or linked_gap.get("missing_combination") or ""
        ).strip()
        linked_gap_support = len(linked_gap.get("cited_papers", []) or [])
        if not linked_gap_id:
            linked_gap_id = str(linked_gap.get("id") or "").strip()

    name = research_question or proposed_method
    description = research_question or rationale or proposed_method
    candidate_text = " ".join(
        part for part in [
            name, description, hypothesis, proposed_method, rationale,
            data_requirements, linked_gap_description,
        ] if part
    )
    region_match_score = score_region_match(candidate_text, target_region_signals)
    context_lines = [
        f"Scenario source: suggestion #{suggestion_index + 1}.",
        f"Operationalized research question: {research_question or proposed_method}",
    ]
    if linked_gap_id:
        context_lines.append(f"Linked gap id: {linked_gap_id}")
    if linked_gap_description:
        context_lines.append(f"Operationalizes gap: {linked_gap_description}")
    if hypothesis:
        context_lines.append(f"Hypothesis: {hypothesis}")
    if proposed_method:
        context_lines.append(f"Proposed method: {proposed_method}")
    if data_requirements:
        context_lines.append(f"Data requirements: {data_requirements}")
    if rationale:
        context_lines.append(f"Rationale: {rationale}")
    if region_match_score > 0:
        context_lines.append("Target-region match detected in the suggestion.")

    return {
        "name": name,
        "description": description,
        "context": " ".join(context_lines),
        "approach_description": proposed_method,
        "candidate_source": "suggestion",
        "source_rank": 1,
        "support_size": linked_gap_support,
        "region_match_score": region_match_score,
        "gap_id": linked_gap_id,
        "priority": 0,
    }


def build_cluster_scenario(
    theme: str,
    support: int,
    target_region_signals: List[str],
) -> Optional[Dict[str, Any]]:
    theme = str(theme or "").strip()
    if not theme:
        return None

    region_match_score = score_region_match(theme, target_region_signals)
    context = f"Scenario source: literature application cluster. Theme support in literature cluster: {support} related papers."
    if region_match_score > 0:
        context += " Target-region match detected in the cluster theme."

    return {
        "name": theme,
        "description": f"Study the application theme: {theme} (fallback from domain knowledge clustering).",
        "context": context,
        "candidate_source": "cluster",
        "source_rank": 2,
        "support_size": support,
        "region_match_score": region_match_score,
        "priority": 0,
    }


def build_scenarios_from_domain_knowledge(
    domain_knowledge: Any,
    count: int,
    dataset_info: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    基于 DomainKnowledge 生成候选场景，采用 gap-first 策略：
    1. 优先从 gaps 中提取候选研究场景；
    2. 其次使用 suggestions 将 gap 落成可执行问题；
    3. 最后才使用 clusters.applications 作为 fallback。
    """
    if not isinstance(domain_knowledge, dict):
        return []

    def _support_size(v: Any) -> int:
        if isinstance(v, list):
            return len(v)
        if isinstance(v, dict):
            return len(v)
        return 0

    count = max(0, int(count))
    if count <= 0:
        return []

    clusters = domain_knowledge.get("clusters", {})
    applications = clusters.get("applications", {}) if isinstance(clusters, dict) else {}
    gaps = domain_knowledge.get("gaps", [])
    suggestions = domain_knowledge.get("suggestions", [])
    target_region_signals = extract_target_region_signals(dataset_info, context)

    scenario_candidates: List[Dict[str, Any]] = []

    if isinstance(gaps, list):
        for gap_index, gap in enumerate(gaps):
            gap_scenario = build_gap_scenario(gap, gap_index, target_region_signals)
            if gap_scenario:
                scenario_candidates.append(gap_scenario)

    gap_map: Dict[str, Dict[str, Any]] = {}
    if isinstance(gaps, list):
        for gap_index, gap in enumerate(gaps):
            if not isinstance(gap, dict):
                continue
            gap_id = str(gap.get("id") or "").strip() or f"gap_{gap_index + 1}"
            gap_map[gap_id] = gap

    if isinstance(suggestions, list):
        for suggestion_index, suggestion in enumerate(suggestions):
            suggestion_gap_id = ""
            if isinstance(suggestion, dict):
                suggestion_gap_id = str(suggestion.get("gap_id") or "").strip()
            if not suggestion_gap_id:
                logger.warning(
                    "skipping suggestion without explicit gap_id at index %s",
                    suggestion_index,
                )
                continue
            linked_gap = gap_map.get(suggestion_gap_id)
            if linked_gap is None:
                logger.warning(
                    "skipping suggestion with unknown gap_id=%s at index %s",
                    suggestion_gap_id,
                    suggestion_index,
                )
                continue
            suggestion_scenario = build_suggestion_scenario(
                suggestion=suggestion,
                suggestion_index=suggestion_index,
                linked_gap=linked_gap,
                linked_gap_id=suggestion_gap_id,
                target_region_signals=target_region_signals,
            )
            if suggestion_scenario:
                scenario_candidates.append(suggestion_scenario)

    if isinstance(applications, dict):
        ranked_clusters = sorted(
            [
                (theme, _support_size(support))
                for theme, support in applications.items()
                if isinstance(theme, str) and theme.strip()
            ],
            key=lambda item: item[1],
            reverse=True,
        )
        for theme, support in ranked_clusters:
            cluster_scenario = build_cluster_scenario(
                theme=theme,
                support=support,
                target_region_signals=target_region_signals,
            )
            if cluster_scenario:
                scenario_candidates.append(cluster_scenario)

    if not scenario_candidates:
        return []

    scenario_candidates.sort(
        key=lambda item: (
            item.get("source_rank", 99),
            -item.get("region_match_score", 0),
            -item.get("support_size", 0),
            item.get("name", ""),
        )
    )

    scenarios: List[Dict[str, Any]] = []
    seen_names: set = set()
    for candidate in scenario_candidates:
        scenario_name = str(candidate.get("name", "")).strip()
        if not scenario_name:
            continue
        dedupe_key = normalize_text(scenario_name)
        if dedupe_key in seen_names:
            continue
        seen_names.add(dedupe_key)

        scenario = dict(candidate)
        # 统一语义：priority 越大越优先
        scenario["priority"] = max(1, int(count) - int(len(scenarios)))
        scenarios.append(scenario)
        if len(scenarios) >= count:
            break

    return scenarios


# ── feedback action id ──────────────────────────────────────────────────

def get_feedback_dataset_action_id(action: Any) -> str:
    if not isinstance(action, dict):
        return ""
    return str(
        action.get("dataset_id")
        or action.get("requirement_id")
        or ""
    ).strip()


# ── replan validation ──────────────────────────────────────────────────

def required_dataset_signature(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    signature_fields = [
        item.get("name", ""),
        item.get("modality", ""),
        item.get("source_preference", ""),
        item.get("retrieval_hint", ""),
        item.get("temporal_requirement_level", ""),
        json.dumps(item.get("allowed_product_families", []) or [], ensure_ascii=False),
        json.dumps(item.get("aliases", []) or [], ensure_ascii=False),
        item.get("spatial_resolution", ""),
        json.dumps(
            item.get("temporal_coverage", {}) or {},
            ensure_ascii=False,
            sort_keys=True,
        ),
        json.dumps(
            item.get("temporal_resolution", {}) or {},
            ensure_ascii=False,
            sort_keys=True,
        ),
        item.get("execution_requirement_level", ""),
        item.get("binding_notes", item.get("notes", "")),
    ]
    normalized = [
        " ".join(re.findall(r"[a-z0-9]+", str(value or "").lower()))
        for value in signature_fields
    ]
    return "|".join(normalized)


def validate_replan_resolution(result: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    plan_refinement_feedback = context.get("plan_refinement_feedback") or {}
    resource_check_feedback = context.get("resource_check_feedback") or {}
    application_patch = result.get("application_patch") or {}
    task_plan = result.get("task_plan", []) or []
    required_datasets = result.get("required_datasets", []) or []
    if isinstance(application_patch, dict):
        task_plan = application_patch.get("task_plan", task_plan) or []
        required_datasets = application_patch.get("required_datasets", required_datasets) or []

    if not plan_refinement_feedback and not resource_check_feedback:
        return {"checked": False, "passed": True, "issues": []}

    issues: List[str] = []
    previous_task_plan = context.get("previous_task_plan") or []
    if previous_task_plan and task_plan == previous_task_plan:
        issues.append("task_plan unchanged after replanning feedback")

    step_failures = resource_check_feedback.get("step_failures", []) or []
    for failure in step_failures:
        if not isinstance(failure, dict):
            continue
        step_id = failure.get("step_id")
        suggested_task_type = failure.get("suggested_task_type") or resource_check_feedback.get("suggested_task_type")
        suggested_conversion_task = failure.get("suggested_conversion_task") or resource_check_feedback.get("suggested_conversion_task")
        matching_steps = [
            step for step in task_plan
            if isinstance(step, dict) and str(step.get("id")) == str(step_id)
        ]
        if suggested_task_type and matching_steps:
            matched_types = {step.get("task_type") for step in matching_steps}
            if suggested_task_type not in matched_types:
                issues.append(f"step {step_id} missing suggested_task_type={suggested_task_type}")
        if suggested_conversion_task:
            has_conversion = any(
                isinstance(step, dict) and step.get("task_type") == suggested_conversion_task
                for step in task_plan
            )
            if not has_conversion:
                issues.append(f"missing suggested_conversion_task={suggested_conversion_task}")

    weakest_step_id = resource_check_feedback.get("weakest_step_id")
    if weakest_step_id is not None:
        weakest_present = any(
            isinstance(step, dict) and str(step.get("id")) == str(weakest_step_id)
            for step in task_plan
        )
        if not weakest_present and not step_failures:
            issues.append(f"weakest_step_id={weakest_step_id} unresolved in replanned task_plan")

    dataset_actions = plan_refinement_feedback.get("dataset_actions", []) or []
    previous_required_datasets = context.get("previous_required_datasets") or []
    previous_required_lookup = {
        get_required_dataset_id(item): item
        for item in previous_required_datasets
        if isinstance(item, dict)
    }
    current_required_lookup = {
        get_required_dataset_id(item): item
        for item in required_datasets
        if isinstance(item, dict)
    }
    dataset_resolution_failures = resource_check_feedback.get("dataset_resolution_failures", []) or []
    for failure in dataset_resolution_failures:
        if not isinstance(failure, dict):
            continue
        dataset_id = str(failure.get("requirement_id", "")).strip()
        if not dataset_id:
            continue
        current_item = current_required_lookup.get(dataset_id)
        previous_item = previous_required_lookup.get(dataset_id, {})
        if not isinstance(current_item, dict):
            continue
        blocking_reasons = [
            str(reason).strip()
            for reason in (failure.get("blocking_reasons", []) or [])
            if str(reason).strip()
        ]
        previous_sig = required_dataset_signature(previous_item)
        current_sig = required_dataset_signature(current_item)
        if "region_conflict" in blocking_reasons and previous_sig and previous_sig == current_sig:
            issues.append(
                f"required_dataset {dataset_id} unchanged after region_conflict feedback"
            )
        if (
            any(
                reason.startswith(("family_", "time_", "region_"))
                or reason == "region_conflict"
                for reason in blocking_reasons
            )
            and previous_sig
            and previous_sig == current_sig
        ):
            issues.append(
                f"required_dataset {dataset_id} did not write family/time/region constraints back into structured fields"
            )
    for action in dataset_actions:
        if not isinstance(action, dict):
            continue
        dataset_id = str(action.get("dataset_id", "")).strip()
        if not dataset_id:
            continue
        current_item = current_required_lookup.get(dataset_id)
        if current_item is None:
            issues.append(f"required_dataset {dataset_id} missing after replanning feedback")
            continue
        if action.get("action") == "replace_with_correct_dataset":
            previous_item = previous_required_lookup.get(dataset_id, {})
            previous_sig = required_dataset_signature(previous_item)
            current_sig = required_dataset_signature(current_item)
            if previous_sig and previous_sig == current_sig:
                issues.append(
                    f"required_dataset {dataset_id} unchanged after replace_with_correct_dataset feedback"
                )

    if not issues:
        return {"checked": True, "passed": True, "issues": []}

    dataset_missing_issues = [i for i in issues if "missing after replanning feedback" in i]
    other_issues = [i for i in issues if "missing after replanning feedback" not in i]
    if not other_issues and dataset_missing_issues:
        logger.warning(
            "validate_replan_resolution: degrading on dataset_id_missing issues: %s",
            dataset_missing_issues,
        )
        return {"checked": True, "passed": True, "degraded": True, "issues": issues}

    return {"checked": True, "passed": False, "issues": issues}

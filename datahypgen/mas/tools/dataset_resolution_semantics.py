import calendar
import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple


FROZEN_IDENTITY_MATCH_ANY = "any"
FROZEN_IDENTITY_MATCH_ALL = "all"
FROZEN_IDENTITY_MATCH_MODES = (FROZEN_IDENTITY_MATCH_ANY, FROZEN_IDENTITY_MATCH_ALL)
FROZEN_IDENTITY_DIMENSIONS = ("entity", "observable", "measure")


def normalize_frozen_identity_dimension(value: Any) -> Tuple[List[str], str]:
    """把一个 frozen identity 维度归一为 (values, match_mode)。

    接受三种写法：裸字符串、字符串列表、`{"values": [...], "match": "any"|"all"}`。
    前两种缺省为 `any`，因为同一维度的多个取值通常是同义写法或互斥备选源，
    逐项 AND 会让命中正确项的候选被同义项否掉。只有复合量（例如气候水平衡
    = 降水 - 蒸散）才需要显式 `match="all"` 逐项校验。

    非法 `match` 值按 `any` 处理：这里是纯读取归一化，不是合同校验点；
    StepPlan 的 `_validate_required_datasets` 负责拒绝非法枚举值。
    """
    if value is None:
        return [], FROZEN_IDENTITY_MATCH_ANY

    match_mode = FROZEN_IDENTITY_MATCH_ANY
    if isinstance(value, dict):
        raw_values = value.get("values")
        declared_match = str(value.get("match", "") or "").strip().lower()
        if declared_match == FROZEN_IDENTITY_MATCH_ALL:
            match_mode = FROZEN_IDENTITY_MATCH_ALL
    else:
        raw_values = value

    if raw_values is None:
        return [], match_mode
    if isinstance(raw_values, (str, bytes)):
        raw_values = [raw_values]
    if not isinstance(raw_values, (list, tuple, set)):
        return [], match_mode

    values: List[str] = []
    for item in raw_values:
        text = str(item or "").strip()
        if text and text not in values:
            values.append(text)
    return values, match_mode


_TEMPORAL_UNIT_SECONDS = {
    "hour": 3600.0,
    "day": 86400.0,
    "month": 2629800.0,
    "year": 31557600.0,
}


def _parse_iso_date(value: Any, *, bound: str = "start") -> tuple[Optional[date], bool]:
    """把一个区间端点解析为日粒度日期。

    `bound` 决定年/年月粒度的补齐方向：`start` 取该年或该月的首日，`end` 取末日。
    方向不能省略——`end="2023"` 表达的是"覆盖到 2023 年末"，若统一补成 2023-01-01
    会把区间收窄将近一年，让本来满足的需求或候选被误判成 incomplete。
    """
    text = str(value or "").strip()
    if not text:
        return None, False
    if text.casefold() == "present":
        return None, True
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date(), False
    except ValueError:
        pass

    # datetime.fromisoformat 不接受 `YYYY` 和 `YYYY-MM`（3.11 起只有 `YYYYMMDD`
    # 这类 basic format 可解析），但 LLM 和 catalog metadata 描述长年份序列时经常
    # 这样写（例如 FAO 农业统计的 1961-2023）。这里按端点方向补齐，只放宽解析粒度，
    # 不放宽任何时间门禁：补齐后的区间仍走同一套 classify_temporal_coverage 判定。
    # 只接受两位月份，与 ISO 8601 和 fromisoformat 保持一致，不猜 `2023-1` 这类写法。
    matched = re.fullmatch(r"(\d{4})(?:-(\d{2}))?", text)
    if not matched:
        return None, False
    year = int(matched.group(1))
    month_text = matched.group(2)
    is_end = str(bound).strip().casefold() == "end"
    try:
        if month_text is None:
            return (date(year, 12, 31) if is_end else date(year, 1, 1)), False
        month = int(month_text)
        if is_end:
            return date(year, month, calendar.monthrange(year, month)[1]), False
        return date(year, month, 1), False
    except ValueError:
        # 非法年月（0000、2023-13 等）保持不可解析，由调用方 fail closed。
        return None, False


def normalize_temporal_coverage(value: Any) -> Dict[str, Any]:
    """Normalize a structured temporal interval without interpreting prose."""
    if not isinstance(value, dict):
        return {}
    start, start_open = _parse_iso_date(value.get("start"), bound="start")
    end, end_open = _parse_iso_date(value.get("end"), bound="end")
    if start is None or start_open or (end is None and not end_open):
        return {}
    normalized = {
        "start": start.isoformat(),
        "end": "present" if end_open else end.isoformat(),
    }
    coverage_mode = str(value.get("coverage_mode") or "").strip()
    if coverage_mode:
        normalized["coverage_mode"] = coverage_mode
    return normalized


def classify_temporal_coverage(required: Any, candidate: Any) -> str:
    """Classify actual date coverage as full, overlap, incomplete, or unverified."""
    if not required:
        return "not_required"
    normalized_required = normalize_temporal_coverage(required)
    if not normalized_required:
        return "invalid_requirement"
    normalized_candidate = normalize_temporal_coverage(candidate)
    if not normalized_candidate:
        return "unverified"

    req_start, _ = _parse_iso_date(normalized_required["start"])
    req_end, req_open = _parse_iso_date(normalized_required["end"])
    cand_start, _ = _parse_iso_date(normalized_candidate["start"])
    cand_end, cand_open = _parse_iso_date(normalized_candidate["end"])
    req_end_value = date.max if req_open else req_end
    cand_end_value = date.max if cand_open else cand_end
    if req_start is None or req_end_value is None or cand_start is None or cand_end_value is None:
        return "unverified"
    if cand_end_value < req_start or cand_start > req_end_value:
        return "no_overlap"

    coverage_mode = normalized_required.get("coverage_mode")
    if coverage_mode == "overlap":
        return "overlap"
    if coverage_mode != "full_cover":
        return "invalid_requirement"
    if cand_start <= req_start and cand_end_value >= req_end_value:
        return "full_cover"
    return "incomplete"


def normalize_temporal_resolution(value: Any) -> Dict[str, Any]:
    """Normalize a structured cadence object."""
    if not isinstance(value, dict):
        return {}
    unit = str(value.get("unit") or "").strip().casefold()
    if unit.endswith("s"):
        unit = unit[:-1]
    interval = value.get("interval")
    if unit not in _TEMPORAL_UNIT_SECONDS or isinstance(interval, bool):
        return {}
    try:
        numeric_interval = float(interval)
    except (TypeError, ValueError):
        return {}
    if numeric_interval <= 0:
        return {}
    normalized: Dict[str, Any] = {
        "unit": unit,
        "interval": int(numeric_interval) if numeric_interval.is_integer() else numeric_interval,
    }
    if "allow_finer_resolution" in value:
        if not isinstance(value["allow_finer_resolution"], bool):
            return {}
        normalized["allow_finer_resolution"] = value["allow_finer_resolution"]
    return normalized


def classify_temporal_resolution(required: Any, candidate: Any) -> str:
    """Compare cadence, allowing explicitly aggregatable finer observations."""
    if not required:
        return "not_required"
    normalized_required = normalize_temporal_resolution(required)
    if not normalized_required:
        return "invalid_requirement"
    normalized_candidate = normalize_temporal_resolution(candidate)
    if not normalized_candidate:
        return "unverified"

    required_seconds = (
        _TEMPORAL_UNIT_SECONDS[normalized_required["unit"]]
        * float(normalized_required["interval"])
    )
    candidate_seconds = (
        _TEMPORAL_UNIT_SECONDS[normalized_candidate["unit"]]
        * float(normalized_candidate["interval"])
    )
    if candidate_seconds == required_seconds:
        return "exact"
    if (
        candidate_seconds < required_seconds
        and normalized_required.get("allow_finer_resolution") is True
    ):
        return "aggregatable"
    return "incompatible"

_PRODUCT_FAMILY_TERMS = {
    "mtbs": ["mtbs", "monitoring trends in burn severity", "burn severity mosaics"],
    "nlcd": ["nlcd", "annual nlcd", "national land cover database", "national land cover"],
    "copernicus": ["copernicus", "c3s", "sentinel"],
    "modis": ["modis", "mod13", "myd13", "mod09", "mod11", "mod14", "mcd64", "mcd12"],
    "era5": ["era5", "era5 land", "ecmwf reanalysis"],
    "landsat": ["landsat", "oli", "etm+", "tm", "landsat 5", "landsat 7", "landsat 8", "landsat 9"],
    "sentinel1": ["sentinel-1", "sentinel 1", "sar", "synthetic aperture radar"],
    "sentinel2": ["sentinel-2", "sentinel 2", "msi", "s2 sr"],
    "srtm": ["srtm", "shuttle radar topography", "digital elevation model", "dem", "elevation"],
    "gridmet": ["gridmet", "university of idaho gridded surface meteorological"],
    "terraclimate": ["terraclimate", "terra climate"],
    "daymet": ["daymet", "daily surface weather"],
    "chirps": ["chirps", "climate hazards group infrared precipitation"],
    "imerg": ["imerg", "gpm imerg", "integrated multi-satellite retrievals for gpm"],
    "prism": ["prism", "parameter elevation regression"],
    "viirs": ["viirs", "visible infrared imaging radiometer suite", "suomi npp"],
    "firms": ["firms", "fire information for resource management", "fire radiative power"],
    "drought_monitor": ["drought monitor", "usdm", "us drought monitor", "drought index"],
    "pdsi": ["pdsi", "palmer drought severity index", "palmer", "palmer drought"],
    "gldas": ["gldas", "global land data assimilation system"],
    "koppen_geiger": ["koppen geiger", "köppen geiger"],
    "thornthwaite": ["thornthwaite"],
    "modis_snow_cover": ["modis snow cover", "mod10", "myd10"],
    "cru_ts": ["cru ts", "climatic research unit time series"],
}

# 查询词同义词扩展表：将常用科学概念名称/缩写 映射到可提升检索命中的附加词
# 用途：在 _build_local_resolution_intent 中对 query_tokens 做扩展，提升召回率
QUERY_SYNONYM_EXPANSION: Dict[str, List[str]] = {
    # 植被指数
    "evi": ["enhanced vegetation index", "vegetation index", "mod13", "myd13"],
    "ndvi": ["normalized difference vegetation index", "vegetation index", "mod13"],
    "nbr": ["normalized burn ratio", "burn ratio", "burn severity"],
    "dnbr": ["delta normalized burn ratio", "burn severity", "fire severity"],
    "lai": ["leaf area index"],
    "savi": ["soil adjusted vegetation index"],
    # 遥感平台/传感器
    "modis": ["mod13", "mcd64", "mod09", "mcd12"],
    "viirs": ["suomi npp", "snpp", "fire radiative power"],
    "landsat": ["oli", "etm+", "landsat 8", "landsat 9"],
    "sentinel": ["sentinel-2", "sentinel-1", "copernicus"],
    # 火灾/燃烧
    "burn severity": ["mtbs", "dnbr", "fire severity", "burn severity mosaics"],
    "burned area": ["mcd64", "fire occurrence", "burn perimeter", "mtbs"],
    "fire perimeter": ["mtbs", "burned area boundary", "fire boundary"],
    # DEM/地形
    "dem": ["srtm", "digital elevation model", "elevation", "terrain"],
    "elevation": ["srtm", "dem", "digital elevation model"],
    "terrain": ["srtm", "dem", "slope", "aspect"],
    "slope": ["srtm", "terrain", "dem"],
    # 土地覆盖
    "land cover": ["nlcd", "lulc", "land use land cover", "classification"],
    "lulc": ["land cover", "land use", "nlcd", "classification"],
    "land use": ["nlcd", "lulc", "land cover"],
    # 气候/气象
    "precipitation": ["chirps", "gridmet", "prism", "era5", "rainfall", "rain"],
    "rainfall": ["chirps", "precipitation", "gridmet"],
    "temperature": ["era5", "gridmet", "daymet", "lst", "land surface temperature"],
    "lst": ["land surface temperature", "mod11", "thermal"],
    "evapotranspiration": ["terraclimate", "gridmet", "pet", "actual evapotranspiration"],
    "pet": ["potential evapotranspiration", "terraclimate"],
    "vpd": ["vapor pressure deficit", "gridmet", "terraclimate"],
    "wind speed": ["gridmet", "era5"],
    "solar radiation": ["gridmet", "terraclimate", "era5"],
    "snowpack": ["swe", "snow water equivalent", "daymet"],
    "swe": ["snow water equivalent", "snowpack"],
    # 干旱指数
    "pdsi": ["palmer drought severity index", "drought", "drought monitor"],
    "palmer drought severity index": ["pdsi", "drought", "drought monitor"],
    "palmer drought": ["pdsi", "drought index", "drought monitor"],
    "spei": ["standardized precipitation evapotranspiration index", "drought"],
    "spi": ["standardized precipitation index", "drought"],
    "drought index": ["pdsi", "spei", "spi", "drought monitor"],
    "drought": ["pdsi", "drought monitor", "spei", "usdm"],
    "drought monitor": ["usdm", "pdsi", "drought"],
    # 土壤
    "soil moisture": ["era5", "gridmet", "terraclimate"],
    "soil": ["soil moisture", "terraclimate"],
    # 大气/气体
    "aerosol": ["modis", "viirs", "aod", "atmospheric optical depth"],
    "aod": ["aerosol optical depth", "modis", "viirs"],
    "co2": ["carbon dioxide", "greenhouse gas"],
    # 水文
    "streamflow": ["nwm", "national water model", "usgs stream gauge"],
    "flood": ["inundation", "surface water", "jrc"],
    "surface water": ["jrc", "gsw", "global surface water"],
    # 土壤
    "soc": ["soil organic carbon", "soil carbon"],
}

_THEME_HINTS = {
    "burn_severity": ["burn severity", "burn", "severity", "mtbs", "dnbr", "nbr", "fire severity"],
    "dem": ["digital elevation model", "dem", "elevation model", "terrain model", "srtm", "elevation"],
    "land_cover": ["land cover", "classification", "lulc", "land use", "nlcd"],
    "ndvi": ["ndvi", "normalized difference vegetation index", "evi", "vegetation index"],
    "fire_weather": ["fire weather", "fire danger", "wildfire", "fire behavior"],
    "vegetation": ["vegetation", "evi", "ndvi", "lai", "plant", "greenness"],
    "drought": ["drought", "pdsi", "spei", "spi", "drought monitor", "palmer"],
    "precipitation": ["precipitation", "rainfall", "chirps", "rain"],
    "temperature": ["temperature", "lst", "land surface temperature", "thermal"],
    "fire": ["fire", "burned area", "burn severity", "wildfire", "mtbs", "firms"],
}

_REGION_TERMS = {
    "california": ["california", "calif"],
    "texas": ["texas", "tx"],
    "united_states": ["united states", "usa", "conus"],
    "north_america": ["north america"],
    "canada": ["canada"],
    "italy": ["italy"],
    "france": ["france"],
    "europe": ["europe"],
    "switzerland": ["switzerland", "swiss", "swisstopo"],
    "africa": ["africa", "african"],
    "west_africa": ["west africa"],
    "south_africa": ["south africa"],
    "south_america": ["south america"],
    "pan_amazonia": ["pan amazonia", "amazonia", "amazonia"],
    "colombia": ["colombia"],
    "chaco": ["chaco"],
    "atlantic_forest": ["atlantic forest"],
    "pampa": ["pampa"],
    "brazil": ["brazil"],
    "global": ["global", "worldwide", "world"],
}

_REGION_COMPAT_MATRIX = {
    "california": {"california", "united_states", "north_america"},
    "texas": {"texas", "united_states", "north_america"},
    "united_states": {"united_states", "north_america"},
    "north_america": {"north_america"},
    "canada": {"canada", "north_america"},
    "italy": {"italy"},
    "france": {"france"},
    "europe": {"europe"},
    "switzerland": {"switzerland", "europe"},
    "africa": {"africa"},
    "west_africa": {"west_africa", "africa"},
    "south_africa": {"south_africa", "africa"},
    "south_america": {"south_america"},
    "pan_amazonia": {"pan_amazonia", "south_america"},
    "colombia": {"colombia", "south_america"},
    "chaco": {"chaco", "south_america"},
    "atlantic_forest": {"atlantic_forest", "south_america"},
    "pampa": {"pampa", "south_america"},
    "brazil": {"brazil", "south_america"},
    "global": {"global"},
}

_TIME_SIGNATURES = {
    "time_series": ["time series", "timeseries", "temporal"],
    "annual": ["annual", "yearly"],
    "monthly": ["monthly"],
    "daily": ["daily"],
    "historical": ["historical", "history", "long term", "longterm"],
}

_CLIMATE_IDENTITY_TERMS = {
    "climate",
    "precipitation",
    "rainfall",
    "temperature",
    "meteorological",
    "weather",
    "terraclimate",
    "gridmet",
    "prism",
    "era5",
}

_LAND_COVER_IDENTITY_TERMS = {
    "land cover",
    "landcover",
    "land_cover",
    "land use land cover",
    "lulc",
    "nlcd",
    "worldcover",
}

_IDENTITY_CONCEPT_TERMS = {
    "climate_classification": {
        "climate classification",
        "climate zone",
        "climatic zone",
        "koppen geiger",
        "koppen",
        "trewartha",
    },
    "land_cover_classification": {
        "land cover",
        "landcover",
        "land use land cover",
        "lulc",
        "nlcd",
        "worldcover",
        "forest classification",
        "forest type",
    },
    "lithology_category": {
        "lithology",
        "lithological",
        "rock type",
        "geological formation",
        "geological map",
        "geology map",
        "glim",
    },
    "bedrock_depth": {
        "bedrock depth",
        "depth to bedrock",
    },
    "categorical_variable": {
        "categorical",
        "classification",
        "class",
    },
    "continuous_aggregate": {
        "weighted sum",
        "wsum",
        "weighted average",
    },
}

_PRODUCT_FAMILY_ALIASES = {
    "era5 land": "era5",
    "era5land": "era5",
    "gpm imerg": "imerg",
    "landsat 8": "landsat",
    "landsat 9": "landsat",
    "landsat 8 9": "landsat",
    "sentinel 1": "sentinel1",
    "sentinel 2": "sentinel2",
    "gldas noah": "gldas",
    "gldas clsm": "gldas",
    "koppen geiger": "koppen_geiger",
    "köppen geiger": "koppen_geiger",
}


def _normalize_text(value: Any) -> str:
    tokens = re.findall(r"[a-z0-9]+", str(value or "").lower())
    return " ".join(tokens)


def _contains_alias(normalized_text: str, alias: str) -> bool:
    normalized_alias = _normalize_text(alias)
    if not normalized_text or not normalized_alias:
        return False
    if normalized_text == normalized_alias:
        return True
    return f" {normalized_alias} " in f" {normalized_text} "


def _dedupe(values: List[str]) -> List[str]:
    deduped: List[str] = []
    seen = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _normalize_product_family(value: Any) -> str:
    normalized = _normalize_text(value)
    return _PRODUCT_FAMILY_ALIASES.get(normalized, normalized.replace(" ", "_"))


def is_climate_identity_text(value: Any) -> bool:
    normalized = _normalize_text(value)
    return any(_contains_alias(normalized, token) for token in _CLIMATE_IDENTITY_TERMS)


def is_land_cover_identity_text(value: Any) -> bool:
    normalized = _normalize_text(value)
    return any(_contains_alias(normalized, token) for token in _LAND_COVER_IDENTITY_TERMS)


def infer_identity_concepts(value: Any) -> List[str]:
    normalized = _normalize_text(value)
    return [
        concept
        for concept, aliases in _IDENTITY_CONCEPT_TERMS.items()
        if any(_contains_alias(normalized, alias) for alias in aliases)
    ]


def is_known_product_family(value: Any) -> bool:
    return _normalize_product_family(value) in _PRODUCT_FAMILY_TERMS


def _extract_time_terms(value: Any) -> List[str]:
    normalized_text = _normalize_text(value)
    matches = [
        label
        for label, aliases in _TIME_SIGNATURES.items()
        if any(_contains_alias(normalized_text, alias) for alias in aliases)
    ]
    return _dedupe(matches)


def extract_region_terms(query_text, explicit_region=None) -> List[str]:
    query_matches: List[str] = []
    normalized_query = _normalize_text(query_text)
    normalized_explicit = _normalize_text(explicit_region)

    for region, aliases in _REGION_TERMS.items():
        if any(_contains_alias(normalized_query, alias) for alias in aliases):
            query_matches.append(region)
        if any(_contains_alias(normalized_explicit, alias) for alias in aliases):
            query_matches.append(region)

    if explicit_region and not query_matches:
        normalized_region = normalize_region_name(explicit_region)
        if normalized_region:
            query_matches.append(normalized_region)

    return _dedupe(query_matches)


def normalize_region_name(value) -> Optional[str]:
    normalized_text = _normalize_text(value)
    if not normalized_text:
        return None

    for region, aliases in _REGION_TERMS.items():
        if any(_contains_alias(normalized_text, alias) for alias in aliases):
            return region

    return normalized_text.replace(" ", "_")


def classify_region_match(required_region, candidate_regions) -> Dict[str, Any]:
    normalized_required = normalize_region_name(required_region)
    normalized_candidates = _dedupe([
        normalized_candidate
        for candidate in (candidate_regions or [])
        for normalized_candidate in [normalize_region_name(candidate)]
        if normalized_candidate
    ])
    if not normalized_required:
        return {
            "status": "none",
            "required_region": None,
            "candidate_regions": normalized_candidates,
            "matched_regions": [],
            "conflicting_regions": [],
        }
    if not normalized_candidates:
        return {
            "status": "unspecified",
            "required_region": normalized_required,
            "candidate_regions": [],
            "matched_regions": [],
            "conflicting_regions": [],
        }
    if "global" in normalized_candidates:
        return {
            "status": "global",
            "required_region": normalized_required,
            "candidate_regions": normalized_candidates,
            "matched_regions": ["global"],
            "conflicting_regions": [],
        }

    compatible_regions = set(_REGION_COMPAT_MATRIX.get(normalized_required, {normalized_required}))
    compatible_regions.add(normalized_required)

    matched_regions = [
        region
        for region in normalized_candidates
        if region in compatible_regions or normalized_required in region or region in normalized_required
    ]
    conflicting_regions = [
        region
        for region in normalized_candidates
        if region not in matched_regions
    ]
    if conflicting_regions:
        return {
            "status": "conflict",
            "required_region": normalized_required,
            "candidate_regions": normalized_candidates,
            "matched_regions": matched_regions,
            "conflicting_regions": conflicting_regions,
        }
    if matched_regions:
        return {
            "status": "compatible",
            "required_region": normalized_required,
            "candidate_regions": normalized_candidates,
            "matched_regions": matched_regions,
            "conflicting_regions": [],
        }

    return {
        "status": "unspecified",
        "required_region": normalized_required,
        "candidate_regions": normalized_candidates,
        "matched_regions": [],
        "conflicting_regions": [],
    }


def detect_family_matches(intent, candidate_text) -> List[str]:
    normalized_text = _normalize_text(candidate_text)
    allowed_product_families = intent.get("allowed_product_families", []) if isinstance(intent, dict) else []
    candidate_tokens = set(normalized_text.split())
    matches: List[str] = []
    for family in allowed_product_families:
        normalized_family = _normalize_product_family(family)
        aliases = _PRODUCT_FAMILY_TERMS.get(normalized_family, [])
        if aliases and any(
            _contains_alias(normalized_text, alias) for alias in aliases
        ):
            matches.append(family)
            continue
        if aliases:
            continue
        family_tokens = set(_normalize_text(family).split())
        if family_tokens and family_tokens.issubset(candidate_tokens):
            matches.append(family)
    return matches


def detect_theme_match(intent, candidate_text) -> bool:
    theme = intent.get("theme") if isinstance(intent, dict) else None
    if not theme:
        return True

    normalized_text = _normalize_text(candidate_text)
    signatures = _THEME_HINTS.get(theme, [theme.replace("_", " ")])
    return any(_contains_alias(normalized_text, signature) for signature in signatures)


def infer_product_families(value: Any) -> List[str]:
    normalized_text = _normalize_text(value)
    matches = [
        family
        for family, aliases in _PRODUCT_FAMILY_TERMS.items()
        if any(_contains_alias(normalized_text, alias) for alias in aliases)
    ]
    return _dedupe(matches)


def infer_theme(value: Any) -> Optional[str]:
    normalized_text = _normalize_text(value)
    for theme, signatures in _THEME_HINTS.items():
        if any(_contains_alias(normalized_text, signature) for signature in signatures):
            return theme
    return None


def detect_type_match(intent, candidate_type) -> bool:
    expected_data_types = intent.get("expected_data_types", []) if isinstance(intent, dict) else []
    if not expected_data_types:
        return True

    normalized_type = _normalize_text(candidate_type).replace(" ", "_")
    return normalized_type in set(expected_data_types)


def detect_time_match(intent, candidate_metadata) -> bool:
    if not isinstance(intent, dict):
        return True

    time_terms = intent.get("time_terms") or _extract_time_terms(
        intent.get("normalized_query") or intent.get("raw_query") or ""
    )
    if not time_terms:
        return True

    candidate_type = _normalize_text(candidate_metadata.get("candidate_type", "")).replace(" ", "_")
    candidate_text = _normalize_text(
        candidate_metadata.get("candidate_text")
        or " ".join(
            str(part)
            for part in [
                candidate_metadata.get("title"),
                candidate_metadata.get("id"),
                " ".join(candidate_metadata.get("tags", []) or []),
                candidate_metadata.get("thematic_group"),
                candidate_metadata.get("temporal_resolution"),
                candidate_metadata.get("temporal_coverage"),
            ]
            if part
        )
    )

    if "time_series" in time_terms and candidate_type == "image_collection":
        return True

    for term in time_terms:
        signatures = _TIME_SIGNATURES.get(term, [])
        if any(_contains_alias(candidate_text, signature) for signature in signatures):
            return True

    return False


def detect_role_match(intent, candidate_metadata) -> bool:
    """
    Binding 层不再使用研究/分析角色语义；保留 API 供旧调用方导入，恒为 True。
    """
    return True


_OBSERVABLE_IDENTITY_PATTERNS = {
    "entity": {
        "crop": (r"\bcrops?\b", r"\bcrop\s+production\b", r"\bmaize\b", r"\bwheat\b", r"\brice\b"),
        "livestock": (r"\blivestock\b", r"\bcattle\b", r"\bsheep\b", r"\bgoats?\b", r"\bpoultry\b"),
    },
    "observable": {
        "yield": (r"\byields?\b", r"\bproduction\s+yield\b"),
        "headcount": (r"\bhead\s*counts?\b", r"\bpopulation\s+count\b", r"\bnumber\s+of\s+(?:animals|livestock)\b"),
    },
    "measure": {
        "rate_or_yield": (r"\byield\b", r"\btonnes?\s+per\s+ha\b", r"\bkg\s+per\s+ha\b"),
        "count": (r"\bhead\s*count\b", r"\bpopulation\s+count\b", r"\bnumber\s+of\b"),
    },
}


def infer_observable_identity(value: Any) -> Dict[str, List[str]]:
    """Infer a small, explicit observable identity used by strong-binding guards."""
    normalized_text = _normalize_text(value)
    identity: Dict[str, List[str]] = {}
    for dimension, labels in _OBSERVABLE_IDENTITY_PATTERNS.items():
        matches = [
            label
            for label, patterns in labels.items()
            if any(re.search(pattern, normalized_text) for pattern in patterns)
        ]
        identity[dimension] = _dedupe(matches)
    return identity


def evaluate_fail_close_rules(intent, candidate_metadata) -> Dict[str, Any]:
    candidate_text = candidate_metadata.get("candidate_text", "")
    candidate_type = candidate_metadata.get("candidate_type", "")
    candidate_regions = candidate_metadata.get("candidate_regions") or extract_region_terms(
        candidate_text,
        explicit_region=candidate_metadata.get("explicit_region"),
    )

    reasons: List[str] = []
    matched_families = detect_family_matches(intent, candidate_text)

    if intent.get("allowed_product_families") and not matched_families:
        reasons.append("family_mismatch:" + ",".join(intent.get("allowed_product_families", [])))

    theme = intent.get("theme")
    if theme and not detect_theme_match(intent, candidate_text):
        reasons.append(f"theme_mismatch:{theme}")

    expected_data_types = intent.get("expected_data_types", [])
    if expected_data_types and not detect_type_match(intent, candidate_type):
        reasons.append(
            f"data_type_mismatch:expected={expected_data_types},actual={candidate_type}"
        )

    if not detect_time_match(intent, candidate_metadata):
        reasons.append("time_mismatch")

    query_regions = intent.get("region_terms", []) if isinstance(intent, dict) else []
    region_scope = "none"
    if query_regions:
        region_results = [
            classify_region_match(required_region, candidate_regions)
            for required_region in query_regions
        ]
        conflicting_regions = sorted({
            region
            for result in region_results
            for region in result.get("conflicting_regions", [])
        })
        if conflicting_regions:
            reasons.append("region_conflict:" + ",".join(conflicting_regions))
            region_scope = "conflict"
        elif any(result.get("status") == "compatible" for result in region_results):
            region_scope = "compatible"
        elif any(result.get("status") == "global" for result in region_results):
            region_scope = "global"
        else:
            region_scope = "unspecified"

    confidence_flags = {
        "matched_families": matched_families,
        "region_scope": region_scope,
        "global_fallback": region_scope == "global",
        "time_match": "time_mismatch" not in reasons,
        "role_match": "role_mismatch" not in reasons,
    }
    return {
        "passed": len(reasons) == 0,
        "reasons": reasons,
        "confidence_flags": confidence_flags,
        "candidate_regions": candidate_regions,
    }


__all__ = [
    "extract_region_terms",
    "normalize_region_name",
    "classify_region_match",
    "infer_product_families",
    "infer_theme",
    "detect_family_matches",
    "detect_theme_match",
    "detect_type_match",
    "detect_time_match",
    "detect_role_match",
    "normalize_temporal_coverage",
    "classify_temporal_coverage",
    "normalize_temporal_resolution",
    "classify_temporal_resolution",
    "is_climate_identity_text",
    "is_land_cover_identity_text",
    "infer_identity_concepts",
    "infer_observable_identity",
    "is_known_product_family",
    "evaluate_fail_close_rules",
    "expand_query_synonyms",
    "QUERY_SYNONYM_EXPANSION",
]


def expand_query_synonyms(query_text: str) -> List[str]:
    """将查询文本中的关键词扩展为同义词 token 集合，提升召回率。

    对 query_text 做 lower+normalize 后，扫描 QUERY_SYNONYM_EXPANSION 的每个 key。
    若 key 出现在 query 中，则把对应的同义词 token 拼接到输出列表里。
    返回的额外 token 列表会被调用方并入 query_tokens，不替换原始 tokens。
    """
    normalized = _normalize_text(query_text)
    extra_tokens: List[str] = []
    for key, synonyms in QUERY_SYNONYM_EXPANSION.items():
        normalized_key = _normalize_text(key)
        if _contains_alias(normalized, normalized_key):
            for synonym in synonyms:
                for token in _normalize_text(synonym).split():
                    if token and token not in extra_tokens:
                        extra_tokens.append(token)
    return extra_tokens

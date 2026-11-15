import asyncio
import json
import logging
import math
import os
import re
import time
from html import unescape
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from urllib.parse import urlsplit, urlunsplit
from urllib.request import getproxies, proxy_bypass

import aiohttp
from aiohttp_socks import ProxyConnector
from shapely.geometry import shape, box, mapping

from .dataset_detail_enricher import DatasetDetailEnricher
from .dataset_index_db_client import DatasetIndexDBClient
from .dataset_source_clients import query_nasa_cmr, query_planetary_computer
from .pangaea_client import search_pangaea
from .dataset_resolution_semantics import (
    FROZEN_IDENTITY_MATCH_ALL,
    classify_temporal_coverage,
    classify_temporal_resolution,
    classify_region_match,
    detect_family_matches,
    detect_theme_match,
    detect_type_match,
    expand_query_synonyms,
    extract_region_terms,
    infer_product_families,
    infer_theme,
    normalize_frozen_identity_dimension,
    normalize_temporal_coverage,
    normalize_temporal_resolution,
)
from .global_coverage_evidence import (
    assess_global_coverage_evidence,
    bbox_implies_global,
)

logger = logging.getLogger(__name__)

MATCH_STOPWORDS = {
    "a",
    "an",
    "and",
    "data",
    "dataset",
    "e",
    "eg",
    "equivalent",
    "example",
    "for",
    "from",
    "g",
    "in",
    "not",
    "of",
    "or",
    "on",
    "specific",
    "the",
    "to",
}

# 数据集"修饰词"：描述建模方法 / 版本 / 时相 / 精度的词，罕见（IDF 高）但对数据集
# 身份无区分意义（如 WorldPop 的 "unconstrained"、"UN-adjusted"）。IDF 的固有弱点是
# 把"罕见"当"高判别力"，这类词会被误当主题词，导致正确候选（含 worldpop+population
# 但不含 unconstrained）覆盖不全。仅在主题词判定时剔除它们，不动全局 tokenize/召回。
DATASET_MODIFIER_TERMS = {
    "unconstrained", "constrained", "adjusted", "unadjusted", "estimated", "estimates",
    "estimate", "final", "run", "version", "product", "products", "gridded", "global",
    "regional", "national", "corrected", "harmonized", "reprocessed", "provisional",
    "preliminary", "interim", "climatology", "composite", "mosaic", "mosaics",
}

RECALL_MIN_SCORE = 0.18
WEAK_BIND_MIN_SCORE = 0.40
STRONG_BIND_MIN_SCORE = 0.65

# datasets.json 里只有 GEE 官方/社区两个源带 type 字段（image / image_collection /
# table）；STAC 源与 OSM tag 源的 type 是空串。缺 type 曾让 `_normalize_catalog_record`
# 直接丢弃记录——实测 11424 条 catalog 有 5139 条（NASA_CMR 全部 4996 条在内）
# 在召回前就消失了，检索侧根本看不见这些数据集。这里按源补出资产形态。
#
# STAC collection 语义上等价于 ImageCollection（一个 collection 下挂多个时相 item），
# 记为 image_collection；OSM tag 是元数据条目，保留 tag 这个独立形态。
SOURCE_DEFAULT_ASSET_TYPE: Dict[str, str] = {
    "NASA_CMR": "image_collection",
    "PLANETARY_COMPUTER": "image_collection",
    "EARTH_SEARCH": "image_collection",
    "OSM_TAGINFO": "tag",
}

# 走 Earth Engine 运行时的源。只有这两个源的 asset 能用 ee.Image / ee.ImageCollection
# 构造，其余源的 type 虽借用了同一套词汇，却必须走各自的客户端。
GEE_CATALOG_SOURCES = frozenset({"GEE_OFFICIAL", "GEE_COMMUNITY"})

# ---------------------------------------------------------------------------
# 方向 3：高频数据集 alias 索引
# 格式：{ alias_文本（小写） -> catalog_id }
# 命中时直接 resolved_strong，跳过 token-overlap 评分
# alias 覆盖 LLM 最常写错/写模糊的高频数据集名称
# ---------------------------------------------------------------------------
GEE_DATASET_ALIAS_INDEX: Dict[str, str] = {
    # MODIS 植被指数
    "modis evi": "MODIS/061/MOD13Q1",
    "modis evi 250m": "MODIS/061/MOD13Q1",
    "modis evi 500m": "MODIS/061/MOD13A1",
    "modis evi monthly": "MODIS/061/MOD13A3",
    "modis ndvi": "MODIS/061/MOD13Q1",
    "modis ndvi 250m": "MODIS/061/MOD13Q1",
    "modis vegetation index": "MODIS/061/MOD13Q1",
    "modis vegetation indices": "MODIS/061/MOD13Q1",
    "mod13q1": "MODIS/061/MOD13Q1",
    # AVHRR / long-term NDVI climate products
    "noaa cdr avhrr ndvi v5": "projects/sat-io/open-datasets/PKU-GIMMS-NDVI/AVHRR_SOLELY",
    "noaa avhrr ndvi climate record": "projects/sat-io/open-datasets/PKU-GIMMS-NDVI/AVHRR_SOLELY",
    "avhrr ndvi climate record": "projects/sat-io/open-datasets/PKU-GIMMS-NDVI/AVHRR_SOLELY",
    "avhrr ndvi cdr": "projects/sat-io/open-datasets/PKU-GIMMS-NDVI/AVHRR_SOLELY",
    "gimms ndvi avhrr": "projects/sat-io/open-datasets/PKU-GIMMS-NDVI/AVHRR_SOLELY",
    "myd13q1": "MODIS/061/MYD13Q1",
    "mod13a1": "MODIS/061/MOD13A1",
    "mod13a3": "MODIS/061/MOD13A3",
    # MODIS 地表温度
    "modis lst": "MODIS/061/MOD11A2",
    "land surface temperature": "MODIS/061/MOD11A2",
    "mod11a2": "MODIS/061/MOD11A2",
    "mod11a1": "MODIS/061/MOD11A1",
    # MODIS 土地覆盖
    "modis land cover": "MODIS/061/MCD12Q1",
    "mcd12q1": "MODIS/061/MCD12Q1",
    # MODIS 燃烧面积
    "modis burned area": "MODIS/061/MCD64A1",
    "mcd64a1": "MODIS/061/MCD64A1",
    "modis fire": "MODIS/061/MCD64A1",
    # Landsat
    "landsat 8": "LANDSAT/LC08/C02/T1_L2",
    "landsat 8 surface reflectance": "LANDSAT/LC08/C02/T1_L2",
    "landsat 9": "LANDSAT/LC09/C02/T1_L2",
    "landsat 7": "LANDSAT/LE07/C02/T1_L2",
    "landsat 5": "LANDSAT/LT05/C02/T1_L2",
    "landsat oli": "LANDSAT/LC08/C02/T1_L2",
    "landsat l2": "LANDSAT/LC08/C02/T1_L2",
    # Sentinel-2
    "sentinel-2": "COPERNICUS/S2_SR_HARMONIZED",
    "sentinel 2": "COPERNICUS/S2_SR_HARMONIZED",
    "sentinel-2 surface reflectance": "COPERNICUS/S2_SR_HARMONIZED",
    "sentinel-2 l2a": "COPERNICUS/S2_SR_HARMONIZED",
    "s2 sr": "COPERNICUS/S2_SR_HARMONIZED",
    # Sentinel-1
    "sentinel-1": "COPERNICUS/S1_GRD",
    "sentinel 1": "COPERNICUS/S1_GRD",
    "sentinel-1 sar": "COPERNICUS/S1_GRD",
    # SRTM
    "srtm": "USGS/SRTMGL1_003",
    "srtm 30m": "USGS/SRTMGL1_003",
    "srtm dem": "USGS/SRTMGL1_003",
    "srtm digital elevation": "USGS/SRTMGL1_003",
    "digital elevation model": "USGS/SRTMGL1_003",
    "dem 30m": "USGS/SRTMGL1_003",
    # Copernicus GLO-30 DEM
    "glo-30": "COPERNICUS/DEM/GLO30",
    "glo 30": "COPERNICUS/DEM/GLO30",
    "glo30": "COPERNICUS/DEM/GLO30",
    "copernicus dem": "COPERNICUS/DEM/GLO30",
    "copernicus dem glo-30": "COPERNICUS/DEM/GLO30",
    "copernicus glo30": "COPERNICUS/DEM/GLO30",
    "copernicus digital elevation model": "COPERNICUS/DEM/GLO30",
    # NLCD
    "nlcd": "projects/sat-io/open-datasets/USGS/ANNUAL_NLCD/LANDCOVER",
    "national land cover": "projects/sat-io/open-datasets/USGS/ANNUAL_NLCD/LANDCOVER",
    "national land cover database": "projects/sat-io/open-datasets/USGS/ANNUAL_NLCD/LANDCOVER",
    "nlcd land cover": "projects/sat-io/open-datasets/USGS/ANNUAL_NLCD/LANDCOVER",
    "usgs land cover": "projects/sat-io/open-datasets/USGS/ANNUAL_NLCD/LANDCOVER",
    # MTBS
    "mtbs": "projects/sat-io/open-datasets/MTBS/BURN_SEVERITY_MOSAICS",
    "mtbs burn severity": "projects/sat-io/open-datasets/MTBS/BURN_SEVERITY_MOSAICS",
    "burn severity mosaics": "projects/sat-io/open-datasets/MTBS/BURN_SEVERITY_MOSAICS",
    "monitoring trends in burn severity": "projects/sat-io/open-datasets/MTBS/BURN_SEVERITY_MOSAICS",
    "mtbs fire perimeter": "projects/sat-io/open-datasets/MTBS/burned_area_boundaries",
    "mtbs burned area boundary": "projects/sat-io/open-datasets/MTBS/burned_area_boundaries",
    # GRIDMET
    "gridmet": "IDAHO_EPSCOR/GRIDMET",
    "gridmet climate": "IDAHO_EPSCOR/GRIDMET",
    "university of idaho gridded surface meteorological": "IDAHO_EPSCOR/GRIDMET",
    # DAYMET
    "daymet": "NASA/ORNL/DAYMET_V4",
    "daymet v4": "NASA/ORNL/DAYMET_V4",
    "daymet climate": "NASA/ORNL/DAYMET_V4",
    # ERA5
    "era5": "ECMWF/ERA5_LAND/MONTHLY_AGGR",
    "era5 land": "ECMWF/ERA5_LAND/MONTHLY_AGGR",
    "era5 daily": "ECMWF/ERA5/DAILY",
    "ecmwf era5": "ECMWF/ERA5_LAND/MONTHLY_AGGR",
    # 干旱监测
    "us drought monitor": "projects/sat-io/open-datasets/us-drought-monitor",
    "usdm": "projects/sat-io/open-datasets/us-drought-monitor",
    "drought monitor": "projects/sat-io/open-datasets/us-drought-monitor",
    "united states drought monitor": "projects/sat-io/open-datasets/us-drought-monitor",
    # Ecoregions
    "ecoregions": "EPA/Ecoregions/2013/L4",
    "usgs ecoregions": "EPA/Ecoregions/2013/L4",
    "epa ecoregions": "EPA/Ecoregions/2013/L4",
    # PDSI 相关（无精确对应，指向最近的替代）
    "pdsi": "projects/sat-io/open-datasets/us-drought-monitor",
    "palmer drought severity index": "projects/sat-io/open-datasets/us-drought-monitor",
    "palmer drought": "projects/sat-io/open-datasets/us-drought-monitor",
}

# 区域/时间匹配的记分表。
#
# 设计原则：匹配上给正分（证据加分），缺证据给中性 0，只有确凿不符才扣分。
# 原表把 region="unspecified"、time="unknown" 各记 -0.05，也就是把"没有证据"
# 当成"不匹配"来罚。实测这两项对全部候选恒定生效：一次侦察的 24 个候选
# （12 入池 + 12 近失）惩罚额取值集合就是 {0.1} 一个值——它不提供任何区分力，
# 只是给所有人整体降 0.1 分，却恰好把 12 个候选压在 strong 阈值 0.65 之下
# （它们的 text_score 是 0.723~0.748）。这与 operation_semantics 里
# "缺证据 ≠ 正向矛盾" 是同一条原则，只是此前藏在惩罚项里而非判级逻辑里。
#
# time="partial" 保留 -0.08：那是有证据的真实时间范围不足，不是未知。
REGION_SCORE_TABLE = {
    "exact": 0.15,
    "compatible_parent": 0.08,
    "global": 0.03,
    "unspecified": 0.0,
}

TIME_SCORE_TABLE = {
    "exact": 0.0,
    "partial": -0.08,
    "unknown": 0.0,
}

@dataclass
class GeoFeature:
    """标准化地理特征数据类 """
    feature_id: str
    name: str
    data_description: str # 数据描述
    feature_type: str  # e.g., 'building', 'road', 'river'
    geometry: Dict     # GeoJSON format
    properties: Dict   # 属性信息
    source: str        # 数据来源

class GeoDataSearch:
    """
    地理数据检索工具，支持多源空间查询
    """
    _catalog_cache: Optional[List[GeoFeature]] = None
    # asset_id -> 本地 catalog 的 GEE asset 类型（image/image_collection/table）
    _gee_catalog_type_index_cache: Optional[Dict[str, str]] = None
    # token -> IDF 权重，基于本地 catalog title/id/tags 语料一次性构建（懒加载 + 类级缓存）
    _token_idf_cache: Optional[Dict[str, float]] = None
    _token_idf_default: float = 1.0
    _ONLINE_GEE_CATALOG_CACHE_KEY = "__catalog__"

    def __init__(self, cache_enabled: bool = True, use_db: bool = True):
        self.cache_enabled = cache_enabled
        self.use_db = use_db
        self._db_client: Optional[DatasetIndexDBClient] = None
        self._cache: Dict[str, List[GeoFeature]] = {} #
        self.session: Optional[aiohttp.ClientSession] = None  # kept for backward compat, no longer used
        self._online_gee_catalog_cache: Dict[str, List[GeoFeature]] = {}
        self._online_gee_catalog_last_failure_at: Optional[float] = None
        self._dataset_detail_enricher = DatasetDetailEnricher()
        geo_settings = self._load_geo_data_search_settings()
        self.region_hard_filter = bool(geo_settings.get("region_hard_filter", True))
        self.time_hard_filter = bool(geo_settings.get("time_hard_filter", True))
        self.role_hard_filter = bool(geo_settings.get("role_hard_filter", True))
        self.recall_min_score = float(geo_settings.get("recall_min_score", RECALL_MIN_SCORE) or RECALL_MIN_SCORE)
        self.weak_bind_min_score = float(
            geo_settings.get("weak_bind_min_score", WEAK_BIND_MIN_SCORE) or WEAK_BIND_MIN_SCORE
        )
        self.strong_bind_min_score = float(
            geo_settings.get("strong_bind_min_score", STRONG_BIND_MIN_SCORE) or STRONG_BIND_MIN_SCORE
        )
        self.online_top_k = int(geo_settings.get("online_top_k", 50) or 50)
        self.online_gee_catalog_enabled = bool(geo_settings.get("online_gee_catalog_enabled", True))
        self.online_gee_catalog_url = str(
            geo_settings.get(
                "online_gee_catalog_url",
                "https://developers.google.com/earth-engine/datasets/catalog",
            ) or "https://developers.google.com/earth-engine/datasets/catalog"
        ).strip()
        self.online_gee_catalog_timeout_seconds = int(
            geo_settings.get("online_gee_catalog_timeout_seconds", 20) or 20
        )

    @property
    def db_client(self) -> Optional[DatasetIndexDBClient]:
        """获取数据库客户端（延迟初始化）。"""
        if not self.use_db:
            return None
        if self._db_client is None:
            try:
                data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
                db_path = os.path.join(data_dir, "datasets.db")
                if os.path.exists(db_path):
                    self._db_client = DatasetIndexDBClient(db_path)
                    logger.info(f"Using dataset index database at {db_path}")
                else:
                    logger.warning(
                        f"Dataset index database not found at {db_path}. "
                        f"Falling back to JSON file. "
                        f"Run 'python -m datahypgen.mas.tools.build_dataset_index_db' to build the database."
                    )
            except Exception as e:
                logger.warning(f"Failed to initialize database client: {e}. Falling back to JSON file.")
        return self._db_client
        self.online_gee_catalog_failure_ttl_seconds = int(
            geo_settings.get("online_gee_catalog_failure_ttl_seconds", 300) or 300
        )

    def _format_exception_for_log(self, exc: Exception) -> str:
        exc_type = type(exc).__name__
        exc_message = str(exc).strip()
        if exc_message:
            return f"{exc_type}: {exc_message}"
        return exc_type

    def _online_gee_session_kwargs(self) -> Dict[str, Any]:
        session_kwargs: Dict[str, Any] = {"trust_env": True}
        target = urlsplit(self.online_gee_catalog_url)
        if not target.hostname or proxy_bypass(target.hostname):
            return session_kwargs

        proxies = getproxies()
        proxy_url = proxies.get(target.scheme.lower()) or proxies.get("all")
        if not proxy_url:
            return session_kwargs

        parsed_proxy = urlsplit(proxy_url)
        proxy_scheme = parsed_proxy.scheme.lower()
        if proxy_scheme not in {"socks4", "socks4a", "socks5", "socks5h"}:
            return session_kwargs

        normalized_scheme = "socks4" if proxy_scheme.startswith("socks4") else "socks5"
        normalized_proxy_url = urlunsplit(parsed_proxy._replace(scheme=normalized_scheme))
        session_kwargs["connector"] = ProxyConnector.from_url(
            normalized_proxy_url,
            rdns=proxy_scheme in {"socks4a", "socks5h"},
        )
        session_kwargs["trust_env"] = False
        return session_kwargs

    def _load_geo_data_search_settings(self) -> Dict[str, Any]:
        settings: Dict[str, Any] = {
            "region_hard_filter": True,
            "time_hard_filter": True,
            "role_hard_filter": True,
            "recall_min_score": RECALL_MIN_SCORE,
            "weak_bind_min_score": WEAK_BIND_MIN_SCORE,
            "strong_bind_min_score": STRONG_BIND_MIN_SCORE,
            "use_database": True,  # 默认使用数据库加速查询
        }
        try:
            import yaml  # type: ignore

            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
            config_candidates = [
                os.path.join(project_root, "config", "config.yaml"),
                os.path.join(project_root, "config", "default_config.yaml"),
                os.path.join(project_root, "config", "debug_config.yaml"),
                os.path.join(project_root, "config", "config", "config.yaml"),
                os.path.join(project_root, "config", "config", "default_config.yaml"),
                os.path.join(project_root, "config", "config", "debug_config.yaml"),
            ]
            for config_path in config_candidates:
                if not os.path.exists(config_path):
                    continue
                with open(config_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                geo_data_search_settings = (((data.get("tools") or {}).get("geo_data_search")) or {})
                if not isinstance(geo_data_search_settings, dict):
                    continue
                settings.update({
                    key: value
                    for key, value in geo_data_search_settings.items()
                    if value is not None
                })
                break
        except Exception as e:
            logger.warning(f"Failed to load geo_data_search settings, using defaults: {e}")

        return settings

    def _load_configured_data_sources(self) -> List[str]:
        env_sources = os.getenv("GEO_DATA_SOURCES", "").strip()
        if env_sources:
            return [s.strip() for s in env_sources.split(",") if s.strip()]

        try:
            import yaml  # type: ignore

            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
            config_candidates = [
                os.path.join(project_root, "config", "config.yaml"),
                os.path.join(project_root, "config", "default_config.yaml"),
                os.path.join(project_root, "config", "config", "config.yaml"),
                os.path.join(project_root, "config", "config", "default_config.yaml"),
            ]
            for config_path in config_candidates:
                if not os.path.exists(config_path):
                    continue
                with open(config_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                sources = (((data.get("tools") or {}).get("geo_data_search") or {}).get("data_sources") or [])
                if isinstance(sources, list) and sources:
                    return [str(s).strip() for s in sources if str(s).strip()]
        except Exception as e:
            logger.warning(f"Failed to load configured geo data sources, using defaults: {e}")

        return ["OpenStreetMap", "Natural Earth"]

    async def _get_all_available_metadata(self) -> List[GeoFeature]:
        """汇总可用数据源的元数据候选，供关键词相关性检索使用。"""
        try:
            configured_sources = self._load_configured_data_sources()
            source_set = {s.lower() for s in configured_sources}

            catalog: List[GeoFeature] = []
            if "openstreetmap" in source_set:
                catalog.extend([
                    GeoFeature(
                        feature_id="osm_facilities",
                        name="OpenStreetMap POI Facilities",
                        data_description="OpenStreetMap points of interest, amenity facilities, and urban infrastructure features.",
                        feature_type="poi",
                        geometry={"type": "GeometryCollection", "geometries": []},
                        properties={"relevance_score": 0.0},
                        source="OpenStreetMap",
                    ),
                    GeoFeature(
                        feature_id="osm_transport",
                        name="OpenStreetMap Transport Network",
                        data_description="OpenStreetMap road network and transport topology for geospatial accessibility analysis.",
                        feature_type="transport",
                        geometry={"type": "GeometryCollection", "geometries": []},
                        properties={"relevance_score": 0.0},
                        source="OpenStreetMap",
                    ),
                ])

            if "natural earth" in source_set:
                catalog.extend([
                    GeoFeature(
                        feature_id="ne_boundaries",
                        name="Natural Earth Administrative Boundaries",
                        data_description="Natural Earth country, state, and administrative boundaries for regional geospatial analysis.",
                        feature_type="boundary",
                        geometry={"type": "GeometryCollection", "geometries": []},
                        properties={"relevance_score": 0.0},
                        source="Natural Earth",
                    ),
                    GeoFeature(
                        feature_id="ne_hydrography",
                        name="Natural Earth Hydrography",
                        data_description="Natural Earth rivers and lakes for environmental and hydrological context analysis.",
                        feature_type="hydrography",
                        geometry={"type": "GeometryCollection", "geometries": []},
                        properties={"relevance_score": 0.0},
                        source="Natural Earth",
                    ),
                ])

            if "gee" in source_set or "earthengine" in source_set or "google earth engine" in source_set:
                catalog.extend(self._load_catalog())

            # 在线 GEE catalog：不依赖 data_sources 配置，只受 online_gee_catalog_enabled 控制
            # 已有 feature_id 集合用于去重（本地 catalog 优先）
            existing_ids = {f.feature_id for f in catalog}
            online_features = await self._load_online_gee_catalog()
            for feature in online_features:
                if feature.feature_id not in existing_ids:
                    catalog.append(feature)
                    existing_ids.add(feature.feature_id)

            # 合并缓存中的实际查询结果，避免丢失在线检索到的候选
            for value in self._cache.values():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, GeoFeature):
                            catalog.append(item)

            return catalog
        except Exception as e:
            logger.warning(f"Failed to collect geo metadata candidates: {e}")
            return []

    async def _get_session(self) -> aiohttp.ClientSession:
        # Deprecated: callers now create their own session via async with
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    def _map_gee_asset_type_to_ee_object_type(self, asset_type: str) -> str:
        """将 catalog type 映射为可直接用于 ee 的对象类型。"""
        normalized = (asset_type or "").strip().lower()

        if normalized == "image":
            return "Image"
        if normalized in {"image_collection", "imagecollection"}:
            return "ImageCollection"
        if normalized in {"table", "feature_collection", "featurecollection"}:
            return "FeatureCollection"

        return "Unknown"

    def _map_gee_asset_type_to_ee_ctor(self, asset_type: str) -> Optional[str]:
        """将 community catalog type 映射为 ee 构造器。"""
        ee_object_type = self._map_gee_asset_type_to_ee_object_type(asset_type)
        if ee_object_type == "Image":
            return "ee.Image"
        if ee_object_type == "ImageCollection":
            return "ee.ImageCollection"
        if ee_object_type == "FeatureCollection":
            return "ee.FeatureCollection"
        return None

    def _normalize_catalog_record(self, dataset: Dict[str, Any]) -> Optional[GeoFeature]:
        """将一条 datasets.json 记录标准化为统一的 GeoFeature 结构。"""
        if not isinstance(dataset, dict):
            return None

        asset_id = str(dataset.get("id", "")).strip()
        title = str(dataset.get("title", asset_id)).strip()
        catalog_source = str(dataset.get("source") or "").strip().upper()
        asset_type = str(dataset.get("type", "")).strip().lower()
        if not asset_type:
            asset_type = SOURCE_DEFAULT_ASSET_TYPE.get(catalog_source, "")
        if not asset_id or not title or not asset_type:
            return None

        tags = dataset.get("tags", [])
        if not tags:
            tags = dataset.get("keywords", [])
        if not isinstance(tags, list):
            tags = re.split(r"[,;]", str(tags))
        normalized_tags = [str(tag).strip() for tag in tags if str(tag).strip()]

        provider = str(dataset.get("provider", "")).strip()
        license_name = str(dataset.get("license", "")).strip()
        thematic_group = str(
            dataset.get("thematic_group") or dataset.get("category") or ""
        ).strip()
        region_name = str(dataset.get("region_name", "")).strip()
        bbox = dataset.get("bbox")
        docs = str(dataset.get("docs") or dataset.get("doc_url") or "").strip()
        sample_code = str(
            dataset.get("sample_code") or dataset.get("sample_code_url") or ""
        ).strip()
        # ee 运行时信息只对 GEE 源有意义。给 STAC/OSM 记录填上 ee_ctor 会让下游
        # 拿 ee.ImageCollection 去构造一个 Earth Engine 里并不存在的 asset，
        # 探测必然失败——非 GEE 源必须显式留空，由各自的客户端负责。
        is_gee_source = catalog_source in GEE_CATALOG_SOURCES or not catalog_source
        if is_gee_source:
            ee_object_type = self._map_gee_asset_type_to_ee_object_type(asset_type)
            ee_ctor = self._map_gee_asset_type_to_ee_ctor(asset_type)
            ee_id = asset_id
        else:
            ee_object_type = ""
            ee_ctor = None
            ee_id = ""

        raw_description = str(dataset.get("description") or "").strip()
        description_parts = [
            title,
            asset_id,
            asset_type,
            ee_object_type,
            provider,
            thematic_group,
            license_name,
            " ".join(normalized_tags),
            raw_description,
        ]
        description = " ".join(part for part in description_parts if part)

        return GeoFeature(
            feature_id=asset_id,
            name=title,
            data_description=description,
            feature_type=asset_type,
            geometry={"type": "GeometryCollection", "geometries": []},
            properties={
                "title": title,
                "id": asset_id,
                "type": asset_type,
                "ee_id": ee_id,
                "ee_object_type": ee_object_type,
                "ee_ctor": ee_ctor,
                "provider": provider,
                "tags": normalized_tags,
                "docs": docs,
                "sample_code": sample_code,
                "license": license_name,
                "thematic_group": thematic_group,
                "region_name": region_name,
                "bbox": bbox,
                "description": raw_description,
                "start_date": str(
                    dataset.get("start_date")
                    or dataset.get("temporal_start")
                    or dataset.get("state_date")
                    or ""
                ).strip(),
                "end_date": str(
                    dataset.get("end_date") or dataset.get("temporal_end") or ""
                ).strip(),
                "source": str(dataset.get("source") or "").strip(),
                "stac_url": str(dataset.get("stac_url") or "").strip(),
                "thumbnail_url": str(dataset.get("thumbnail_url") or "").strip(),
                "terms_of_use": str(dataset.get("terms_of_use") or "").strip(),
                # 弃用状态是 soft 信息：只透传，不参与过滤与打分。下游据此提示
                # 官方继任者（replacement_id / replacement_in_catalog），可用性
                # 仍由时间合同门禁按 start_date / end_date 判定。
                "deprecated": bool(dataset.get("deprecated", False)),
                "deprecation": dataset.get("deprecation") or None,
            },
            # 前缀必须反映真实来源。此前对所有记录一律写 "GEE:{id}"，
            # 于是下游 `"GEE" in source.upper()` 恒真，非 GEE 源永远被路由到
            # Earth Engine 探测分支，外部源的 manifest 分支成了不可达代码。
            source=f"{catalog_source or 'GEE'}:{asset_id}",
        )

    def _normalize_online_gee_catalog_record(self, dataset: Dict[str, Any]) -> Optional[GeoFeature]:
        if not isinstance(dataset, dict):
            return None

        asset_id = str(dataset.get("id", "")).strip()
        title = str(dataset.get("title", asset_id)).strip()
        asset_type = str(dataset.get("type", "")).strip().lower()
        if not asset_id or not title or not asset_type:
            return None

        tags = dataset.get("tags", [])
        if not isinstance(tags, list):
            tags = [str(tags)]
        normalized_tags = [str(tag).strip() for tag in tags if str(tag).strip()]

        provider = str(dataset.get("provider", "Google Earth Engine Catalog")).strip()
        thematic_group = str(dataset.get("thematic_group", "")).strip()
        docs = str(dataset.get("docs", "")).strip()
        sample_code = str(dataset.get("sample_code", "")).strip()
        description = str(dataset.get("description", "")).strip()
        license_name = str(dataset.get("license", "")).strip()
        ee_object_type = self._map_gee_asset_type_to_ee_object_type(asset_type)
        ee_ctor = self._map_gee_asset_type_to_ee_ctor(asset_type)
        region_name = str(dataset.get("region_name", "")).strip()
        bbox = dataset.get("bbox")

        description_parts = [
            title,
            asset_id,
            asset_type,
            ee_object_type,
            provider,
            thematic_group,
            description,
            " ".join(normalized_tags),
        ]

        return GeoFeature(
            feature_id=asset_id,
            name=title,
            data_description=" ".join(part for part in description_parts if part),
            feature_type=asset_type,
            geometry={"type": "GeometryCollection", "geometries": []},
            properties={
                "title": title,
                "id": asset_id,
                "type": asset_type,
                "ee_id": asset_id,
                "ee_object_type": ee_object_type,
                "ee_ctor": ee_ctor,
                "provider": provider,
                "tags": normalized_tags,
                "docs": docs,
                "sample_code": sample_code,
                "license": license_name,
                "thematic_group": thematic_group,
                "region_name": region_name,
                "bbox": bbox,
                "description": description,
                "start_date": str(
                    dataset.get("start_date") or dataset.get("state_date") or ""
                ).strip(),
                "end_date": str(dataset.get("end_date") or "").strip(),
            },
            source=f"GEE_ONLINE:{asset_id}",
        )

    def _get_local_catalog_type_index(self) -> Dict[str, str]:
        """构建并缓存 asset_id -> 本地 catalog asset 类型 的索引。"""
        if type(self)._gee_catalog_type_index_cache is not None:
            return type(self)._gee_catalog_type_index_cache

        type_by_id: Dict[str, str] = {}
        for feature in self._load_catalog():
            feature_id = str(feature.feature_id or "").strip()
            local_type = str((feature.properties or {}).get("type") or "").strip().lower()
            if feature_id and local_type:
                type_by_id[feature_id] = local_type

        type(self)._gee_catalog_type_index_cache = type_by_id
        return type_by_id

    def get_local_catalog_asset_type(self, asset_id: str) -> Optional[str]:
        """按 Earth Engine asset id 反查本地 catalog 声明的 asset 类型。

        返回 datasets.json 中 GEE 条目的原始 type（image / image_collection / table），
        未收录该 asset 时返回 None——调用方不得据此臆断类型。
        """
        normalized_id = str(asset_id or "").strip()
        if not normalized_id:
            return None
        return self._get_local_catalog_type_index().get(normalized_id)

    def _trust_local_catalog_type(self, online_features: List[GeoFeature]) -> List[GeoFeature]:
        """用本地 catalog 的 asset type 覆盖在线 catalog 按 ID 推断的 type。

        在线 catalog 的 type 只能按 asset_id 字符串推断，容易把 ImageCollection
        误判成 Image。本地 catalog（datasets.json）带有明确的 type 字段，
        若在线候选与本地条目同 ID，则以本地 type 为准。
        """
        local_type_by_id = self._get_local_catalog_type_index()
        if not local_type_by_id:
            return online_features

        for feature in online_features:
            feature_id = str(feature.feature_id or "").strip()
            local_type = local_type_by_id.get(feature_id)
            if not local_type:
                continue
            current_type = str((feature.properties or {}).get("type") or "").strip().lower()
            if current_type == local_type:
                continue
            feature.feature_type = local_type
            feature.properties["type"] = local_type
            feature.properties["ee_object_type"] = self._map_gee_asset_type_to_ee_object_type(local_type)
            feature.properties["ee_ctor"] = self._map_gee_asset_type_to_ee_ctor(local_type)
        return online_features

    def _load_catalog(self) -> List[GeoFeature]:
        """
        从 datahypgen/mas/data 目录加载统一数据集 catalog（datasets.json 或 datasets.db）。

        优先使用 SQLite 数据库（datasets.db），回退到 JSON 文件（datasets.json）。
        数据库提供更快的查询性能，尤其是首次加载和按 ID 查询时。
        """
        if type(self)._catalog_cache is not None:
            return type(self)._catalog_cache

        catalog: List[GeoFeature] = []
        seen_feature_ids: set[str] = set()

        # 尝试使用数据库
        db_client = self.db_client
        if db_client is not None:
            try:
                logger.info("Loading catalog from database...")
                start_time = time.time()
                datasets = db_client.get_all_datasets(include_deprecated=True)
                load_time = time.time() - start_time
                logger.info(f"Loaded {len(datasets)} datasets from database in {load_time:.3f}s")

                for dataset in datasets:
                    normalized_feature = self._normalize_catalog_record(dataset)
                    if normalized_feature is None:
                        continue
                    feature_id = str(normalized_feature.feature_id or "").strip()
                    if not feature_id or feature_id in seen_feature_ids:
                        continue
                    seen_feature_ids.add(feature_id)
                    catalog.append(normalized_feature)

                type(self)._catalog_cache = catalog
                return type(self)._catalog_cache
            except Exception as e:
                logger.warning(f"Failed to load catalog from database: {e}. Falling back to JSON file.")

        # 回退到 JSON 文件
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

        candidate_paths = [
            os.path.join(data_dir, "datasets.json"),
        ]

        existing_paths = [path for path in candidate_paths if os.path.exists(path)]
        if not existing_paths:
            logger.warning(
                "Dataset catalog JSON not found. Tried: %s",
                candidate_paths,
            )
            type(self)._catalog_cache = []
            return type(self)._catalog_cache

        logger.info("Loading catalog from JSON file...")
        start_time = time.time()
        for gee_metadata_path in existing_paths:
            try:
                with open(gee_metadata_path, "r", encoding="utf-8") as f:
                    gee_metadata = json.load(f) or []

                if not isinstance(gee_metadata, list):
                    logger.warning(
                        "Unexpected dataset catalog format in %s: expected list, got %s",
                        gee_metadata_path,
                        type(gee_metadata).__name__,
                    )
                    continue

                for dataset in gee_metadata:
                    # 弃用资产不在装载层剔除：GEE 对其只发 DeprecationWarning，数据仍可读，
                    # 本地 catalog 内的弃用条目 removal_date 全部为空（只停更、无删除计划）。
                    # 直接剔除会误伤历史时段研究——GPWv4、CORINE、COPERNICUS/DEM/GLO30 都在其中。
                    # 弃用状态经 _normalize_catalog_record 透传给下游作 soft 信息。
                    normalized_feature = self._normalize_catalog_record(dataset)
                    if normalized_feature is None:
                        continue
                    feature_id = str(normalized_feature.feature_id or "").strip()
                    if not feature_id or feature_id in seen_feature_ids:
                        continue
                    seen_feature_ids.add(feature_id)
                    catalog.append(normalized_feature)
            except Exception as e:
                logger.warning(f"Failed to load dataset catalog metadata from {gee_metadata_path}: {e}")

        load_time = time.time() - start_time
        logger.info(f"Loaded {len(catalog)} datasets from JSON in {load_time:.3f}s")

        type(self)._catalog_cache = catalog
        return type(self)._catalog_cache

    def _get_token_idf_table(self) -> Dict[str, float]:
        """
        基于本地 catalog 的 title/id/tags/thematic_group 语料，一次性构建 token->IDF 权重表。

        动机：原 _calculate_relevance 用纯召回率（命中词数/query词数），数据集显示名里的
        修饰词（unconstrained / final / run / density / estimates ...）在候选规范名里往往
        不出现，却撑大分母、腰斩得分；而罕见主题词（worldpop / imerg / nasadem）与常见词
        （precipitation / data / global / monthly）权重相同。IDF 让罕见主题词主导得分、
        常见修饰词自然弱化。IDF = ln((N + 1) / (df + 1)) + 1，恒为正。
        """
        if type(self)._token_idf_cache is not None:
            return type(self)._token_idf_cache

        catalog = self._load_catalog()
        doc_freq: Dict[str, int] = {}
        total_docs = 0
        for feature in catalog:
            props = feature.properties or {}
            tags = props.get("tags", [])
            if not isinstance(tags, list):
                tags = [str(tags)]
            doc_text = " ".join(
                part
                for part in [
                    str(props.get("title") or feature.name or "").strip(),
                    str(props.get("id") or feature.feature_id or "").strip(),
                    str(props.get("ee_id") or "").strip(),
                    " ".join(str(tag).strip() for tag in tags if str(tag).strip()),
                    str(props.get("thematic_group") or "").strip(),
                    str(props.get("description") or "").strip(),
                ]
                if part
            )
            doc_tokens = self._tokenize_match_text(doc_text)
            if not doc_tokens:
                continue
            total_docs += 1
            for token in doc_tokens:
                doc_freq[token] = doc_freq.get(token, 0) + 1

        if total_docs == 0:
            # catalog 为空时不构建，_calculate_relevance 会退化回召回率公式
            type(self)._token_idf_cache = {}
            return type(self)._token_idf_cache

        idf_table: Dict[str, float] = {}
        for token, df in doc_freq.items():
            idf_table[token] = math.log((total_docs + 1) / (df + 1)) + 1.0
        # 未登录 token（catalog 里没出现过，通常是拼写变体或极罕见词）给最高权重，
        # 视为高判别力，避免被当噪声吞掉
        type(self)._token_idf_default = math.log(total_docs + 1) + 1.0
        type(self)._token_idf_cache = idf_table
        return type(self)._token_idf_cache

    def _resolve_backend(
        self,
        backend: Optional[str] = None,
        source_preference: Optional[str] = None,
    ) -> str:
        """解析检索后端（auto/gee/osm/stac）。"""
        if backend:
            normalized_backend = str(backend).strip().lower()
            if normalized_backend in {"gee", "osm", "stac", "auto", "pangaea"}:
                return normalized_backend

        if source_preference:
            normalized_source = str(source_preference).strip().lower()
            if normalized_source == "pangaea":
                return "pangaea"
            if normalized_source == "gee":
                return "gee"
            if normalized_source in {"osm", "openstreetmap"}:
                return "osm"
            if normalized_source in {"stac", "nasa_cmr", "planetary_computer", "earth_search"}:
                return "stac"

        return "auto"

    def _infer_gee_asset_type_from_id(self, asset_id: str) -> str:
        normalized_id = str(asset_id or "").strip()
        upper_id = normalized_id.upper()
        if not normalized_id:
            return "image"
        if normalized_id.startswith("projects/"):
            return "table"
        if any(token in upper_id for token in ["/C02/", "/C01/", "COLLECTION", "TIMESERIES", "TIME_SERIES"]):
            return "image_collection"
        return "image"

    def _strip_html(self, text: str) -> str:
        stripped = re.sub(r"<[^>]+>", " ", str(text or ""))
        stripped = unescape(stripped)
        return re.sub(r"\s+", " ", stripped).strip()

    def _parse_online_gee_catalog_html(self, html: str, source_url: str) -> List[GeoFeature]:
        page_text = str(html or "")
        if not page_text.strip():
            return []

        dataset_records: Dict[str, Dict[str, Any]] = {}

        # 按数据集卡片分块：每个 href 到下一个 href 之间的文本块包含该数据集的完整卡片内容
        # 结构：<a href="/earth-engine/datasets/catalog/SLUG"> ... title ... description ... tags-comment ... </a>
        card_blocks = re.split(
            r'(?=href="/earth-engine/datasets/catalog/[^"#?]{3,}\")',
            page_text,
            flags=re.IGNORECASE,
        )

        for block in card_blocks:
            # 提取 slug
            slug_match = re.match(
                r'href="/earth-engine/datasets/catalog/([^"#?]+)"',
                block,
                flags=re.IGNORECASE,
            )
            if not slug_match:
                continue
            slug = str(slug_match.group(1) or "").strip()
            if not slug or "/" in slug.split("_")[0]:
                # 跳过分类导航链接（如 "landsat"、"modis" 等不含下划线的顶级 slug）
                if "_" not in slug and slug.islower() and len(slug) < 20:
                    continue

            asset_id = slug.replace("_", "/")

            # 提取 title：data-text 属性最准确（包含完整名称）
            title = ""
            title_match = re.search(r'data-text="([^"]{3,200})"', block)
            if title_match:
                title = self._strip_html(title_match.group(1))
            if not title:
                h3_match = re.search(r'<h3[^>]*>(.*?)</h3>', block, re.DOTALL | re.IGNORECASE)
                if h3_match:
                    title = self._strip_html(h3_match.group(1))
            if not title:
                title = asset_id.split("/")[-1].replace("_", " ")

            # 提取 description snippet：<td class="ee-dataset-description-snippet">
            description = ""
            desc_match = re.search(
                r'class=["\']ee-dataset-description-snippet["\'][^>]*>(.*?)(?:</td>|</p>|</div>)',
                block,
                re.DOTALL | re.IGNORECASE,
            )
            if desc_match:
                description = self._strip_html(desc_match.group(1))

            # 提取 tags 和真实 asset_id：HTML 注释块格式为
            # <!-- tag1 tag2 tag3\n     REAL/ASSET/ID\n  Provider Name -->
            tags: List[str] = []
            comment_match = re.search(r'<!--(.*?)-->', block, re.DOTALL)
            if comment_match:
                comment_text = comment_match.group(1).strip()
                lines = [l.strip() for l in comment_text.splitlines() if l.strip()]
                if lines:
                    # 第一行是空格分隔的 tags
                    raw_tags = lines[0]
                    tags = [t.strip() for t in raw_tags.split() if t.strip() and t.strip() != "-->"]
                if len(lines) >= 2:
                    # 第二行是真实 asset_id（含斜杠）
                    candidate_id = lines[1].strip()
                    if "/" in candidate_id and not candidate_id.startswith("http"):
                        asset_id = candidate_id

            record = dataset_records.setdefault(
                asset_id,
                {
                    "id": asset_id,
                    "title": title,
                    "type": self._infer_gee_asset_type_from_id(asset_id),
                    "provider": "Google Earth Engine Catalog",
                    "tags": tags,
                    "docs": f"https://developers.google.com/earth-engine/datasets/catalog/{slug}",
                    "sample_code": "",
                    "thematic_group": "",
                    "description": description,
                },
            )
            # 已存在时补充空缺字段（description 和 tags 优先更新）
            if description and not record.get("description"):
                record["description"] = description
            if tags and not record.get("tags"):
                record["tags"] = tags
            if title and record.get("title") in ("", record["id"].split("/")[-1].replace("_", " ")):
                record["title"] = title

        if not dataset_records:
            return []

        normalized_features: List[GeoFeature] = []
        for record in dataset_records.values():
            feature = self._normalize_online_gee_catalog_record(record)
            if feature is not None:
                normalized_features.append(feature)
        return normalized_features

    async def _load_online_gee_catalog(self, query_hint: Optional[str] = None) -> List[GeoFeature]:
        if not self.online_gee_catalog_enabled or not self.online_gee_catalog_url:
            return []

        cache_key = self._ONLINE_GEE_CATALOG_CACHE_KEY
        if self.cache_enabled and cache_key in self._online_gee_catalog_cache:
            return self._online_gee_catalog_cache[cache_key]
        if (
            self.cache_enabled
            and self._online_gee_catalog_last_failure_at is not None
            and self.online_gee_catalog_failure_ttl_seconds > 0
        ):
            elapsed = time.time() - self._online_gee_catalog_last_failure_at
            if elapsed < self.online_gee_catalog_failure_ttl_seconds:
                logger.info(
                    "Skipping online GEE catalog fetch due to recent failure: "
                    "elapsed_seconds=%.1f failure_ttl_seconds=%s",
                    elapsed,
                    self.online_gee_catalog_failure_ttl_seconds,
                )
                return []

        try:
            timeout = aiohttp.ClientTimeout(total=self.online_gee_catalog_timeout_seconds)
            async with aiohttp.ClientSession(**self._online_gee_session_kwargs()) as session:
                async with session.get(
                    self.online_gee_catalog_url,
                    timeout=timeout,
                    headers={"User-Agent": "DataGenHyp/1.0 (GEE catalog search)"},
                ) as resp:
                    if resp.status != 200:
                        self._online_gee_catalog_last_failure_at = time.time()
                        logger.warning("Online GEE catalog request failed: status=%s", resp.status)
                        return []
                    html = await resp.text()
        except Exception as e:
            self._online_gee_catalog_last_failure_at = time.time()
            logger.warning(
                "Failed to fetch online GEE catalog: error=%s url=%s timeout_seconds=%s",
                self._format_exception_for_log(e),
                self.online_gee_catalog_url,
                self.online_gee_catalog_timeout_seconds,
            )
            return []

        features = self._parse_online_gee_catalog_html(html, self.online_gee_catalog_url)
        features = self._trust_local_catalog_type(features)
        if self.cache_enabled:
            self._online_gee_catalog_cache[cache_key] = features
        self._online_gee_catalog_last_failure_at = None
        return features

    async def resolve_online_dataset(self, intent) -> Dict[str, Any]:
        """在线检索数据集，支持 GEE、NASA_CMR、PLANETARY_COMPUTER。"""
        normalized_intent = self._build_local_resolution_intent(intent)
        candidates = await self._recall_online_candidates(normalized_intent)
        direct = self._resolve_exact_pangaea_candidate(normalized_intent, candidates)
        if direct is not None:
            return direct
        return self._resolve_gee_dataset_from_candidates(normalized_intent, candidates)

    async def rank_online_datasets(
        self,
        intent: Any,
        *,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """在线检索数据集排序列表，支持 GEE、NASA_CMR、PLANETARY_COMPUTER。"""
        normalized_intent = self._build_local_resolution_intent(intent)
        candidates = await self._recall_online_candidates(normalized_intent)
        ranked = self._build_ranked_resolution_results(
            normalized_intent,
            candidates,
            limit=limit,
        )
        direct = self._resolve_exact_pangaea_candidate(normalized_intent, candidates)
        if direct is not None:
            direct_id = self._feature_asset_id(direct["candidate"])
            return [direct] + [item for item in ranked if self._feature_asset_id(item.get("candidate")) != direct_id][:max(0, limit - 1)]
        return ranked

    def _resolve_exact_pangaea_candidate(self, intent, candidates):
        query = " ".join([str(intent.get("query_text", "")), *self._extract_probe_identity_ids(intent)])
        for candidate in candidates:
            feature = candidate["feature"]
            props = feature.properties
            if props.get("source") != "PANGAEA":
                continue
            if self._query_contains_exact_catalog_id(query, self._normalize_match_text(query), candidate["id"]):
                return self._build_direct_resolution_result(
                    intent, feature, candidate, {"exact_id_hit": candidate["id"]},
                )
        return None

    async def _recall_online_candidates(
        self,
        normalized_intent: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """召回在线候选：并行查询 GEE、NASA_CMR、PLANETARY_COMPUTER。"""
        query_hint = normalized_intent.get("query_text", "")
        explicit_ids = normalized_intent.get("explicit_dataset_ids", [])

        # 并行抓取三个源
        # GEE 使用原有的 HTML 抓取逻辑
        gee_task = self._load_online_gee_catalog(query_hint=query_hint)

        # NASA CMR 和 PC 使用 explicit_ids（如果有）或 query_hint
        search_queries = explicit_ids[:3] if explicit_ids else [query_hint]

        nasa_tasks = [self._search_nasa_cmr_online(q) for q in search_queries if q]
        pc_tasks = [self._search_planetary_computer_online(q) for q in search_queries if q]

        all_tasks = [gee_task] + nasa_tasks + pc_tasks
        if query_hint:
            all_tasks.append(self._search_pangaea_online(query_hint))
        results = await asyncio.gather(*all_tasks, return_exceptions=True)

        # 收集所有特征
        all_features: List[GeoFeature] = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Online source failed: {result}")
            elif isinstance(result, list):
                all_features.extend(result)

        # 归一化
        candidates = [
            self._normalize_local_candidate(feature)
            for feature in all_features
        ]
        candidates = [candidate for candidate in candidates if candidate]

        # 去重（按 id）
        seen_ids = set()
        deduped = []
        for candidate in candidates:
            cand_id = candidate.get("id")
            if cand_id and cand_id not in seen_ids:
                seen_ids.add(cand_id)
                deduped.append(candidate)

        # 文本打分和过滤
        recall_min_score = float(
            normalized_intent.get("recall_min_score", self.recall_min_score)
        )
        filtered: List[Dict[str, Any]] = []
        for candidate in deduped:
            relevance_result = self._score_text_relevance(normalized_intent, candidate)
            candidate["score_breakdown"] = relevance_result["score_breakdown"]
            candidate["text_score"] = relevance_result["total_score"]
            if candidate.get("text_score", 0.0) >= recall_min_score:
                filtered.append(candidate)

        return sorted(
            filtered,
            key=lambda candidate: (
                1 if self._is_strict_feature_match(query_hint, candidate["feature"]) else 0,
                candidate.get("text_score", 0.0),
            ),
            reverse=True,
        )[: max(1, int(normalized_intent["search_k"]))]

    async def _search_pangaea_online(self, query: str, bbox=None) -> List[GeoFeature]:
        records = await search_pangaea(query, limit=5, bbox=bbox)
        features = []
        for record in records:
            feature = self._normalize_catalog_record(record)
            if feature is not None:
                for key in ("doi", "relations", "parameters", "metadata_url", "data_url", "is_collection", "access_note", "distributions", "conditions_of_access", "is_accessible_for_free", "collected_start", "collected_end"):
                    feature.properties[key] = record.get(key)
                if record.get("bbox") is not None:
                    feature.geometry = mapping(box(*record["bbox"]))
                features.append(feature)
        return features

    async def _search_nasa_cmr_online(self, query: str) -> List[GeoFeature]:
        """在线搜索 NASA CMR，返回 GeoFeature 列表。

        query 应该是完整的 collection_id（如 datasets.json 中的 id）。
        """
        if not query or len(query.strip()) < 2:
            return []

        try:
            result = await query_nasa_cmr(collection_id=query.strip(), limit=1)
            if result.get("error"):
                logger.debug(f"NASA CMR search for '{query}' failed: {result['error']}")
                return []

            # 从 collection metadata 构造 catalog record
            collection_id = result.get("collection_id", query.strip())
            title = result.get("title") or collection_id

            record = {
                "id": collection_id,
                "title": title,
                "description": result.get("description", ""),
                "source": "NASA_CMR",
                "type": "image_collection",
                "provider": result.get("provider_id", "NASA"),
                "doc_url": result.get("collection_url", ""),
                "stac_url": result.get("collection_url", ""),
            }

            feature = self._normalize_catalog_record(record)
            if feature:
                # 标记为在线来源
                feature.source = f"NASA_CMR_ONLINE:{collection_id}"
                return [feature]

            return []
        except Exception as exc:
            logger.warning(f"NASA CMR online search failed for '{query}': {exc}")
            return []

    async def _search_planetary_computer_online(self, query: str) -> List[GeoFeature]:
        """在线搜索 Planetary Computer，返回 GeoFeature 列表。

        query 应该是 collection_id（通常是小写连字符格式）。
        """
        if not query or len(query.strip()) < 2:
            return []

        try:
            # PC 的 collection id 通常是小写连字符格式
            normalized_query = query.lower().replace(" ", "-").replace("_", "-")
            result = await query_planetary_computer(collection_id=normalized_query, limit=1)

            if result.get("error"):
                logger.debug(f"Planetary Computer search for '{query}' failed: {result['error']}")
                return []

            # 从 collection metadata 构造 catalog record
            collection_id = result.get("collection_id", normalized_query)
            title = result.get("title") or collection_id

            record = {
                "id": collection_id,
                "title": title,
                "description": result.get("description", ""),
                "source": "PLANETARY_COMPUTER",
                "type": "image_collection",
                "provider": "Microsoft Planetary Computer",
                "doc_url": result.get("collection_url", ""),
                "stac_url": result.get("collection_url", ""),
            }

            feature = self._normalize_catalog_record(record)
            if feature:
                feature.source = f"PLANETARY_COMPUTER_ONLINE:{collection_id}"
                return [feature]

            return []
        except Exception as exc:
            logger.warning(f"Planetary Computer online search failed for '{query}': {exc}")
            return []

    def _filter_features_by_keyword(
        self,
        keyword: str,
        features: List[GeoFeature],
        spatial_filter: Optional[Dict] = None,
        min_relevance_score: float = 0.7,
        limit: Optional[int] = None,
        parsed_query: Optional[Dict[str, Any]] = None,
    ) -> List[GeoFeature]:
        """对候选特征按关键词相关性进行过滤和排序。"""
        matched_features: List[GeoFeature] = []

        for feature in features:
            relevance = self._score_feature_relevance(
                keyword,
                feature,
                parsed_query=parsed_query,
            )
            strict_match = self._is_strict_feature_match(keyword, feature)
            feature.properties = feature.properties or {}
            feature.properties["relevance_score"] = relevance
            feature.properties["strict_match"] = strict_match

            if relevance >= min_relevance_score:
                if spatial_filter:
                    if self._check_spatial_overlap(feature.geometry, spatial_filter):
                        matched_features.append(feature)
                else:
                    matched_features.append(feature)

        matched_features.sort(
            key=lambda item: (
                bool(item.properties.get("strict_match")),
                item.properties.get("relevance_score", 0),
            ),
            reverse=True,
        )

        top_k = 3
        if isinstance(limit, int) and limit > 0:
            top_k = limit
        return matched_features[:top_k]

    def _search_gee_by_keyword(
        self,
        keyword: str,
        spatial_filter: Optional[Dict] = None,
        min_relevance_score: float = 0.7,
        limit: Optional[int] = None,
        parsed_query: Optional[Dict[str, Any]] = None,
    ) -> List[GeoFeature]:
        """
        GEE 集合检索：
        1) 先查本地 community catalog
        2) 返回可直接用于 ee 的对象类型与资产 ID

        注意：`_load_catalog()` 现在装的是 datasets.json 的全部 6 个源，
        方法名是历史遗留（早期 catalog 只有 GEE）。这里不按 dataset_source
        过滤——GEE 检索历来就是"全量候选池"的语义，过滤会缩小召回而非收紧。
        真正要限定到 GEE 的场景应显式传 backend="gee" 由调用方在结果里
        按 dataset_source 二次过滤，或使用 `_search_stac_by_keyword` 拿纯
        STAC 候选。
        """
        gee_features = self._load_catalog()
        matched = self._filter_features_by_keyword(
            keyword=keyword,
            features=gee_features,
            spatial_filter=spatial_filter,
            min_relevance_score=min_relevance_score,
            limit=limit,
            parsed_query=parsed_query,
        )
        return self._attach_gee_runtime_metadata(matched)

    # STAC 源集合：datasets.json 里通过 STAC API 提供 collection 元数据的源。
    # 与 GEE_CATALOG_SOURCES 互斥,合起来加 OSM_TAGINFO 构成 catalog 的全部来源。
    STAC_CATALOG_SOURCES = frozenset({"NASA_CMR", "PLANETARY_COMPUTER", "EARTH_SEARCH"})

    def _search_stac_by_keyword(
        self,
        keyword: str,
        spatial_filter: Optional[Dict] = None,
        min_relevance_score: float = 0.7,
        limit: Optional[int] = None,
        parsed_query: Optional[Dict[str, Any]] = None,
    ) -> List[GeoFeature]:
        """STAC 源检索：只在 NASA_CMR / Planetary Computer / Earth Search 三个
        STAC catalog 里找,不混入 GEE 或 OSM 候选。

        这几个源都没有 ee_ctor,可达性验证走 dataset_source_clients 里各自的
        STAC 客户端,而不是 Earth Engine 运行时——调用方拿到结果后应该用
        dataset_source_clients.probe_dataset,不要往 ee.* 里塞。
        """
        catalog = self._load_catalog()
        stac_features = [
            feature
            for feature in catalog
            if str((feature.properties or {}).get("source") or "").strip().upper()
            in self.STAC_CATALOG_SOURCES
        ]
        matched = self._filter_features_by_keyword(
            keyword=keyword,
            features=stac_features,
            spatial_filter=spatial_filter,
            min_relevance_score=min_relevance_score,
            limit=limit,
            parsed_query=parsed_query,
        )
        return self._attach_gee_runtime_metadata(matched)


    def _attach_gee_runtime_metadata(self, features: List[GeoFeature]) -> List[GeoFeature]:
        """
        为 GEE 检索结果补充可直接用于 ee 的运行时信息。

        非 GEE 源（STAC collection、OSM tag）不参与补全：它们的 type 借用了
        image_collection 这类词汇，但 asset 并不在 Earth Engine 里，推断出
        ee_ctor 只会让探测阶段拿错的构造器去访问不存在的 asset。
        """
        for feature in features:
            if not isinstance(feature, GeoFeature):
                continue

            feature.properties = feature.properties or {}

            asset_type = str(
                feature.properties.get("type")
                or feature.feature_type
                or ""
            ).strip().lower()
            catalog_source = str(feature.properties.get("source") or "").strip().upper()
            feature.properties["title"] = feature.properties.get("title") or feature.name
            feature.properties["type"] = asset_type

            if catalog_source and catalog_source not in GEE_CATALOG_SOURCES:
                feature.properties["id"] = (
                    feature.properties.get("id") or feature.feature_id or ""
                )
                feature.properties.setdefault("ee_id", "")
                feature.properties.setdefault("ee_object_type", "")
                feature.properties.setdefault("ee_ctor", None)
                continue

            ee_id = str(
                feature.properties.get("ee_id")
                or feature.properties.get("id")
                or feature.feature_id
                or ""
            ).strip()

            ee_object_type = str(
                feature.properties.get("ee_object_type")
                or self._map_gee_asset_type_to_ee_object_type(asset_type)
            ).strip()
            ee_ctor = feature.properties.get("ee_ctor") or self._map_gee_asset_type_to_ee_ctor(asset_type)

            feature.properties["id"] = feature.properties.get("id") or ee_id
            feature.properties["ee_id"] = ee_id
            feature.properties["ee_object_type"] = ee_object_type
            feature.properties["ee_ctor"] = ee_ctor

        return features

    def _search_osm_by_keyword(
        self,
        keyword: str,
        spatial_filter: Optional[Dict] = None,
        min_relevance_score: float = 0.7,
        limit: Optional[int] = None,
    ) -> List[GeoFeature]:
        """OSM 元数据检索。"""
        osm_features = [
            GeoFeature(
                feature_id="osm_facilities",
                name="OpenStreetMap POI Facilities",
                data_description="OpenStreetMap points of interest, amenity facilities, and urban infrastructure features.",
                feature_type="poi",
                geometry={"type": "GeometryCollection", "geometries": []},
                properties={"relevance_score": 0.0},
                source="OpenStreetMap",
            ),
            GeoFeature(
                feature_id="osm_transport",
                name="OpenStreetMap Transport Network",
                data_description="OpenStreetMap road network and transport topology for geospatial accessibility analysis.",
                feature_type="transport",
                geometry={"type": "GeometryCollection", "geometries": []},
                properties={"relevance_score": 0.0},
                source="OpenStreetMap",
            ),
        ]
        return self._filter_features_by_keyword(
            keyword=keyword,
            features=osm_features,
            spatial_filter=spatial_filter,
            min_relevance_score=min_relevance_score,
            limit=limit,
        )

    async def search_by_bbox(self, bbox: Dict[str, float], data_type: str) -> List[GeoFeature]:
        """
        通过矩形范围(BBOX)查询地理数据
        bbox format: {'min_lat', 'min_lon', 'max_lat', 'max_lon'}
        """
        cache_key = f"{bbox}_{data_type}"
        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]


        # 默认实现：OpenStreetMap Overpass API
        features = await self._query_osm_overpass(bbox, data_type)
        try:
            features.extend(await self._search_pangaea_online(
                data_type, bbox=[bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"]],
            ))
        except Exception as exc:
            logger.warning("PANGAEA spatial search failed: %s", exc)
            return features
        
        if self.cache_enabled:
            self._cache[cache_key] = features
        return features

    async def _query_osm_overpass(self, bbox: Dict, data_type: str) -> List[GeoFeature]:
        """实现第一个数据源连接器：OSM """
        # 构建 Overpass QL 查询语句
        query = f"""
        [out:json][timeout:25];
        (
          node["amenity"="{data_type}"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
          way["amenity"="{data_type}"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
        );
        out body; >; out skel qt;
        """
        url = "https://overpass-api.de/api/interpreter"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data={'data': query}) as resp:
                    if resp.status != 200:
                        logger.error(f"OSM API Error: {resp.status}")
                        return []

                    data = await resp.json()
                    return self._parse_osm_to_features(data)
        except Exception as e:
            logger.error(f"Failed to fetch geo data: {str(e)}")
            return []

    def _parse_osm_to_features(self, osm_data: Dict) -> List[GeoFeature]:
        results = []
        for element in osm_data.get('elements', []):
            if 'lat' in element and 'lon' in element:
                geom = {"type": "Point", "coordinates": [element['lon'], element['lat']]}
                actual_name = element.get('tags', {}).get('name', 'Unknown')
                feature = GeoFeature(
                    feature_id=str(element['id']),
                    name=actual_name,
                    data_description=f"OSM facility: {actual_name}", 
                    feature_type=element.get('tags', {}).get('amenity', 'point'),
                    geometry=geom,
                    properties=element.get('tags', {}),
                    source="OpenStreetMap"
                )
                results.append(feature)
        return results

    async def search_by_keyword(
        self, 
        keyword: str, 
        spatial_filter: Optional[Dict] = None,
        min_relevance_score: float = 0.7,
        limit: Optional[int] = None,
        backend: Optional[str] = None,
        source_preference: Optional[str] = None,
    ) -> List[GeoFeature]:
        """
        基于关键词进行多后端检索（向后兼容原接口）。

        Args:
            keyword: 查询关键词
            spatial_filter: 可选空间过滤
            min_relevance_score: 最小相关性阈值
            limit: 返回结果上限（默认 3）
            backend: 显式后端（gee/osm/auto）
            source_preference: 来源偏好（用于路由 gee/osm）
        """
        effective_backend = self._resolve_backend(
            backend=backend,
            source_preference=source_preference,
        )

        # 1. 检查缓存
        cache_key = (
            f"keyword:{keyword}:{spatial_filter}:"
            f"backend={effective_backend}:limit={limit}:source_pref={source_preference}"
        )
        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        logger.info(f"正在根据关键词检索相关地理数据: {keyword} (backend={effective_backend})")

        if effective_backend == "pangaea":
            features = []
        elif effective_backend == "gee":
            features = self._search_gee_by_keyword(
                keyword=keyword,
                spatial_filter=spatial_filter,
                min_relevance_score=min_relevance_score,
                limit=limit,
            )
        elif effective_backend == "osm":
            features = self._search_osm_by_keyword(
                keyword=keyword,
                spatial_filter=spatial_filter,
                min_relevance_score=min_relevance_score,
                limit=limit,
            )
        elif effective_backend == "stac":
            features = self._search_stac_by_keyword(
                keyword=keyword,
                spatial_filter=spatial_filter,
                min_relevance_score=min_relevance_score,
                limit=limit,
            )
        else:
            all_candidate_features = await self._get_all_available_metadata()
            features = self._filter_features_by_keyword(
                keyword=keyword,
                features=all_candidate_features,
                spatial_filter=spatial_filter,
                min_relevance_score=min_relevance_score,
                limit=limit,
            )

        # 如果启用了缓存机制，将筛选后的结果存入缓存
        try:
            pangaea_features = await self._search_pangaea_online(keyword)
        except Exception as exc:
            logger.warning("PANGAEA keyword search failed: %s", exc)
            # Preserve other providers, but do not cache an incomplete search.
            return features
        features = self._filter_features_by_keyword(
            keyword=keyword,
            features=features + pangaea_features,
            spatial_filter=spatial_filter,
            min_relevance_score=min_relevance_score,
            limit=limit,
        )
        if self.cache_enabled:
            self._cache[cache_key] = features
            
        return features
    
    def build_gee_call(self, keyword: str) -> Optional[Dict[str, Any]]:
        """
        根据关键词返回可直接执行的 ee 调用信息。
        例如:
        {
            "ee_object_type": "ImageCollection",
            "ee_id": "projects/xxx/yyy",
            "expression": 'ee.ImageCollection("projects/xxx/yyy")'
        }
        """
        intent = self._build_local_resolution_intent(keyword)
        resolved = self.resolve_local_dataset(intent)
        if resolved.get("status") != "resolved_strong" or resolved.get("candidate") is None:
            return None

        feature = resolved["candidate"]
        props = feature.properties or {}
        ee_ctor = props.get("ee_ctor")
        ee_id = props.get("ee_id") or props.get("id") or feature.feature_id

        if not ee_ctor or not ee_id:
            return None

        return {
            "title": props.get("title") or feature.name,
            "id": props.get("id") or feature.feature_id,
            "type": props.get("type") or feature.feature_type,
            "ee_id": ee_id,
            "ee_object_type": props.get("ee_object_type"),
            "ee_ctor": ee_ctor,
            "docs": props.get("docs"),
            "sample_code": props.get("sample_code"),
            "provider": props.get("provider"),
            "tags": props.get("tags", []),
            "thematic_group": props.get("thematic_group"),
            "license": props.get("license"),
            "source": feature.source,
            "binding_decision": resolved.get("binding_decision", {}),
            "resolution_status": resolved.get("status"),
            "expression": f'{ee_ctor}("{ee_id}")',
        }

    def resolve_local_dataset(self, intent) -> Dict[str, Any]:
        normalized_intent = self._build_local_resolution_intent(intent)
        # 改法①：精确 catalog id 优先于 alias 查表。
        # 前提：alias 的键都是"人写模糊名"（modis ndvi / noaa cdr avhrr ndvi v5），
        # 不会是完整 catalog id；而 `_query_contains_exact_catalog_id` 只在该 query 字面
        # 含完整 id 时才命中，二者不重叠。故把精确 id 提到最前不会破坏 alias 重定向，
        # 只会让"query/canonical/entity 已明确给出 id"的情况按用户裁定优先绑定到该 id。
        # 精确 id 匹配同时纳入从 canonical_requirement_name / frozen_constraints.entity
        # 提取的可探测身份（改法②），使仅靠声明侧 id 标识的需求无需目录语义匹配。
        probe_ids = self._extract_probe_identity_ids(normalized_intent)
        exact_id_result = self._resolve_by_exact_catalog_id(normalized_intent, probe_ids=probe_ids)
        if exact_id_result is not None:
            return exact_id_result
        alias_result = self._resolve_by_alias_index(normalized_intent)
        if alias_result is not None:
            return alias_result
        candidates = self._recall_local_candidates(normalized_intent)
        return self._resolve_gee_dataset_from_candidates(normalized_intent, candidates)

    def rank_local_datasets(
        self,
        intent: Any,
        *,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Return evaluated local candidates without collapsing the ranked list to top-1."""
        normalized_intent = self._build_local_resolution_intent(intent)
        probe_ids = self._extract_probe_identity_ids(normalized_intent)
        direct_result = self._resolve_by_exact_catalog_id(normalized_intent, probe_ids=probe_ids)
        if direct_result is None:
            direct_result = self._resolve_by_alias_index(normalized_intent)
        ranked_results = self._build_ranked_resolution_results(
            normalized_intent,
            self._recall_local_candidates(normalized_intent),
            limit=limit,
        )
        if direct_result is None or direct_result.get("candidate") is None:
            return ranked_results

        direct_id = self._feature_asset_id(direct_result["candidate"])
        merged = [direct_result]
        merged.extend(
            result
            for result in ranked_results
            if self._feature_asset_id(result.get("candidate")) != direct_id
        )
        return merged[: max(1, int(limit))]

    def _build_direct_resolution_result(
        self,
        intent: Dict[str, Any],
        feature: GeoFeature,
        candidate: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        candidate = self._enrich_normalized_candidate(candidate)
        constraint_report = self._build_constraint_report(intent, candidate)
        hard_rejections = self._build_blocking_reasons(
            intent,
            constraint_report,
            candidate,
        )
        verification_gaps = self._build_verification_gaps(
            intent,
            constraint_report,
        )
        if not intent.get("expected_data_types_explicit"):
            hard_rejections = [
                reason for reason in hard_rejections if reason != "type_mismatch_hard"
            ]
        frozen_report = self._evaluate_frozen_constraints(
            intent,
            candidate,
            constraint_report,
        )
        frozen_constraints_satisfied = bool(frozen_report["satisfied"])
        if hard_rejections:
            binding_grade = "unresolved_semantic"
            accepted_for_binding = False
        elif verification_gaps:
            binding_grade = "resolved_weak"
            accepted_for_binding = False
        elif frozen_constraints_satisfied:
            binding_grade = "resolved_strong"
            accepted_for_binding = True
        else:
            binding_grade = "resolved_weak"
            accepted_for_binding = False
        binding_decision = {
            "accepted_for_binding": accepted_for_binding,
            "binding_grade": binding_grade,
            "blocking_reasons": hard_rejections,
            "hard_rejections": hard_rejections,
            "verification_gaps": verification_gaps,
            "soft_reasons": list(frozen_report["reasons"]),
            "constraint_report": constraint_report,
            "score_breakdown": {
                "title": 1.0,
                "id": 1.0,
                "tags": 1.0,
                "thematic_group": 1.0,
                "family_bonus": 0.15,
            },
            "text_score": 1.0,
            "final_score": 1.0,
            "frozen_constraints_satisfied": frozen_constraints_satisfied,
        }
        binding_decision.update(metadata)
        confidence_flags = dict(metadata)
        confidence_flags["direct_match"] = True
        return {
            "status": binding_grade,
            "candidate": feature,
            "top_candidates": [{"id": candidate.get("id", ""), "title": candidate.get("title", "")}],
            "binding_decision": binding_decision,
            "reasons": (
                hard_rejections
                or verification_gaps
                or list(frozen_report["reasons"])
            ),
            "confidence_flags": confidence_flags,
            "penalty_report": {"region_penalty": 0.0, "time_penalty": 0.0, "role_penalty": 0.0},
            "frozen_constraints_satisfied": frozen_constraints_satisfied,
        }

    def _query_contains_exact_catalog_id(
        self,
        raw_query: str,
        normalized_query: str,
        candidate_id: str,
    ) -> bool:
        normalized_id = self._normalize_match_text(candidate_id)
        if not candidate_id or not normalized_id:
            return False
        raw_query_cf = raw_query.casefold()
        candidate_id_cf = candidate_id.casefold()
        if candidate_id_cf == raw_query_cf:
            return True
        if normalized_id == normalized_query:
            return True
        if f" {normalized_id} " in f" {normalized_query} ":
            return True
        pattern = re.compile(rf"(?<![a-z0-9]){re.escape(candidate_id)}(?![a-z0-9])", re.IGNORECASE)
        return bool(pattern.search(raw_query))

    def _resolve_by_exact_catalog_id(
        self,
        normalized_intent: Dict[str, Any],
        probe_ids: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        raw_query = str(normalized_intent.get("query_text") or "").strip()
        if not raw_query:
            return None

        probe_set = {
            str(item).strip().casefold()
            for item in (probe_ids or [])
            if str(item or "").strip()
        }
        normalized_query = self._normalize_match_text(raw_query)
        for feature in self._load_catalog():
            props = feature.properties or {}
            candidate_ids = [
                str(props.get("id") or "").strip(),
                str(props.get("ee_id") or "").strip(),
                str(feature.feature_id or "").strip(),
            ]

            def _candidate_matches(candidate_id: str) -> bool:
                if not candidate_id:
                    return False
                if probe_set and candidate_id.casefold() in probe_set:
                    return True
                return self._query_contains_exact_catalog_id(
                    raw_query=raw_query,
                    normalized_query=normalized_query,
                    candidate_id=candidate_id,
                )

            matched_id = next(
                (candidate_id for candidate_id in candidate_ids if _candidate_matches(candidate_id)),
                None,
            )
            if matched_id is None:
                continue
            candidate = self._normalize_local_candidate(feature)
            if candidate is None:
                return None
            return self._build_direct_resolution_result(
                normalized_intent,
                feature,
                candidate,
                {"exact_id_hit": matched_id},
            )
        return None

    def _resolve_by_alias_index(self, normalized_intent: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """在 GEE_DATASET_ALIAS_INDEX 中做精确/子串匹配，命中则构造 resolved_strong 结果。

        匹配逻辑：
        1. 对 query_text 做规范化（小写、合并空白）
        2. 逐 key 检查规范化 key 是否是规范化 query 的子串（或完整匹配）
        3. 命中最长 key（最具体的 alias 优先）
        4. 用命中的 catalog_id 在本地 catalog 中找对应 feature
        5. 找到则返回 resolved_strong；找不到（catalog 未收录）则返回 None，降回普通检索
        """
        raw_query = str(normalized_intent.get("query_text") or "").strip()
        if not raw_query:
            return None
        normalized_q = self._normalize_match_text(raw_query)

        # 找命中最长 alias（避免短 key 误匹配）
        best_key: Optional[str] = None
        best_catalog_id: Optional[str] = None
        for alias_key, catalog_id in GEE_DATASET_ALIAS_INDEX.items():
            norm_key = self._normalize_match_text(alias_key)
            if not norm_key:
                continue
            # 完整词边界匹配：key 必须作为完整词组出现在 query 中
            if norm_key == normalized_q or f" {norm_key} " in f" {normalized_q} ":
                if best_key is None or len(norm_key) > len(self._normalize_match_text(best_key)):
                    best_key = alias_key
                    best_catalog_id = catalog_id

        if best_catalog_id is None:
            return None

        # 在本地 catalog 中找 catalog_id 对应的 feature
        for feature in self._load_catalog():
            props = feature.properties or {}
            fid = str(props.get("id") or props.get("ee_id") or feature.feature_id or "").strip()
            if fid == best_catalog_id:
                candidate = self._normalize_local_candidate(feature)
                if candidate is None:
                    break
                return self._build_direct_resolution_result(
                    normalized_intent,
                    feature,
                    candidate,
                    {"alias_hit": best_key},
                )

        # catalog_id 在 alias 中有定义但本地 catalog 中未收录，降回普通检索
        logger.debug(
            "Alias hit '%s' -> '%s' but catalog entry not found locally; falling back to token search.",
            best_key, best_catalog_id,
        )
        asset_type = self._infer_gee_asset_type_from_id(best_catalog_id)
        ee_object_type = self._map_gee_asset_type_to_ee_object_type(asset_type)
        ee_ctor = self._map_gee_asset_type_to_ee_ctor(asset_type)
        alias_feature = GeoFeature(
            feature_id=best_catalog_id,
            name=best_key,
            data_description=f"{best_key} {best_catalog_id} alias-index Earth Engine dataset",
            feature_type=asset_type,
            geometry={"type": "GeometryCollection", "geometries": []},
            properties={
                "title": best_key,
                "id": best_catalog_id,
                "type": asset_type,
                "ee_id": best_catalog_id,
                "ee_object_type": ee_object_type,
                "ee_ctor": ee_ctor,
                "provider": "GEE alias index",
                "tags": [best_key],
                "license": "",
                "thematic_group": "",
            },
            source=f"GEE_ALIAS:{best_catalog_id}",
        )
        alias_candidate = self._normalize_local_candidate(alias_feature)
        if alias_candidate is None:
            return None
        return self._build_direct_resolution_result(
            normalized_intent,
            alias_feature,
            alias_candidate,
            {"alias_hit": best_key, "alias_key": best_key},
        )

    def _resolve_gee_dataset_from_candidates(self, intent, candidates) -> Dict[str, Any]:
        evaluation_result = self._evaluate_candidate_set(intent, candidates)
        evaluated_candidates = evaluation_result.get("evaluated_candidates", [])
        ranked_candidates = self._rank_candidate_set(intent, evaluated_candidates)
        top_candidates = [candidate["summary"] for candidate in ranked_candidates[:5]]

        if not evaluated_candidates:
            return {
                "status": "unresolved_real_missing",
                "candidate": None,
                "top_candidates": top_candidates,
                "binding_decision": {
                    "accepted_for_binding": False,
                    "binding_grade": "unresolved_real_missing",
                    "blocking_reasons": evaluation_result.get("blocking_reasons", ["no_candidates"]),
                    "hard_rejections": [],
                    "verification_gaps": [],
                    "constraint_report": {
                        "region_match": "unspecified",
                        "time_match": "unknown",
                        "role_match": "acceptable",
                        "type_match": False,
                        "theme_match": False,
                        "family_match": False,
                    },
                    "score_breakdown": {
                        "title": 0.0,
                        "id": 0.0,
                        "tags": 0.0,
                        "thematic_group": 0.0,
                        "family_bonus": 0.0,
                    },
                    "text_score": 0.0,
                    "final_score": 0.0,
                },
                "reasons": evaluation_result.get("blocking_reasons", ["no_candidates"]),
                "confidence_flags": {},
                "penalty_report": {
                    "region_penalty": 0.0,
                    "time_penalty": 0.0,
                    "role_penalty": 0.0,
                },
                "frozen_constraints_satisfied": False,
            }

        composite_resolution = self._try_multi_collection_composite_resolution(intent, ranked_candidates)
        if composite_resolution is not None:
            return composite_resolution

        best_candidate = ranked_candidates[0]
        binding_decision = best_candidate["binding_decision"]
        status = str(binding_decision.get("binding_grade") or "unresolved_semantic")
        constraint_report = binding_decision.get("constraint_report", {})

        # 提取 binding_status（用于 manifest）
        binding_status = "bindable" if binding_decision.get("accepted_for_binding") else "unbound"

        return {
            "status": status,
            "binding_status": binding_status,
            "candidate": best_candidate.get("feature"),
            "top_candidates": top_candidates,
            "binding_decision": binding_decision,
            "reasons": (
                binding_decision.get("blocking_reasons", [])
                or binding_decision.get("soft_reasons", [])
            ),
            "confidence_flags": {
                "constraint_report": constraint_report,
                "region_sort_tier": best_candidate.get("region_sort_tier"),
                "family_match": constraint_report.get("family_match", False),
            },
            "penalty_report": binding_decision.get("penalty_report", {}),
            "frozen_constraints_satisfied": bool(
                binding_decision.get("frozen_constraints_satisfied", False)
            ),
        }

    @staticmethod
    def _feature_asset_id(feature: Any) -> str:
        if feature is None:
            return ""
        properties = getattr(feature, "properties", None) or {}
        return str(
            properties.get("id")
            or properties.get("ee_id")
            or getattr(feature, "feature_id", "")
            or ""
        ).strip()

    def _build_ranked_resolution_results(
        self,
        intent: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        *,
        limit: int,
    ) -> List[Dict[str, Any]]:
        evaluation_result = self._evaluate_candidate_set(intent, candidates)
        ranked_candidates = self._rank_candidate_set(
            intent,
            evaluation_result.get("evaluated_candidates", []),
        )
        results: List[Dict[str, Any]] = []
        for ranked_candidate in ranked_candidates[: max(1, int(limit))]:
            binding_decision = ranked_candidate["binding_decision"]
            results.append({
                "status": str(
                    binding_decision.get("binding_grade") or "unresolved_semantic"
                ),
                "candidate": ranked_candidate.get("feature"),
                "binding_decision": binding_decision,
                "reasons": (
                    binding_decision.get("blocking_reasons", [])
                    or binding_decision.get("soft_reasons", [])
                ),
                "confidence_flags": {
                    "constraint_report": binding_decision.get("constraint_report", {}),
                    "region_sort_tier": ranked_candidate.get("region_sort_tier"),
                },
                "penalty_report": binding_decision.get("penalty_report", {}),
                "frozen_constraints_satisfied": bool(
                    binding_decision.get("frozen_constraints_satisfied", False)
                ),
            })
        return results

    def _try_multi_collection_composite_resolution(
        self,
        intent: Dict[str, Any],
        ranked_candidates: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        对于 Landsat 一类的长时间序列需求，允许用多个同 family collection 组合满足。

        触发条件（最小保守策略）：
          - intent 明确要求 annual/time_series
          - allowed/preferred family 指向 Landsat
          - 当前单候选失败主因是 temporal_coverage_incomplete
          - 候选都是 image_collection 且 family/theme/type 匹配
        """
        if not ranked_candidates:
            return None

        time_terms = set(intent.get("time_terms") or [])
        requires_timeseries = bool({"annual", "historical", "time_series"} & time_terms)
        if not requires_timeseries:
            return None

        families = {
            str(family).strip().lower()
            for family in (intent.get("allowed_product_families") or [])
            if str(family).strip()
        }
        preferred_family = str(intent.get("preferred_family", "") or "").strip().lower()
        if preferred_family:
            families.add(preferred_family)
        if "landsat" not in families:
            return None

        composite_members: List[Dict[str, Any]] = []
        for candidate in ranked_candidates:
            decision = candidate.get("binding_decision") or {}
            report = decision.get("constraint_report") or {}
            blocking = set(decision.get("blocking_reasons") or [])
            if candidate.get("candidate_type") != "image_collection":
                continue
            if "temporal_coverage_incomplete" not in blocking:
                continue
            if not report.get("type_match", False):
                continue
            if not report.get("theme_match", False):
                continue
            if families:
                candidate_text = self._normalize_match_text(
                    " ".join(
                        str(part)
                        for part in [
                            candidate.get("title", ""),
                            candidate.get("id", ""),
                            candidate.get("ee_id", ""),
                            " ".join(candidate.get("tags", []) or []),
                        ]
                        if part
                    )
                )
                if "landsat" not in candidate_text:
                    continue
            composite_members.append(candidate)

        if len(composite_members) < 2:
            return None

        members = [
            {
                "name": member.get("title") or member.get("feature").name,
                "id": member.get("id"),
                "ee_id": member.get("ee_id"),
                "type": member.get("candidate_type"),
            }
            for member in composite_members[:5]
        ]
        reasons = [
            "multi_collection_temporal_stitching_required",
            "single_collection_temporal_coverage_resolved_by_family_composition",
        ]
        binding_decision = {
            "accepted_for_binding": False,
            "binding_grade": "resolved_weak",
            "blocking_reasons": [],
            "hard_rejections": [],
            "verification_gaps": [],
            "soft_reasons": reasons,
            "constraint_report": {
                "region_match": "global",
                "time_match": "partial",
                "role_match": "acceptable",
                "type_match": True,
                "theme_match": True,
                "family_match": True,
            },
            "score_breakdown": {
                "title": max((member.get("text_score", 0.0) for member in composite_members), default=0.0),
                "id": 0.0,
                "tags": 0.0,
                "thematic_group": 0.0,
                "family_bonus": 0.15,
            },
            "text_score": max((member.get("text_score", 0.0) for member in composite_members), default=0.0),
            "final_score": max((member.get("final_score", 0.0) for member in composite_members), default=0.0),
            "penalty_report": {
                "region_penalty": 0.03,
                "time_penalty": -0.08,
                "role_penalty": 0.0,
            },
            "frozen_constraints_satisfied": True,
            "composite_binding": {
                "strategy": "multi_collection_family_composition",
                "family": preferred_family or "landsat",
                "members": members,
            },
        }
        return {
            "status": "resolved_weak",
            "candidate": None,
            "top_candidates": [
                {
                    "name": "Composite binding via multiple Landsat collections",
                    "id": "composite:landsat_multi_collection",
                    "type": "image_collection",
                    "status": "resolved_weak",
                    "binding_decision": binding_decision,
                }
            ] + [candidate["summary"] for candidate in ranked_candidates[:4]],
            "binding_decision": binding_decision,
            "reasons": reasons,
            "confidence_flags": {
                "constraint_report": binding_decision["constraint_report"],
                "region_sort_tier": 2,
                "family_match": True,
                "composite_binding": True,
            },
            "penalty_report": binding_decision["penalty_report"],
            "frozen_constraints_satisfied": True,
        }

    def _build_local_resolution_intent(self, intent) -> Dict[str, Any]:
        raw_intent = intent if isinstance(intent, dict) else {}
        query_text = str(
            raw_intent.get("query_text")
            or raw_intent.get("raw_query")
            or raw_intent.get("keyword")
            or intent
            or ""
        ).strip()
        normalized_query = self._normalize_match_text(query_text)
        query_tokens = self._tokenize_match_text(query_text)
        # 原始 query token（未经 synonym 扩展），用于识别真正的主题词。
        # synonym 扩展会注入一堆兄弟数据集名（如 CHIRPS query 被塞进 era5/prism/rain），
        # 混进"主题词"会污染 topic-cover 判定，故主题词只认原始 token。
        raw_query_tokens = set(query_tokens)
        # 方向 2：同义词扩展，提升 token overlap 召回率
        synonym_tokens = expand_query_synonyms(query_text)
        for token in synonym_tokens:
            if token and token not in MATCH_STOPWORDS:
                query_tokens.add(token)
        excluded_tokens: set[str] = set()

        raw_tokens = re.findall(r"[a-z0-9]+", query_text.lower())
        boundary_tokens = MATCH_STOPWORDS.union({"not"})
        for idx, token in enumerate(raw_tokens):
            if token != "not":
                continue
            next_tokens = raw_tokens[idx + 1: idx + 2]
            for candidate in next_tokens:
                if candidate in boundary_tokens:
                    continue
                excluded_tokens.add(candidate)
        if excluded_tokens:
            query_tokens = {token for token in query_tokens if token not in excluded_tokens}
            raw_query_tokens = {token for token in raw_query_tokens if token not in excluded_tokens}

        allowed_product_families = infer_product_families(query_text)
        theme = infer_theme(query_text)

        region_terms = extract_region_terms(
            query_text,
            explicit_region=raw_intent.get("explicit_region"),
        )
        expected_data_types: List[str] = []
        if "time series" in normalized_query:
            expected_data_types = ["image_collection"]
        elif any(term in normalized_query for term in ["mosaic", "mosaics", "classification", "land cover", "dem", "ndvi"]):
            expected_data_types = ["image", "image_collection"]
        elif any(term in normalized_query for term in ["boundary", "boundaries", "footprint", "footprints", "locations"]):
            expected_data_types = ["table"]

        time_terms: List[str] = []
        if "time series" in normalized_query:
            time_terms.append("time_series")
        for label, signatures in {
            "annual": ["annual", "yearly"],
            "monthly": ["monthly"],
            "daily": ["daily"],
            "historical": ["historical", "history", "long term", "longterm"],
        }.items():
            if any(signature in normalized_query for signature in signatures):
                time_terms.append(label)

        normalized_intent = {
            "query_text": query_text,
            "raw_query": query_text,
            "normalized_query": normalized_query,
            "query_tokens": query_tokens,
            "raw_query_tokens": raw_query_tokens,
            "excluded_tokens": sorted(excluded_tokens),
            "theme": raw_intent.get("theme") or theme,
            "region_terms": raw_intent.get("region_terms") or region_terms,
            "required_region": raw_intent.get("required_region") or (region_terms[0] if len(region_terms) == 1 else None),
            "allowed_product_families": raw_intent.get("allowed_product_families") or allowed_product_families,
            "preferred_family": raw_intent.get("preferred_family"),
            "aliases": raw_intent.get("aliases") or [],
            "dataset_name": raw_intent.get("dataset_name") or "",
            "canonical_requirement_name": raw_intent.get("canonical_requirement_name") or query_text,
            "requirement_identity_text": raw_intent.get("requirement_identity_text") or "",
            "explicit_dataset_ids": raw_intent.get("explicit_dataset_ids") or [],
            "retrieval_product_terms": raw_intent.get("retrieval_product_terms") or [],
            "frozen_constraints": raw_intent.get("frozen_constraints") or {},
            "soft_constraints": raw_intent.get("soft_constraints") or {},
            "expected_data_types": raw_intent.get("expected_data_types") or expected_data_types,
            "expected_data_types_explicit": bool(
                raw_intent.get("expected_data_types")
            ),
            "time_terms": raw_intent.get("time_terms") or time_terms,
            "temporal_coverage": raw_intent.get("temporal_coverage") or {},
            "temporal_resolution": raw_intent.get("temporal_resolution") or {},
            "search_k": max(1, int(raw_intent.get("search_k", 50) or 50)),
            "recall_min_score": float(raw_intent.get("recall_min_score", self.recall_min_score) or self.recall_min_score),
            "weak_bind_min_score": float(
                raw_intent.get("weak_bind_min_score", self.weak_bind_min_score) or self.weak_bind_min_score
            ),
            "strong_bind_min_score": float(
                raw_intent.get("strong_bind_min_score", self.strong_bind_min_score) or self.strong_bind_min_score
            ),
            "region_hard_filter": bool(raw_intent.get("region_hard_filter", self.region_hard_filter)),
            "time_hard_filter": bool(raw_intent.get("time_hard_filter", self.time_hard_filter)),
            "role_hard_filter": bool(raw_intent.get("role_hard_filter", self.role_hard_filter)),
        }
        return normalized_intent

    def _recall_local_candidates(self, intent) -> List[Dict[str, Any]]:
        search_limit = max(1, int(intent.get("search_k", 20) or 20))
        normalized_candidates = [
            self._normalize_local_candidate(feature, include_detail=False)
            for feature in self._load_catalog()
        ]
        normalized_candidates = [candidate for candidate in normalized_candidates if candidate]

        for candidate in normalized_candidates:
            relevance_result = self._score_text_relevance(intent, candidate)
            candidate["score_breakdown"] = relevance_result["score_breakdown"]
            candidate["text_score"] = relevance_result["total_score"]
        recall_min_score = float(intent.get("recall_min_score", self.recall_min_score) or self.recall_min_score)
        normalized_candidates = [
            candidate for candidate in normalized_candidates
            if candidate.get("text_score", 0.0) >= recall_min_score
        ]

        normalized_candidates.sort(
            key=lambda candidate: (
                1 if self._is_strict_feature_match(intent.get("query_text", ""), candidate["feature"]) else 0,
                candidate.get("text_score", 0.0),
            ),
            reverse=True,
        )
        return normalized_candidates[:search_limit]

    @staticmethod
    def _bbox_implies_global_land_coverage(value: Any) -> bool:
        """Thin delegate kept for existing call sites and tests.

        阈值与解析逻辑统一由 global_coverage_evidence.bbox_implies_global 持有，
        避免同一判据在两处各自维护后漂移。北界取 60 而非 75 的沿革见该模块注释与
        开发过程.md 8.4：SRTM 族受轨道倾角限制截止在 60°N/56°S，却是事实上的
        全球陆地 DEM 标准产品。
        """
        return bbox_implies_global(value)

    def _enrich_normalized_candidate(
        self,
        candidate: Dict[str, Any],
    ) -> Dict[str, Any]:
        if candidate.get("detail_loaded"):
            return candidate
        enriched_entry = self._dataset_detail_enricher.enrich(
            {
                "id": candidate.get("id"),
                "title": candidate.get("title"),
            }
        )
        detail = enriched_entry.get("detail") or {}
        temporal_extent = detail.get("temporal_extent") or []
        detail_coverage: Dict[str, Any] = {}
        if isinstance(temporal_extent, list) and len(temporal_extent) >= 2:
            detail_coverage = {
                "start": temporal_extent[0],
                "end": temporal_extent[1] or "present",
            }
        temporal_coverage = normalize_temporal_coverage(
            detail_coverage or candidate.get("temporal_coverage")
        )
        cadence = detail.get("cadence") or candidate.get("temporal_resolution")
        temporal_resolution = normalize_temporal_resolution(cadence)

        bands = detail.get("bands") or []
        band_text = " ".join(
            " ".join(
                str(part or "").strip()
                for part in [band.get("name"), band.get("description")]
                if str(part or "").strip()
            )
            for band in bands
            if isinstance(band, dict)
        )
        keywords = detail.get("keywords") or []
        if isinstance(keywords, str):
            keywords = [keywords]
        evidence_parts = [
            candidate.get("candidate_text", ""),
            detail.get("description", ""),
            " ".join(str(item) for item in keywords),
            band_text,
        ]
        enriched = dict(candidate)
        enriched["candidate_text"] = " ".join(
            str(part).strip() for part in evidence_parts if str(part).strip()
        )
        enriched["temporal_coverage"] = temporal_coverage
        enriched["temporal_resolution"] = temporal_resolution
        enriched["detail"] = detail
        enriched["detail_loaded"] = True
        return enriched

    def _normalize_local_candidate(
        self,
        feature,
        *,
        include_detail: bool = True,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(feature, GeoFeature):
            return None

        feature = self._attach_gee_runtime_metadata([feature])[0]
        props = feature.properties or {}
        title = str(props.get("title") or feature.name or "").strip()
        asset_id = str(props.get("id") or feature.feature_id or "").strip()
        candidate_type = str(props.get("type") or feature.feature_type or "").strip().lower()
        tags = props.get("tags", [])
        if not isinstance(tags, list):
            tags = [str(tags)]
        thematic_group = str(props.get("thematic_group") or "").strip()
        candidate_text = " ".join(
            part
            for part in [
                title,
                asset_id,
                str(props.get("ee_id") or "").strip(),
                " ".join(str(tag).strip() for tag in tags if str(tag).strip()),
                thematic_group,
                str(props.get("provider") or "").strip(),
                str(props.get("region_name") or "").strip(),
                str(props.get("description") or "").strip(),
            ]
            if part
        )
        # 区域词提取的输入刻意不含 description：长文本里的 "global" 证据强度远低于
        # 结构化 bbox，混进同一个词袋会让弱证据与强证据等价。全球覆盖改由分层判据
        # assess_global_coverage_evidence 单独裁定，其中 bbox 拥有一票否决权。
        candidate_regions = extract_region_terms(
            " ".join(
                part
                for part in [
                    title,
                    asset_id,
                    str(props.get("ee_id") or "").strip(),
                    " ".join(str(tag).strip() for tag in tags if str(tag).strip()),
                    thematic_group,
                    str(props.get("provider") or "").strip(),
                    str(props.get("region_name") or "").strip(),
                ]
                if part
            ),
            explicit_region=props.get("region_name"),
        )
        # 实测：description 声称 global 的 3761 条 catalog 记录里 468 条被自身 bbox
        # 证否（如 Allen Coral Atlas，description 写 global 而 bbox 仅 ±33° 珊瑚带）。
        # 旧实现把 description 拼进 candidate_text 再抽区域词，这批记录会被误判成
        # 全球覆盖；现在 bbox 存在且明确非全球时，文本声明一律作废。
        global_evidence = assess_global_coverage_evidence(
            bbox=props.get("bbox"),
            short_fields=[
                title,
                asset_id,
                " ".join(str(tag).strip() for tag in tags if str(tag).strip()),
            ],
            description=props.get("description"),
        )
        if global_evidence["proves_global"]:
            candidate_regions.append("global")
        elif global_evidence["bbox_refutes"]:
            # bbox 已证否全球，任何来源推断出的 global 都不得留在候选区域里。
            candidate_regions = [
                region for region in candidate_regions
                if str(region).strip().casefold() != "global"
            ]
        candidate_regions = list(dict.fromkeys(candidate_regions))
        raw_coverage = {
            "start": props.get("start_date") or props.get("state_date"),
            "end": props.get("end_date"),
        }
        candidate = {
            "feature": feature,
            "title": title,
            "id": asset_id,
            "candidate_type": candidate_type,
            "tags": [str(tag).strip() for tag in tags if str(tag).strip()],
            "thematic_group": thematic_group,
            "provider": str(props.get("provider") or "").strip(),
            "license": str(props.get("license") or "").strip(),
            "docs": props.get("docs"),
            "sample_code": props.get("sample_code"),
            "ee_id": str(props.get("ee_id") or asset_id).strip(),
            "ee_object_type": props.get("ee_object_type"),
            "ee_ctor": props.get("ee_ctor"),
            "source": feature.source,
            "candidate_text": candidate_text,
            "candidate_regions": candidate_regions,
            "temporal_coverage": normalize_temporal_coverage(raw_coverage),
            "temporal_resolution": normalize_temporal_resolution(
                props.get("cadence")
            ),
        }
        return self._enrich_normalized_candidate(candidate) if include_detail else candidate

    @staticmethod
    def _looks_like_gee_asset_id(text: str) -> bool:
        """判定一段文本是否形如 GEE 资产 id（可作探测身份）。

        与目录语义匹配不同，这里需要的是"声明侧给了明确在线身份"的证据。保守起见
        只认：①顶层是 projects/ 社区资产；②含 `/` 的官方/项目 id（如 ECMWF/ERA5_LAND/
        HOURLY）。形如 'Forest Stand 1'/'Arnsberg Region' 等地名实体不含 `/`，不会误判。
        """
        stripped = str(text or "").strip()
        if not stripped:
            return False
        if stripped.startswith(("http://", "https://")):
            return False
        if stripped.startswith("GEE:"):
            return True
        if stripped.startswith("projects/"):
            return True
        # 官方集合形如 COLLECTION/ID/…，至少一段 `/`；斜杠两侧不允许是普通地名，
        # 再叠加 id 不以空格开头的约束（地名很少含 `/`）。
        return "/" in stripped and not stripped[0].isspace()

    def _extract_probe_identity_ids(self, normalized_intent: Dict[str, Any]) -> List[str]:
        """从声明侧提取可作为 GEE 探测身份的资产 id。

        身份来源（改法②）：
        1. canonical_requirement_name：形如 'ECMWF/ERA5_LAND/HOURLY:temperature'，
           取冒号前的 id 部分（冒号后是 observable/维度切片，不是资产 id）。
        2. frozen_constraints.entity（字符串或列表）：仅当该项形如 GEE 资产 id 时
           纳入（地名实体『Forest Stand 1』等不含 `/`，被 _looks_like_gee_asset_id 排除）。

        只做"形如 id"的保守收集，不假定实体一定是在线资产；探测失败由调用方留痕放行。
        """
        collected: List[str] = []

        def _add(value):
            stripped = str(value or "").strip()
            if stripped and stripped not in collected:
                collected.append(stripped)

        canonical = str(normalized_intent.get("canonical_requirement_name") or "").strip()
        if canonical and ":" in canonical:
            head, _, _ = canonical.partition(":")
            head = head.strip()
            if self._looks_like_gee_asset_id(head):
                _add(head)

        frozen = normalized_intent.get("frozen_constraints") or {}
        entity = None
        if isinstance(frozen, dict):
            entity = frozen.get("entity")
        if isinstance(entity, str):
            entity_values = [entity]
        elif isinstance(entity, list):
            entity_values = [item for item in entity if isinstance(item, str)]
        else:
            entity_values = []
        for value in entity_values:
            if self._looks_like_gee_asset_id(value):
                _add(value)
        return collected

    def _candidate_matches_explicit_dataset_id(
        self,
        intent: Dict[str, Any],
        candidate: Dict[str, Any],
    ) -> bool:
        explicit_ids = intent.get("explicit_dataset_ids") or []
        if isinstance(explicit_ids, str):
            explicit_ids = [explicit_ids]
        explicit_ids = list(explicit_ids) + self._extract_probe_identity_ids(intent)
        normalized_explicit = {
            str(item).strip().casefold()
            for item in explicit_ids
            if str(item or "").strip()
        }
        if not normalized_explicit:
            return False
        candidate_ids = {
            str(item).strip().casefold()
            for item in [
                candidate.get("id"),
                candidate.get("ee_id"),
                candidate.get("source"),
            ]
            if str(item or "").strip()
        }
        return bool(normalized_explicit.intersection(candidate_ids))

    def _candidate_has_strong_title_hit(
        self,
        intent: Dict[str, Any],
        candidate: Dict[str, Any],
    ) -> bool:
        candidate_title = self._normalize_match_text(candidate.get("title", ""))
        if not candidate_title:
            return False
        title_tokens = self._tokenize_match_text(candidate_title)
        phrases = []
        for field in (
            "dataset_name",
        ):
            value = str(intent.get(field) or "").strip()
            if value:
                phrases.append(value)
        retrieval_terms = intent.get("retrieval_product_terms") or []
        if isinstance(retrieval_terms, str):
            retrieval_terms = [retrieval_terms]
        aliases = intent.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        phrases.extend(str(item).strip() for item in retrieval_terms if str(item).strip())
        phrases.extend(str(item).strip() for item in aliases[:5] if str(item).strip())
        for phrase in phrases:
            normalized_phrase = self._normalize_match_text(phrase)
            if not normalized_phrase:
                continue
            phrase_tokens = self._tokenize_match_text(normalized_phrase)
            if len(phrase_tokens) < 2:
                continue
            if normalized_phrase == candidate_title:
                return True
            if normalized_phrase in candidate_title or candidate_title in normalized_phrase:
                return True
            if phrase_tokens.issubset(title_tokens):
                return True
        return False

    def _candidate_is_trusted_identity_hit(
        self,
        intent: Dict[str, Any],
        candidate: Dict[str, Any],
    ) -> bool:
        return (
            self._candidate_matches_explicit_dataset_id(intent, candidate)
            or self._candidate_has_strong_title_hit(intent, candidate)
        )

    def _query_topic_terms(self, query_tokens: set) -> set:
        """
        从 query token 中挑出高判别力主题词（IDF 高于 query 内中位数的词）。
        用于识别"主题词全中、只差修饰词"的候选。IDF 表不可用时返回空集（信号关闭）。
        先剔除数据集修饰词（unconstrained/adjusted/final...）：它们罕见（IDF 高）但对
        数据集身份无区分意义，若混进主题词会让正确候选覆盖不全、topic-cover 失效。
        """
        idf_table = self._get_token_idf_table()
        if not idf_table or not query_tokens:
            return set()
        content_tokens = {t for t in query_tokens if t not in DATASET_MODIFIER_TERMS}
        if not content_tokens:
            content_tokens = set(query_tokens)
        default_idf = type(self)._token_idf_default
        idf_of = lambda tok: idf_table.get(tok, default_idf)
        sorted_idf = sorted(idf_of(t) for t in content_tokens)
        median_idf = sorted_idf[len(sorted_idf) // 2]
        return {t for t in content_tokens if idf_of(t) >= median_idf}

    def _candidate_covers_topic_terms(self, topic_terms: set, candidate: Dict[str, Any]) -> bool:
        """候选的 title/id/tags/thematic 文本是否完整覆盖 query 的高判别力主题词。"""
        if not topic_terms:
            return False
        candidate_tokens = self._tokenize_match_text(candidate.get("candidate_text", ""))
        return topic_terms.issubset(candidate_tokens)

    def _evaluate_candidate_set(self, intent, candidates) -> Dict[str, Any]:
        evaluated_candidates: List[Dict[str, Any]] = []
        blocking_reason_pool: List[str] = []
        semantic_candidate_found = False
        strong_bind_min_score = float(intent.get("strong_bind_min_score", self.strong_bind_min_score) or self.strong_bind_min_score)
        weak_bind_min_score = float(intent.get("weak_bind_min_score", self.weak_bind_min_score) or self.weak_bind_min_score)
        # top1 信心提升：识别 query 的高判别力主题词，候选若完整覆盖它们
        # （主题词全中、只差 final/density 等低 IDF 修饰词），放宽 weak 门槛，
        # 救回"排序已对但绝对分擦阈值"的候选。
        topic_terms = self._query_topic_terms(
            intent.get("raw_query_tokens") or intent.get("query_tokens", set())
        )
        topic_weak_floor = weak_bind_min_score * 0.75

        for recalled_candidate in candidates:
            candidate = self._enrich_normalized_candidate(recalled_candidate)
            score_result = self._score_text_relevance(intent, candidate)
            constraint_report = self._build_constraint_report(intent, candidate)
            hard_rejections = self._build_blocking_reasons(
                intent,
                constraint_report,
                candidate,
            )
            verification_gaps = self._build_verification_gaps(
                intent,
                constraint_report,
            )
            penalty_report = self._build_penalty_report(constraint_report)
            final_score = min(
                0.99,
                max(
                    0.0,
                    score_result["total_score"]
                    + score_result["score_breakdown"].get("family_bonus", 0.0)
                    + penalty_report["region_penalty"]
                    + penalty_report["time_penalty"]
                    + penalty_report["role_penalty"],
                ),
            )
            frozen_report = self._evaluate_frozen_constraints(intent, candidate, constraint_report)
            frozen_constraints_satisfied = bool(frozen_report["satisfied"])
            soft_reasons = list(frozen_report["reasons"])
            binding_grade = "unresolved_semantic"
            accepted_for_binding = False
            covers_topic = (
                not hard_rejections
                and self._candidate_covers_topic_terms(topic_terms, candidate)
            )
            # 判断是否为外部源（非 GEE）
            source_prefix = str(candidate.get("source", "")).split(":", 1)[0] if candidate.get("source") else ""
            # 去掉 _ONLINE 后缀（在线检索的 source 是 NASA_CMR_ONLINE: 等形式）
            source_prefix = source_prefix.replace("_ONLINE", "")
            is_external_source = source_prefix in {"NASA_CMR", "PLANETARY_COMPUTER", "EARTH_SEARCH", "WORLDCLIM", "OSM_GEOFABRIK", "OSM_TAGINFO"}

            if hard_rejections:
                binding_grade = "unresolved_semantic"
            elif (
                not verification_gaps
                and final_score >= strong_bind_min_score
                and frozen_constraints_satisfied
            ):
                binding_grade = "resolved_strong"
                accepted_for_binding = True
            elif final_score >= weak_bind_min_score:
                binding_grade = "resolved_weak"
                # 外部源：达到 weak 阈值即可绑定（它们缺详情，不强求冻结约束与高分）
                if is_external_source:
                    accepted_for_binding = True
                    soft_reasons.append("external_source_binding_relaxed")
                elif final_score >= strong_bind_min_score and not frozen_constraints_satisfied:
                    soft_reasons.append("frozen_constraints_not_satisfied")
            elif covers_topic and final_score >= topic_weak_floor:
                # 主题词全中、仅差低 IDF 修饰词：救到 weak，避免真实数据集被误判缺失
                binding_grade = "resolved_weak"
                soft_reasons.append(
                    f"topic_terms_fully_covered:weak_floor:{final_score:.2f}>={topic_weak_floor:.2f}"
                )
            else:
                binding_grade = "unresolved_semantic"
                soft_reasons.append(
                    f"binding_score_below_weak:{final_score:.2f}<{weak_bind_min_score:.2f}"
                )
            if not hard_rejections:
                semantic_candidate_found = True
            binding_decision = {
                "accepted_for_binding": accepted_for_binding,
                "binding_grade": binding_grade,
                "blocking_reasons": hard_rejections,
                "hard_rejections": hard_rejections,
                "verification_gaps": verification_gaps,
                "soft_reasons": soft_reasons,
                "constraint_report": constraint_report,
                "score_breakdown": score_result["score_breakdown"],
                "text_score": score_result["total_score"],
                "final_score": final_score,
                "penalty_report": penalty_report,
                "frozen_constraints_satisfied": frozen_constraints_satisfied,
            }
            evaluated_candidate = {
                **candidate,
                "binding_decision": binding_decision,
                "region_sort_tier": self._map_region_sort_tier(constraint_report["region_match"]),
                "text_score": score_result["total_score"],
                "final_score": final_score,
                "summary": {
                    "name": candidate["title"] or candidate["feature"].name,
                    "id": candidate["id"],
                    "type": candidate["candidate_type"],
                    "status": binding_grade,
                    "binding_decision": binding_decision,
                },
            }
            evaluated_candidates.append(evaluated_candidate)
            blocking_reason_pool.extend(hard_rejections or soft_reasons)

        deduped_blocking_reasons: List[str] = []
        seen_reasons = set()
        for reason in blocking_reason_pool:
            if reason in seen_reasons:
                continue
            seen_reasons.add(reason)
            deduped_blocking_reasons.append(reason)

        return {
            "evaluated_candidates": evaluated_candidates,
            "blocking_reasons": deduped_blocking_reasons or ["no_candidates"],
            "semantic_candidate_found": semantic_candidate_found,
        }

    def _rank_candidate_set(self, intent, evaluated_candidates) -> List[Dict[str, Any]]:
        return sorted(
            evaluated_candidates,
            key=lambda candidate: (
                1 if candidate["binding_decision"].get("binding_grade") == "resolved_strong" else 0,
                1 if candidate["binding_decision"].get("binding_grade") == "resolved_weak" else 0,
                1 if candidate["binding_decision"].get("frozen_constraints_satisfied") else 0,
                -candidate.get("region_sort_tier", 99),
                candidate.get("final_score", 0.0),
                1 if candidate["binding_decision"]["constraint_report"].get("theme_match") else 0,
                1 if candidate["binding_decision"]["constraint_report"].get("type_match") else 0,
            ),
            reverse=True,
        )

    def _score_text_relevance(self, intent, candidate) -> Dict[str, Any]:
        # 打分以原始 query token 为主导（不含 synonym 扩展）。synonym 表把通用主题词
        # 映射到竞品数据集 ID（如 precipitation→chirps/era5/prism），若直接进主打分，
        # 查 GPM IMERG 时 chirps 会帮 CHIRPS 候选抢到高分、挤掉真正的 IMERG。
        # 故：raw token 主导排序，synonym-only token 命中只给受限 bonus（封顶、不主导），
        # 既杜绝竞品污染，又保留"查 rainfall 也能召回 CHIRPS"的近义召回价值。
        query_tokens = intent.get("query_tokens", set())
        raw_query_tokens = intent.get("raw_query_tokens") or query_tokens
        synonym_only_tokens = set(query_tokens) - set(raw_query_tokens)

        raw_query = " ".join(sorted(raw_query_tokens))
        title_score = self._calculate_relevance(raw_query, candidate.get("title", ""))
        id_score = self._calculate_relevance(raw_query, candidate.get("id", ""))
        tags_score = self._calculate_relevance(raw_query, " ".join(candidate.get("tags", [])))
        thematic_group_score = self._calculate_relevance(
            raw_query,
            candidate.get("thematic_group", ""),
        )
        family_bonus = 0.15 if detect_family_matches(intent, candidate.get("candidate_text", "")) else 0.0

        # synonym bonus：synonym-only token 命中候选文本时给小幅加分（最高 0.15），
        # 让纯近义词 query 仍能召回相关数据集，但压不过 raw token 的直接匹配。
        synonym_bonus = 0.0
        if synonym_only_tokens:
            candidate_tokens = self._tokenize_match_text(candidate.get("candidate_text", ""))
            synonym_hits = synonym_only_tokens & candidate_tokens
            if synonym_hits:
                synonym_bonus = min(0.15, 0.05 * len(synonym_hits))

        score_breakdown = {
            "title": title_score,
            "id": id_score,
            "tags": tags_score,
            "thematic_group": thematic_group_score,
            "family_bonus": family_bonus,
            "synonym_bonus": synonym_bonus,
        }
        total_score = min(
            0.99,
            (0.55 * title_score)
            + (0.20 * id_score)
            + (0.15 * tags_score)
            + (0.10 * thematic_group_score)
            + synonym_bonus
        )
        return {
            "score_breakdown": score_breakdown,
            "total_score": total_score,
        }

    def _build_constraint_report(self, intent, candidate) -> Dict[str, Any]:
        region_match = self._classify_candidate_region_match(intent, candidate)
        temporal_coverage_match = classify_temporal_coverage(
            intent.get("temporal_coverage"),
            candidate.get("temporal_coverage"),
        )
        temporal_resolution_match = classify_temporal_resolution(
            intent.get("temporal_resolution"),
            candidate.get("temporal_resolution"),
        )
        time_match = self._classify_candidate_time_match(
            temporal_coverage_match,
            temporal_resolution_match,
        )
        type_match = detect_type_match(intent, candidate.get("candidate_type", ""))
        theme_match = detect_theme_match(intent, candidate.get("candidate_text", ""))
        family_match = (
            True
            if not intent.get("allowed_product_families")
            else bool(detect_family_matches(intent, candidate.get("candidate_text", "")))
        )
        return {
            "region_match": region_match,
            "time_match": time_match,
            "temporal_coverage_match": temporal_coverage_match,
            "temporal_resolution_match": temporal_resolution_match,
            "role_match": "acceptable",
            "type_match": type_match,
            "theme_match": theme_match,
            "family_match": family_match,
        }

    def _build_blocking_reasons(
        self,
        intent: Dict[str, Any],
        constraint_report: Dict[str, Any],
        candidate: Dict[str, Any],
    ) -> List[str]:
        blocking_reasons: List[str] = []
        if constraint_report["region_match"] == "conflict":
            blocking_reasons.append("region_conflict")
        # 要求全球覆盖时，"候选声明了别的区域"与"候选没有区域信息"必须分开：
        # 前者是正向矛盾（region_match == "conflict"，已由上一条拦下），后者只是
        # metadata 缺口。GEBCO 全球水深格网这类数据集真实存在且全球覆盖是其定义
        # 属性，只因 catalog metadata 未写 region 就被硬拦属于误杀；按项目边界，
        # metadata gap 应留在 verification_gaps 供下游判断，不得当成已确认 conflict。
        if (
            "global" in (intent.get("region_terms") or [])
            and constraint_report["region_match"] not in {"global", "unspecified"}
        ):
            blocking_reasons.append("global_coverage_unverified")
        if intent.get("time_hard_filter", True):
            coverage_reason = {
                "no_overlap": "temporal_no_overlap",
                "incomplete": "temporal_coverage_incomplete",
                "invalid_requirement": "temporal_coverage_invalid",
            }.get(constraint_report.get("temporal_coverage_match"))
            if coverage_reason:
                blocking_reasons.append(coverage_reason)
            resolution_reason = {
                "incompatible": "temporal_resolution_incompatible",
                "invalid_requirement": "temporal_resolution_invalid",
            }.get(constraint_report.get("temporal_resolution_match"))
            if resolution_reason:
                blocking_reasons.append(resolution_reason)
        if not constraint_report["type_match"] and intent.get("expected_data_types"):
            blocking_reasons.append("type_mismatch_hard")
        if intent.get("allowed_product_families") and not constraint_report["family_match"]:
            blocking_reasons.append("family_mismatch_hard")
        blocking_reasons.extend(
            self._build_frozen_identity_blocking_reasons(intent, candidate)
        )
        return blocking_reasons

    @staticmethod
    def _build_verification_gaps(
        intent: Dict[str, Any],
        constraint_report: Dict[str, Any],
    ) -> List[str]:
        verification_gaps: List[str] = []
        # 空间缺口与 time_hard_filter 无关，必须在时间开关之外先记录：候选未声明
        # 任何区域时全球覆盖既未被证明也未被证伪，降级放行后要留下可审计的缺口，
        # 否则就成了静默放行。
        if (
            "global" in (intent.get("region_terms") or [])
            and constraint_report.get("region_match") == "unspecified"
        ):
            verification_gaps.append("global_coverage_unverified")
        if not intent.get("time_hard_filter", True):
            return verification_gaps
        if constraint_report.get("temporal_coverage_match") == "unverified":
            verification_gaps.append("temporal_coverage_unverified")
        if constraint_report.get("temporal_resolution_match") == "unverified":
            verification_gaps.append("temporal_resolution_unverified")
        return verification_gaps

    def _build_frozen_identity_blocking_reasons(
        self,
        intent: Dict[str, Any],
        candidate: Dict[str, Any],
    ) -> List[str]:
        frozen_constraints = intent.get("frozen_constraints") or {}
        if not isinstance(frozen_constraints, dict):
            return []

        candidate_text = self._normalize_match_text(candidate.get("candidate_text", ""))
        candidate_tokens = self._tokenize_match_text(candidate_text)
        trusted_identity_hit = self._candidate_is_trusted_identity_hit(intent, candidate)
        aliases = intent.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]

        reasons: List[str] = []
        ambiguous_acronyms: set[str] = set()
        for dimension in ("entity", "observable", "measure"):
            values, match_mode = normalize_frozen_identity_dimension(
                frozen_constraints.get(dimension)
            )
            unmatched: List[str] = []
            matched_any = False
            for value in values:
                normalized_value = self._normalize_match_text(value)
                if not normalized_value:
                    continue
                required_tokens = self._tokenize_match_text(normalized_value)
                exact_phrase_match = normalized_value in candidate_text
                token_match = bool(required_tokens) and required_tokens.issubset(candidate_tokens)
                if exact_phrase_match or token_match:
                    matched_any = True
                    continue

                if trusted_identity_hit:
                    # A concrete catalog id or strong title hit is allowed to carry
                    # semantically compatible fields whose exact frozen wording is
                    # absent from catalog metadata. Clear family/time/type/region
                    # conflicts are still produced by the other hard gates.
                    continue

                unmatched.append(normalized_value)
                if len(required_tokens) < 2:
                    continue
                expected_acronym = "".join(
                    token[0] for token in normalized_value.split() if token
                )
                for alias in aliases:
                    alias_text = str(alias or "").strip()
                    normalized_alias = self._normalize_match_text(alias_text)
                    if (
                        alias_text.isupper()
                        and normalized_alias == expected_acronym
                        and normalized_alias in candidate_tokens
                    ):
                        ambiguous_acronyms.add(normalized_alias)

            # `any` 是缺省语义：同一维度的多个取值通常是同义写法或互斥备选源
            # （`[soil_moisture, water_content]`、`[GRACE, GRACE-FO, GLDAS, CLSM]`），
            # 逐项 AND 会让命中正确项的候选被同义项否掉，而互斥源列表在 AND 下
            # 物理上无法满足。复合量（气候水平衡 = P - ET）必须由 requirement
            # 显式声明 `match="all"` 才逐项校验。
            if match_mode == FROZEN_IDENTITY_MATCH_ALL or not matched_any:
                reasons.extend(
                    f"frozen_{dimension}_unverified:{value.replace(' ', '_')}"
                    for value in unmatched
                )

        reasons.extend(
            f"ambiguous_acronym_only_match:{acronym}"
            for acronym in sorted(ambiguous_acronyms)
        )
        return reasons

    def _build_penalty_report(self, constraint_report: Dict[str, Any]) -> Dict[str, float]:
        return {
            "region_penalty": REGION_SCORE_TABLE.get(constraint_report.get("region_match", ""), 0.0),
            "time_penalty": TIME_SCORE_TABLE.get(constraint_report.get("time_match", ""), 0.0),
            "role_penalty": 0.0,
        }

    def _evaluate_frozen_constraints(
        self,
        intent: Dict[str, Any],
        candidate: Dict[str, Any],
        constraint_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        frozen_constraints = intent.get("frozen_constraints") or {}
        if not isinstance(frozen_constraints, dict) or not frozen_constraints:
            return {"satisfied": True, "reasons": []}

        reasons: List[str] = []
        frozen_region = str(frozen_constraints.get("region", "") or "").strip()
        if frozen_region and constraint_report.get("region_match") == "conflict":
            reasons.append("frozen_region_conflict")

        frozen_modality = str(frozen_constraints.get("modality", "") or "").strip().lower()
        intent_modality = str(intent.get("modality", "") or "").strip().lower()
        candidate_type = str(candidate.get("candidate_type", "") or "").strip().lower()
        if frozen_modality and intent_modality and frozen_modality != intent_modality:
            reasons.append("frozen_modality_mismatch")

        frozen_candidate_types = frozen_constraints.get("candidate_type") or []
        if isinstance(frozen_candidate_types, str):
            frozen_candidate_types = [frozen_candidate_types]
        normalized_candidate_types = {
            str(item).strip().lower() for item in frozen_candidate_types if str(item).strip()
        }
        if normalized_candidate_types and candidate_type and candidate_type not in normalized_candidate_types:
            reasons.append("frozen_candidate_type_mismatch")

        return {"satisfied": len(reasons) == 0, "reasons": reasons}

    def _classify_candidate_region_match(self, intent, candidate) -> str:
        query_regions = intent.get("region_terms", [])
        if not query_regions:
            return "unspecified"

        candidate_regions = candidate.get("candidate_regions", [])
        best_match = "unspecified"
        for required_region in query_regions:
            match_result = classify_region_match(required_region, candidate_regions)
            status = match_result.get("status")
            if status == "conflict":
                return "conflict"
            if status == "global":
                best_match = "global" if best_match == "unspecified" else best_match
                continue
            if status != "compatible":
                continue
            matched_regions = set(match_result.get("matched_regions", []))
            if required_region in matched_regions:
                return "exact"
            best_match = "compatible_parent"
        return best_match

    def _classify_candidate_time_match(
        self,
        temporal_coverage_match: str,
        temporal_resolution_match: str,
    ) -> str:
        if "invalid_requirement" in {
            temporal_coverage_match,
            temporal_resolution_match,
        }:
            return "conflict"
        if temporal_coverage_match == "no_overlap" or temporal_resolution_match == "incompatible":
            return "conflict"
        if temporal_coverage_match == "incomplete":
            return "partial"
        if "unverified" in {
            temporal_coverage_match,
            temporal_resolution_match,
        }:
            return "unknown"
        if (
            temporal_coverage_match in {"full_cover", "overlap", "not_required"}
            and temporal_resolution_match in {"exact", "aggregatable", "not_required"}
        ):
            if temporal_coverage_match == temporal_resolution_match == "not_required":
                return "unknown"
            return "exact"
        return "unknown"

    def _map_region_sort_tier(self, region_match: str) -> int:
        return {
            "exact": 0,
            "compatible_parent": 1,
            "global": 2,
            "unspecified": 3,
            "conflict": 4,
        }.get(region_match, 5)

    def _normalize_match_text(self, text: str) -> str:
        tokens = re.findall(r"[a-z0-9]+", str(text or "").lower())
        return " ".join(tokens)

    def _tokenize_match_text(self, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", str(text or "").lower())
            if token and token not in MATCH_STOPWORDS
        }

    def _is_strict_feature_match(self, query: str, feature: GeoFeature) -> bool:
        normalized_query = self._normalize_match_text(query)
        if not normalized_query:
            return False

        props = feature.properties or {}
        candidates = [
            feature.name,
            props.get("title"),
            props.get("id"),
            props.get("ee_id"),
        ]
        return any(
            self._normalize_match_text(candidate) == normalized_query
            for candidate in candidates
            if candidate
        )

    def _score_feature_relevance(
        self,
        query: str,
        feature: GeoFeature,
        parsed_query: Optional[Dict[str, Any]] = None,
    ) -> float:
        if self._is_strict_feature_match(query, feature):
            return 1.0

        parsed = parsed_query or self._build_local_resolution_intent(query)
        props = feature.properties or {}
        title_text = " ".join(
            str(part)
            for part in [feature.name, props.get("title")]
            if part
        )
        id_text = " ".join(
            str(part)
            for part in [props.get("id"), props.get("ee_id")]
            if part
        )
        tags_text = " ".join(props.get("tags", []))
        group_text = str(props.get("thematic_group") or "")

        query_tokens = parsed.get("query_tokens", set())
        title_score = self._calculate_relevance(" ".join(sorted(query_tokens)), title_text)
        id_score = self._calculate_relevance(" ".join(sorted(query_tokens)), id_text)
        tags_score = self._calculate_relevance(" ".join(sorted(query_tokens)), tags_text)
        group_score = self._calculate_relevance(" ".join(sorted(query_tokens)), group_text)

        family_bonus = 0.0
        family_matches = detect_family_matches(
            parsed,
            " ".join([title_text, id_text, tags_text, group_text]),
        )
        if family_matches:
            family_bonus = 0.15

        return min(
            0.99,
            (0.55 * title_score)
            + (0.20 * id_score)
            + (0.15 * tags_score)
            + (0.10 * group_score)
            + family_bonus,
        )

    def _calculate_relevance(self, query: str, description: str) -> float:
        """
        计算查询词与数据描述的相关性得分。

        IDF 加权 F 型：结合 query 覆盖率（recall）与判别力主题词命中（precision）。
        - recall = 命中词 IDF 之和 / query 词 IDF 之和（罕见主题词主导，修饰词弱化）
        - topic  = 命中的高判别力词 IDF 之和 / query 高判别力词 IDF 之和
          （只看主题词是否被覆盖，屏蔽 final/density/precipitation 这类低 IDF 修饰词）
        取二者加权，让"主题词全中但修饰词没中"的候选拿到高分。
        IDF 表不可用（catalog 为空）时退化回原始召回率公式，保证可回落、不崩。
        """
        if not description:
            return 0.0

        query_words = self._tokenize_match_text(query)
        if not query_words:
            return 0.0
        desc_words = self._tokenize_match_text(description)
        intersection = query_words.intersection(desc_words)

        idf_table = self._get_token_idf_table()
        if not idf_table:
            return len(intersection) / len(query_words)

        default_idf = type(self)._token_idf_default
        query_weight = sum(idf_table.get(token, default_idf) for token in query_words)
        if query_weight <= 0.0:
            return len(intersection) / len(query_words)
        hit_weight = sum(idf_table.get(token, default_idf) for token in intersection)
        recall = hit_weight / query_weight

        # 高判别力主题词：IDF 高于 query 内中位数的词。屏蔽低 IDF 修饰词后，
        # 只要候选覆盖了主题词就给高 topic 分。
        idf_of = lambda tok: idf_table.get(tok, default_idf)
        sorted_idf = sorted(idf_of(t) for t in query_words)
        median_idf = sorted_idf[len(sorted_idf) // 2]
        topic_terms = {t for t in query_words if idf_of(t) >= median_idf}
        topic_weight = sum(idf_of(t) for t in topic_terms)
        if topic_weight > 0.0:
            topic_hit = sum(idf_of(t) for t in topic_terms & intersection)
            topic = topic_hit / topic_weight
        else:
            topic = recall

        return max(recall, 0.5 * recall + 0.5 * topic)

    def _check_spatial_overlap(self, feature_geom: Dict, filter_geom: Dict) -> bool:
        """
        利用 Shapely 进行空间相交验证
        """
        try:
            f_shape = shape(feature_geom)
            filter_shape = shape(filter_geom)
            return f_shape.intersects(filter_shape)
        except Exception as e:
            logger.error(f"空间过滤计算错误: {e}")
            return False
        
    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("AIOHTTP session closed.")

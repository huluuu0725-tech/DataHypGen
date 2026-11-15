"""
core.py · 共享数据结构、task_type 枚举、LLM 标注接口
"""

from __future__ import annotations
import re
from dataclasses import dataclass, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# task_type 枚举（地理领域）
# ---------------------------------------------------------------------------

TASK_TYPE_KEYWORDS: dict[str, list[str]] = {
    "spatial_join":           ["sjoin", "spatial join", "join nearest",
                               "merge vector", "spatial merge"],
    "raster_clip":            ["clip", "crop", "mask", "cutout", "crop_image",
                               "raster_geometry_mask"],
    "raster_reproject":       ["reproject", "warp", "resample crs", "transform crs",
                               "aligned_target", "calculate_default_transform"],
    "raster_resample":        ["resample", "resize", "rescale", "zoom"],
    "raster_mosaic":          ["mosaic", "merge raster", "composite", "copy_first",
                               "copy_last", "copy_max", "copy_min", "copy_sum"],
    "raster_band_math":       ["band math", "ndvi", "ndwi", "evi", "spectral index",
                               "band ratio", "normalize", "stack bands",
                               "normalized_diff", "stack_raster", "bytescale",
                               "rasterize"],
    "temporal_aggregation":   ["temporal", "resample time", "groupby time",
                               "monthly mean", "annual mean", "time series aggregate",
                               "open_mfdataset", "rolling", "isel"],
    "temporal_trend":         ["trend", "mann-kendall", "sen slope", "mannkendall"],
    "temporal_decomposition": ["decompose", "seasonal", "stl", "anomaly", "detrend"],
    "spatial_autocorrelation":["moran", "autocorrelation", "spatial lag",
                               "spatial weights", "geary", "getis", "join_counts",
                               "queen", "rook", "contiguity", "distance band"],
    "spatial_regression":     ["spatial regression", "lag model", "error model",
                               "geographically weighted", "gwr", "ml_lag",
                               "ml_error", "twosls", "spreg"],
    "crs_transform":          ["crs", "projection", "epsg", "to_crs",
                               "from_epsg", "from_wkt", "to_wkt", "from_proj4",
                               "transformer", "coordinate reference", "datum",
                               "ellipsoid", "is_projected", "is_geographic",
                               "utm_zone", "pyproj"],
    "dem_analysis":           ["dem", "elevation", "slope", "aspect", "hillshade",
                               "terrain", "curvature", "roughness", "tpi",
                               "richdem", "terrainattribute"],
    "hydrology":              ["hydrology", "watershed", "flow accumulation",
                               "flow direction", "depression fill", "stream",
                               "flowaccumulation", "filldepressions", "basin"],
    "spectral_index":         ["ndvi", "ndwi", "evi", "savi", "nbr",
                               "normalized difference", "band index"],
    "land_classification":    ["classify", "classification", "land cover",
                               "supervised", "unsupervised", "cluster",
                               "random forest", "kmeans", "land use"],
    "change_detection":       ["change detection", "bitemporal", "before after"],
    "network_analysis":       ["network", "graph", "shortest path", "routing",
                               "centrality", "osm", "graph_from", "street",
                               "isochrone", "betweenness", "travel time"],
    "point_interpolation":    ["interpolat", "kriging", "idw", "griddata", "tin",
                               "inverse distance"],
    "zonal_statistics":       ["zonal", "zonal_stat", "extract by mask",
                               "zonal_stats", "point_query"],
    "buffer_analysis":        ["buffer", "voronoi", "thiessen"],
    "overlay_analysis":       ["overlay", "union", "dissolve", "clip vector", "erase"],
    "geometry_operation":     ["centroid", "convex hull", "simplify", "affine",
                               "rotate", "translate", "skew", "delaunay",
                               "polygonize", "line_merge", "shared_paths",
                               "snap", "unary_union", "strtree", "polylabel",
                               "substring", "split"],
    "data_format_conversion": ["read_file", "to_file", "open_dataset",
                               "geojson", "shapefile", "netcdf", "from_file",
                               "wkt", "wkb", "plotting_extent", "reshape_as",
                               "rasterio.open"],
    "catalog_search":         ["stac", "catalog", "collection", "item", "search",
                               "pystac", "provider", "asset", "spatial_extent",
                               "temporal_extent", "link"],
    "visualization":          ["plot", "visualiz", "render", "explore", "show(",
                               "draw_legend", "colorbar", "plot_bands", "plot_rgb",
                               "show_hist", "adjust_band", "choropleth"],
    "data_quality_check":     ["validate", "is_valid", "crs_check", "missing",
                               "fix geometry", "extent_to_json"],
}

# 数据类型关键词
INPUT_TYPE_KEYWORDS: dict[str, list[str]] = {
    "vector":           ["geodataframe", "shapefile", "geojson", "polygon",
                         "point", "linestring", "vector", "feature"],
    "raster":           ["raster", "tiff", "geotiff", "band", "pixel",
                         "gridded", "array", "image"],
    "raster_timeseries":["xarray", "dataset", "netcdf", "datacube",
                         "time series raster"],
    "table":            ["dataframe", "csv", "tabular", "attribute", "column"],
    "spatial_weights":  ["weights", "w matrix", "queen", "rook", "knn"],
    "graph":            ["graph", "network", "node", "edge"],
    "file":             ["file", "path", "filename", "read"],
}

OUTPUT_TYPE_KEYWORDS: dict[str, list[str]] = {
    "vector":           ["geodataframe", "vector", "polygon", "point"],
    "raster":           ["raster", "array", "numpy", "band"],
    "raster_timeseries":["xarray", "dataset", "datacube"],
    "table":            ["dataframe", "dict", "list", "table"],
    "statistic":        ["moran", "statistic", "p-value", "result", "score"],
    "model":            ["model", "classifier", "regressor", "fitted"],
    "graph":            ["graph", "network"],
    "plot":             ["figure", "axes", "plot", "map"],
    "path":             ["path", "route", "list of nodes"],
}

DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "urban":          ["urban", "city", "street", "building", "land use"],
    "ecology":        ["ecology", "habitat", "species", "vegetation", "biodiversity"],
    "climate":        ["climate", "temperature", "precipitation", "weather",
                       "atmospheric"],
    "hydrology":      ["hydrology", "water", "river", "basin", "watershed",
                       "flood"],
    "remote_sensing": ["remote sensing", "satellite", "landsat", "modis",
                       "sentinel", "imagery"],
    "agriculture":    ["agriculture", "crop", "farm", "soil"],
    "geomorphology":  ["geomorphology", "terrain", "elevation", "dem", "slope"],
    "transportation": ["transport", "road", "route", "network", "traffic"],
    "land_cover":     ["land cover", "land use", "classification", "lulc"],
    "oceanography":   ["ocean", "sea", "marine", "coastal"],
    "general":        [],  # 兜底标签，始终附加
}


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class RawFunction:
    """爬取后的原始函数信息，未经 LLM 标注"""
    function_name: str        # 完整调用路径
    package: str              # 包名（不含版本）
    raw_description: str      # 文档字符串或描述文本
    signature: str            # 函数签名字符串
    source_url: str           # 来源 URL
    source_type: str          # official_doc | task_lib | cookbook
    layer: int                # 1 | 2 | 3


@dataclass
class IndexRecord:
    function_name: str
    package: str              # 包名+版本
    task_type: str
    input_data_type: list
    output_data_type: str
    domain_tags: list
    signature: str
    pip_verified: bool = False
    confidence: float = 0.0
    source_type: str = "official_doc"
    source_url: str = ""
    raw_description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# 基于关键词的自动标注（无需 LLM，作为基础标注层）
# ---------------------------------------------------------------------------

def infer_task_type(text: str) -> str:
    text_lower = text.lower()
    scores: dict[str, int] = {}
    for task, keywords in TASK_TYPE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score:
            scores[task] = score
    if not scores:
        return "general"
    return max(scores, key=lambda k: scores[k])


def infer_input_types(text: str) -> list[str]:
    text_lower = text.lower()
    found = []
    for dtype, keywords in INPUT_TYPE_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            found.append(dtype)
    return found if found else ["unknown"]


def infer_output_type(text: str) -> str:
    text_lower = text.lower()
    for dtype, keywords in OUTPUT_TYPE_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return dtype
    return "unknown"


def infer_domain_tags(text: str) -> list[str]:
    text_lower = text.lower()
    tags = ["general"]
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if domain == "general":
            continue
        if any(kw in text_lower for kw in keywords):
            tags.append(domain)
    return list(set(tags))


def auto_annotate(raw: RawFunction) -> dict:
    """用关键词规则对 RawFunction 做全字段标注，返回标注结果字典。"""
    combined = f"{raw.function_name} {raw.raw_description} {raw.signature}"
    return {
        "task_type":        infer_task_type(combined),
        "input_data_type":  infer_input_types(combined),
        "output_data_type": infer_output_type(combined),
        "domain_tags":      infer_domain_tags(combined),
    }


# ---------------------------------------------------------------------------
# 置信度计算
# ---------------------------------------------------------------------------

def compute_confidence(
    pip_verified: bool,
    source_type: str,
    layer: int,
    has_signature: bool,
) -> float:
    if pip_verified:
        base = 0.92
    elif source_type == "official_doc":
        base = 0.70
    elif source_type == "task_lib":
        base = 0.62
    elif source_type == "cookbook":
        base = 0.55
    else:
        base = 0.40

    # 层级加成
    if layer == 1:
        base = min(base + 0.05, 1.0)
    elif layer == 3:
        base = max(base - 0.05, 0.0)

    if pip_verified and has_signature:
        base = min(base + 0.05, 1.0)

    return round(base, 2)
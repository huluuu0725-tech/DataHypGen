"""
数据集画像智能体 (DatasetProfilerAgent)

在文献检索前运行，自动读取数据文件并对输入数据集进行描述性分析，
生成结构化的"数据集画像"（Dataset Portrait），作为强制上下文注入到
后续所有 idea 生成提示中。

设计目标：
  - 确保 idea 锚定在"这个数据集能回答什么问题"，而非"该领域通用研究方向"
  - 兼容任意数据类型：分类栅格 / 连续值栅格 / 向量 / 时序 CSV 等
  - 无需 rasterio（使用 struct + numpy 解析 GeoTIFF）；若 rasterio 可用则自动切换
  - 输出标准化 portrait dict，可序列化后注入 orchestration context

流程位置：
  orchestration pipeline:
    [DatasetProfilerAgent]   ← 新增，在 SurveyAgent 之前
      ↓ dataset_portrait
    [SurveyAgent]
      ↓ survey_results
    [DomainKnowledgeAgent]
      ↓ domain_knowledge
    [PlanAgent]
      ...
"""

import csv
import asyncio
import copy
import hashlib
import json
import logging
import os
import re
import struct
import math
import threading
import time
import warnings
from contextlib import nullcontext
from datetime import datetime
from urllib.request import Request, urlopen
from urllib.parse import quote, unquote, urlparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..tools.gee_auth import initialize_gee_from_service_account
from ..tools.sentinel_safe_reader import SentinelSAFEReader
from ..tools.dataset_source_clients import (
    query_earth_search,
    query_nasa_cmr,
    query_osm_overpass,
    query_osm_taginfo,
    get_planetary_computer_sas_token,
    query_planetary_computer,
    sign_planetary_computer_item,
)

logger = logging.getLogger(__name__)

# 向量要素属性采样：取前 N 个要素的属性实测值。此前 _summarize_gee_target 只保留属性
# 键名、丢掉值，导致"某属性的单位/编码含义是什么"这类问题在画像阶段无解，一路穿到
# idea 成熟化后以零召回收口。N 取小值即可定论单位，且只发一次 getInfo。
VECTOR_ATTRIBUTE_SAMPLE_SIZE = 3
# geometry().area() 的容许误差（米）。面积只用于反推单位量级，不需要高精度。
VECTOR_AREA_MAX_ERROR_M = 10
# declared/实测m² 的比值与理论值的容许倍数。要素几何经过简化、且投影不同会带来偏差，
# 但 m²/公顷/km² 三档之间相差 1e4 量级，5 倍容忍足以区分且不会误判。
VECTOR_AREA_UNIT_TOLERANCE = 5.0

EXTERNAL_PROFILE_SOURCES = frozenset({
    "PANGAEA",
    "NASA_CMR",
    "PLANETARY_COMPUTER",
    "EARTH_SEARCH",
    "OSM_TAGINFO",
    "OSM_OVERPASS",
})

NASA_DAAC_S3_CREDENTIAL_ENDPOINTS = {
    "ornl-cumulus-prod-protected": (
        "https://data.ornldaac.earthdata.nasa.gov/s3credentials"
    ),
}

NASA_DAAC_HTTPS_DISTRIBUTION_ENDPOINTS = {
    "ornl-cumulus-prod-protected": (
        "https://data.ornldaac.earthdata.nasa.gov/protected"
    ),
}


def _normal_two_sided_pvalue(z_score: float) -> float:
    return float(math.erfc(abs(z_score) / math.sqrt(2.0)))

# ── 可选依赖（有则用，无则降级） ─────────────────────────────────────────────
try:
    import rasterio
    from rasterio.windows import Window
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False

try:
    import boto3
    from rasterio.session import AWSSession
    _HAS_BOTO3 = True
except ImportError:
    boto3 = None
    AWSSession = None
    _HAS_BOTO3 = False

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

try:
    import shapefile as pyshp
    _HAS_PYSHP = True
except ImportError:
    _HAS_PYSHP = False

try:
    import xarray as xr
    _HAS_XARRAY = True
except ImportError:
    _HAS_XARRAY = False


@dataclass
class ResolvedProfilerConfig:
    dataset_type: str
    dataset_name: str
    topic: str
    region: str
    temporal_scope: str
    spatial_resolution: str
    pixel_area_m2: float
    file_paths: List[str] = field(default_factory=list)
    class_labels: Dict[int, str] = field(default_factory=dict)
    hotspot_target_classes: List[int] = field(default_factory=list)
    nodata_classes: List[int] = field(default_factory=list)
    remote_source: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 通用文件读取层（Reader Registry）
# ═══════════════════════════════════════════════════════════════════════════════

class RasterReader:
    """
    统一的栅格读取接口。
    优先使用 rasterio；若不可用，则用轻量 GeoTIFF 解析器（仅支持
    单波段、标准 TIFF，适合 MTBS 这类 uint8/int16 分类栅格）。
    返回：(data: np.ndarray [H, W], nodata: Optional[float], meta: dict)
    """

    @classmethod
    def read(
        cls,
        path: str,
        band: int = 1,
        max_pixels: int = 4_000_000,  # 降采样上限，防止内存溢出
    ) -> Tuple[np.ndarray, Optional[float], Dict[str, Any]]:
        if _HAS_RASTERIO:
            return cls._read_rasterio(path, band, max_pixels)
        return cls._read_minimal_tiff(path, max_pixels)

    @classmethod
    def _read_rasterio(cls, path, band, max_pixels):
        with rasterio.open(path) as src:
            meta = {
                "width": src.width,
                "height": src.height,
                "crs": str(src.crs),
                "transform": list(src.transform),
                "count": src.count,
                "dtype": str(src.dtypes[band - 1]),
                "nodata": src.nodata,
            }
            nodata = src.nodata
            # 降采样
            total = src.width * src.height
            if total > max_pixels:
                scale = math.sqrt(max_pixels / total)
                out_h = max(1, int(src.height * scale))
                out_w = max(1, int(src.width * scale))
                data = src.read(
                    band,
                    out_shape=(out_h, out_w),
                    resampling=rasterio.enums.Resampling.nearest,
                )
            else:
                data = src.read(band)
        return data, nodata, meta

    @classmethod
    def _read_minimal_tiff(cls, path, max_pixels):
        """
        最小化 GeoTIFF 解析器，不依赖 rasterio。
        支持：LZW 压缩 / 无压缩，uint8 / uint16 / int16 / float32 单波段。
        若解析失败抛出 ValueError；调用方负责 fallback。
        """
        try:
            import tifffile  # 常见于 scientific Python 环境
            data = tifffile.imread(path)
            if data.ndim == 3:
                data = data[0]  # 取第一波段
            nodata = None
            meta = {"width": data.shape[1], "height": data.shape[0],
                    "dtype": str(data.dtype), "source": "tifffile"}
        except ImportError:
            # 使用纯 struct 解析 TIFF IFD，获取宽高和像素数据
            data, meta = cls._parse_tiff_struct(path, max_pixels=max_pixels)
            nodata = None

        # 降采样
        h, w = data.shape
        if h * w > max_pixels:
            step = math.ceil(math.sqrt(h * w / max_pixels))
            data = data[::step, ::step]
        return data, nodata, meta

    @staticmethod
    def _decode_packbits(encoded: bytes) -> bytes:
        '''Decode one TIFF PackBits strip and reject truncated runs.'''
        decoded = bytearray()
        offset = 0
        while offset < len(encoded):
            header = encoded[offset]
            offset += 1
            if header <= 127:
                end = offset + header + 1
                if end > len(encoded):
                    raise ValueError('Truncated PackBits literal run')
                decoded.extend(encoded[offset:end])
                offset = end
            elif header >= 129:
                if offset >= len(encoded):
                    raise ValueError('Truncated PackBits repeat run')
                decoded.extend(encoded[offset:offset + 1] * (257 - header))
                offset += 1
        return bytes(decoded)

    @classmethod
    def _parse_tiff_struct(cls, path: str, max_pixels: int = 4_000_000):
        """
        解析单波段、无压缩（或简单 Strip TIFF）的 GeoTIFF。
        仅用于 rasterio 和 tifffile 均不可用时的最后兜底。
        """
        with open(path, "rb") as f:
            raw = f.read()

        # 字节序
        bo = raw[:2]
        if bo == b"II":
            endian = "<"
        elif bo == b"MM":
            endian = ">"
        else:
            raise ValueError(f"Not a valid TIFF file: {path}")

        bigtiff = struct.unpack_from(endian + "H", raw, 2)[0] == 43
        if bigtiff:
            raise ValueError("BigTIFF not supported by minimal parser")

        ifd_offset = struct.unpack_from(endian + "I", raw, 4)[0]

        def read_ifd(offset):
            n_entries = struct.unpack_from(endian + "H", raw, offset)[0]
            tags = {}
            for i in range(n_entries):
                base = offset + 2 + i * 12
                tag, typ, count = struct.unpack_from(endian + "HHI", raw, base)
                val_offset = base + 8
                # 类型映射
                type_sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8}
                sz = type_sizes.get(typ, 4)
                if count * sz <= 4:
                    raw_val = raw[val_offset: val_offset + count * sz]
                else:
                    ptr = struct.unpack_from(endian + "I", raw, val_offset)[0]
                    raw_val = raw[ptr: ptr + count * sz]
                if typ == 3:
                    vals = [struct.unpack_from(endian + "H", raw_val, j * 2)[0] for j in range(count)]
                elif typ == 4:
                    vals = [struct.unpack_from(endian + "I", raw_val, j * 4)[0] for j in range(count)]
                elif typ == 2:
                    vals = [raw_val.decode("ascii", errors="ignore").rstrip("\x00")]
                else:
                    vals = list(raw_val)
                tags[tag] = vals[0] if count == 1 else vals
            return tags

        tags = read_ifd(ifd_offset)
        width = tags.get(256, 0)
        height = tags.get(257, 0)
        bits = tags.get(258, 8)
        compression = tags.get(259, 1)

        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise ValueError(f'Invalid TIFF dimensions: {width}x{height}')
        samples_per_pixel = tags.get(277, 1)
        if samples_per_pixel != 1:
            raise ValueError(f'Minimal TIFF parser supports one band, got {samples_per_pixel}')
        if isinstance(bits, list):
            if len(bits) != 1:
                raise ValueError(f'Unsupported BitsPerSample: {bits}')
            bits = bits[0]

        if compression not in (1, 32773):  # 仅支持无压缩和 PackBits
            raise ValueError(f"Unsupported TIFF compression: {compression}")

        strip_offsets = tags.get(273, [])
        strip_byte_counts = tags.get(279, [])
        if isinstance(strip_offsets, int):
            strip_offsets = [strip_offsets]
        if isinstance(strip_byte_counts, int):
            strip_byte_counts = [strip_byte_counts]

        rows_per_strip = tags.get(278, height)
        if not isinstance(rows_per_strip, int) or rows_per_strip <= 0:
            raise ValueError(f'Invalid RowsPerStrip: {rows_per_strip}')
        expected_strips = math.ceil(height / rows_per_strip)
        if len(strip_offsets) != expected_strips or len(strip_byte_counts) != expected_strips:
            raise ValueError(
                f'TIFF strip table mismatch: expected {expected_strips}, '
                f'got {len(strip_offsets)} offsets and {len(strip_byte_counts)} byte counts'
            )

        sample_format = tags.get(339, 1)
        dtype_codes = {
            (1, 8): 'u1', (1, 16): 'u2', (1, 32): 'u4',
            (2, 8): 'i1', (2, 16): 'i2', (2, 32): 'i4',
            (3, 32): 'f4',
        }
        dtype_code = dtype_codes.get((sample_format, bits))
        if dtype_code is None:
            raise ValueError(
                f'Unsupported TIFF sample format/bit depth: {sample_format}/{bits}'
            )
        dtype = np.dtype(dtype_code if bits == 8 else endian + dtype_code)
        row_bytes = width * dtype.itemsize
        step = max(1, math.ceil(math.sqrt(width * height / max(1, max_pixels))))
        sampled_strips = []

        for strip_index, (off, cnt) in enumerate(zip(strip_offsets, strip_byte_counts)):
            encoded = raw[off: off + cnt]
            if len(encoded) != cnt:
                raise ValueError(f'TIFF strip {strip_index} extends beyond the file')
            decoded = cls._decode_packbits(encoded) if compression == 32773 else encoded
            row_start = strip_index * rows_per_strip
            strip_rows = min(rows_per_strip, height - row_start)
            expected_bytes = strip_rows * row_bytes
            if len(decoded) != expected_bytes:
                raise ValueError(
                    f'TIFF strip {strip_index} decoded to {len(decoded)} bytes, '
                    f'expected {expected_bytes}'
                )
            strip_data = np.frombuffer(decoded, dtype=dtype).reshape(strip_rows, width)
            first_sampled_row = (-row_start) % step
            if first_sampled_row < strip_rows:
                sampled_strips.append(
                    strip_data[first_sampled_row::step, ::step].copy()
                )

        if not sampled_strips:
            raise ValueError('TIFF contains no sampleable rows')
        arr = np.concatenate(sampled_strips, axis=0)
        meta = {"width": width, "height": height, "dtype": str(dtype), "source": "struct_parser"}
        meta.update({
            'compression': 'packbits' if compression == 32773 else 'none',
            'sample_step': step,
        })
        return arr, meta


class RemoteAssetReader:
    """远程资产读取协议。

    GEE 的服务端 ``sample`` 不是文件窗口读取，因此只复用其采样预算和
    统计语义；STAC/HTTP 资产由本适配器负责 URL、窗口和有限暂存。
    """

    def read_window(
        self,
        asset: Any,
        *,
        band: int = 1,
        max_pixels: int = 250_000,
        window: Optional[Any] = None,
    ) -> Tuple[np.ndarray, Optional[float], Dict[str, Any]]:
        raise NotImplementedError

    def stage(self, asset: Any) -> str:
        raise NotImplementedError


class NASADaacS3CredentialsProvider:
    """Fetch and cache temporary AWS credentials for protected NASA DAAC assets."""

    def __init__(
        self,
        earthdata_token: str,
        endpoints: Optional[Dict[str, str]] = None,
        refresh_margin_seconds: int = 300,
    ):
        self.earthdata_token = str(earthdata_token or "").strip()
        self.endpoints = {
            **NASA_DAAC_S3_CREDENTIAL_ENDPOINTS,
            **(endpoints or {}),
        }
        self.refresh_margin_seconds = max(0, int(refresh_margin_seconds))
        self._cache: Dict[str, Tuple[Dict[str, str], float]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _bucket_from_asset(asset: Any) -> str:
        href = STACAssetReader.asset_href(asset)
        parsed = urlparse(href)
        if parsed.scheme.lower() == "s3":
            return parsed.netloc.lower()
        host = (parsed.hostname or "").lower()
        marker = ".s3"
        return host.split(marker, 1)[0] if marker in host else ""

    @staticmethod
    def _expiration_timestamp(value: Any) -> float:
        if isinstance(value, (int, float)):
            timestamp = float(value)
            return timestamp / 1000 if timestamp > 10_000_000_000 else timestamp
        text = str(value or "").strip()
        if not text:
            return time.time() + 3600
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        except ValueError as exc:
            raise ValueError("NASA DAAC credentials returned an invalid expiration") from exc

    @staticmethod
    def _normalize_credentials(payload: Any) -> Tuple[Dict[str, str], float]:
        if not isinstance(payload, dict):
            raise ValueError("NASA DAAC credentials response must be a JSON object")
        access_key = payload.get("accessKeyId") or payload.get("AccessKeyId")
        secret_key = payload.get("secretAccessKey") or payload.get("SecretAccessKey")
        session_token = payload.get("sessionToken") or payload.get("SessionToken")
        if not access_key or not secret_key or not session_token:
            raise ValueError("NASA DAAC credentials response is missing required fields")
        expiration = NASADaacS3CredentialsProvider._expiration_timestamp(
            payload.get("expiration") or payload.get("Expiration")
        )
        return {
            "AccessKeyId": str(access_key),
            "SecretAccessKey": str(secret_key),
            "SessionToken": str(session_token),
        }, expiration

    def _endpoint_for_bucket(self, bucket: str) -> Optional[str]:
        for prefix, endpoint in self.endpoints.items():
            if bucket == prefix or bucket.startswith(f"{prefix}-"):
                return endpoint
        return None

    def __call__(self, asset: Any) -> Optional[Dict[str, str]]:
        bucket = self._bucket_from_asset(asset)
        endpoint = self._endpoint_for_bucket(bucket)
        if endpoint is None:
            return None
        if not self.earthdata_token:
            raise RuntimeError(
                f"NASA Earthdata token is required for protected bucket {bucket}"
            )

        now = time.time()
        with self._lock:
            cached = self._cache.get(endpoint)
            if cached and now < cached[1] - self.refresh_margin_seconds:
                return dict(cached[0])

            request = Request(
                endpoint,
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self.earthdata_token}",
                    "User-Agent": "DataGenHyp-dataset-profiler/1.0",
                },
            )
            with urlopen(request, timeout=30) as response:  # noqa: S310
                credentials, expiration = self._normalize_credentials(
                    json.loads(response.read().decode("utf-8"))
                )
            self._cache[endpoint] = (credentials, expiration)
            return dict(credentials)


class STACAssetReader(RemoteAssetReader):
    """读取 STAC item asset 的有限窗口，必要时暂存小文件。

    COG/GeoTIFF 优先走 rasterio 的 HTTP range 读取；Zarr/NetCDF 只打开
    第一个数据变量并压缩到有界样本。任何超过 ``max_stage_bytes`` 的
    资产都不会被整文件下载。
    """

    RASTER_TYPES = frozenset({
        "image/tiff", "image/tiff; application=geotiff",
        "image/tiff; application=geotiff; profile=cloud-optimized",
        "image/vnd.stac.geotiff",
    })

    def __init__(
        self,
        staging_dir: Optional[str] = None,
        max_stage_bytes: int = 64 * 1024 * 1024,
        headers: Optional[Dict[str, str]] = None,
        s3_access_mode: str = "auto",
        s3_credentials_provider: Optional[Any] = None,
        asset_signer: Optional[Any] = None,
        nasa_edl_token: str = "",
        nasa_s3_credentials_endpoints: Optional[Dict[str, str]] = None,
        nasa_https_distribution_endpoints: Optional[Dict[str, str]] = None,
    ):
        configured = staging_dir if staging_dir is not None else self._configured_staging_dir()
        self.staging_dir = Path(configured)
        self.max_stage_bytes = max(1, int(max_stage_bytes))
        self.headers = {
            "User-Agent": "DataGenHyp-dataset-profiler/1.0",
            **{key: value for key, value in (headers or {}).items() if value},
        }
        normalized_mode = str(s3_access_mode).strip().lower()
        if normalized_mode not in {"auto", "public", "private"}:
            raise ValueError("s3_access_mode must be one of: auto, public, private")
        self.s3_access_mode = normalized_mode
        self.s3_credentials_provider = s3_credentials_provider
        self.asset_signer = asset_signer
        self.nasa_edl_token = str(nasa_edl_token or "").strip()
        if self.nasa_edl_token and "Authorization" not in self.headers:
            self.headers["Authorization"] = f"Bearer {self.nasa_edl_token}"
        self.nasa_https_distribution_endpoints = {
            **NASA_DAAC_HTTPS_DISTRIBUTION_ENDPOINTS,
            **(nasa_https_distribution_endpoints or {}),
        }
        self.nasa_s3_credentials_provider = NASADaacS3CredentialsProvider(
            self.nasa_edl_token,
            endpoints=nasa_s3_credentials_endpoints,
        )

    @staticmethod
    def _configured_staging_dir() -> str:
        return os.getenv(
            "DATASET_PROFILE_STAGING_DIR",
            r"F:\codex\tmp\dataset_profiler_assets",
        )

    @staticmethod
    def asset_href(asset: Any) -> str:
        if isinstance(asset, str):
            return asset.strip()
        if isinstance(asset, dict):
            return str(asset.get("href") or asset.get("url") or "").strip()
        return ""

    @staticmethod
    def asset_type(asset: Any) -> str:
        return str(asset.get("type") or "").lower().strip() if isinstance(asset, dict) else ""

    @staticmethod
    def is_s3_href(href: str) -> bool:
        return urlparse(str(href)).scheme.lower() == "s3"

    @staticmethod
    def is_presigned_https(href: str) -> bool:
        parsed = urlparse(str(href))
        query = parsed.query.lower()
        return parsed.scheme.lower() == "https" and (
            "x-amz-signature=" in query or "x-amz-credential=" in query
        )

    @classmethod
    def is_s3_https_href(cls, href: str) -> bool:
        parsed = urlparse(str(href))
        host = (parsed.hostname or "").lower()
        return (
            parsed.scheme.lower() == "https"
            and not cls.is_presigned_https(href)
            and (
                ".s3.amazonaws.com" in host
                or ".s3-" in host and host.endswith(".amazonaws.com")
                or ".s3." in host and host.endswith(".amazonaws.com")
            )
        )

    @classmethod
    def is_s3_asset_href(cls, href: str) -> bool:
        return cls.is_s3_href(href) or cls.is_s3_https_href(href)

    @staticmethod
    def is_protected_s3_asset(asset: Any) -> bool:
        href = STACAssetReader.asset_href(asset)
        parsed = urlparse(href)
        bucket = parsed.netloc if parsed.scheme.lower() == "s3" else (parsed.hostname or "").split(".s3", 1)[0]
        return "protected" in bucket.lower()

    @classmethod
    def normalize_s3_href(cls, href: str) -> str:
        if cls.is_s3_href(href):
            return href
        if not cls.is_s3_https_href(href):
            return href
        parsed = urlparse(href)
        host = (parsed.hostname or "").lower()
        bucket = host.split(".s3", 1)[0]
        key = unquote(parsed.path.lstrip("/"))
        if not bucket or not key:
            raise ValueError("S3 HTTPS asset URL must include a bucket and object key")
        return f"s3://{bucket}/{key}"

    @staticmethod
    def s3_region_from_href(href: str) -> Optional[str]:
        host = (urlparse(str(href)).hostname or "").lower()
        match = re.search(r"\.s3(?:-|\.)([a-z0-9-]+)\.amazonaws\.com$", host)
        return match.group(1) if match else None

    @staticmethod
    def _s3_bucket_and_key(href: str) -> Tuple[str, str]:
        normalized = STACAssetReader.normalize_s3_href(href)
        parsed = urlparse(normalized)
        if parsed.scheme.lower() != "s3":
            return "", ""
        return parsed.netloc.lower(), unquote(parsed.path.lstrip("/"))

    def _nasa_https_distribution_href(self, asset: Any) -> Optional[str]:
        bucket, key = self._s3_bucket_and_key(self.asset_href(asset))
        if not bucket or not key:
            return None
        for prefix, endpoint in self.nasa_https_distribution_endpoints.items():
            if bucket == prefix or bucket.startswith(f"{prefix}-"):
                return f"{endpoint.rstrip('/')}/{quote(key, safe='/')}"
        return None

    def _prepare_asset(self, asset: Any) -> Any:
        if self.asset_signer is None:
            return asset
        signed = self.asset_signer(asset)
        if not signed:
            raise ValueError("asset signer returned an empty asset")
        if isinstance(signed, str):
            if isinstance(asset, dict):
                return {**asset, "href": signed}
            return signed
        return signed

    def _s3_mode_for_asset(self, asset: Any) -> str:
        if self.is_protected_s3_asset(asset):
            return "private"
        if isinstance(asset, dict):
            declared = asset.get("s3_access_mode")
            if isinstance(declared, str) and declared.strip():
                mode = declared.strip().lower()
                if mode not in {"auto", "public", "private"}:
                    raise ValueError("asset s3_access_mode must be one of: auto, public, private")
                return mode
            if asset.get("public") is True:
                return "public"
        return self.s3_access_mode

    def _aws_session_for_asset(self, asset: Any) -> Any:
        if self.s3_credentials_provider is not None:
            provided = self.s3_credentials_provider(asset)
            if provided is not None:
                return self._normalize_aws_credentials(provided)
        nasa_credentials = self.nasa_s3_credentials_provider(asset)
        if nasa_credentials is not None:
            return nasa_credentials
        if not _HAS_BOTO3:
            return None
        session = boto3.Session()
        return session if session.get_credentials() is not None else None

    @staticmethod
    def _normalize_aws_credentials(provided: Any) -> Any:
        if hasattr(provided, "get_credentials"):
            return provided
        if not isinstance(provided, dict):
            raise TypeError("s3_credentials_provider must return a boto3 Session or credential mapping")
        access_key = (
            provided.get("AccessKeyId")
            or provided.get("accessKeyId")
            or provided.get("aws_access_key_id")
        )
        secret_key = (
            provided.get("SecretAccessKey")
            or provided.get("secretAccessKey")
            or provided.get("aws_secret_access_key")
        )
        session_token = (
            provided.get("SessionToken")
            or provided.get("sessionToken")
            or provided.get("aws_session_token")
        )
        if not access_key or not secret_key:
            raise ValueError("dynamic AWS credentials must include access key and secret key")
        return {
            "AccessKeyId": str(access_key),
            "SecretAccessKey": str(secret_key),
            "SessionToken": str(session_token) if session_token else "",
        }

    @classmethod
    def choose_asset(cls, item: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        assets = item.get("assets") or {}
        if not isinstance(assets, dict):
            return None, None
        ranked: List[Tuple[int, str, Dict[str, Any]]] = []
        for key, asset in assets.items():
            if not isinstance(asset, dict) or not cls.asset_href(asset):
                continue
            media = cls.asset_type(asset)
            href = cls.asset_href(asset).lower().split("?", 1)[0]
            safe_product = ("slc" in href and (href.endswith(".zip") or ".safe" in href))
            # Catalog links and SAFE/ZIP packages are not directly readable windows.
            if not ("tiff" in media or "geotiff" in media or "netcdf" in media
                    or "zarr" in media or safe_product or href.endswith((".tif", ".tiff", ".nc", ".nc4", ".zarr"))):
                continue
            roles = asset.get("roles") or []
            score = 0
            if safe_product:
                score += 15
            if media in cls.RASTER_TYPES or "tiff" in media or "geotiff" in media:
                score += 30
            if "data" in roles:
                score += 10
            if str(asset.get("href", "")).lower().split("?", 1)[0].endswith((".tif", ".tiff", ".nc", ".zarr")):
                score += 5
            ranked.append((score, str(key), asset))
        if not ranked:
            return None, None
        _, key, asset = max(ranked, key=lambda entry: (entry[0], entry[1]))
        return key, asset

    def stage(self, asset: Any) -> str:
        href = self.asset_href(asset)
        if not href.startswith(("http://", "https://")):
            raise ValueError("remote asset href must be an HTTP(S) URL")
        digest = hashlib.sha256(href.encode("utf-8")).hexdigest()[:24]
        suffix = Path(href.split("?", 1)[0]).suffix or ".bin"
        target = self.staging_dir / f"{digest}{suffix}"
        if target.exists():
            return str(target)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        request = Request(href, headers=self.headers)
        try:
            with urlopen(request, timeout=30) as response:  # noqa: S310
                length = response.headers.get("Content-Length")
                if length and int(length) > self.max_stage_bytes:
                    raise ValueError(f"asset exceeds staging limit ({length} > {self.max_stage_bytes} bytes)")
                written = 0
                with open(target, "wb") as output:
                    while True:
                        chunk = response.read(min(1024 * 1024, self.max_stage_bytes - written + 1))
                        if not chunk:
                            break
                        written += len(chunk)
                        if written > self.max_stage_bytes:
                            raise ValueError("asset exceeds staging limit")
                        output.write(chunk)
        except Exception:
            target.unlink(missing_ok=True)
            raise
        return str(target)

    @staticmethod
    def _bounded_window(width: int, height: int, max_pixels: int) -> Optional[Any]:
        if width <= 0 or height <= 0 or width * height <= max_pixels or not _HAS_RASTERIO:
            return None
        scale = math.sqrt(max_pixels / float(width * height))
        win_w = max(1, int(width * scale))
        win_h = max(1, int(height * scale))
        return Window(max(0, (width - win_w) // 2), max(0, (height - win_h) // 2), win_w, win_h)

    def _read_raster_with_env(
        self,
        href: str,
        band: int,
        max_pixels: int,
        window: Optional[Any] = None,
        *,
        aws_session: Optional[Any] = None,
        unsigned: bool = False,
        aws_region: Optional[str] = None,
    ) -> Tuple[np.ndarray, Optional[float], Dict[str, Any]]:
        if not _HAS_RASTERIO:
            raise ImportError("rasterio is required for remote GeoTIFF window reads")
        if isinstance(aws_session, dict):
            if not _HAS_BOTO3 or AWSSession is None:
                raise ImportError(
                    "boto3 is required for authenticated S3 raster reads; "
                    "install the runtime requirements"
                )
            environment = rasterio.Env(AWSSession(
                aws_access_key_id=aws_session["AccessKeyId"],
                aws_secret_access_key=aws_session["SecretAccessKey"],
                aws_session_token=aws_session.get("SessionToken") or None,
                region_name=aws_region,
            ))
            applied_s3_mode = "private"
        elif aws_session is not None:
            if AWSSession is None:
                raise ImportError("rasterio AWS session support requires boto3")
            environment = rasterio.Env(AWSSession(aws_session))
            applied_s3_mode = "private"
        elif unsigned:
            environment_options = {"AWS_NO_SIGN_REQUEST": "YES"}
            if aws_region:
                environment_options["AWS_REGION"] = aws_region
            environment = rasterio.Env(**environment_options)
            applied_s3_mode = "public"
        else:
            environment = nullcontext()
            applied_s3_mode = None
        with environment:
            with rasterio.open(href) as src:
                selected_window = window or self._bounded_window(src.width, src.height, max_pixels)
                data = src.read(band, window=selected_window, masked=False) if selected_window else src.read(band)
                meta = {
                    "width": src.width, "height": src.height, "count": src.count,
                    "dtype": str(src.dtypes[band - 1]), "crs": str(src.crs),
                    "transform": list(src.window_transform(selected_window)) if selected_window else list(src.transform),
                    "window": [selected_window.col_off, selected_window.row_off, selected_window.width, selected_window.height]
                    if selected_window else None,
                    "nominal_scale_m": float((abs(src.res[0]) + abs(src.res[1])) / 2)
                    if src.res and all(math.isfinite(float(value)) for value in src.res) else None,
                    "reader": "rasterio_remote_window",
                }
                if applied_s3_mode is not None:
                    meta["s3_access_mode"] = applied_s3_mode
                return data, src.nodata, meta

    def _read_nasa_authenticated_https(
        self,
        asset: Any,
        band: int,
        max_pixels: int,
        window: Optional[Any],
    ) -> Tuple[np.ndarray, Optional[float], Dict[str, Any]]:
        href = self._nasa_https_distribution_href(asset)
        if href is None:
            raise RuntimeError("NASA HTTPS distribution endpoint is not configured for this bucket")
        if not self.nasa_edl_token:
            raise RuntimeError("NASA_EDL_TOKEN is required for protected NASA HTTPS assets")
        staged = self.stage({"href": href, "type": "image/tiff"})
        data, nodata, meta = RasterReader.read(
            staged,
            band=band,
            max_pixels=max_pixels,
        )
        return data, nodata, {
            **meta,
            "reader": "staged_nasa_authenticated_https",
            "staged_path": staged,
        }

    def _read_raster(
        self,
        asset: Any,
        href: str,
        band: int,
        max_pixels: int,
        window: Optional[Any] = None,
    ) -> Tuple[np.ndarray, Optional[float], Dict[str, Any]]:
        if not self.is_s3_asset_href(href):
            return self._read_raster_with_env(href, band, max_pixels, window)

        normalized_href = self.normalize_s3_href(href)
        aws_region = self.s3_region_from_href(href)
        mode = self._s3_mode_for_asset(asset)
        if self.is_protected_s3_asset(asset):
            return self._read_nasa_authenticated_https(
                asset, band, max_pixels, window
            )
        if mode == "public":
            return self._read_raster_with_env(
                normalized_href, band, max_pixels, window,
                unsigned=True, aws_region=aws_region,
            )

        aws_session = self._aws_session_for_asset(asset)
        if mode == "private":
            if aws_session is None:
                raise RuntimeError(
                    "private S3 asset requires dynamic AWS credentials; "
                    "configure s3_credentials_provider or the boto3 credential chain"
                )
            return self._read_raster_with_env(
                normalized_href, band, max_pixels, window,
                aws_session=aws_session, aws_region=aws_region,
            )

        if aws_session is not None:
            try:
                return self._read_raster_with_env(
                    normalized_href, band, max_pixels, window,
                    aws_session=aws_session, aws_region=aws_region,
                )
            except Exception as credential_error:
                try:
                    return self._read_raster_with_env(
                        normalized_href, band, max_pixels, window,
                        unsigned=True, aws_region=aws_region,
                    )
                except Exception as unsigned_error:
                    raise RuntimeError(
                        "automatic S3 access failed with dynamic credentials "
                        f"({type(credential_error).__name__}: {credential_error}) and anonymously "
                        f"({type(unsigned_error).__name__}: {unsigned_error})"
                    ) from unsigned_error
        return self._read_raster_with_env(
            normalized_href, band, max_pixels, window,
            unsigned=True, aws_region=aws_region,
        )

    def _read_xarray(self, href: str, max_pixels: int) -> Tuple[np.ndarray, Optional[float], Dict[str, Any]]:
        if not _HAS_XARRAY:
            raise ImportError("xarray is required for remote Zarr/NetCDF reads")
        if ".zarr" in href.lower():
            dataset_context = xr.open_zarr(href, consolidated=False)
        else:
            dataset_context = xr.open_dataset(href)
        with dataset_context as dataset:
            variables = list(dataset.data_vars)
            if not variables:
                raise ValueError("remote dataset contains no data variables")
            variable = dataset[variables[0]]
            while variable.ndim > 2:
                variable = variable.isel({variable.dims[0]: 0})
            data = np.asarray(variable.values)
        if data.ndim != 2:
            data = np.asarray(data).reshape(1, -1)
        if data.size > max_pixels:
            step = max(1, math.ceil(math.sqrt(data.size / max_pixels)))
            data = data[::step, ::step]
        return data, None, {"variable": variables[0], "reader": "xarray_bounded_sample"}

    def read_window(
        self,
        asset: Any,
        *,
        band: int = 1,
        max_pixels: int = 250_000,
        window: Optional[Any] = None,
    ):
        prepared_asset = self._prepare_asset(asset)
        href = self.asset_href(prepared_asset)
        if not href:
            raise ValueError("STAC asset has no href")
        media = self.asset_type(prepared_asset)
        product_path = href.lower().split("?", 1)[0]
        if "slc" in product_path and (product_path.endswith(".zip") or ".safe" in product_path):
            return SentinelSAFEReader(self).read(prepared_asset, band, max_pixels, window)
        if not ("tiff" in media or "geotiff" in media or "zarr" in media
                or "netcdf" in media or href.lower().split("?", 1)[0].endswith(
                    (".tif", ".tiff", ".zarr", ".nc", ".nc4"))):
            raise ValueError(f"unsupported remote window format: {media or 'unspecified'}")
        try:
            if "tiff" in media or "geotiff" in media or href.lower().split("?", 1)[0].endswith((".tif", ".tiff")):
                return self._read_raster(prepared_asset, href, band, max_pixels, window=window)
            if "zarr" in media or "netcdf" in media or href.lower().split("?", 1)[0].endswith((".zarr", ".nc", ".nc4")):
                return self._read_xarray(href, max_pixels)
            raise ValueError(f"unsupported remote window format: {media or 'unspecified'}")
        except Exception as direct_error:
            if self.is_s3_asset_href(href):
                raise RuntimeError(
                    f"S3 asset read failed ({type(direct_error).__name__}: {direct_error})"
                ) from direct_error
            try:
                staged = self.stage(prepared_asset)
                data, nodata, meta = RasterReader.read(staged, band=band, max_pixels=max_pixels)
                meta = {**meta, "reader": "staged_local", "staged_path": staged,
                        "direct_error": f"{type(direct_error).__name__}: {direct_error}"}
                return data, nodata, meta
            except Exception as staged_error:
                raise RuntimeError(
                    f"remote asset read failed ({type(direct_error).__name__}: {direct_error}); "
                    f"staging failed ({type(staged_error).__name__}: {staged_error})"
                ) from staged_error


class VectorReader:
    """向量 / CSV 时序数据读取器"""

    @classmethod
    def read_csv(cls, path: str) -> "pd.DataFrame":
        if not _HAS_PANDAS:
            raise ImportError("pandas required for CSV reading")
        return pd.read_csv(path)

    @classmethod
    def list_vector_files(cls, directory: str, max_files: int = 20) -> List[str]:
        vector_exts = {".geojson", ".json", ".shp"}
        found: List[str] = []
        try:
            for root, _, files in os.walk(directory):
                for fname in sorted(files):
                    if Path(fname).suffix.lower() in vector_exts:
                        found.append(os.path.join(root, fname))
                        if len(found) >= max_files:
                            return found
        except (OSError, PermissionError) as e:
            logger.warning(f"Cannot list vector directory {directory}: {e}")
        return found

    @classmethod
    def list_tabular_files(cls, directory: str, max_files: int = 20) -> List[str]:
        table_exts = {".csv", ".tsv"}
        found: List[str] = []
        try:
            for root, _, files in os.walk(directory):
                for fname in sorted(files):
                    if Path(fname).suffix.lower() in table_exts:
                        found.append(os.path.join(root, fname))
                        if len(found) >= max_files:
                            return found
        except (OSError, PermissionError) as e:
            logger.warning(f"Cannot list tabular directory {directory}: {e}")
        return found

    @classmethod
    def read_geojson(cls, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if not isinstance(payload, dict) or payload.get("type") != "FeatureCollection":
            raise ValueError(f"Unsupported GeoJSON structure: {path}")

        normalized_features = []
        bbox = cls._normalize_bbox(payload.get("bbox"))
        for feature in payload.get("features", []) or []:
            if not isinstance(feature, dict):
                continue
            geometry = feature.get("geometry") or {}
            properties = feature.get("properties") or {}
            geometry_type = str((geometry or {}).get("type") or "Unknown")
            feature_bbox = cls._bbox_from_coordinates((geometry or {}).get("coordinates"))
            bbox = cls._merge_bbox(bbox, feature_bbox)
            normalized_features.append({
                "geometry_type": geometry_type,
                "properties": properties if isinstance(properties, dict) else {},
            })

        return {
            "features": normalized_features,
            "bbox": bbox,
        }

    @classmethod
    def read_shapefile(cls, path: str) -> Dict[str, Any]:
        if not _HAS_PYSHP:
            raise ImportError("pyshp required for shapefile reading")

        reader = pyshp.Reader(path)
        field_names = [field[0] for field in reader.fields[1:]]
        features = []
        for shape_record in reader.iterShapeRecords():
            features.append({
                "geometry_type": getattr(shape_record.shape, "shapeTypeName", "Unknown"),
                "properties": dict(zip(field_names, shape_record.record)),
            })

        return {
            "features": features,
            "bbox": cls._normalize_bbox(getattr(reader, "bbox", None)),
        }

    @classmethod
    def _bbox_from_coordinates(cls, coords: Any) -> Optional[Dict[str, float]]:
        if not isinstance(coords, (list, tuple)) or not coords:
            return None

        points: List[Tuple[float, float]] = []

        def _walk(node: Any) -> None:
            if isinstance(node, (list, tuple)):
                if len(node) >= 2 and all(isinstance(item, (int, float)) for item in node[:2]):
                    points.append((float(node[0]), float(node[1])))
                    return
                for child in node:
                    _walk(child)

        _walk(coords)
        if not points:
            return None

        xs = [item[0] for item in points]
        ys = [item[1] for item in points]
        return {
            "min_x": round(min(xs), 6),
            "min_y": round(min(ys), 6),
            "max_x": round(max(xs), 6),
            "max_y": round(max(ys), 6),
        }

    @classmethod
    def _normalize_bbox(cls, bbox: Any) -> Optional[Dict[str, float]]:
        if isinstance(bbox, dict):
            required = ("min_x", "min_y", "max_x", "max_y")
            if all(key in bbox for key in required):
                return {key: round(float(bbox[key]), 6) for key in required}
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            try:
                return {
                    "min_x": round(float(bbox[0]), 6),
                    "min_y": round(float(bbox[1]), 6),
                    "max_x": round(float(bbox[2]), 6),
                    "max_y": round(float(bbox[3]), 6),
                }
            except (TypeError, ValueError):
                return None
        return None

    @classmethod
    def _merge_bbox(
        cls,
        bbox_a: Optional[Dict[str, float]],
        bbox_b: Optional[Dict[str, float]],
    ) -> Optional[Dict[str, float]]:
        if not bbox_a:
            return bbox_b
        if not bbox_b:
            return bbox_a
        return {
            "min_x": round(min(bbox_a["min_x"], bbox_b["min_x"]), 6),
            "min_y": round(min(bbox_a["min_y"], bbox_b["min_y"]), 6),
            "max_x": round(max(bbox_a["max_x"], bbox_b["max_x"]), 6),
            "max_y": round(max(bbox_a["max_y"], bbox_b["max_y"]), 6),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 通用分析器（AnalysisEngine）
# ═══════════════════════════════════════════════════════════════════════════════

class AnalysisEngine:
    """
    与数据类型无关的统计分析方法集。
    输入均为 numpy array 或 dict，不依赖任何地理库。
    """

    # ── 2.1 分类栅格：逐年像素分布 ──────────────────────────────────────────

    @staticmethod
    def class_distribution(
        data: np.ndarray,
        nodata: Optional[float],
        class_labels: Dict[int, str],
        pixel_area_m2: float = 900.0,
    ) -> Dict[str, Any]:
        """
        计算各类别的像素数量、面积占比。
        返回：{class_id: {label, pixel_count, area_ha, pct}}
        """
        flat = data.flatten()
        if nodata is not None:
            flat = flat[flat != nodata]
        total_valid = len(flat)
        if total_valid == 0:
            return {}

        result = {}
        for raw_val, count in sorted(Counter(flat.tolist()).items()):
            val = int(raw_val)
            label = class_labels.get(val, f"class_{val}")
            area_ha = count * pixel_area_m2 / 10000
            pct = round(count / total_valid * 100, 2)
            result[str(val)] = {
                "label": label,
                "pixel_count": int(count),
                "area_ha": round(area_ha, 1),
                "pct": pct,
            }
        return result

    @staticmethod
    def timeseries_class_area(
        yearly_data: Dict[int, np.ndarray],  # year → 2D array
        nodata: Optional[float],
        target_classes: List[int],
        pixel_area_m2: float = 900.0,
    ) -> Dict[int, Dict[int, float]]:
        """
        对每年计算指定类别的面积（公顷）。
        返回：{year: {class_id: area_ha}}
        """
        result = {}
        for year, arr in sorted(yearly_data.items()):
            flat = arr.flatten()
            if nodata is not None:
                flat = flat[flat != nodata]
            year_result = {}
            for cls in target_classes:
                count = int(np.sum(flat == cls))
                year_result[cls] = round(count * pixel_area_m2 / 10000, 1)
            result[year] = year_result
        return result

    # ── 2.2 异常年份检测 ────────────────────────────────────────────────────

    @staticmethod
    def detect_anomaly_years(
        yearly_values: Dict[int, float],
        z_threshold: float = 1.5,
    ) -> List[Dict[str, Any]]:
        """
        基于 Z-score 检测显著偏高年份。
        返回：[{year, value, z_score, direction}]
        """
        if len(yearly_values) < 4:
            return []
        years = sorted(yearly_values)
        vals = np.array([yearly_values[y] for y in years], dtype=float)
        mean, std = float(np.mean(vals)), float(np.std(vals))
        if std < 1e-9:
            return []
        anomalies = []
        for y, v in zip(years, vals):
            z = (v - mean) / std
            if abs(z) >= z_threshold:
                anomalies.append({
                    "year": y,
                    "value": round(float(v), 1),
                    "z_score": round(float(z), 2),
                    "direction": "above_mean" if z > 0 else "below_mean",
                })
        anomalies.sort(key=lambda x: abs(x["z_score"]), reverse=True)
        return anomalies

    # ── 2.3 Mann-Kendall 趋势检验 ───────────────────────────────────────────

    @staticmethod
    def mann_kendall_trend(
        yearly_values: Dict[int, float],
    ) -> Dict[str, Any]:
        """
        无参数时序趋势检验。
        返回：{trend, p_value, tau, slope_per_year, significant}
        """
        years = sorted(yearly_values)
        n = len(years)
        if n < 4:
            return {"trend": "insufficient_data", "p_value": None,
                    "significant": False, "n": n}
        vals = [yearly_values[y] for y in years]
        # Mann-Kendall S 统计量
        s = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                diff = vals[j] - vals[i]
                s += (1 if diff > 0 else (-1 if diff < 0 else 0))
        # 方差（无连续值调整版）
        var_s = n * (n - 1) * (2 * n + 5) / 18.0
        if var_s <= 0:
            return {"trend": "flat", "p_value": 1.0, "significant": False, "n": n}
        z = (s - 1) / math.sqrt(var_s) if s > 0 else (
            (s + 1) / math.sqrt(var_s) if s < 0 else 0)
        p = _normal_two_sided_pvalue(z)
        tau = s / (n * (n - 1) / 2.0)
        # Sen's slope
        slopes = []
        for i in range(n - 1):
            for j in range(i + 1, n):
                if years[j] != years[i]:
                    slopes.append((vals[j] - vals[i]) / (years[j] - years[i]))
        slope = float(np.median(slopes)) if slopes else 0.0

        trend_str = ("increasing" if slope > 0 else "decreasing") if abs(z) > 1.96 else "no_significant_trend"
        return {
            "trend": trend_str,
            "p_value": round(float(p), 4),
            "tau": round(float(tau), 3),
            "slope_per_year_ha": round(slope, 2),
            "significant": bool(p < 0.05),
            "n_years": n,
        }

    # ── 2.4 空间热点（基于重心和密度方块） ──────────────────────────────────

    @staticmethod
    def spatial_hotspot_summary(
        data: np.ndarray,
        nodata: Optional[float],
        hotspot_classes: List[int],
        n_quadrants: int = 4,
    ) -> Dict[str, Any]:
        """
        将图像分为 n_quadrants×n_quadrants 象限，计算热点类别在各象限的密度。
        返回：{hotspot_centroid_yx, quadrant_density, dominant_quadrant}
        不依赖地理坐标，只输出相对位置。
        """
        h, w = data.shape
        mask = np.zeros_like(data, dtype=bool)
        for cls in hotspot_classes:
            mask |= (data == cls)
        if nodata is not None:
            mask &= (data != nodata)

        hotspot_pixels = np.argwhere(mask)
        if len(hotspot_pixels) == 0:
            return {"hotspot_pixels": 0, "centroid_relative": None,
                    "quadrant_density": {}, "dominant_quadrant": None}

        centroid_y = float(np.mean(hotspot_pixels[:, 0])) / h
        centroid_x = float(np.mean(hotspot_pixels[:, 1])) / w

        # 象限密度
        qh, qw = h // n_quadrants, w // n_quadrants
        quad_density = {}
        for qi in range(n_quadrants):
            for qj in range(n_quadrants):
                y0, y1 = qi * qh, min((qi + 1) * qh, h)
                x0, x1 = qj * qw, min((qj + 1) * qw, w)
                region = mask[y0:y1, x0:x1]
                total = (y1 - y0) * (x1 - x0)
                density = float(np.sum(region)) / total if total > 0 else 0.0
                quad_density[f"Q{qi}_{qj}"] = round(density * 100, 2)  # %

        dominant = max(quad_density, key=quad_density.get)
        return {
            "hotspot_pixel_count": int(np.sum(mask)),
            "centroid_relative": {
                "y_pct": round(centroid_y * 100, 1),
                "x_pct": round(centroid_x * 100, 1),
                "note": "relative position within raster (0=top/left, 100=bottom/right)",
            },
            "quadrant_density_pct": quad_density,
            "dominant_quadrant": dominant,
        }

    # ── 2.5 连续值栅格：基本统计 ───────────────────────────────────────────

    @staticmethod
    def continuous_stats(
        data: np.ndarray,
        nodata: Optional[float],
        percentiles: List[float] = [5, 25, 50, 75, 95],
    ) -> Dict[str, Any]:
        flat = data.flatten().astype(float)
        if nodata is not None:
            flat = flat[flat != nodata]
        if len(flat) == 0:
            return {}
        pct_vals = {f"p{int(p)}": round(float(np.percentile(flat, p)), 4)
                    for p in percentiles}
        return {
            "count": int(len(flat)),
            "mean": round(float(np.mean(flat)), 4),
            "std": round(float(np.std(flat)), 4),
            "min": round(float(np.min(flat)), 4),
            "max": round(float(np.max(flat)), 4),
            **pct_vals,
        }

    @staticmethod
    def local_stddev(data: np.ndarray, radius: int = 1) -> np.ndarray:
        """计算有界窗口的局部标准差，供 GEE 与 STAC 样本共用。"""
        arr = np.asarray(data, dtype=float)
        if arr.ndim != 2 or arr.size == 0:
            return np.asarray([], dtype=float)
        radius = max(1, int(radius))
        padded = np.pad(arr, radius, mode="edge")
        size = radius * 2 + 1
        try:
            windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size))
            finite = np.isfinite(windows)
            counts = np.sum(finite, axis=(-2, -1))
            values = np.where(finite, windows, 0.0)
            sums = np.sum(values, axis=(-2, -1))
            squared_sums = np.sum(values * values, axis=(-2, -1))
            variance = np.full(counts.shape, np.nan, dtype=float)
            valid = counts > 0
            means = np.zeros(counts.shape, dtype=float)
            means[valid] = sums[valid] / counts[valid]
            variance[valid] = (
                squared_sums[valid] / counts[valid] - means[valid] * means[valid]
            )
            variance[valid] = np.maximum(variance[valid], 0.0)
            return np.sqrt(variance).reshape(-1)
        except AttributeError:
            windows = []
            for row in range(arr.shape[0]):
                for col in range(arr.shape[1]):
                    windows.append(padded[row:row + size, col:col + size])
            deviations = []
            for window in windows:
                values = window[np.isfinite(window)]
                deviations.append(float(np.std(values)) if values.size else np.nan)
            return np.asarray(deviations, dtype=float)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 数据类型检测器（TypeDetector）
# ═══════════════════════════════════════════════════════════════════════════════

class DatasetTypeDetector:
    """
    根据 data_description 的字段自动推断数据集类型，
    决定采用哪条分析路径。

    返回的 dataset_type 枚举：
      - "classified_raster"   : 整数分类栅格（如 MTBS 烧伤严重度）
      - "continuous_raster"   : 连续值栅格（如 NDVI、DEM）
      - "multitemporal_raster": 多时相分类/连续栅格（annual mosaics）
      - "vector_polygon"      : 面向量
      - "tabular_timeseries"  : CSV 时序
      - "unknown"             : 无法识别，跳过数值分析
    """

    @classmethod
    def detect(cls, data_description: Dict[str, Any]) -> str:
        # 优先判断栅格
        data_type_str = str(data_description.get("data_type", "")).lower()
        data_format_str = str(data_description.get("data_format", "")).lower()
        raster_content = data_description.get("raster_content", {})
        pixel_values = raster_content.get("pixel_values", {}) if isinstance(raster_content, dict) else {}
        temporal_unit = str(data_description.get("temporal_unit", "")).lower()
        records = str(data_description.get("records", "")).lower()

        # data_type 由上游自由文本描述（如 "Phase-preserving SAR image collection"），
        # 不是规范枚举，只能按去空格后的子串判定，否则长句形态一律落到 unknown。
        data_type_compact = data_type_str.replace("_", "").replace(" ", "")
        data_format_compact = data_format_str.replace("_", "").replace(" ", "")
        is_gee_image = (
            data_type_compact == "image"
            or "imagecollection" in data_type_compact
            or "imagecollection" in data_format_compact
        )
        has_raster_bands = bool(
            isinstance(raster_content, dict)
            and (
                raster_content.get("first_band_names")
                or raster_content.get("first_band_data_types")
                or raster_content.get("band_semantics")
                or raster_content.get("gee_asset_id")
            )
        )
        is_raster = (
            any(kw in data_type_str for kw in ("raster", "geotiff", "tiff", "tif", "mosaic"))
            or is_gee_image
            or has_raster_bands
        )
        is_classified = (
            bool(pixel_values)
            or "classif" in str(raster_content.get("classification_type", "")).lower()
            or "classif" in data_type_str
        )
        is_temporal = (
            "annual" in temporal_unit
            or "monthly" in temporal_unit
            or "per year" in records
            or "time series" in data_type_str.lower()
        )

        if is_raster and is_classified and is_temporal:
            return "multitemporal_classified_raster"
        if is_raster and is_classified:
            return "classified_raster"
        if is_raster:
            return "continuous_raster"

        # 向量
        has_vector_content = "vector_content" in data_description
        is_feature_collection = "featurecollection" in data_type_str.replace(" ", "")
        if (
            any(kw in data_type_str for kw in ("vector", "shapefile", "geojson", "polygon", "point", "line"))
            or is_feature_collection
            or has_vector_content
        ):
            return "vector_polygon"

        # CSV/表格
        if any(kw in data_type_str for kw in ("csv", "tabular", "table", "spreadsheet")):
            return "tabular_timeseries"

        return "unknown"

    @classmethod
    def extract_class_labels(cls, data_description: Dict[str, Any]) -> Dict[int, str]:
        raster_content = data_description.get("raster_content", {})
        if not isinstance(raster_content, dict):
            return {}
        pixel_values = raster_content.get("pixel_values", {})
        if not isinstance(pixel_values, dict):
            return {}
        labels: Dict[int, str] = {}
        for raw_key, raw_value in pixel_values.items():
            try:
                labels[int(raw_key)] = str(raw_value)
            except (TypeError, ValueError):
                # Ignore range-like keys such as "0-100" for continuous/thematic cover rasters.
                continue
        return labels

    @classmethod
    def extract_pixel_area(cls, data_description: Dict[str, Any]) -> float:
        unit = data_description.get("unit_of_measurement", {})
        if isinstance(unit, dict):
            pa = unit.get("pixel_area")
            if pa is not None:
                # 解析 "900 m²" 或数值
                try:
                    return float(str(pa).split()[0])
                except (ValueError, IndexError):
                    pass
        # 从 spatial_resolution 推算
        res_str = str(data_description.get("spatial_resolution", "30"))
        try:
            res_m = float("".join(c for c in res_str if c.isdigit() or c == "."))
            return res_m ** 2
        except ValueError:
            return 900.0  # 默认 30m


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 上下文解析器（ContextResolver）
# ═══════════════════════════════════════════════════════════════════════════════

class ContextResolver:
    """将 context / params 解析为确定性的 profiler 配置。"""

    @classmethod
    def resolve(
        cls,
        context: Dict[str, Any],
        params: Dict[str, Any],
        data_description: Dict[str, Any],
        agent_config: Dict[str, Any],
    ) -> ResolvedProfilerConfig:
        dataset_info = context.get("dataset_info", {}) or {}
        dataset_type = DatasetTypeDetector.detect(data_description)
        file_paths = DataPathResolver.resolve(dataset_info)
        if not file_paths:
            file_paths = DataPathResolver.resolve(context)
        remote_source = RemoteDatasetResolver.resolve(dataset_info)
        if not remote_source:
            remote_source = RemoteDatasetResolver.resolve(context)

        class_labels = DatasetTypeDetector.extract_class_labels(data_description)
        pixel_area_m2 = DatasetTypeDetector.extract_pixel_area(data_description)
        hotspot_target_classes = cls._resolve_hotspot_classes(params, agent_config, class_labels)
        nodata_classes = cls._resolve_nodata_classes(data_description)

        spatial_coverage = data_description.get("spatial_coverage") or {}
        dataset_name = (
            data_description.get("dataset_name")
            or context.get("dataset_name")
            or "Unknown Dataset"
        )
        topic = cls._resolve_topic(context, data_description, dataset_name)
        region = context.get("region") or spatial_coverage.get("region_name") or "Unknown region"
        temporal_scope = data_description.get("temporal_coverage") or "Unknown period"
        spatial_resolution = data_description.get("spatial_resolution") or "Unknown resolution"

        return ResolvedProfilerConfig(
            dataset_type=dataset_type,
            dataset_name=str(dataset_name),
            topic=str(topic),
            region=str(region),
            temporal_scope=str(temporal_scope),
            spatial_resolution=str(spatial_resolution),
            pixel_area_m2=float(pixel_area_m2),
            file_paths=file_paths,
            class_labels=class_labels,
            hotspot_target_classes=hotspot_target_classes,
            nodata_classes=nodata_classes,
            remote_source=remote_source,
        )

    @classmethod
    def _resolve_topic(
        cls,
        context: Dict[str, Any],
        data_description: Dict[str, Any],
        dataset_name: str,
    ) -> str:
        topic_candidates = [
            context.get("topic"),
            data_description.get("thematic_focus"),
            data_description.get("dataset_name"),
            data_description.get("description"),
            dataset_name,
        ]
        for candidate in topic_candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return "Unknown topic"

    @classmethod
    def _resolve_hotspot_classes(
        cls,
        params: Dict[str, Any],
        agent_config: Dict[str, Any],
        class_labels: Dict[int, str],
    ) -> List[int]:
        requested = params.get("hotspot_target_classes")
        if requested is None:
            requested = agent_config.get("hotspot_target_classes")

        normalized = cls._normalize_int_list(requested)
        if normalized:
            return normalized

        candidate_classes = sorted(int(cid) for cid in class_labels.keys())
        candidate_classes = [cid for cid in candidate_classes if cid > 0]
        if candidate_classes:
            return [candidate_classes[-1]]
        return []

    @classmethod
    def _resolve_nodata_classes(cls, data_description: Dict[str, Any]) -> List[int]:
        raster_content = data_description.get("raster_content", {}) or {}
        explicit_values = []
        for key in ("nodata_values", "nodata_classes"):
            values = raster_content.get(key)
            explicit_values.extend(cls._normalize_int_list(values))

        nodata_value = raster_content.get("nodata")
        explicit_values.extend(cls._normalize_int_list([nodata_value]))

        inferred_values = []
        pixel_values = raster_content.get("pixel_values", {}) or {}
        for raw_key, raw_label in pixel_values.items():
            label = str(raw_label).strip().lower()
            if label in {"nodata", "no data", "fill", "mask"}:
                try:
                    inferred_values.append(int(raw_key))
                except (TypeError, ValueError):
                    continue

        ordered = []
        seen = set()
        for value in explicit_values + inferred_values:
            if value not in seen:
                seen.add(value)
                ordered.append(value)
        return ordered

    @staticmethod
    def _normalize_int_list(value: Any) -> List[int]:
        if value is None:
            return []
        if not isinstance(value, (list, tuple, set)):
            value = [value]

        normalized = []
        for item in value:
            try:
                normalized.append(int(item))
            except (TypeError, ValueError):
                continue
        return normalized


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 文件路径解析器（PathResolver）
# ═══════════════════════════════════════════════════════════════════════════════

class DataPathResolver:
    """
    从 spatial_extent / dataset_info 中解析文件路径列表。
    支持：
      - local_stage_path（单个目录）
      - source 字段的 LOCAL: 前缀
      - data_access_manifest 列表
    """

    @classmethod
    def resolve(cls, dataset_info: Dict[str, Any]) -> List[str]:
        """
        返回所有可用数据文件路径（不保证存在）。
        """
        paths = []

        # 从 base_dataset 直接路径
        base = dataset_info.get("base_dataset", {}) or {}
        if isinstance(base, dict):
            local_path = base.get("data_path")
            if local_path and not cls._is_remote_reference(local_path):
                paths.append(str(local_path))

        # 从 spatial_extent
        se = dataset_info.get("spatial_extent", {}) or {}
        if isinstance(se, dict):
            bd = se.get("base_dataset", {}) or {}
            if isinstance(bd, dict):
                lp = bd.get("data_path")
                if lp and not cls._is_remote_reference(lp):
                    paths.append(str(lp))

        # 从 data_access_manifest
        manifest = dataset_info.get("data_access_manifest", []) or []
        for entry in manifest:
            if isinstance(entry, dict):
                src = entry.get("source", "") or ""
                if src.startswith("LOCAL:"):
                    local_source = src[6:].strip()
                    if local_source and not cls._is_remote_reference(local_source):
                        paths.append(local_source)
                lsp = entry.get("local_stage_path")
                if lsp and not cls._is_remote_reference(lsp):
                    paths.append(str(lsp))

        # 去重，保留存在的
        seen = set()
        result = []
        for p in paths:
            if p and p not in seen:
                seen.add(p)
                result.append(p)
        return result

    @staticmethod
    def _is_remote_reference(value: Any) -> bool:
        normalized = str(value or "").strip().upper()
        if normalized.startswith("LOCAL:"):
            normalized = normalized[6:].strip()
        return normalized.startswith((
            "PANGAEA:",
            "GEE:",
            "NASA_CMR:",
            "PLANETARY_COMPUTER:",
            "EARTH_SEARCH:",
            "OSM_TAGINFO:",
            "OSM_OVERPASS:",
            "HTTP://",
            "HTTPS://",
        ))

    @classmethod
    def list_raster_files(cls, directory: str, max_files: int = 50) -> List[str]:
        """列出目录下的栅格文件。"""
        raster_exts = {".tif", ".tiff", ".geotiff", ".img", ".nc"}
        found = []
        try:
            for root, _, files in os.walk(directory):
                for fname in sorted(files):
                    if Path(fname).suffix.lower() in raster_exts:
                        found.append(os.path.join(root, fname))
                        if len(found) >= max_files:
                            return found
        except (OSError, PermissionError) as e:
            logger.warning(f"Cannot list directory {directory}: {e}")
        return found

    @classmethod
    def list_vector_files(cls, directory: str, max_files: int = 20) -> List[str]:
        return VectorReader.list_vector_files(directory, max_files=max_files)

    @classmethod
    def list_tabular_files(cls, directory: str, max_files: int = 20) -> List[str]:
        return VectorReader.list_tabular_files(directory, max_files=max_files)

    @classmethod
    def parse_year_from_filename(cls, path: str) -> Optional[int]:
        """从文件名提取年份，如 mtbs_TX_1997.tif -> 1997。"""
        import re

        name = Path(path).stem
        m = re.search(r"(19|20)\d{2}", name)
        if m:
            return int(m.group(0))
        return None


class RemoteDatasetResolver:
    """Resolve GEE and external dataset references for profiling."""

    @classmethod
    def resolve(cls, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """按 dataset_source/source/data_path 统一解析远程数据源。"""
        base = dataset_info.get("base_dataset", {}) or {}
        if not isinstance(base, dict):
            return {}
        description = base.get("data_description", {}) or {}
        if not isinstance(description, dict):
            description = {}
        metadata = base.get("source_metadata") or description.get("source_metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {}

        source = str(base.get("dataset_source") or metadata.get("source") or "").strip().upper()
        source_ref = str(base.get("source") or base.get("data_path") or "").strip()
        if source_ref.upper().startswith("LOCAL:"):
            source_ref = source_ref[6:].strip()
        if not source and ":" in source_ref:
            source = source_ref.split(":", 1)[0].strip().upper()
        if source in {"GEE", "GEE_OFFICIAL", "GEE_COMMUNITY"} or source == "":
            return cls.resolve_gee_asset(dataset_info)
        if source not in EXTERNAL_PROFILE_SOURCES:
            return {}

        identifier = str(base.get("id") or metadata.get("id") or "").strip()
        if not identifier and ":" in source_ref:
            identifier = source_ref.split(":", 1)[1].strip()
        if not identifier:
            return {}
        doc_url = str(base.get("doc_url") or metadata.get("doc_url") or "").strip()
        stac_url = str(base.get("stac_url") or metadata.get("stac_url") or doc_url).strip()
        return {
            "source_type": source.lower(),
            "dataset_source": source,
            "dataset_id": identifier,
            "source": f"{source}:{identifier}",
            "doc_url": doc_url,
            "stac_url": stac_url,
            "provider": base.get("provider") or metadata.get("provider"),
            "overpass_ql": base.get("overpass_ql") or metadata.get("overpass_ql"),
            "endpoint": base.get("endpoint") or metadata.get("endpoint"),
        }
    """解析 base_dataset 中的远程 GEE 资产引用。"""

    @classmethod
    def resolve_gee_asset(cls, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        base = dataset_info.get("base_dataset", {}) or {}
        if not isinstance(base, dict):
            return {}

        data_description = base.get("data_description", {}) or {}
        if not isinstance(data_description, dict):
            data_description = {}
        raster_content = data_description.get("raster_content", {}) or {}
        if not isinstance(raster_content, dict):
            raster_content = {}
        gee_asset = data_description.get("gee_asset", {}) or {}
        if not isinstance(gee_asset, dict):
            gee_asset = {}

        ee_id = str(
            base.get("ee_id")
            or base.get("gee_id")
            or data_description.get("ee_id")
            or data_description.get("gee_id")
            or raster_content.get("gee_asset_id")
            or gee_asset.get("id")
            or ""
        ).strip()
        source = str(base.get("source") or "").strip()
        data_path = str(base.get("data_path") or "").strip()
        if not ee_id:
            for reference in (source, data_path):
                normalized = reference
                if normalized.startswith("LOCAL:"):
                    normalized = normalized[6:].strip()
                if normalized.startswith("GEE:"):
                    ee_id = normalized[4:].strip()
                    break

        ee_ctor = str(
            base.get("ee_ctor")
            or data_description.get("ee_ctor")
            or ""
        ).strip()
        if not ee_ctor and ee_id:
            declared_type = " ".join(
                str(value or "")
                for value in (
                    gee_asset.get("type"),
                    data_description.get("data_type"),
                    data_description.get("data_format"),
                )
            ).lower().replace("_", "")
            if "featurecollection" in declared_type or any(
                token in declared_type for token in ("vector", "table")
            ):
                ee_ctor = "ee.FeatureCollection"
            elif "imagecollection" in declared_type:
                ee_ctor = "ee.ImageCollection"
            elif re.search(r"(^|\s)image($|\s)", declared_type):
                ee_ctor = "ee.Image"
            elif ee_id.startswith("projects/"):
                ee_ctor = "ee.FeatureCollection"
            else:
                ee_ctor = "ee.ImageCollection"

        if not ee_id:
            return {}

        return {
            "source_type": "gee",
            "ee_id": ee_id,
            "ee_ctor": ee_ctor,
            "source": f"GEE:{ee_id}",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. 约束构建器 + Portrait 格式化器
# ═══════════════════════════════════════════════════════════════════════════════

class ConstraintBuilder:
    """基于确定性统计结果构造问题轴与研究钩子。"""

    @classmethod
    def build(
        cls,
        config: ResolvedProfilerConfig,
        analysis_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        candidate_axes = cls._build_candidate_question_axes(config, analysis_results)
        return {
            "candidate_question_axes": candidate_axes,
            "suggested_research_hooks": cls._build_hooks(config, analysis_results, candidate_axes),
            "profile_quality": cls._build_profile_quality(config, analysis_results, candidate_axes),
        }

    @classmethod
    def _build_candidate_question_axes(
        cls,
        config: ResolvedProfilerConfig,
        analysis_results: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        axes = []

        global_dist = analysis_results.get("global_class_distribution", {})
        if global_dist:
            dominant_class, dominant_info = max(
                global_dist.items(),
                key=lambda item: item[1].get("pct", 0),
            )
            axes.append({
                "axis": "composition_baseline",
                "dataset_only": True,
                "summary": (
                    f"The mapped composition is dominated by {dominant_info.get('label', dominant_class)} "
                    f"({dominant_info.get('pct', '?')}% of valid observations)."
                ),
                "evidence": {
                    "dominant_class": dominant_class,
                    "dominant_label": dominant_info.get("label", dominant_class),
                    "dominant_pct": dominant_info.get("pct"),
                },
            })

        value_distribution = analysis_results.get("value_distribution", {}) or {}
        if value_distribution:
            axes.append({
                "axis": "continuous_value_distribution",
                "dataset_only": True,
                "summary": (
                    f"Representative values span {value_distribution.get('min')} to "
                    f"{value_distribution.get('max')}, with median {value_distribution.get('p50')} "
                    f"and interquartile range {value_distribution.get('p25')} to "
                    f"{value_distribution.get('p75')}."
                ),
                "evidence": value_distribution,
            })

        texture_summary = analysis_results.get("spatial_texture_summary", {}) or {}
        texture_distribution = texture_summary.get("distribution", {}) or {}
        if texture_distribution:
            axes.append({
                "axis": "local_spatial_texture",
                "dataset_only": True,
                "summary": (
                    "Local 3x3-neighborhood variability has median standard deviation "
                    f"{texture_distribution.get('p50')} and 95th percentile "
                    f"{texture_distribution.get('p95')}."
                ),
                "evidence": {
                    "metric": texture_summary.get("metric"),
                    **texture_distribution,
                },
            })

        scale_summary = analysis_results.get("scale_summary", {}) or {}
        if scale_summary.get("nominal_scale_m_median") is not None:
            axes.append({
                "axis": "spatial_scale",
                "dataset_only": True,
                "summary": (
                    f"The sampled native scale is {scale_summary.get('nominal_scale_m_median')} m "
                    f"and the texture statistic uses an approximately "
                    f"{scale_summary.get('texture_window_m_median')} m window."
                ),
                "evidence": scale_summary,
            })

        significant_trends = []
        for cls_id, trend in (analysis_results.get("trend_by_class", {}) or {}).items():
            if trend.get("significant"):
                significant_trends.append((cls_id, trend))
        for cls_id, trend in significant_trends[:3]:
            label = cls._class_label(config, cls_id)
            axes.append({
                "axis": "long_term_trend",
                "dataset_only": True,
                "summary": (
                    f"{label} shows a {trend.get('trend', 'detected')} pattern "
                    f"with slope {trend.get('slope_per_year_ha', '?')} ha/yr "
                    f"(p={trend.get('p_value', '?')})."
                ),
                "evidence": {
                    "class_id": cls_id,
                    "class_label": label,
                    "trend": trend.get("trend"),
                    "p_value": trend.get("p_value"),
                    "slope_per_year_ha": trend.get("slope_per_year_ha"),
                },
            })

        for cls_id, anomalies in (analysis_results.get("anomaly_years", {}) or {}).items():
            if not anomalies:
                continue
            top = anomalies[0]
            label = cls._class_label(config, cls_id)
            axes.append({
                "axis": "temporal_anomaly",
                "dataset_only": True,
                "summary": (
                    f"{label} has an anomalous time step in {top.get('year')} "
                    f"(z={top.get('z_score')}, value={top.get('value')})."
                ),
                "evidence": {
                    "class_id": cls_id,
                    "class_label": label,
                    "year": top.get("year"),
                    "z_score": top.get("z_score"),
                    "value": top.get("value"),
                },
            })

        hotspot = analysis_results.get("spatial_hotspot", {}) or {}
        if hotspot.get("hotspot_pixel_count", 0) > 0:
            centroid = hotspot.get("centroid_relative", {}) or {}
            axes.append({
                "axis": "spatial_concentration",
                "dataset_only": True,
                "summary": (
                    f"Target mapped class is spatially concentrated in {hotspot.get('dominant_quadrant')} "
                    f"with centroid near ({centroid.get('x_pct', '?')}%, {centroid.get('y_pct', '?')}%)."
                ),
                "evidence": {
                    "dominant_quadrant": hotspot.get("dominant_quadrant"),
                    "representative_year": hotspot.get("representative_year"),
                    "hotspot_pixel_count": hotspot.get("hotspot_pixel_count"),
                },
            })

        trend_of_mean = analysis_results.get("trend_of_mean", {}) or {}
        if trend_of_mean.get("significant"):
            axes.append({
                "axis": "aggregate_value_trend",
                "dataset_only": True,
                "summary": (
                    f"Aggregate values show a {trend_of_mean.get('trend')} pattern "
                    f"(p={trend_of_mean.get('p_value')})."
                ),
                "evidence": {
                    "trend": trend_of_mean.get("trend"),
                    "p_value": trend_of_mean.get("p_value"),
                    "slope_per_year_ha": trend_of_mean.get("slope_per_year_ha"),
                },
            })

        anomaly_years_by_mean = analysis_results.get("anomaly_years_by_mean") or []
        if anomaly_years_by_mean:
            top = anomaly_years_by_mean[0]
            axes.append({
                "axis": "aggregate_value_anomaly",
                "dataset_only": True,
                "summary": (
                    f"Aggregate values are anomalous in {top.get('year')} "
                    f"(z={top.get('z_score')}, value={top.get('value')})."
                ),
                "evidence": {
                    "year": top.get("year"),
                    "z_score": top.get("z_score"),
                    "value": top.get("value"),
                },
            })

        geometry_summary = analysis_results.get("geometry_summary", {}) or {}
        if geometry_summary.get("feature_count", 0) > 0:
            dominant_geometry = geometry_summary.get("dominant_geometry_type") or "mixed geometry"
            axes.append({
                "axis": "feature_inventory",
                "dataset_only": True,
                "summary": (
                    f"The vector layer contains {geometry_summary.get('feature_count')} features, "
                    f"dominated by {dominant_geometry}."
                ),
                "evidence": {
                    "feature_count": geometry_summary.get("feature_count"),
                    "geometry_types": geometry_summary.get("geometry_types", {}),
                    "bbox": geometry_summary.get("bbox", {}),
                },
            })

        attribute_summary = analysis_results.get("attribute_summary", {}) or {}
        for field_name, info in attribute_summary.items():
            if info.get("type") == "numeric":
                axes.append({
                    "axis": "attribute_gradient",
                    "dataset_only": True,
                    "summary": (
                        f"Numeric attribute '{field_name}' spans {info.get('min')} to {info.get('max')} "
                        f"(mean={info.get('mean')})."
                    ),
                    "evidence": {
                        "field": field_name,
                        "min": info.get("min"),
                        "max": info.get("max"),
                        "mean": info.get("mean"),
                    },
                })
                break
        else:
            for field_name, info in attribute_summary.items():
                top_values = info.get("top_values") or []
                if info.get("type") in {"categorical", "string"} and top_values:
                    top_value = top_values[0]
                    axes.append({
                        "axis": "attribute_composition",
                        "dataset_only": True,
                        "summary": (
                            f"Attribute '{field_name}' is dominated by '{top_value.get('value')}' "
                            f"({top_value.get('count')} records)."
                        ),
                        "evidence": {
                            "field": field_name,
                            "top_values": top_values[:3],
                        },
                    })
                    break

        timeseries_summary = analysis_results.get("timeseries_summary", {}) or {}
        if timeseries_summary.get("time_column"):
            axes.append({
                "axis": "temporal_coverage",
                "dataset_only": True,
                "summary": (
                    f"The table exposes a usable time column '{timeseries_summary.get('time_column')}' "
                    f"covering {timeseries_summary.get('n_time_steps', 0)} aggregated time steps."
                ),
                "evidence": {
                    "time_column": timeseries_summary.get("time_column"),
                    "value_column": timeseries_summary.get("value_column"),
                    "year_range": timeseries_summary.get("year_range"),
                    "n_time_steps": timeseries_summary.get("n_time_steps"),
                },
            })

        if not axes and analysis_results.get("note"):
            axes.append({
                "axis": "metadata_scope",
                "dataset_only": False,
                "summary": analysis_results.get("note"),
                "evidence": {},
            })

        return axes

    @classmethod
    def _build_hooks(
        cls,
        config: ResolvedProfilerConfig,
        analysis_results: Dict[str, Any],
        candidate_axes: List[Dict[str, Any]],
    ) -> List[str]:
        hooks = []

        for cls_id, anomalies in (analysis_results.get("anomaly_years", {}) or {}).items():
            if not anomalies:
                continue
            top = anomalies[0]
            label = cls._class_label(config, cls_id)
            hooks.append(
                f"What distinguishes {top.get('year')} from the baseline period for {label}, "
                f"given its anomalous mapped area (z={top.get('z_score')})?"
            )

        for cls_id, trend in (analysis_results.get("trend_by_class", {}) or {}).items():
            if not trend.get("significant"):
                continue
            label = cls._class_label(config, cls_id)
            hooks.append(
                f"How robust is the detected {trend.get('trend')} pattern in {label} "
                f"({trend.get('slope_per_year_ha', '?')} ha/yr) across the observation window?"
            )

        hotspot = analysis_results.get("spatial_hotspot", {}) or {}
        if hotspot.get("hotspot_pixel_count", 0) > 0:
            representative_year = hotspot.get("representative_year", "the representative time step")
            hooks.append(
                f"Is the concentration of target mapped pixels in {hotspot.get('dominant_quadrant')} "
                f"persistent across years, or mainly driven by {representative_year}?"
            )

        trend_of_mean = analysis_results.get("trend_of_mean", {}) or {}
        if trend_of_mean.get("significant"):
            hooks.append(
                f"What does the detected {trend_of_mean.get('trend')} shift in aggregate values imply "
                f"for the monitored variable across {config.temporal_scope}?"
            )

        anomaly_years_by_mean = analysis_results.get("anomaly_years_by_mean") or []
        if anomaly_years_by_mean:
            top = anomaly_years_by_mean[0]
            hooks.append(
                f"What conditions correspond to the anomalous aggregate value observed in {top.get('year')} "
                f"(z={top.get('z_score')}) relative to neighboring time steps?"
            )

        geometry_summary = analysis_results.get("geometry_summary", {}) or {}
        if geometry_summary.get("feature_count", 0) > 0:
            hooks.append(
                f"How does the observed {geometry_summary.get('dominant_geometry_type', 'feature')} inventory "
                f"within the layer bbox relate to the study target in {config.region}?"
            )

        attribute_summary = analysis_results.get("attribute_summary", {}) or {}
        for field_name, info in attribute_summary.items():
            if info.get("type") == "numeric":
                hooks.append(
                    f"What spatial or contextual pattern could explain variation in numeric attribute '{field_name}' "
                    f"across the mapped features?"
                )
                break
            top_values = info.get("top_values") or []
            if info.get("type") in {"categorical", "string"} and top_values:
                hooks.append(
                    f"Why is '{top_values[0].get('value')}' the dominant category in attribute '{field_name}', "
                    f"and where does it cluster within the mapped extent?"
                )
                break

        timeseries_summary = analysis_results.get("timeseries_summary", {}) or {}
        if timeseries_summary.get("time_column"):
            hooks.append(
                f"How stable is the {timeseries_summary.get('aggregation')} series for '{timeseries_summary.get('value_column')}' "
                f"across {timeseries_summary.get('year_range', 'the detected period')}?"
            )

        if not hooks and candidate_axes:
            first_axis = candidate_axes[0]
            hooks.append(
                f"How can the dataset-only axis '{first_axis.get('axis')}' be converted into a concrete, testable question "
                f"within {config.region} during {config.temporal_scope}?"
            )

        if not hooks:
            hooks.append(
                f"Which descriptive patterns are directly observable in {config.topic} within {config.region} "
                f"using this dataset alone?"
            )

        return cls._dedupe(hooks)[:5]

    @classmethod
    def _build_profile_quality(
        cls,
        config: ResolvedProfilerConfig,
        analysis_results: Dict[str, Any],
        candidate_axes: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        coverage = analysis_results.get("coverage_summary", {}) or {}
        loaded_years = coverage.get("loaded_years", []) or []
        missing_years = coverage.get("missing_years", []) or []
        load_errors = coverage.get("load_errors", []) or []
        significant_trend_count = sum(
            1 for trend in (analysis_results.get("trend_by_class", {}) or {}).values()
            if trend.get("significant")
        )
        anomaly_count = sum(
            len(items) for items in (analysis_results.get("anomaly_years", {}) or {}).values()
        )
        hotspot_detected = 1 if (analysis_results.get("spatial_hotspot", {}) or {}).get("hotspot_pixel_count", 0) > 0 else 0
        aggregate_trend_detected = 1 if (analysis_results.get("trend_of_mean", {}) or {}).get("significant") else 0
        aggregate_anomaly_count = len(analysis_results.get("anomaly_years_by_mean") or [])
        vector_signal_count = 1 if (analysis_results.get("geometry_summary", {}) or {}).get("feature_count", 0) > 0 else 0
        signal_count = (
            significant_trend_count
            + anomaly_count
            + hotspot_detected
            + aggregate_trend_detected
            + aggregate_anomaly_count
            + vector_signal_count
        )

        has_dataset_axis = any(axis.get("dataset_only") for axis in candidate_axes)

        if analysis_results.get("error"):
            status = "analysis_failed"
        elif any(
            analysis_results.get(key, {}).get("status") in {"failed", "empty", "unsupported_asset_format"}
            for key in ("remote_sampling_summary", "gee_sampling_summary")
        ):
            status = "metadata_only"
        elif has_dataset_axis:
            status = "analysis_ready"
        else:
            status = "metadata_only"

        if status == "analysis_failed":
            quality_tier = "low"
        elif status == "metadata_only":
            quality_tier = "low"
        elif has_dataset_axis and len(candidate_axes) >= 3 and (len(loaded_years) >= 4 or analysis_results.get("yearly_stats")):
            quality_tier = "high"
        elif has_dataset_axis:
            quality_tier = "medium"
        else:
            quality_tier = "low"

        notes = []
        if not config.file_paths:
            notes.append("No resolved local data path was provided.")
        if missing_years:
            notes.append(f"Missing expected time steps: {missing_years}")
        if load_errors:
            notes.append(f"Read failures encountered for {len(load_errors)} file(s).")
        for summary_key in ("remote_sampling_summary", "gee_sampling_summary"):
            sampling = analysis_results.get(summary_key, {})
            if sampling.get("status") in {"failed", "empty", "unsupported_asset_format"}:
                notes.append(f"{summary_key}: {sampling['status']}; no quantitative profile was established.")
        if not candidate_axes:
            notes.append("No strong dataset-grounded question axis was extracted.")

        return {
            "status": status,
            "quality_tier": quality_tier,
            "candidate_axis_count": len(candidate_axes),
            "detected_signal_count": signal_count,
            "resolved_path_count": len(config.file_paths),
            "loaded_year_count": len(loaded_years),
            "missing_year_count": len(missing_years),
            "load_error_count": len(load_errors),
            "notes": notes,
        }

    @classmethod
    def _class_label(cls, config: ResolvedProfilerConfig, class_id: Any) -> str:
        try:
            normalized = int(class_id)
        except (TypeError, ValueError):
            return str(class_id)
        return config.class_labels.get(normalized, f"class_{normalized}")

    @staticmethod
    def _dedupe(items: List[str]) -> List[str]:
        ordered = []
        seen = set()
        for item in items:
            if item and item not in seen:
                seen.add(item)
                ordered.append(item)
        return ordered


class PortraitFormatter:
    """将确定性统计分析结果格式化为结构化画像与注入文本。"""

    @classmethod
    def format(
        cls,
        config: ResolvedProfilerConfig,
        data_description: Dict[str, Any],
        analysis_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        derived = ConstraintBuilder.build(config, analysis_results)
        dataset_features = cls._build_dataset_features(data_description)
        portrait = {
            "dataset_type": config.dataset_type,
            "resolved_profile_context": {
                "dataset_name": config.dataset_name,
                "topic": config.topic,
                "region": config.region,
                "temporal_scope": config.temporal_scope,
                "spatial_resolution": config.spatial_resolution,
                "hotspot_target_classes": config.hotspot_target_classes,
                "dataset_source": config.remote_source.get("dataset_source")
                or config.remote_source.get("source_type"),
                "dataset_id": config.remote_source.get("dataset_id")
                or config.remote_source.get("ee_id"),
            },
            "source_description": data_description,
            "dataset_features": dataset_features,
            "analysis": analysis_results,
            "portrait_text": cls._build_text(
                config,
                analysis_results,
                derived,
                dataset_features,
            ),
            "candidate_question_axes": derived["candidate_question_axes"],
            "suggested_research_hooks": derived["suggested_research_hooks"],
            "profile_quality": derived["profile_quality"],
        }
        return portrait

    @classmethod
    def _build_dataset_features(
        cls,
        data_description: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """
        Promote dataset-description facts into authoritative dataset features.

        Downstream agents should treat these as traits that exist on the dataset.
        """
        if not isinstance(data_description, dict):
            return []

        features: List[Dict[str, str]] = []

        def add(feature: Any, source_field: str) -> None:
            if feature in (None, "", [], {}):
                return
            if isinstance(feature, (dict, list)):
                text = json.dumps(feature, ensure_ascii=False, sort_keys=True)
            else:
                text = str(feature).strip()
            if not text:
                return
            features.append({
                "feature": text,
                "evidence_field": source_field,
                "availability": "available_in_dataset",
            })

        for item in data_description.get("analytical_capabilities") or []:
            add(item, "analytical_capabilities")

        source_capability = data_description.get("source_dataset_capability")
        if isinstance(source_capability, list):
            for item in source_capability:
                add(item, "source_dataset_capability")
        elif isinstance(source_capability, dict):
            for key, value in source_capability.items():
                add({key: value}, "source_dataset_capability")
        else:
            add(source_capability, "source_dataset_capability")

        for field_name in (
            "temporal_coverage",
            "spatial_coverage",
            "spatial_resolution",
            "coordinate_system",
            "temporal_unit",
            "records",
            "spectral_bands",
            "band_details",
            "ground_truth_source",
            "mask_type",
            "raster_content",
            "data_source",
            "reference",
        ):
            add(data_description.get(field_name), field_name)

        seen = set()
        deduped = []
        for feature in features:
            key = (feature["feature"], feature["evidence_field"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(feature)
        return deduped

    @classmethod
    def _build_text(
        cls,
        config: ResolvedProfilerConfig,
        analysis_results: Dict[str, Any],
        derived: Dict[str, Any],
        dataset_features: List[Dict[str, str]],
    ) -> str:
        """生成注入后续 agent prompt 的自然语言画像摘要。"""
        lines = []
        lines.append(f"=== Dataset Portrait: {config.dataset_name} ===")
        lines.append(
            f"Topic: {config.topic} | Region: {config.region} | "
            f"Period: {config.temporal_scope} | Resolution: {config.spatial_resolution}"
        )
        lines.append(f"Dataset type: {config.dataset_type}")
        lines.append("")

        if dataset_features:
            lines.append("Dataset features available for downstream reasoning:")
            for item in dataset_features[:8]:
                lines.append(
                    f"  - {item.get('feature')} "
                    f"(from {item.get('evidence_field')})"
                )
            lines.append("")

        coverage = analysis_results.get("coverage_summary", {}) or {}
        if coverage:
            if coverage.get("n_years_loaded") is not None:
                lines.append(
                    f"Coverage: {coverage.get('n_years_loaded', '?')} time steps loaded, "
                    f"{coverage.get('n_files_total', '?')} files discovered."
                )
                if coverage.get("missing_years"):
                    lines.append(f"  Missing time steps: {coverage['missing_years']}")
            elif coverage.get("n_rows_loaded") is not None:
                lines.append(
                    f"Coverage: {coverage.get('n_rows_loaded', '?')} rows loaded, "
                    f"{coverage.get('n_files_total', '?')} files discovered."
                )
            else:
                lines.append(
                    f"Coverage: {coverage.get('n_files_analyzed', '?')} files analyzed, "
                    f"{coverage.get('n_files_total', '?')} files discovered."
                )

        profile_status = str(analysis_results.get("profile_status") or "").strip()
        if profile_status:
            lines.append(f"Profile status: {profile_status}.")

        stac_summary = analysis_results.get("stac_metadata_summary", {}) or {}
        if stac_summary:
            title = stac_summary.get("title")
            if title:
                lines.append(f"Remote collection: {title}.")
            if stac_summary.get("sampled_item_count") is not None:
                lines.append(
                    f"Remote item sample: {stac_summary.get('sampled_item_count')} items, "
                    f"{stac_summary.get('asset_count', 0)} assets."
                )
            if stac_summary.get("sampled_datetime_range"):
                lines.append(
                    f"Sampled item time range: {stac_summary['sampled_datetime_range']}."
                )

        osm_summary = analysis_results.get("osm_metadata_summary", {}) or {}
        if osm_summary.get("tag_key"):
            lines.append(
                f"OSM tag: {osm_summary.get('tag_key')} "
                f"(exists={osm_summary.get('exists')})."
            )
        elif osm_summary.get("element_count") is not None:
            lines.append(f"OSM element sample: {osm_summary.get('element_count')} elements.")

        global_dist = analysis_results.get("global_class_distribution", {}) or {}
        if global_dist:
            lines.append("\nOverall mapped class distribution:")
            for cls_id, info in global_dist.items():
                lines.append(
                    f"  [{cls_id}] {info['label']}: {info['pct']}% ({info['area_ha']:.0f} ha)"
                )

        value_distribution = analysis_results.get("value_distribution", {}) or {}
        if value_distribution:
            lines.append(
                "\nRepresentative continuous-value distribution: "
                f"min={value_distribution.get('min')}, p25={value_distribution.get('p25')}, "
                f"median={value_distribution.get('p50')}, p75={value_distribution.get('p75')}, "
                f"max={value_distribution.get('max')}, n={value_distribution.get('count')}."
            )

        texture_summary = analysis_results.get("spatial_texture_summary", {}) or {}
        texture_distribution = texture_summary.get("distribution", {}) or {}
        if texture_distribution:
            lines.append(
                "Local spatial texture (3x3 neighborhood standard deviation): "
                f"median={texture_distribution.get('p50')}, "
                f"p95={texture_distribution.get('p95')}."
            )

        scale_summary = analysis_results.get("scale_summary", {}) or {}
        if scale_summary.get("nominal_scale_m_median") is not None:
            lines.append(
                "Scale summary: "
                f"native median={scale_summary.get('nominal_scale_m_median')} m, "
                f"texture window={scale_summary.get('texture_window_m_median')} m."
            )

        trend_by_class = analysis_results.get("trend_by_class", {}) or {}
        if trend_by_class:
            lines.append("\nDetected temporal trends:")
            for cls_id, trend in trend_by_class.items():
                label = ConstraintBuilder._class_label(config, cls_id)
                sig = "★ significant" if trend.get("significant") else ""
                lines.append(
                    f"  [{cls_id}] {label}: {trend.get('trend', '?')} "
                    f"(p={trend.get('p_value', '?')}, slope={trend.get('slope_per_year_ha', '?')} ha/yr) {sig}"
                )

        anomaly_years = analysis_results.get("anomaly_years", {}) or {}
        if anomaly_years:
            lines.append("\nAnomalous time steps:")
            for cls_id, anomalies in anomaly_years.items():
                if not anomalies:
                    continue
                label = ConstraintBuilder._class_label(config, cls_id)
                parts = [f"{item['year']} (z={item['z_score']})" for item in anomalies[:3]]
                lines.append(f"  [{cls_id}] {label}: {', '.join(parts)}")

        hotspot = analysis_results.get("spatial_hotspot", {}) or {}
        if hotspot.get("hotspot_pixel_count", 0) > 0:
            centroid = hotspot.get("centroid_relative", {}) or {}
            lines.append(
                f"\nSpatial concentration: dominant quadrant = {hotspot.get('dominant_quadrant', '?')}, "
                f"centroid at ({centroid.get('x_pct', '?')}% x, {centroid.get('y_pct', '?')}% y)."
            )

        trend_of_mean = analysis_results.get("trend_of_mean", {}) or {}
        if trend_of_mean:
            lines.append(
                f"\nAggregate trend: {trend_of_mean.get('trend', '?')} "
                f"(p={trend_of_mean.get('p_value', '?')})."
            )

        anomaly_years_by_mean = analysis_results.get("anomaly_years_by_mean") or []
        if anomaly_years_by_mean:
            top = anomaly_years_by_mean[0]
            lines.append(
                f"Anomalous aggregate time step: {top.get('year')} "
                f"(z={top.get('z_score')}, value={top.get('value')})."
            )

        geometry_summary = analysis_results.get("geometry_summary", {}) or {}
        if geometry_summary.get("feature_count", 0) > 0:
            lines.append(
                f"\nVector geometry summary: {geometry_summary.get('feature_count')} features, "
                f"geometry types = {geometry_summary.get('geometry_types', {})}."
            )
            if geometry_summary.get("bbox"):
                lines.append(f"  BBox: {geometry_summary.get('bbox')}")

        attribute_summary = analysis_results.get("attribute_summary", {}) or {}
        if attribute_summary:
            lines.append("\nBasic attribute summary:")
            for field_name, info in list(attribute_summary.items())[:4]:
                if info.get("type") == "numeric":
                    lines.append(
                        f"  - {field_name}: numeric, mean={info.get('mean')}, "
                        f"range=({info.get('min')}, {info.get('max')})"
                    )
                else:
                    lines.append(
                        f"  - {field_name}: {info.get('type')}, top={info.get('top_values', [])[:2]}"
                    )

        timeseries_summary = analysis_results.get("timeseries_summary", {}) or {}
        if timeseries_summary.get("time_column"):
            lines.append(
                f"\nTabular time series: time column = {timeseries_summary.get('time_column')}, "
                f"value column = {timeseries_summary.get('value_column')}, "
                f"aggregation = {timeseries_summary.get('aggregation')}, "
                f"range = {timeseries_summary.get('year_range', 'unknown')}."
            )

        candidate_axes = derived.get("candidate_question_axes", []) or []
        if candidate_axes:
            lines.append("\nCandidate question axes:")
            for axis in candidate_axes[:4]:
                lines.append(f"  - {axis.get('axis')}: {axis.get('summary')}")

        obs = analysis_results.get("observability_note", "")
        if obs:
            lines.append(f"\nObservability note: {obs}")

        quality = derived.get("profile_quality", {}) or {}
        lines.append(
            f"\nProfile quality: {quality.get('status', 'unknown')} / {quality.get('quality_tier', 'unknown')}"
        )
        lines.append("\n[END OF DATASET PORTRAIT]")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. DatasetProfilerAgent 主类
# ═══════════════════════════════════════════════════════════════════════════════

class DatasetProfilerAgent:
    """
    数据集画像智能体。

    接口：
        portrait = await agent.execute(context, params)

    输入 context 字段（与现有 pipeline 兼容）：
        - dataset_info.base_dataset.local_stage_path  : 数据目录
        - dataset_info.base_dataset.data_description  : 现有 JSON 描述
        - spatial_extent                               : 同上（兼容字段）

    输出 portrait dict：
        - dataset_type          : 推断的数据集类型
        - analysis              : 原始统计结果
        - portrait_text         : 自然语言摘要（注入 LLM prompt 用）
        - suggested_research_hooks  : 具体研究钩子
        - enriched_data_description : 原始描述 + 新增字段的合并版本
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        # 最多读取多少年的文件（避免内存超限）
        self.max_years_to_load = self.config.get("max_years_to_load", 24)
        # 热点分析的目标类别（默认：高严重度）
        self.hotspot_target_classes = self.config.get("hotspot_target_classes", [3])
        # 降采样上限
        self.max_pixels = self.config.get("max_pixels_per_file", 4_000_000)
        self.gee_profile_max_images = int(self.config.get("gee_profile_max_images", 6))
        self.gee_profile_pixels_per_image = int(
            self.config.get("gee_profile_pixels_per_image", 512)
        )
        self.remote_profile_max_pixels = max(1, int(
            self.config.get("remote_profile_max_pixels", self.gee_profile_pixels_per_image * 9)
        ))
        configured_reader = self.config.get("remote_asset_reader")
        if configured_reader is None:
            configured_reader = STACAssetReader(
                staging_dir=self.config.get("remote_staging_dir"),
                max_stage_bytes=int(self.config.get("remote_max_stage_bytes", 64 * 1024 * 1024)),
                s3_access_mode=self.config.get("remote_s3_access_mode", "auto"),
                s3_credentials_provider=self.config.get("remote_s3_credentials_provider"),
                asset_signer=self.config.get("remote_asset_signer"),
                nasa_edl_token=os.getenv("NASA_EDL_TOKEN", "").strip(),
                nasa_s3_credentials_endpoints=self.config.get(
                    "nasa_s3_credentials_endpoints"
                ),
                nasa_https_distribution_endpoints=self.config.get(
                    "nasa_https_distribution_endpoints"
                ),
                headers={
                    "Authorization": f"Bearer {os.getenv('NASA_EDL_TOKEN', '').strip()}"
                    if os.getenv("NASA_EDL_TOKEN", "").strip() else "",
                    "Ocp-Apim-Subscription-Key": os.getenv(
                        "PLANETARY_COMPUTER_SUBSCRIPTION_KEY", ""
                    ).strip(),
                },
            )
        self.remote_asset_reader = configured_reader

    async def _execute_composite(
        self,
        context: Dict[str, Any],
        params: Dict[str, Any],
        dataset_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Profile every declared component and return one role-aware composite portrait."""
        base_dataset = dataset_info.get("base_dataset", {}) or {}
        components = [
            component for component in (base_dataset.get("dataset_components") or [])
            if isinstance(component, dict)
        ]
        component_portraits = []
        for component in components:
            component = copy.deepcopy(component)
            parent_description = base_dataset.get("data_description", {})
            if isinstance(parent_description, dict):
                component_description = component.setdefault("data_description", {})
                if isinstance(component_description, dict):
                    for scope_key in ("remote_profile_scope", "gee_profile_scope"):
                        if scope_key not in component_description and scope_key in parent_description:
                            component_description[scope_key] = copy.deepcopy(parent_description[scope_key])
            component_info = {
                "base_dataset": component,
                "retrieved_datasets": [],
                "available_datasets": [component.get("name")],
            }
            component_context = dict(context)
            component_context["dataset_info"] = component_info
            component_context["spatial_extent"] = component_info
            portrait = await self.execute(component_context, params)
            component_portraits.append({
                "role": component.get("role"),
                "dataset_name": component.get("name"),
                "source": component.get("source"),
                "portrait": portrait,
            })

        features = []
        axes = []
        hooks = []
        analyses = {}
        component_identity = []
        for item in component_portraits:
            portrait = item["portrait"] or {}
            role = item.get("role") or item.get("dataset_name") or "component"
            analyses[role] = portrait.get("analysis", {})
            member_context = portrait.get("resolved_profile_context") or {}
            component_identity.append({
                "role": role,
                "dataset_name": item.get("dataset_name"),
                "source": item.get("source"),
                "dataset_type": portrait.get("dataset_type"),
                "dataset_id": member_context.get("dataset_id"),
            })
            features.extend({"role": role, **feature} for feature in (portrait.get("dataset_features") or []) if isinstance(feature, dict))
            axes.extend({"role": role, **axis} for axis in (portrait.get("candidate_question_axes") or []) if isinstance(axis, dict))
            # 单数据集画像里 suggested_research_hooks 是字符串列表（ConstraintBuilder._build_hooks
            # 返回 List[str]），复合层必须维持同一类型，否则下游（survey 的 query 种子、
            # plan_utils、idea 评估的证据上下文）拿到的是无法当文本用的结构。此前用
            # isinstance(hook, dict) 过滤，成员 hooks 是字符串，结果被整批静默丢弃。
            hooks.extend(
                f"[{role}] {str(hook).strip()}"
                for hook in (portrait.get("suggested_research_hooks") or [])
                if str(hook).strip()
            )

        source_description = copy.deepcopy(base_dataset.get("data_description") or {})
        composite_description = copy.deepcopy(source_description)
        if not isinstance(composite_description, dict):
            composite_description = {"description": str(composite_description)}
        composite_description["dataset_name"] = base_dataset.get("name") or "Composite observation"
        composite_description["data_type"] = "Composite multi-dataset observation"
        composite_description["components"] = [
            {
                "role": item.get("role"),
                "dataset_name": item.get("dataset_name"),
                "source": item.get("source"),
                "portrait": item.get("portrait", {}).get("enriched_data_description", {}),
            }
            for item in component_portraits
        ]
        contract = base_dataset.get("combination_contract")
        if isinstance(contract, dict):
            composite_description["combination_contract"] = copy.deepcopy(contract)

        # 复合层补齐单数据集画像本来就有的身份字段。缺 resolved_profile_context 时，
        # 下游按 dataset_source/dataset_id 定位数据集的分支在复合任务上会拿到空值。
        composite_name = base_dataset.get("name") or "Composite observation"
        spatial_coverage = source_description.get("spatial_coverage") if isinstance(source_description, dict) else None
        region = ""
        if isinstance(spatial_coverage, dict):
            region = str(spatial_coverage.get("region_name") or "").strip()
        if not region:
            region = str(base_dataset.get("region_name") or "").strip()

        return {
            "dataset_type": "composite_dataset",
            "resolved_profile_context": {
                "dataset_name": composite_name,
                "topic": composite_name,
                "region": region,
                "dataset_source": base_dataset.get("dataset_source") or base_dataset.get("source"),
                "dataset_id": base_dataset.get("anchor_asset_id") or base_dataset.get("id"),
                "component_count": len(component_portraits),
                "components": component_identity,
            },
            "source_description": source_description,
            "analysis": {"components": analyses, "component_count": len(component_portraits)},
            "dataset_features": features,
            "candidate_question_axes": axes,
            "suggested_research_hooks": hooks,
            "profile_quality": {
                "status": "componentwise",
                "quality_tier": "metadata_or_remote_component_profiles",
                "component_count": len(component_portraits),
            },
            "portrait_text": "\n\n".join(
                f"[{item.get('role') or item.get('dataset_name')}] {item.get('portrait', {}).get('portrait_text', '')}"
                for item in component_portraits
            ),
            "enriched_data_description": composite_description,
            "component_portraits": component_portraits,
        }

    async def execute(
        self,
        context: Dict[str, Any],
        params: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        主入口。若文件不可读则 graceful degrade：
        返回仅含元数据的基础 portrait，不阻断流程。
        """
        params = params or {}
        dataset_info = context.get("dataset_info", {}) or {}
        base_dataset = dataset_info.get("base_dataset", {}) if isinstance(dataset_info, dict) else {}
        if isinstance(base_dataset, dict) and base_dataset.get("type") == "composite_dataset":
            logger.info(
                "DatasetProfilerAgent Profiling composite anchor with %s components",
                len(base_dataset.get("dataset_components") or []),
            )
            return await self._execute_composite(context, params, dataset_info)

        data_description = self._extract_data_description(context)

        logger.info(f"DatasetProfilerAgent Starting profiling for: "
                    f"{data_description.get('dataset_name', 'Unknown')}")

        # 1. 解析确定性 profiler 配置
        resolved_config = ContextResolver.resolve(
            context=context,
            params=params,
            data_description=data_description,
            agent_config=self.config,
        )
        dataset_type = resolved_config.dataset_type
        file_paths = resolved_config.file_paths
        logger.info(f"DatasetProfilerAgent Detected dataset type: {dataset_type}")

        # 3. 执行分析
        analysis_results = {}
        try:
            if not file_paths and resolved_config.remote_source.get("source_type") == "gee":
                analysis_results = self._analyze_gee_metadata(
                    resolved_config.remote_source,
                    data_description,
                )
            elif (
                not file_paths
                and resolved_config.remote_source.get("dataset_source")
                in EXTERNAL_PROFILE_SOURCES
            ):
                analysis_results = await self._analyze_external_remote(
                    resolved_config.remote_source,
                    data_description,
                )
            elif dataset_type in ("multitemporal_classified_raster", "classified_raster"):
                analysis_results = await self._analyze_classified_raster(
                    file_paths,
                    data_description,
                    dataset_type,
                    resolved_config.hotspot_target_classes,
                )
            elif dataset_type == "continuous_raster":
                analysis_results = await self._analyze_continuous_raster(
                    file_paths, data_description
                )
            elif dataset_type == "vector_polygon":
                analysis_results = await self._analyze_vector_dataset(
                    file_paths, data_description
                )
            elif dataset_type == "tabular_timeseries":
                analysis_results = await self._analyze_tabular_timeseries(
                    file_paths, data_description
                )
            else:
                logger.info(
                    f"DatasetProfilerAgent dataset_type={dataset_type}: "
                    "numerical analysis skipped, metadata-only portrait."
                )
                analysis_results = {"note": f"Numerical analysis not implemented for type: {dataset_type}"}
        except Exception as e:
            logger.warning(f"DatasetProfilerAgent Analysis failed: {e}. "
                           "Returning metadata-only portrait.")
            analysis_results = {"error": str(e), "note": "Analysis failed; metadata portrait only."}

        # 4. 格式化输出
        portrait = PortraitFormatter.format(resolved_config, data_description, analysis_results)

        # 5. 生成 enriched_data_description（原始描述 + 画像增量字段）
        portrait["enriched_data_description"] = self._enrich_description(
            data_description, portrait
        )

        logger.info(
            f"DatasetProfilerAgent Portrait generated. "
            f"Hooks: {len(portrait.get('suggested_research_hooks', []))}, "
            f"Question axes: {len(portrait.get('candidate_question_axes', []))}, "
            f"Anomaly years found: "
            f"{sum(len(v) for v in analysis_results.get('anomaly_years', {}).values())}"
        )
        return portrait

    # ── 内部分析方法 ────────────────────────────────────────────────────────

    @staticmethod
    def _stac_bbox_from_item(item: Dict[str, Any]) -> Optional[List[float]]:
        bbox = item.get("bbox")
        if isinstance(bbox, list) and len(bbox) >= 4:
            try:
                return [float(value) for value in bbox[:4]]
            except (TypeError, ValueError):
                return None
        geometry = item.get("geometry") or {}
        coordinates = geometry.get("coordinates") if isinstance(geometry, dict) else None
        points: List[Tuple[float, float]] = []

        def collect(value: Any) -> None:
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                if all(isinstance(part, (int, float)) for part in value[:2]):
                    points.append((float(value[0]), float(value[1])))
                else:
                    for nested in value:
                        collect(nested)

        collect(coordinates)
        if not points:
            return None
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return [min(xs), min(ys), max(xs), max(ys)]

    @classmethod
    def _summarize_stac_items(cls, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        item_ids: List[str] = []
        datetimes: List[str] = []
        asset_count = 0
        media_types: Counter[str] = Counter()
        bbox_values: List[List[float]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id") or "").strip()
            if item_id:
                item_ids.append(item_id)
            properties = item.get("properties") or {}
            if isinstance(properties, dict):
                value = properties.get("datetime") or properties.get("start_datetime")
                if value:
                    datetimes.append(str(value))
            assets = item.get("assets") or {}
            if isinstance(assets, dict):
                asset_count += len(assets)
                for asset in assets.values():
                    if isinstance(asset, dict):
                        media_type = str(asset.get("type") or "").strip()
                        if media_type:
                            media_types[media_type] += 1
            bbox = cls._stac_bbox_from_item(item)
            if bbox:
                bbox_values.append(bbox)
        summary: Dict[str, Any] = {
            "sampled_item_count": len(item_ids),
            "sampled_item_ids": item_ids[:12],
            "asset_count": asset_count,
            "media_types": dict(media_types),
        }
        if datetimes:
            summary["sampled_datetime_range"] = [min(datetimes), max(datetimes)]
        if bbox_values:
            summary["sampled_bbox"] = [
                min(value[0] for value in bbox_values),
                min(value[1] for value in bbox_values),
                max(value[2] for value in bbox_values),
                max(value[3] for value in bbox_values),
            ]
        return summary

    def _build_remote_sample_analysis(
        self,
        *,
        value_samples: List[float],
        texture_samples: List[float],
        sampled_asset_count: int,
        sampled_asset_ids: List[str],
        sample_errors: List[str],
        scales_m: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """把任意远程读取器输出的 NumPy 样本转换成画像统计。"""
        value_distribution = AnalysisEngine.continuous_stats(
            np.asarray(value_samples, dtype=float), None
        ) if value_samples else {}
        texture_distribution = AnalysisEngine.continuous_stats(
            np.asarray(texture_samples, dtype=float), None
        ) if texture_samples else {}
        scale_summary: Dict[str, Any] = {"neighborhood_window_pixels": 3}
        if scales_m:
            scale_summary.update({
                "nominal_scale_m_min": round(min(scales_m), 4),
                "nominal_scale_m_max": round(max(scales_m), 4),
                "nominal_scale_m_median": round(float(np.median(scales_m)), 4),
                "texture_window_m_median": round(float(np.median(scales_m)) * 3, 4),
            })
        status = "sampled" if value_distribution else ("failed" if sample_errors else "empty")
        result: Dict[str, Any] = {
            "remote_sampling_summary": {
                "status": status,
                "sampling_method": "bounded_remote_asset_window",
                "sampled_asset_count": sampled_asset_count,
                "sampled_asset_ids": sampled_asset_ids[:12],
                "sampled_pixel_count": len(value_samples),
                "errors": sample_errors,
            },
            "scale_summary": scale_summary,
        }
        if value_distribution:
            result["value_distribution"] = value_distribution
        if texture_distribution:
            result["spatial_texture_summary"] = {
                "metric": "3x3_neighborhood_local_stddev",
                "distribution": texture_distribution,
            }
        return result

    def _sample_stac_assets(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        value_samples: List[float] = []
        texture_samples: List[float] = []
        sampled_ids: List[str] = []
        errors: List[str] = []
        scales_m: List[float] = []
        asset_read_audit: List[Dict[str, Any]] = []
        max_assets = max(1, min(int(self.gee_profile_max_images), 12))
        for item in items[:max_assets]:
            item_id = str(item.get("id") or "").strip()
            key, asset = STACAssetReader.choose_asset(item)
            if asset is None:
                errors.append(f"{item_id or '<unknown>'}: no supported window-readable raster asset; archive/SAFE packages require a product-specific reader")
                continue
            try:
                data, nodata, meta = self.remote_asset_reader.read_window(
                    asset, max_pixels=self.remote_profile_max_pixels
                )
                if isinstance(meta, dict):
                    asset_read_audit.append({"item_id": item_id, "asset_key": key, **meta})
                arr = np.asarray(data)
                if arr.ndim != 2:
                    arr = np.asarray(arr).reshape(1, -1)
                flat = arr.astype(float).reshape(-1)
                finite = np.isfinite(flat)
                if nodata is not None:
                    finite &= flat != float(nodata)
                values = flat[finite]
                if values.size:
                    value_samples.extend(values.tolist())
                    texture_arr = arr.astype(float, copy=True)
                    texture_arr[~np.isfinite(texture_arr)] = np.nan
                    if nodata is not None:
                        texture_arr[texture_arr == float(nodata)] = np.nan
                    textures = AnalysisEngine.local_stddev(texture_arr)
                    if textures.size:
                        texture_samples.extend(textures[np.isfinite(textures)].tolist())
                    sampled_ids.append(item_id or key or "<asset>")
                scale = meta.get("nominal_scale_m") if isinstance(meta, dict) else None
                if isinstance(scale, (int, float)) and math.isfinite(float(scale)) and float(scale) > 0:
                    scales_m.append(float(scale))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{item_id or '<unknown>'}: {type(exc).__name__}: {exc}")
        analysis = self._build_remote_sample_analysis(
            value_samples=value_samples,
            texture_samples=texture_samples,
            sampled_asset_count=len(sampled_ids),
            sampled_asset_ids=sampled_ids,
            sample_errors=errors,
            scales_m=scales_m,
        )
        analysis["remote_sampling_summary"]["asset_read_audit"] = asset_read_audit
        if any(item.get("sample_semantics") == "complex_amplitude" for item in asset_read_audit):
            analysis["remote_sampling_summary"]["value_semantics"] = "SLC complex amplitude; no phase, coherence or displacement was derived"
        return analysis

    async def _analyze_external_remote(
        self,
        remote_source: Dict[str, Any],
        data_description: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Profile non-GEE sources from collection/item or OSM metadata.

        This is deliberately bounded: it reads collection metadata and at most
        ``gee_profile_max_images`` STAC items, then reads only bounded windows;
        full assets are never downloaded beyond the staging byte limit.
        """
        source = str(remote_source.get("dataset_source") or "").strip().upper()
        dataset_id = str(remote_source.get("dataset_id") or "").strip()
        if not source or not dataset_id:
            return {
                "profile_status": "remote_source_invalid",
                "error": "external source and dataset id are required",
                "observability_note": "External source metadata was incomplete; no remote request was made.",
            }

        limit = max(1, min(int(self.gee_profile_max_images), 12))
        scope = data_description.get("remote_profile_scope", data_description.get("gee_profile_scope", {}))
        if not isinstance(scope, dict):
            raise ValueError("remote_profile_scope must be an object")
        query_scope = {}
        if "bbox" in scope:
            bbox = scope["bbox"]
            if (not isinstance(bbox, (list, tuple)) or len(bbox) != 4
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in bbox)
                    or not (-180 <= bbox[0] < bbox[2] <= 180)
                    or not (-90 <= bbox[1] < bbox[3] <= 90)):
                raise ValueError("remote_profile_scope.bbox must be [west, south, east, north] in WGS84")
            query_scope["bbox"] = list(bbox)
        if "start_date" in scope or "end_date" in scope:
            if "start_date" not in scope or "end_date" not in scope:
                raise ValueError("remote_profile_scope requires both start_date and end_date")
            start = datetime.strptime(scope["start_date"], "%Y-%m-%d")
            end = datetime.strptime(scope["end_date"], "%Y-%m-%d")
            if start >= end:
                raise ValueError("remote_profile_scope dates must define a positive interval")
            query_scope["datetime_range"] = f"{scope['start_date']}/{scope['end_date']}"
        try:
            if source == "PANGAEA":
                from ..tools.pangaea_client import query_pangaea
                result = await query_pangaea(dataset_id)
            elif source == "NASA_CMR":
                result = await query_nasa_cmr(dataset_id, limit=limit, **query_scope)
            elif source == "PLANETARY_COMPUTER":
                result = await query_planetary_computer(dataset_id, limit=limit, **query_scope)
            elif source == "EARTH_SEARCH":
                result = await query_earth_search(dataset_id, limit=limit, **query_scope)
            elif source == "OSM_TAGINFO":
                result = await query_osm_taginfo(dataset_id)
            elif source == "OSM_OVERPASS":
                query = str(remote_source.get("overpass_ql") or "").strip()
                if not query:
                    return {
                        "profile_status": "remote_source_invalid",
                        "error": "OSM_OVERPASS requires overpass_ql",
                        "observability_note": "No Overpass query was supplied; no remote request was made.",
                    }
                result = await query_osm_overpass(
                    query,
                    endpoint=str(remote_source.get("endpoint") or "").strip()
                    or "https://overpass-api.de/api/interpreter",
                    timeout=30,
                )
            else:
                return {"profile_status": "unsupported_remote_source", "error": source}
        except Exception as exc:  # noqa: BLE001
            logger.warning("External dataset profiling failed for %s:%s: %s", source, dataset_id, exc)
            return {
                "profile_status": "remote_request_failed",
                "error": f"{type(exc).__name__}: {exc}",
                "observability_note": "Remote metadata request failed; quantitative profiling was not performed.",
            }

        if result.get("error"):
            return {
                "profile_status": "remote_request_failed",
                "error": str(result["error"]),
                "remote_source": source,
                "remote_dataset_id": dataset_id,
                "observability_note": "Remote metadata request failed; quantitative profiling was not performed.",
            }

        if source == "PANGAEA":
            return {
                "profile_status": "remote_metadata_profiled",
                "remote_source": source,
                "remote_dataset_id": dataset_id,
                "pangaea_metadata_summary": result["record"],
                "coverage_summary": {"n_files_analyzed": 0},
                "observability_note": "PANGAEA DOI metadata and typed relations only; no measurements were downloaded or statistically profiled. Collected dates may differ from the time represented by a palaeoclimate record.",
            }

        if source in {"NASA_CMR", "PLANETARY_COMPUTER", "EARTH_SEARCH"}:
            items = [item for item in (result.get("items") or []) if isinstance(item, dict)]
            if source == "PLANETARY_COMPUTER":
                token = await get_planetary_computer_sas_token(dataset_id)
                if token:
                    for item in items:
                        sign_planetary_computer_item(item, token)
            item_summary = self._summarize_stac_items(items)
            profile: Dict[str, Any] = {
                "profile_status": "remote_metadata_profiled",
                "profile_scope": scope,
                "scope_note": "Task-scoped catalog sample." if query_scope else "Dataset sample only; study region and event dates are unverified.",
                "remote_source": source,
                "remote_dataset_id": dataset_id,
                "note": "Remote STAC metadata profile completed with a bounded item sample.",
                "coverage_summary": {
                    "n_files_total": len(items),
                    "n_files_analyzed": 0,
                    "n_metadata_items_analyzed": len(items),
                    "sample_is_bounded": True,
                },
                "stac_metadata_summary": {
                    "collection_url": result.get("collection_url") or remote_source.get("stac_url"),
                    "title": result.get("title") or data_description.get("dataset_name"),
                    "description": str(result.get("description") or "")[:1000],
                    "declared_temporal_coverage": data_description.get("temporal_coverage"),
                    "declared_spatial_coverage": data_description.get("spatial_coverage"),
                    **item_summary,
                },
                "observability_note": (
                    "Profile derived from STAC collection metadata and a bounded item sample; "
                    "pixel-level statistics require explicit asset staging."
                ),
            }
            if DatasetTypeDetector.detect(data_description) == "continuous_raster" and items:
                sample_analysis = await asyncio.to_thread(self._sample_stac_assets, items)
                profile.update(sample_analysis)
                sampled_count = sample_analysis.get("remote_sampling_summary", {}).get("sampled_asset_count")
                if isinstance(sampled_count, int):
                    profile["coverage_summary"]["n_files_analyzed"] = sampled_count
                if sample_analysis.get("value_distribution"):
                    profile["profile_status"] = "remote_assets_profiled"
                    profile["observability_note"] = (
                        "Profile includes bounded remote asset windows; values and texture are "
                        "representative estimates, not an exhaustive dataset census."
                    )
                else:
                    profile["observability_note"] = (
                        "STAC metadata was profiled, but no bounded asset window was readable; "
                        "quantitative statistics were omitted."
                    )
            return profile

        if source == "OSM_TAGINFO":
            return {
                "profile_status": "remote_metadata_profiled",
                "remote_source": source,
                "remote_dataset_id": dataset_id,
                "note": "Remote OSM tag metadata profile completed.",
                "coverage_summary": {"n_files_total": 1, "n_files_analyzed": 1},
                "osm_metadata_summary": {
                    "tag_key": dataset_id,
                    "exists": bool(result.get("exists")),
                    "api_url": result.get("api_url"),
                },
                "observability_note": "Profile derived from OSM Taginfo metadata; feature-level statistics require an Overpass query.",
            }

        elements = [item for item in (result.get("elements") or []) if isinstance(item, dict)]
        tag_keys = Counter(
            key
            for element in elements
            for key in ((element.get("tags") or {}).keys() if isinstance(element.get("tags"), dict) else [])
        )
        geometry_types = Counter(str(element.get("type") or "unknown") for element in elements)
        points = [
            (float(element["lon"]), float(element["lat"]))
            for element in elements
            if isinstance(element.get("lon"), (int, float))
            and isinstance(element.get("lat"), (int, float))
        ]
        geometry_summary: Dict[str, Any] = {
            "feature_count": len(elements),
            "geometry_types": dict(geometry_types),
        }
        if points:
            geometry_summary["bbox"] = [
                min(point[0] for point in points),
                min(point[1] for point in points),
                max(point[0] for point in points),
                max(point[1] for point in points),
            ]
        return {
            "profile_status": "remote_metadata_profiled",
            "remote_source": source,
            "remote_dataset_id": dataset_id,
            "note": "Remote OSM Overpass metadata profile completed with a bounded response.",
            "coverage_summary": {"n_rows_loaded": len(elements), "n_files_total": 1},
            "geometry_summary": geometry_summary,
            "osm_metadata_summary": {
                "element_count": len(elements),
                "top_tag_keys": tag_keys.most_common(12),
            },
            "observability_note": "Profile derived from a bounded OSM Overpass response.",
        }

    def _analyze_gee_metadata(
        self,
        remote_source: Dict[str, Any],
        data_description: Dict[str, Any],
    ) -> Dict[str, Any]:
        ee_id = str(remote_source.get("ee_id") or "").strip()
        ee_ctor = str(remote_source.get("ee_ctor") or "").strip()
        if not ee_id or not ee_ctor:
            return {"note": "GEE source declared but ee_id/ee_ctor is incomplete."}

        import ee

        initialize_gee_from_service_account(ee)
        target = self._build_gee_target(ee, ee_ctor, ee_id)
        scope = data_description.get("gee_profile_scope", {})
        if not isinstance(scope, dict):
            raise ValueError("gee_profile_scope must be an object")
        target = self._apply_gee_profile_scope(ee, target, ee_ctor, scope)
        if ee_ctor == "ee.ImageCollection":
            target = target.limit(max(1, self.gee_profile_max_images))
        logger.info("GEE profile metadata asset=%s scope=%s image_limit=%s",
                    ee_id, scope, self.gee_profile_max_images)
        gee_summary = self._summarize_gee_target(target, ee_ctor)
        gee_summary["profile_scope"] = scope

        collection_size = gee_summary.get("collection_size")
        remote_item_count = (
            int(collection_size)
            if isinstance(collection_size, (int, float)) and collection_size > 0
            else 1
        )
        result = {
            "coverage_summary": {
                "remote_source": "gee",
                "n_files_total": remote_item_count,
                "n_files_analyzed": 1,
                "sample_is_bounded": ee_ctor == "ee.ImageCollection",
                "count_scope": "profile_subset" if ee_ctor == "ee.ImageCollection" else "asset",
            },
            "observability_note": "Profile derived from Google Earth Engine metadata without local staging.",
            "gee_metadata_summary": {
                "ee_id": ee_id,
                "ee_ctor": ee_ctor,
                **gee_summary,
                "profile_scope": scope,
                "scope_note": (
                    "Explicit profile filters; sampled statistics describe a bounded subset."
                    if scope else
                    "Dataset sample only; study region, event dates and polarizations are unverified."
                ),
            },
        }

        if DatasetTypeDetector.detect(data_description) == "continuous_raster":
            try:
                sample_analysis = self._sample_gee_continuous_raster(
                    ee,
                    target,
                    ee_ctor,
                    gee_summary,
                )
                result.update(sample_analysis)
                sampled_image_count = (
                    sample_analysis.get("gee_sampling_summary", {})
                    .get("sampled_image_count")
                )
                if sampled_image_count:
                    result["coverage_summary"]["n_files_analyzed"] = sampled_image_count
                if sample_analysis.get("value_distribution"):
                    result["observability_note"] = (
                        "Profile includes bounded windows from a limited GEE asset subset; "
                        "statistics describe only the sampled windows, not the study region or full archive."
                    )
            except Exception as e:
                result["gee_sampling_summary"] = {
                    "status": "failed",
                    "error": self._format_sampling_error(e),
                    "sampling_method": "bounded_collection_window_sample",
                }
                logger.warning("DatasetProfilerAgent bounded GEE sampling failed: %s", e)

        if ee_ctor == "ee.FeatureCollection":
            try:
                result.update(self._sample_gee_vector_attributes(ee, target))
                result["observability_note"] = (
                    "Profile includes sampled feature attributes with observed values and a "
                    "geometry-based unit check; attribute semantics are inferred from a bounded "
                    "sample, not an exhaustive census."
                )
            except Exception as e:
                result["gee_vector_sampling_summary"] = {
                    "status": "failed",
                    "error": self._format_sampling_error(e),
                    "sampling_method": "first_feature_attribute_probe",
                }
                logger.warning(
                    "DatasetProfilerAgent GEE vector attribute sampling failed: %s", e
                )

        data_type = str(data_description.get("data_type") or "").lower()
        if "raster" in data_type or ee_ctor in {"ee.Image", "ee.ImageCollection"}:
            bands = gee_summary.get("band_names") or []
            if bands:
                result["attribute_summary"] = {
                    "bands": {
                        "type": "categorical",
                        "distinct_count": len(bands),
                        "top_values": [{"value": band, "count": 1} for band in bands[:5]],
                    }
                }
        return result

    def _apply_gee_profile_scope(
        self, ee: Any, target: Any, ee_ctor: str, scope: Dict[str, Any]
    ) -> Any:
        if not scope:
            return target
        if ee_ctor != "ee.ImageCollection" and ("start_date" in scope or "end_date" in scope):
            raise ValueError("date-scoped profiling requires an ImageCollection")
        unknown = set(scope) - {"bbox", "start_date", "end_date", "bands"}
        if unknown:
            raise ValueError(f"Unsupported gee_profile_scope fields: {sorted(unknown)}")
        if "bbox" in scope:
            bbox = scope["bbox"]
            if (
                not isinstance(bbox, (list, tuple)) or len(bbox) != 4
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) for v in bbox)
                or not (-180 <= bbox[0] < bbox[2] <= 180)
                or not (-90 <= bbox[1] < bbox[3] <= 90)
            ):
                raise ValueError("gee_profile_scope.bbox must be [west, south, east, north] in WGS84")
            if ee_ctor == "ee.ImageCollection":
                target = target.filterBounds(ee.Geometry.Rectangle(bbox))
        if "start_date" in scope or "end_date" in scope:
            if "start_date" not in scope or "end_date" not in scope:
                raise ValueError("gee_profile_scope requires both start_date and end_date")
            start = datetime.strptime(scope["start_date"], "%Y-%m-%d")
            end = datetime.strptime(scope["end_date"], "%Y-%m-%d")
            if start >= end:
                raise ValueError("gee_profile_scope dates must define a positive interval")
            target = target.filterDate(scope["start_date"], scope["end_date"])
        if "bands" in scope:
            bands = scope["bands"]
            if not isinstance(bands, list) or not bands or any(
                not isinstance(band, str) or not band.strip() for band in bands
            ):
                raise ValueError("gee_profile_scope.bands must contain non-empty band names")
            if ee_ctor == "ee.ImageCollection":
                target = target.map(lambda image: image.set("__profile_bands", image.bandNames()))
                for band in bands:
                    target = target.filter(ee.Filter.listContains("__profile_bands", band))
            target = target.select(bands)
        return target

    def _sample_gee_vector_attributes(self, ee: Any, target: Any) -> Dict[str, Any]:
        """采样向量要素的属性实测值，并对面积类属性做单位反推。

        `_summarize_gee_target` 对 FeatureCollection 已经取到 first feature 的 properties，
        但只留了键名、丢掉了值。而"某个属性的单位是什么"这类问题的答案就在值里：把
        AREA 字段的数值与 `geometry().area()` 的实测平方米对比，比值接近 1 说明是 m²、
        接近 1e-6 说明是 km²、接近 1e-4 说明是公顷。

        这类问题此前会一路穿到 idea 成熟化：任务描述里写着"AREA 单位未在 runtime probe
        中声明，聚合前需核实"，质询据此正确提出 finding，通道判定把它判成需要数据集，
        于是拿 `verification area attribute square` 去查 catalog，召回的是 WorldPop 人口
        数据（描述里含 "per square kilometre"），最终零召回收口、idea 空转。答案本来
        一次属性探测就能给出。
        """
        sampled = target.limit(VECTOR_ATTRIBUTE_SAMPLE_SIZE).getInfo() or {}
        features = sampled.get("features") or []
        if not features:
            return {
                "gee_vector_sampling_summary": {
                    "status": "empty",
                    "sampling_method": "first_feature_attribute_probe",
                }
            }

        observed: Dict[str, List[Any]] = {}
        for feature in features:
            if not isinstance(feature, dict):
                continue
            for key, value in (feature.get("properties") or {}).items():
                observed.setdefault(str(key), []).append(value)

        attribute_observations = {
            key: {
                "observed_values": values[:VECTOR_ATTRIBUTE_SAMPLE_SIZE],
                "value_type": type(values[0]).__name__ if values else "unknown",
            }
            for key, values in observed.items()
        }

        summary: Dict[str, Any] = {
            "gee_vector_sampling_summary": {
                "status": "ok",
                "sampling_method": "first_feature_attribute_probe",
                "sampled_feature_count": len(features),
            },
            "vector_attribute_observations": attribute_observations,
        }

        unit_checks = self._infer_vector_area_units(ee, target, observed)
        if unit_checks:
            summary["vector_attribute_unit_checks"] = unit_checks
        return summary

    def _infer_vector_area_units(
        self,
        ee: Any,
        target: Any,
        observed: Dict[str, List[Any]],
    ) -> Dict[str, Any]:
        """用几何实测面积反推面积类属性的单位。

        只对名字里含 area 且取到数值的属性做，避免对无关字段发起额外 GEE 调用。
        """
        area_fields = [
            key
            for key, values in observed.items()
            if "area" in key.lower()
            and values
            and isinstance(values[0], (int, float))
            and not isinstance(values[0], bool)
        ]
        if not area_fields:
            return {}

        first = ee.Feature(target.first())
        geometry_area_m2 = float(
            first.geometry().area(maxError=VECTOR_AREA_MAX_ERROR_M).getInfo()
        )
        if geometry_area_m2 <= 0:
            return {}

        checks: Dict[str, Any] = {}
        for field in area_fields:
            declared = float(observed[field][0])
            ratio = declared / geometry_area_m2
            checks[field] = {
                "declared_value": declared,
                "geometry_area_m2": round(geometry_area_m2, 3),
                "declared_to_m2_ratio": round(ratio, 9),
                "inferred_unit": self._classify_area_unit(ratio),
            }
        return checks

    @staticmethod
    def _classify_area_unit(ratio: float) -> str:
        """按 declared/实测m² 的比值判定单位；容忍投影与简化带来的偏差。"""
        for unit, expected in (
            ("square_meters", 1.0),
            ("hectares", 1e-4),
            ("square_kilometers", 1e-6),
        ):
            if expected / VECTOR_AREA_UNIT_TOLERANCE <= ratio <= expected * VECTOR_AREA_UNIT_TOLERANCE:
                return unit
        return "unresolved"

    @staticmethod
    def _format_sampling_error(exc: Exception) -> str:
        message = str(exc).strip()
        return f"{type(exc).__name__}: {message}" if message else type(exc).__name__

    def _sample_gee_continuous_raster(
        self,
        ee: Any,
        target: Any,
        ee_ctor: str,
        gee_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        band_names = gee_summary.get("band_names") or []
        if not band_names:
            return {
                "gee_sampling_summary": {
                    "status": "skipped_no_band",
                    "sampling_method": "bounded_collection_window_sample",
                }
            }

        if ee_ctor == "ee.ImageCollection" and gee_summary.get("collection_size") == 0:
            return {"gee_sampling_summary": {"status": "skipped_empty_collection"}}
        sample_images, image_offsets = self._select_gee_profile_images(
            ee,
            target,
            ee_ctor,
            gee_summary.get("collection_size"),
        )
        value_samples: List[float] = []
        texture_samples: List[float] = []
        nominal_scales: List[float] = []
        sample_errors: List[str] = []
        sampled_image_count = 0

        for sample_index, image in enumerate(sample_images):
            try:
                logger.info("GEE profile reading projection image %s band %s",
                            sample_index, band_names[0])
                value_image = image.select([band_names[0]]).rename("profile_value")
                nominal_scale = float(value_image.projection().nominalScale().getInfo())
                if not math.isfinite(nominal_scale) or nominal_scale <= 0:
                    raise ValueError("GEE profile requires a finite positive native scale")
                nominal_scales.append(nominal_scale)

                logger.info("GEE profile sampling image %s band %s at %s m",
                            sample_index, band_names[0], nominal_scale)
                # Bound the evaluated area, not just the number of returned pixels.
                radius = nominal_scale * math.sqrt(self.remote_profile_max_pixels) / 2
                footprint = value_image.geometry()
                profile_scope = gee_summary.get("profile_scope", {})
                if "bbox" in profile_scope:
                    footprint = footprint.intersection(
                        ee.Geometry.Rectangle(profile_scope["bbox"]), maxError=nominal_scale
                    )
                # A footprint centroid may lie outside a concave or disjoint footprint.
                # Select an interior anchor instead, including for global DEM assets.
                anchor = ee.FeatureCollection.randomPoints(
                    region=footprint, points=1, seed=1729 + sample_index,
                    maxError=nominal_scale,
                ).first().geometry()
                region = anchor.buffer(
                    radius, maxError=nominal_scale
                ).bounds(maxError=nominal_scale).intersection(
                    footprint, maxError=nominal_scale
                )
                value_image = value_image.clip(region)

                texture_image = value_image.reduceNeighborhood(
                    reducer=ee.Reducer.stdDev(),
                    kernel=ee.Kernel.square(
                        radius=1,
                        units="pixels",
                        normalize=False,
                    ),
                ).rename("profile_local_stddev")
                profile_image = value_image.addBands(texture_image)
                sample_payload = profile_image.sample(
                    region=region,
                    scale=nominal_scale,
                    numPixels=max(1, self.gee_profile_pixels_per_image),
                    seed=1729 + sample_index,
                    dropNulls=True,
                    geometries=False,
                    tileScale=4,
                ).getInfo()

                sampled_values, sampled_textures = self._extract_gee_sample_values(sample_payload)
                value_samples.extend(sampled_values)
                texture_samples.extend(sampled_textures)
                sampled_image_count += 1
            except Exception as e:
                sample_errors.append(self._format_sampling_error(e))
                logger.warning("GEE profile sampling image %s failed: %s", sample_index, e)

        return self._build_gee_sample_analysis(
            value_samples=value_samples,
            texture_samples=texture_samples,
            nominal_scales=nominal_scales,
            sampled_image_count=sampled_image_count,
            sampled_band=str(band_names[0]),
            image_offsets=image_offsets,
            sample_errors=sample_errors,
        )

    def _select_gee_profile_images(
        self,
        ee: Any,
        target: Any,
        ee_ctor: str,
        collection_size: Any,
    ) -> Tuple[List[Any], List[int]]:
        if ee_ctor == "ee.Image":
            return [target], [0]
        if ee_ctor != "ee.ImageCollection":
            return [], []

        size = int(collection_size) if collection_size is not None else 0
        if size <= 0:
            return [ee.Image(target.first())], [0]

        image_count = min(max(1, self.gee_profile_max_images), size)
        offsets = list(range(image_count))
        bounded = target.limit(image_count).toList(image_count)
        images = [ee.Image(bounded.get(offset)) for offset in offsets]
        return images, offsets

    @staticmethod
    def _extract_gee_sample_values(sample_payload: Any) -> Tuple[List[float], List[float]]:
        features = sample_payload.get("features", []) if isinstance(sample_payload, dict) else []
        values: List[float] = []
        textures: List[float] = []
        for feature in features:
            properties = feature.get("properties", {}) if isinstance(feature, dict) else {}
            value = properties.get("profile_value")
            texture = properties.get("profile_local_stddev")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_value = float(value)
                if math.isfinite(numeric_value):
                    values.append(numeric_value)
            if isinstance(texture, (int, float)) and not isinstance(texture, bool):
                numeric_texture = float(texture)
                if math.isfinite(numeric_texture):
                    textures.append(numeric_texture)
        return values, textures

    def _build_gee_sample_analysis(
        self,
        *,
        value_samples: List[float],
        texture_samples: List[float],
        nominal_scales: List[float],
        sampled_image_count: int,
        sampled_band: str,
        image_offsets: List[int],
        sample_errors: List[str],
    ) -> Dict[str, Any]:
        result = self._build_remote_sample_analysis(
            value_samples=value_samples,
            texture_samples=texture_samples,
            sampled_asset_count=sampled_image_count,
            sampled_asset_ids=[f"offset:{offset}" for offset in image_offsets],
            sample_errors=sample_errors,
            scales_m=nominal_scales,
        )
        summary = result.pop("remote_sampling_summary")
        result["gee_sampling_summary"] = {
            **summary,
            "sampling_method": "bounded_collection_window_sample",
            "window_pixel_budget": self.remote_profile_max_pixels,
            "sampled_band": sampled_band,
            "requested_image_limit": max(1, self.gee_profile_max_images),
            "requested_pixels_per_image": max(1, self.gee_profile_pixels_per_image),
            "sampled_image_count": sampled_image_count,
            "image_offsets": image_offsets,
        }
        return result

    def _build_gee_target(self, ee: Any, ee_ctor: str, ee_id: str) -> Any:
        if ee_ctor == "ee.Image":
            return ee.Image(ee_id)
        if ee_ctor == "ee.ImageCollection":
            return ee.ImageCollection(ee_id)
        if ee_ctor == "ee.FeatureCollection":
            return ee.FeatureCollection(ee_id)
        raise ValueError(f"Unsupported GEE constructor for dataset profiling: {ee_ctor}")

    def _summarize_gee_target(self, target: Any, ee_ctor: str) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}

        if ee_ctor == "ee.Image":
            info = target.getInfo() or {}
            bands = info.get("bands", []) if isinstance(info, dict) else []
            summary["band_names"] = [
                str(band.get("id") or "").strip()
                for band in bands
                if isinstance(band, dict) and str(band.get("id") or "").strip()
            ]
            summary["properties"] = sorted((info.get("properties") or {}).keys())[:20]
            return summary

        if ee_ctor == "ee.ImageCollection":
            first = target.first()
            info = first.getInfo() if first else {}
            bands = info.get("bands", []) if isinstance(info, dict) else []
            summary["band_names"] = [
                str(band.get("id") or "").strip()
                for band in bands
                if isinstance(band, dict) and str(band.get("id") or "").strip()
            ]
            try:
                summary["collection_size"] = int(target.size().getInfo())
            except Exception:
                summary["collection_size"] = None
            return summary

        if ee_ctor == "ee.FeatureCollection":
            try:
                summary["feature_count"] = int(target.size().getInfo())
            except Exception:
                summary["feature_count"] = None
            first = target.first()
            info = first.getInfo() if first else {}
            props = {}
            if isinstance(info, dict):
                props = info.get("properties") or {}
            summary["properties"] = sorted(props.keys())[:20]
            geometry = info.get("geometry") if isinstance(info, dict) else None
            if isinstance(geometry, dict):
                summary["geometry_type"] = geometry.get("type")
            return summary

        return summary

    async def _analyze_classified_raster(
        self,
        file_paths: List[str],
        data_description: Dict[str, Any],
        dataset_type: str,
        hotspot_target_classes: List[int],
    ) -> Dict[str, Any]:
        class_labels = DatasetTypeDetector.extract_class_labels(data_description)
        pixel_area = DatasetTypeDetector.extract_pixel_area(data_description)

        # 找到所有栅格文件（可能是目录路径）
        all_raster_files: List[Tuple[Optional[int], str]] = []
        for path in file_paths:
            if os.path.isdir(path):
                rfiles = DataPathResolver.list_raster_files(path)
                for rf in rfiles:
                    year = DataPathResolver.parse_year_from_filename(rf)
                    all_raster_files.append((year, rf))
            elif os.path.isfile(path):
                year = DataPathResolver.parse_year_from_filename(path)
                all_raster_files.append((year, path))

        if not all_raster_files:
            logger.warning("DatasetProfilerAgent No raster files found.")
            return {"note": "No raster files found at resolved paths."}

        # 按年份排序，限制加载数量
        all_raster_files.sort(key=lambda x: (x[0] or 0, x[1]))
        files_to_load = all_raster_files[:self.max_years_to_load]

        # 逐年加载
        yearly_data: Dict[int, np.ndarray] = {}
        nodata_val = None
        file_meta = None
        load_errors = []
        dummy_year_counter = 9000  # 无法解析年份时的占位

        for year, fpath in files_to_load:
            try:
                arr, nd, meta = RasterReader.read(fpath, max_pixels=self.max_pixels)
                if nodata_val is None and nd is not None:
                    nodata_val = nd
                if file_meta is None:
                    file_meta = meta
                key = year if year is not None else dummy_year_counter
                dummy_year_counter += 1
                yearly_data[key] = arr
            except Exception as e:
                load_errors.append({"file": fpath, "error": str(e)})
                logger.warning(f"DatasetProfilerAgent Cannot load {fpath}: {e}")

        if not yearly_data:
            return {
                "note": "All raster files failed to load.",
                "load_errors": load_errors,
            }

        # ── 全局汇总（所有年份叠加）
        global_concat = np.concatenate([a.flatten() for a in yearly_data.values()])
        global_dist = AnalysisEngine.class_distribution(
            global_concat.reshape(1, -1), nodata_val, class_labels, pixel_area
        )

        # ── 逐年类别面积（仅真实年份）
        real_year_data = {y: arr for y, arr in yearly_data.items() if y < 9000}
        # 默认分析所有非背景类别（排除 nodata 值类别如 5）
        active_classes = [int(k) for k in class_labels if int(k) not in (5,)]
        if nodata_val is not None:
            active_classes = [c for c in active_classes if c != int(nodata_val)]

        yearly_area = AnalysisEngine.timeseries_class_area(
            real_year_data, nodata_val, active_classes, pixel_area
        )

        # ── 趋势检验（每个类别）
        trend_by_class = {}
        for cls in active_classes:
            cls_series = {y: yearly_area[y].get(cls, 0.0) for y in sorted(yearly_area)}
            trend_by_class[cls] = AnalysisEngine.mann_kendall_trend(cls_series)

        # ── 异常年份（高严重度类别，默认 class 3）
        anomaly_years = {}
        for cls in hotspot_target_classes:
            if cls in active_classes:
                cls_series = {y: yearly_area[y].get(cls, 0.0) for y in sorted(yearly_area)}
                anomaly_years[cls] = AnalysisEngine.detect_anomaly_years(cls_series)

        # ── 空间热点（取面积最大年份的栅格做代表）
        hotspot = {}
        if real_year_data and hotspot_target_classes:
            hotspot_cls = hotspot_target_classes[0]
            if hotspot_cls in active_classes and yearly_area:
                best_year = max(yearly_area, key=lambda y: yearly_area[y].get(hotspot_cls, 0))
                if best_year in real_year_data:
                    hotspot = AnalysisEngine.spatial_hotspot_summary(
                        real_year_data[best_year], nodata_val, [hotspot_cls]
                    )
                    hotspot["representative_year"] = best_year

        # ── 年际变异系数（量化年际波动幅度）
        cv_by_class = {}
        for cls in active_classes:
            vals = [yearly_area[y].get(cls, 0.0) for y in sorted(yearly_area)]
            if len(vals) >= 3 and np.mean(vals) > 1e-6:
                cv_by_class[cls] = round(float(np.std(vals) / np.mean(vals)), 3)

        # ── 加载的年份列表
        loaded_years = sorted(y for y in yearly_data if y < 9000)
        try:
            temporal_str = str(data_description.get("temporal_coverage", ""))
            expected_years = []
            if "-" in temporal_str:
                parts = temporal_str.split("-")
                start, end = int(parts[0].strip()), int(parts[1].strip())
                expected_years = list(range(start, end + 1))
        except (ValueError, IndexError):
            expected_years = []

        missing_years = [y for y in expected_years if y not in loaded_years]

        return {
            "coverage_summary": {
                "n_files_total": len(all_raster_files),
                "n_years_loaded": len(loaded_years),
                "loaded_years": loaded_years,
                "missing_years": missing_years,
                "load_errors": load_errors,
            },
            "global_class_distribution": global_dist,
            "yearly_class_area_ha": yearly_area,
            "trend_by_class": trend_by_class,
            "anomaly_years": anomaly_years,
            "spatial_hotspot": hotspot,
            "interannual_cv": cv_by_class,
            "observability_note": (
                f"Dataset covers {len(loaded_years)} time steps of mapped categorical raster observations. "
                "Direct observables: class composition, per-time-step mapped area by class, "
                "spatial concentration of selected classes, and inter-temporal variability."
            ),
        }

    async def _analyze_continuous_raster(
        self,
        file_paths: List[str],
        data_description: Dict[str, Any],
    ) -> Dict[str, Any]:
        all_raster_files = []
        for path in file_paths:
            if os.path.isdir(path):
                rfiles = DataPathResolver.list_raster_files(path)
                for rf in rfiles:
                    year = DataPathResolver.parse_year_from_filename(rf)
                    all_raster_files.append((year, rf))
            elif os.path.isfile(path):
                year = DataPathResolver.parse_year_from_filename(path)
                all_raster_files.append((year, path))

        if not all_raster_files:
            return {"note": "No raster files found."}

        all_raster_files.sort(key=lambda x: (x[0] or 0, x[1]))
        files_to_load = all_raster_files[:self.max_years_to_load]

        yearly_stats = {}
        nodata_val = None
        for year, fpath in files_to_load:
            try:
                arr, nd, _ = RasterReader.read(fpath, max_pixels=self.max_pixels)
                if nodata_val is None and nd is not None:
                    nodata_val = nd
                stats_result = AnalysisEngine.continuous_stats(arr, nd)
                key = year if year is not None else fpath
                yearly_stats[key] = stats_result
            except Exception as e:
                logger.warning(f"DatasetProfilerAgent Cannot load {fpath}: {e}")

        # 全局汇总
        all_means = {k: v["mean"] for k, v in yearly_stats.items()
                     if isinstance(k, int) and "mean" in v}
        trend = AnalysisEngine.mann_kendall_trend(all_means) if len(all_means) >= 4 else {}
        anomalies = AnalysisEngine.detect_anomaly_years(all_means) if len(all_means) >= 4 else []

        return {
            "yearly_stats": yearly_stats,
            "trend_of_mean": trend,
            "anomaly_years_by_mean": anomalies,
            "observability_note": "Continuous raster; statistics describe value distributions and temporal changes in aggregate metrics.",
        }

    async def _analyze_vector_dataset(
        self,
        file_paths: List[str],
        data_description: Dict[str, Any],
    ) -> Dict[str, Any]:
        all_vector_files: List[str] = []
        for path in file_paths:
            if os.path.isdir(path):
                all_vector_files.extend(DataPathResolver.list_vector_files(path))
            elif os.path.isfile(path):
                all_vector_files.append(path)

        vector_files = [
            item for item in all_vector_files
            if Path(item).suffix.lower() in {".geojson", ".json", ".shp"}
        ]
        if not vector_files:
            return {"note": "No supported vector files found at resolved paths."}

        features: List[Dict[str, Any]] = []
        load_errors = []
        analyzed_files = []
        merged_bbox = None

        for fpath in vector_files[:5]:
            try:
                suffix = Path(fpath).suffix.lower()
                if suffix in {".geojson", ".json"}:
                    payload = VectorReader.read_geojson(fpath)
                else:
                    payload = VectorReader.read_shapefile(fpath)
                features.extend(payload.get("features", []))
                merged_bbox = VectorReader._merge_bbox(merged_bbox, payload.get("bbox"))
                analyzed_files.append(fpath)
            except Exception as e:
                load_errors.append({"file": fpath, "error": str(e)})
                logger.warning(f"DatasetProfilerAgent Cannot load vector file {fpath}: {e}")

        if not features:
            return {
                "note": "All vector files failed to load.",
                "load_errors": load_errors,
            }

        geometry_summary = self._summarize_vector_geometries(features, merged_bbox)
        attribute_summary = self._summarize_records(
            [item.get("properties", {}) for item in features],
            max_fields=8,
        )

        return {
            "coverage_summary": {
                "n_files_total": len(vector_files),
                "n_files_analyzed": len(analyzed_files),
                "analyzed_files": analyzed_files,
                "load_errors": load_errors,
            },
            "geometry_summary": geometry_summary,
            "attribute_summary": attribute_summary,
            "basic_attributes": attribute_summary,
            "observability_note": (
                f"Vector dataset with {geometry_summary.get('feature_count', 0)} features; direct observables include "
                "geometry inventory, bounding-box extent, and basic attribute distributions."
            ),
        }

    async def _analyze_tabular_timeseries(
        self,
        file_paths: List[str],
        data_description: Dict[str, Any],
    ) -> Dict[str, Any]:
        all_tabular_files: List[str] = []
        for path in file_paths:
            if os.path.isdir(path):
                all_tabular_files.extend(DataPathResolver.list_tabular_files(path))
            elif os.path.isfile(path):
                all_tabular_files.append(path)

        table_files = [
            item for item in all_tabular_files
            if Path(item).suffix.lower() in {".csv", ".tsv"}
        ]
        if not table_files:
            return {"note": "No tabular files found at resolved paths."}

        rows: List[Dict[str, Any]] = []
        load_errors = []
        analyzed_files = []
        max_rows = 200000
        for fpath in table_files[:5]:
            try:
                rows.extend(self._read_tabular_rows(fpath, max_rows=max_rows - len(rows)))
                analyzed_files.append(fpath)
                if len(rows) >= max_rows:
                    break
            except Exception as e:
                load_errors.append({"file": fpath, "error": str(e)})
                logger.warning(f"DatasetProfilerAgent Cannot load tabular file {fpath}: {e}")

        if not rows:
            return {
                "note": "All tabular files failed to load.",
                "load_errors": load_errors,
            }

        time_detection = self._detect_time_column(rows)
        attribute_summary = self._summarize_records(rows, max_fields=8)
        timeseries_summary = {
            "time_column": time_detection.get("time_column"),
            "detected_ratio": time_detection.get("detected_ratio"),
            "year_range": None,
            "n_time_steps": 0,
            "value_column": None,
            "aggregation": None,
            "yearly_values": {},
        }
        trend = {}
        anomalies = []

        if time_detection.get("time_column"):
            value_column = self._pick_primary_numeric_column(rows, exclude={time_detection["time_column"]})
            yearly_values, aggregation = self._aggregate_yearly_values(
                rows,
                time_column=time_detection["time_column"],
                value_column=value_column,
            )
            if yearly_values:
                years = sorted(yearly_values)
                timeseries_summary = {
                    "time_column": time_detection.get("time_column"),
                    "detected_ratio": time_detection.get("detected_ratio"),
                    "year_range": f"{years[0]}-{years[-1]}",
                    "n_time_steps": len(years),
                    "value_column": value_column or "__record_count__",
                    "aggregation": aggregation,
                    "yearly_values": yearly_values,
                }
                if len(yearly_values) >= 4:
                    trend = AnalysisEngine.mann_kendall_trend(yearly_values)
                    anomalies = AnalysisEngine.detect_anomaly_years(yearly_values)

        return {
            "coverage_summary": {
                "n_files_total": len(table_files),
                "n_files_analyzed": len(analyzed_files),
                "n_rows_loaded": len(rows),
                "analyzed_files": analyzed_files,
                "load_errors": load_errors,
            },
            "time_column_detection": time_detection,
            "timeseries_summary": timeseries_summary,
            "trend_of_mean": trend,
            "anomaly_years_by_mean": anomalies,
            "attribute_summary": attribute_summary,
            "observability_note": (
                f"Tabular time series with detected time column '{timeseries_summary.get('time_column') or 'unresolved'}'; "
                "direct observables include temporal coverage, aggregated trend, and anomaly screening."
            ),
        }

    def _read_tabular_rows(self, path: str, max_rows: int = 200000) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        suffix = Path(path).suffix.lower()
        delimiter = "	" if suffix == ".tsv" else ","
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                rows.append(dict(row))
                if len(rows) >= max_rows:
                    break
        return rows

    def _coerce_year(self, value: Any) -> Optional[int]:
        if value in (None, ""):
            return None
        if isinstance(value, (int, float)) and 1800 <= int(value) <= 2200:
            return int(value)
        text = str(value).strip()
        if not text:
            return None
        match = re.search(r"(19|20)\d{2}", text)
        if match:
            try:
                return int(match.group(0))
            except ValueError:
                return None
        return None

    def _detect_time_column(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not rows:
            return {"time_column": None, "detected_ratio": 0.0}

        columns: List[str] = []
        seen = set()
        for row in rows[:50]:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    columns.append(key)

        best = {"time_column": None, "score": 0.0, "detected_ratio": 0.0}
        for column in columns:
            raw_values = [row.get(column) for row in rows]
            non_empty = [value for value in raw_values if value not in (None, "")]
            if not non_empty:
                continue
            valid_years = [self._coerce_year(value) for value in non_empty]
            detected = sum(year is not None for year in valid_years)
            ratio = detected / len(non_empty)
            name_bonus = 0.15 if re.search(r"year|date|time|month", column, flags=re.IGNORECASE) else 0.0
            score = ratio + name_bonus
            if ratio >= 0.6 and score > best["score"]:
                best = {
                    "time_column": column,
                    "score": score,
                    "detected_ratio": round(ratio, 3),
                }
        return {k: v for k, v in best.items() if k != "score"}

    def _pick_primary_numeric_column(
        self,
        rows: List[Dict[str, Any]],
        exclude: Optional[set] = None,
    ) -> Optional[str]:
        exclude = exclude or set()
        candidates: Dict[str, Tuple[int, float]] = {}
        for row in rows[:5000]:
            for key, value in row.items():
                if key in exclude:
                    continue
                numeric_value = self._coerce_float(value)
                if numeric_value is None:
                    continue
                count, total = candidates.get(key, (0, 0.0))
                candidates[key] = (count + 1, total + abs(numeric_value))

        if not candidates:
            return None
        ranked = sorted(candidates.items(), key=lambda item: (item[1][0], item[1][1]), reverse=True)
        return ranked[0][0]

    def _aggregate_yearly_values(
        self,
        rows: List[Dict[str, Any]],
        time_column: str,
        value_column: Optional[str],
    ) -> Tuple[Dict[int, float], str]:
        grouped: Dict[int, List[float]] = defaultdict(list)
        record_counts: Dict[int, int] = defaultdict(int)

        for row in rows:
            year = self._coerce_year(row.get(time_column))
            if year is None:
                continue
            record_counts[year] += 1
            if value_column:
                numeric_value = self._coerce_float(row.get(value_column))
                if numeric_value is not None:
                    grouped[year].append(numeric_value)

        if value_column and any(grouped.values()):
            yearly_values = {
                year: round(float(np.mean(values)), 4)
                for year, values in grouped.items()
                if values
            }
            return dict(sorted(yearly_values.items())), "mean_by_year"

        yearly_counts = {
            year: float(count)
            for year, count in record_counts.items()
        }
        return dict(sorted(yearly_counts.items())), "record_count_by_year"

    def _summarize_vector_geometries(
        self,
        features: List[Dict[str, Any]],
        bbox: Optional[Dict[str, float]],
    ) -> Dict[str, Any]:
        geometry_types = Counter()
        for feature in features:
            geometry_types[str(feature.get("geometry_type") or "Unknown")] += 1
        dominant_geometry_type = None
        if geometry_types:
            dominant_geometry_type = max(geometry_types, key=geometry_types.get)
        return {
            "feature_count": len(features),
            "geometry_types": dict(geometry_types),
            "dominant_geometry_type": dominant_geometry_type,
            "bbox": bbox,
        }

    def _summarize_records(
        self,
        records: List[Dict[str, Any]],
        max_fields: int = 8,
    ) -> Dict[str, Any]:
        field_values: Dict[str, List[Any]] = defaultdict(list)
        for record in records:
            if not isinstance(record, dict):
                continue
            for key, value in record.items():
                field_values[str(key)].append(value)

        summaries: Dict[str, Any] = {}
        for field_name in sorted(field_values)[:max_fields]:
            summaries[field_name] = self._summarize_value_list(field_values[field_name])
        return summaries

    def _summarize_value_list(self, values: List[Any]) -> Dict[str, Any]:
        non_null = [value for value in values if value not in (None, "")]
        summary = {
            "non_null_count": len(non_null),
            "null_count": len(values) - len(non_null),
        }
        if not non_null:
            summary["type"] = "empty"
            return summary

        numeric_values = [self._coerce_float(value) for value in non_null]
        valid_numeric = [value for value in numeric_values if value is not None]
        if len(valid_numeric) >= max(3, int(len(non_null) * 0.7)):
            summary.update({
                "type": "numeric",
                "min": round(float(min(valid_numeric)), 4),
                "max": round(float(max(valid_numeric)), 4),
                "mean": round(float(np.mean(valid_numeric)), 4),
            })
            return summary

        normalized = [str(value).strip() for value in non_null if str(value).strip()]
        counts = Counter(normalized)
        summary.update({
            "type": "categorical" if len(counts) <= 20 else "string",
            "distinct_count": len(counts),
            "top_values": [
                {"value": label, "count": count}
                for label, count in counts.most_common(3)
            ],
        })
        return summary

    def _coerce_float(self, value: Any) -> Optional[float]:
        if value in (None, ""):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip().replace(",", "")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    # ── 辅助方法 ────────────────────────────────────────────────────────────

    def _extract_data_description(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """从多个可能的 context 字段提取 data_description"""
        # 优先从 dataset_info.base_dataset.data_description
        di = context.get("dataset_info", {}) or {}
        if isinstance(di, dict):
            bd = di.get("base_dataset", {}) or {}
            if isinstance(bd, dict):
                dd = bd.get("data_description")
                if isinstance(dd, dict) and dd:
                    enriched = dict(dd)
                    for field in ("ee_id", "gee_id", "ee_ctor", "source", "data_path", "type"):
                        if bd.get(field) is not None and field not in enriched:
                            enriched[field] = bd.get(field)
                    return enriched

        # spatial_extent
        se = context.get("spatial_extent", {}) or {}
        if isinstance(se, dict):
            bd = se.get("base_dataset", {}) or {}
            if isinstance(bd, dict):
                dd = bd.get("data_description")
                if isinstance(dd, dict) and dd:
                    enriched = dict(dd)
                    for field in ("ee_id", "gee_id", "ee_ctor", "source", "data_path", "type"):
                        if bd.get(field) is not None and field not in enriched:
                            enriched[field] = bd.get(field)
                    return enriched

        # 直接的 data_description
        dd = context.get("data_description")
        if isinstance(dd, dict) and dd:
            return dd

        return {}

    def _enrich_description(
        self,
        original: Dict[str, Any],
        portrait: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        将画像分析结果增量合并到原始 data_description 中，
        生成 enriched_data_description 供所有下游 agent 使用。
        """
        enriched = dict(original)

        # 新增：实测分析字段
        analysis = portrait.get("analysis", {})
        enriched["profiled_statistics"] = {
            "anomaly_years": analysis.get("anomaly_years", {}),
            "trend_by_class": analysis.get("trend_by_class", {}),
            "global_class_distribution": analysis.get("global_class_distribution", {}),
            "interannual_cv": analysis.get("interannual_cv", {}),
            "spatial_hotspot": analysis.get("spatial_hotspot", {}),
            "geometry_summary": analysis.get("geometry_summary", {}),
            "attribute_summary": analysis.get("attribute_summary", {}),
            "timeseries_summary": analysis.get("timeseries_summary", {}),
            "trend_of_mean": analysis.get("trend_of_mean", {}),
            "anomaly_years_by_mean": analysis.get("anomaly_years_by_mean", []),
            "profile_status": analysis.get("profile_status"),
            "remote_source": analysis.get("remote_source"),
            "remote_dataset_id": analysis.get("remote_dataset_id"),
            "stac_metadata_summary": analysis.get("stac_metadata_summary", {}),
            "osm_metadata_summary": analysis.get("osm_metadata_summary", {}),
        }

        # 向量属性实测值与面积单位反推结果并入 vector_content，覆盖任务描述里遗留的
        # "单位未在 runtime probe 中声明"。这类不确定性此前会一路穿到 idea 成熟化，
        # 质询提出"AREA 单位是什么"，通道判定判成需要数据集，检索 catalog 必然空手。
        attribute_observations = analysis.get("vector_attribute_observations") or {}
        unit_checks = analysis.get("vector_attribute_unit_checks") or {}
        if attribute_observations or unit_checks:
            vector_content = dict(enriched.get("vector_content") or {})
            if attribute_observations:
                vector_content["observed_attribute_values"] = attribute_observations
            if unit_checks:
                vector_content["attribute_unit_checks"] = unit_checks
                # 单位已确定的属性，把结论写进语义提示，替换掉"需要核实"的说法。
                hints = dict(vector_content.get("property_semantics_hint") or {})
                for field_name, check in unit_checks.items():
                    unit = str(check.get("inferred_unit") or "")
                    if unit and unit != "unresolved":
                        hints[field_name] = (
                            f"Numeric area attribute; unit measured as {unit} by comparing the "
                            f"declared value against geometry().area() "
                            f"(ratio {check.get('declared_to_m2_ratio')})."
                        )
                if hints:
                    vector_content["property_semantics_hint"] = hints
            enriched["vector_content"] = vector_content

        # 新增：数据集已经存在的特征。下游不得因为本地样本字段未展开而否定这些特征。
        enriched["dataset_features"] = portrait.get("dataset_features", [])
        enriched["dataset_feature_contract"] = {
            "dataset_features_are_authoritative": True,
        }

        # 新增：问题轴与质量标记，供 idea 评估前过滤
        enriched["candidate_question_axes"] = portrait.get("candidate_question_axes", [])
        enriched["profile_quality"] = portrait.get("profile_quality", {})
        enriched["resolved_profile_context"] = portrait.get("resolved_profile_context", {})

        # 新增：数据驱动的研究钩子
        enriched["data_driven_research_hooks"] = portrait.get("suggested_research_hooks", [])

        # 保留画像文本供 prompt 注入
        enriched["portrait_summary_text"] = portrait.get("portrait_text", "")

        return enriched

"""多源数据集查询与下载客户端。

为 datasets.json 中的非 GEE 源提供统一的查询/探测/下载入口：
- NASA_CMR：STAC API 查询
- PLANETARY_COMPUTER：STAC API 查询
- EARTH_SEARCH：STAC API 查询
- OSM_TAGINFO：OSM Taginfo API 查询
- OSM_OVERPASS：OSM Overpass API 查询

只接入"能不下载数据就确认数据集存在"的源。纯静态下载镜像
（Geofabrik PBF、WorldClim zip）不在此列——验证它们等于起一次
GB 级下载，代价与收益不成比例，这类源已从 datasets.json 移除。

所有需要密钥的源均从环境变量读取；密钥缺失时只影响需要认证的操作，
元数据查询尽量保持可匿名访问。
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urljoin

import aiohttp

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=30)
USER_AGENT = "DataGenHyp-dataset-source-clients/1.0"

# ---------------------------------------------------------------------------
# 环境变量名（同步在 .env 中声明，方便用户补充密钥）
# ---------------------------------------------------------------------------
ENV_NASA_EDL_TOKEN = "NASA_EDL_TOKEN"
ENV_PLANETARY_COMPUTER_SUBSCRIPTION_KEY = "PLANETARY_COMPUTER_SUBSCRIPTION_KEY"
ENV_DATASET_DOWNLOAD_DIR = "DATASET_DOWNLOAD_DIR"

# ---------------------------------------------------------------------------
# 通用 HTTP 工具
# ---------------------------------------------------------------------------


def _default_headers() -> Dict[str, str]:
    return {"User-Agent": USER_AGENT}


async def probe_http_head(url: str, timeout: int = 10) -> Dict[str, Any]:
    """对 URL 发 HEAD 请求，返回可达性结论，不下载内容。"""
    import time as time_module
    from datetime import datetime

    start = time_module.time()
    result: Dict[str, Any] = {
        "status": "unknown",
        "checked_at": datetime.now().isoformat(),
        "reason": "",
        "latency_ms": 0,
    }

    if not url or not url.startswith(("http://", "https://")):
        result["status"] = "fail"
        result["reason"] = "invalid or missing URL"
        return result

    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout),
            headers=_default_headers(),
        ) as session:
            async with session.head(url, allow_redirects=True) as resp:
                result["latency_ms"] = int((time_module.time() - start) * 1000)
                if resp.status < 400:
                    result["status"] = "pass"
                    result["reason"] = f"HTTP {resp.status}"
                else:
                    result["status"] = "fail"
                    result["reason"] = f"HTTP {resp.status}"
    except asyncio.TimeoutError:
        result["status"] = "fail"
        result["reason"] = "timeout"
    except Exception as exc:  # noqa: BLE001
        result["status"] = "fail"
        result["reason"] = str(exc)

    return result


async def probe_stac_collection(stac_url: str, timeout: int = 15) -> Dict[str, Any]:
    """GET STAC collection URL，验证其为合法 STAC collection 并解析基础元数据。"""
    import time as time_module
    from datetime import datetime

    start = time_module.time()
    result: Dict[str, Any] = {
        "status": "unknown",
        "checked_at": datetime.now().isoformat(),
        "reason": "",
        "latency_ms": 0,
        "collection": None,
    }

    if not stac_url or not stac_url.startswith(("http://", "https://")):
        result["status"] = "fail"
        result["reason"] = "invalid or missing STAC URL"
        return result

    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout),
            headers={**_default_headers(), "Accept": "application/json"},
        ) as session:
            async with session.get(stac_url, allow_redirects=True) as resp:
                result["latency_ms"] = int((time_module.time() - start) * 1000)
                if resp.status >= 400:
                    result["status"] = "fail"
                    result["reason"] = f"HTTP {resp.status}"
                    return result

                try:
                    payload = await resp.json()
                except Exception as exc:  # noqa: BLE001
                    result["status"] = "fail"
                    result["reason"] = f"invalid JSON: {exc}"
                    return result

                if not isinstance(payload, dict):
                    result["status"] = "fail"
                    result["reason"] = "STAC payload is not an object"
                    return result

                stac_version = payload.get("stac_version") or payload.get("stac")
                if not stac_version:
                    result["status"] = "fail"
                    result["reason"] = "missing stac_version"
                    return result

                result["status"] = "pass"
                result["reason"] = f"STAC {stac_version}"
                result["collection"] = {
                    "id": payload.get("id"),
                    "title": payload.get("title"),
                    "description": payload.get("description"),
                    "license": payload.get("license"),
                    "extent": payload.get("extent"),
                }
    except asyncio.TimeoutError:
        result["status"] = "fail"
        result["reason"] = "timeout"
    except Exception as exc:  # noqa: BLE001
        result["status"] = "fail"
        result["reason"] = str(exc)

    return result


async def download_to_local(
    url: str,
    local_path: str,
    chunk_size: int = 8192,
    timeout: int = 300,
) -> Dict[str, Any]:
    """通用 HTTP 下载，保存到本地路径。"""
    from datetime import datetime

    result: Dict[str, Any] = {
        "status": "unknown",
        "checked_at": datetime.now().isoformat(),
        "reason": "",
        "local_path": local_path,
        "bytes": 0,
    }

    if not url or not url.startswith(("http://", "https://")):
        result["status"] = "fail"
        result["reason"] = "invalid or missing URL"
        return result

    target = Path(local_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout),
            headers=_default_headers(),
        ) as session:
            async with session.get(url, allow_redirects=True) as resp:
                if resp.status >= 400:
                    result["status"] = "fail"
                    result["reason"] = f"HTTP {resp.status}"
                    return result

                with open(target, "wb") as f:
                    async for chunk in resp.content.iter_chunked(chunk_size):
                        if chunk:
                            f.write(chunk)
                            result["bytes"] += len(chunk)

                result["status"] = "pass"
                result["reason"] = f"downloaded {result['bytes']} bytes"
    except Exception as exc:  # noqa: BLE001
        result["status"] = "fail"
        result["reason"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# 源级客户端
# ---------------------------------------------------------------------------


class SourceClientError(Exception):
    """数据集源客户端异常。"""


CMR_SEARCH_BASE = "https://cmr.earthdata.nasa.gov/search"
CMR_STAC_BASE = "https://cmr.earthdata.nasa.gov/stac"


async def resolve_nasa_cmr_stac_url(
    entry_id: str,
    session: Optional[aiohttp.ClientSession] = None,
) -> Dict[str, Any]:
    """按 CMR entry id 解析出该 collection 真正的 STAC endpoint。

    CMR 的 STAC 按 provider 分目录，`/stac/ALL/collections/{id}` 只在根
    catalog 层存在，取具体 collection 时恒 404——datasets.json 里抓来的
    doc_url 正是这个形式，直接拿去探测会把全部 collection 判成不可达。
    正确路径是先用 search API 查出 provider-id，再拼
    `/stac/{provider}/collections/{entry_id}`。

    CMR 的 `entry_id` 检索参数匹配的是 `<ShortName>_<VersionId>`（如
    `SENTINEL-1A_SLC_1`），而本仓库部分 CMR 资产 id 只落到 ShortName
    （如 `SENTINEL-1A_SLC`，即 collection 的 native-id），用 `entry_id=`
    检索恒 0 命中。因此 entry_id 检索为空时回退用 `short_name=` 检索，
    命中后按 `<ShortName>_<Version>` 还原 canonical entry id 再拼 STAC
    URL，并在结果里以 `resolved_entry_id` / `resolved_via` 标注。
    """
    result: Dict[str, Any] = {
        "entry_id": entry_id,
        "provider_id": None,
        "concept_id": None,
        "stac_url": None,
        "resolved_entry_id": None,
        "resolved_via": None,
        "error": None,
    }
    if not entry_id:
        result["error"] = "missing entry id"
        return result

    token = os.getenv(ENV_NASA_EDL_TOKEN, "").strip()
    headers: Dict[str, str] = {"Accept": "application/json", **_default_headers()}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    search_url = f"{CMR_SEARCH_BASE}/collections.umm_json"

    async def _search(params: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        async with session.get(search_url, params=params) as resp:
            resp.raise_for_status()
            payload = await resp.json()
        return payload.get("items") or []

    owns_session = session is None
    if owns_session:
        session = aiohttp.ClientSession(timeout=DEFAULT_TIMEOUT, headers=headers)
    try:
        items = await _search({"entry_id": entry_id})
        stac_entry_id = entry_id
        if items:
            result["resolved_via"] = "entry_id"
        else:
            # 回退：部分资产 id 只到 ShortName（native-id），CMR 的
            # entry_id 参数不匹配它，须按 short_name 检索并还原带版本的
            # canonical entry id（ShortName_VersionId）。
            items = await _search({"short_name": entry_id})
            if not items:
                result["error"] = "collection not found in CMR search"
                return result
            umm = items[0].get("umm") or {}
            short_name = str(umm.get("ShortName") or entry_id).strip()
            version = str(umm.get("Version") or "").strip()
            stac_entry_id = f"{short_name}_{version}" if version else short_name
            result["resolved_via"] = "short_name_fallback"
        result["resolved_entry_id"] = stac_entry_id
        meta = items[0].get("meta") or {}
        provider_id = str(meta.get("provider-id") or "").strip()
        result["provider_id"] = provider_id or None
        result["concept_id"] = meta.get("concept-id")
        if not provider_id:
            result["error"] = "CMR search returned no provider-id"
            return result
        result["stac_url"] = (
            f"{CMR_STAC_BASE}/{quote(provider_id)}/collections/{quote(stac_entry_id)}"
        )
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)
    finally:
        if owns_session:
            await session.close()

    return result


async def query_nasa_cmr(
    collection_id: str,
    bbox: Optional[List[float]] = None,
    datetime_range: Optional[str] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """查询 NASA CMR STAC 的 collection 或 items。

    Args:
        collection_id: CMR collection entry id（datasets.json 中的 id）
        bbox: [minx, miny, maxx, maxy]
        datetime_range: ISO 8601 范围，如 "2023-01-01T00:00:00Z/2023-12-31T23:59:59Z"
        limit: 最大返回 item 数
    """
    token = os.getenv(ENV_NASA_EDL_TOKEN, "").strip()
    headers: Dict[str, str] = {"Accept": "application/json", **_default_headers()}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    result: Dict[str, Any] = {
        "source": "NASA_CMR",
        "collection_id": collection_id,
        "collection_url": None,
        "items": [],
        "error": None,
    }

    try:
        async with aiohttp.ClientSession(
            timeout=DEFAULT_TIMEOUT, headers=headers
        ) as session:
            resolved = await resolve_nasa_cmr_stac_url(collection_id, session=session)
            collection_url = resolved.get("stac_url")
            result["collection_url"] = collection_url
            result["provider_id"] = resolved.get("provider_id")
            result["concept_id"] = resolved.get("concept_id")
            if not collection_url:
                result["error"] = resolved.get("error") or "unable to resolve STAC URL"
                return result

            async with session.get(collection_url) as resp:
                if resp.status == 404:
                    result["error"] = "collection not found"
                    return result
                resp.raise_for_status()
                collection = await resp.json()

            result["title"] = collection.get("title")
            result["description"] = collection.get("description")

            # 尝试搜索 items（可选）。items 链接有两种形态：
            # - search 端点（如 Planetary Computer）：.../search?collection=...
            # - CMR STAC 的 per-collection items：.../collections/{id}/items
            # 只认含 "search" 的链接会把 CMR 的 items 链接整条丢弃。
            items_links = [
                link.get("href", "")
                for link in collection.get("links", [])
                if link.get("rel") == "items" and link.get("href")
            ]
            search_links = [href for href in items_links if "search" in href] or items_links
            if search_links:
                search_url = search_links[0]
                params: Dict[str, Any] = {"limit": limit}
                if bbox:
                    params["bbox"] = ",".join(str(v) for v in bbox)
                if datetime_range:
                    params["datetime"] = datetime_range

                try:
                    async with session.get(search_url, params=params) as resp:
                        resp.raise_for_status()
                        payload = await resp.json()
                        result["items"] = payload.get("features", [])
                except Exception:  # noqa: BLE001
                    # items 搜索失败不影响 collection 本身的有效性
                    pass
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)

    return result


PLANETARY_COMPUTER_SAS_BASE = "https://planetarycomputer.microsoft.com/api/sas/v1"
_PC_SAS_CACHE = {}


async def get_planetary_computer_sas_token(
    collection_id: str,
    session=None,
):
    """获取指定 Planetary Computer Collection 的 SAS Token，具备内存缓存与自动过期刷新。"""
    import time as time_module
    from datetime import datetime

    normalized_id = str(collection_id or "").strip()
    if not normalized_id:
        return None

    now = time_module.time()
    cached = _PC_SAS_CACHE.get(normalized_id)
    if cached:
        token, expiry = cached
        if now < expiry - 300:
            return token

    subscription_key = os.getenv(ENV_PLANETARY_COMPUTER_SUBSCRIPTION_KEY, "").strip()
    headers = {"Accept": "application/json", **_default_headers()}
    if subscription_key:
        headers["Ocp-Apim-Subscription-Key"] = subscription_key

    url = f"{PLANETARY_COMPUTER_SAS_BASE}/token/{quote(normalized_id)}"
    owns_session = session is None
    if owns_session:
        session = aiohttp.ClientSession(timeout=DEFAULT_TIMEOUT, headers=headers)
    try:
        async with session.get(url, headers=headers) as resp:
            if resp.status != 200:
                logger.warning(
                    "Failed to fetch Planetary Computer SAS token for %s: HTTP %s",
                    normalized_id,
                    resp.status,
                )
                return None
            payload = await resp.json()
            token = payload.get("token")
            if not token:
                return None

            expiry_str = payload.get("msft:expiry")
            expiry_ts = now + 3000
            if expiry_str:
                try:
                    clean_expiry = expiry_str.rstrip("Z")
                    expiry_ts = datetime.fromisoformat(clean_expiry).timestamp()
                except Exception:
                    pass

            _PC_SAS_CACHE[normalized_id] = (token, expiry_ts)
            return token
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Error fetching Planetary Computer SAS token for %s: %s",
            normalized_id,
            exc,
        )
        return None
    finally:
        if owns_session:
            await session.close()


def sign_planetary_computer_url(url: str, token: str) -> str:
    """将 SAS Token 拼接到 Planetary Computer 资源 URL 中。"""
    if not url or not token:
        return url
    clean_token = token.lstrip("?&")
    delimiter = "&" if "?" in url else "?"
    return f"{url}{delimiter}{clean_token}"


def sign_planetary_computer_item(item: dict, token: str) -> dict:
    """将 SAS Token 拼接到 STAC item 内部的所有 asset href 中。"""
    if not token or not isinstance(item, dict):
        return item
    assets = item.get("assets")
    if not isinstance(assets, dict):
        return item
    for asset in assets.values():
        if not isinstance(asset, dict):
            continue
        href = asset.get("href")
        if isinstance(href, str) and href.startswith(("http://", "https://")):
            if "sig=" not in href:
                asset["href"] = sign_planetary_computer_url(href, token)
    return item


async def query_planetary_computer(
    collection_id: str,
    bbox: Optional[List[float]] = None,
    datetime_range: Optional[str] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """查询 Planetary Computer STAC。"""
    subscription_key = os.getenv(ENV_PLANETARY_COMPUTER_SUBSCRIPTION_KEY, "").strip()
    headers: Dict[str, str] = {"Accept": "application/json", **_default_headers()}
    if subscription_key:
        headers["Ocp-Apim-Subscription-Key"] = subscription_key

    base_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
    collection_url = f"{base_url}/collections/{collection_id}"
    result: Dict[str, Any] = {
        "source": "PLANETARY_COMPUTER",
        "collection_id": collection_id,
        "collection_url": collection_url,
        "items": [],
        "error": None,
    }

    try:
        async with aiohttp.ClientSession(
            timeout=DEFAULT_TIMEOUT, headers=headers
        ) as session:
            async with session.get(collection_url) as resp:
                if resp.status == 404:
                    result["error"] = "collection not found"
                    return result
                resp.raise_for_status()
                collection = await resp.json()

            result["title"] = collection.get("title")
            result["description"] = collection.get("description")

            search_url = f"{base_url}/search"
            params: Dict[str, Any] = {
                "collections": [collection_id],
                "limit": limit,
            }
            if bbox:
                params["bbox"] = bbox
            if datetime_range:
                params["datetime"] = datetime_range

            async with session.post(search_url, json=params) as resp:
                resp.raise_for_status()
                payload = await resp.json()
                items = payload.get("features", [])

            token = await get_planetary_computer_sas_token(collection_id, session=session)
            if token:
                for item in items:
                    sign_planetary_computer_item(item, token)

            result["items"] = items
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)

    return result


async def query_earth_search(
    collection_id: str,
    bbox: Optional[List[float]] = None,
    datetime_range: Optional[str] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """查询 Earth Search STAC（Element84）。"""
    base_url = "https://earth-search.aws.element84.com/v1"
    collection_url = f"{base_url}/collections/{collection_id}"
    result: Dict[str, Any] = {
        "source": "EARTH_SEARCH",
        "collection_id": collection_id,
        "collection_url": collection_url,
        "items": [],
        "error": None,
    }

    try:
        async with aiohttp.ClientSession(
            timeout=DEFAULT_TIMEOUT, headers=_default_headers()
        ) as session:
            async with session.get(collection_url) as resp:
                if resp.status == 404:
                    result["error"] = "collection not found"
                    return result
                resp.raise_for_status()
                collection = await resp.json()

            result["title"] = collection.get("title")
            result["description"] = collection.get("description")

            search_url = f"{base_url}/search"
            params: Dict[str, Any] = {
                "collections": [collection_id],
                "limit": limit,
            }
            if bbox:
                params["bbox"] = bbox
            if datetime_range:
                params["datetime"] = datetime_range

            async with session.post(search_url, json=params) as resp:
                resp.raise_for_status()
                payload = await resp.json()
                result["items"] = payload.get("features", [])
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)

    return result


async def query_osm_taginfo(tag_key: str) -> Dict[str, Any]:
    """查询 OSM taginfo API 获取 tag 元数据。

    tag key 必须转义后再拼进 URL：catalog 里存在 `*`、`*:traffic` 这类含
    保留字符的 key，直接插值会构造出畸形 URL 并以连接错误的形式失败，
    看起来像网络问题，实则是本地拼串问题。
    """
    # datasets.json 中的 id 形如 "highway" 或 "*:2"
    encoded_key = quote(str(tag_key or ""), safe="")
    api_url = (
        "https://taginfo.openstreetmap.org/api/4/key/values"
        f"?key={encoded_key}&page=1&rp=1"
    )
    wiki_url = f"https://wiki.openstreetmap.org/wiki/Key:{encoded_key}"

    result: Dict[str, Any] = {
        "source": "OSM_TAGINFO",
        "tag_key": tag_key,
        "wiki_url": wiki_url,
        "api_url": api_url,
        "exists": False,
        "error": None,
    }

    try:
        async with aiohttp.ClientSession(
            timeout=DEFAULT_TIMEOUT, headers=_default_headers()
        ) as session:
            async with session.get(api_url) as resp:
                resp.raise_for_status()
                payload = await resp.json()
                result["exists"] = bool(payload.get("data"))
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)

    return result


async def query_osm_overpass(
    overpass_ql: str,
    endpoint: str = "https://overpass-api.de/api/interpreter",
    timeout: int = 60,
) -> Dict[str, Any]:
    """通过 OSM Overpass API 执行查询并返回结果。

    Args:
        overpass_ql: Overpass QL 查询字符串，例如
            ``node["amenity"="hospital"](around:1000,51.5,-0.1);out;``
        endpoint: Overpass API endpoint，可替换为其他公共/私有实例
        timeout: 请求超时（秒）
    """
    result: Dict[str, Any] = {
        "source": "OSM_OVERPASS",
        "endpoint": endpoint,
        "elements": [],
        "error": None,
    }

    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout),
            headers={**_default_headers(), "Content-Type": "application/x-www-form-urlencoded"},
        ) as session:
            async with session.post(endpoint, data={"data": overpass_ql}) as resp:
                resp.raise_for_status()
                payload = await resp.json()
                result["elements"] = payload.get("elements", [])
                result["count"] = len(result["elements"])
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# 统一调度入口
# ---------------------------------------------------------------------------


async def query_dataset(
    record: Dict[str, Any],
    bbox: Optional[List[float]] = None,
    datetime_range: Optional[str] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """根据 datasets.json 记录分发到对应源客户端。"""
    source = str(record.get("source") or "").strip().upper()
    dataset_id = str(record.get("id") or "").strip()

    if source == "PANGAEA":
        from .pangaea_client import query_pangaea
        return await query_pangaea(dataset_id)

    if source in {"GEE_OFFICIAL", "GEE_COMMUNITY"}:
        return {
            "source": source,
            "dataset_id": dataset_id,
            "note": "use GEE/earthengine-api directly",
        }

    if source == "NASA_CMR":
        return await query_nasa_cmr(dataset_id, bbox=bbox, datetime_range=datetime_range, limit=limit)
    if source == "PLANETARY_COMPUTER":
        return await query_planetary_computer(
            dataset_id, bbox=bbox, datetime_range=datetime_range, limit=limit
        )
    if source == "EARTH_SEARCH":
        return await query_earth_search(
            dataset_id, bbox=bbox, datetime_range=datetime_range, limit=limit
        )
    if source == "OSM_TAGINFO":
        return await query_osm_taginfo(dataset_id)
    if source == "OSM_OVERPASS":
        overpass_ql = str(record.get("overpass_ql") or "").strip()
        if not overpass_ql:
            return {"source": source, "dataset_id": dataset_id, "error": "OSM_OVERPASS requires overpass_ql"}
        endpoint = str(record.get("endpoint") or "").strip()
        if endpoint:
            return await query_osm_overpass(overpass_ql, endpoint=endpoint)
        return await query_osm_overpass(overpass_ql)

    return {
        "source": source,
        "dataset_id": dataset_id,
        "error": f"unsupported source: {source}",
    }


# 已知使用 STAC API 的数据源
_STAC_API_SOURCES = frozenset({"NASA_CMR", "PLANETARY_COMPUTER", "EARTH_SEARCH"})


def _looks_like_stac_url(url: str) -> bool:
    """判断 URL 是否指向 STAC collection/item endpoint。"""
    url_lower = url.lower()
    # NASA CMR、Planetary Computer、Earth Search 等 STAC endpoint 的常见模式
    if "/stac/" in url_lower or "/api/stac/" in url_lower:
        return True
    if "/collections/" in url_lower and (
        "earth-search.aws.element84.com" in url_lower
        or "planetarycomputer.microsoft.com" in url_lower
        or "cmr.earthdata.nasa.gov" in url_lower
    ):
        return True
    return False


async def probe_dataset(record: Dict[str, Any]) -> Dict[str, Any]:
    """轻量探测 datasets.json 记录的可达性，不下载内容。"""
    from datetime import datetime

    source = str(record.get("source") or "").strip().upper()
    dataset_id = str(record.get("id") or "").strip()
    doc_url = str(record.get("doc_url") or "").strip()
    stac_url = str(record.get("stac_url") or "").strip()

    if source == "PANGAEA":
        from .pangaea_client import query_pangaea
        try:
            metadata = await query_pangaea(dataset_id)
            data_url = metadata["record"]["data_url"]
            if not data_url:
                return {"source": source, "dataset_id": dataset_id, "status": "unknown",
                        "reason": "PANGAEA metadata resolved but no distribution was published",
                        "checked_at": datetime.now().isoformat()}
            probe = await probe_http_head(data_url)
            return {**probe, "source": source, "dataset_id": dataset_id, "url": data_url,
                    "reason": "PANGAEA distribution: " + str(probe.get("reason", ""))}
        except Exception as exc:
            return {"source": source, "dataset_id": dataset_id, "status": "fail",
                    "reason": str(exc), "checked_at": datetime.now().isoformat()}

    if source in {"GEE_OFFICIAL", "GEE_COMMUNITY"}:
        return {
            "source": source,
            "dataset_id": dataset_id,
            "status": "advisory",
            "reason": "use GEE probe in resourcescheck_agent",
            "checked_at": datetime.now().isoformat(),
        }

    # NASA_CMR 的 doc_url 抓的是 /stac/ALL/collections/{id}，取具体 collection 时恒 404，
    # 必须先查 provider 才能拼出可用的 STAC endpoint。
    if source == "NASA_CMR":
        resolved = await resolve_nasa_cmr_stac_url(dataset_id)
        probe_url = resolved.get("stac_url")
        if not probe_url:
            return {
                "source": source,
                "dataset_id": dataset_id,
                "status": "fail",
                "reason": resolved.get("error") or "unable to resolve CMR STAC URL",
                "checked_at": datetime.now().isoformat(),
            }
        result = await probe_stac_collection(probe_url)
        return {
            "source": source,
            "dataset_id": dataset_id,
            "status": result["status"],
            "reason": result["reason"],
            "latency_ms": result["latency_ms"],
            "url": probe_url,
            "provider_id": resolved.get("provider_id"),
            "checked_at": result.get("checked_at", datetime.now().isoformat()),
        }

    # PLANETARY_COMPUTER / EARTH_SEARCH 的 doc_url 本身就是 STAC collection endpoint
    probe_url = stac_url or doc_url
    if probe_url and (source in _STAC_API_SOURCES or _looks_like_stac_url(probe_url)):
        result = await probe_stac_collection(probe_url)
        return {
            "source": source,
            "dataset_id": dataset_id,
            "status": result["status"],
            "reason": result["reason"],
            "latency_ms": result["latency_ms"],
            "url": probe_url,
            "checked_at": result.get("checked_at", datetime.now().isoformat()),
        }

    # OSM_TAGINFO 的 doc_url 指向 wiki 页面，而大量真实存在的 tag 并没有 wiki 条目
    # （实测 4066、AHI_refnum 等均 404）。tag 是否存在应由 taginfo API 回答。
    if source == "OSM_TAGINFO":
        taginfo = await query_osm_taginfo(dataset_id)
        if taginfo.get("error"):
            return {
                "source": source,
                "dataset_id": dataset_id,
                "status": "fail",
                "reason": str(taginfo["error"]),
                "url": taginfo.get("api_url"),
                "checked_at": datetime.now().isoformat(),
            }
        return {
            "source": source,
            "dataset_id": dataset_id,
            "status": "pass" if taginfo.get("exists") else "fail",
            "reason": (
                "tag present in taginfo"
                if taginfo.get("exists")
                else "tag absent from taginfo"
            ),
            "url": taginfo.get("api_url"),
            "checked_at": datetime.now().isoformat(),
        }

    if source == "OSM_OVERPASS":
        overpass_ql = str(record.get("overpass_ql") or "").strip()
        if not overpass_ql:
            return {
                "source": source,
                "dataset_id": dataset_id,
                "status": "fail",
                "reason": "OSM_OVERPASS requires overpass_ql",
                "checked_at": datetime.now().isoformat(),
            }
        endpoint = str(record.get("endpoint") or "").strip()
        overpass = (
            await query_osm_overpass(overpass_ql, endpoint=endpoint)
            if endpoint else await query_osm_overpass(overpass_ql)
        )
        if overpass.get("error"):
            return {
                "source": source,
                "dataset_id": dataset_id,
                "status": "fail",
                "reason": str(overpass["error"]),
                "url": overpass.get("endpoint"),
                "checked_at": datetime.now().isoformat(),
            }
        return {
            "source": source,
            "dataset_id": dataset_id,
            "status": "pass",
            "reason": f"Overpass query returned {overpass.get('count', 0)} elements",
            "url": overpass.get("endpoint"),
            "checked_at": datetime.now().isoformat(),
        }

    if doc_url:
        result = await probe_http_head(doc_url)
        return {
            "source": source,
            "dataset_id": dataset_id,
            "status": result["status"],
            "reason": result["reason"],
            "latency_ms": result["latency_ms"],
            "url": doc_url,
            "checked_at": result.get("checked_at", datetime.now().isoformat()),
        }

    return {
        "source": source,
        "dataset_id": dataset_id,
        "status": "unknown",
        "reason": "no doc_url or stac_url",
        "checked_at": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# 本地下载目录工具
# ---------------------------------------------------------------------------


def get_dataset_download_dir() -> Path:
    """返回环境变量 DATASET_DOWNLOAD_DIR 指定的本地下载目录，默认项目根目录 datasets/。"""
    raw = os.getenv(ENV_DATASET_DOWNLOAD_DIR, "")
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(__file__).resolve().parent.parent.parent.parent / "datasets"


def local_dataset_path(record: Dict[str, Any], filename: Optional[str] = None) -> Path:
    """根据 datasets.json 记录推导本地存储路径。

    约定：datasets/<source>/<id>/<filename>
    """
    source = str(record.get("source") or "UNKNOWN").strip().upper()
    dataset_id = str(record.get("id") or "").strip().replace("/", "_")
    if not filename:
        # 尝试从 doc_url 推断
        doc_url = str(record.get("doc_url") or "").strip()
        filename = Path(doc_url).name if doc_url else f"{dataset_id}.dat"
    return get_dataset_download_dir() / source / dataset_id / filename

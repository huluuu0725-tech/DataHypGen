"""候选数据集详情富化工具。

顶层 catalog（datasets.json）逐条只有 id/title/description 等平铺字段，
没有波段清单、分辨率、时间步长；本工具在候选数据集
（CandidateDatasetScout 产出）上补全这些详情：

- 官方数据集：ee_catalog_datasets/{id 规范化}.json（STAC），确定性 join，
  抽 description / 波段 / 分辨率 / 时间步长与范围 / 空间范围 / 关键词
- 社区数据集：community_asset_doc_index.json 查 id -> md 文件名，
  从 community_project_docs/*.md 抽描述与 EE 示例代码
- 未命中（含 OSM）：返回原 entry，不加 detail（fail-open，不阻塞）
"""

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from .dataset_resolution_semantics import normalize_temporal_resolution

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_OFFICIAL_DIR = DATA_DIR / "ee_catalog_datasets"
DEFAULT_COMMUNITY_DOCS_DIR = DATA_DIR / "community_project_docs"
DEFAULT_COMMUNITY_INDEX = DATA_DIR / "community_asset_doc_index.json"
DEFAULT_DESCRIPTION_INDEX = Path(__file__).resolve().parents[3] / "artifacts" / "dataset_description_index.json"
DEFAULT_DETAIL_DB = DATA_DIR / "datasets.db"

_JS_SNIPPET_RE = re.compile(r"```(?:js|javascript)\s*\n(.*?)```", re.DOTALL)
_SAMPLE_CODE_RE = re.compile(r"Sample Code:\s*(https?://\S+)", re.IGNORECASE)
_IMAGE_MD_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_YEAR_RANGE_RE = re.compile(
    r"^\s*(19\d{2}|20\d{2})\s*(?:to|[-–])\s*(19\d{2}|20\d{2})\s*$",
    re.IGNORECASE,
)
_CADENCE_WORDS = {
    "hourly": "hour",
    "daily": "day",
    "monthly": "month",
    "annual": "year",
    "annually": "year",
    "yearly": "year",
}


class DatasetDetailEnricher:
    """按候选 entry 的 id 富化详情；所有失败路径 fail-open。"""

    _community_index_cache: Optional[Dict[str, str]] = None
    _description_index_cache: Optional[Dict[str, Dict[str, Any]]] = None
    _detail_db_conn: Optional[sqlite3.Connection] = None

    def __init__(
        self,
        official_dir: Optional[Path] = None,
        community_docs_dir: Optional[Path] = None,
        community_index_path: Optional[Path] = None,
        description_index_path: Optional[Path] = None,
        detail_db_path: Optional[Path] = None,
        max_description_chars: int = 1200,
        max_snippets: int = 2,
        use_sqlite: bool = True,
    ):
        self.official_dir = Path(official_dir or DEFAULT_OFFICIAL_DIR)
        self.community_docs_dir = Path(community_docs_dir or DEFAULT_COMMUNITY_DOCS_DIR)
        self.community_index_path = Path(community_index_path or DEFAULT_COMMUNITY_INDEX)
        self.description_index_path = Path(description_index_path or DEFAULT_DESCRIPTION_INDEX)
        self.detail_db_path = Path(detail_db_path or DEFAULT_DETAIL_DB)
        self.max_description_chars = max_description_chars
        self.max_snippets = max_snippets
        self.use_sqlite = use_sqlite
        # 自定义路径的实例不共享类级缓存，避免测试与生产互相污染
        self._use_class_cache = community_index_path is None
        self._use_description_class_cache = description_index_path is None
        self._use_detail_db_cache = detail_db_path is None

        # 初始化 SQLite 连接（如果启用且数据库存在）
        if self.use_sqlite and self.detail_db_path.exists():
            try:
                if self._use_detail_db_cache and type(self)._detail_db_conn is not None:
                    self._db_conn = type(self)._detail_db_conn
                else:
                    self._db_conn = sqlite3.connect(str(self.detail_db_path), check_same_thread=False)
                    if self._use_detail_db_cache:
                        type(self)._detail_db_conn = self._db_conn
                logger.info("Dataset detail database loaded: %s", self.detail_db_path)
            except Exception as exc:
                logger.warning("Failed to connect to detail database %s: %s (fallback to file mode)",
                             self.detail_db_path, exc)
                self._db_conn = None
        else:
            self._db_conn = None
            if self.use_sqlite and not self.detail_db_path.exists():
                logger.warning("Dataset detail database not found at %s (fallback to file mode)", self.detail_db_path)

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def enrich(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """返回补充了 detail 字段的 entry 副本；未命中或异常时原样返回。"""
        if not isinstance(entry, dict):
            return entry
        asset_id = str(entry.get("id") or "").strip()
        if not asset_id:
            return entry
        try:
            detail = self._official_detail(asset_id) or self._community_detail(asset_id)
            temporal_detail = self._description_index_temporal_detail(asset_id)
            if temporal_detail:
                detail = dict(detail or {})
                index_provenance = temporal_detail.get("provenance") or {}
                adopted_provenance: Dict[str, str] = {}
                for field in ("temporal_extent", "cadence"):
                    # 官方 STAC detail 是权威来源，描述索引只填空、不覆盖：索引的
                    # cadence 由频率词正则推出，interval 恒为 1，会把官方的 8-day
                    # 打成 1-day、3-hour 打成 1-hour，进而让 ResourceCheck 的
                    # temporal_resolution 门禁静默放行本不满足的 requirement。
                    if temporal_detail.get(field) and not detail.get(field):
                        detail[field] = temporal_detail[field]
                        if index_provenance.get(field):
                            adopted_provenance[field] = index_provenance[field]
                if adopted_provenance:
                    detail["temporal_provenance"] = adopted_provenance
                    detail.setdefault("detail_source", "dataset_description_index")
        except Exception as exc:  # fail-open：富化失败不阻塞候选
            logger.warning("Detail enrichment failed for %s: %s", asset_id, exc)
            detail = None
        if not detail:
            return entry
        enriched = dict(entry)
        enriched["detail"] = detail
        return enriched

    def enrich_batch(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.enrich(entry) for entry in entries]

    # ------------------------------------------------------------------
    # 官方 STAC
    # ------------------------------------------------------------------
    @staticmethod
    def official_filename(asset_id: str) -> str:
        return asset_id.replace("/", "_").replace(".", "_") + ".json"

    def _official_detail(self, asset_id: str) -> Optional[Dict[str, Any]]:
        # 优先使用 SQLite
        if self._db_conn is not None:
            try:
                cursor = self._db_conn.cursor()
                cursor.execute("SELECT detail_json FROM datasets WHERE id = ?", (asset_id,))
                row = cursor.fetchone()
                # detail_json 只有 877 行有值，其余行命中主键但取到 NULL，
                # 必须判空后再解析，否则会退化成每次都异常 + 文件回退。
                if row and row[0]:
                    return json.loads(row[0])
            except Exception as exc:
                logger.warning("SQLite query failed for %s: %s (fallback to file)", asset_id, exc)

        # 降级到文件读取模式
        stac_path = self.official_dir / self.official_filename(asset_id)
        if not stac_path.is_file():
            return None
        try:
            stac = json.loads(stac_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load official STAC %s: %s", stac_path, exc)
            return None

        # official_filename 把 "/" 和 "." 都折成 "_"，是有损映射：catalog 里
        # USGS/3DEP/1m（官方 1m DEM）与 USGS/3DEP_1m（社区 1 弧秒 DEM，约 30m）
        # 会落到同一个文件名，后者曾借走前者的 bands/gsd/时间范围。文件命中后必须
        # 校验文件内的真实 id，不一致就当作未命中。
        stac_id = str(stac.get("id") or "").strip()
        if stac_id and stac_id != asset_id:
            logger.debug(
                "Official STAC filename collision: %s resolves to %s, skipping",
                asset_id, stac_id,
            )
            return None

        summaries = stac.get("summaries") or {}
        bands = []
        for band in (summaries.get("eo:bands") or [])[:24]:
            if isinstance(band, dict):
                bands.append({
                    "name": band.get("name"),
                    "description": band.get("description"),
                    "gsd": band.get("gsd"),
                })
        known_band_names = {
            str(band.get("name") or "").strip()
            for band in bands
            if isinstance(band, dict)
        }
        for name, summary in summaries.items():
            if len(bands) >= 24:
                break
            if name in known_band_names or not isinstance(summary, dict):
                continue
            if not {"minimum", "maximum"}.intersection(summary):
                continue
            bands.append({"name": name, "description": None, "gsd": None})
        extent = stac.get("extent") or {}
        temporal = ((extent.get("temporal") or {}).get("interval") or [[None, None]])[0]
        spatial = ((extent.get("spatial") or {}).get("bbox") or [None])[0]

        description = str(stac.get("description") or "").strip()
        detail = {
            "detail_source": "official_stac",
            "description": description[: self.max_description_chars],
            "bands": bands,
            "gsd": summaries.get("gsd"),
            "cadence": stac.get("gee:interval"),
            "temporal_extent": temporal,
            # 键名必须是 spatial_extent：下游三处 detail allowlist
            # （orchestration_agent 的审计与 StepPlan 裁剪、idea_generation 的
            # compact_detail、idea_maturation 的 evidence 视图）取的都是这个名字。
            # 此前这里叫 spatial_bbox，于是 STAC 的 bbox 在每个消费点都被静默丢掉。
            # 值沿用 STAC 原始的 [minx, miny, maxx, maxy]，与 Task.spatial_extent
            # 的 {type, coordinates} 是互不相干的两套命名空间——detail 只是喂给
            # prompt 的事实，不参与 get_spatial_center 那条几何计算路径。
            "spatial_extent": spatial,
            "keywords": stac.get("keywords") or [],
            "temporal_provenance": {
                "temporal_extent": "official_stac.extent.temporal.interval",
                **(
                    {"cadence": "official_stac.gee:interval"}
                    if stac.get("gee:interval")
                    else {}
                ),
            },
        }
        return detail

    def _load_description_index(self) -> Dict[str, Dict[str, Any]]:
        if (
            self._use_description_class_cache
            and type(self)._description_index_cache is not None
        ):
            return type(self)._description_index_cache
        index: Dict[str, Dict[str, Any]] = {}
        try:
            payload = json.loads(self.description_index_path.read_text(encoding="utf-8"))
            datasets = payload.get("datasets") if isinstance(payload, dict) else None
            if isinstance(datasets, dict):
                index = {
                    str(key): value
                    for key, value in datasets.items()
                    if isinstance(value, dict)
                }
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Failed to load dataset description index %s: %s",
                self.description_index_path,
                exc,
            )
        if self._use_description_class_cache:
            type(self)._description_index_cache = index
        return index

    @staticmethod
    def _temporal_extent_from_text(value: Any) -> List[str]:
        text = " ".join(str(value or "").split())
        if not text:
            return []
        if "/" in text:
            start, end = [part.strip() for part in text.split("/", 1)]
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", start) and (
                re.fullmatch(r"\d{4}-\d{2}-\d{2}", end)
                or end.casefold() == "present"
            ):
                return [start, end]
        match = _YEAR_RANGE_RE.fullmatch(text)
        if match:
            return [f"{match.group(1)}-01-01", f"{match.group(2)}-12-31"]
        return []

    @staticmethod
    def _explicit_product_cadence(value: Any, *, title: bool = False) -> Dict[str, Any]:
        text = " ".join(str(value or "").split()).casefold()
        if not text:
            return {}
        if title:
            matches = re.findall(
                r"\b(hourly|daily|monthly|annual|annually|yearly)\b", text
            )
        else:
            matches = re.findall(
                r"\b(?:dataset|product|collection|we)\b.{0,100}?"
                r"\b(hourly|daily|monthly|annual|annually|yearly)\b.{0,60}?"
                r"\b(?:maps?|grids?|classification|composites?|data|anomal(?:y|ies))\b",
                text,
            )
        if len(set(matches)) != 1:
            return {}
        return {"unit": _CADENCE_WORDS[matches[0]], "interval": 1}

    def _description_index_temporal_detail(self, asset_id: str) -> Dict[str, Any]:
        record = self._load_description_index().get(asset_id)
        if not isinstance(record, dict):
            return {}
        metadata = record.get("temporal_metadata") or {}
        if isinstance(metadata, dict) and (
            metadata.get("temporal_extent") or metadata.get("cadence")
        ):
            return {
                "temporal_extent": list(metadata.get("temporal_extent") or []),
                "cadence": dict(metadata.get("cadence") or {}),
                "provenance": dict(metadata.get("provenance") or {}),
            }

        data_description = record.get("data_description") or {}
        official_stac = record.get("official_stac") or {}
        temporal_extent = self._temporal_extent_from_text(
            data_description.get("temporal_coverage")
        )
        provenance: Dict[str, str] = {}
        if temporal_extent:
            provenance["temporal_extent"] = "data_description.temporal_coverage"

        # 索引里 gee:interval 落在 official_stac.cadence（见构建脚本的字段映射）；
        # "official_stac.gee:interval" 只是 provenance 说明，不是产物字段路径。
        cadence = normalize_temporal_resolution(official_stac.get("cadence"))
        if cadence:
            provenance["cadence"] = "official_stac.gee:interval"
        else:
            for value, source, is_title in (
                (official_stac.get("title"), "official_stac.title", True),
                (official_stac.get("description"), "official_stac.description", False),
                (record.get("title"), "data_description.dataset_name", True),
                (data_description.get("description"), "data_description.description", False),
            ):
                cadence = self._explicit_product_cadence(value, title=is_title)
                if cadence:
                    provenance["cadence"] = source
                    break
        if data_description.get("temporal_semantics") == "static_snapshot":
            cadence = {}
            provenance.pop("cadence", None)
        return {
            "temporal_extent": temporal_extent,
            "cadence": cadence,
            "provenance": provenance,
        }

    # ------------------------------------------------------------------
    # 社区 md
    # ------------------------------------------------------------------
    def _load_community_index(self) -> Dict[str, str]:
        if self._use_class_cache and type(self)._community_index_cache is not None:
            return type(self)._community_index_cache
        index: Dict[str, str] = {}
        try:
            payload = json.loads(self.community_index_path.read_text(encoding="utf-8"))
            raw_index = payload.get("index") if isinstance(payload, dict) else None
            if isinstance(raw_index, dict):
                index = {str(k): str(v) for k, v in raw_index.items()}
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Failed to load community asset doc index %s: %s",
                self.community_index_path, exc,
            )
        if self._use_class_cache:
            type(self)._community_index_cache = index
        return index

    def _community_detail(self, asset_id: str) -> Optional[Dict[str, Any]]:
        doc_name = self._load_community_index().get(asset_id)
        if not doc_name:
            return None
        doc_path = self.community_docs_dir / doc_name
        try:
            text = doc_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Failed to read community doc %s: %s", doc_path, exc)
            return None

        description = self._extract_doc_description(text)
        snippets = [
            snippet.strip()
            for snippet in _JS_SNIPPET_RE.findall(text)[: self.max_snippets]
            if snippet.strip()
        ]
        sample_match = _SAMPLE_CODE_RE.search(text)
        return {
            "detail_source": "community_doc",
            "doc": doc_name,
            "description": description[: self.max_description_chars],
            "ee_snippets": snippets,
            "sample_code": sample_match.group(1) if sample_match else None,
        }

    @staticmethod
    def _extract_doc_description(text: str) -> str:
        """取首个标题后的正文段落（到下一个标题/代码块为止），剔除图片。"""
        lines: List[str] = []
        started = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                if started:
                    break
                started = True
                continue
            if stripped.startswith("```"):
                break
            lines.append(line)
        description = "\n".join(lines).strip()
        return _IMAGE_MD_RE.sub("", description).strip()

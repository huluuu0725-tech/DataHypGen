"""Shared experiment resource checks, independent of planning contracts and agents."""

import asyncio
import copy
import json
import re
from pathlib import Path

from datahypgen.mas.tools.capability_details import CapabilityDetailStore
from datahypgen.mas.tools.gee_auth import initialize_gee_from_service_account, set_gee_deadline
from datahypgen.mas.tools.operation_semantics import evaluate_operation_compatibility


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 声明里「一个字段塞多个资产 ID」的分隔符。注意 GEE 资产 ID 自身含 `/`
# （如 LANDSAT/LC08/C02/T1_L2），故只把**两侧带空白**的斜杠当分隔符，避免误拆。
# 实测三种形态：'GEE:A; GEE:B; GEE:C'（task_010）、'A / B / C'（task_035 的
# gee_asset.id）、'NASA/OMI/Aura/NO2/CLIMATOLOGY_V4 and COPERNICUS/S5P/OFFL/L3_NO2'
# （task_011）。这样的声明违反「组合不得写成单伪资产 ID」的口径，但检查器必须能容错。
ASSET_ID_SEPARATOR = re.compile(r"\s+/\s+|;|\||\s+and\s+", re.IGNORECASE)


def looks_like_gee_asset_id(text):
    """判定一段文本是否形如 GEE 资产 id（可作探测身份）。

    与 geo_data_search._looks_like_gee_asset_id 保持一致：只认 projects/ 社区资产、含
    GEE: 前缀，或含 `/` 的官方/项目 id。地名实体（'Forest Stand 1'）不含 `/`，不会误判。
    """
    stripped = str(text or "").strip()
    if not stripped:
        return False
    if stripped.startswith(("http://", "https://")):
        return False
    if stripped.startswith("GEE:") or stripped.startswith("projects/"):
        return True
    return "/" in stripped and not stripped[0].isspace()


def extract_probe_identity_ids(requirement):
    """从声明侧提取可作为 GEE 探测身份的资产 id（改法②）。

    身份来源：canonical_requirement_name（形如 'ECMWF/ERA5_LAND/HOURLY:temperature'，
    取冒号前 id 部分）与 frozen_constraints.entity（字符串/列表，仅收形如 id 者）。
    只做保守收集，探测失败由调用方留痕放行。
    """
    collected = []
    canonical = str(requirement.get("canonical_requirement_name") or "").strip()
    if canonical and ":" in canonical:
        head = canonical.partition(":")[0].strip()
        if looks_like_gee_asset_id(head) and head not in collected:
            collected.append(head)
    frozen = requirement.get("frozen_constraints") or {}
    entity = frozen.get("entity") if isinstance(frozen, dict) else None
    entity_values = [entity] if isinstance(entity, str) else (
        [item for item in entity if isinstance(item, str)] if isinstance(entity, list) else []
    )
    for value in entity_values:
        value = str(value or "").strip()
        if looks_like_gee_asset_id(value) and value not in collected:
            collected.append(value)
    return collected


def split_asset_ids(value):
    """拆开声明里的多值资产 ID，按出现顺序去重；单值原样返回。"""
    if not isinstance(value, str):
        return []
    out = []
    for part in ASSET_ID_SEPARATOR.split(value):
        part = part.strip()
        if part and part not in out:
            out.append(part)
    return out


# 平台前缀 → `dataset_source_clients` 的源名。**只收该模块 probe_dataset 已支持的源**，
# 于是「声明了平台」才探测，与「无前缀裸标识交目录匹配」的既有边界一致；`Derived:`、
# `External:` 一类非平台标记不在表内，不得送进探测（Benchmark-v3.md §4.2.2）。
PLATFORM_PREFIX_SOURCES = {
    "NASA_CMR": "NASA_CMR",
    "PLANETARY_COMPUTER": "PLANETARY_COMPUTER",
    "EARTH_SEARCH": "EARTH_SEARCH",
    "OSM_TAGINFO": "OSM_TAGINFO",
    "OSM_OVERPASS": "OSM_OVERPASS",
    "API": "HTTP",
    "HTTP": "HTTP",
}

# 平台未给 stac_url 时按平台约定补 collection endpoint——PC/Earth Search 的标识即
# collection id（实测 dual_dataset/task_017 的 source_metadata.doc_url 正是
# `.../api/stac/v1/collections/gridmet`，与此约定一致）。
PLATFORM_COLLECTION_URL = {
    "PLANETARY_COMPUTER": "https://planetarycomputer.microsoft.com/api/stac/v1/collections/{id}",
    "EARTH_SEARCH": "https://earth-search.aws.element84.com/v1/collections/{id}",
}


def extract_platform_probe_records(requirement, path_text):
    """从声明侧提取**非 GEE 平台**的可探测身份（改法⑤）。

    身份有两处来源，按可信度排序，去重后合并：
      ① `source_metadata`：锚点声明自带的结构化身份
         （`{source, id, [doc_url], [stac_url]}`），与 datasets.json 记录同构，可直接
         交给 `probe_dataset`。此前该字段已随 `anchor_members()` 投影进 member 字典，
         却不被 `check_dataset` 读取，于是身份齐全的锚点仍落目录语义匹配被判缺口。
      ② `data_path` 的平台前缀 `<PLATFORM>:<标识>`，PLATFORM 限 `PLATFORM_PREFIX_SOURCES`；
         `API:`/`HTTP:` 的标识本身即端点 URL。

    返回值形如 `{"source": ..., "id": ..., ["doc_url"/"stac_url"]}`，可直接交给探测。
    无前缀的裸标识不产生记录——平台未限定不得推定（同上规格）。
    """
    records = []

    def add(source, identifier, *, doc_url="", stac_url=""):
        source = str(source or "").strip().upper()
        identifier = str(identifier or "").strip()
        if not source or not identifier:
            return
        if source not in PLATFORM_PREFIX_SOURCES.values():
            return
        record = {"source": source, "id": identifier}
        if doc_url:
            record["doc_url"] = doc_url
        if stac_url:
            record["stac_url"] = stac_url
        elif not doc_url and source in PLATFORM_COLLECTION_URL:
            # 仅在声明**完全没给端点**时才按平台约定补，避免用一个约定 URL 盖掉
            # source_metadata 里的权威 doc_url。
            record["stac_url"] = PLATFORM_COLLECTION_URL[source].format(id=identifier)
        if not any(item["source"] == source and item["id"] == identifier for item in records):
            records.append(record)

    metadata = requirement.get("source_metadata")
    if isinstance(metadata, dict):
        add(
            metadata.get("source"), metadata.get("id"),
            doc_url=str(metadata.get("doc_url") or "").strip(),
            stac_url=str(metadata.get("stac_url") or "").strip(),
        )

    prefix, _, rest = str(path_text or "").partition(":")
    rest = rest.strip()
    if rest:
        source = PLATFORM_PREFIX_SOURCES.get(prefix.strip().upper())
        if source == "HTTP":
            # 端点本身即标识；非 URL 形态无法探测，留给目录匹配。
            if rest.startswith(("http://", "https://")):
                add("HTTP", rest, doc_url=rest)
        elif source:
            add(source, rest)
    return records


def normalize_semantic_text(value):
    """把一段文本归一成可比较的语义记号（小写、只留字母数字汉字与空格）。

    与 mas 原版 _normalize_semantic_text 的用途一致：让 query 去重的比较不受大小写/
    标点干扰。实现保持确定性，不依赖外部模型。
    """
    text = str(value or "").strip().lower()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", text)


def build_search_queries(requirement):
    """从声明侧展开多组检索 query（基准侧版，对应 mas 原版 _build_search_queries_from_intent）。

    完全确定性、不依赖 agent/LLM。mas 原版还从 retrieval_product_terms / preferred_family
    展开，基准侧声明里这两者较少出现，这里保留：aliases、dataset_name ，
    canonical_requirement_name（去 `ID:slice` 冒号后的纯 id 形态）、frozen_constraints.entity
    （形如 GEE id 者）、region 拼接，以及常见的语义关键词组合（drought/precipitation/temperature）。
    返回值按出现顺序去重。
    """
    name = str(requirement.get("dataset_name") or requirement.get("name") or "").strip()
    canonical = str(requirement.get("canonical_requirement_name") or name).strip()
    # canonical 可能是 'ECMWF/ERA5_LAND/HOURLY:temperature' 形态，只取冒号前的 id 主体。
    canonical_id = canonical.partition(":")[0].strip()
    aliases = requirement.get("aliases") or []
    if isinstance(aliases, str):
        aliases = [aliases]
    entity = ((requirement.get("frozen_constraints") or {}).get("entity") if isinstance(requirement.get("frozen_constraints"), dict) else None)
    entity_terms = [entity] if isinstance(entity, str) else (
        [item for item in entity if isinstance(item, str)] if isinstance(entity, list) else []
    )
    entity_ids = [term for term in entity_terms if looks_like_gee_asset_id(term)]
    region = ""
    frozen_region = (requirement.get("frozen_constraints") or {}).get("region") if isinstance(requirement.get("frozen_constraints"), dict) else ""
    if frozen_region:
        region = str(frozen_region).strip().replace("_", " ")
    required_region = str(requirement.get("required_region") or "").strip()
    if required_region and not region:
        region = required_region.replace("_", " ")

    queries = []
    # 显式 id 形态优先（含 canonical 的 id 部分与 entity 里的 GEE id）。
    if canonical_id and looks_like_gee_asset_id(canonical_id):
        queries.append(canonical_id)
    queries.extend(entity_ids)
    # 语义关键词组合（与 mas 原版一致）。
    req_text = normalize_semantic_text(" ".join([name, canonical, str(requirement.get("semantic_requirement_text") or "")]))
    if any(token in req_text for token in {"drought", "spi", "pdsi", "spei"}):
        queries.append(f"{canonical_id or name} drought index spi pdsi".strip())
    if any(token in req_text for token in {"precipitation", "precip", "rainfall", "temperature", "temp", "climate"}):
        queries.append(f"{canonical_id or name} precipitation temperature climate".strip())
    # aliases（前 3 条）与 dataset_name：每条先带 region 再不带。
    for alias in aliases[:3]:
        alias_text = str(alias).strip()
        if alias_text:
            queries.append(f"{alias_text} {region}".strip())
            queries.append(alias_text)
    if name:
        queries.append(f"{name} {region}".strip())
        queries.append(name)
    if canonical_id and canonical_id != name:
        queries.append(f"{canonical_id} {region}".strip())
        queries.append(canonical_id)

    deduped = []
    seen = set()
    for query in queries:
        norm = normalize_semantic_text(query)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        deduped.append(query)
    return deduped


def anchor_members(prompt):
    datasets = prompt.get("datasets")
    if isinstance(datasets, list) and datasets:
        descriptions = prompt.get("data_descriptions")
        if isinstance(descriptions, list) and len(descriptions) == len(datasets):
            paths = prompt.get("data_paths")
            if not isinstance(paths, list) or len(paths) != len(datasets):
                raise ValueError("Composite anchor data_paths must match datasets")
            return [dict(description, data_path=path, role=item.get("role", "input"))
                    for description, path, item in zip(descriptions, paths, datasets)]
        return [dict(item.get("data_description", {}), **{
            "data_path": item.get("data_path", item.get("data_description", {}).get("data_path", "")),
            "role": item.get("role", "input"),
        }) for item in datasets]
    description = prompt.get("data_description")
    if not isinstance(description, dict):
        raise ValueError("Task prompt is missing anchor data_description")
    # 单数据集分支此前直接返回 data_description，把**顶层的 data_path 丢了**，于是锚点在
    # manifest 里 source 为空、被判 verification_gap（实测 41 任务 / 250 条目）。
    return [dict(description, **{
        "data_path": prompt.get("data_path", description.get("data_path", "")),
        "role": prompt.get("role", "input"),
    })]


class ExperimentResourceChecker:
    def __init__(self, config, *, search=None, probe=None):
        self.config = config
        self.rc_config = config["agents"]["resources_check"]
        self.search = search
        self.probe_override = probe
        self.cache = {}
        index_path = PROJECT_ROOT / config["tools"]["capability_index"]["path"]
        index = json.loads(index_path.read_text(encoding="utf-8"))
        entries = index if isinstance(index, list) else index["entries"]
        self.functions = {entry["function_name"]: entry for entry in entries}
        self.details = CapabilityDetailStore(index_path.parent / "geo_capability_details")

    def catalog_evidence(self, requirement, dataset_id):
        """Enrich a probed asset only with catalog metadata for that exact identity."""
        if self.search is None:
            from datahypgen.mas.tools.geo_data_search import GeoDataSearch
            self.search = GeoDataSearch()
        intent = dict(requirement, query_text=dataset_id, raw_query=dataset_id,
                      explicit_dataset_ids=[dataset_id])
        result = self.search.resolve_local_dataset(intent)
        candidate = result.get("candidate")
        if candidate is None:
            return {}
        if isinstance(candidate, dict):
            record = dict(candidate)
        else:
            record = dict(candidate.properties)
            record.setdefault("id", candidate.feature_id)
        identity = record.get("ee_id", record.get("id"))
        return record if identity == dataset_id else {}

    async def _probe(self, record):
        if self.probe_override is not None:
            return await self.probe_override(record)
        if record["source"] not in {"GEE_OFFICIAL", "GEE_COMMUNITY"}:
            from datahypgen.mas.tools.dataset_source_clients import probe_dataset
            return await probe_dataset(record)
        import ee
        from dotenv import load_dotenv
        gee_config = self.config["tools"]["gee"]
        load_dotenv(PROJECT_ROOT / gee_config["env_file"], override=gee_config["reload_env_each_call"])
        timeout = self.rc_config["remote_probe"]["timeout_seconds"]

        def query():
            initialize_gee_from_service_account(
                ee, credential_env=gee_config["api_key_env"],
                project_env=gee_config["project_env"],
            )
            set_gee_deadline(ee, timeout * 1000)
            return ee.data.getAsset(record["id"])

        try:
            metadata = await asyncio.wait_for(asyncio.to_thread(query), timeout=timeout)
            return {"status": "pass", "reason": "GEE asset metadata verified", "asset_type": metadata.get("type"), "metadata": metadata}
        except Exception as error:
            # Network/authentication failures are verification gaps, not proof of absent data.
            return {"status": "unknown", "reason": str(error)}

    async def check_dataset(self, requirement, requirement_id, *, mandatory=True, anchor=False):
        key = json.dumps([requirement, requirement_id, mandatory, anchor], sort_keys=True)
        if key in self.cache:
            return copy.deepcopy(self.cache[key])
        name = requirement.get("dataset_name", requirement.get("name", requirement_id))
        path_text = str(requirement.get("data_path", "")).strip()
        entry = {
            "requirement_id": requirement_id, "dataset_title": name,
            "mandatory": mandatory, "resource_role": requirement.get("role", "input"),
            "is_anchor": anchor, "bind_status": "unbound", "binding_status": "verification_gap",
            "online_attempted": False, "connectivity_probe": None,
            "failure_class": "verification_gap", "source": path_text,
        }
        path_candidate = path_text
        if path_text.startswith("LOCAL:"):
            path_candidate = path_text[6:]
        is_remote = path_text.startswith(("GEE:", "External:", "http://", "https://"))
        if path_candidate and not is_remote:
            local_path = Path(path_candidate)
            if not local_path.is_absolute():
                local_path = PROJECT_ROOT / local_path
            if local_path.is_file() or local_path.is_dir():
                is_dir = local_path.is_dir()
                try:
                    if is_dir:
                        # 目录形式数据集（如 mosaics）同样只靠可读性验证。此前只判
                        # is_file()，目录被直接漏过并回落成目录匹配缺口——实测
                        # LOCAL:datasets/MTBS_BSmosaics 就在本地却判成 verification_gap。
                        next(iter(local_path.iterdir()), None)
                    else:
                        with local_path.open("rb") as handle:
                            handle.read(1)
                    entry.update(
                        bind_status="bindable",
                        binding_status="local_directory_verified" if is_dir else "local_file_verified",
                        source=str(local_path),
                        local_stage_path=str(local_path),
                        failure_class="",
                        resolution_source="local_directory" if is_dir else "local_file",
                        connectivity_probe={
                            "status": "pass",
                            "reason": "readable local directory" if is_dir else "readable local file",
                        },
                    )
                    self.cache[key] = entry
                    return copy.deepcopy(entry)
                except OSError as error:
                    entry["connectivity_probe"] = {"status": "fail", "reason": str(error)}
                    self.cache[key] = entry
                    return copy.deepcopy(entry)

        gee_asset = requirement.get("gee_asset", {})
        # 声明可能一个字段塞多个 ID（见 ASSET_ID_SEPARATOR）；拆开后逐个使用。
        # 显式 data_path 优先于 gee_asset.id（沿用原优先级），且同样支持多值：
        # 'GEE:A; GEE:B' 拆成 ['A','B']，每段各自剥掉 GEE: 前缀。
        asset_ids = split_asset_ids(gee_asset.get("id")) if isinstance(gee_asset, dict) else []
        if path_text.startswith("GEE:"):
            asset_ids = [p[4:] if p.startswith("GEE:") else p for p in split_asset_ids(path_text)]
        # 改法②：canonical_requirement_name（ID:slice 形态）与 frozen_constraints.entity
        # 里形如 GEE 资产 id 的实体，同样提升为可探测身份。锚点常把 id 写在这里而非
        # gee_asset.id（实测 task_015 的 ECMWF/ERA5_LAND/HOURLY 即无 gee_asset 字段），
        # 此前它们只能落目录语义匹配、被弱信号判缺口。
        for probe_id in extract_probe_identity_ids(requirement):
            probe_norm = probe_id[4:] if probe_id.startswith("GEE:") else probe_id
            if probe_norm not in asset_ids:
                asset_ids.append(probe_norm)
        direct_url = path_text if path_text.startswith(("http://", "https://")) else requirement.get("stac_url")

        # 显式身份先直接探测。锚点与带 GEE id 的需求，其身份是任务声明给定的，
        # 一次 getAsset 就能定论；此前它们要等目录语义匹配达到 resolved_strong 才轮到
        # 探测，于是被弱信号一票否决（实测 605 条锚点缺口里绝大多数都有明确 GEE id）。
        # 多值时逐个试、首个通过即定论。探测未通过（含网络/鉴权失败）不构成
        # "产品不存在"的证据，留痕后继续走目录匹配。
        # 改法⑤：身份不止 GEE。锚点声明常把非 GEE 平台身份写在 source_metadata 或
        # data_path 前缀（`NASA_CMR:`/`PLANETARY_COMPUTER:`/`API:`），此前一律落目录
        # 语义匹配被判缺口（实测 13 个任务各 5 个 idea 全灭）。这里一并纳入显式探测，
        # 由 `_probe` 按 source 分派到 probe_dataset / GEE getAsset。
        probe_records = []
        if asset_ids and not direct_url:
            probe_records.extend(
                ("explicit_asset_probe", "GEE:" + candidate_id,
                 {"source": "GEE_OFFICIAL", "id": candidate_id})
                for candidate_id in asset_ids
            )
        probe_records.extend(
            ("platform_asset_probe", f"{record['source']}:{record['id']}", record)
            for record in extract_platform_probe_records(requirement, path_text)
        )
        for resolution_source, bind_source, record in probe_records:
            entry["resolution_source"] = resolution_source
            probe = await self._probe(record)
            entry["connectivity_probe"] = probe
            entry["resource_evidence"] = {"record": record, "probe": probe}
            if probe["status"] == "pass":
                entry.update(
                    source=bind_source,
                    dataset_id=record["id"],
                    bind_status="bindable",
                    binding_status="verified",
                    failure_class="",
                )
                self.cache[key] = entry
                return copy.deepcopy(entry)

        if direct_url:
            if requirement.get("stac_url"):
                probe = await self._probe({"source": requirement.get("dataset_source", "STAC"), "id": name, "stac_url": direct_url})
            else:
                from datahypgen.mas.tools.dataset_source_clients import probe_http_head
                probe = await probe_http_head(direct_url)
            entry.update(source=direct_url, resolution_source="declared_endpoint", connectivity_probe=probe)
            entry["resource_evidence"] = {"probe": probe}
        else:
            if self.search is None:
                from datahypgen.mas.tools.geo_data_search import GeoDataSearch
                self.search = GeoDataSearch()
            # 多 query 展开（对应 mas 原版 _build_search_queries_from_intent）：aliases /
            # dataset_name / canonical / entity id / region 会展开成多组词。这里对每组词都
            # 先走本地目录语义匹配，本地未达可绑定标准时再在线回退一次；取全部 query 中
            # 最优结果（首个 resolved_strong / 已 accepted_for_binding / 首个不触发回退者）。
            # 确定性实现，不依赖 agent/LLM。显式 id（asset_ids）始终作为第一组 query 与
            # explicit_dataset_ids 传入，最高优先。
            queries = list(asset_ids) if asset_ids else []
            for query in build_search_queries(requirement):
                if query not in queries:
                    queries.append(query)
            if not queries:
                queries = [str(name)]
            best = None
            best_source = ""
            best_queried = ""
            # 对应 mas 原版 :1124-1145：本地重复给出同一个 weak 候选时，换 query 词再打
            # 在线目录不会改善结论（在线要么返回同一 asset，要么返回不匹配的别的东西），
            # 而每次在线都要并行等 GEE/CMR/PC 三源、网络差时单次数十秒，一个 requirement
            # 重复 10 次就是十几分钟。故同一候选出现第二次起跳过在线，只记本地轮。
            seen_local_candidate_ids = set()
            for query in queries:
                intent = dict(requirement)
                intent["query_text"] = query
                intent["raw_query"] = query
                if asset_ids:
                    intent["explicit_dataset_ids"] = list(asset_ids)
                result = self.search.resolve_local_dataset(intent)
                source = "local_catalog"
                local_candidate_id = self._resolution_candidate_id(result)
                repeated_local_candidate = (
                    local_candidate_id is not None
                    and local_candidate_id in seen_local_candidate_ids
                    and result.get("status") == "resolved_weak"
                )
                if local_candidate_id is not None:
                    seen_local_candidate_ids.add(local_candidate_id)
                # 在线回退不再只认「本地无候选」：本地给了候选但未达可绑定标准时同样回退一次。
                # "本地有候选"不等于"该产品可用"——候选可能只是语义近邻，并没有回答它到底在不在。
                if not repeated_local_candidate and self._needs_online_fallback(result) \
                        and self.rc_config["allow_online_gee_fallback"]:
                    entry["online_attempted"] = True
                    online = await self.search.resolve_online_dataset(intent)
                    if not self._needs_online_fallback(online):
                        result = online
                        source = "online_catalog"
                if best is None or self._resolution_score(result) > self._resolution_score(best):
                    best = result
                    best_source = source
                    best_queried = query
                if result.get("status") == "resolved_strong":
                    break
            result = best
            entry["resolution_source"] = best_source
            entry["resolution_query"] = best_queried
            candidate = result.get("candidate")
            entry["resolution_status"] = result.get("status")
            entry["binding_decision"] = result.get("binding_decision", {})
            # A rejected candidate says nothing about the availability of the requested product.
            if self._needs_online_fallback(result):
                entry["reason"] = "No verified matching catalog binding"
                self.cache[key] = entry
                return copy.deepcopy(entry)
            if isinstance(candidate, dict):
                record = dict(candidate)
            else:
                record = dict(candidate.properties)
                record["id"] = record.get("id", candidate.feature_id)
                source_prefix = candidate.source.split(":", 1)[0]
                record["source"] = record.get("source") if record.get("source") else source_prefix
                if record["source"] in {"GEE", "GEE_ONLINE"}:
                    record["source"] = "GEE_OFFICIAL"
                if record["source"].endswith("_ONLINE"):
                    record["source"] = record["source"][:-7]
                if record.get("docs"):
                    record["doc_url"] = record["docs"]
            if record.get("ee_id"):
                record["id"] = record["ee_id"]
            # 多值声明时，命中的候选只要等于其中任意一个 ID 即算身份相符。
            if asset_ids and record["id"] not in asset_ids:
                entry["reason"] = "Explicit anchor asset identity mismatch"
                self.cache[key] = entry
                return copy.deepcopy(entry)
            if not record.get("source") or not record.get("id"):
                entry["reason"] = "Catalog binding is missing source or identity"
                self.cache[key] = entry
                return copy.deepcopy(entry)
            entry.update(source=record["source"] + ":" + record["id"], dataset_id=record["id"])
            entry["connectivity_probe"] = await self._probe(record)
            entry["resource_evidence"] = {"record": record, "probe": entry["connectivity_probe"]}
        if entry["connectivity_probe"]["status"] == "pass":
            entry.update(bind_status="bindable", binding_status="verified", failure_class="")
        else:
            entry["reason"] = entry["connectivity_probe"].get("reason", "Probe did not verify access")
        self.cache[key] = entry
        return copy.deepcopy(entry)

    @staticmethod
    def _needs_online_fallback(result):
        """目录结果是否未达「可绑定」标准。

        `accepted_for_binding` 才是目录侧给出的绑定判据，`status` 只是它的粗粒度标签。
        两者不一致时以 accepted_for_binding 为准——实测有 12 条 `resolved_weak` 已被目录侧
        接受绑定，却因为只看 `status != "resolved_strong"` 而被误判成 verification_gap。
        """
        if result.get("candidate") is None:
            return True
        if result.get("status") == "resolved_strong":
            return False
        return not (result.get("binding_decision") or {}).get("accepted_for_binding")

    @staticmethod
    def _resolution_score(result):
        """给目录/在线结果打一个可排序的单调标量，用于跨 query 选取最优。

        与 _needs_online_fallback/绑定判据保持一致：resolved_strong 最高，
        accepted_for_binding 次之，其余按是否给出 candidate 区分。score 越大越好；返回
        0 表示不可绑定（等同 _needs_online_fallback 判定为真）。
        """
        if result.get("status") == "resolved_strong":
            return 3
        if (result.get("binding_decision") or {}).get("accepted_for_binding"):
            return 2
        return 1 if result.get("candidate") is not None else 0

    @staticmethod
    def _resolution_candidate_id(result):
        """从目录/在线结果里取候选的稳定标识，用于判定"同一 weak 候选重复出现"。

        对应 mas 原版 _resolution_candidate_id（resourcescheck_agent.py:2005-2019）。
        只取 identity（id / ee_id / feature_id），不取来源/评分——同一产品换词应命中同一 id。
        """
        candidate = result.get("candidate")
        if candidate is None:
            return None
        properties = getattr(candidate, "properties", None)
        if not isinstance(properties, dict) and isinstance(candidate, dict):
            properties = candidate.get("properties") or candidate
        if not isinstance(properties, dict):
            properties = {}
        candidate_id = (
            properties.get("id")
            or properties.get("ee_id")
            or getattr(candidate, "feature_id", None)
        )
        return str(candidate_id) if candidate_id and str(candidate_id) != "None" else None

    def check_tools(self, plan):
        validations = []
        for step in plan if isinstance(plan, list) else []:
            if not isinstance(step, dict):
                continue
            hint = step.get("implementation_hint", {})
            if not isinstance(hint, dict):
                hint = {}
            entry = self.functions.get(hint.get("function_name"))
            score = 1.0 if entry and entry.get("pip_verified") else 0.6 if entry else 0.3
            evidence = self.details.merge(entry, self.details.get(entry)) if entry else hint
            compatibility = evaluate_operation_compatibility(step, evidence)
            advisory = compatibility.get("advisory", compatibility["status"] == "operation_unproven")
            validations.append({
                "step_id": step.get("id"), "function_name": hint.get("function_name", ""),
                "score": score, "reachability_passed": score >= 0.6,
                "operation_compatible": compatibility["compatible"] or advisory,
                "operation_compatibility_status": compatibility["status"],
                "operation_compatibility_reason": compatibility["reason"],
            })
        return validations

    async def evaluate(self, applications, prompt):
        # No plan DAG / planning-contract admission gate for baselines.
        evaluated = copy.deepcopy(applications)
        anchors = []
        for index, member in enumerate(anchor_members(prompt), start=1):
            anchors.append(await self.check_dataset(member, f"anchor_component_{index}", anchor=True))
        audit = []
        for app in evaluated:
            if not isinstance(app, dict):
                raise ValueError("Benchmark application must be an object")
            if not isinstance(app.get("required_datasets"), list) or not isinstance(app.get("task_plan"), list):
                raise ValueError("Structured resource requirements are missing; extract native prose first")
            if not app["required_datasets"] and not app["task_plan"]:
                raise ValueError("Cannot verify an application with no resource requirements")
            manifest = copy.deepcopy(anchors)
            for index, requirement in enumerate(app.get("required_datasets", []), start=1):
                if not isinstance(requirement, dict):
                    raise ValueError("Dataset requirement must be an object")
                manifest.append(await self.check_dataset(
                    requirement, str(requirement.get("id", f"required_dataset_{index}")),
                    mandatory=requirement.get("execution_requirement_level") != "optional",
                ))
            tools = self.check_tools(app.get("task_plan", []))
            missing = [item for item in manifest if item["mandatory"] and item["bind_status"] != "bindable"]
            hard_tools = [item for item in tools if not item["operation_compatible"]]
            tool_gaps = [item for item in tools if not item["reachability_passed"]]
            rule = ""
            failure_class = ""
            if any(item["is_anchor"] for item in missing):
                rule, failure_class = "RC-L2_anchor_verification_gap", "verification_gap"
            elif missing or tool_gaps:
                rule, failure_class = "RC-L2_resource_verification_gap", "verification_gap"
            if hard_tools:
                rule, failure_class = "RC-L1_implementation_incompatible", "implementation_incompatible"
            passed = not (missing or hard_tools or tool_gaps)
            guaranteed = [item["dataset_title"] for item in manifest if item["bind_status"] == "bindable"]
            app["resource_check"] = {
                "guaranteed_datasets": guaranteed,
                "execution_contract": {
                    "resource_check_passed": passed, "should_terminate": bool(hard_tools),
                    "terminal_rule": rule, "data_access_manifest": manifest,
                    "tool_reachability_score": min((item["score"] for item in tools), default=None),
                    "step_tool_validations": tools,
                },
            }
            app["resource_check_feedback"] = {"terminal_rule": rule, "failure_class": failure_class}
            # Keep generation statuses intact; resource verdicts are a separate evaluation axis.
            audit.append({"idea_id": app.get("idea_id", ""), "source": "shared_resource_check",
                          "executed_by_evaluator": True, "resource_check_passed": passed})
        return evaluated, {"execution_owner": "evaluator", "application_audit": audit,
                           "executed_count": len(audit), "contract_validation": False}

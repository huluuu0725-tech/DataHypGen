#!/usr/bin/env python3
"""
同步 Earth Engine 官方弃用资产清单，并回填本地 catalog 的弃用状态。

判据来源是 Earth Engine 官方 STAC 弃用清单，而不是 datasets.json 里 title 的
`[deprecated]` 后缀。后者实测零误报但会漏判：COPERNICUS/DEM/GLO30、NOAA/CFSR、
NASA/TEMPO/* 等条目的本地标题看不出任何弃用痕迹，只有官方清单知道。

回填只写事实，不做过滤也不做降权：弃用资产在 GEE 上仍可正常读取，本地 catalog
内的弃用条目 removal_date 全部为空（只停更、无删除计划），个别条目甚至仍在更新
（如 NOAA/CFSR 更新至 2026-07-06）。数据能不能用应由时间合同门禁依据
temporal_start / temporal_end 判定，弃用状态只作为 soft 信息透传给下游。

用法：
    python -m datahypgen.mas.tools.sync_deprecated_assets --dry-run
    python -m datahypgen.mas.tools.sync_deprecated_assets
    python -m datahypgen.mas.tools.sync_deprecated_assets --offline

回填后需重建索引库（datasets.db 由 build_dataset_index_db 构建，9 列表结构）：
    python -m datahypgen.mas.tools.build_dataset_index_db
"""
import argparse
import json
import logging
import os
import shutil
import ssl
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEPRECATED_CATALOG_URL = (
    "https://storage.googleapis.com/earthengine-stac/catalog/catalog_deprecated.json"
)
DEPRECATION_SOURCE = "gee_stac_catalog_deprecated"

_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
DEFAULT_SNAPSHOT_PATH = os.path.join(_DATA_DIR, "gee_deprecated_assets.json")
DEFAULT_DATASETS_PATH = os.path.join(_DATA_DIR, "datasets.json")


def fetch_deprecated_catalog(
    snapshot_path: str,
    offline: bool = False,
    retries: int = 4,
) -> Dict[str, Any]:
    """
    获取官方弃用清单。offline 模式直接复用本地快照，否则联网抓取并覆盖快照。

    网络失败不静默回退到本地快照——那会让"同步"变成一次无声的空操作。
    需要离线复现时必须显式传 --offline。
    """
    if offline:
        if not os.path.exists(snapshot_path):
            raise FileNotFoundError(
                f"离线模式需要已有快照，但 {snapshot_path} 不存在。"
                f"请先联网运行一次以生成快照。"
            )
        with open(snapshot_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        logger.info("离线模式：复用快照 %s", snapshot_path)
        return payload

    context = ssl.create_default_context()
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            raw = urllib.request.urlopen(
                DEPRECATED_CATALOG_URL, timeout=120, context=context
            ).read()
            payload = json.loads(raw)
            break
        except (urllib.error.URLError, ssl.SSLError, TimeoutError, ValueError) as exc:
            last_error = exc
            logger.warning("第 %d/%d 次抓取失败：%s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(2 * attempt)
    else:
        raise RuntimeError(f"抓取官方弃用清单失败：{last_error}")

    os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
    with open(snapshot_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("已抓取官方清单并落地快照：%s", snapshot_path)
    return payload


def extract_deprecated_index(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    从官方 STAC payload 提取 asset_id -> 弃用元信息。

    官方用 link 的 `title` 字段承载 asset id（见 ee/deprecation.py 的
    DeprecatedAsset.FromStacLink），不是人类可读标题。
    """
    index: Dict[str, Dict[str, Any]] = {}
    for link in payload.get("links", []):
        if not isinstance(link, dict) or not link.get("deprecated"):
            continue
        asset_id = str(link.get("title") or "").strip()
        if not asset_id:
            continue
        removal_date = str(link.get("gee:removal_date") or "").strip()
        index[asset_id] = {
            "replacement_id": str(link.get("gee:replacement_id") or "").strip() or None,
            "removal_date": removal_date or None,
            "learn_more_url": str(link.get("gee:learn_more_url") or "").strip() or None,
        }
    return index


def backfill_datasets(
    datasets: List[Dict[str, Any]],
    index: Dict[str, Dict[str, Any]],
    synced_at: str,
) -> Dict[str, Any]:
    """
    就地回填 datasets 记录的弃用状态，返回统计信息。

    幂等：官方清单里已撤销弃用的条目会被清回 deprecated=False 并移除 deprecation 块。
    """
    local_ids = {str(r.get("id") or "").strip() for r in datasets if isinstance(r, dict)}

    marked = 0
    cleared = 0
    changed = 0
    with_replacement = 0
    replacement_in_catalog = 0
    with_removal_date = 0
    missing_replacement: List[Tuple[str, str]] = []

    for record in datasets:
        if not isinstance(record, dict):
            continue
        asset_id = str(record.get("id") or "").strip()
        entry = index.get(asset_id)

        if entry is None:
            # 不在官方清单里：确保状态为未弃用，清掉可能存在的历史残留
            if record.get("deprecated") is not False or "deprecation" in record:
                if record.get("deprecated") is True or "deprecation" in record:
                    cleared += 1
                    changed += 1
                record["deprecated"] = False
                record.pop("deprecation", None)
            continue

        replacement_id = entry["replacement_id"]
        in_catalog = bool(replacement_id and replacement_id in local_ids)
        deprecation_block = {
            "replacement_id": replacement_id,
            "replacement_in_catalog": in_catalog,
            "removal_date": entry["removal_date"],
            "learn_more_url": entry["learn_more_url"],
            "source": DEPRECATION_SOURCE,
            "synced_at": synced_at,
        }

        previous = record.get("deprecation")
        previous_comparable = (
            {k: v for k, v in previous.items() if k != "synced_at"}
            if isinstance(previous, dict)
            else None
        )
        current_comparable = {
            k: v for k, v in deprecation_block.items() if k != "synced_at"
        }
        if record.get("deprecated") is not True or previous_comparable != current_comparable:
            changed += 1

        record["deprecated"] = True
        record["deprecation"] = deprecation_block

        marked += 1
        if replacement_id:
            with_replacement += 1
            if in_catalog:
                replacement_in_catalog += 1
            else:
                missing_replacement.append((asset_id, replacement_id))
        if entry["removal_date"]:
            with_removal_date += 1

    return {
        "total_records": len(datasets),
        "official_deprecated": len(index),
        "marked": marked,
        "cleared": cleared,
        "changed": changed,
        "official_not_in_catalog": len(index) - marked,
        "with_replacement": with_replacement,
        "replacement_in_catalog": replacement_in_catalog,
        "with_removal_date": with_removal_date,
        "missing_replacement": missing_replacement,
    }


def write_datasets(datasets: List[Dict[str, Any]], path: str) -> None:
    """
    写回 catalog。保持原文件的 CRLF 行尾与 indent=2 格式，避免整文件产生假改动。
    """
    with open(path, "w", encoding="utf-8", newline="\r\n") as f:
        json.dump(datasets, f, ensure_ascii=False, indent=2)


def backup_datasets(path: str) -> str:
    """datasets.json 不受 git 跟踪，改写前必须自备份。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{path}.backup_{stamp}"
    shutil.copy2(path, backup_path)
    logger.info("已备份原 catalog：%s", backup_path)
    return backup_path


def report(stats: Dict[str, Any]) -> None:
    total = stats["total_records"]
    marked = stats["marked"]
    rate = (marked / total * 100) if total else 0.0
    logger.info("=" * 62)
    logger.info("本地 catalog 记录总数        : %d", total)
    logger.info("官方清单弃用条目             : %d", stats["official_deprecated"])
    logger.info(
        "回填标记为弃用               : %d / %d (%.2f%%)", marked, total, rate
    )
    logger.info("官方弃用但本地 catalog 无此条 : %d", stats["official_not_in_catalog"])
    logger.info("撤销历史弃用标记             : %d", stats["cleared"])
    logger.info("本轮实际发生变更的记录       : %d", stats["changed"])
    logger.info("-" * 62)
    if marked:
        with_repl = stats["with_replacement"]
        in_cat = stats["replacement_in_catalog"]
        logger.info(
            "有官方继任者                 : %d / %d (%.1f%%)",
            with_repl, marked, with_repl / marked * 100,
        )
        logger.info(
            "继任者本地 catalog 可用      : %d / %d (%.1f%%)",
            in_cat, marked, in_cat / marked * 100,
        )
        logger.info("带 removal_date（有删除计划）: %d", stats["with_removal_date"])
    for asset_id, replacement in stats["missing_replacement"]:
        logger.warning("继任者不在本地 catalog：%s -> %s", asset_id, replacement)
    logger.info("=" * 62)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="同步 GEE 官方弃用清单并回填本地 catalog"
    )
    parser.add_argument("--datasets", default=DEFAULT_DATASETS_PATH, help="datasets.json 路径")
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT_PATH, help="官方清单快照落地路径")
    parser.add_argument("--offline", action="store_true", help="复用已有快照，不联网")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写回 catalog")
    parser.add_argument("--no-backup", action="store_true", help="跳过备份（不推荐）")
    args = parser.parse_args()

    payload = fetch_deprecated_catalog(args.snapshot, offline=args.offline)
    index = extract_deprecated_index(payload)
    if not index:
        raise RuntimeError("官方清单解析结果为空，拒绝以空清单回填 catalog")

    with open(args.datasets, "r", encoding="utf-8") as f:
        datasets = json.load(f)
    if not isinstance(datasets, list):
        raise ValueError(f"{args.datasets} 必须是数据集记录列表")

    synced_at = datetime.now().strftime("%Y-%m-%d")
    stats = backfill_datasets(datasets, index, synced_at)
    report(stats)

    if args.dry_run:
        logger.info("dry-run：未写回 %s", args.datasets)
        return

    if stats["changed"] == 0:
        logger.info("无变更，跳过写回")
        return

    if not args.no_backup:
        backup_datasets(args.datasets)
    write_datasets(datasets, args.datasets)
    logger.info("已写回 %s", args.datasets)
    logger.info("请重建索引库：python -m datahypgen.mas.tools.build_dataset_index_db")


if __name__ == "__main__":
    main()

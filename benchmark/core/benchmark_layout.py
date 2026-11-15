"""Canonical grouped task layout helpers for the benchmark scripts."""

from __future__ import annotations

import json
import re
from pathlib import Path


GROUPS = ("single_dataset", "dual_dataset", "three_or_more_datasets")
TASK_ID_PATTERN = re.compile(r"^(single_dataset|dual_dataset|three_or_more_datasets)/(task_\d{3})$")

SYSTEM_KEYS = ("ours",)


def system_outputs_root(benchmark_dir: Path, system_key: str) -> Path:
    """单个系统的产物根：``system_outputs/<system_key>/``。

    2026-10-07 起每系统一层。此前 ours 直接铺在 ``system_outputs/`` 根下、基线用
    ``<system_key>_benchmark/``，两个 ``SYSTEM_ROOTS`` 常量各写一份因而漂移；本函数
    是唯一来源。注意目录名与产物 payload 里的 ``system`` 字段（仍为
    ``<system_key>_benchmark`` / ``internagent_benchmark``）是有意不同的两套标识：
    前者是布局，后者是实验身份。
    """
    if system_key not in SYSTEM_KEYS:
        raise ValueError(f"未知 system_key: {system_key!r}；可选 {SYSTEM_KEYS}")
    return benchmark_dir / "system_outputs" / system_key


def system_outputs_roots(benchmark_dir: Path) -> dict:
    return {key: system_outputs_root(benchmark_dir, key) for key in SYSTEM_KEYS}


def load_task_manifest(benchmark_dir: Path) -> dict:
    path = benchmark_dir / "dataset" / "task_groups.json"
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest.get("tasks"), list):
        raise ValueError(f"task manifest has no tasks array: {path}")
    return manifest


def canonical_task_ids(benchmark_dir: Path) -> list[str]:
    manifest = load_task_manifest(benchmark_dir)
    task_ids = [entry.get("new_task_id") for entry in manifest["tasks"]]
    invalid = [task_id for task_id in task_ids if not isinstance(task_id, str) or not TASK_ID_PATTERN.fullmatch(task_id)]
    if invalid:
        raise ValueError(f"invalid canonical task IDs in task_groups.json: {invalid[:5]}")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("duplicate canonical task IDs in task_groups.json")
    return sorted(task_ids)


def split_task_id(task_id: str) -> tuple[str, str]:
    match = TASK_ID_PATTERN.fullmatch(task_id.strip())
    if not match:
        raise ValueError(
            f"task ID must be canonical <group>/task_NNN, got {task_id!r}"
        )
    return match.group(1), match.group(2)


def require_canonical_task_id(benchmark_dir: Path, task_id: str) -> str:
    split_task_id(task_id)
    if task_id not in set(canonical_task_ids(benchmark_dir)):
        raise ValueError(f"task ID is not present in task_groups.json: {task_id!r}")
    return task_id


def grouped_json_path(root: Path, task_id: str) -> Path:
    group, local_id = split_task_id(task_id)
    return root / group / f"{local_id}.json"


def grouped_task_dir(root: Path, task_id: str) -> Path:
    group, local_id = split_task_id(task_id)
    return root / group / local_id


def grouped_result_path(root: Path, task_id: str, suffix: str = ".json") -> Path:
    group, local_id = split_task_id(task_id)
    return root / group / f"{local_id}{suffix}"

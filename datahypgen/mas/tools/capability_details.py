"""Lazy access to per-function capability detail records."""

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional


class CapabilityDetailStore:
    """Load only the package detail files needed by selected capability candidates."""

    def __init__(self, details_dir: Path) -> None:
        self.details_dir = Path(details_dir)
        self._module_cache: Dict[str, Dict[str, Any]] = {}
        self._misc_cache: Optional[Dict[str, Any]] = None

    @staticmethod
    def _module_from_entry(entry: Dict[str, Any]) -> str:
        module = str(entry.get("module") or "").strip()
        if module:
            return module
        function_name = str(entry.get("function_name") or "").strip()
        if "." in function_name:
            return function_name.split(".", 1)[0]
        package = str(entry.get("package") or "").strip()
        return package.split("==", 1)[0].split(">=", 1)[0].strip()

    def _load_module(self, module: str) -> Dict[str, Any]:
        module = module.strip()
        if not module:
            return {}
        if module in self._module_cache:
            return self._module_cache[module]

        path = self.details_dir / f"{module}.json"
        if not path.exists():
            self._module_cache[module] = {}
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._module_cache[module] = {}
            return {}
        if not isinstance(data, dict):
            data = {}
        self._module_cache[module] = data
        return data

    def _load_misc(self) -> Dict[str, Any]:
        if self._misc_cache is not None:
            return self._misc_cache
        path = self.details_dir / "_misc.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        if not isinstance(data, dict):
            data = {}
        self._misc_cache = data
        return data

    def get(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        function_name = str(entry.get("function_name") or "").strip()
        if not function_name:
            return {}

        module_data = self._load_module(self._module_from_entry(entry))
        detail = module_data.get(function_name)
        if not isinstance(detail, dict):
            detail = self._load_misc().get(function_name)
        return copy.deepcopy(detail) if isinstance(detail, dict) else {}

    @staticmethod
    def merge(entry: Dict[str, Any], detail: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(entry)
        if not isinstance(detail, dict) or not detail:
            return merged
        for key in ("raw_description", "parameters", "returns", "source_type", "execution_mode"):
            if key in detail and detail[key] not in (None, "", [], {}):
                merged[key] = detail[key]
        return merged


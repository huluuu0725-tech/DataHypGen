"""
工具查找管理器
用于快速查找和匹配工具（基于文本匹配，不使用embedding）
"""
import json
import os
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class ToolFinder:
    """
    工具查找管理器
    
    功能：
    1. 加载工具列表（从geo_capability_index.json）
    2. 文本匹配查找工具（基于名称、描述、标签）
    3. 获取工具schema信息
    """
    
    def __init__(self, tools_list_path: str = None, tool_schema_dir: str = None):
        """
        初始化工具查找管理器
        
        Args:
            tools_list_path: 工具列表文件路径（默认：tools/data/geo_capability_index.json）
            tool_schema_dir: 工具schema目录（默认：tools/data/tool_schemas）
        """
        if tools_list_path is None:
            base_dir = Path(__file__).parent / "data"
            tools_list_path = str(base_dir / "geo_capability_index.json")
        
        if tool_schema_dir is None:
            base_dir = Path(__file__).parent / "data"
            tool_schema_dir = str(base_dir / "tool_schemas")
        
        self.tools_list_path = tools_list_path
        self.tool_schema_dir = tool_schema_dir
        
        # 确保目录存在
        os.makedirs(os.path.dirname(self.tools_list_path), exist_ok=True)
        os.makedirs(self.tool_schema_dir, exist_ok=True)
        
        # 加载工具列表
        self.tools_list: List[Dict[str, Any]] = []
        self._cap_entries: List[Dict[str, Any]] = []
        self._name_index: Dict[str, Dict] = {}  # 名称 -> 工具信息
        self._load_tools_list()
    
    def _load_tools_list(self):
        if not os.path.exists(self.tools_list_path):
            logger.info(f"Tools list file not found: {self.tools_list_path}")
            # index file is read-only, do NOT call _save_tools_list()
            return
        try:
            with open(self.tools_list_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                # geo_capability_index format (JSON array)
                self._cap_entries = data
                self.tools_list = data  # keep legacy methods working
                for entry in data:
                    fn = entry.get("function_name", "")
                    if fn:
                        self._name_index[fn.lower()] = entry
                logger.info(f"Loaded {len(data)} capability entries from {self.tools_list_path}")
            elif isinstance(data, dict):
                # legacy tools_list format (backward compat)
                self.tools_list = data.get("tools", [])
                for tool in self.tools_list:
                    name = tool.get("name", "")
                    if name:
                        self._name_index[name.lower()] = tool
                logger.info(f"Loaded {len(self.tools_list)} tools from {self.tools_list_path}")
        except Exception as e:
            logger.warning(f"Failed to load tools list: {e}")
    
    def find_tool_by_name(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        通过名称精确查找工具
        
        Args:
            tool_name: 工具名称
            
        Returns:
            工具信息字典，如果不存在返回None
        """
        return self._name_index.get(tool_name.lower())
    
    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Return the capability index entry for a tool.
        First tries exact match by function_name via _name_index.
        Falls back to search() Level 2 (keyword-only) if not found.
        Does NOT read tool_schemas/ directory.
        """
        entry = self._name_index.get(tool_name.lower())
        if entry:
            return entry
        results = self.search(tool_name, ["unknown"], top_k=1)
        return results[0] if results else None
    
    def search(
        self,
        keyword: str,
        input_data_types: List[str],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Two-level fallback search:
          Level 1: keyword match + input_data_type filter
          Level 2: keyword match only (when input_data_types all "unknown" or Level 1 empty)

        keyword matches against: function_name / task_type / domain_tags / raw_description
        Results sorted by confidence descending, capped at top_k.
        """
        if not self._cap_entries:
            return []

        kw = keyword.lower()
        all_unknown = not input_data_types or all(
            t.lower() == "unknown" for t in input_data_types
        )

        def _keyword_match(entry: Dict[str, Any]) -> bool:
            if kw in entry.get("function_name", "").lower():
                return True
            if kw in entry.get("task_type", "").lower():
                return True
            if any(kw in tag.lower() for tag in (entry.get("domain_tags") or [])):
                return True
            if kw in entry.get("raw_description", "").lower():
                return True
            return False

        def _input_type_match(entry: Dict[str, Any]) -> bool:
            entry_types = [t.lower() for t in (entry.get("input_data_type") or [])]
            return any(t.lower() in entry_types for t in input_data_types)

        # Level 1
        if not all_unknown:
            level1 = [
                e for e in self._cap_entries
                if _keyword_match(e) and _input_type_match(e)
            ]
            if level1:
                level1.sort(key=lambda e: e.get("confidence", 0.0), reverse=True)
                return level1[:top_k]

        # Level 2 (fallback)
        level2 = [e for e in self._cap_entries if _keyword_match(e)]
        level2.sort(key=lambda e: e.get("confidence", 0.0), reverse=True)
        return level2[:top_k]

    def list_all_tools(self) -> List[Dict[str, Any]]:
        """
        获取所有工具列表
        
        Returns:
            所有工具的列表
        """
        return self.tools_list.copy()
    

# 全局单例实例
_tool_finder: Optional[ToolFinder] = None


def get_tool_finder() -> ToolFinder:
    """获取全局工具查找管理器单例"""
    global _tool_finder
    if _tool_finder is None:
        _tool_finder = ToolFinder()
    return _tool_finder








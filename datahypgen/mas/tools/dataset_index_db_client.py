"""
数据集索引数据库查询客户端

提供对 datasets.db 的查询接口，返回与 JSON 加载格式一致的数据集字典。
"""
import json
import logging
import sqlite3
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DatasetIndexDBClient:
    """数据集索引数据库查询客户端"""

    def __init__(self, db_path: str):
        """
        初始化数据库客户端

        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()

    def get_all_datasets(self, include_deprecated: bool = True) -> List[Dict[str, Any]]:
        """
        获取所有数据集

        Args:
            include_deprecated: 是否包含已弃用的数据集，默认包含。

                弃用不等于不可读：GEE 对弃用 asset 只发 DeprecationWarning，数据仍能
                正常访问；本地 catalog 内的弃用条目 removal_date 全部为空（只停更、
                无删除计划），个别条目甚至仍在更新。能否使用应由时间合同门禁依据
                temporal_start / temporal_end 判定，不在 catalog 装载层剔除。

        Returns:
            数据集字典列表
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT data_json FROM datasets")
        datasets = [json.loads(row["data_json"]) for row in cursor.fetchall()]

        if not include_deprecated:
            datasets = [d for d in datasets if not d.get("deprecated", False)]

        return datasets

    def search_by_source(self, source: str) -> List[Dict[str, Any]]:
        """
        按数据源搜索

        Args:
            source: 数据源名称（如 "GEE_OFFICIAL", "GEE_COMMUNITY"）

        Returns:
            匹配的数据集列表
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT data_json FROM datasets WHERE data_source = ?",
            (source,)
        )
        rows = cursor.fetchall()
        return [json.loads(row["data_json"]) for row in rows]

    def search_by_keyword(
        self,
        keyword: str,
        fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        按关键词搜索

        Args:
            keyword: 搜索关键词
            fields: 搜索字段列表，默认为 ["title", "description", "tags"]

        Returns:
            匹配的数据集列表
        """
        if fields is None:
            fields = ["title", "description", "tags"]

        # 构建 WHERE 子句
        conditions = []
        params = []
        for field in fields:
            conditions.append(f"{field} LIKE ?")
            params.append(f"%{keyword}%")

        where_clause = " OR ".join(conditions)
        query = f"SELECT data_json FROM datasets WHERE {where_clause}"

        cursor = self.conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [json.loads(row["data_json"]) for row in rows]

    def search_by_time_range(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        按时间范围搜索

        Args:
            start: 起始时间（ISO 格式，如 "2020-01-01"）
            end: 结束时间（ISO 格式）

        Returns:
            时间范围有交集的数据集列表
        """
        conditions = []
        params = []

        if start:
            conditions.append("temporal_end >= ?")
            params.append(start)
        if end:
            conditions.append("temporal_start <= ?")
            params.append(end)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"SELECT data_json FROM datasets WHERE {where_clause}"

        cursor = self.conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [json.loads(row["data_json"]) for row in rows]

    def search_by_ee_id(self, ee_id: str) -> Optional[Dict[str, Any]]:
        """
        按 Earth Engine ID 精确查找

        Args:
            ee_id: GEE asset ID

        Returns:
            匹配的数据集，如果不存在则返回 None
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT data_json FROM datasets WHERE ee_id = ?",
            (ee_id,)
        )
        row = cursor.fetchone()
        return json.loads(row["data_json"]) if row else None

    def count_datasets(self) -> int:
        """
        统计数据集总数

        Returns:
            数据集数量
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM datasets")
        return cursor.fetchone()["count"]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

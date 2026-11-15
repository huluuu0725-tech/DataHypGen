"""结构化文献检索查询构造器

将扁平的词条拼接改造为分层的语义查询，提高 OpenAlex 检索质量。
"""
import logging
import re
from typing import Any, Dict, List, Set, Tuple

logger = logging.getLogger(__name__)

# 核心科学概念的最大词条数（主查询）
PRIMARY_QUERY_MAX_TERMS = 5
# 次要术语最大数（后筛选用）
SECONDARY_TERMS_MAX = 8

# 数据集名称模式（这些应该降权处理）
DATASET_NAME_PATTERNS = frozenset({
    "3dep", "smap", "modis", "landsat", "sentinel", "goes", "viirs",
    "dem", "ndvi", "evi", "lai", "gpm", "trmm", "chirps", "era5",
})

# 技术/方法学词汇（应该降权或排除）
TECHNICAL_TERMS = frozenset({
    "temporal", "spatial", "resolution", "metadata", "acquisition",
    "processing", "algorithm", "product", "band", "spectral",
    "validation", "calibration", "preprocessing", "resampling",
})

# 缺陷类型词汇（应该完全排除）
DEFECT_TERMS = frozenset({
    "undeclared", "unavailable", "dependency", "conflict", "missing",
    "incomplete", "unverified", "defect", "error", "mismatch",
})


class StructuredLiteratureQuery:
    """结构化文献查询构造器"""

    @classmethod
    def build_structured_query(
        cls,
        finding: Dict[str, Any],
        idea: Dict[str, Any],
        dataset_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """构造结构化查询

        Returns:
            {
                "primary_query": str,  # 核心科学概念，用于 OpenAlex search 参数
                "topic_names": List[str],  # 领域关键词，用于 topic filter
                "secondary_terms": List[str],  # 次要术语，用于结果后筛选
                "excluded_terms": List[str],  # 被排除的术语（审计用）
            }
        """
        # 1. 提取所有候选词条
        all_terms = cls._extract_all_terms(finding, idea, dataset_info)

        # 2. 分类：核心概念 vs 次要术语 vs 排除
        primary, secondary, excluded = cls._classify_terms(all_terms)

        # 3. 提取领域锚点作为 topic filter
        topic_names = cls._extract_topic_names(idea, dataset_info)

        # 4. 构造最终查询字符串
        primary_query = " ".join(primary[:PRIMARY_QUERY_MAX_TERMS])

        result = {
            "primary_query": primary_query,
            "topic_names": topic_names,
            "secondary_terms": secondary[:SECONDARY_TERMS_MAX],
            "excluded_terms": excluded,
        }

        logger.info(
            "Structured query: primary=%s, topics=%s, secondary=%d terms, excluded=%d terms",
            primary_query,
            topic_names,
            len(result["secondary_terms"]),
            len(excluded),
        )

        return result

    @classmethod
    def _extract_all_terms(
        cls,
        finding: Dict[str, Any],
        idea: Dict[str, Any],
        dataset_info: Dict[str, Any],
    ) -> List[str]:
        """提取所有候选词条（未分类）"""
        terms: List[str] = []

        # 从 finding 提取
        for field in ("evidence_need", "modification_direction", "question"):
            value = finding.get(field)
            if value:
                terms.extend(cls._tokenize(value))

        # 从 idea 提取核心字段
        for field in ("research_question", "core_hypothesis", "theoretical_mechanism"):
            value = idea.get(field)
            if value:
                terms.extend(cls._tokenize(value))

        # 从 knowledge_anchor 提取
        knowledge_anchor = idea.get("knowledge_anchor_and_motivation")
        if isinstance(knowledge_anchor, dict):
            macro_phenomenon = knowledge_anchor.get("macro_phenomenon")
            if macro_phenomenon:
                terms.extend(cls._tokenize(macro_phenomenon))

        # 去重但保持顺序
        seen: Set[str] = set()
        unique_terms: List[str] = []
        for term in terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)

        return unique_terms

    @classmethod
    def _classify_terms(
        cls,
        terms: List[str],
    ) -> Tuple[List[str], List[str], List[str]]:
        """将词条分类为：核心概念、次要术语、排除

        Returns:
            (primary_terms, secondary_terms, excluded_terms)
        """
        primary: List[str] = []
        secondary: List[str] = []
        excluded: List[str] = []

        for term in terms:
            term_lower = term.lower()

            # 1. 排除：缺陷类型词汇
            if term_lower in DEFECT_TERMS:
                excluded.append(term)
                continue

            # 2. 次要：数据集名称
            if term_lower in DATASET_NAME_PATTERNS:
                secondary.append(term)
                continue

            # 3. 次要：技术方法学词汇
            if term_lower in TECHNICAL_TERMS:
                secondary.append(term)
                continue

            # 4. 核心：长度足够的专业术语（>= 4 个字符）
            if len(term) >= 4:
                primary.append(term)
            # 短词进入次要（可能是缩写）
            elif len(term) >= 2:
                secondary.append(term)
            else:
                excluded.append(term)

        return primary, secondary, excluded

    @classmethod
    def _extract_topic_names(
        cls,
        idea: Dict[str, Any],
        dataset_info: Dict[str, Any],
    ) -> List[str]:
        """提取领域关键词，用于 OpenAlex topic filter

        优先级：
        1. dataset_info.domain（如果是明确的学科领域）
        2. idea.knowledge_anchor.macro_phenomenon（宏观现象）
        """
        topics: List[str] = []

        # 1. 明确的学科领域
        domain = dataset_info.get("domain") if isinstance(dataset_info, dict) else ""
        if domain and isinstance(domain, str):
            domain_clean = domain.strip()
            if domain_clean and len(domain_clean) > 3:
                topics.append(domain_clean)

        # 2. 宏观现象（作为备选）
        knowledge_anchor = idea.get("knowledge_anchor_and_motivation")
        if isinstance(knowledge_anchor, dict):
            macro_phenomenon = knowledge_anchor.get("macro_phenomenon")
            if macro_phenomenon and isinstance(macro_phenomenon, str):
                # 提取前 2-3 个核心词作为 topic
                phenomenon_terms = cls._tokenize(macro_phenomenon)
                # 过滤掉太短和技术词汇
                valid_terms = [
                    term for term in phenomenon_terms
                    if len(term) >= 4 and term.lower() not in TECHNICAL_TERMS
                ]
                topics.extend(valid_terms[:2])

        # 去重
        seen: Set[str] = set()
        unique_topics: List[str] = []
        for topic in topics:
            if topic not in seen:
                seen.add(topic)
                unique_topics.append(topic)

        return unique_topics[:3]  # 最多 3 个 topic

    @staticmethod
    def _tokenize(value: Any) -> List[str]:
        """基础分词，保留字母数字和连字符"""
        if not isinstance(value, str):
            if isinstance(value, dict):
                # 递归处理字典
                tokens: List[str] = []
                for v in value.values():
                    tokens.extend(StructuredLiteratureQuery._tokenize(v))
                return tokens
            return []

        text = value.lower()
        # 提取单词（字母开头，可包含数字、下划线、连字符）
        tokens = re.findall(r"[a-z][a-z0-9_-]*", text)

        # 清理并过滤
        cleaned: List[str] = []
        for token in tokens:
            token = token.strip("_-")
            # 过滤停用词和太短的词
            if len(token) >= 2 and token not in {"the", "and", "for", "with", "from"}:
                cleaned.append(token)

        return cleaned

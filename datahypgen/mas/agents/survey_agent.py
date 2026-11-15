"""
文献检索智能体
本模块实现了文献调查代理，该代理可对研究主题进行全面的文献调查。
该代理生成智能搜索查询，从多个来源检索相关的学术论文，
并根据相关性对论文进行评分，
然后进行深度阅读分析，从优秀论文中提取方法论细节（重点关注文献之间是如何提出假设->为了验证这个假设如何确定需要分析的数据集->结合待验证假设和待分析数据集确定数据分析思路）。
该代理支持带有查询优化功能的自动化迭代文献综述。
重点：这个智能体需要检索与用户问题相似度高的问题的文献，并进行深入阅读分析，提取其中的方法论细节。
"""

import asyncio
import logging
import json
import re
from typing import Dict, Any, List, Optional, Tuple, Union
import os
from .base_agent import BaseAgent, AgentExecutionError
from ..tools.filtered_literature_cache import FilteredLiteratureCache
from ..tools.literature_search import LiteratureSearch, publication_year_is_allowed
from ..tools.utils import PaperMetadata, parse_io_description, format_papers_for_printing_next_query,\
    download_pdf, extract_text_from_pdf, download_pdf_by_doi, select_papers

logger = logging.getLogger(__name__)


class SurveyAgent(BaseAgent):
    """
Survey Agent 针对研究主题进行全面的文献调研。
该代理通过以下方式进行智能文献检索：
- 根据研究主题生成上下文感知的搜索查询
- 从多个学术资源（Semantic Sc​​holar、arxiv）检索论文
- 迭代优化搜索查询以扩大论文覆盖范围
- 根据相关性、创新性和方法学质量对论文进行评分
- 对排名靠前的论文进行深度阅读分析，以提取方法学细节
- 对排名靠前的论文综合分析当前研究的空白
该代理采用迭代搜索策略，从关键词查询开始
并逐步使用论文相似性和基于参考文献的查询
    构建一个全面的文献库。
    """

    _SEARCH_DEPTH_QUERY_LIMITS = {
        "shallow": 2,
        "moderate": 4,
        "deep": 6,
    }

    _DOMAIN_RULES = (
        (
            re.compile(
                r"\b(geomorphology|geomorphometry|geomorphic|terrain|topograph(?:y|ic)|slope|dem)\b"
            ),
            "geomorphology terrain",
        ),
        (
            re.compile(r"\b(wildfire|forest fire|burn scar|burn severity)\b"),
            "wildfire remote sensing",
        ),
        (
            re.compile(
                r"\b(hydrolog(?:y|ic)|drainage|river|flood|evapotranspiration|aridity)\b"
            ),
            "hydrology remote sensing",
        ),
        (
            re.compile(r"\b(land cover|land use|urban|mangrove|vegetation|mining)\b"),
            "land cover remote sensing",
        ),
        (
            re.compile(r"\b(pm2\.?5|air quality|atmospher(?:e|ic))\b"),
            "atmospheric remote sensing",
        ),
    )

    _DOMAIN_ANCHOR_TERMS = {
        "geomorphology terrain": (
            "geomorphology",
            "geomorphometry",
            "geomorphic",
            "terrain",
            "topography",
            "topographic",
        ),
        "wildfire remote sensing": (
            "wildfire",
            "remote sensing",
            "satellite",
            "landsat",
        ),
        "hydrology remote sensing": (
            "hydrology",
            "hydrologic",
            "remote sensing",
            "satellite",
        ),
        "land cover remote sensing": (
            "land cover",
            "land use",
            "remote sensing",
            "satellite",
        ),
        "atmospheric remote sensing": (
            "atmospheric",
            "air quality",
            "remote sensing",
            "satellite",
        ),
        "geospatial remote sensing": (
            "geospatial",
            "remote sensing",
            "earth observation",
            "satellite",
            "landsat",
        ),
    }
    
    def __init__(self, model, config: Dict[str, Any]):
        """
        Initialize the survey agent.
        
        Args:
            model: Language model to use
            config: Debug configuration dictionary
        """
        super().__init__(model, config)
        
        # Load agent-specific configuration
        self.max_papers = config.get("max_papers", 5)
        self.search_depth = config.get("search_depth", "moderate")  # shallow, moderate, deep
        self.sources = config.get("sources", ["arxiv", "nature"])
        # 过滤阈值：保留高分文献进入上下文，其余仅作审计留档
        self.score_threshold = config.get("score_threshold", 5)
        self.paper_top_k = config.get("paper_top_k", 30)
        configured_initial_query_limit = config.get("initial_query_limit")
        self.initial_query_limit = (
            int(configured_initial_query_limit)
            if configured_initial_query_limit is not None
            else None
        )
        self.iterative_query_limit = int(config.get("iterative_query_limit", 1))
        self.max_results_per_query = int(config.get("max_results_per_query", 10))
        self.enable_similar_paper_expansion = bool(config.get("enable_similar_paper_expansion", False))
        self.enable_literature_cache_fallback = bool(config.get("enable_literature_cache_fallback", True))
        self.literature_cache_path = config.get(
            "literature_cache_path",
            os.path.join(
                "experiments",
                "results",
                "novelty_evidence_cache",
                "filtered_literature.json",
            ),
        )
        self._filtered_literature_cache = (
            FilteredLiteratureCache(self.literature_cache_path)
            if self.enable_literature_cache_fallback and self.literature_cache_path
            else None
        )
        
        # Initialize tools
        tools_config = config.get("_global_config", {}).get("tools", {})
        self.literature_search = None
        self.max_publication_year: Optional[int] = None
        self._init_literature_search(tools_config.get("literature_search", {}))

    def _init_literature_search(self, config: Dict[str, Any]) -> None:
        """
        Initialize the literature search tool.
        
        Args:
            config: Literature search configuration
        """
        api_keys = dict(config.get("api_keys", {}))
        if config.get("openalex_key") and not api_keys.get("openalex"):
            api_keys["openalex"] = config.get("openalex_key")
        if config.get("springer_nature_key") and not api_keys.get("springer_nature"):
            api_keys["springer_nature"] = config.get("springer_nature_key")
        
        try:
            self.max_publication_year = config.get("max_publication_year")
            self.literature_search = LiteratureSearch(
                api_keys=api_keys,
                max_publication_year=self.max_publication_year,
            )
            logger.info("Literature search tool initialized successfully")
        except ValueError:
            raise
        except Exception as e:
            logger.warning(f"Failed to initialize literature search: {str(e)}")

    def _openalex_enabled(self) -> bool:
        return "openalex" in {str(source).strip().lower() for source in self.sources}

    def _get_paper_download_credentials(
        self,
        paper: Dict[str, Any],
    ) -> Tuple[Optional[str], Optional[str]]:
        open_access = paper.get("open_access") or {}
        pdf_url = (
            open_access.get("pdf_url")
            or open_access.get("oa_url")
            or paper.get("url")
        )
        doi = paper.get("doi")
        return pdf_url, doi

        
    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行文献检索，返回标准化结果：
        {
            "papers": [...],          # 已筛选文献列表
            "papers_all": [...],      # 原始全集
            "search_queries": [...],
            "aggregated_views": {...},
            "filter_params": {...},
            "raw_count": int
        }
        """
        papers_filtered, papers_all, search_queries, aggregated_views, filter_params = await self.advanced_query_paper(context=context)
        
        return {
            "papers": papers_filtered,
            "papers_all": papers_all,
            "search_queries": search_queries,
            "aggregated_views": aggregated_views,
            "filter_params": filter_params,
            "raw_count": len(papers_all)
        }

    @staticmethod
    def _apply_paper_scores(paper_bank: List[Dict[str, Any]], response: Dict[str, Any]) -> None:
        paper_lookup = {str(paper.get("id", "")): paper for paper in paper_bank}

        for paper_id, score in response.items():
            target_paper = paper_lookup.get(str(paper_id))
            if target_paper is None:
                logger.warning(f"Scored paper id {paper_id!r} not found after domain filtering; skipping score writeback")
                continue
            target_paper["score"] = score

    def _stringify_data_description(self, data_description: Any) -> str:
        if isinstance(data_description, str):
            return data_description
        if isinstance(data_description, (dict, list)):
            return json.dumps(data_description, ensure_ascii=False)
        return str(data_description or "")

    def _normalize_text_field(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (list, tuple, set)):
            parts = [self._normalize_text_field(item) for item in value]
            return "; ".join(part for part in parts if part)
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
        return str(value).strip()

    def _normalize_string_list(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, (list, tuple, set)):
            normalized: List[str] = []
            for item in value:
                normalized.extend(self._normalize_string_list(item))
            return normalized
        if isinstance(value, dict):
            normalized = self._normalize_text_field(value)
            return [normalized] if normalized else []
        normalized = str(value).strip()
        return [normalized] if normalized else []

    def _resolve_search_domain(
        self,
        explicit_domain: Any,
        data_description: Any,
        dataset_portrait: Optional[Dict[str, Any]] = None,
    ) -> str:
        explicit_domain_text = self._normalize_text_field(explicit_domain)
        portrait = dataset_portrait if isinstance(dataset_portrait, dict) else {}
        dataset_evidence_text = " ".join(
            part
            for part in (
                self._stringify_data_description(data_description),
                self._normalize_text_field(portrait.get("portrait_text")),
                self._normalize_text_field(portrait.get("candidate_question_axes")),
                self._normalize_text_field(portrait.get("suggested_research_hooks")),
            )
            if part
        ).lower()

        for evidence_text in (explicit_domain_text.lower(), dataset_evidence_text):
            if not evidence_text:
                continue
            for pattern, canonical_domain in self._DOMAIN_RULES:
                if pattern.search(evidence_text):
                    return canonical_domain

        if explicit_domain_text:
            return explicit_domain_text
        return "geospatial remote sensing"

    # 锚定豁免的通用停用词：这些词不足以证明查询落在某个跨领域方向上。
    _CROSS_DOMAIN_STOPWORDS = frozenset({
        "analysis", "and", "based", "data", "dataset", "effect", "effects",
        "for", "from", "high", "impact", "impacts", "influence", "large",
        "low", "measurement", "model", "modeling", "modelling", "observed",
        "of", "on", "over", "pattern", "patterns", "process", "processes",
        "regional", "remote", "resolution", "role", "satellite", "scale",
        "sensing", "spatial", "study", "the", "using", "variability", "with",
    })

    @classmethod
    def _significant_terms(cls, text: Any) -> set:
        return {
            term
            for term in re.findall(r"[a-z][a-z0-9-]{2,}", str(text or "").lower())
            if term not in cls._CROSS_DOMAIN_STOPWORDS
        }

    def _build_cross_domain_anchor_context(
        self,
        dataset_portrait: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Collect the terms that let a cross-domain query stand in for a domain anchor.

        领域锚定的真实目的是阻止同形异义误召回（"burn severity" 单独出现会命中核物理）。
        跨领域查询若同时带有本数据集的核心变量词与某个跨领域方向的专有词，本身已经
        消歧完成，不应再被拼上领域前缀拽回原领域——那会抵消跨领域推演的全部意义。
        注意核心变量词（如 slope）单独出现仍是歧义的，因此两个条件必须同时满足。
        """
        portrait = dataset_portrait if isinstance(dataset_portrait, dict) else {}
        axes = portrait.get("cross_domain_axes") or []
        if not isinstance(axes, list):
            return {"variable_terms": frozenset(), "direction_terms": ()}

        direction_terms = []
        for axis_record in axes:
            if not isinstance(axis_record, dict):
                continue
            terms = self._significant_terms(axis_record.get("domain_name"))
            terms |= self._significant_terms(
                " ".join(
                    str(term)
                    for term in (axis_record.get("dataset_search_terms") or [])
                )
            )
            if terms:
                direction_terms.append(frozenset(terms))
        if not direction_terms:
            return {"variable_terms": frozenset(), "direction_terms": ()}

        # 核心变量词：expander 要求每条 literature_query 都保留它，因此跨方向的
        # 公共词就是变量词本身；不足两个方向时退回画像文本与数据集名。
        query_term_sets = [
            self._significant_terms((axis or {}).get("literature_query"))
            for axis in axes
            if isinstance(axis, dict) and (axis or {}).get("literature_query")
        ]
        variable_terms: set = set()
        if len(query_term_sets) >= 2:
            variable_terms = set.intersection(*(set(s) for s in query_term_sets))
        if not variable_terms:
            variable_terms = self._significant_terms(
                portrait.get("dataset_name")
            ) | self._significant_terms(portrait.get("primary_variable"))
        return {
            "variable_terms": frozenset(variable_terms),
            "direction_terms": tuple(direction_terms),
        }

    def _query_is_grounded_cross_domain(
        self,
        normalized_query: str,
        cross_domain_context: Optional[Dict[str, Any]],
    ) -> bool:
        if not cross_domain_context:
            return False
        variable_terms = cross_domain_context.get("variable_terms") or frozenset()
        direction_terms = cross_domain_context.get("direction_terms") or ()
        if not variable_terms or not direction_terms:
            return False
        query_terms = self._significant_terms(normalized_query)
        if not query_terms & variable_terms:
            return False
        return any(query_terms & terms for terms in direction_terms)

    def _query_has_domain_anchor(
        self,
        query: str,
        domain: str,
        cross_domain_context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        normalized_query = re.sub(r"\s+", " ", str(query or "").lower()).strip()
        anchor_terms = self._DOMAIN_ANCHOR_TERMS.get(domain)
        if anchor_terms is None:
            anchor_terms = tuple(
                term
                for term in re.findall(r"[a-z][a-z-]{2,}", domain.lower())
                if term not in {"global", "regional", "region", "area", "areas"}
            )
        if any(
            re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", normalized_query)
            for term in anchor_terms
        ):
            return True
        return self._query_is_grounded_cross_domain(
            normalized_query, cross_domain_context
        )

    def _enforce_query_domain_anchor(
        self,
        queries: List[Any],
        domain: str,
        cross_domain_context: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        anchored_queries: List[str] = []
        seen = set()
        for raw_query in queries:
            query = re.sub(r"\s+", " ", str(raw_query or "")).strip()
            if not query:
                continue
            if not self._query_has_domain_anchor(query, domain, cross_domain_context):
                query = f"{domain} {query}"
                logger.info("Added required literature-query domain anchor: %s", query)
            elif self._query_is_grounded_cross_domain(query.lower(), cross_domain_context):
                logger.info(
                    "Kept cross-domain literature query without domain prefix: %s", query
                )
            normalized = query.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            anchored_queries.append(query)
        return anchored_queries

    def _resolve_initial_query_limit(self, search_depth: str) -> int:
        if self.initial_query_limit is not None:
            return max(0, self.initial_query_limit)
        return self._SEARCH_DEPTH_QUERY_LIMITS.get(search_depth, 4)

    def _split_semantic_items(self, value: Any) -> List[str]:
        normalized_text = self._normalize_text_field(value)
        if not normalized_text:
            return []
        return [
            part.strip()
            for part in re.split(r"[;\n]+", normalized_text)
            if part and part.strip()
        ]

    def _normalize_paper_detail_fields(self, paper: Dict[str, Any]) -> None:
        if not isinstance(paper, dict):
            return
        paper["methods"] = self._normalize_text_field(paper.get("methods"))
        paper["applications"] = self._normalize_text_field(paper.get("applications"))
        paper["datasets_used"] = self._normalize_string_list(paper.get("datasets_used"))

    def _paper_identity_key(self, paper: Dict[str, Any]) -> str:
        doi = self._normalize_text_field(paper.get("doi")).lower()
        if doi:
            return f"doi:{doi}"
        title = re.sub(r"\s+", " ", self._normalize_text_field(paper.get("title")).lower())
        return f"title:{title}" if title else ""

    def _extract_cache_match_terms(self, *values: Any) -> List[str]:
        text = " ".join(self._normalize_text_field(value).lower() for value in values)
        tokens = re.findall(r"[a-z][a-z0-9_-]{2,}", text)
        stopwords = {
            "the", "and", "for", "with", "from", "that", "this", "using", "into",
            "about", "analysis", "study", "research", "dataset", "data", "task",
            "description", "global", "regional", "method", "methods", "paper",
            "papers", "trend", "trends",
        }
        terms = [token for token in tokens if token not in stopwords]
        return list(dict.fromkeys(terms))[:40]

    def _score_cached_literature_paper(self, paper: Dict[str, Any], terms: List[str]) -> int:
        if not terms:
            return 0
        haystack = " ".join(
            self._normalize_text_field(paper.get(field)).lower()
            for field in ("title", "abstract", "journal", "query")
        )
        return sum(1 for term in terms if term in haystack)

    def _load_literature_cache_papers(self) -> List[Dict[str, Any]]:
        if self._filtered_literature_cache is None:
            return []
        papers = self._filtered_literature_cache.load()

        normalized: List[Dict[str, Any]] = []
        for paper in papers:
            if not isinstance(paper, dict):
                continue
            if not publication_year_is_allowed(
                paper.get("year"),
                self.max_publication_year,
            ):
                continue
            title = self._normalize_text_field(paper.get("title"))
            abstract = self._normalize_text_field(paper.get("abstract"))
            if not title:
                continue
            cached_paper = dict(paper)
            cached_paper["title"] = title
            cached_paper["abstract"] = abstract
            cached_paper["source"] = cached_paper.get("source") or "literature_cache"
            cached_paper["retrieval_fallback"] = "filtered_literature_cache"
            normalized.append(cached_paper)
        return normalized

    def _backfill_from_literature_cache(
        self,
        paper_bank: List[Dict[str, Any]],
        target_count: int,
        *,
        domain: str,
        task_description: str,
        data_description: str,
        search_queries: List[str],
    ) -> List[Dict[str, Any]]:
        if len(paper_bank) >= target_count:
            return paper_bank

        cached_papers = self._load_literature_cache_papers()
        if not cached_papers:
            return paper_bank

        terms = self._extract_cache_match_terms(
            domain,
            task_description,
            data_description,
            " ".join(search_queries),
        )
        existing_keys = {
            key
            for key in (self._paper_identity_key(paper) for paper in paper_bank)
            if key
        }

        ranked_candidates: List[Tuple[int, Dict[str, Any]]] = []
        for cached_paper in cached_papers:
            key = self._paper_identity_key(cached_paper)
            if not key or key in existing_keys:
                continue
            match_score = self._score_cached_literature_paper(cached_paper, terms)
            if match_score <= 0:
                continue
            ranked_candidates.append((match_score, cached_paper))

        ranked_candidates.sort(
            key=lambda item: (
                item[0],
                item[1].get("score", float("-inf")),
                item[1].get("year") or 0,
            ),
            reverse=True,
        )

        backfilled = paper_bank[:]
        next_id = len(backfilled)
        for match_score, cached_paper in ranked_candidates:
            if len(backfilled) >= target_count:
                break
            key = self._paper_identity_key(cached_paper)
            if not key or key in existing_keys:
                continue
            paper = dict(cached_paper)
            paper["id"] = str(next_id)
            paper["cache_match_score"] = match_score
            paper["domain_filter_fallback"] = "filtered_literature_cache"
            backfilled.append(paper)
            existing_keys.add(key)
            next_id += 1

        added = len(backfilled) - len(paper_bank)
        if added:
            logger.info(
                "Literature cache fallback added %d papers from %s; total=%d, target=%d",
                added,
                self.literature_cache_path,
                len(backfilled),
                target_count,
            )
        else:
            logger.info("Literature cache fallback found no matching papers")
        return backfilled

    def _resolve_dataset_portrait(
        self,
        context: Dict[str, Any],
        data_description: Any,
    ) -> Dict[str, Any]:
        portrait = context.get("dataset_portrait", {})
        if isinstance(portrait, dict) and portrait:
            return portrait
        return {}

    def _build_portrait_query_constraints(self, dataset_portrait: Dict[str, Any]) -> str:
        if not isinstance(dataset_portrait, dict) or not dataset_portrait:
            return ""

        axes = dataset_portrait.get("candidate_question_axes", []) or []
        hooks = dataset_portrait.get("suggested_research_hooks", []) or []
        portrait_text = str(dataset_portrait.get("portrait_text", "") or "").strip()
        cross_domain_axes = dataset_portrait.get("cross_domain_axes", []) or []

        lines = ["Dataset Portrait Query Hints:"]
        if portrait_text:
            lines.append(f"- Portrait summary: {portrait_text}")
        if axes:
            lines.append(f"- Candidate question axes to prioritize: {axes[:4]}")
        if hooks:
            lines.append(f"- Data-driven research hooks to use as query seeds: {hooks[:4]}")
        if cross_domain_axes:
            lines.append(
                "- Cross-domain exploration seeds (dedicate up to half of the queries to "
                "these directions; keep the dataset's core variable term in each):"
            )
            for axis_record in cross_domain_axes[:6]:
                if not isinstance(axis_record, dict):
                    continue
                literature_query = str(axis_record.get("literature_query") or "").strip()
                domain_name = str(axis_record.get("domain_name") or "").strip()
                if literature_query:
                    lines.append(f"  * [{domain_name}] {literature_query}")
        lines.append("- Prefer papers whose methods can inform the portrait summary, question axes, or research hooks above.")
        return "\n".join(lines)

    def _build_portrait_scoring_constraints(self, dataset_portrait: Dict[str, Any]) -> str:
        if not isinstance(dataset_portrait, dict) or not dataset_portrait:
            return ""

        axes = dataset_portrait.get("candidate_question_axes", []) or []
        hooks = dataset_portrait.get("suggested_research_hooks", []) or []
        cross_domain_axes = dataset_portrait.get("cross_domain_axes", []) or []

        lines = ["Dataset Portrait Scoring Hints:"]
        if axes:
            lines.append(f"- Reward papers aligned with these candidate question axes: {axes[:4]}")
        if hooks:
            lines.append(f"- Reward papers that provide methodological support for these data-driven hooks: {hooks[:4]}")
        if cross_domain_axes:
            lines.append(
                "- Also reward papers from cross-domain directions listed below — "
                "these bring methodological diversity and should NOT be penalised for "
                "being outside the dataset's primary field:"
            )
            for axis_record in cross_domain_axes[:6]:
                if not isinstance(axis_record, dict):
                    continue
                domain_name = str(axis_record.get("domain_name") or "").strip()
                role = str(axis_record.get("role") or "").strip()
                lit_q = str(axis_record.get("literature_query") or "").strip()
                if domain_name:
                    hint = f"  * [{domain_name}]"
                    if role:
                        hint += f" role={role}"
                    if lit_q:
                        hint += f": {lit_q}"
                    lines.append(hint)
        lines.append("- Prefer papers whose methods are useful for turning the dataset portrait into researchable questions.")
        return "\n".join(lines)

    async def _llm_generate_initial_queries(
        self,
        task_description: str,
        data_description: str,
        domain: str,
        search_depth: str,
        dataset_portrait: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        使用 LLM 直接基于 task_description / data_description 生成初始检索关键词列表。
        根据 search_depth 控制生成数量（shallow:2, moderate:4, deep:6）。
        若 LLM 调用失败，回退到原始输入文本查询。
        """
        if search_depth == "shallow":
            num_queries = 2
            depth_instruction = (
                "Generate 1-2 focused, specific queries targeting the most central aspect."
            )
        elif search_depth == "deep":
            num_queries = 6
            depth_instruction = (
                "Generate 4-6 diverse queries covering different methods, approaches, "
                "and technical aspects related to the task and dataset."
            )
        else:  # moderate
            num_queries = 4
            depth_instruction = (
                "Generate 2-4 balanced queries covering the main technical aspects."
            )

        domain = self._resolve_search_domain(
            domain,
            data_description,
            dataset_portrait,
        )
        data_description_text = self._stringify_data_description(data_description)
        portrait_query_constraints = self._build_portrait_query_constraints(dataset_portrait or {})
        cross_domain_context = self._build_cross_domain_anchor_context(dataset_portrait)

        if task_description:
            query_context = f"Task: {task_description}\nDataset: {data_description_text}"
            task_instruction = (
                "Generate effective search queries to find scientific literature that provides "
                "methods or approaches relevant to the task and dataset described above. "
            )
            system_prompt = (
                "You are a scientific literature search specialist.\n"
                "Your task is to formulate effective search queries for academic databases "
                "based on a task description and dataset information.\n\n"
                "Guidelines:\n"
                "- Create queries using scientific terminology likely to appear in research papers.\n"
                "- Ground every query in the specific technical content of the task and dataset "
                "  (e.g., sensor names, data types, spatial resolution, geographic region, "
                "  analytical method, target variable).\n"
                "- Use Boolean operators (AND, OR) when helpful.\n"
                "- Be specific enough to find relevant papers without being so narrow that "
                "  important evidence is missed.\n"
                "- Do NOT use generic filler phrases such as 'real-world applications', "
                "  'case studies', or 'events' as standalone query terms.\n"
                "- Generate the majority of queries grounded in the dataset's own domain.\n"
                "- If cross-domain exploration seeds are supplied below, dedicate up to half "
                "  of the queries to those directions; each cross-domain query MUST still keep "
                "  the dataset's core variable term so it stays grounded in this dataset.\n"
                "- Do NOT introduce topics that are neither in the task/dataset description "
                "  nor in the supplied cross-domain seeds.\n"
                f"- IMPORTANT: Every query MUST include at least one domain-specific anchor term "
                f"  (derived from the domain '{domain}' and dataset) to prevent ambiguous matches "
                f"  from unrelated fields. For example, a term like 'burn' or 'mosaic' without "
                f"  domain context may match physics or biology papers — always pair such terms "
                f"  with domain qualifiers."
            )
        else:
            query_context = f"Dataset: {data_description_text}"
            task_instruction = (
                "Generate effective search queries to find scientific literature that analyzes "
                "or applies datasets similar to the one described above. "
            )
            system_prompt = (
                "You are a scientific literature search specialist.\n"
                "Your task is to formulate effective search queries for academic databases "
                "based on a dataset description.\n\n"
                "Guidelines:\n"
                "- Extract specific technical terms from the dataset description: sensor or "
                "  product names, data modality, spatial/temporal resolution, geographic focus, "
                "  target variable, time period.\n"
                "- Build queries around those concrete terms, NOT generic phrases.\n"
                "- Use Boolean operators (AND, OR) when helpful.\n"
                "- Do NOT use generic phrases like 'real-world applications', 'case studies', "
                "  'historical events', or 'urban changes' as standalone query terms.\n"
                "- Generate the majority of queries grounded in the dataset's own domain.\n"
                "- If cross-domain exploration seeds are supplied below, dedicate up to half "
                "  of the queries to those directions; each cross-domain query MUST still keep "
                "  the dataset's core variable term so it stays grounded in this dataset.\n"
                "- Do NOT introduce topics that are neither in the dataset description "
                "  nor in the supplied cross-domain seeds.\n"
                f"- IMPORTANT: Every query MUST include at least one domain-specific anchor term "
                f"  (derived from the domain '{domain}' and dataset) to prevent ambiguous matches "
                f"  from unrelated fields. Pair potentially ambiguous terms with domain qualifiers."
            )

        output_schema = {
            "type": "object",
            "properties": {
                "search_queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of plain-keyword search query strings (2-6 words each) "
                        "for academic databases. Do not wrap in KeywordQuery()."
                    ),
                }
            },
            "required": ["search_queries"],
        }

        prompt = (
            f"{query_context}\n\n"
            f"# Task\n{task_instruction}"
            f"{depth_instruction}\n\n"
            "Rules:\n"
            "- Each query must be a plain keyword string (2-6 words), not wrapped in any format.\n"
            "- Every query must be grounded in the specific content above.\n"
            "- Avoid repeating the same concept across multiple queries; maximise coverage.\n"
            f"- Every query MUST contain at least one domain-specific anchor term from the domain "
            f"'{domain}' or dataset to prevent off-domain ambiguous matches. "
            f"Do NOT generate queries that consist solely of terms that are common across multiple "
            f"unrelated scientific disciplines (e.g., 'burn severity' alone matches nuclear physics; "
            f"pair it with domain terms like 'wildfire', 'remote sensing', or 'Landsat')."
        )

        if portrait_query_constraints:
            system_prompt += "\n\n" + portrait_query_constraints
            prompt += "\n\n" + portrait_query_constraints

        try:
            response = await self._call_model(
                prompt=prompt,
                system_prompt=system_prompt,
                schema=output_schema,
            )
            raw_queries = response.get("search_queries", [])
            queries = self._enforce_query_domain_anchor(
                raw_queries, domain, cross_domain_context
            )[:num_queries]
            if queries:
                logger.info(
                    f"LLM generated {len(queries)} initial queries "
                    f"(search_depth={search_depth}): {queries}"
                )
                return queries
        except Exception as e:
            logger.warning(
                f"LLM initial query generation failed ({e}); "
                "falling back to raw-text query."
            )

        # 直接从用户原始输入文本截取前 8 个词作为查询
        fallback_text = (task_description or data_description_text or domain or "").strip()
        if fallback_text:
            words = fallback_text.split()[:8]
            fallback_query = " ".join(words)
            fallback_query = self._enforce_query_domain_anchor([fallback_query], domain)[0]
            logger.info(f"Raw-text fallback initial query: {fallback_query}")
            return [fallback_query]
        return []

    async def advanced_query_paper(self, context) -> tuple:
        """

        Returns:
            tuple: (
                papers_filtered,
                papers_all,
                search_queries,
                aggregated_views,
                filter_params
            )
        """
        def parse_query(query_str: str) -> tuple:
            """Parse query string to extract type and content."""
            query_str = (query_str or "").strip()
            keyword_match = re.match(r'KeywordQuery\(\s*["\'](.+?)["\']\s*\)$', query_str)
            if keyword_match:
                return "keyword", keyword_match.group(1).strip()

            paper_match = re.match(r'PaperQuery\(\s*["\'](.+?)["\']\s*\)$', query_str)
            if paper_match:
                return "paper", paper_match.group(1).strip()

            if query_str.startswith("KeywordQuery"):
                return "keyword", query_str[len("KeywordQuery"):].strip("() ").strip('"\'')
            if query_str.startswith("PaperQuery"):
                return "paper", query_str[len("PaperQuery"):].strip("() ").strip('"\'')
            return "keyword", query_str

        search_queries = []
        seen_queries = set()
        used_keyword_queries = set()

        def append_query_if_new(query: str) -> bool:
            normalized = (query or "").strip().lower()
            if not normalized or normalized in seen_queries:
                return False
            seen_queries.add(normalized)
            search_queries.append(query.strip())
            return True
        task_description = (
            context.get("description")
            or (context.get("goal") or {}).get("description", "")
            or ""
        )
        raw_data_description = context.get("data_description", "")
        dataset_portrait = self._resolve_dataset_portrait(context, raw_data_description)
        domain = self._resolve_search_domain(
            context.get("domain", ""),
            raw_data_description,
            dataset_portrait,
        )
        data_description_text = self._stringify_data_description(raw_data_description)
        data_description = data_description_text
        portrait_query_constraints = self._build_portrait_query_constraints(dataset_portrait)
        portrait_scoring_constraints = self._build_portrait_scoring_constraints(dataset_portrait)
        cross_domain_context = self._build_cross_domain_anchor_context(dataset_portrait)
        search_depth = context.get("search_depth", self.search_depth)
        output_schema_paper_score={
            "type": "object",
            "properties": {
                "^[a-zA-Z0-9_]+$": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": 10
                }
            },
            "description": "A dictionary where each key is a paperID and each value is a score between 1 and 10."
        }
        # 统一论文详情schema，无论是否有任务描述都提取全部字段
        output_schema_paper_details = {
            "type": "object",
            "properties": {
                "background": {
                    "type": "string",
                    "description": "Core problem context and motivation"
                },
                "contributions": {
                    "type": "string",
                    "description": "Novel contributions or key insights derived from the analysis"
                },
                "methods": {
                    "type": "string",
                    "description": "Methodological framework and analytical reasoning approach (conceptual/logical level, not tool names)"
                },
                "applications": {
                    "type": "string",
                    "description": "Specific real-world events, case studies, or scenarios where the dataset was applied (e.g., specific floods, urban changes, historical events)"
                },
                "datasets_used": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Product families, data modalities, or data source names explicitly mentioned in the paper (e.g., MTBS, MODIS NDVI, ERA5, Landsat)"
                }
            },
            "required": ["background", "contributions", "methods", "applications", "datasets_used"]
        }
        
        ###
        # 根据是否有任务描述，决定是否定义 I/O（供后续论文评分 prompt 使用）
        # 有任务描述时：定义 I/O 以匹配方法
        # 无任务描述时：不需要 I/O，关注数据集应用场景
        if task_description:
            define_task_attribute_prompt = f"""
                You are a Senior Geography Scientist.
                Target Research Domain: "{domain}"
                Task Description: "{task_description}"
                Dataset Description: "{data_description}"

                Before answering, internally consider:
                - Data modality: Raster / Vector / PointCloud / TimeSeries / Text / Graph
                - Core geo-analytical task type (e.g., segmentation, change detection, interpolation)

                Input: <...>
                Output: <...>

                Rules:
                - Output exactly 2 lines starting with "Input:" and "Output:".
                - No extra text.
            """
            try:
                response = await self._call_model(prompt=define_task_attribute_prompt)
                io_description = parse_io_description(response)  # -> (input_str, output_str)
                logger.info(f"Defined Geo I/O: {io_description}")
            except Exception as e:
                logger.error(f"Error defining task attribute: {str(e)}")
                io_description = ("Geography Data", "Analysis Result")
        else:
            # 无任务描述时，不需要定义 I/O，关注数据集的应用场景而非方法
            io_description = None
            logger.info("No task description provided, skipping I/O definition (focus on dataset applications)")

        # 初始检索词由 LLM 直接基于 task_description / data_description 生成，
        # 按 search_depth 控制数量（shallow:2, moderate:4, deep:6）。
        initial_query_strings = await self._llm_generate_initial_queries(
            task_description=task_description,
            data_description=data_description,
            domain=domain or "",
            search_depth=search_depth,
            dataset_portrait=dataset_portrait,
        )

        initial_query_strings = self._enforce_query_domain_anchor(
            initial_query_strings,
            domain,
            cross_domain_context,
        )[:self._resolve_initial_query_limit(search_depth)]

        # 对每条初始查询执行检索，结果去重后并入 paper_bank
        paper_bank: dict = {}
        for raw_q in initial_query_strings:
            formatted = f'KeywordQuery("{raw_q}")'
            append_query_if_new(formatted)
            query_type, query_content = parse_query(formatted)
            try:
                if query_type == "keyword":
                    init_paper_lst = await self.literature_search.multi_source_search(
                        query=query_content,
                        sources=self.sources,
                        max_results=self.max_results_per_query,
                    )
                elif query_type == "paper":
                    if self.enable_similar_paper_expansion and self._openalex_enabled():
                        similar_papers = await self.literature_search.search_similar_papers(
                            paper_id=query_content,
                            max_results=self.max_results_per_query,
                        )
                        init_paper_lst = {"openalex": similar_papers}
                    else:
                        logger.info(
                            "Skipping OpenAlex similar-paper expansion because it is disabled or openalex is not in sources"
                        )
                        init_paper_lst = {}
                else:
                    init_paper_lst = await self.literature_search.multi_source_search(
                        query=query_content,
                        sources=self.sources,
                        max_results=self.max_results_per_query,
                    )
            except Exception as e:
                logger.error(f"Initial query search failed for '{query_content}': {e}")
                continue

            if init_paper_lst:
                existing_titles = {p["title"] for p in paper_bank.values()}
                for source, papers in init_paper_lst.items():
                    for paper in papers:
                        paper_dict = paper.to_dict()
                        paper_dict["source"] = source
                        if paper_dict["title"] not in existing_titles:
                            paper_bank[str(len(paper_bank))] = paper_dict
                            existing_titles.add(paper_dict["title"])

        if not paper_bank:
            logger.warning("No papers found for any initial query.")

        logger.info(
            f"Initial paper bank size after {len(initial_query_strings)} LLM queries: "
            f"{len(paper_bank)}"
        )

        # make advanced query — 迭代扩展检索
        # 根据有/无 task_description 分别定义迭代扩展的 system_prompt
        if task_description:
            iter_system_prompt = (
                "You are a scientific literature search specialist helping to expand a paper collection.\n"
                "Your goal is to find more papers whose methods or approaches are relevant to the given task and dataset.\n\n"
                "Rules:\n"
                "- Ground every new query in the specific task description and dataset characteristics "
                "  (sensor type, data modality, target variable, geographic region).\n"
                "- Diversify from previous queries: cover different sub-topics, methods, or angles.\n"
                "- Keep keyword queries short (2-6 words); avoid overly generic phrases.\n"
                "- Do NOT repeat queries already asked.\n"
                f"- IMPORTANT: Every query MUST include at least one domain-specific anchor term "
                f"  from the domain '{domain}' to prevent off-domain ambiguous matches. "
                f"  Ambiguous terms (e.g., 'burn', 'mosaic', 'severity') MUST be paired with "
                f"  domain qualifiers (e.g., 'wildfire', 'remote sensing', 'land cover').\n"
                "- Output ONLY the query string in KeywordQuery(\"...\") or PaperQuery(\"...\") format, "
                "  with no additional text."
            )
        else:
            iter_system_prompt = (
                "You are a scientific literature search specialist helping to expand a paper collection.\n"
                "Your goal is to find more papers that analyze or apply datasets similar to the one described.\n\n"
                "Rules:\n"
                "- Ground every new query in the specific technical attributes of the dataset description: "
                "  sensor or product name, data type, spatial/temporal resolution, geographic focus, "
                "  target variable, time period.\n"
                "- Diversify from previous queries: explore different methodological angles, "
                "  geographic sub-regions, or temporal windows mentioned in the description.\n"
                "- Keep keyword queries short (2-6 words); do NOT use generic phrases like "
                "  'real-world applications', 'case studies', 'events', or 'historical events'.\n"
                "- Do NOT repeat queries already asked.\n"
                "- Output ONLY the query string in KeywordQuery(\"...\") or PaperQuery(\"...\") format, "
                "  with no additional text."
            )

        if portrait_query_constraints:
            iter_system_prompt += "\n\n" + portrait_query_constraints

        grounding_k = 10
        iteration = 0
        while len(paper_bank) < self.max_papers and iteration < self.iterative_query_limit:
            ## select the top k papers with highest scores for grounding
            data_list = [{'id': id, **info} for id, info in paper_bank.items()]
            scored_papers = [p for p in data_list if 'score' in p]
            unscored_papers = [p for p in data_list if 'score' not in p]
            sorted_papers = sorted(scored_papers, key=lambda x: x['score'], reverse=True) + unscored_papers
            grounding_papers = sorted_papers[:grounding_k]
            grounding_papers_str = format_papers_for_printing_next_query(
                grounding_papers,
                include_abstract=False,
                include_score=False,
                include_id=True
            )
            # 根据是否有任务描述，调整新查询生成的重点
            if task_description:
                # 有任务描述：关注方法匹配（I/O）
                if io_description is not None:
                    query_prompt = (
                        f"Task: {task_description}\n"
                        f"Dataset: {data_description}\n"
                        f"Target I/O — input: {io_description[0]}, output: {io_description[1]}\n\n"
                        "Query formats:\n"
                        "(1) KeywordQuery(\"keyword\"): find papers matching the keyword "
                        "(2-6 words, not too specific).\n"
                        "(2) PaperQuery(\"paperId\"): find papers similar to the given paper.\n\n"
                        f"Already collected papers:\n{grounding_papers_str}\n\n"
                        f"Already asked queries:\n{search_queries}\n\n"
                        "Propose ONE new query that expands coverage with diverse, relevant papers. "
                        "Output only the query:"
                    )
                else:
                    query_prompt = (
                        f"Task: {task_description}\n"
                        f"Dataset: {data_description}\n\n"
                        "Query formats:\n"
                        "(1) KeywordQuery(\"keyword\"): find papers matching the keyword "
                        "(2-6 words, not too specific).\n"
                        "(2) PaperQuery(\"paperId\"): find papers similar to the given paper.\n\n"
                        f"Already collected papers:\n{grounding_papers_str}\n\n"
                        f"Already asked queries:\n{search_queries}\n\n"
                        "Propose ONE new query that expands coverage with diverse, relevant papers. "
                        "Output only the query:"
                    )
            else:
                # 无任务描述：围绕 data_description 的具体技术特征展开，不使用通用示例词
                query_prompt = (
                    f"Dataset description: {data_description}\n\n"
                    "Query formats:\n"
                    "(1) KeywordQuery(\"keyword\"): find papers matching the keyword "
                    "(2-6 words drawn from the dataset description above, not generic phrases).\n"
                    "(2) PaperQuery(\"paperId\"): find papers similar to the given paper.\n\n"
                    f"Already collected papers:\n{grounding_papers_str}\n\n"
                    f"Already asked queries:\n{search_queries}\n\n"
                    "Propose ONE new query grounded in the specific dataset attributes above "
                    "(e.g., sensor name, data type, resolution, geographic region, target variable). "
                    "Output only the query:"
                )
            if portrait_query_constraints:
                query_prompt += "\n\n" + portrait_query_constraints

            try:
                response = await self._call_model(
                    prompt=query_prompt,
                    system_prompt=iter_system_prompt,
                )
                raw_query = response
                query_type, query_content = parse_query(raw_query)
                if query_type == "keyword":
                    candidate = (query_content or "").replace('"', " ").strip()
                    anchored_candidates = self._enforce_query_domain_anchor(
                        [candidate], domain, cross_domain_context
                    )
                    if not anchored_candidates:
                        raise AgentExecutionError("Generated an empty literature keyword query")
                    candidate = anchored_candidates[0]
                    dedup_suffixes = ["remote sensing", "process analysis", "spatiotemporal"]
                    dedup_attempt = 0
                    while candidate.lower() in used_keyword_queries and dedup_attempt < 3:
                        candidate = f"{candidate} {dedup_suffixes[dedup_attempt]}"
                        dedup_attempt += 1
                    candidate = re.sub(r"\s+", " ", candidate).strip()
                    used_keyword_queries.add(candidate.lower())
                    query = f'KeywordQuery("{candidate}")'
                elif query_type == "paper":
                    query = f'PaperQuery("{query_content}")'
                else:
                    query = f'KeywordQuery("{query_content}")'

                if not append_query_if_new(query):
                    logger.info(f"Duplicate query skipped: {query}")
                    iteration += 1
                    continue
            except Exception as e:
                logger.error(f"Error generating new query: {str(e)}")
                raise AgentExecutionError("Failed to generate new query")
            
            new_paper_lst = {}
            try:
                logger.info(f"Searching new query {query}")
                query_type, query_content = parse_query(query)
                if query_type == "keyword":
                    new_paper_lst = await self.literature_search.multi_source_search(
                        query=query_content,
                        sources=self.sources,
                        max_results=self.max_results_per_query
                    )
                elif query_type == "paper":
                    if self.enable_similar_paper_expansion and self._openalex_enabled():
                        similar_papers = await self.literature_search.search_similar_papers(
                            paper_id=query_content,
                            max_results=self.max_results_per_query
                        )
                        new_paper_lst = {"openalex": similar_papers}
                    else:
                        logger.info(
                            "Skipping OpenAlex similar-paper expansion because it is disabled or openalex is not in sources"
                        )
                        new_paper_lst = {}
                else:
                    # Fallback to keyword search
                    new_paper_lst = await self.literature_search.multi_source_search(
                        query=query_content,
                        sources=self.sources,
                        max_results=self.max_results_per_query
                    )
            except Exception as e:
                logger.error(f"survey error: {e}")
            
            if new_paper_lst:
                flattened_papers = []
                for source, papers in new_paper_lst.items():
                    for paper in papers:
                        # Convert PaperMetadata to dict
                        paper_dict = paper.to_dict()
                        paper_dict['source'] = source  # Add source info
                        flattened_papers.append(paper_dict)
                existing_titles = {paper['title'] for paper in paper_bank.values()}
                new_papers = [paper for paper in flattened_papers if paper['title'] not in existing_titles]
                logger.info(f"Size of new_papers after filtering: {len(new_papers)}")
                if new_papers:
                    # Assign new unique indices to new papers
                    start_index = len(paper_bank)
                    new_paper_bank = {str(start_index + i): paper for i, paper in enumerate(new_papers)}

                    # Update paper_bank with new papers
                    paper_bank.update(new_paper_bank)
                else:
                    logger.info("No NEW papers found for the query")
            else:
                logger.info("No papers found for the query")
            
            iteration += 1
        
        data_list = [{'id': id, **info} for id, info in paper_bank.items()]
        paper_bank = [
            paper
            for paper in data_list
            if publication_year_is_allowed(
                paper.get("year"),
                self.max_publication_year,
            )
        ]

        # ── 领域过滤：评分前剔除明显跨域噪声论文 ──────────────────────────────
        # 基于 title + abstract 批量判断论文是否属于目标领域，
        # 避免多义关键词（如 "mosaic"、"burn"、"severity"）召回天文、核物理、医学等跨域文献。
        paper_bank = await self._filter_off_domain_papers(
            paper_bank=paper_bank,
            domain=domain,
            task_description=task_description,
            data_description=data_description,
            dataset_portrait=dataset_portrait,
        )
        logger.info(f"After domain filter: {len(paper_bank)} papers remain")
        fallback_target_count = self.max_papers
        if self.paper_top_k and self.paper_top_k > 0:
            fallback_target_count = min(fallback_target_count, self.paper_top_k)
        fallback_target_count = max(fallback_target_count, self.max_results_per_query)
        paper_bank = self._backfill_from_literature_cache(
            paper_bank=paper_bank,
            target_count=fallback_target_count,
            domain=domain,
            task_description=task_description,
            data_description=data_description,
            search_queries=search_queries,
        )
        # ────────────────────────────────────────────────────────────────────────

        BATCH_SIZE = 10
        task_name = context.get("task_name", "default_task")
        session_id = context.get("session_id", "unknown_session")

        async def score_batch(batch_index: int, batch: List[Dict[str, Any]]) -> None:
            abs_batch = [{'id': paper['id'], 'title': paper['title'], 'abstract': paper['abstract']} for paper in batch]
            # 根据是否有任务描述，调整论文评分标准
            if task_description:
                # 有任务描述：关注方法相关性和I/O匹配
                if io_description is not None:
                    paper_score_prompt = f"You are a helpful literature review assistant whose job is to read the below set of papers and score each paper.\n Task: {task_description}\n Dataset: {data_description}\n The criteria for scoring are: \n (1) The paper is directly relevant to the topic of: {domain}. Note that it should be specific to solve the problem of focus, rather than just generic methods. \n (2) The input and output of the proposed method in this paper is same with input: {io_description[0]}, output: {io_description[1]}. Note that if the input and output are not match, the paper should get a low score. \n (3) The paper proposes a clear methodological framework — a reasoning strategy or analytical paradigm (e.g., spatially-aware ensemble learning, time-series trend analysis, spatial hotspot detection) — NOT merely a list of tools. Papers that only describe a pipeline of off-the-shelf libraries without a novel analytical idea should get a lower score. \n (4) The paper is interesting, meaningful, and has potential to inspire new analysis approaches for the given task. \n The papers are: \n {abs_batch} \n Please score each paper from 1 to 10. \n Write the response in JSON format with \"paperID: score\" as the key and value for each paper. \n\n ONLY output the JSON dict with NO additional text. DO NOT output newline characters. DO NOT output any markdown modifier so that we can call json.loads() on the output later."
                else:
                    paper_score_prompt = f"You are a helpful literature review assistant whose job is to read the below set of papers and score each paper.\n Task: {task_description}\n Dataset: {data_description}\n The criteria for scoring are: \n (1) The paper is directly relevant to the topic of: {domain}. Note that it should be specific to solve the problem of focus, rather than just generic methods. \n (2) The paper proposes a clear methodological framework — a reasoning strategy or analytical paradigm (e.g., spatially-aware ensemble learning, time-series trend analysis, spatial hotspot detection) — NOT merely a list of tools. Papers that only describe a pipeline of off-the-shelf libraries without a novel analytical idea should get a lower score. \n (3) The paper is interesting, meaningful, and has potential to inspire new analysis approaches for the given task. \n The papers are: \n {abs_batch} \n Please score each paper from 1 to 10. \n MUST Write the response in JSON format with \"paperID: score\" as the key and value for each paper. \n\n ONLY output the JSON dict with NO additional text. DO NOT output newline characters. DO NOT output any markdown modifier so that we can call json.loads() on the output later."
            else:
                # 无任务描述：关注数据集应用场景的相关性
                paper_score_prompt = f"You are a helpful literature review assistant whose job is to read the below set of papers and score each paper.\n Dataset Description: {data_description}\n The criteria for scoring are: \n (1) The paper uses datasets similar to the provided dataset description. Focus on papers that applied similar data types, resolutions, or temporal characteristics. \n (2) The paper describes specific real-world applications, case studies, or events where the dataset was used (e.g., specific floods, wildfires, urban changes, historical events). Papers that only describe generic methods without specific applications should get lower scores. \n (3) The paper provides concrete examples of how the dataset was applied to analyze actual events or scenarios, rather than just theoretical discussions. \n (4) The paper is interesting and provides insights into how similar datasets can be used for practical analysis. \n The papers are: \n {abs_batch} \n Please score each paper from 1 to 10. \n MUST Write the response in JSON format with \"paperID: score\" as the key and value for each paper. \n\n ONLY output the JSON dict with NO additional text. DO NOT output newline characters. DO NOT output any markdown modifier so that we can call json.loads() on the output later."

            if portrait_scoring_constraints:
                paper_score_prompt += f"\n\n{portrait_scoring_constraints}"

            try:
                response = await self._call_model(
                    prompt=paper_score_prompt,
                    schema=output_schema_paper_score
                )
                self._apply_paper_scores(paper_bank, response)
            except Exception as e:
                logger.warning(
                    f"Failed to score papers at batch starting index {batch_index}, "
                    f"batch_size={len(batch)}: {e}. "
                    "Skipping score writeback for this batch; papers will be ranked last."
                )

        await asyncio.gather(*[
            score_batch(batch_index, paper_bank[batch_index:batch_index + BATCH_SIZE])
            for batch_index in range(0, len(paper_bank), BATCH_SIZE)
        ])
        
        logger.info(f"Number of papers in paper_bank: {len(paper_bank)}")

        rag_read_depth = 3
        selected_for_deep_read = select_papers(paper_bank, self.max_papers, rag_read_depth)

        if selected_for_deep_read:
            base_dir = os.path.join("results", task_name, f"traj_{session_id}")
            pdf_dir = os.path.join(base_dir, "pdf")
            os.makedirs(pdf_dir, exist_ok=True)

            async def deep_read_paper(paper: Dict[str, Any]) -> None:
                paper_id = paper["id"]
                pdf_url, doi = self._get_paper_download_credentials(paper)

                logger.info(
                    "Deep read paper: paper_id=%s url=%s doi=%s",
                    paper_id,
                    pdf_url,
                    doi,
                )
                if not (pdf_url or doi):
                    return

                pdf_path = None
                try:
                    if pdf_url:
                        pdf_path = await asyncio.to_thread(download_pdf, pdf_url, save_folder=pdf_dir)

                    if pdf_path is None and doi:
                        pdf_path = await asyncio.to_thread(download_pdf_by_doi, doi, pdf_dir)
                except Exception as e:
                    logger.warning(
                        f"Skipping paper {paper_id} during deep read because PDF download failed: {e}"
                    )
                    return

                if pdf_path is None:
                    logger.warning(
                        f"Skipping paper {paper_id} during deep read because no PDF could be downloaded. "
                        f"url={pdf_url}, doi={doi}"
                    )
                    return

                text = await asyncio.to_thread(extract_text_from_pdf, pdf_path)
                if not text:
                    return

                get_detail_prompt = (
                    f"Analyze the following paper text and extract structured information:\n{text}\n\n"
                    "Extract ALL of the following fields:\n"
                    "- Background: Core problem context and motivation\n"
                    "- Contributions: Novel contributions or key insights derived from the analysis\n"
                    "- Methods: The methodological framework and analytical reasoning — describe the THINKING APPROACH "
                    "used to solve the problem (conceptual/logical level, e.g., 'spatially-aware ensemble learning that "
                    "accounts for autocorrelation among burn severity patches'). DO NOT list software libraries or tool names.\n"
                    "- Applications: Specific real-world events, case studies, or scenarios where the dataset was applied "
                    "(e.g., specific floods, urban changes, historical events)\n"
                    "- Datasets_used: Product families, data modalities, or data source names explicitly mentioned in the paper "
                    "(e.g., MTBS, MODIS NDVI, ERA5, Landsat). Return as a JSON array of strings.\n\n"
                    "Return JSON with keys: background, contributions, methods, applications, datasets_used. "
                    "ONLY output the JSON dict with NO additional text."
                )
                try:
                    response = await self._call_model(
                        prompt=get_detail_prompt,
                        schema=output_schema_paper_details
                    )
                    details = response
                except Exception as e:
                    logger.error(f"survey error: {e}")
                    raise AgentExecutionError("Failed to get paper details")

                if details:
                    paper["background"] = details.get("background", "")
                    paper["contributions"] = details.get("contributions", "")
                    paper["methods"] = details.get("methods", "")
                    paper["applications"] = details.get("applications", "")
                    paper["datasets_used"] = details.get("datasets_used") or []
                    self._normalize_paper_detail_fields(paper)
                else:
                    paper["background"] = None
                    paper["contributions"] = None
                    paper["methods"] = None
                    paper["applications"] = None
                    paper["datasets_used"] = []

            results = await asyncio.gather(
                *[deep_read_paper(paper) for paper in selected_for_deep_read],
                return_exceptions=True,
            )
            # 保持原语义：任一论文的结构化抽取失败则整个 agent 失败。
            for result in results:
                if isinstance(result, AgentExecutionError):
                    raise result
      
        for paper in paper_bank:
            paper['is_deep_read'] = paper['id'] in [p['id'] for p in selected_for_deep_read]
            
        # Generate aggregated views for expert analysis
        # 过滤视图：限制进入上下文的文献集合
        filter_params = {
            "score_min": self.score_threshold,
            "top_k": self.paper_top_k,
        }
        if self.max_publication_year is not None:
            filter_params["max_publication_year"] = self.max_publication_year
        papers_all = paper_bank[:]
        papers_filtered = self._filter_papers(papers_all, filter_params)

        # 聚合视图使用过滤后的集合，若为空则退回全集
        aggregated_source = papers_filtered if papers_filtered else papers_all
        aggregated_views = self._generate_aggregated_views(aggregated_source)
        
        return papers_filtered, papers_all, search_queries, aggregated_views, filter_params

    def _filter_papers(self, papers: List[Dict[str, Any]], filter_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """根据分数阈值/Top-K生成精简文献列表

        规则：
        - 先按分数降序。
        - 仅保留 score >= score_min 的文献；数量不足 top_k 时不补低分。
        - 若没有任何文献达到阈值，则退化为取最高分 Top-K（容忍低分但不为空）。
        """
        score_min = filter_params.get("score_min", 0)
        top_k = filter_params.get("top_k", 0)

        scored = sorted(
            papers,
            key=lambda x: x.get("score", float("-inf")),
            reverse=True
        )

        filtered = [p for p in scored if p.get("score", float("-inf")) >= score_min]

        # 若阈值集为空，使用最高分 Top-K 兜底避免空集
        if not filtered:
            fallback = scored
            filtered = fallback[:top_k] if top_k and top_k > 0 else fallback
            return filtered

        if top_k and top_k > 0:
            filtered = filtered[:top_k]

        return filtered

    def _generate_aggregated_views(self, paper_bank: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate aggregated views for expert analysis.
        
        Args:
            paper_bank: List of papers with extracted information
            
        Returns:
            Dictionary with aggregated clusters and examples
        """
        # Initialize clusters
        method_clusters = {}
        application_clusters = {}
        dataset_clusters = {}
        triples = []
        
        for paper in paper_bank:
            self._normalize_paper_detail_fields(paper)
            paper_id = paper['id']
            
            # Method clustering
            methods = self._split_semantic_items(paper.get('methods'))
            for method in methods:
                if method:
                    if method not in method_clusters:
                        method_clusters[method] = []
                    method_clusters[method].append(paper_id)
            
            # Application clustering
            applications = self._split_semantic_items(paper.get('applications'))
            for app in applications:
                if app:
                    if app not in application_clusters:
                        application_clusters[app] = []
                    application_clusters[app].append(paper_id)
            
            # Dataset clustering: prefer deep-read datasets_used, fallback to keyword heuristics
            datasets = []
            if paper.get('datasets_used'):
                datasets = self._normalize_string_list(paper.get('datasets_used'))
            else:
                text = paper.get('abstract', '') + ' ' + paper.get('title', '')
                if 'satellite' in text.lower() or 'remote sensing' in text.lower():
                    datasets.append('Remote Sensing Data')
                if 'urban' in text.lower():
                    datasets.append('Urban Data')

            for dataset in datasets:
                if dataset not in dataset_clusters:
                    dataset_clusters[dataset] = []
                dataset_clusters[dataset].append(paper_id)

            # Generate triples only when all three dimensions have values
            evidence_source = 'deep_read' if paper.get('is_deep_read') else 'fallback_keyword'
            for method in methods:
                for app in applications:
                    for dataset in datasets:
                        if method and app and dataset:
                            triples.append({
                                'method': method,
                                'application': app,
                                'dataset': dataset,
                                'paper_ids': [paper_id],
                                'evidence_source': evidence_source,
                            })
        
        # Limit triples to top examples (e.g., 5)
        unique_triples = []
        seen = set()
        for triple in triples:
            key = (triple['method'], triple['application'], triple['dataset'])
            if key not in seen:
                seen.add(key)
                unique_triples.append(triple)
                if len(unique_triples) >= 5:
                    break
        
        return {
            'method_clusters': method_clusters,
            'application_clusters': application_clusters,
            'dataset_clusters': dataset_clusters,
            'method_application_dataset_triples': unique_triples
        }

    async def _filter_off_domain_papers(
        self,
        paper_bank: List[Dict[str, Any]],
        domain: str,
        task_description: str,
        data_description: str,
        dataset_portrait: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        评分前领域过滤：基于 title + abstract 批量判断每篇论文的相关性类别，
        直接丢弃真正不相关的噪声文献，不依赖 score 阈值。

        判据是三值而非二值。启用跨领域检索后，"学科距离"不再是有效判据——被有意
        召回的跨领域文献在学科上本就远离本域。真正的区分是**是否存在一条经由本数据集
        核心变量的机制链**：
          - in_domain              : 属于本域
          - cross_domain_relevant  : 学科不同，但核心变量在其中充当
                                     driver/response/covariate/stratifier/constraint
          - off_domain             : 无机制链，或核心变量只是同形异义词
                                     （如天体物理的 "spectral slope"）
        前两类保留，仅第三类丢弃。同形异义误召回（"burn severity" 命中核物理）才是
        本过滤器要拦的目标，跨领域组合不是。

        动态从 domain / task_description / data_description / dataset_portrait 推断
        目标领域与跨领域方向，而非硬编码特定领域词汇。

        Args:
            paper_bank: 原始论文列表（dict列表，每条含 title / abstract）
            domain: 研究领域描述（如 "geography"、"wildfire ecology"）
            task_description: 任务描述
            data_description: 数据集描述
            dataset_portrait: 数据集画像，提供 cross_domain_axes 作为跨领域判据；
                              缺失时退化为"本域 + 邻近方法学"的二类保留语义

        Returns:
            过滤后的论文列表（保留本域与跨领域相关论文）
        """
        if not paper_bank:
            return paper_bank

        # 构建领域上下文，供 LLM 判断目标领域
        domain_context_parts = []
        if domain:
            domain_context_parts.append(f"Research domain: {domain}")
        if task_description:
            domain_context_parts.append(f"Task: {task_description[:300]}")
        if data_description:
            domain_context_parts.append(f"Dataset: {data_description[:300]}")
        domain_context = "\n".join(domain_context_parts)

        # 跨领域方向：把"该保留的跨领域"从"真不相关"里明确切分出来。
        cross_domain_block = ""
        portrait = dataset_portrait if isinstance(dataset_portrait, dict) else {}
        cross_axes = portrait.get("cross_domain_axes")
        if isinstance(cross_axes, list) and cross_axes:
            direction_lines = []
            for axis_record in cross_axes[:8]:
                if not isinstance(axis_record, dict):
                    continue
                axis_domain = str(axis_record.get("domain_name") or "").strip()
                if not axis_domain:
                    continue
                role = str(axis_record.get("role") or "").strip()
                direction_lines.append(
                    f"- {axis_domain}" + (f" (core variable acts as: {role})" if role else "")
                )
            if direction_lines:
                cross_domain_block = (
                    "\n\n# Intended Cross-Domain Directions\n"
                    "These directions were deliberately searched for. Papers matching them are "
                    "WANTED, not noise, even though they sit in a different discipline:\n"
                    + "\n".join(direction_lines)
                )

        FILTER_BATCH_SIZE = 15
        kept_papers: List[Dict[str, Any]] = []
        removed_count = 0
        cross_domain_kept = 0

        for batch_start in range(0, len(paper_bank), FILTER_BATCH_SIZE):
            batch = paper_bank[batch_start: batch_start + FILTER_BATCH_SIZE]

            # 构造供 LLM 判断的论文摘要列表
            paper_snippets = []
            for p in batch:
                title = p.get("title", "")
                abstract = (p.get("abstract") or "")[:200]
                journal = p.get("journal", "")
                paper_snippets.append({
                    "id": p["id"],
                    "title": title,
                    "journal": journal,
                    "abstract_snippet": abstract,
                })

            output_schema = {
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "relevance": {
                                    "type": "string",
                                    "enum": [
                                        "in_domain",
                                        "cross_domain_relevant",
                                        "off_domain",
                                    ],
                                    "description": (
                                        "in_domain: belongs to the target research domain. "
                                        "cross_domain_relevant: different discipline, but the "
                                        "dataset's core variable participates in a real "
                                        "mechanism/causal link. "
                                        "off_domain: no such link, or the variable term is only "
                                        "a homonym from an unrelated field."
                                    ),
                                },
                                "reason": {
                                    "type": "string",
                                    "description": (
                                        "One sentence. For cross_domain_relevant, name the "
                                        "mechanism link through the core variable."
                                    ),
                                },
                            },
                            "required": ["id", "relevance"],
                        },
                    }
                },
                "required": ["results"],
            }

            prompt = (
                f"# Target Research Context\n{domain_context}"
                f"{cross_domain_block}\n\n"
                "# Task\n"
                "Classify each paper below into exactly one relevance class. The question is NOT "
                "'how far is this discipline from the target domain' — cross-disciplinary papers "
                "are searched for on purpose. The question is whether a real mechanism links the "
                "paper to the dataset's core variable.\n\n"
                "Rules:\n"
                "- `in_domain`: the paper is about the target domain, or a closely adjacent field "
                "  supplying relevant methodology (e.g., a spatial-autocorrelation statistics "
                "  paper for geography).\n"
                "- `cross_domain_relevant`: the paper sits in a different discipline, BUT the "
                "  dataset's core variable plays a real role in it — as driver, response, "
                "  covariate, stratifier, or constraint. Example: for a terrain-slope dataset, an "
                "  epidemiology paper on how hillslope gradient shapes vector breeding habitat "
                "  IS cross_domain_relevant, because slope drives the mechanism. Prefer this "
                "  class for papers matching the intended cross-domain directions above.\n"
                "- `off_domain`: no such mechanism link. This is the class for homonym "
                "  false-positives, where the variable term means something unrelated in that "
                "  field (e.g., 'spectral slope' in astrophysics, 'burn severity' in nuclear "
                "  fusion, 'slope' in a speech-processing filter). Also use it for papers from "
                "  unrelated disciplines that merely share a keyword.\n"
                "- Decide by asking: can you state one sentence of the form 'the dataset's core "
                "  variable causes / constrains / stratifies X in this paper'? If yes it is "
                "  relevant (in_domain or cross_domain_relevant); if the variable is only a "
                "  lexical coincidence, it is off_domain.\n"
                "- When genuinely uncertain, prefer keeping the paper: choose "
                "  `cross_domain_relevant` over `off_domain`.\n\n"
                f"# Papers\n{paper_snippets}\n\n"
                "Return a JSON object with a 'results' array, one entry per paper."
            )

            system_prompt = (
                "You are a relevance filter for scientific literature feeding a cross-disciplinary "
                "research pipeline. You remove only papers whose apparent match is a lexical "
                "coincidence — a term that means something different in an unrelated field. "
                "You deliberately keep papers from other disciplines when the dataset's core "
                "variable genuinely participates in their mechanism, because those papers supply "
                "the methodological diversity the pipeline needs."
            )

            try:
                response = await self._call_model(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    schema=output_schema,
                )
                results = response.get("results", [])
                relevance_map = self._normalize_relevance_results(results)

                for p in batch:
                    pid = str(p["id"])
                    # 默认保留：判定缺失时不丢弃（与既有保守语义一致）
                    relevance = relevance_map.get(pid, "in_domain")
                    if relevance != "off_domain":
                        if relevance == "cross_domain_relevant":
                            p["relevance_class"] = relevance
                            cross_domain_kept += 1
                        kept_papers.append(p)
                    else:
                        removed_count += 1
                        logger.info(
                            f"DomainFilter Removed off-domain paper: "
                            f"id={pid}, title={p.get('title', '')[:80]}"
                        )

            except Exception as e:
                logger.warning(
                    f"DomainFilter Batch filter failed (batch_start={batch_start}): {e}. "
                    "Keeping all papers in this batch as fallback."
                )
                kept_papers.extend(batch)

        logger.info(
            f"DomainFilter Removed {removed_count} off-domain papers, "
            f"kept {len(kept_papers)} / {len(paper_bank)} "
            f"(cross-domain relevant: {cross_domain_kept})"
        )
        return kept_papers

    _RELEVANCE_CLASSES = frozenset({
        "in_domain",
        "cross_domain_relevant",
        "off_domain",
    })

    @classmethod
    def _normalize_relevance_results(cls, results: Any) -> Dict[str, str]:
        """Map model output to id -> relevance class.

        枚举值缺失或不可识别时不写入映射，由调用方按"判定缺失即保留"处理，
        避免在这里伪造一个具体类别。
        """
        relevance_map: Dict[str, str] = {}
        if not isinstance(results, list):
            return relevance_map
        for record in results:
            if not isinstance(record, dict):
                continue
            paper_id = record.get("id")
            if paper_id is None:
                continue
            relevance = (
                str(record.get("relevance") or "")
                .strip()
                .lower()
                .replace("-", "_")
                .replace(" ", "_")
            )
            if relevance not in cls._RELEVANCE_CLASSES:
                continue
            relevance_map[str(paper_id)] = relevance
        return relevance_map

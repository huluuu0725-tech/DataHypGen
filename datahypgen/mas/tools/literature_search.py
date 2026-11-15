"""
Literature Search Tool for DataGenHyp

This module provides tools for scientific literature search, citation management, and metadata extraction.
It integrates with multiple academic search engines and databases.
"""

import os
import asyncio
import logging
import random
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import aiohttp
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def normalize_publication_year(value: Any) -> Optional[int]:
    """Normalize provider/cache year values without guessing from arbitrary text."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        normalized = value.strip()
        if re.fullmatch(r"\d{4}", normalized):
            return int(normalized)
    return None


def publication_year_is_allowed(
    value: Any,
    max_publication_year: Optional[int],
) -> bool:
    """Apply a fail-closed upper publication-year bound when one is configured."""
    if max_publication_year is None:
        return True
    year = normalize_publication_year(value)
    return year is not None and year <= max_publication_year


def validate_max_publication_year(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("max_publication_year must be an integer or None")
    if value < 1000 or value > 9999:
        raise ValueError("max_publication_year must be a four-digit year")
    return value


def _load_project_env() -> None:
    """Load the project .env so literature API keys are available at runtime."""
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    env_path = os.path.join(project_root, ".env")

    if not os.path.exists(env_path):
        return

    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=env_path, override=False)
    except Exception as e:
        logger.warning(f"Failed to load literature search env file '{env_path}': {e}")


@dataclass
class PaperMetadata:
    """Data class for paper metadata."""
    
    title: str
    authors: List[str]
    abstract: str
    year: Optional[int] = None
    doi: Optional[str] = None
    journal: Optional[str] = None
    keywords: Optional[List[str]] = None
    full_text: Optional[str] = None
    url: Optional[str] = None
    open_access: Optional[Dict[str, Any]] = None
    publication_date: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "year": self.year,
            "doi": self.doi,
            "journal": self.journal,
            "keywords": self.keywords,
            "url": self.url,
            "open_access": self.open_access,
            "publication_date": self.publication_date,
        }
    
    def to_citation(self, format_type: str = "apa") -> str:
        """
        Generate a formatted citation.
        
        Args:
            format_type: Citation format ("apa", "mla", "chicago", "harvard", "bibtex")
            
        Returns:
            Formatted citation string
        """
        if format_type == "apa":
            # APA format
            author_text = ""
            if self.authors:
                if len(self.authors) == 1:
                    author_text = f"{self.authors[0]}."
                elif len(self.authors) == 2:
                    author_text = f"{self.authors[0]} & {self.authors[1]}."
                else:
                    author_text = f"{self.authors[0]} et al."
            
            year_text = f" ({self.year})." if self.year else ""
            journal_text = f" {self.journal}," if self.journal else ""
            doi_text = f" doi:{self.doi}" if self.doi else ""
            
            return f"{author_text}{year_text} {self.title}.{journal_text}{doi_text}"
            
        elif format_type == "bibtex":
            # BibTeX format
            first_author = self.authors[0].split(" ")[-1] if self.authors else "Unknown"
            year = self.year or "Unknown"
            key = f"{first_author}{year}"
            
            authors = " and ".join(self.authors) if self.authors else "Unknown"
            
            return (
                f"@article{{{key},\n"
                f"  author = {{{authors}}},\n"
                f"  title = {{{self.title}}},\n"
                f"  journal = {{{self.journal or 'Unknown'}}},\n"
                f"  year = {{{self.year or 'Unknown'}}},\n"
                f"  doi = {{{self.doi or ''}}}\n"
                f"}}"
            )
            
        # Default to a basic citation
        authors = ", ".join(self.authors) if self.authors else "Unknown"
        year = f"({self.year})" if self.year else ""
        journal = f"{self.journal}" if self.journal else ""
        
        return f"{authors} {year}. {self.title}. {journal}"



class LiteratureSearch:
    """
    Tool for searching scientific literature across multiple sources.
    """
    _global_request_lock: Optional[asyncio.Lock] = None
    _arxiv_cooldown_reason: Optional[str] = None
    _arxiv_cooldown_until: float = 0.0
    _arxiv_last_request_at: float = 0.0
    _nature_disabled_reason: Optional[str] = None
    
    def __init__(self,
                api_keys: Optional[Dict[str, str]] = None,
                email: Optional[str] = None,
                max_publication_year: Optional[int] = None):
        """
        Initialize the literature search tool.
        
        Args:
            api_keys: Dictionary of API keys for different sources
            email: Contact email for OpenAlex polite pool
            max_publication_year: Inclusive publication-year upper bound. Papers
                with missing or invalid years are excluded when configured.
        """
        max_publication_year = validate_max_publication_year(max_publication_year)
        self.api_keys = api_keys or {}
        _load_project_env()
        self.email = email
        self.max_publication_year = max_publication_year
        if not self.email:
            env_email = os.getenv("OPENALEX_EMAIL")
            if env_email:
                self.email = env_email
        
        # Default search parameters
        self.default_max_results = 10
        self.default_sort = "relevance"  # or "date"
        
        # Cache for search results
        self._cache = {}

    def is_publication_year_allowed(self, value: Any) -> bool:
        return publication_year_is_allowed(value, self.max_publication_year)

    def _publication_year_cache_key(self) -> str:
        value = self.max_publication_year
        return f"max_year={value}" if value is not None else "max_year=none"

    def _filter_papers_by_publication_year(
        self,
        papers: List[PaperMetadata],
        *,
        source_name: str,
    ) -> List[PaperMetadata]:
        if self.max_publication_year is None:
            return papers
        filtered = [paper for paper in papers if self.is_publication_year_allowed(paper.year)]
        dropped = len(papers) - len(filtered)
        if dropped:
            logger.info(
                "%s publication cutoff <=%d removed %d papers with newer or unknown years",
                source_name,
                self.max_publication_year,
                dropped,
            )
        return filtered

    @classmethod
    def _get_global_request_lock(cls) -> asyncio.Lock:
        if cls._global_request_lock is None:
            cls._global_request_lock = asyncio.Lock()
        return cls._global_request_lock

    @asynccontextmanager
    async def _serialized_session_get(
        self,
        source_name: str,
        session: aiohttp.ClientSession,
        url: str,
        *,
        min_interval_seconds: float = 0.0,
        **kwargs,
    ):
        # Conservative mode: serialize every outbound literature-source request.
        async with self._get_global_request_lock():
            if source_name == "arXiv" and min_interval_seconds > 0.0:
                elapsed = time.monotonic() - type(self)._arxiv_last_request_at
                remaining = min_interval_seconds - elapsed
                if remaining > 0.0:
                    await self._sleep(remaining)
                type(self)._arxiv_last_request_at = time.monotonic()
            logger.debug("Serialized literature request start: %s", source_name)
            async with session.get(url, **kwargs) as response:
                yield response

    def _make_client_session(
        self,
        headers: Optional[Dict[str, str]] = None,
    ) -> aiohttp.ClientSession:
        """让 aiohttp 显式继承环境代理配置，避免与 curl 行为不一致。"""
        session_headers = {
            "Accept-Encoding": "gzip, deflate",
            "User-Agent": "DataGenHyp literature search",
        }
        if headers:
            session_headers.update(headers)
        return aiohttp.ClientSession(
            trust_env=True,
            headers=session_headers,
        )

    @staticmethod
    def _get_required_env(name: str) -> str:
        value = os.getenv(name)
        if value is None or not value.strip():
            raise RuntimeError(f"Required environment variable {name} is not configured")
        return value.strip()

    def _get_arxiv_config(self) -> tuple[str, Dict[str, str], float]:
        api_url = self._get_required_env("ARXIV_API_URL")
        if not api_url.startswith("https://"):
            raise ValueError("ARXIV_API_URL must use HTTPS")
        contact_email = self._get_required_env("ARXIV_CONTACT_EMAIL")
        if "@" not in contact_email:
            raise ValueError("ARXIV_CONTACT_EMAIL must be a valid contact email")
        interval_value = self._get_required_env("ARXIV_REQUEST_INTERVAL_SECONDS")
        try:
            request_interval_seconds = float(interval_value)
        except ValueError as exc:
            raise ValueError("ARXIV_REQUEST_INTERVAL_SECONDS must be numeric") from exc
        if request_interval_seconds <= 0.0:
            raise ValueError("ARXIV_REQUEST_INTERVAL_SECONDS must be greater than zero")
        headers = {
            "User-Agent": f"DataGenHyp/1.0 (mailto:{contact_email})",
            "Accept": "application/atom+xml",
        }
        return api_url, headers, request_interval_seconds

    async def _sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)

    @classmethod
    def _set_arxiv_cooldown(cls, cooldown_seconds: float, reason: str) -> None:
        cls._arxiv_cooldown_until = max(time.monotonic() + max(0.0, cooldown_seconds), cls._arxiv_cooldown_until)
        cls._arxiv_cooldown_reason = reason
        logger.warning(
            "Cooling down arXiv search for %.0f seconds: %s",
            max(0.0, cooldown_seconds),
            reason,
        )

    @classmethod
    def _reset_arxiv_cooldown_for_tests(cls) -> None:
        cls._arxiv_cooldown_reason = None
        cls._arxiv_cooldown_until = 0.0

    @classmethod
    def _arxiv_cooldown_active(cls) -> bool:
        if cls._arxiv_cooldown_until <= 0.0:
            return False
        if time.monotonic() >= cls._arxiv_cooldown_until:
            cls._reset_arxiv_cooldown_for_tests()
            return False
        return True

    @classmethod
    def _get_arxiv_cooldown_remaining_seconds(cls) -> float:
        if not cls._arxiv_cooldown_active():
            return 0.0
        return max(0.0, cls._arxiv_cooldown_until - time.monotonic())

    async def _wait_for_arxiv_cooldown(self) -> None:
        while True:
            remaining_seconds = self._get_arxiv_cooldown_remaining_seconds()
            if remaining_seconds <= 0.0:
                return

            cooldown_deadline = self._arxiv_cooldown_until
            logger.info(
                "Waiting %.2f seconds for arXiv cooldown to expire: %s",
                remaining_seconds,
                self._arxiv_cooldown_reason or "rate limit cooldown",
            )
            await self._sleep(remaining_seconds)

            if self._arxiv_cooldown_until <= cooldown_deadline:
                self._reset_arxiv_cooldown_for_tests()
                return

    @classmethod
    def _disable_nature_for_process(cls, reason: str) -> None:
        if cls._nature_disabled_reason:
            return
        cls._nature_disabled_reason = reason
        logger.warning("Disabling Springer Nature search for current process: %s", reason)

    @classmethod
    def _reset_nature_disable_for_tests(cls) -> None:
        cls._nature_disabled_reason = None

    @staticmethod
    def _parse_retry_after_seconds(headers: Any) -> Optional[float]:
        if not headers:
            return None
        retry_after = headers.get("retry-after")
        if retry_after is None:
            return None
        try:
            return float(str(retry_after).strip())
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _is_connectivity_error(exc: Exception) -> bool:
        return isinstance(
            exc,
            (
                aiohttp.ClientConnectionError,
                aiohttp.ClientConnectorError,
                asyncio.TimeoutError,
                TimeoutError,
            ),
        )

    async def _sleep_with_backoff(
        self,
        source_name: str,
        attempt: int,
        tries: int,
        *,
        base_delay: float = 10.0,
        max_delay: float = 60.0,
    ) -> None:
        """指数退避 + 抖动，避免固定重试节奏持续撞限流窗口。"""
        if attempt >= tries - 1:
            return

        backoff_cap = min(max_delay, base_delay * (2 ** attempt))
        delay = backoff_cap + random.uniform(0, min(1.0, backoff_cap * 0.25))
        logger.info(
            "%s retrying in %.2f seconds (attempt %d/%d)",
            source_name,
            delay,
            attempt + 2,
            tries,
        )
        await self._sleep(delay)

    def _get_semantic_scholar_key(self) -> Optional[str]:
        env_key = os.getenv("S2_API_KEY")
        if env_key:
            return env_key

        legacy_env_key = os.getenv("SS_API_KEY")
        if legacy_env_key:
            return legacy_env_key

        semantic_key = self.api_keys.get("semantic_scholar")
        if semantic_key:
            return semantic_key

        semantic_key_alt = self.api_keys.get("semantic_scholar_key")
        if semantic_key_alt:
            return semantic_key_alt

        return None

    def _get_openalex_key(self) -> Optional[str]:
        env_key = os.getenv("OPENALEX_API_KEY")
        if env_key:
            return env_key

        env_key_alt = os.getenv("OPENALEX_KEY")
        if env_key_alt:
            return env_key_alt

        config_key = self.api_keys.get("openalex")
        if config_key:
            return config_key

        config_key_alt = self.api_keys.get("openalex_key")
        if config_key_alt:
            return config_key_alt

        return None

    def _get_springer_nature_key(self) -> Optional[str]:
        env_key = os.getenv("SPRINGER_NATURE_API_KEY")
        if env_key:
            return env_key

        legacy_env_key = os.getenv("SPRINGER_API_KEY")
        if legacy_env_key:
            return legacy_env_key

        for key_name in ("springer_nature", "springer_nature_key", "nature"):
            config_key = self.api_keys.get(key_name)
            if config_key:
                return config_key

        return None

    def _build_openalex_params(self) -> Dict[str, str]:
        params: Dict[str, str] = {}

        api_key = self._get_openalex_key()
        if api_key:
            params["api_key"] = api_key

        if self.email:
            params["mailto"] = self.email

        return params

    def _extract_doi(self, doi_value: Optional[str]) -> Optional[str]:
        if not doi_value:
            return None

        doi = doi_value.strip()
        for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
            if doi.lower().startswith(prefix):
                return doi[len(prefix):]
        return doi

    def _extract_year_from_date(self, date_value: Any) -> Optional[int]:
        if date_value is None:
            return None

        match = re.search(r"(\d{4})", str(date_value))
        if not match:
            return None

        try:
            return int(match.group(1))
        except ValueError:
            return None

    def _sanitize_openalex_query(self, query: str) -> str:
        sanitized = str(query or "")
        sanitized = sanitized.replace("*", " ").replace("?", " ")
        sanitized = re.sub(r"\s+", " ", sanitized).strip()
        return sanitized

    def _reconstruct_openalex_abstract(self, abstract_index: Any) -> str:
        if not isinstance(abstract_index, dict) or not abstract_index:
            return ""

        try:
            max_position = -1
            for positions in abstract_index.values():
                if isinstance(positions, list) and positions:
                    max_position = max(max_position, max(positions))

            if max_position < 0:
                return ""

            ordered_tokens = [""] * (max_position + 1)
            for token, positions in abstract_index.items():
                if not isinstance(positions, list):
                    continue
                for pos in positions:
                    if isinstance(pos, int) and 0 <= pos <= max_position:
                        ordered_tokens[pos] = token

            return " ".join([token for token in ordered_tokens if token]).strip()
        except Exception as e:
            logger.warning(f"Failed to reconstruct OpenAlex abstract: {e}")
            return ""

    def _normalize_openalex_id(self, work_id: str) -> Optional[str]:
        raw_id = (work_id or "").strip()
        if not raw_id:
            return None

        if raw_id.startswith("https://openalex.org/"):
            short_id = raw_id.rsplit("/", 1)[-1]
            if re.fullmatch(r"W\d+", short_id, re.IGNORECASE):
                return short_id.upper()
            return None

        if re.fullmatch(r"W\d+", raw_id, re.IGNORECASE):
            return raw_id.upper()

        return None

    def _normalize_openalex_topic_id(self, topic_id: str) -> Optional[str]:
        raw_id = (topic_id or "").strip()
        if not raw_id:
            return None

        if raw_id.startswith("https://openalex.org/"):
            short_id = raw_id.rsplit("/", 1)[-1]
            if re.fullmatch(r"T\d+", short_id, re.IGNORECASE):
                return short_id.upper()
            return None

        if re.fullmatch(r"T\d+", raw_id, re.IGNORECASE):
            return raw_id.upper()

        return None

    def _parse_openalex_work(self, work_data: Dict[str, Any]) -> Optional[PaperMetadata]:
        if not isinstance(work_data, dict):
            return None

        title = work_data.get("display_name", "")
        if not title:
            return None

        authors = []
        for authorship in work_data.get("authorships", []):
            author_name = authorship.get("author", {}).get("display_name")
            if author_name:
                authors.append(author_name)

        abstract = self._reconstruct_openalex_abstract(work_data.get("abstract_inverted_index"))
        open_access = work_data.get("open_access") or {}
        primary_location = work_data.get("primary_location") or {}
        source_info = primary_location.get("source") or {}

        url = primary_location.get("pdf_url")
        if not url:
            url = open_access.get("oa_url")
        if not url:
            url = primary_location.get("landing_page_url")
        if not url:
            url = work_data.get("doi")

        keywords = []
        for concept in work_data.get("concepts", []):
            if concept.get("score", 0) < 0.3:
                continue
            concept_name = concept.get("display_name")
            if concept_name:
                keywords.append(concept_name)
            if len(keywords) >= 8:
                break

        publication_year = normalize_publication_year(work_data.get("publication_year"))
        if not self.is_publication_year_allowed(publication_year):
            return None

        paper = PaperMetadata(
            title=title,
            authors=authors,
            abstract=abstract,
            year=publication_year,
            doi=self._extract_doi(work_data.get("doi")),
            journal=source_info.get("display_name"),
            keywords=keywords or None,
            url=url,
            open_access=open_access or None,
            publication_date=work_data.get("publication_date"),
        )
        return paper

    def _extract_nature_abstract(self, abstract_data: Any) -> str:
        if isinstance(abstract_data, str):
            return abstract_data.strip()

        if isinstance(abstract_data, list):
            parts = [self._extract_nature_abstract(item) for item in abstract_data]
            return " ".join([part for part in parts if part]).strip()

        if isinstance(abstract_data, dict):
            for key in ("p", "content", "text", "value"):
                if key in abstract_data:
                    return self._extract_nature_abstract(abstract_data.get(key))

            parts = [self._extract_nature_abstract(value) for value in abstract_data.values()]
            return " ".join([part for part in parts if part]).strip()

        return ""

    def _extract_nature_authors(self, creators: Any) -> List[str]:
        authors: List[str] = []

        if isinstance(creators, list):
            for creator in creators:
                if isinstance(creator, dict):
                    author_name = (
                        creator.get("creator")
                        or creator.get("name")
                        or creator.get("displayName")
                    )
                    if author_name:
                        authors.append(str(author_name).strip())
                elif isinstance(creator, str):
                    authors.append(creator.strip())

        return [author for author in authors if author]

    def _classify_nature_url_candidate(
        self,
        candidate: str,
        hint: str,
        pdf_urls: List[str],
        landing_urls: List[str],
    ) -> None:
        normalized_candidate = (candidate or "").strip()
        if not normalized_candidate:
            return

        if not normalized_candidate.lower().startswith(("http://", "https://")):
            return

        normalized_hint = hint.lower()
        normalized_url = normalized_candidate.lower()
        if (
            normalized_url.endswith(".pdf")
            or ".pdf?" in normalized_url
            or "/pdf" in normalized_url
            or "application/pdf" in normalized_hint
            or " pdf" in normalized_hint
            or normalized_hint.startswith("pdf")
        ):
            pdf_urls.append(normalized_candidate)
            return

        landing_urls.append(normalized_candidate)

    def _collect_nature_url_candidates(
        self,
        value: Any,
        pdf_urls: List[str],
        landing_urls: List[str],
        hint: str = "",
    ) -> None:
        if isinstance(value, dict):
            current_hint_parts = [
                hint,
                str(value.get("format", "")),
                str(value.get("type", "")),
                str(value.get("contentType", "")),
                str(value.get("platform", "")),
            ]
            current_hint = " ".join([part for part in current_hint_parts if part]).strip()
            for field_name in ("value", "url", "href"):
                field_value = value.get(field_name)
                if isinstance(field_value, str):
                    self._classify_nature_url_candidate(
                        field_value,
                        current_hint,
                        pdf_urls,
                        landing_urls,
                    )
            for nested_value in value.values():
                self._collect_nature_url_candidates(
                    nested_value,
                    pdf_urls,
                    landing_urls,
                    current_hint,
                )
            return

        if isinstance(value, list):
            for item in value:
                self._collect_nature_url_candidates(item, pdf_urls, landing_urls, hint)
            return

        if isinstance(value, str):
            self._classify_nature_url_candidate(value, hint, pdf_urls, landing_urls)

    def _dedupe_preserve_order(self, values: List[str]) -> List[str]:
        seen = set()
        deduped = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            deduped.append(value)
        return deduped

    def _parse_nature_record(self, record: Dict[str, Any]) -> Optional[PaperMetadata]:
        if not isinstance(record, dict):
            return None

        title = str(record.get("title") or "").strip()
        if not title:
            return None

        doi = self._extract_doi(record.get("doi"))
        authors = self._extract_nature_authors(record.get("creators"))
        abstract = self._extract_nature_abstract(record.get("abstract"))
        journal = record.get("journalTitle") or record.get("publicationName") or record.get("journal")
        publication_date = (
            record.get("publicationDate")
            or record.get("onlineDate")
            or record.get("printDate")
        )
        publication_year = self._extract_year_from_date(publication_date)
        if not self.is_publication_year_allowed(publication_year):
            return None

        pdf_urls: List[str] = []
        landing_urls: List[str] = []
        for field_name in ("url", "urls", "identifier"):
            self._collect_nature_url_candidates(record.get(field_name), pdf_urls, landing_urls)

        pdf_urls = self._dedupe_preserve_order(pdf_urls)
        landing_urls = self._dedupe_preserve_order(landing_urls)

        doi_url = f"https://doi.org/{doi}" if doi else None
        url = pdf_urls[0] if pdf_urls else None
        if not url and landing_urls:
            url = landing_urls[0]
        if not url:
            url = doi_url

        open_access: Dict[str, Any] = {}
        if pdf_urls:
            open_access["pdf_url"] = pdf_urls[0]
            open_access["oa_url"] = pdf_urls[0]
        if landing_urls:
            open_access["landing_page_url"] = landing_urls[0]
        if doi_url:
            open_access["doi_url"] = doi_url

        return PaperMetadata(
            title=title,
            authors=authors,
            abstract=abstract,
            year=publication_year,
            doi=doi,
            journal=journal,
            url=url,
            open_access=open_access or None,
            publication_date=str(publication_date) if publication_date else None,
        )

    
    async def search_arxiv(self, 
                         query: str, 
                         max_results: int = 10, 
                         sort: str = "relevance",
                         categories: Optional[List[str]] = None,
                         **kwargs) -> List[PaperMetadata]:
        """
        Search arXiv for papers matching the query.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            sort: Sort order ("relevance" or "date")
            categories: List of arXiv categories to search
            
        Returns:
            List of paper metadata
        """
        if self._arxiv_cooldown_active():
            await self._wait_for_arxiv_cooldown()

        # Build the cache key
        cats_str = ",".join(categories) if categories else "all"
        cache_key = (
            f"arxiv:{query}:{max_results}:{sort}:{cats_str}:"
            f"{self._publication_year_cache_key()}"
        )
        if cache_key in self._cache:
            logger.info(f"Using cached results for arXiv query: {query}")
            return self._cache[cache_key]
            
        logger.debug(f"Searching arXiv for: {query}")
        
        search_url, arxiv_headers, request_interval_seconds = self._get_arxiv_config()
        
        # Sort parameter
        sort_param = "relevance" if sort == "relevance" else "submittedDate"
        
        # Category filter
        cat_filter = ""
        if categories:
            cat_filter = " AND (" + " OR ".join([f"cat:{cat}" for cat in categories]) + ")"
        
        # Search parameters
        search_query = f"all:{query}{cat_filter}"
        if self.max_publication_year is not None:
            search_query += (
                " AND submittedDate:[199001010000 TO "
                f"{self.max_publication_year}12312359]"
            )
        search_params = {
            "search_query": search_query,
            "max_results": max_results,
            "sortBy": sort_param,
            "sortOrder": "descending"
        }
        
        tries = 3
        for attempt in range(tries):
            try:
                async with self._make_client_session(headers=arxiv_headers) as session:
                    async with self._serialized_session_get(
                        "arXiv",
                        session,
                        search_url,
                        min_interval_seconds=request_interval_seconds,
                        params=search_params,
                    ) as response:
                        if response.status != 200:
                            logger.error(f"arXiv search error: {response.status}")
                            if response.status == 429:
                                retry_after_seconds = self._parse_retry_after_seconds(
                                    getattr(response, "headers", {})
                                )
                                cooldown_seconds = retry_after_seconds if retry_after_seconds is not None else 300.0
                                reason = "API request limit exceeded"
                                if retry_after_seconds is not None:
                                    reason = f"{reason} (Retry-After={retry_after_seconds:g}s)"
                                self._set_arxiv_cooldown(cooldown_seconds, reason)
                                if attempt < tries - 1:
                                    await self._wait_for_arxiv_cooldown()
                                    continue
                                return []
                            if attempt < tries - 1:
                                await self._sleep_with_backoff("arXiv", attempt, tries)
                                continue
                            else:
                                return []

                        xml_data = await response.text()
                        logger.info(f'arXiv REQUEST {query} success!')

                        papers = self._filter_papers_by_publication_year(
                            self._parse_arxiv_xml(xml_data),
                            source_name="arXiv",
                        )
                        
                        # Cache the results
                        self._cache[cache_key] = papers
                        
                        logger.info(f"Get {len(papers)} papers from arXiv")
                        
                        return papers
                        
            except Exception as e:
                logger.error(f"Error searching arXiv: {e}")
                if attempt < tries - 1:
                    await self._sleep_with_backoff("arXiv", attempt, tries)
                    continue
                return []
    
    async def search_semantic_scholar(self,
                                    query: str,
                                    max_results: int = 10,
                                    use_api_key: bool = True,
                                    **kwargs) -> List[PaperMetadata]:
        """
        Search Semantic Scholar for papers matching the query.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of paper metadata
        """
        # Check if API key is available
        api_key = self._get_semantic_scholar_key() if use_api_key else None
        if not api_key:
            logger.warning("No API key for Semantic Scholar, using limited access")
            
        # Build the cache key
        auth_mode = "keyed" if api_key else "anonymous"
        cache_key = (
            f"semantic:{auth_mode}:{query}:{max_results}:"
            f"{self._publication_year_cache_key()}"
        )
        if cache_key in self._cache:
            logger.info(f"Using cached results for Semantic Scholar query: {query}")
            return self._cache[cache_key]
            
        logger.info(f"Searching Semantic Scholar for: {query}")
        
        # Semantic Scholar API URL
        search_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        
        # Search parameters
        search_params = {
            "query": query,
            "limit": max_results,
            "fields": "title,abstract,authors,year,journal,citationCount,doi,url,openAccessPdf,isOpenAccess"
        }
        if self.max_publication_year is not None:
            search_params["year"] = f"-{self.max_publication_year}"
        
        # Headers
        headers = {"x-api-key": api_key} if api_key else {}
        
        tries = 3
        for attempt in range(tries):
            search_data = {}
            papers = []
            try:
                # Rate limit between requests
                await self._sleep(1)
                async with self._make_client_session() as session:
                    async with self._serialized_session_get(
                        "Semantic Scholar",
                        session,
                        search_url,
                        params=search_params,
                        headers=headers,
                    ) as response:
                        if response.status != 200:
                            logger.error(f"Semantic Scholar search error: {response.status}")
                            if attempt < tries - 1:
                                await self._sleep_with_backoff("Semantic Scholar", attempt, tries)
                                continue
                            else:
                                return []
                        else:
                            search_data = await response.json() if response.content_type == 'application/json' else {}
                            papers = []
                    
                        for paper_data in search_data.get("data", []):
                            author_list = [author.get("name", "") for author in paper_data.get("authors", [])]
                            open_access_pdf = paper_data.get("openAccessPdf") or {}
                            
                            publication_year = normalize_publication_year(paper_data.get("year"))
                            if not self.is_publication_year_allowed(publication_year):
                                continue
                            paper = PaperMetadata(
                                title=paper_data.get("title", ""),
                                authors=author_list,
                                abstract=paper_data.get("abstract", ""),
                                year=publication_year,
                                doi=self._extract_doi(paper_data.get("doi")),
                                journal=paper_data.get("journal", {}).get("name") if paper_data.get("journal") else None,
                                url=open_access_pdf.get("url") or paper_data.get("url"),
                                open_access={
                                    "is_oa": paper_data.get("isOpenAccess"),
                                    "oa_url": open_access_pdf.get("url"),
                                }
                            )
                            papers.append(paper)
                        
                        # Cache the results
                        self._cache[cache_key] = papers
                        
                        
                        return papers
      
            except Exception as e:
                logger.error(f"Error searching Semantic Scholar: {str(e)}")
                if attempt < tries - 1:
                    await self._sleep_with_backoff("Semantic Scholar", attempt, tries)
                    continue
                return []

    async def search_semantic_scholar_no_key(
        self,
        query: str,
        max_results: int = 10,
        **kwargs,
    ) -> List[PaperMetadata]:
        return await self.search_semantic_scholar(
            query,
            max_results=max_results,
            use_api_key=False,
            **kwargs,
        )

    async def search_openalex(self,
                            query: str,
                            max_results: int = 10,
                            topic_names: Optional[List[str]] = None,
                            **kwargs) -> List[PaperMetadata]:
        """
        Search OpenAlex for papers matching the query.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of paper metadata
        """
        topic_key = ",".join(topic_names or []) if topic_names else "all"
        cache_key = (
            f"openalex:{query}:{max_results}:{topic_key}:"
            f"{self._publication_year_cache_key()}"
        )
        if cache_key in self._cache:
            logger.info(f"Using cached results for OpenAlex query: {query}")
            return self._cache[cache_key]

        sanitized_query = self._sanitize_openalex_query(query)
        logger.info(f"Searching OpenAlex for: {sanitized_query}")

        search_url = "https://api.openalex.org/works"
        search_params = {
            "search": sanitized_query,
            "per-page": max_results,
        }
        topic_ids = await self._resolve_openalex_topic_ids(topic_names or [])
        openalex_filter = self._build_openalex_topic_filter(topic_ids)
        filter_parts = [openalex_filter] if openalex_filter else []
        if self.max_publication_year is not None:
            filter_parts.append(
                f"to_publication_date:{self.max_publication_year}-12-31"
            )
        if filter_parts:
            search_params["filter"] = ",".join(filter_parts)
        search_params.update(self._build_openalex_params())

        tries = 3
        for attempt in range(tries):
            try:
                await self._sleep(1)
                async with self._make_client_session() as session:
                    async with self._serialized_session_get(
                        "OpenAlex",
                        session,
                        search_url,
                        params=search_params,
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(
                                f"OpenAlex search error: {response.status}, body={error_text[:200]}"
                            )
                            if 400 <= response.status < 500:
                                return []
                            if attempt < tries - 1:
                                await self._sleep_with_backoff("OpenAlex", attempt, tries)
                                continue
                            else:
                                return []

                        search_data = await response.json(content_type=None)
                        papers = []
                        for work_data in search_data.get("results", []):
                            parsed = self._parse_openalex_work(work_data)
                            if parsed:
                                papers.append(parsed)

                        logger.info("Get %d papers from OpenAlex", len(papers))
                        self._cache[cache_key] = papers
                        return papers

            except Exception as e:
                logger.error(f"Error searching OpenAlex: {str(e)}")
                if attempt < tries - 1:
                    await self._sleep_with_backoff("OpenAlex", attempt, tries)
                    continue
                return []

        return []

    async def _resolve_openalex_topic_ids(self, topic_names: List[str]) -> List[str]:
        normalized_topics = []
        for topic in topic_names:
            cleaned = self._sanitize_openalex_query(topic)
            if cleaned:
                normalized_topics.append(cleaned)

        normalized_topics = list(dict.fromkeys(normalized_topics))[:4]
        if not normalized_topics:
            return []

        topic_ids: List[str] = []
        for topic in normalized_topics:
            resolved = await self._search_openalex_topic_ids(topic)
            for topic_id in resolved:
                if topic_id not in topic_ids:
                    topic_ids.append(topic_id)
        return topic_ids[:8]

    async def _search_openalex_topic_ids(self, topic_name: str) -> List[str]:
        cache_key = f"openalex_topics:{topic_name}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        search_url = "https://api.openalex.org/topics"
        search_params = {
            "search": topic_name,
            "per-page": 3,
        }
        search_params.update(self._build_openalex_params())

        tries = 3
        for attempt in range(tries):
            try:
                await self._sleep(1)
                async with self._make_client_session() as session:
                    async with self._serialized_session_get(
                        "OpenAlex topics",
                        session,
                        search_url,
                        params=search_params,
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(
                                f"OpenAlex topic search error: {response.status}, body={error_text[:200]}"
                            )
                            if 400 <= response.status < 500:
                                return []
                            if attempt < tries - 1:
                                await self._sleep_with_backoff("OpenAlex topics", attempt, tries)
                                continue
                            return []

                        search_data = await response.json(content_type=None)
                        topic_ids: List[str] = []
                        for topic_data in search_data.get("results", []):
                            normalized_id = self._normalize_openalex_topic_id(
                                str(topic_data.get("id") or "")
                            )
                            if normalized_id and normalized_id not in topic_ids:
                                topic_ids.append(normalized_id)

                        self._cache[cache_key] = topic_ids
                        return topic_ids
            except Exception as e:
                logger.error(f"Error searching OpenAlex topics: {str(e)}")
                if attempt < tries - 1:
                    await self._sleep_with_backoff("OpenAlex topics", attempt, tries)
                    continue
                return []

        return []

    def _build_openalex_topic_filter(self, topic_ids: List[str]) -> str:
        normalized_topic_ids = []
        for topic_id in topic_ids:
            normalized_id = self._normalize_openalex_topic_id(topic_id)
            if normalized_id:
                normalized_topic_ids.append(normalized_id)

        normalized_topic_ids = list(dict.fromkeys(normalized_topic_ids))[:8]
        if not normalized_topic_ids:
            return ""

        return f"topics.id:{'|'.join(normalized_topic_ids)}"

    async def search_nature(self,
                            query: str,
                            max_results: int = 10,
                            **kwargs) -> List[PaperMetadata]:
        """Search Springer Nature for papers matching the query."""
        if self._nature_disabled_reason:
            return []

        api_key = self._get_springer_nature_key()
        if not api_key:
            logger.warning("No API key for Springer Nature; skipping nature search")
            return []

        cache_key = (
            f"nature:{query}:{max_results}:{self._publication_year_cache_key()}"
        )
        if cache_key in self._cache:
            logger.info(f"Using cached results for Nature query: {query}")
            return self._cache[cache_key]

        logger.info(f"Searching Nature for: {query}")

        search_url = "https://api.springernature.com/meta/v2/json"
        search_params = {
            "q": query,
            "p": max_results,
            "api_key": api_key,
        }

        tries = 3
        for attempt in range(tries):
            try:
                await self._sleep(1)
                async with self._make_client_session() as session:
                    async with self._serialized_session_get(
                        "Nature",
                        session,
                        search_url,
                        params=search_params,
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(
                                f"Nature search error: {response.status}, body={error_text[:200]}"
                            )
                            if response.status == 429:
                                reason = "API request limit exceeded"
                                if "retry-after" in getattr(response, "headers", {}):
                                    reason = f"{reason} (Retry-After={response.headers.get('retry-after')})"
                                self._disable_nature_for_process(reason)
                                return []
                            if response.status in (401, 403):
                                self._disable_nature_for_process(f"HTTP {response.status}")
                                return []
                            if 400 <= response.status < 500:
                                return []
                            if attempt < tries - 1:
                                await self._sleep_with_backoff("Nature", attempt, tries)
                                continue
                            else:
                                self._disable_nature_for_process(f"HTTP {response.status}")
                                return []

                        search_data = await response.json(content_type=None)
                        papers = []
                        for record in search_data.get("records", []):
                            parsed = self._parse_nature_record(record)
                            if parsed:
                                papers.append(parsed)

                        self._cache[cache_key] = papers
                        return papers

            except Exception as e:
                logger.error(f"Error searching Nature: {str(e)}")
                if self._is_connectivity_error(e):
                    self._disable_nature_for_process(f"connectivity_error: {type(e).__name__}")
                    return []
                if attempt < tries - 1:
                    await self._sleep_with_backoff("Nature", attempt, tries)
                    continue
                return []

        return []

        
    async def multi_source_search(self, 
                               query: str, 
                               sources: List[str] = None,
                               max_results: int = 10,
                               **kwargs) -> Dict[str, List[PaperMetadata]]:
        """
        Search multiple sources simultaneously.
        
        Args:
            query: Search query
            sources: List of sources to search
            max_results: Maximum results per source
            
        Returns:
            Dictionary mapping source names to result lists
        """
        if not sources:
            sources = ["arxiv", "nature"]

        combined_results = {}
        for source in sources:
            if source == "arxiv":
                try:
                    combined_results[source] = await self.search_arxiv(query, max_results, **kwargs)
                except Exception as e:
                    logger.error(f"Error searching {source}: {str(e)}")
                    combined_results[source] = []
            elif source == "openalex":
                try:
                    combined_results[source] = await self.search_openalex(query, max_results, **kwargs)
                except Exception as e:
                    logger.error(f"Error searching {source}: {str(e)}")
                    combined_results[source] = []
            elif source == "nature":
                try:
                    combined_results[source] = await self.search_nature(query, max_results, **kwargs)
                except Exception as e:
                    logger.error(f"Error searching {source}: {str(e)}")
                    combined_results[source] = []
            elif source == "semantic_scholar":
                try:
                    combined_results[source] = await self.search_semantic_scholar(query, max_results, **kwargs)
                except Exception as e:
                    logger.error(f"Error searching {source}: {str(e)}")
                    combined_results[source] = []
            elif source == "semantic_scholar_no_key":
                try:
                    combined_results[source] = await self.search_semantic_scholar_no_key(query, max_results, **kwargs)
                except Exception as e:
                    logger.error(f"Error searching {source}: {str(e)}")
                    combined_results[source] = []
            else:
                logger.warning(f"Unknown literature source: {source}")

        for source in sources:
            if source not in combined_results:
                combined_results[source] = []
                
        return combined_results
    
    async def search_similar_papers(self,
                                  paper_id: str,
                                  max_results: int = 20,
                                  **kwargs) -> List[PaperMetadata]:
        """
        Search for papers similar to the given paper ID using OpenAlex related works.
        
        Args:
            paper_id: Paper ID to find similar papers for
            max_results: Maximum number of results
            
        Returns:
            List of similar paper metadata
        """
        normalized_id = self._normalize_openalex_id(paper_id)
        if not normalized_id:
            logger.warning(
                f"OpenAlex similar papers query requires OpenAlex work id like W123..., got: {paper_id}"
            )
            return []
            
        # Build the cache key
        cache_key = (
            f"openalex_similar:{normalized_id}:{max_results}:"
            f"{self._publication_year_cache_key()}"
        )
        if cache_key in self._cache:
            logger.info(f"Using cached results for similar papers: {normalized_id}")
            return self._cache[cache_key]
            
        logger.info(f"Searching OpenAlex similar papers for: {normalized_id}")
        work_url = f"https://api.openalex.org/works/{normalized_id}"
        base_params = self._build_openalex_params()
        
        tries = 3
        for attempt in range(tries):
            try:
                await self._sleep(1)
                async with self._make_client_session() as session:
                    async with self._serialized_session_get(
                        "OpenAlex similar",
                        session,
                        work_url,
                        params=base_params,
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(
                                f"OpenAlex similar search error: {response.status}, body={error_text[:200]}"
                            )
                            if attempt < tries - 1:
                                await self._sleep_with_backoff("OpenAlex similar", attempt, tries)
                                continue
                            else:
                                return []

                        work_data = await response.json(content_type=None)
                        related_work_ids = work_data.get("related_works", [])[:max_results]
                        if not related_work_ids:
                            self._cache[cache_key] = []
                            return []

                    papers: List[PaperMetadata] = []
                    for related_work_id in related_work_ids:
                        normalized_related_id = self._normalize_openalex_id(related_work_id)
                        if not normalized_related_id:
                            continue

                        related_url = f"https://api.openalex.org/works/{normalized_related_id}"
                        async with self._serialized_session_get(
                            "OpenAlex similar",
                            session,
                            related_url,
                            params=base_params,
                        ) as related_response:
                            if related_response.status != 200:
                                continue
                            related_work_data = await related_response.json(content_type=None)

                        parsed_paper = self._parse_openalex_work(related_work_data)
                        if parsed_paper:
                            papers.append(parsed_paper)

                    self._cache[cache_key] = papers
                    return papers

            except Exception as e:
                logger.error(f"Error searching OpenAlex similar papers: {str(e)}")
                if attempt < tries - 1:
                    await self._sleep_with_backoff("OpenAlex similar", attempt, tries)
                    continue
                return []
    
    
    def _parse_arxiv_xml(self, xml_data: str) -> List[PaperMetadata]:
        """
        Parse arXiv XML response to extract paper metadata.
        
        Args:
            xml_data: XML response from arXiv
            
        Returns:
            List of paper metadata
        """
        papers = []
        soup = BeautifulSoup(xml_data, "xml")
        
        for entry in soup.find_all("entry"):
            try:
                # Title
                title_elem = entry.find("title")
                title_text = title_elem.text.strip() if title_elem else ""
                
                # Abstract
                summary_elem = entry.find("summary")
                abstract_text = summary_elem.text.strip() if summary_elem else ""
                
                # Authors
                authors = []
                for author in entry.find_all("author"):
                    name_elem = author.find("name")
                    if name_elem:
                        authors.append(name_elem.text.strip())
                
                # Publication year
                published_elem = entry.find("published")
                year = None
                if published_elem:
                    try:
                        pub_date = published_elem.text.strip()
                        match = re.search(r"(\d{4})", pub_date)
                        if match:
                            year = int(match.group(1))
                    except ValueError:
                        pass
                
                # DOI and URL
                doi = None
                url = None
                for link in entry.find_all("link"):
                    href = link.get("href", "")
                    if link.get("title") == "doi":
                        doi = href.replace("http://dx.doi.org/", "")
                    elif link.get("rel") == "alternate":
                        url = href.replace("/abs/", "/pdf/")
                
                # Create paper metadata
                paper = PaperMetadata(
                    title=title_text,
                    authors=authors,
                    abstract=abstract_text,
                    year=year,
                    doi=doi,
                    journal="arXiv",
                    url=url,
                )
                papers.append(paper)
                
            except Exception as e:
                logger.error(f"Error parsing arXiv entry: {str(e)}")
        
        return papers
    
    def clear_cache(self) -> None:
        """Clear the search cache."""
        self._cache.clear()

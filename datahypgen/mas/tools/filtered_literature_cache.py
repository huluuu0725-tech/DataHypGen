"""Persistent pool of previously filtered literature evidence."""

import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union


logger = logging.getLogger(__name__)


class FilteredLiteratureCache:
    """Store reusable papers independently of the query that first found them."""

    INDEX_FILENAME = "filtered_literature.json"
    PAYLOAD_VERSION = "filtered_literature_index_v1"
    PAPER_FIELDS = (
        "source",
        "query",
        "rank",
        "title",
        "abstract",
        "year",
        "doi",
        "url",
        "journal",
        "methods",
        "applications",
        "datasets_used",
        "score",
        "similarity_or_rank",
        "retrieval_source",
        "retrieval_fallback",
        "cache_match_score",
    )

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path).expanduser().resolve()

    @staticmethod
    def paper_identity_key(paper: Dict[str, Any]) -> str:
        if not isinstance(paper, dict):
            return ""
        doi = str(paper.get("doi") or "").strip().lower()
        if doi:
            return f"doi:{doi}"
        url = str(paper.get("url") or "").strip().lower()
        if url:
            return f"url:{url}"
        title = " ".join(str(paper.get("title") or "").lower().split())
        return f"title:{title}" if title else ""

    def load(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            with self.path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            logger.warning("Failed to read filtered literature cache %s: %s", self.path, exc)
            return []

        papers = payload.get("papers", []) if isinstance(payload, dict) else payload
        if not isinstance(papers, list):
            logger.warning("Invalid filtered literature cache payload at %s", self.path)
            return []
        return [copy.deepcopy(item) for item in papers if isinstance(item, dict)]

    def add_papers(
        self,
        papers: List[Dict[str, Any]],
        *,
        source_stage: str,
        task_name: str = "",
        system: str = "",
    ) -> int:
        if not isinstance(papers, list) or not papers:
            return 0

        existing = self.load()
        by_key = {
            self.paper_identity_key(item): copy.deepcopy(item)
            for item in existing
            if self.paper_identity_key(item)
        }
        written_count = 0
        for paper in papers:
            normalized = self._normalize_paper(
                paper,
                source_stage=source_stage,
                task_name=task_name,
                system=system,
            )
            key = self.paper_identity_key(normalized)
            if not key:
                continue
            written_count += 1
            if key not in by_key:
                by_key[key] = normalized
                continue
            current = by_key[key]
            for field in self.PAPER_FIELDS:
                if current.get(field) in (None, "", [], {}) and normalized.get(field) not in (
                    None,
                    "",
                    [],
                    {},
                ):
                    current[field] = copy.deepcopy(normalized[field])
            provenance = list(current.get("cache_provenance") or [])
            provenance.extend(normalized.get("cache_provenance") or [])
            current["cache_provenance"] = self._dedupe_provenance(provenance)

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
            with temporary_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "cache_payload_version": self.PAYLOAD_VERSION,
                        "papers": sorted(
                            by_key.values(),
                            key=lambda item: str(item.get("title") or "").lower(),
                        ),
                    },
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
            temporary_path.replace(self.path)
        except Exception as exc:
            logger.warning("Failed to write filtered literature cache %s: %s", self.path, exc)
            return 0
        return written_count

    @classmethod
    def _normalize_paper(
        cls,
        paper: Dict[str, Any],
        *,
        source_stage: str,
        task_name: str,
        system: str,
    ) -> Dict[str, Any]:
        if not isinstance(paper, dict):
            return {}
        title = str(paper.get("title") or "").strip()
        if not title:
            return {}
        normalized = {
            field: copy.deepcopy(paper[field])
            for field in cls.PAPER_FIELDS
            if paper.get(field) not in (None, "", [], {})
        }
        normalized["title"] = title
        abstract = paper.get("abstract")
        if isinstance(abstract, str) and abstract.strip():
            normalized["abstract"] = abstract.strip()
        if not normalized.get("source"):
            normalized["source"] = str(
                paper.get("retrieval_source") or source_stage or "filtered_literature_cache"
            )
        normalized["cache_source_stage"] = source_stage
        provenance = list(paper.get("cache_provenance") or [])
        provenance.append(
            {
                "source_stage": source_stage,
                "task_name": task_name,
                "system": system,
            }
        )
        normalized["cache_provenance"] = cls._dedupe_provenance(provenance)
        return normalized

    @staticmethod
    def _dedupe_provenance(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        deduped: List[Dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            key = (
                str(item.get("source_stage") or ""),
                str(item.get("task_name") or ""),
                str(item.get("system") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(
                {
                    "source_stage": key[0],
                    "task_name": key[1],
                    "system": key[2],
                }
            )
        return deduped

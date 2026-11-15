"""
Utility Tools for Scientific Literature Management

This module provides a comprehensive suite of utility functions and classes for managing
scientific literature, including:
- Paper metadata structures (PaperMetadata dataclass)
- Multi-source paper search (openalex, arXiv)
- PDF downloading and text extraction
- Paper filtering and deduplication
- Citation formatting (APA, BibTeX)
- Query parsing and execution
- DOI resolution and publisher page scraping

These utilities support the literature search and survey capabilities of the DataGenHyp system.
"""

import logging
import re
import os
import time
import requests
import httpx
import subprocess
from pathlib import Path
try:
    import pdfplumber
except ImportError:  # pragma: no cover - optional dependency in test/runtime environments
    pdfplumber = None
from urllib.parse import urljoin
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import random

from bs4 import BeautifulSoup


_arxiv_last_request_at = 0.0

logger = logging.getLogger(__name__)

# Define the paper search endpoint URL
search_url = 'https://api.semanticscholar.org/graph/v1/paper/search/'
graph_url = 'https://api.semanticscholar.org/graph/v1/paper/'
rec_url = "https://api.semanticscholar.org/recommendations/v1/papers/forpaper/"

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
    source: Optional[str] = None
    url: Optional[str] = None
    
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
            "source": self.source,
            "url": self.url,
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
    
# Search tools

def _get_arxiv_request_config() -> tuple[str, dict, float]:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[3] / ".env", override=False)
    required_names = (
        "ARXIV_API_URL",
        "ARXIV_CONTACT_EMAIL",
        "ARXIV_REQUEST_INTERVAL_SECONDS",
    )
    values = {}
    for name in required_names:
        value = os.getenv(name)
        if value is None or not value.strip():
            raise RuntimeError(f"Required environment variable {name} is not configured")
        values[name] = value.strip()
    api_url = values["ARXIV_API_URL"]
    if not api_url.startswith("https://"):
        raise ValueError("ARXIV_API_URL must use HTTPS")
    contact_email = values["ARXIV_CONTACT_EMAIL"]
    if "@" not in contact_email:
        raise ValueError("ARXIV_CONTACT_EMAIL must be a valid contact email")
    try:
        request_interval_seconds = float(values["ARXIV_REQUEST_INTERVAL_SECONDS"])
    except ValueError as exc:
        raise ValueError("ARXIV_REQUEST_INTERVAL_SECONDS must be numeric") from exc
    if request_interval_seconds <= 0.0:
        raise ValueError("ARXIV_REQUEST_INTERVAL_SECONDS must be greater than zero")
    headers = {
        "User-Agent": f"DataGenHyp/1.0 (mailto:{contact_email})",
        "Accept": "application/atom+xml",
        "Accept-Encoding": "gzip, deflate",
    }
    return api_url, headers, request_interval_seconds


def fetch_arxiv_papers(query: str, max_results: int = 20, sort: str = "relevance", categories: list = None) -> list:
    """
    Fetch papers from arXiv based on the query.
    
    Args:
        query: Search query
        max_results: Maximum number of results (default: 20)
        sort: Sort order ("relevance" or "date")
        categories: List of arXiv categories to search (default: None)
    
    Returns:
        List of paper metadata in JSON format
    """
    logger.info(f"Searching arXiv for: {query}")
    
    search_url, headers, request_interval_seconds = _get_arxiv_request_config()
    
    # Sort parameter
    sort_param = "relevance" if sort == "relevance" else "submittedDate"
    
    # Category filter
    cat_filter = ""
    if categories:
        cat_filter = " AND (" + " OR ".join([f"cat:{cat}" for cat in categories]) + ")"
    
    # Search parameters
    search_params = {
        "search_query": f"all:{query}{cat_filter}",
        "max_results": max_results,
        "sortBy": sort_param,
        "sortOrder": "descending"
    }
    
    try:
        global _arxiv_last_request_at
        elapsed = time.monotonic() - _arxiv_last_request_at
        remaining = request_interval_seconds - elapsed
        if remaining > 0.0:
            time.sleep(remaining)
        _arxiv_last_request_at = time.monotonic()
        response = requests.get(search_url, params=search_params, headers=headers)
        if response.status_code != 200:
            logger.error(f"arXiv search error: {response.status_code}")
            return []
        
        xml_data = response.text
        papers = parse_arxiv_xml(xml_data)  # 假设你有一个解析函数
        
        logger.info(f"Get {len(papers)} papers from arXiv")

        return papers
    
    except Exception as e:
        logger.error(f"Error searching arXiv: {e}")
        return []

def select_papers(paper_bank, max_papers, rag_read_depth):
    selected_for_deep_read = []
    count = 0
    for paper in sorted(paper_bank, key=lambda x: x.get("score", float("-inf")), reverse=True):
        if count >= rag_read_depth:
            break

        open_access = paper.get("open_access") or {}
        download_url = (
            open_access.get("pdf_url")
            or open_access.get("oa_url")
            or paper.get("url")
        )
        doi = paper.get("doi")

        if download_url or doi:
            selected_for_deep_read.append(paper)
            count += 1

    selected_for_deep_read = selected_for_deep_read[:max_papers]
    return selected_for_deep_read

def parse_arxiv_xml(xml_data: str) -> list:
    
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
                    url = href.replace("abs", "pdf")
            
            paper = PaperMetadata(
                    title=title_text,
                    authors=authors,
                    abstract=abstract_text,
                    year=year,
                    doi=doi,
                    journal="arXiv",
                    url=url,
                    source='arXiv'
                )
            papers.append(paper.to_dict())# NOTE: placeholder for paper metadata 
            
        except Exception as e:
            logger.error(f"Error parsing arXiv entry: {str(e)}")
    
    return papers


# IO tools

def parse_io_description(output):
    match_input = re.match(r'Input\("([^"]+)"\)', output)
    input_description = match_input.group(1) if match_input else None
    match_output = re.match(r'.*Output\("([^"]+)"\)', output)
    output_description = match_output.group(1) if match_output else None
    return input_description, output_description


def format_papers_for_printing(paper_lst, include_abstract=True, include_score=True, include_id=True):
    """
    Convert a list of papers to a string for printing or as part of a prompt.
    """
    output_str = ""
    for idx, paper in enumerate(paper_lst):
        # if include_id and "paperId" in paper:
        #     output_str += "paperId: " + paper["paperId"].strip() + "\n"
        if include_id:
            output_str += "paperId: " + str(idx) + "\n" 
        elif include_id and "title" in paper:
            output_str += "paperId: " + paper["title"].strip() + "\n"
        
        output_str += "title: " + paper.get("title", "").strip() + "\n"
        
        if include_abstract:
            if "abstract" in paper and paper["abstract"]:
                output_str += "abstract: " + paper["abstract"].strip() + "\n"
            elif "tldr" in paper and paper["tldr"] and paper["tldr"].get("text"):
                output_str += "tldr: " + paper["tldr"]["text"].strip() + "\n"
        
        if "year" in paper:
            output_str += "year: " + str(paper["year"]) + "\n"
        
        if "score" in paper and include_score:
            output_str += "relevance score: " + str(paper["score"]) + "\n"
        
        output_str += "\n"
    
    return output_str

def format_papers_for_printing_next_query(paper_lst, include_abstract=True, include_score=True, include_id=True):
    """
    Convert a list of papers to a string for printing or as part of a prompt.
    """
    output_str = ""
    for idx, paper in enumerate(paper_lst):
        if include_id:
            output_str += "paperId: " + str(idx) + "\n" 
        elif include_id and "title" in paper:
            output_str += "paperId: " + paper["title"].strip() + "\n"
        
        output_str += "title: " + paper.get("title", "").strip() + "\n"
        
        output_str += "\n"
    
    return output_str

def print_top_papers_from_paper_bank(paper_bank, top_k=10):
    data_list = [{'id': id, **info} for id, info in paper_bank.items()]
    top_papers = sorted(data_list, key=lambda x: x['score'], reverse=True)[: top_k]
    logger.debug(format_papers_for_printing(top_papers, include_abstract=False))


def dedup_paper_bank(sorted_paper_bank):
    idx_to_remove = []

    for i in reversed(range(len(sorted_paper_bank))):
        for j in range(i):
            if sorted_paper_bank[i]["paperId"].strip() == sorted_paper_bank[j]["paperId"].strip():
                idx_to_remove.append(i)
                break
            if ''.join(sorted_paper_bank[i]["title"].lower().split()) == ''.join(
                    sorted_paper_bank[j]["title"].lower().split()):
                idx_to_remove.append(i)
                break
            if sorted_paper_bank[i]["abstract"] == sorted_paper_bank[j]["abstract"]:
                idx_to_remove.append(i)
                break

    deduped_paper_bank = [paper for i, paper in enumerate(sorted_paper_bank) if i not in idx_to_remove]
    return deduped_paper_bank


def download_pdf(pdf_url, save_folder="pdfs"):
    logger.info(f"downloading pdf from {pdf_url}")
    
    if not pdf_url:
        return None
    
    os.makedirs(save_folder, exist_ok=True)
    
    file_name = pdf_url.split("/")[-1].split("?")[0]
    if not file_name.endswith('.pdf'):
        file_name = file_name + '.pdf'
    save_path = os.path.join(save_folder, file_name)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36",
    }
    try:
        response = httpx.get(url=pdf_url,headers=headers, timeout=10, verify=False)
        if response.status_code != 200:
            logger.error(f"Failed to download PDF from {pdf_url}: {response.status_code}")
            return None

        content_type = response.headers.get("content-type", "").lower()
        is_pdf_response = "application/pdf" in content_type or response.content.startswith(b"%PDF-")
        if not is_pdf_response:
            logger.warning(
                f"Skipping non-PDF response from {pdf_url}: content-type={content_type or 'unknown'}"
            )
            return None

        with open(save_path, "wb") as file:
            file.write(response.content)
        return save_path
    except Exception as e:
        logger.error(f"Error downloading PDF from {pdf_url}: {e}")
        return None
    
    
def download_pdf_by_doi(doi: str, download_dir: str = "downloaded_papers"):
    doi = doi.strip()
    if doi.lower().startswith('doi:'):
        doi = doi[4:].strip()
    if doi.lower().startswith('https://doi.org/'):
        doi = doi[16:].strip()

    doi_url = f"https://doi.org/{doi}"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(doi_url, headers=headers, allow_redirects=True, timeout=20)
        response.raise_for_status()
        publisher_url = response.url
        logger.info(f"Redirected to the publisher page: {publisher_url}")

        soup = BeautifulSoup(response.text, 'html.parser')
        pdf_links = []

        for link in soup.find_all('a', href=True):
            href = link['href']
            link_text = link.get_text().lower()
            if ('pdf' in href.lower() or
                'pdf' in link_text or
                'download' in link_text and ('full' in link_text or 'article' in link_text) or
                'full text' in link_text):
                pdf_links.append(urljoin(publisher_url, href))

        if not pdf_links:
            logger.warning(f"No PDF link found on DOI landing page: {doi_url}")
            return None

        logger.info(f"Found {len(pdf_links)} candidate PDF links for DOI {doi}")
        pdf_url = pdf_links[0]
        logger.info(f"Attempting DOI PDF download: {pdf_url}")

        pdf_response = requests.get(pdf_url, headers=headers, stream=True, timeout=20)
        pdf_response.raise_for_status()
        if 'application/pdf' not in pdf_response.headers.get('Content-Type', ''):
            logger.warning(
                f"Skipping DOI PDF download with non-PDF content-type for {doi}: "
                f"{pdf_response.headers.get('Content-Type', 'unknown')}"
            )
            return None

        os.makedirs(download_dir, exist_ok=True)

        filename = f"{doi.replace('/', '_')}.pdf"
        filepath = os.path.join(download_dir, filename)

        with open(filepath, 'wb') as f:
            for chunk in pdf_response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        logger.info(f"PDF downloaded to: {filepath}")
        return filepath
    except Exception as e:
        logger.warning(f"Failed to download PDF by DOI {doi}: {e}")
        return None

def extract_text_from_pdf(pdf_path):
    try:
        if pdfplumber is None:
            logger.warning("pdfplumber is not installed; skipping PDF text extraction for %s", pdf_path)
            return None
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None
    
    
def get_pdf_url(paper_id, max_retries=5):

    base_url = "https://api.semanticscholar.org/graph/v1/paper/"
    url = f"{base_url}{paper_id}"
    params = {"fields": "openAccessPdf"}  

    headers = {'x-api-key': os.environ['S2_API_KEY']}
    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200:
        data = response.json()
        return data.get("openAccessPdf", {}).get("url")

    elif response.status_code == 429:
        attempt = 0
        while attempt < max_retries:
            print("Rate limit exceeded. Sleeping for 10 seconds...")
            time.sleep(10) 
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                return data.get("openAccessPdf", {}).get("url")
            attempt += 1
        print("Max retries exceeded. Could not retrieve PDF URL.")
        return None

    else:
        print(f"Failed to retrieve PDF URL. Status code: {response.status_code}")
        return None

        
def PaperQuery(paper_id):
    query_params = {
        'paperId': paper_id,
        'limit': 20,
        'fields': 'title,year,citationCount,abstract'
    }
    headers = {'x-api-key': os.environ['S2_API_KEY']}
    response = requests.get(url=rec_url + paper_id, params=query_params, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        return None


def PaperDetails(paper_id, fields='title,year,abstract,authors,citationCount,venue,tldr'):

    ## get paper details based on paper id
    paper_data_query_params = {'fields': fields}
    headers = {'x-api-key': os.environ['S2_API_KEY']}
    response = requests.get(url=graph_url + paper_id, params=paper_data_query_params, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        return None


def GetAbstract(paper_id):
    ## get the abstract of a paper based on paper id
    paper_details = PaperDetails(paper_id)

    if paper_details is not None:
        return paper_details["abstract"]
    else:
        return None


def GetCitationCount(paper_id):
    ## get the citation count of a paper based on paper id
    paper_details = PaperDetails(paper_id)

    if paper_details is not None:
        return int(paper_details["citationCount"])
    else:
        return None



def is_valid_paper(paper):
    paper = paper
    # Check for specific keywords indicating non-research papers
    title = paper.get("title", "").lower() if paper.get("title") else ""
    abstract = paper.get("abstract", "").lower() if paper.get("abstract") else ""
    if ("survey" in title or "survey" in abstract or
        "review" in title or "review" in abstract or
        "position paper" in title or "position paper" in abstract):
        return False
    
    # Check abstract length (new rule)
    if len(abstract.split()) <= 50:
        return False
    
    return True

def paper_filter(paper_lst):
    """
    Filter out papers based on some basic heuristics.
    Args:
        paper_lst (dict): A dictionary where keys are sources (e.g.,'arxiv') and values are lists of papers.
    Returns:
        dict: A dictionary with the same structure as input, but with filtered papers.
    """
    filtered_paper_lst = {}
    
    # Iterate through each source and filter papers
    for source, papers in paper_lst.items():
        if isinstance(papers, list):  # Ensure the value is a list
            filtered_papers = [paper for paper in papers if is_valid_paper(paper)]
            filtered_paper_lst[source] = filtered_papers
        else:
            # If the value is not a list, skip or handle differently
            filtered_paper_lst[source] = papers  # Keep the original structure
    
    # print("Filtered paper list: ", filtered_paper_lst)
    return filtered_paper_lst

'''
def multi_source_search(query: str, sources: list[str] = None, max_results: int = 10, **kwargs) -> dict[str, list[dict]]:
    
    if not sources:
        sources = ["arxiv", "openalex"]
    
    combined_results = {}
    
    for source in sources:
        if source == "arxiv":
            combined_results[source] = fetch_arxiv_papers(query, max_results, **kwargs)
        elif source == "openalex":
            combined_results[source] = fetch_openalex_papers(query, max_results, **kwargs)
        else:
            logger.warning(f"Unknown source: {source}. Skipping.")
    
    return combined_results
'''

def parse_and_execute(output, max_results):
    ## parse gpt4 output and execute corresponding functions
    if output.startswith("KeywordQuery"):
        match = re.match(r'KeywordQuery\("([^"]+)"\)', output)
        keyword = match.group(1) if match else None
        if keyword:
            response = multi_source_search(keyword, max_results=max_results)
            if response is not None:
                paper_lst = response
            # print("paper_lst: ",paper_lst)
            return paper_filter(paper_lst)
        else:
            return None
    elif output.startswith("PaperQuery"):
        match = re.match(r'PaperQuery\("([^"]+)"\)', output)
        paper_id = match.group(1) if match else None
        if paper_id:
            response = PaperQuery(paper_id)
            if response is not None and response["recommendedPapers"]:
                paper_lst = response["recommendedPapers"]
                return paper_filter(paper_lst)
    elif output.startswith("GetAbstract"):
        match = re.match(r'GetAbstract\("([^"]+)"\)', output)
        paper_id = match.group(1) if match else None
        if paper_id:
            return GetAbstract(paper_id)
    elif output.startswith("GetCitationCount"):
        match = re.match(r'GetCitationCount\("([^"]+)"\)', output)
        paper_id = match.group(1) if match else None
        if paper_id:
            return GetCitationCount(paper_id)
    return None

def replace_and_with_or(query, max_keep=1):
    parts = query.split(" AND ")
    
    if len(parts) <= max_keep + 1:
        return query
    
    if max_keep > 0:
        keep_positions = random.sample(range(len(parts) - 1), max_keep)
    else:
        keep_positions = []
    
    result = parts[0]
    for i in range(len(parts) - 1):
        if i in keep_positions:
            result += " AND " + parts[i + 1]  # 保留 AND
        else:
            result += " OR " + parts[i + 1]  # 将 AND 替换为 OR
    
    return result

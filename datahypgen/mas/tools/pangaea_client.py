"""PANGAEA search and DOI metadata access, independent of the local catalog.

Search parameters follow the official pangaeapy PanQuery client. DataCite
relations retain their direction and type; they do not assert empirical use.
"""

import asyncio
import logging
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List

import aiohttp

SEARCH_URL = "https://www.pangaea.de/advanced/search.php"
NS = {"d": "http://datacite.org/schema/kernel-4"}
logger = logging.getLogger(__name__)


def normalize_pangaea_doi(value: str) -> str:
    value = re.sub(r"^(?:PANGAEA:|doi:|https?://(?:doi\.org|doi\.pangaea\.de)/)", "", value.strip(), flags=re.I)
    if value.isdigit():
        value = f"10.1594/PANGAEA.{value}"
    if not re.fullmatch(r"10\.1594/PANGAEA\.\d+", value, re.I):
        raise ValueError("Expected a PANGAEA dataset DOI or numeric dataset ID")
    return "10.1594/PANGAEA." + value.rsplit(".", 1)[1]


def parse_pangaea_metadata(xml: str) -> Dict[str, Any]:
    root = ET.fromstring(xml)
    def text(path):
        return (root.findtext(path, default="", namespaces=NS)).strip()
    doi = normalize_pangaea_doi(text("d:identifier"))
    relations = [
        {"source": doi, "relation_type": node.get("relationType", ""),
         "target": (node.text or "").strip(),
         "target_identifier_type": node.get("relatedIdentifierType", ""),
         "target_resource_type": node.get("resourceTypeGeneral", ""),
         "provenance": "PANGAEA DataCite metadata"}
        for node in root.findall("d:relatedIdentifiers/d:relatedIdentifier", NS)
    ]
    subjects = root.findall("d:subjects/d:subject", NS)
    boxes = []
    for node in root.findall("d:geoLocations/d:geoLocation/d:geoLocationBox", NS):
        values = [node.findtext("d:" + key, namespaces=NS) for key in
                  ("westBoundLongitude", "southBoundLatitude", "eastBoundLongitude", "northBoundLatitude")]
        if all(value is not None for value in values):
            boxes.append([float(value) for value in values])
    for node in root.findall("d:geoLocations/d:geoLocation/d:geoLocationPoint", NS):
        lon, lat = node.findtext("d:pointLongitude", namespaces=NS), node.findtext("d:pointLatitude", namespaces=NS)
        if lon is not None and lat is not None:
            boxes.append([float(lon), float(lat), float(lon), float(lat)])
    bbox = [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)] if boxes else None
    collected = [node.text.strip() for node in root.findall("d:dates/d:date", NS)
                 if node.get("dateType") == "Collected" and node.text]
    intervals = [value.split("/", 1) for value in collected]
    rights = root.find("d:rightsList/d:rights", NS)
    url = f"https://doi.pangaea.de/{doi}"
    resource = root.find("d:resourceType", NS)
    resource_type = (resource.text or "").strip() if resource is not None else ""
    is_collection = "series" in resource_type.lower() or any(r["relation_type"] == "HasPart" for r in relations)
    formats = [(node.text or "").strip() for node in root.findall("d:formats/d:format", NS)]
    return {
        "id": doi, "doi": doi, "source": "PANGAEA",
        "type": "table" if "text/tab-separated-values" in formats else "dataset",
        "title": text("d:titles/d:title"), "provider": text("d:publisher"),
        "description": "\n".join(" ".join(node.itertext()).strip() for node in
                                    root.findall("d:descriptions/d:description", NS)),
        "tags": list(dict.fromkeys((node.text or "").strip() for node in subjects if node.text)),
        "parameters": [(node.text or "").strip() for node in subjects if node.get("subjectScheme") == "Parameter"],
        "bbox": bbox,
        "region_name": "; ".join((node.text or "").strip() for node in root.findall("d:geoLocations/d:geoLocation/d:geoLocationPlace", NS)),
        # Collection dates are not necessarily the time represented by the data.
        "start_date": "", "end_date": "",
        "collected_start": min((v[0] for v in intervals), default=""),
        "collected_end": max((v[-1] for v in intervals), default=""),
        "license": rights.get("rightsIdentifier", "") if rights is not None else "",
        "terms_of_use": rights.get("rightsURI", "") if rights is not None else "",
        "docs": url, "metadata_url": url + "?format=metadata_datacite4",
        "data_url": url + "?format=textfile" if not is_collection and "text/tab-separated-values" in formats else "",
        "is_collection": is_collection,
        "resource_type": resource_type, "formats": formats,
        "relations": relations,
        "access_note": "Data access must be verified separately; a DOI or metadata response does not establish unrestricted download access.",
    }


async def _metadata(session, doi):
    url = f"https://doi.pangaea.de/{normalize_pangaea_doi(doi)}"
    async with session.get(url, params={"format": "metadata_datacite4"}) as response:
        response.raise_for_status()
        record = parse_pangaea_metadata(await response.text())
    if record["doi"] != normalize_pangaea_doi(doi):
        raise ValueError("PANGAEA metadata returned a different dataset DOI")
    async with session.get(url, params={"format": "metadata_jsonld"}) as response:
        response.raise_for_status()
        linked = await response.json(content_type=None)
    return enrich_pangaea_links(record, linked)


def enrich_pangaea_links(record: Dict[str, Any], linked: Dict[str, Any]) -> Dict[str, Any]:
    def objects(value):
        if isinstance(value, list):
            return value
        return [value] if value is not None else []
    record["distributions"] = [
        {"url": item["contentUrl"], "format": item.get("encodingFormat", "")}
        for item in objects(linked.get("distribution"))
        if isinstance(item, dict) and isinstance(item.get("contentUrl"), str)
    ]
    record["data_url"] = record["distributions"][0]["url"] if record["distributions"] else ""
    record["conditions_of_access"] = linked.get("conditionsOfAccess", "")
    record["is_accessible_for_free"] = linked.get("isAccessibleForFree")
    for key, relation_type in (("hasPart", "HasPart"), ("isPartOf", "IsPartOf")):
        for item in objects(linked.get(key)):
            target = item.get("@id", item.get("identifier", item.get("url", ""))) if isinstance(item, dict) else item
            if isinstance(target, str) and target:
                record["relations"].append({
                    "source": record["doi"], "relation_type": relation_type,
                    "target": target, "target_identifier_type": "URL",
                    "target_resource_type": "", "provenance": "PANGAEA JSON-LD metadata",
                })
    if any(r["relation_type"] == "HasPart" for r in record["relations"]):
        record["is_collection"] = True
    return record


async def query_pangaea(dataset_id: str) -> Dict[str, Any]:
    async with aiohttp.ClientSession(trust_env=True, timeout=aiohttp.ClientTimeout(total=45)) as session:
        record = await _metadata(session, dataset_id)
    return {"source": "PANGAEA", "dataset_id": record["id"], "record": record,
            "title": record["title"], "description": record["description"],
            "relations": record["relations"], "metadata_url": record["metadata_url"]}


async def search_pangaea(query: str, *, limit: int = 5, bbox=None) -> List[Dict[str, Any]]:
    if not query.strip():
        return []
    if not 1 <= limit <= 500:
        raise ValueError("PANGAEA search limit must be between 1 and 500")
    async with aiohttp.ClientSession(trust_env=True, timeout=aiohttp.ClientTimeout(total=45)) as session:
        if re.fullmatch(r"(?:https?://(?:doi\.org|doi\.pangaea\.de)/|doi:|PANGAEA:)?(?:10\.1594/PANGAEA\.)?\d+", query.strip(), re.I):
            return [await _metadata(session, query)]
        params = {"q": query, "count": limit, "offset": 0}
        if bbox is not None:
            if len(bbox) != 4:
                raise ValueError("PANGAEA bbox requires west, south, east, north")
            params.update(dict(zip(("minlon", "minlat", "maxlon", "maxlat"), bbox)))
        async with session.get(SEARCH_URL, params=params) as response:
            response.raise_for_status()
            payload = await response.json(content_type=None)
        if not isinstance(payload.get("results"), list):
            raise ValueError("Invalid PANGAEA search response: missing results")
        # Bound metadata enrichment concurrency.
        semaphore = asyncio.Semaphore(3)
        async def enrich(hit):
            async with semaphore:
                return await _metadata(session, hit["URI"])
        results = await asyncio.gather(*(enrich(hit) for hit in payload["results"][:limit]), return_exceptions=True)
        records = []
        for hit, result in zip(payload["results"][:limit], results):
            if isinstance(result, Exception):
                logger.warning("PANGAEA metadata enrichment failed for %s: %s", hit.get("URI"), result)
            else:
                records.append(result)
        if results and not records:
            raise RuntimeError("PANGAEA search succeeded but all dataset metadata requests failed")
        return records

"""
Entity normalization for drug and condition names.

Uses the ClinicalTrials.gov autocomplete API to resolve informal names,
brand names, and abbreviations to their canonical forms before the
planning step. This prevents hallucination from synonym ambiguity.

Examples:
  "keytruda" -> "Pembrolizumab"
  "opdivo"   -> "Nivolumab"
  "T2D"      -> "Type 2 Diabetes"
"""

import logging
from functools import partial
import asyncio
import requests

logger = logging.getLogger(__name__)

CT_AUTOCOMPLETE_URL = "https://clinicaltrials.gov/api/v2/stats/fieldValues"
CT_AUTOCOMPLETE_SUGGEST = "https://clinicaltrials.gov/api/v2/autocomplete"

# Common brand name -> generic name mappings as a fast local cache
# Covers the most frequent cases without an API call
BRAND_TO_GENERIC: dict[str, str] = {
    "keytruda": "Pembrolizumab",
    "opdivo": "Nivolumab",
    "yervoy": "Ipilimumab",
    "avastin": "Bevacizumab",
    "herceptin": "Trastuzumab",
    "rituxan": "Rituximab",
    "humira": "Adalimumab",
    "ozempic": "Semaglutide",
    "wegovy": "Semaglutide",
    "mounjaro": "Tirzepatide",
    "zepbound": "Tirzepatide",
    "tecentriq": "Atezolizumab",
    "imfinzi": "Durvalumab",
    "bavencio": "Avelumab",
    "libtayo": "Cemiplimab",
    "lynparza": "Olaparib",
    "ibrance": "Palbociclib",
    "kisqali": "Ribociclib",
    "verzenio": "Abemaciclib",
    "tagrisso": "Osimertinib",
    "gleevec": "Imatinib",
    "gleevec": "Imatinib",
}

# Common abbreviations and informal names for conditions
CONDITION_ALIASES: dict[str, str] = {
    "t2d": "type 2 diabetes",
    "t1d": "type 1 diabetes",
    "nsclc": "non-small cell lung cancer",
    "sclc": "small cell lung cancer",
    "tnbc": "triple negative breast cancer",
    "crc": "colorectal cancer",
    "hcc": "hepatocellular carcinoma",
    "rcc": "renal cell carcinoma",
    "aml": "acute myeloid leukemia",
    "cll": "chronic lymphocytic leukemia",
    "dlbcl": "diffuse large b-cell lymphoma",
    "ra": "rheumatoid arthritis",
    "ms": "multiple sclerosis",
    "copd": "chronic obstructive pulmonary disease",
    "ckd": "chronic kidney disease",
    "hf": "heart failure",
    "af": "atrial fibrillation",
    "mdd": "major depressive disorder",
    "ptsd": "post-traumatic stress disorder",
    "ad": "alzheimer's disease",
    "pd": "parkinson's disease",
}


def _sync_autocomplete(term: str, category: str) -> str | None:
    """
    Call ClinicalTrials.gov autocomplete API to resolve a term.
    Returns the top canonical result or None if no match.
    category: 'InterventionName' or 'Condition'
    """
    try:
        resp = requests.get(
            CT_AUTOCOMPLETE_SUGGEST,
            params={"type": category, "query": term, "limit": 1},
            timeout=5,
        )
        if resp.status_code == 200:
            data = resp.json()
            hits = data.get("hits", [])
            if hits:
                canonical = hits[0].get("term") or hits[0].get("value")
                if canonical and canonical.lower() != term.lower():
                    logger.info(f"Normalized '{term}' -> '{canonical}' via CT autocomplete")
                    return canonical
    except Exception as e:
        logger.debug(f"Autocomplete lookup failed for '{term}': {e}")
    return None


async def normalize_drug(name: str | None) -> str | None:
    """Normalize a drug/intervention name to its canonical form."""
    if not name:
        return name

    # Check local cache first
    lower = name.lower().strip()
    if lower in BRAND_TO_GENERIC:
        canonical = BRAND_TO_GENERIC[lower]
        logger.info(f"Normalized drug '{name}' -> '{canonical}' via local cache")
        return canonical

    # Fall back to CT autocomplete API
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, partial(_sync_autocomplete, name, "InterventionName"))
    return result or name


async def normalize_condition(name: str | None) -> str | None:
    """Normalize a condition/disease name to its canonical form."""
    if not name:
        return name

    lower = name.lower().strip()
    if lower in CONDITION_ALIASES:
        canonical = CONDITION_ALIASES[lower]
        logger.info(f"Normalized condition '{name}' -> '{canonical}' via local cache")
        return canonical

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, partial(_sync_autocomplete, name, "Condition"))
    return result or name


async def normalize_request_entities(
    query: str,
    drug_name: str | None,
    condition: str | None,
) -> tuple[str, str | None, str | None]:
    """
    Normalize drug and condition names in parallel.
    Returns (query, normalized_drug, normalized_condition).
    Query text is returned unchanged — normalization applies to structured fields only.
    """
    drug_task = normalize_drug(drug_name)
    condition_task = normalize_condition(condition)

    normalized_drug, normalized_condition = await asyncio.gather(drug_task, condition_task)
    return query, normalized_drug, normalized_condition
"""
Agent tools — the functions the LLM can invoke during the agentic loop.

Three tools, coarse-grained by design:
  - search_trials: fetches studies from ClinicalTrials.gov
  - aggregate: groups and counts studies by a field to produce viz data points
  - get_study_details: fetches a single study for citation excerpts

Each tool has:
  - A JSON schema definition (passed to the Anthropic API as tool definitions)
  - An async implementation function
  - Pydantic input validation before touching the API
"""

from typing import Any
import logging

from app.client.ct_client import ClinicalTrialsClient
from app.schemas.ct_api import CTSearchParams, CTStudy
from app.schemas.citations import Citation

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Tool JSON schema definitions — passed to Anthropic API
# -----------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "search_trials",
        "description": (
            "Search ClinicalTrials.gov for studies matching the given filters. "
            "Returns a list of normalized study records. "
            "Use this first to retrieve the raw data before aggregating."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "interventions": {
                    "type": "string",
                    "description": "Drug or intervention name to search for.",
                },
                "conditions": {
                    "type": "string",
                    "description": "Disease or condition to search for.",
                },
                "sponsor": {
                    "type": "string",
                    "description": "Sponsor organization name.",
                },
                "phase": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of phases to filter by, e.g. ['PHASE2', 'PHASE3'].",
                },
                "country": {
                    "type": "string",
                    "description": "Country name to filter by.",
                },
                "start_date_gte": {
                    "type": "string",
                    "description": "Filter studies starting on or after this date (YYYY-MM-DD).",
                },
                "start_date_lte": {
                    "type": "string",
                    "description": "Filter studies starting on or before this date (YYYY-MM-DD).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "aggregate",
        "description": (
            "Aggregate a list of study records by a specified field to produce "
            "visualization data points. Returns counts grouped by the field value. "
            "Call this after search_trials once you have the raw studies."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "studies_key": {
                    "type": "string",
                    "description": (
                        "Key referencing which search result to aggregate. "
                        "Use 'primary' for the first search, 'secondary' for the second."
                    ),
                },
                "field": {
                    "type": "string",
                    "enum": [
                        "phase",
                        "status",
                        "sponsor_name",
                        "sponsor_class",
                        "start_year",
                        "country",
                        "condition",
                        "intervention",
                        "enrollment_bucket",
                    ],
                    "description": "The field to group and count studies by.",
                },
                "top_n": {
                    "type": "integer",
                    "description": "Return only the top N groups by count. Omit for all groups.",
                },
                "label": {
                    "type": "string",
                    "description": (
                        "Label for this series in the visualization, used when "
                        "producing grouped bar charts from multiple searches."
                    ),
                },
            },
            "required": ["studies_key", "field"],
        },
    },
    {
        "name": "get_study_details",
        "description": (
            "Fetch detailed information for a single study by NCT ID. "
            "Use this to retrieve the brief summary excerpt for citations. "
            "Only call this when you need a specific text excerpt."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "nct_id": {
                    "type": "string",
                    "description": "The NCT ID of the study, e.g. 'NCT01234567'.",
                },
            },
            "required": ["nct_id"],
        },
    },
]


# -----------------------------------------------------------------------
# Tool implementations
# -----------------------------------------------------------------------

async def search_trials(
    client: ClinicalTrialsClient,
    interventions: str | None = None,
    conditions: str | None = None,
    sponsor: str | None = None,
    phase: list[str] | None = None,
    country: str | None = None,
    start_date_gte: str | None = None,
    start_date_lte: str | None = None,
) -> list[CTStudy]:
    """Fetch studies from ClinicalTrials.gov with the given filters."""
    params = CTSearchParams(
        interventions=interventions,
        conditions=conditions,
        sponsor=sponsor,
        phase=phase,
        country=country,
        start_date_gte=start_date_gte,
        start_date_lte=start_date_lte,
    )
    studies = await client.search(params)
    logger.info(f"search_trials returned {len(studies)} studies")
    return studies


def aggregate(
    studies: list[CTStudy],
    field: str,
    top_n: int | None = None,
    label: str | None = None,
) -> list[dict[str, Any]]:
    """
    Aggregate studies by a field, returning sorted count data points with citations.

    Each data point includes:
      - The field value (e.g. "PHASE3")
      - trial_count
      - series label (for grouped charts)
      - citations: up to 3 representative nct_ids + excerpts
    """
    counts: dict[str, list[CTStudy]] = {}

    for study in studies:
        keys = _extract_field_values(study, field)
        for key in keys:
            if key not in counts:
                counts[key] = []
            counts[key].append(study)

    # Sort by count descending
    sorted_items = sorted(counts.items(), key=lambda x: len(x[1]), reverse=True)

    if top_n:
        sorted_items = sorted_items[:top_n]

    result = []
    for value, group_studies in sorted_items:
        citations = _build_citations(group_studies, field, value)
        point: dict[str, Any] = {
            field: value,
            "trial_count": len(group_studies),
            "citations": [c.model_dump() for c in citations],
        }
        if label:
            point["series"] = label
        result.append(point)

    return result


async def get_study_details(
    client: ClinicalTrialsClient,
    nct_id: str,
) -> dict[str, Any] | None:
    """Fetch a single study's details for citation excerpts."""
    study = await client.get_study(nct_id)
    if not study:
        return None
    return {
        "nct_id": study.nct_id,
        "brief_title": study.brief_title,
        "brief_summary": study.brief_summary,
        "phase": study.phase,
        "status": study.status,
        "url": study.ct_url,
    }


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _extract_field_values(study: CTStudy, field: str) -> list[str]:
    """
    Extract one or more grouping values from a study for a given field.
    Multi-value fields (countries, conditions, interventions) return a list.
    """
    match field:
        case "phase":
            return [_normalize_phase(study.phase)] if study.phase else ["Unknown"]
        case "status":
            return [study.status or "Unknown"]
        case "sponsor_name":
            return [study.sponsor_name or "Unknown"]
        case "sponsor_class":
            return [study.sponsor_class or "Unknown"]
        case "enrollment_bucket":
            if study.enrollment is None:
                return ["Unknown"]
            e = study.enrollment
            if e < 50:
                return ["<50"]
            elif e < 100:
                return ["50-99"]
            elif e < 250:
                return ["100-249"]
            elif e < 500:
                return ["250-499"]
            elif e < 1000:
                return ["500-999"]
            elif e < 2000:
                return ["1000-1999"]
            else:
                return ["2000+"]
        case "start_year":
            if study.start_date:
                return [study.start_date[:4]]
            return ["Unknown"]
        case "country":
            return study.countries if study.countries else ["Unknown"]
        case "condition":
            return study.conditions if study.conditions else ["Unknown"]
        case "intervention":
            return study.interventions if study.interventions else ["Unknown"]
        case _:
            return ["Unknown"]


def _normalize_phase(phase: str | None) -> str:
    """Normalize CT API phase strings to human-readable form."""
    if not phase:
        return "Unknown"
    mapping = {
        "PHASE1": "Phase 1",
        "PHASE2": "Phase 2",
        "PHASE3": "Phase 3",
        "PHASE4": "Phase 4",
        "EARLY_PHASE1": "Early Phase 1",
        "NA": "N/A",
    }
    return mapping.get(phase.upper(), phase)


def _build_citations(
    studies: list[CTStudy],
    field: str,
    value: str,
    max_citations: int = 3,
) -> list[Citation]:
    """
    Build up to max_citations Citation objects for a data point.
    Picks studies with a brief_summary for the excerpt.
    """
    citations = []
    candidates = [s for s in studies if s.brief_summary][:max_citations]

    for study in candidates:
        excerpt = study.brief_summary[:300].strip()
        if not excerpt.endswith("..."):
            excerpt = excerpt[:297] + "..." if len(study.brief_summary) > 300 else excerpt

        citations.append(Citation(
            nct_id=study.nct_id,
            excerpt=excerpt,
            field_name=f"protocolSection.descriptionModule.briefSummary",
            url=study.ct_url,
        ))

    return citations
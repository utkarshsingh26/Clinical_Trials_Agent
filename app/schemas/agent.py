from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

from .visualization import VizType


class IntentType(str, Enum):
    """
    Classifies the user's analytical intent.
    Drives tool selection and aggregation strategy.
    """
    TREND = "trend"                  # How has X changed over time?
    DISTRIBUTION = "distribution"    # How are trials distributed across Y?
    COMPARISON = "comparison"        # Compare X vs Y across some dimension
    GEOGRAPHIC = "geographic"        # Which countries/locations...?
    NETWORK = "network"              # Relationships between entities
    SUMMARY = "summary"              # General stats / counts


# Intent -> natural viz type mapping (used as fallback validation)
INTENT_VIZ_MAP: dict[IntentType, list[VizType]] = {
    IntentType.TREND: [VizType.TIME_SERIES, VizType.BAR_CHART],
    IntentType.DISTRIBUTION: [VizType.BAR_CHART, VizType.HISTOGRAM, VizType.PIE_CHART],
    IntentType.COMPARISON: [VizType.GROUPED_BAR_CHART, VizType.BAR_CHART, VizType.SCATTER],
    IntentType.GEOGRAPHIC: [VizType.BAR_CHART, VizType.SCATTER],
    IntentType.NETWORK: [VizType.NETWORK_GRAPH],
    IntentType.SUMMARY: [VizType.BAR_CHART, VizType.HISTOGRAM, VizType.PIE_CHART],
}


class ExtractedFilters(BaseModel):
    """
    Structured filters the agent extracted from the query + optional input fields.
    These are passed to the ClinicalTrials.gov API.
    """
    drug_name: Optional[str] = None
    condition: Optional[str] = None
    trial_phase: Optional[str] = None
    sponsor: Optional[str] = None
    country: Optional[str] = None
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    # For comparison queries — second entity
    secondary_drug: Optional[str] = None
    secondary_condition: Optional[str] = None


class AgentPlan(BaseModel):
    """
    Structured plan the agent must produce BEFORE making any tool calls.
    Validated with Pydantic — if the LLM produces an invalid plan, the
    request fails fast rather than proceeding with bad assumptions.

    This is the primary hallucination guard: forces the agent to commit
    to an explicit intent + viz type + filters before touching the API.
    """
    intent: IntentType = Field(
        ...,
        description="The analytical intent behind the user's query.",
    )
    viz_type: VizType = Field(
        ...,
        description="Visualization type chosen for this intent. Must be compatible with intent.",
    )
    filters: ExtractedFilters = Field(
        ...,
        description="Structured filters extracted from the query and optional input fields.",
    )
    aggregation_field: str = Field(
        ...,
        description=(
            "The ClinicalTrials.gov field to aggregate/group by. "
            "E.g. 'phase', 'start_date', 'location_country', 'sponsor_name'."
        ),
        examples=["phase", "start_date", "location_country"],
    )
    reasoning: str = Field(
        ...,
        max_length=500,
        description="One or two sentences explaining why this intent/viz/aggregation was chosen.",
    )
    requires_multiple_searches: bool = Field(
        False,
        description="True if the query requires more than one API call (e.g. comparison queries).",
    )

    def validate_viz_intent_compatibility(self) -> bool:
        """Returns True if the chosen viz type is valid for the detected intent."""
        valid_types = INTENT_VIZ_MAP.get(self.intent, [])
        return self.viz_type in valid_types
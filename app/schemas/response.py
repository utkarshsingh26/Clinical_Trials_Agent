from typing import Any, Optional
from pydantic import BaseModel, Field

from .visualization import VisualizationSpec
from .agent import AgentPlan


class ResponseMetadata(BaseModel):
    """
    Contextual information about how the query was interpreted and executed.
    Helps frontend show appropriate context and helps debugging.
    """
    query_interpretation: str = Field(
        ...,
        description="Human-readable summary of how the agent interpreted the query.",
    )
    filters_applied: dict[str, Any] = Field(
        default_factory=dict,
        description="The actual filters sent to the ClinicalTrials.gov API.",
    )
    total_trials_retrieved: int = Field(
        ...,
        description="Total number of trial records fetched before aggregation.",
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Any assumptions made during query interpretation or aggregation.",
    )
    tool_calls_made: int = Field(
        ...,
        description="Number of tool calls the agent made. Bounded at 5.",
    )
    source: str = Field(
        default="clinicaltrials.gov",
        description="Data source.",
    )


class QueryResponse(BaseModel):
    """
    Top-level response from the query-to-visualization agent.

    The visualization field is the primary output — a structured spec
    a frontend can render without additional API calls or LLM calls.

    Citations are embedded in visualization.data under a 'citations' key
    on each data point (bonus traceability feature).
    """
    visualization: VisualizationSpec
    meta: ResponseMetadata
    plan: AgentPlan = Field(
        ...,
        description="The agent's planning step output — useful for debugging and transparency.",
    )
    error: Optional[str] = Field(
        None,
        description="Set if the agent encountered a recoverable error. Visualization may be partial.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "visualization": {
                        "type": "bar_chart",
                        "title": "Trials by Phase for Pembrolizumab",
                        "encoding": {
                            "encoding_type": "cartesian",
                            "x": {"field": "phase", "label": "Trial Phase", "type": "nominal"},
                            "y": {"field": "trial_count", "label": "Number of Trials", "type": "quantitative"},
                        },
                        "data": [
                            {
                                "phase": "Phase 2",
                                "trial_count": 78,
                                "citations": [
                                    {
                                        "nct_id": "NCT01234567",
                                        "excerpt": "Phase 2 randomized study evaluating pembrolizumab...",
                                        "field_name": "protocolSection.descriptionModule.briefSummary",
                                        "url": "https://clinicaltrials.gov/study/NCT01234567",
                                    }
                                ],
                            }
                        ],
                    },
                    "meta": {
                        "query_interpretation": "User asked about trial phase distribution for Pembrolizumab",
                        "filters_applied": {"interventions": "Pembrolizumab"},
                        "total_trials_retrieved": 110,
                        "assumptions": ["Phase 'NA' trials excluded from distribution"],
                        "tool_calls_made": 2,
                        "source": "clinicaltrials.gov",
                    },
                }
            ]
        }
    }
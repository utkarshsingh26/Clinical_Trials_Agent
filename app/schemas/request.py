from typing import Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """
    Input schema for the query-to-visualization agent.

    Required:
        query: Natural language question about clinical trials.

    Optional structured fields help the agent extract params without relying
    solely on LLM parsing, reducing hallucination risk on specific identifiers.
    """

    query: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Natural language question about clinical trials.",
        examples=["How has the number of trials for Pembrolizumab changed over time?"],
    )

    # Optional structured hints — candidate-defined per spec section 3.1
    drug_name: Optional[str] = Field(
        None,
        description="Specific drug or intervention name.",
        examples=["Pembrolizumab", "Ozempic"],
    )
    condition: Optional[str] = Field(
        None,
        description="Disease or condition of interest.",
        examples=["lung cancer", "type 2 diabetes"],
    )
    trial_phase: Optional[str] = Field(
        None,
        pattern=r"^(PHASE[1-4]|EARLY_PHASE1|NA)$",
        description="Trial phase filter. Must be one of: PHASE1, PHASE2, PHASE3, PHASE4, EARLY_PHASE1, NA.",
        examples=["PHASE3"],
    )
    sponsor: Optional[str] = Field(
        None,
        description="Sponsor organization name.",
        examples=["Merck", "NIH"],
    )
    country: Optional[str] = Field(
        None,
        description="Country filter using ISO 3166-1 alpha-2 code or full name.",
        examples=["US", "United States"],
    )
    start_year: Optional[int] = Field(
        None,
        ge=1990,
        le=2030,
        description="Filter trials starting on or after this year.",
        examples=[2015],
    )
    end_year: Optional[int] = Field(
        None,
        ge=1990,
        le=2030,
        description="Filter trials starting on or before this year.",
        examples=[2024],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "How has the number of trials for Pembrolizumab changed over time?",
                    "drug_name": "Pembrolizumab",
                    "start_year": 2015,
                },
                {
                    "query": "Compare trial phases for lung cancer vs breast cancer",
                    "condition": "lung cancer",
                },
                {
                    "query": "Which countries have the most recruiting trials for type 2 diabetes?",
                    "condition": "type 2 diabetes",
                },
            ]
        }
    }
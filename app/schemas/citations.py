from typing import Optional
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """
    Source traceability record linking a visualization data point
    back to the underlying ClinicalTrials.gov trial record.

    Attached to individual data points in the visualization data array
    under a 'citations' key.
    """
    nct_id: str = Field(
        ...,
        pattern=r"^NCT\d{8}$",
        description="ClinicalTrials.gov trial identifier.",
        examples=["NCT01234567"],
    )
    excerpt: str = Field(
        ...,
        max_length=500,
        description=(
            "Exact text excerpt or field value from the API response "
            "that supports this data point."
        ),
        examples=["Phase 3 randomized study evaluating pembrolizumab in NSCLC..."],
    )
    field_name: Optional[str] = Field(
        None,
        description="The specific API field this excerpt came from.",
        examples=["protocolSection.descriptionModule.briefSummary"],
    )
    url: Optional[str] = Field(
        None,
        description="Direct URL to the trial record on ClinicalTrials.gov.",
        examples=["https://clinicaltrials.gov/study/NCT01234567"],
    )
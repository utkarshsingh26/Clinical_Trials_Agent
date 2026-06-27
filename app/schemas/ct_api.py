from typing import Any, Optional
from pydantic import BaseModel, Field


class CTSearchParams(BaseModel):
    query_term: Optional[str] = Field(None, description="Free-text search term.")
    conditions: Optional[str] = Field(None, description="Condition/disease filter.")
    interventions: Optional[str] = Field(None, description="Drug/intervention filter.")
    sponsor: Optional[str] = Field(None, description="Sponsor name filter.")
    phase: Optional[list[str]] = Field(None, description="Phase filter list.")
    country: Optional[str] = Field(None, description="Country filter.")
    start_date_gte: Optional[str] = Field(None, description="Start date lower bound, YYYY-MM-DD.")
    start_date_lte: Optional[str] = Field(None, description="Start date upper bound, YYYY-MM-DD.")
    fields: Optional[list[str]] = Field(None, description="Specific fields to return.")
    page_size: int = Field(250, le=1000, description="Max results per page.")


class CTStudy(BaseModel):
    nct_id: str
    brief_title: Optional[str] = None
    brief_summary: Optional[str] = None
    phase: Optional[str] = None
    status: Optional[str] = None
    sponsor_name: Optional[str] = None
    sponsor_class: Optional[str] = None
    start_date: Optional[str] = None
    completion_date: Optional[str] = None
    conditions: list[str] = Field(default_factory=list)
    interventions: list[str] = Field(default_factory=list)
    countries: list[str] = Field(default_factory=list)
    enrollment: Optional[int] = None

    @property
    def ct_url(self) -> str:
        return f"https://clinicaltrials.gov/study/{self.nct_id}"

    @classmethod
    def from_api_response(cls, raw: dict[str, Any]) -> "CTStudy":
        protocol = raw.get("protocolSection", {})
        id_module = protocol.get("identificationModule", {})
        desc_module = protocol.get("descriptionModule", {})
        status_module = protocol.get("statusModule", {})
        design_module = protocol.get("designModule", {})
        sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
        conditions_module = protocol.get("conditionsModule", {})
        arms_module = protocol.get("armsInterventionsModule", {})
        contacts_module = protocol.get("contactsLocationsModule", {})

        interventions = [
            i.get("name", "")
            for i in arms_module.get("interventions", [])
            if i.get("name")
        ]

        countries = list({
            loc.get("country", "")
            for loc in contacts_module.get("locations", [])
            if loc.get("country")
        })

        lead_sponsor = sponsor_module.get("leadSponsor", {})
        phases = design_module.get("phases", [])
        phase = phases[0] if phases else None

        return cls(
            nct_id=id_module.get("nctId", ""),
            brief_title=id_module.get("briefTitle"),
            brief_summary=desc_module.get("briefSummary"),
            phase=phase,
            status=status_module.get("overallStatus"),
            sponsor_name=lead_sponsor.get("name"),
            sponsor_class=lead_sponsor.get("class"),
            start_date=status_module.get("startDateStruct", {}).get("date"),
            completion_date=status_module.get("completionDateStruct", {}).get("date"),
            conditions=conditions_module.get("conditions", []),
            interventions=interventions,
            countries=countries,
            enrollment=design_module.get("enrollmentInfo", {}).get("count"),
        )
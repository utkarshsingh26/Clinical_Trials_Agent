"""
ClinicalTrials.gov API v2 client.

Uses `requests` (not httpx) because clinicaltrials.gov blocks httpx via TLS
fingerprinting. Requests is run in a thread pool executor to keep the async
interface intact for the FastAPI/asyncio environment.
"""

import asyncio
import logging
from typing import Optional
from functools import partial

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.schemas.ct_api import CTSearchParams, CTStudy

logger = logging.getLogger(__name__)

CT_API_BASE = "https://clinicaltrials.gov/api/v2"

REQUIRED_FIELDS = [
    "NCTId",
    "BriefTitle",
    "BriefSummary",
    "Phase",
    "OverallStatus",
    "StartDate",
    "CompletionDate",
    "LeadSponsorName",
    "LeadSponsorClass",
    "Condition",
    "InterventionName",
    "LocationCountry",
    "EnrollmentCount",
]


class CTAPIError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"CT API error {status_code}: {message}")


def _make_session() -> requests.Session:
    """Create a requests session with retry logic."""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class ClinicalTrialsClient:
    def __init__(
        self,
        base_url: str = CT_API_BASE,
        timeout: float = 30.0,
        max_retries: int = 3,
        mock: bool = False,
    ):
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.mock = mock
        self._session: Optional[requests.Session] = None

    async def __aenter__(self) -> "ClinicalTrialsClient":
        self._session = _make_session()
        return self

    async def __aexit__(self, *args) -> None:
        if self._session:
            self._session.close()

    def _build_query_params(self, params: CTSearchParams) -> dict:
        p: dict = {"pageSize": params.page_size, "format": "json"}

        if params.query_term:
            p["query.term"] = params.query_term
        if params.conditions:
            p["query.cond"] = params.conditions
        if params.interventions:
            p["query.intr"] = params.interventions
        if params.sponsor:
            p["query.spons"] = params.sponsor

        filter_parts = []
        if params.phase:
            phase_filter = " OR ".join(
                f'AREA[Phase]{ph}' for ph in params.phase
            )
            filter_parts.append(f"({phase_filter})")
        if params.country:
            filter_parts.append(f'AREA[LocationCountry]{params.country}')
        if params.start_date_gte:
            filter_parts.append(f'AREA[StartDate]RANGE[{params.start_date_gte},MAX]')
        if params.start_date_lte:
            filter_parts.append(f'AREA[StartDate]RANGE[MIN,{params.start_date_lte}]')
        if filter_parts:
            p["filter.advanced"] = " AND ".join(filter_parts)

        p["fields"] = ",".join(params.fields or REQUIRED_FIELDS)
        return p

    def _sync_get(self, url: str, params: dict) -> dict:
        """Synchronous GET — run in thread pool to avoid blocking event loop."""
        resp = self._session.get(url, params=params, timeout=self.timeout)
        if resp.status_code == 200:
            return resp.json()
        raise CTAPIError(resp.status_code, resp.text[:200])

    async def _get(self, url: str, params: dict) -> dict:
        """Run sync GET in thread pool executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, partial(self._sync_get, url, params))

    async def search(
        self,
        params: CTSearchParams,
        max_pages: int = 1,
    ) -> list[CTStudy]:
        if self.mock:
            return await self._mock_search(params)

        if not self._session:
            raise RuntimeError("Client not initialized. Use async with.")

        query_params = self._build_query_params(params)
        url = f"{self.base_url}/studies"

        studies: list[CTStudy] = []
        seen_ids: set[str] = set()
        page_token: Optional[str] = None
        pages_fetched = 0

        while pages_fetched < max_pages:
            if page_token:
                query_params["pageToken"] = page_token

            data = await self._get(url, query_params)
            raw_studies = data.get("studies", [])

            for raw in raw_studies:
                study = CTStudy.from_api_response(raw)
                if study.nct_id and study.nct_id not in seen_ids:
                    studies.append(study)
                    seen_ids.add(study.nct_id)

            pages_fetched += 1
            page_token = data.get("nextPageToken")

            if not page_token:
                break

            logger.info(f"Fetched page {pages_fetched}, total so far: {len(studies)}")

        logger.info(f"Search complete: {len(studies)} studies across {pages_fetched} page(s)")
        return studies

    async def get_study(self, nct_id: str) -> Optional[CTStudy]:
        if self.mock:
            return await self._mock_get_study(nct_id)

        if not self._session:
            raise RuntimeError("Client not initialized. Use async with.")

        url = f"{self.base_url}/studies/{nct_id}"
        try:
            data = await self._get(url, {"format": "json"})
            return CTStudy.from_api_response(data)
        except CTAPIError as e:
            if e.status_code == 404:
                return None
            raise

    async def _mock_search(self, params: CTSearchParams) -> list[CTStudy]:
        await asyncio.sleep(0.05)

        drug = params.interventions or "TestDrug"
        condition = params.conditions or "Cancer"

        import random
        random.seed(42)

        phases = ["PHASE1", "PHASE2", "PHASE3", "PHASE4"]
        sponsors = ["Merck", "Pfizer", "NIH", "Roche", "AstraZeneca"]
        countries_pool = [
            ["United States"], ["United States", "Canada"],
            ["Germany", "France"], ["United Kingdom"],
            ["United States", "Japan"], ["China"],
        ]
        statuses = ["COMPLETED", "RECRUITING", "ACTIVE_NOT_RECRUITING", "TERMINATED"]

        mock_studies = []
        for i in range(80):
            year = random.randint(2010, 2024)
            month = random.randint(1, 12)
            mock_studies.append(CTStudy(
                nct_id=f"NCT{str(i + 1000000).zfill(8)}",
                brief_title=f"{drug} Study {i + 1} for {condition}",
                brief_summary=(
                    f"{random.choice(phases)} randomized study evaluating "
                    f"{drug} in patients with {condition}."
                ),
                phase=random.choice(phases),
                status=random.choice(statuses),
                sponsor_name=random.choice(sponsors),
                sponsor_class=random.choice(["INDUSTRY", "NIH", "OTHER"]),
                start_date=f"{year}-{str(month).zfill(2)}",
                completion_date=f"{min(year + 3, 2025)}-{str(month).zfill(2)}",
                conditions=[condition],
                interventions=[drug],
                countries=random.choice(countries_pool),
                enrollment=random.randint(50, 2000),
            ))

        if params.start_date_gte:
            min_year = int(params.start_date_gte[:4])
            mock_studies = [
                s for s in mock_studies
                if s.start_date and int(s.start_date[:4]) >= min_year
            ]

        return mock_studies

    async def _mock_get_study(self, nct_id: str) -> Optional[CTStudy]:
        await asyncio.sleep(0.02)
        return CTStudy(
            nct_id=nct_id,
            brief_title=f"Mock Study {nct_id}",
            brief_summary="Phase 3 randomized study evaluating treatment in oncology patients.",
            phase="PHASE3",
            status="COMPLETED",
            sponsor_name="Merck",
            sponsor_class="INDUSTRY",
            start_date="2020-01",
            completion_date="2023-06",
            conditions=["Lung Cancer"],
            interventions=["Pembrolizumab"],
            countries=["United States", "Germany"],
            enrollment=450,
        )
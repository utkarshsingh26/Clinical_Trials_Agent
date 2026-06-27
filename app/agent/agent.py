"""
ClinicalTrials Query-to-Visualization Agent.

Architecture:
  1. PLAN   — LLM parses query into structured AgentPlan (intent, viz_type, filters)
              Validated with Pydantic before any API calls. Fails fast on bad plans.
  2. LOOP   — LLM calls tools (search_trials, aggregate, get_study_details) to fetch
              and process data. Hard cap of MAX_TOOL_CALLS per request.
  3. ASSEMBLE — LLM produces final VisualizationSpec JSON from aggregated data.
              Validated against our schema before returning.

Design decisions:
  - Raw OpenAI SDK, no LangChain/LangGraph — explicit state, easy to debug
  - Forced planning step before tool use — primary hallucination guard
  - Tool inputs/outputs validated with Pydantic at every step
  - viz_type decided upfront, not inferred from data — prevents hallucination
  - Network graph built from aggregated adjacency data, not raw LLM output
"""

import json
import logging
import os
from typing import Any, AsyncGenerator
import asyncio

from openai import OpenAI

from app.client.ct_client import ClinicalTrialsClient
from app.schemas.agent import AgentPlan, ExtractedFilters, IntentType, INTENT_VIZ_MAP
from app.schemas.ct_api import CTStudy
from app.schemas.request import QueryRequest
from app.schemas.response import QueryResponse, ResponseMetadata
from app.schemas.visualization import (
    VizType,
    VisualizationSpec,
    CartesianEncoding,
    NetworkEncoding,
    AxisField,
    NodeDef,
    EdgeDef,
)
from app.tools.tools import (
    search_trials,
    aggregate,
    get_study_details,
    TOOL_DEFINITIONS,
)

logger = logging.getLogger(__name__)

MAX_TOOL_CALLS = 5

PLAN_SYSTEM_PROMPT = """You are a clinical trials data analyst. Your job is to interpret a user's 
natural language question about clinical trials and produce a structured analysis plan.

You must respond with ONLY valid JSON matching this exact schema — no prose, no markdown:
{
  "intent": one of ["trend", "distribution", "comparison", "geographic", "network", "summary"],
  "viz_type": one of ["bar_chart", "grouped_bar_chart", "time_series", "scatter", "histogram", "network_graph", "pie_chart"],
  "filters": {
    "drug_name": string or null,
    "condition": string or null,
    "trial_phase": string or null,
    "sponsor": string or null,
    "country": string or null,
    "start_year": integer or null,
    "end_year": integer or null,
    "secondary_drug": string or null,
    "secondary_condition": string or null
  },
  "aggregation_field": one of ["phase", "status", "sponsor_name", "sponsor_class", "start_year", "country", "condition", "intervention", "enrollment_bucket"],
  "reasoning": "one or two sentences explaining your choices",
  "requires_multiple_searches": boolean
}

Intent to viz_type rules:
- trend -> time_series or bar_chart
- distribution -> bar_chart or histogram
- comparison -> grouped_bar_chart or bar_chart
- geographic -> bar_chart
- network -> network_graph
- summary -> bar_chart or histogram
- For sponsor type/class breakdown queries, use distribution intent with viz_type="pie_chart" and aggregation_field="sponsor_class".

Be conservative: if unsure between network and distribution, choose distribution.
For enrollment/size/distribution queries, use histogram intent with aggregation_field="enrollment_bucket"."""

AGENT_SYSTEM_PROMPT = """You are a clinical trials data analyst agent with access to the ClinicalTrials.gov API.

You have three tools:
1. search_trials — fetch studies matching filters
2. aggregate — group studies by a field to produce visualization data
3. get_study_details — get a single study's details for citations

Workflow you MUST follow:
1. Call search_trials first with appropriate filters from the plan
2. Call aggregate on the results to produce data points
3. For comparison queries: search twice (primary + secondary), aggregate both, merge
4. For network queries: search once, then build node/edge data from the results
5. Stop when you have enough data to answer the question

IMPORTANT:
- Do not fabricate data. Only use values returned by the tools.
- Keep titles concise and descriptive.
- Always include citations in each data point from the aggregate tool output.
- For histogram queries about enrollment size, call aggregate with field="enrollment_bucket".
- Always call both search_trials AND aggregate — never stop after just search_trials."""

# OpenAI tool definitions — same structure as Anthropic but wrapped in "function"
OPENAI_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"],
        },
    }
    for tool in TOOL_DEFINITIONS
]


class TrialsAgent:
    """
    Query-to-Visualization agent for ClinicalTrials.gov data.

    Usage:
        agent = TrialsAgent(mock=True)
        response = await agent.run(request)
    """

    def __init__(self, mock: bool = False, api_key: str | None = None):
        self.mock = mock
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY")
        )
        self.progress_queue: asyncio.Queue | None = None

    async def emit(self, event: str, detail: str = ""):
        """Emit a progress event to the SSE queue if one is attached."""
        if self.progress_queue:
            await self.progress_queue.put({"event": event, "detail": detail})

    async def run(self, request: QueryRequest, progress_queue: asyncio.Queue | None = None) -> QueryResponse:
        """
        Main entry point. Runs the full plan -> tool loop -> assemble pipeline.
        """
        self.progress_queue = progress_queue
        logger.info(f"Agent starting for query: {request.query[:80]}")

        await self.emit("Planning query...", "Interpreting: " + request.query[:60])

        # Step 1: Plan
        plan = await self._plan(request)
        logger.info(f"Plan: intent={plan.intent}, viz={plan.viz_type}, field={plan.aggregation_field}")

        await self.emit("Plan complete", f"Intent: {plan.intent} · Viz: {plan.viz_type} · Field: {plan.aggregation_field}")

        # Step 2: Agentic tool loop
        viz_data, tool_calls_made, raw_studies = await self._tool_loop(request, plan)

        # Step 3: Assemble final response
        await self.emit("Building visualization...", "Assembling chart spec")
        response = self._assemble_response(request, plan, viz_data, tool_calls_made, raw_studies)
        return response

    # ------------------------------------------------------------------
    # Step 1: Planning
    # ------------------------------------------------------------------

    async def _plan(self, request: QueryRequest) -> AgentPlan:
        """
        Ask the LLM to produce a structured plan. Validates output with Pydantic.
        Falls back to a rule-based plan if LLM output is invalid.
        """
        context = f"Query: {request.query}"
        if request.drug_name:
            context += f"\nDrug: {request.drug_name}"
        if request.condition:
            context += f"\nCondition: {request.condition}"
        if request.trial_phase:
            context += f"\nPhase: {request.trial_phase}"
        if request.start_year:
            context += f"\nStart year: {request.start_year}"
        if request.end_year:
            context += f"\nEnd year: {request.end_year}"

        response = self.client.chat.completions.create(
            model="gpt-4.1",
            max_tokens=1000,
            messages=[
                {"role": "system", "content": PLAN_SYSTEM_PROMPT},
                {"role": "user", "content": context},
            ],
            response_format={"type": "json_object"},  # forces valid JSON output
        )

        raw = response.choices[0].message.content.strip()

        try:
            plan_dict = json.loads(raw)
            plan = AgentPlan(**plan_dict)

            # Validate viz/intent compatibility — fallback if invalid
            if not plan.validate_viz_intent_compatibility():
                valid_types = INTENT_VIZ_MAP.get(plan.intent, [])
                if valid_types:
                    logger.warning(
                        f"LLM chose incompatible viz {plan.viz_type} for intent {plan.intent}. "
                        f"Falling back to {valid_types[0]}"
                    )
                    plan = plan.model_copy(update={"viz_type": valid_types[0]})

            return plan

        except Exception as e:
            logger.warning(f"Plan parsing failed ({e}), using rule-based fallback")
            return self._fallback_plan(request)

    def _fallback_plan(self, request: QueryRequest) -> AgentPlan:
        """Rule-based plan when LLM output is unparseable."""
        query_lower = request.query.lower()

        if any(w in query_lower for w in ["over time", "per year", "trend", "changed", "history"]):
            intent = IntentType.TREND
            viz_type = VizType.TIME_SERIES
            agg_field = "start_year"
        elif any(w in query_lower for w in ["country", "countries", "where", "geographic"]):
            intent = IntentType.GEOGRAPHIC
            viz_type = VizType.BAR_CHART
            agg_field = "country"
        elif any(w in query_lower for w in ["network", "relationship", "sponsor", "connected"]):
            intent = IntentType.NETWORK
            viz_type = VizType.NETWORK_GRAPH
            agg_field = "sponsor_name"
        elif any(w in query_lower for w in ["compare", "vs", "versus"]):
            intent = IntentType.COMPARISON
            viz_type = VizType.GROUPED_BAR_CHART
            agg_field = "phase"
        else:
            intent = IntentType.DISTRIBUTION
            viz_type = VizType.BAR_CHART
            agg_field = "phase"

        return AgentPlan(
            intent=intent,
            viz_type=viz_type,
            filters=ExtractedFilters(
                drug_name=request.drug_name,
                condition=request.condition,
                trial_phase=request.trial_phase,
                sponsor=request.sponsor,
                country=request.country,
                start_year=request.start_year,
                end_year=request.end_year,
            ),
            aggregation_field=agg_field,
            reasoning="Rule-based fallback plan used due to LLM parsing failure.",
            requires_multiple_searches=False,
        )

    # ------------------------------------------------------------------
    # Step 2: Agentic tool loop
    # ------------------------------------------------------------------

    async def _tool_loop(
        self,
        request: QueryRequest,
        plan: AgentPlan,
    ) -> tuple[list[dict], int, dict[str, list[CTStudy]]]:
        """
        Runs the bounded agentic tool loop. Hard cap at MAX_TOOL_CALLS.

        Returns:
          - viz_data: list of aggregated data points
          - tool_calls_made: count for metadata
          - raw_studies: dict of search results keyed by 'primary'/'secondary'
        """
        async with ClinicalTrialsClient(mock=self.mock) as ct_client:
            plan_context = (
                f"User query: {request.query}\n\n"
                f"Analysis plan:\n"
                f"- Intent: {plan.intent}\n"
                f"- Viz type: {plan.viz_type}\n"
                f"- Aggregation field: {plan.aggregation_field}\n"
                f"- Filters: {plan.filters.model_dump(exclude_none=True)}\n"
                f"- Requires multiple searches: {plan.requires_multiple_searches}\n\n"
                f"Execute this plan using the available tools."
            )

            messages = [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": plan_context},
            ]

            study_store: dict[str, list[CTStudy]] = {}
            viz_data: list[dict] = []
            tool_calls_made = 0

            while tool_calls_made < MAX_TOOL_CALLS:
                response = self.client.chat.completions.create(
                    model="gpt-4.1",
                    max_tokens=4000,
                    tools=OPENAI_TOOL_DEFINITIONS,
                    tool_choice="auto",
                    messages=messages,
                )

                message = response.choices[0].message
                finish_reason = response.choices[0].finish_reason

                # Add assistant message to history
                messages.append(message)

                if finish_reason == "stop" or not message.tool_calls:
                    break

                # Process tool calls
                for tool_call in message.tool_calls:
                    if tool_calls_made >= MAX_TOOL_CALLS:
                        logger.warning("Tool call cap reached, stopping loop")
                        break

                    tool_calls_made += 1
                    tool_input = json.loads(tool_call.function.arguments)

                    result = await self._dispatch_tool(
                        tool_call.function.name,
                        tool_input,
                        ct_client,
                        study_store,
                    )

                    if tool_call.function.name == "aggregate" and isinstance(result, list):
                        viz_data = result

                    # Add tool result to message history
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, default=str),
                    })

            return viz_data, tool_calls_made, study_store

    async def _dispatch_tool(
        self,
        tool_name: str,
        tool_input: dict,
        ct_client: ClinicalTrialsClient,
        study_store: dict[str, list[CTStudy]],
    ) -> Any:
        """Route a tool call to the correct implementation."""
        logger.info(f"Tool call: {tool_name}({list(tool_input.keys())})")

        match tool_name:
            case "search_trials":
                label = tool_input.pop("studies_key", "primary")
                filters_desc = ", ".join(f"{k}={v}" for k, v in tool_input.items() if v)
                await self.emit(f"Searching ClinicalTrials.gov...", filters_desc or "Fetching trials")
                studies = await search_trials(ct_client, **tool_input)
                study_store[label] = studies
                await self.emit(f"Retrieved {len(studies):,} trials", f"Search complete for {label} query")
                return {
                    "studies_key": label,
                    "count": len(studies),
                    "sample": [
                        {"nct_id": s.nct_id, "phase": s.phase, "start_date": s.start_date}
                        for s in studies[:5]
                    ],
                }

            case "aggregate":
                studies_key = tool_input.get("studies_key", "primary")
                field = tool_input.get("field", "phase")
                studies = study_store.get(studies_key, study_store.get("primary", []))
                await self.emit(f"Aggregating {len(studies):,} trials...", f"Grouping by {field}")
                result = aggregate(
                    studies=studies,
                    field=field,
                    top_n=tool_input.get("top_n"),
                    label=tool_input.get("label"),
                )
                await self.emit(f"Aggregation complete", f"{len(result)} data points produced")
                return result

            case "get_study_details":
                await self.emit("Fetching citation details...", tool_input["nct_id"])
                return await get_study_details(ct_client, tool_input["nct_id"])

            case _:
                logger.warning(f"Unknown tool: {tool_name}")
                return {"error": f"Unknown tool: {tool_name}"}

    # ------------------------------------------------------------------
    # Step 3: Assemble final QueryResponse
    # ------------------------------------------------------------------

    def _assemble_response(
        self,
        request: QueryRequest,
        plan: AgentPlan,
        viz_data: list[dict],
        tool_calls_made: int,
        study_store: dict[str, list[CTStudy]],
    ) -> QueryResponse:
        total_studies = sum(len(v) for v in study_store.values())
        viz_spec = self._build_viz_spec(plan, viz_data, study_store)
        meta = ResponseMetadata(
            query_interpretation=plan.reasoning,
            filters_applied=plan.filters.model_dump(exclude_none=True),
            total_trials_retrieved=total_studies,
            assumptions=self._infer_assumptions(plan, viz_data),
            tool_calls_made=tool_calls_made,
        )
        return QueryResponse(visualization=viz_spec, meta=meta, plan=plan)

    def _build_viz_spec(
        self,
        plan: AgentPlan,
        viz_data: list[dict],
        study_store: dict[str, list[CTStudy]],
    ) -> VisualizationSpec:
        field = plan.aggregation_field
        title = self._generate_title(plan)

        match plan.viz_type:
            case VizType.NETWORK_GRAPH:
                encoding = self._build_network_encoding(study_store)
                # Build sample citations from top studies for network graph
                from app.tools.tools import _build_citations
                top_studies = study_store.get("primary", [])[:10]
                network_citations = _build_citations(top_studies, "sponsor_name", "", max_citations=10)
                network_data = [{"citations": [c.model_dump() for c in network_citations]}] if network_citations else []
                return VisualizationSpec(
                    type=plan.viz_type,
                    title=title,
                    encoding=encoding,
                    data=network_data,
                )
            case VizType.TIME_SERIES:
                encoding = CartesianEncoding(
                    x=AxisField(field="start_year", label="Year", type="temporal"),
                    y=AxisField(field="trial_count", label="Number of Trials", type="quantitative"),
                )
            case VizType.GROUPED_BAR_CHART:
                encoding = CartesianEncoding(
                    x=AxisField(field=field, label=_field_label(field), type="nominal"),
                    y=AxisField(field="trial_count", label="Number of Trials", type="quantitative"),
                    series=AxisField(field="series", label="Group", type="nominal"),
                )
            case VizType.SCATTER:
                encoding = CartesianEncoding(
                    x=AxisField(field="start_year", label="Year", type="temporal"),
                    y=AxisField(field="enrollment", label="Enrollment", type="quantitative"),
                )
            case VizType.PIE_CHART:
                encoding = CartesianEncoding(
                    x=AxisField(field=field, label=_field_label(field), type="nominal"),
                    y=AxisField(field="trial_count", label="Number of Trials", type="quantitative"),
                )

            case VizType.HISTOGRAM:
                # Sort enrollment buckets in logical order
                bucket_order = ["<50","50-99","100-249","250-499","500-999","1000-1999","2000+","Unknown"]
                viz_data = sorted(viz_data, key=lambda d: bucket_order.index(d.get("enrollment_bucket", "Unknown")) if d.get("enrollment_bucket") in bucket_order else 99)
                encoding = CartesianEncoding(
                    x=AxisField(field="enrollment_bucket", label="Enrollment Size", type="ordinal"),
                    y=AxisField(field="trial_count", label="Number of Trials", type="quantitative"),
                )

            case _:
                encoding = CartesianEncoding(
                    x=AxisField(field=field, label=_field_label(field), type="nominal"),
                    y=AxisField(field="trial_count", label="Number of Trials", type="quantitative"),
                )

        return VisualizationSpec(
            type=plan.viz_type,
            title=title,
            encoding=encoding,
            data=viz_data,
            color_scheme="categorical",
        )

    def _build_network_encoding(
        self,
        study_store: dict[str, list[CTStudy]],
    ) -> NetworkEncoding:
        studies = study_store.get("primary", [])
        sponsor_nodes: dict[str, NodeDef] = {}
        condition_nodes: dict[str, NodeDef] = {}
        edge_weights: dict[tuple[str, str], int] = {}

        for study in studies:
            sponsor = study.sponsor_name or "Unknown"
            if sponsor not in sponsor_nodes:
                sponsor_nodes[sponsor] = NodeDef(
                    id=f"sponsor_{sponsor}",
                    label=sponsor,
                    type="sponsor",
                )
            for condition in (study.conditions or [])[:2]:
                cond_key = condition[:50]
                if cond_key not in condition_nodes:
                    condition_nodes[cond_key] = NodeDef(
                        id=f"condition_{cond_key}",
                        label=cond_key,
                        type="condition",
                    )
                edge_key = (f"sponsor_{sponsor}", f"condition_{cond_key}")
                edge_weights[edge_key] = edge_weights.get(edge_key, 0) + 1

        top_sponsors = sorted(
            sponsor_nodes.values(),
            key=lambda n: sum(w for (s, _), w in edge_weights.items() if s == n.id),
            reverse=True,
        )[:20]
        top_conditions = sorted(
            condition_nodes.values(),
            key=lambda n: sum(w for (_, c), w in edge_weights.items() if c == n.id),
            reverse=True,
        )[:15]

        top_sponsor_ids = {n.id for n in top_sponsors}
        top_condition_ids = {n.id for n in top_conditions}

        edges = [
            EdgeDef(source=s, target=c, weight=w)
            for (s, c), w in edge_weights.items()
            if s in top_sponsor_ids and c in top_condition_ids
        ]

        return NetworkEncoding(
            nodes=list(top_sponsors) + list(top_conditions),
            edges=edges,
            node_color_by="type",
            edge_weight_label="Number of co-occurring trials",
        )

    def _generate_title(self, plan: AgentPlan) -> str:
        f = plan.filters
        subject = f.drug_name or f.condition or f.sponsor or "Clinical Trials"
        field_label = _field_label(plan.aggregation_field)

        match plan.intent:
            case IntentType.TREND:
                return f"Trials Over Time: {subject}"
            case IntentType.DISTRIBUTION:
                return f"{subject} Trials by {field_label}"
            case IntentType.COMPARISON:
                secondary = f.secondary_drug or f.secondary_condition or "Comparison"
                return f"Trial Comparison: {subject} vs {secondary} by {field_label}"
            case IntentType.GEOGRAPHIC:
                return f"Geographic Distribution of {subject} Trials"
            case IntentType.NETWORK:
                return f"Sponsor-Condition Network: {subject}"
            case _:
                return f"{subject} Clinical Trials Summary"

    def _infer_assumptions(self, plan: AgentPlan, viz_data: list[dict]) -> list[str]:
        assumptions = []
        if plan.viz_type == VizType.NETWORK_GRAPH:
            assumptions.append("Network limited to top 20 sponsors and 15 conditions by connectivity.")
        if any(d.get(plan.aggregation_field) == "Unknown" for d in viz_data):
            assumptions.append("Some trials had missing values and are grouped under 'Unknown'.")
        if plan.aggregation_field == "country":
            assumptions.append("Trials with multiple locations are counted once per country.")
        return assumptions


def _field_label(field: str) -> str:
    return {
        "phase": "Phase",
        "status": "Status",
        "sponsor_name": "Sponsor",
        "sponsor_class": "Sponsor Type",
        "start_year": "Year",
        "country": "Country",
        "condition": "Condition",
        "intervention": "Intervention",
    }.get(field, field.replace("_", " ").title())

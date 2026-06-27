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
  - Strict JSON schema on plan call — LLM constrained at token level
  - Few-shot examples in plan prompt — reduces intent misclassification
  - Retry-on-failure planning — one retry before rule-based fallback
  - Entity normalization — resolves brand names/abbreviations before planning
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

from app.agent.normalizer import normalize_request_entities
from app.observability import new_trace, StepTrace, Timer
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

# Strict JSON schema for plan output — constrains LLM at token level
PLAN_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "agent_plan",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": ["trend", "distribution", "comparison", "geographic", "network", "summary"]
                },
                "viz_type": {
                    "type": "string",
                    "enum": ["bar_chart", "grouped_bar_chart", "time_series", "scatter", "histogram", "network_graph", "pie_chart"]
                },
                "filters": {
                    "type": "object",
                    "properties": {
                        "drug_name": {"type": ["string", "null"]},
                        "condition": {"type": ["string", "null"]},
                        "trial_phase": {"type": ["string", "null"]},
                        "sponsor": {"type": ["string", "null"]},
                        "country": {"type": ["string", "null"]},
                        "start_year": {"type": ["integer", "null"]},
                        "end_year": {"type": ["integer", "null"]},
                        "secondary_drug": {"type": ["string", "null"]},
                        "secondary_condition": {"type": ["string", "null"]}
                    },
                    "required": ["drug_name", "condition", "trial_phase", "sponsor", "country", "start_year", "end_year", "secondary_drug", "secondary_condition"],
                    "additionalProperties": False
                },
                "aggregation_field": {
                    "type": "string",
                    "enum": ["phase", "status", "sponsor_name", "sponsor_class", "start_year", "country", "condition", "intervention", "enrollment_bucket"]
                },
                "reasoning": {"type": "string"},
                "requires_multiple_searches": {"type": "boolean"}
            },
            "required": ["intent", "viz_type", "filters", "aggregation_field", "reasoning", "requires_multiple_searches"],
            "additionalProperties": False
        }
    }
}

PLAN_SYSTEM_PROMPT = """You are a clinical trials data analyst. Your job is to interpret a user's 
natural language question about clinical trials and produce a structured analysis plan.

Intent to viz_type rules:
- trend -> time_series or bar_chart
- distribution -> bar_chart or histogram
- comparison -> grouped_bar_chart or bar_chart
- geographic -> bar_chart
- network -> network_graph
- summary -> bar_chart or histogram
- For sponsor type/class breakdown queries, use distribution intent with viz_type="pie_chart" and aggregation_field="sponsor_class".

Be conservative: if unsure between network and distribution, choose distribution.
For enrollment/size/distribution queries, use histogram intent with aggregation_field="enrollment_bucket".

Examples of correct plans:

Query: "How has the number of trials for Pembrolizumab changed per year since 2015?"
Plan: {"intent": "trend", "viz_type": "time_series", "filters": {"drug_name": "Pembrolizumab", "condition": null, "trial_phase": null, "sponsor": null, "country": null, "start_year": 2015, "end_year": null, "secondary_drug": null, "secondary_condition": null}, "aggregation_field": "start_year", "reasoning": "User asks about change over time, trend maps to time_series aggregated by year.", "requires_multiple_searches": false}

Query: "How are lung cancer trials distributed across phases?"
Plan: {"intent": "distribution", "viz_type": "bar_chart", "filters": {"drug_name": null, "condition": "lung cancer", "trial_phase": null, "sponsor": null, "country": null, "start_year": null, "end_year": null, "secondary_drug": null, "secondary_condition": null}, "aggregation_field": "phase", "reasoning": "User asks about distribution across phases, bar_chart grouped by phase.", "requires_multiple_searches": false}

Query: "Compare trial phases for Pembrolizumab vs Nivolumab"
Plan: {"intent": "comparison", "viz_type": "grouped_bar_chart", "filters": {"drug_name": "Pembrolizumab", "condition": null, "trial_phase": null, "sponsor": null, "country": null, "start_year": null, "end_year": null, "secondary_drug": "Nivolumab", "secondary_condition": null}, "aggregation_field": "phase", "reasoning": "User compares two drugs, grouped_bar_chart with phase on x-axis.", "requires_multiple_searches": true}

Query: "Which countries have the most recruiting trials for type 2 diabetes?"
Plan: {"intent": "geographic", "viz_type": "bar_chart", "filters": {"drug_name": null, "condition": "type 2 diabetes", "trial_phase": null, "sponsor": null, "country": null, "start_year": null, "end_year": null, "secondary_drug": null, "secondary_condition": null}, "aggregation_field": "country", "reasoning": "User asks about geographic distribution, bar_chart by country.", "requires_multiple_searches": false}

Query: "Show a network of sponsors and conditions for breast cancer trials"
Plan: {"intent": "network", "viz_type": "network_graph", "filters": {"drug_name": null, "condition": "breast cancer", "trial_phase": null, "sponsor": null, "country": null, "start_year": null, "end_year": null, "secondary_drug": null, "secondary_condition": null}, "aggregation_field": "sponsor_name", "reasoning": "User asks for relationship network, network_graph between sponsors and conditions.", "requires_multiple_searches": false}"""

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
        self.trace = new_trace(request.query)
        logger.info(f"Agent starting for query: {request.query[:80]}")

        await self.emit("Planning query...", "Interpreting: " + request.query[:60])

        # Step 0: Normalize entity names before planning
        with Timer() as t:
            _, norm_drug, norm_condition = await normalize_request_entities(
                request.query,
                request.drug_name,
                request.condition,
            )
        self.trace.add_step(StepTrace(
            step="normalization",
            input={"drug_name": request.drug_name, "condition": request.condition},
            output={"drug_name": norm_drug, "condition": norm_condition},
            duration_ms=t.elapsed_ms,
        ))
        if norm_drug != request.drug_name or norm_condition != request.condition:
            await self.emit(
                "Entity normalization",
                ("Drug: " + str(request.drug_name) + " -> " + str(norm_drug) if norm_drug != request.drug_name else "")
                + ("Condition: " + str(request.condition) + " -> " + str(norm_condition) if norm_condition != request.condition else "")
            )
            request = request.model_copy(update={
                "drug_name": norm_drug,
                "condition": norm_condition,
            })

        # Step 1: Plan
        with Timer() as t:
            plan = await self._plan(request)
        logger.info(f"Plan: intent={plan.intent}, viz={plan.viz_type}, field={plan.aggregation_field}")
        self.trace.intent = str(plan.intent)
        self.trace.viz_type = str(plan.viz_type)
        self.trace.fallback_used = plan.reasoning == "Rule-based fallback plan used due to LLM parsing failure."
        self.trace.add_step(StepTrace(
            step="planning",
            input={"model": "gpt-4.1-mini"},
            output={"intent": str(plan.intent), "viz_type": str(plan.viz_type), "aggregation_field": plan.aggregation_field},
            duration_ms=t.elapsed_ms,
        ))
        await self.emit("Plan complete", "Intent: " + str(plan.intent) + " - Viz: " + str(plan.viz_type) + " - Field: " + str(plan.aggregation_field))

        # Step 2: Agentic tool loop
        viz_data, tool_calls_made, raw_studies = await self._tool_loop(request, plan)

        # Step 3: Assemble final response
        await self.emit("Building visualization...", "Assembling chart spec")
        with Timer() as t:
            response = self._assemble_response(request, plan, viz_data, tool_calls_made, raw_studies)
        self.trace.add_step(StepTrace(
            step="assembly",
            input={"viz_type": str(plan.viz_type)},
            output={"data_points": len(viz_data)},
            duration_ms=t.elapsed_ms,
        ))
        self.trace.complete("success")
        return response

    # ------------------------------------------------------------------
    # Step 1: Planning
    # ------------------------------------------------------------------

    async def _plan(self, request: QueryRequest) -> AgentPlan:
        """
        Ask the LLM to produce a structured plan using strict JSON schema.
        Validates output with Pydantic. Retries once on failure before
        falling back to rule-based planning.
        """
        context = "Query: " + request.query
        if request.drug_name:
            context += "\nDrug: " + request.drug_name
        if request.condition:
            context += "\nCondition: " + request.condition
        if request.trial_phase:
            context += "\nPhase: " + request.trial_phase
        if request.start_year:
            context += "\nStart year: " + str(request.start_year)
        if request.end_year:
            context += "\nEnd year: " + str(request.end_year)

        response = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            max_tokens=1000,
            messages=[
                {"role": "system", "content": PLAN_SYSTEM_PROMPT},
                {"role": "user", "content": context},
            ],
            response_format=PLAN_JSON_SCHEMA,
        )

        raw = response.choices[0].message.content.strip()

        try:
            plan_dict = json.loads(raw)
            plan = AgentPlan(**plan_dict)

            if not plan.validate_viz_intent_compatibility():
                valid_types = INTENT_VIZ_MAP.get(plan.intent, [])
                if valid_types:
                    logger.warning(
                        "LLM chose incompatible viz " + str(plan.viz_type) +
                        " for intent " + str(plan.intent) +
                        ". Falling back to " + str(valid_types[0])
                    )
                    plan = plan.model_copy(update={"viz_type": valid_types[0]})

            return plan

        except Exception as e:
            logger.warning("Plan parsing failed (" + str(e) + "), retrying with error context")
            return await self._retry_plan(request, context, str(e))

    async def _retry_plan(self, request: QueryRequest, original_context: str, error: str) -> AgentPlan:
        """
        Re-prompt the LLM with the validation error to get a corrected plan.
        Falls back to rule-based planning if the retry also fails.
        """
        retry_context = (
            original_context
            + chr(10) + chr(10)
            + "Your previous response failed validation with this error: "
            + error
            + chr(10)
            + "Please fix the plan and return valid JSON matching the required schema exactly."
        )
        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                max_tokens=1000,
                messages=[
                    {"role": "system", "content": PLAN_SYSTEM_PROMPT},
                    {"role": "user", "content": retry_context},
                ],
                response_format=PLAN_JSON_SCHEMA,
            )
            raw = response.choices[0].message.content.strip()
            plan_dict = json.loads(raw)
            plan = AgentPlan(**plan_dict)
            logger.info("Retry plan succeeded")
            return plan
        except Exception as e2:
            logger.warning("Retry plan also failed (" + str(e2) + "), using rule-based fallback")
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
        """
        async with ClinicalTrialsClient(mock=self.mock) as ct_client:
            plan_context = (
                "User query: " + request.query + "\n\n"
                "Analysis plan:\n"
                "- Intent: " + str(plan.intent) + "\n"
                "- Viz type: " + str(plan.viz_type) + "\n"
                "- Aggregation field: " + plan.aggregation_field + "\n"
                "- Filters: " + str(plan.filters.model_dump(exclude_none=True)) + "\n"
                "- Requires multiple searches: " + str(plan.requires_multiple_searches) + "\n\n"
                "Execute this plan using the available tools."
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

                messages.append(message)

                if finish_reason == "stop" or not message.tool_calls:
                    break

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
        logger.info("Tool call: " + tool_name + "(" + str(list(tool_input.keys())) + ")")

        match tool_name:
            case "search_trials":
                label = tool_input.pop("studies_key", "primary")
                filters_desc = ", ".join(str(k) + "=" + str(v) for k, v in tool_input.items() if v)
                await self.emit("Searching ClinicalTrials.gov...", filters_desc or "Fetching trials")
                with Timer() as t:
                    studies = await search_trials(ct_client, **tool_input)
                study_store[label] = studies
                self.trace.add_step(StepTrace(
                    step="tool_call",
                    input={"tool": "search_trials", "filters": tool_input},
                    output={"result_count": len(studies), "label": label},
                    duration_ms=t.elapsed_ms,
                ))
                await self.emit("Retrieved " + str(len(studies)) + " trials", "Search complete for " + label + " query")
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
                await self.emit("Aggregating " + str(len(studies)) + " trials...", "Grouping by " + field)
                with Timer() as t:
                    result = aggregate(
                        studies=studies,
                        field=field,
                        top_n=tool_input.get("top_n"),
                        label=tool_input.get("label"),
                    )
                self.trace.add_step(StepTrace(
                    step="tool_call",
                    input={"tool": "aggregate", "field": field},
                    output={"result_count": len(result)},
                    duration_ms=t.elapsed_ms,
                ))
                await self.emit("Aggregation complete", str(len(result)) + " data points produced")
                return result

            case "get_study_details":
                await self.emit("Fetching citation details...", tool_input["nct_id"])
                with Timer() as t:
                    result = await get_study_details(ct_client, tool_input["nct_id"])
                self.trace.add_step(StepTrace(
                    step="tool_call",
                    input={"tool": "get_study_details", "nct_id": tool_input["nct_id"]},
                    output={"found": result is not None},
                    duration_ms=t.elapsed_ms,
                ))
                return result

            case _:
                logger.warning("Unknown tool: " + tool_name)
                return {"error": "Unknown tool: " + tool_name}

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
                bucket_order = ["<50", "50-99", "100-249", "250-499", "500-999", "1000-1999", "2000+", "Unknown"]
                viz_data = sorted(
                    viz_data,
                    key=lambda d: bucket_order.index(d.get("enrollment_bucket", "Unknown"))
                    if d.get("enrollment_bucket") in bucket_order else 99
                )
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
                    id="sponsor_" + sponsor,
                    label=sponsor,
                    type="sponsor",
                )
            for condition in (study.conditions or [])[:2]:
                cond_key = condition[:50]
                if cond_key not in condition_nodes:
                    condition_nodes[cond_key] = NodeDef(
                        id="condition_" + cond_key,
                        label=cond_key,
                        type="condition",
                    )
                edge_key = ("sponsor_" + sponsor, "condition_" + cond_key)
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
                return "Trials Over Time: " + subject
            case IntentType.DISTRIBUTION:
                return subject + " Trials by " + field_label
            case IntentType.COMPARISON:
                secondary = f.secondary_drug or f.secondary_condition or "Comparison"
                return "Trial Comparison: " + subject + " vs " + secondary + " by " + field_label
            case IntentType.GEOGRAPHIC:
                return "Geographic Distribution of " + subject + " Trials"
            case IntentType.NETWORK:
                return "Sponsor-Condition Network: " + subject
            case _:
                return subject + " Clinical Trials Summary"

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
        "enrollment_bucket": "Enrollment Size",
    }.get(field, field.replace("_", " ").title())
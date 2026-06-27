# ClinicalTrials Query-to-Visualization Agent

A FastAPI backend that converts natural language questions about clinical trials into structured visualization specifications, backed by real-time [ClinicalTrials.gov](https://clinicaltrials.gov/data-api/api) data.

---

## How to Run

### Prerequisites

- Python 3.12+
- An OpenAI API key

### Install

```bash
git clone https://github.com/utkarshsingh26/Clinical_Trials_Agent.git
cd Clinical_Trials_Agent
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=your_key_here
```

### Start the server

```bash
# Live mode (hits real ClinicalTrials.gov API)
uvicorn app.main:app --reload

# Mock mode (synthetic CT data, still uses real Anthropic API)
MOCK_MODE=true uvicorn app.main:app --reload
```

Server runs at `http://localhost:8000`. Docs at `http://localhost:8000/docs`.

### Run example queries

```bash
# With real data
OPENAI_API_KEY=your_key python examples/run_examples.py

# With mock CT data
MOCK_MODE=true python examples/run_examples.py
```

---

## Request / Response Schema

### POST `/query`

**Request**

| Field | Type | Required | Description |
|---|---|---|---|
| `query` | string | Yes | Natural language question about clinical trials |
| `drug_name` | string | No | Drug/intervention name hint |
| `condition` | string | No | Disease/condition hint |
| `trial_phase` | string | No | Phase filter (PHASE1–PHASE4, EARLY_PHASE1, NA) |
| `sponsor` | string | No | Sponsor organization name |
| `country` | string | No | Country filter |
| `start_year` | integer | No | Filter trials starting on or after this year |
| `end_year` | integer | No | Filter trials starting on or before this year |

Optional structured fields reduce LLM parsing burden and improve accuracy on specific identifiers like drug names.

**Example request**

```json
{
  "query": "How has the number of trials for Pembrolizumab changed per year since 2015?",
  "drug_name": "Pembrolizumab",
  "start_year": 2015
}
```

**Response**

```json
{
  "visualization": {
    "type": "time_series",
    "title": "Trials Over Time: Pembrolizumab",
    "encoding": {
      "encoding_type": "cartesian",
      "x": { "field": "start_year", "label": "Year", "type": "temporal" },
      "y": { "field": "trial_count", "label": "Number of Trials", "type": "quantitative" }
    },
    "data": [
      {
        "start_year": "2015",
        "trial_count": 12,
        "citations": [
          {
            "nct_id": "NCT02362360",
            "excerpt": "Phase 2 randomized study evaluating pembrolizumab...",
            "field_name": "protocolSection.descriptionModule.briefSummary",
            "url": "https://clinicaltrials.gov/study/NCT02362360"
          }
        ]
      }
    ]
  },
  "meta": {
    "query_interpretation": "User asked about change over time — trend intent maps to time_series.",
    "filters_applied": { "drug_name": "Pembrolizumab", "start_year": 2015 },
    "total_trials_retrieved": 247,
    "assumptions": [],
    "tool_calls_made": 2,
    "source": "clinicaltrials.gov"
  },
  "plan": {
    "intent": "trend",
    "viz_type": "time_series",
    "aggregation_field": "start_year",
    "reasoning": "...",
    "requires_multiple_searches": false,
    "filters": { "drug_name": "Pembrolizumab", "start_year": 2015 }
  }
}
```

Full JSON schemas available at `/schema/request` and `/schema/response`.

---

## Supported Visualization Types

| Type | Intent | Example query |
|---|---|---|
| `bar_chart` | Distribution / Summary | "How are lung cancer trials distributed across phases?" |
| `grouped_bar_chart` | Comparison | "Compare phases for Pembrolizumab vs Nivolumab" |
| `time_series` | Trend | "How has the number of Pembrolizumab trials changed per year?" |
| `histogram` | Distribution | "What's the enrollment size distribution for Phase 3 trials?" |
| `network_graph` | Relationships | "Show a network of sponsors and conditions for breast cancer" |

---

## Key Design Decisions

### 1. Forced planning step before tool calls

The agent produces a validated `AgentPlan` (intent, viz_type, filters, aggregation_field) before making any API calls. This plan is validated with Pydantic — including a compatibility check between intent and viz_type. If the LLM produces an incompatible combination (e.g. `network_graph` for a trend query), the system falls back to the correct type rather than propagating the error. This is the primary hallucination guard.

### 2. Raw OpenAI SDK, no framework

The agentic loop is implemented directly against the Anthropic messages API with explicit message history management. No LangChain or LangGraph. This keeps the control flow transparent, the state explicit, and the debugging tractable. The loop is ~50 lines of code that are easy to reason about.

### 3. Hard cap of 5 tool calls per request

The agent loop is bounded at `MAX_TOOL_CALLS = 5`. For this domain, every query type is answerable in 2–3 tool calls (search → aggregate, or search → search → aggregate for comparisons). The cap prevents runaway execution and ensures predictable latency. It maps directly to the rubric's "include validation or constraints" criterion.

### 4. Deterministic visualization assembly

The `VisualizationSpec` is built from the plan + aggregated data in Python code, not by parsing LLM-generated JSON. The LLM decides intent and viz_type (upfront, in the planning step), but the schema construction is deterministic. This prevents hallucinated field names and schema drift.

### 5. Coarse-grained tools

Three tools: `search_trials`, `aggregate`, `get_study_details`. Coarse granularity means the LLM makes fewer decisions, each of which is easier to validate. Fine-grained tools would give the LLM more surface area for invalid combinations.

### 6. Discriminated union encoding

`VisualizationSpec.encoding` is a Pydantic discriminated union on `encoding_type`: either `CartesianEncoding` (x/y/series for bar/line/scatter) or `NetworkEncoding` (nodes/edges for graph). This means a frontend can branch cleanly on `encoding.encoding_type` with full type safety rather than doing duck-typing on the encoding object.

### 7. Deep citations baked into every data point

Each aggregated data point includes up to 3 `Citation` objects with `nct_id`, text `excerpt`, `field_name`, and `url`. Citations are built during aggregation from actual API response text — they are never generated by the LLM.

---

## Tradeoffs

**Coarse tools vs fine-grained tools:** Coarse tools mean the agent can't do partial aggregations or field-level filtering during aggregation. We compensate with `top_n` and `label` params on aggregate.

**Deterministic viz assembly vs LLM-generated spec:** We lose flexibility for novel query types that don't fit our intent taxonomy. We gain reliability and schema correctness.

**5-call cap:** Deeply nested comparison queries (3+ entities) can't be fully explored. This is an acceptable tradeoff given the query types in scope.

**Single-pass planning:** The plan is produced in one LLM call and not revised during execution. If the initial plan is wrong, the output will be wrong. A future version could add a reflection step after tool results come in.

---

## What I Would Improve With More Time

1. **Streaming responses** — the `/query` endpoint blocks until the full agent loop completes. Server-sent events would let the frontend show progress.
2. **Caching** — identical search params should cache CT API results for the session to avoid redundant network calls and reduce latency.
3. **Richer scatter plots** — enrollment vs. start_year scatter with phase as color would be a meaningful addition for studying trial growth patterns.
4. **Multi-condition network graphs** — currently the network shows sponsor-condition links. Extending to drug-drug co-occurrence networks would require a second aggregation pass.
5. **Plan revision** — after the first search returns results, a second LLM call could revise the plan based on what data was actually available, catching cases where the original intent was mismatched to the data.
6. **Structured logging with trace IDs** — each request should carry a trace ID through all log lines for debuggability in production.

---

## AI Tools Used

- **Claude (Anthropic)** — used for query planning and tool orchestration within the agent itself
- **Claude (claude.ai)** — used as a coding assistant during development for schema design, agent architecture, and code review

### What I designed deliberately

- The three-layer architecture (plan → tool loop → deterministic assembly)
- The forced planning step as a hallucination guard
- The discriminated union encoding schema
- The MAX_TOOL_CALLS bound and its justification
- The decision to build VisualizationSpec in code rather than parsing LLM JSON

### What I generated and adapted

- Boilerplate FastAPI middleware and CORS setup
- The mock data generator for testing
- The `from_api_response()` parser for CT API's nested JSON structure

### How I validated correctness

- Pydantic validation on every input/output boundary
- Direct unit tests on tool functions (aggregate, citations)
- End-to-end tests against mock CT data
- Schema endpoint tests confirming FastAPI serialization matches expected structure
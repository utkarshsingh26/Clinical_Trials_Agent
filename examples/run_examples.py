"""
Example runs for submission — 5 queries covering all supported viz types.

Run with:
    OPENAI_API_KEY=your_key python examples/run_examples.py

Set MOCK_MODE=true to run without hitting ClinicalTrials.gov (uses synthetic data).
The LLM calls still run against the real Anthropic API.
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

from app.agent.agent import TrialsAgent
from app.schemas.request import QueryRequest

MOCK_MODE = os.environ.get("MOCK_MODE", "false").lower() == "true"

EXAMPLES = [
    {
        "id": 1,
        "description": "Time trend — trials over time",
        "request": QueryRequest(
            query="How has the number of trials for Pembrolizumab changed per year since 2015?",
            drug_name="Pembrolizumab",
            start_year=2015,
        ),
    },
    {
        "id": 2,
        "description": "Distribution — phase breakdown",
        "request": QueryRequest(
            query="How are lung cancer trials distributed across phases?",
            condition="lung cancer",
        ),
    },
    {
        "id": 3,
        "description": "Comparison — two drugs by phase",
        "request": QueryRequest(
            query="Compare trial phases for Pembrolizumab vs Nivolumab",
            drug_name="Pembrolizumab",
        ),
    },
    {
        "id": 4,
        "description": "Geographic — country distribution",
        "request": QueryRequest(
            query="Which countries have the most recruiting trials for type 2 diabetes?",
            condition="type 2 diabetes",
        ),
    },
    {
        "id": 5,
        "description": "Network — sponsor-condition relationships",
        "request": QueryRequest(
            query="Show a network of sponsors and conditions for breast cancer trials",
            condition="breast cancer",
        ),
    },
]


async def run_example(agent: TrialsAgent, example: dict) -> dict:
    print(f"\n{'='*60}")
    print(f"Example {example['id']}: {example['description']}")
    print(f"Query: {example['request'].query}")
    print("Running...")

    try:
        response = await agent.run(example["request"])
        result = {
            "example_id": example["id"],
            "description": example["description"],
            "query": example["request"].query,
            "output": response.model_dump(mode="json"),
        }
        print(f"Viz type: {response.visualization.type}")
        print(f"Title: {response.visualization.title}")
        print(f"Data points: {len(response.visualization.data)}")
        print(f"Tool calls: {response.meta.tool_calls_made}")
        print(f"Trials retrieved: {response.meta.total_trials_retrieved}")
        return result
    except Exception as e:
        print(f"ERROR: {e}")
        return {
            "example_id": example["id"],
            "description": example["description"],
            "query": example["request"].query,
            "error": str(e),
        }


async def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    print(f"Running {len(EXAMPLES)} example queries")
    print(f"Mode: {'MOCK (synthetic CT data)' if MOCK_MODE else 'LIVE (real ClinicalTrials.gov)'}")

    agent = TrialsAgent(mock=MOCK_MODE)
    results = []

    for example in EXAMPLES:
        result = await run_example(agent, example)
        results.append(result)

    # Save full outputs
    output_path = os.path.join(os.path.dirname(__file__), "example_outputs.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"All examples complete. Full outputs saved to {output_path}")
    successful = sum(1 for r in results if "error" not in r)
    print(f"Success: {successful}/{len(results)}")


if __name__ == "__main__":
    asyncio.run(main())
"""
Observability / request tracing for the ClinicalTrials Agent.

Captures a structured trace for every request — each step, its inputs,
outputs, duration, and outcome. Stored in an in-memory ring buffer.

In production this would be forwarded to Datadog, Prometheus, or a
logging aggregator via the /traces endpoint or a log shipper sidecar.
"""

import time
import uuid
from collections import deque
from typing import Any
from dataclasses import dataclass, field, asdict


MAX_TRACES = 100  # Ring buffer — oldest traces dropped when full


@dataclass
class StepTrace:
    step: str
    input: dict = field(default_factory=dict)
    output: dict = field(default_factory=dict)
    duration_ms: int = 0
    status: str = "success"
    error: str | None = None


@dataclass
class RequestTrace:
    request_id: str
    timestamp: str
    query: str
    steps: list[StepTrace] = field(default_factory=list)
    total_duration_ms: int = 0
    status: str = "pending"
    viz_type: str | None = None
    intent: str | None = None
    fallback_used: bool = False

    def add_step(self, step: StepTrace):
        self.steps.append(step)

    def complete(self, status: str = "success"):
        self.status = status

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# Global in-memory ring buffer
_traces: deque = deque(maxlen=MAX_TRACES)


def new_trace(query: str) -> RequestTrace:
    """Create a new request trace and register it."""
    from datetime import datetime, timezone
    trace = RequestTrace(
        request_id=str(uuid.uuid4())[:8],
        timestamp=datetime.now(timezone.utc).isoformat(),
        query=query,
    )
    _traces.append(trace)
    return trace


def get_traces() -> list[dict]:
    """Return all stored traces as dicts, most recent first."""
    return [t.to_dict() for t in reversed(list(_traces))]


def get_trace(request_id: str) -> dict | None:
    """Return a single trace by request_id."""
    for t in _traces:
        if t.request_id == request_id:
            return t.to_dict()
    return None


class Timer:
    """Context manager for timing a step."""
    def __init__(self):
        self.elapsed_ms = 0
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = round((time.perf_counter() - self._start) * 1000)
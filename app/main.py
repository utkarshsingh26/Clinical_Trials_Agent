import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.agent.agent import TrialsAgent
from app.observability import get_traces, get_trace
from app.schemas.request import QueryRequest
from app.schemas.response import QueryResponse

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MOCK_MODE = os.environ.get("MOCK_MODE", "false").lower() == "true"


@asynccontextmanager
async def lifespan(app: FastAPI):
    mode = "MOCK" if MOCK_MODE else "LIVE"
    logger.info(f"Starting ClinicalTrials Agent in {mode} mode")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="ClinicalTrials Query-to-Visualization Agent",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000)
    logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({duration}ms)")
    return response


@app.get("/health")
async def health():
    return {"status": "ok", "mock_mode": MOCK_MODE}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    if not os.environ.get("OPENAI_API_KEY") and not MOCK_MODE:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set.")
    try:
        agent = TrialsAgent(mock=MOCK_MODE)
        response = await agent.run(request)
        return response
    except Exception as e:
        logger.exception(f"Agent error for query: {request.query[:80]}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    if not os.environ.get("OPENAI_API_KEY") and not MOCK_MODE:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set.")

    async def event_generator():
        progress_queue = asyncio.Queue()
        agent = TrialsAgent(mock=MOCK_MODE)
        agent_task = asyncio.create_task(agent.run(request, progress_queue=progress_queue))

        while not agent_task.done():
            try:
                event = await asyncio.wait_for(progress_queue.get(), timeout=0.3)
                payload = json.dumps({"type": "progress", "event": event["event"], "detail": event["detail"]})
                yield "data: " + payload + "\n\n"
            except asyncio.TimeoutError:
                yield "data: " + json.dumps({"type": "ping"}) + "\n\n"

        while not progress_queue.empty():
            event = await progress_queue.get()
            payload = json.dumps({"type": "progress", "event": event["event"], "detail": event["detail"]})
            yield "data: " + payload + "\n\n"

        try:
            response = await agent_task
            result = json.dumps({"type": "result", "data": response.model_dump(mode="json")})
            yield "data: " + result + "\n\n"
        except Exception as e:
            logger.exception("Agent error during streaming")
            error = json.dumps({"type": "error", "message": str(e)})
            yield "data: " + error + "\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/traces")
async def traces():
    """
    Returns all stored request traces for observability.
    In production, forward these to Datadog, Prometheus, or a log aggregator.
    Stored in-memory — resets on server restart. Max 100 traces retained.
    """
    return {"traces": get_traces(), "count": len(get_traces())}


@app.get("/traces/{request_id}")
async def trace(request_id: str):
    """Returns a single trace by request_id."""
    t = get_trace(request_id)
    if not t:
        raise HTTPException(status_code=404, detail="Trace not found")
    return t


@app.get("/schema/request")
async def request_schema():
    return QueryRequest.model_json_schema()


@app.get("/schema/response")
async def response_schema():
    return QueryResponse.model_json_schema()
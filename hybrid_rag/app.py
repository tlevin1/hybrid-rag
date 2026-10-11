"""FastAPI service: POST /ask with a choice of retriever.

    uvicorn hybrid_rag.app:app --reload

With MONGODB_URI set, every request is traced to MongoDB (hybrid_rag/traces.py),
and POST /feedback and GET /stats are enabled.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Literal

import anthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .corpus import load_chunks
from .generate import ANSWER_MODEL, NO_ANSWER, answer
from .retrieval import Mode, Retriever

PERSIST_DIR = os.getenv("CHROMA_DIR", "data/chroma")
state: dict = {}
log = logging.getLogger(__name__)


def make_trace_store():
    """A TraceStore if MONGODB_URI is set, otherwise None (tracing off)."""
    uri = os.getenv("MONGODB_URI")
    if not uri:
        return None
    from .traces import TraceStore

    return TraceStore(uri)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Index once at startup; the Chroma index persists, so restarts don't re-embed.
    state["retriever"] = Retriever(load_chunks(), persist_dir=PERSIST_DIR)
    state["llm"] = anthropic.Anthropic()
    state["traces"] = make_trace_store()
    yield


app = FastAPI(title="Hybrid RAG", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(examples=["What did the author work on before college?"])
    mode: Mode = "hybrid"
    top_k: int = Field(4, ge=1, le=20)


class Source(BaseModel):
    chunk_id: str
    text: str
    score: float
    ranks: dict[str, int]


class AskResponse(BaseModel):
    answer: str
    mode: Mode
    model: str
    sources: list[Source]
    trace_id: str | None = None  # pass to POST /feedback


class FeedbackRequest(BaseModel):
    trace_id: str
    rating: Literal[1, -1]
    comment: str | None = Field(None, max_length=2000)


def require_traces():
    if state.get("traces") is None:
        raise HTTPException(503, "Tracing is off: set MONGODB_URI to enable feedback and stats.")
    return state["traces"]


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(state["retriever"].bm25.chunks)}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty.")
    t0 = time.perf_counter()
    hits = state["retriever"].search(req.question, k=req.top_k, mode=req.mode)
    t1 = time.perf_counter()
    try:
        text = answer(state["llm"], req.question, hits)
    except anthropic.APIError as e:
        raise HTTPException(502, f"LLM API error: {e}") from e
    t2 = time.perf_counter()
    sources = [Source(chunk_id=h.chunk_id, text=h.text, score=h.score, ranks=h.ranks) for h in hits]
    return AskResponse(
        answer=text, mode=req.mode, model=ANSWER_MODEL, sources=sources,
        trace_id=record_trace(req, text, sources, {
            "retrieval": round((t1 - t0) * 1000, 1),
            "generation": round((t2 - t1) * 1000, 1),
            "total": round((t2 - t0) * 1000, 1),
        }),
    )


def record_trace(req: AskRequest, text: str, sources: list[Source], latency_ms: dict) -> str | None:
    """Tracing is observability, not the product: if MongoDB is down, the user
    still gets their answer and the failure is logged."""
    store = state.get("traces")
    if store is None:
        return None
    try:
        return store.record(
            question=req.question, mode=req.mode, top_k=req.top_k, model=ANSWER_MODEL,
            answer=text, refused=text == NO_ANSWER, latency_ms=latency_ms,
            sources=[{"chunk_id": s.chunk_id, "score": s.score, "ranks": s.ranks} for s in sources],
        )
    except Exception:
        log.exception("could not record trace")
        return None


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    if not require_traces().add_feedback(req.trace_id, req.rating, req.comment):
        raise HTTPException(404, "Unknown or expired trace_id.")
    return {"ok": True}


@app.get("/stats")
def stats(hours: int = 24):
    """Per retrieval mode over the last `hours`: volume, refusal rate, latency
    percentiles, feedback; plus the most recent refusals to read."""
    store = require_traces()
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    return {
        "since": since.isoformat(),
        "by_mode": store.stats(since),
        "recent_refusals": [
            {"trace_id": r["_id"], "ts": r["ts"].isoformat(), "mode": r["mode"], "question": r["question"],
             "retrieved": [s["chunk_id"] for s in r.get("sources", [])]}
            for r in store.recent_refusals(since)
        ],
    }

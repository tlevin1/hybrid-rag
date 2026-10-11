"""FastAPI service: POST /ask with a choice of retriever.

    uvicorn hybrid_rag.app:app --reload
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

import anthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .corpus import load_chunks
from .generate import ANSWER_MODEL, answer
from .retrieval import Mode, Retriever

PERSIST_DIR = os.getenv("CHROMA_DIR", "data/chroma")
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Index once at startup; the Chroma index persists, so restarts don't re-embed.
    state["retriever"] = Retriever(load_chunks(), persist_dir=PERSIST_DIR)
    state["llm"] = anthropic.Anthropic()
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


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(state["retriever"].bm25.chunks)}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty.")
    hits = state["retriever"].search(req.question, k=req.top_k, mode=req.mode)
    try:
        text = answer(state["llm"], req.question, hits)
    except anthropic.APIError as e:
        raise HTTPException(502, f"LLM API error: {e}") from e
    return AskResponse(
        answer=text, mode=req.mode, model=ANSWER_MODEL,
        sources=[Source(chunk_id=h.chunk_id, text=h.text, score=h.score, ranks=h.ranks) for h in hits],
    )

"""API and answer generation with a fake LLM: no network, no API key."""

import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from hybrid_rag import app as app_module
from hybrid_rag.generate import NO_ANSWER, answer, build_prompt
from hybrid_rag.retrieval import Hit, Retriever

from .conftest import CHUNKS, HashedBagOfWords


class FakeLLM:
    def __init__(self, reply: str):
        self.reply, self.calls = reply, []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.reply)])


def test_prompt_labels_each_passage_with_its_id():
    prompt = build_prompt("Q?", [Hit("c007", "some text", 1.0)])
    assert "[c007]\nsome text" in prompt and prompt.endswith("Question: Q?")


def test_no_hits_means_no_llm_call():
    llm = FakeLLM("should not be used")
    assert answer(llm, "Q?", []) == NO_ANSWER and llm.calls == []


def test_refusal_with_extra_explanation_is_normalized():
    llm = FakeLLM(NO_ANSWER + "\n\nThe passages discuss painting, not geography.")
    assert answer(llm, "Q?", [Hit("c0", "t", 1.0)]) == NO_ANSWER


@pytest.fixture
def client(monkeypatch):
    llm = FakeLLM("Yahoo bought it [c002].")
    retriever = Retriever(CHUNKS, embedding_function=HashedBagOfWords(), collection=f"t{uuid.uuid4().hex}")
    monkeypatch.setattr(app_module, "load_chunks", lambda: CHUNKS)
    monkeypatch.setattr(app_module, "Retriever", lambda chunks, persist_dir: retriever)
    monkeypatch.setattr(app_module.anthropic, "Anthropic", lambda: llm)
    with TestClient(app_module.app) as c:
        yield c


def test_ask_returns_answer_and_sources_with_ranks(client):
    r = client.post("/ask", json={"question": "Who bought Viaweb?", "mode": "hybrid", "top_k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Yahoo bought it [c002]." and body["mode"] == "hybrid"
    assert body["sources"][0]["chunk_id"] == "c002"
    assert set(body["sources"][0]["ranks"]) == {"bm25", "vector"}


def test_ask_rejects_unknown_mode_and_empty_question(client):
    assert client.post("/ask", json={"question": "x", "mode": "sparse"}).status_code == 422
    assert client.post("/ask", json={"question": "  "}).status_code == 400

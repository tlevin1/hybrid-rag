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


# Tracing, feedback and stats (with an in-memory store; the MongoDB store itself
# is tested against a real database in test_traces.py)


class FakeTraceStore:
    def __init__(self, fail=False):
        self.records, self.feedback, self.fail = [], {}, fail

    def record(self, **kw):
        if self.fail:
            raise ConnectionError("MongoDB is down")
        self.records.append(kw)
        return f"t{len(self.records)}"

    def add_feedback(self, trace_id, rating, comment=None):
        if trace_id not in {f"t{i}" for i in range(1, len(self.records) + 1)}:
            return False
        self.feedback[trace_id] = (rating, comment)
        return True

    def stats(self, since):
        return [{"mode": "hybrid", "requests": len(self.records)}]

    def recent_refusals(self, since):
        return []


def traced_client(monkeypatch, store, reply="Yahoo bought it [c002]."):
    retriever = Retriever(CHUNKS, embedding_function=HashedBagOfWords(), collection=f"t{uuid.uuid4().hex}")
    monkeypatch.setattr(app_module, "load_chunks", lambda: CHUNKS)
    monkeypatch.setattr(app_module, "Retriever", lambda chunks, persist_dir: retriever)
    monkeypatch.setattr(app_module.anthropic, "Anthropic", lambda: FakeLLM(reply))
    monkeypatch.setattr(app_module, "make_trace_store", lambda: store)
    return TestClient(app_module.app)


def test_ask_records_a_trace_and_returns_its_id(monkeypatch):
    store = FakeTraceStore()
    with traced_client(monkeypatch, store) as c:
        body = c.post("/ask", json={"question": "Who bought Viaweb?", "mode": "bm25"}).json()
    assert body["trace_id"] == "t1"
    rec = store.records[0]
    assert rec["mode"] == "bm25" and rec["refused"] is False
    assert set(rec["latency_ms"]) == {"retrieval", "generation", "total"}
    assert "text" not in rec["sources"][0]


def test_refusals_are_flagged_in_the_trace(monkeypatch):
    store = FakeTraceStore()
    with traced_client(monkeypatch, store, reply=NO_ANSWER) as c:
        c.post("/ask", json={"question": "Capital of Australia?"})
    assert store.records[0]["refused"] is True


def test_a_tracing_outage_does_not_break_answers(monkeypatch):
    with traced_client(monkeypatch, FakeTraceStore(fail=True)) as c:
        r = c.post("/ask", json={"question": "Who bought Viaweb?"})
    assert r.status_code == 200 and r.json()["trace_id"] is None and r.json()["answer"]


def test_feedback_endpoint(monkeypatch):
    store = FakeTraceStore()
    with traced_client(monkeypatch, store) as c:
        tid = c.post("/ask", json={"question": "Who bought Viaweb?"}).json()["trace_id"]
        assert c.post("/feedback", json={"trace_id": tid, "rating": -1, "comment": "wrong"}).status_code == 200
        assert c.post("/feedback", json={"trace_id": "nope", "rating": 1}).status_code == 404
        assert c.post("/feedback", json={"trace_id": tid, "rating": 5}).status_code == 422
    assert store.feedback[tid] == (-1, "wrong")


def test_stats_endpoint(monkeypatch):
    with traced_client(monkeypatch, FakeTraceStore()) as c:
        c.post("/ask", json={"question": "Who bought Viaweb?"})
        body = c.get("/stats?hours=1").json()
    assert body["by_mode"] == [{"mode": "hybrid", "requests": 1}] and body["recent_refusals"] == []


def test_feedback_and_stats_need_tracing(monkeypatch):
    with traced_client(monkeypatch, None) as c:
        assert c.post("/ask", json={"question": "Who bought Viaweb?"}).json()["trace_id"] is None
        assert c.post("/feedback", json={"trace_id": "t1", "rating": 1}).status_code == 503
        assert c.get("/stats").status_code == 503

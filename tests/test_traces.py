"""TraceStore against a real MongoDB (local Docker, or the service container in
CI). Each test gets its own database, dropped afterwards. Skipped when no
MongoDB is reachable."""

import os
import uuid
from datetime import timedelta

import pytest

pymongo = pytest.importorskip("pymongo")

from hybrid_rag.traces import RETENTION_DAYS, TraceStore, now  # noqa: E402

URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")


@pytest.fixture
def store():
    client = pymongo.MongoClient(URI, serverSelectionTimeoutMS=1500, tz_aware=True)
    try:
        client.admin.command("ping")
    except pymongo.errors.PyMongoError:
        pytest.skip(f"no MongoDB at {URI}")
    db = f"test_{uuid.uuid4().hex[:12]}"
    yield TraceStore(db=db, client=client)
    client.drop_database(db)


def record(store, mode="hybrid", refused=False, total_ms=100.0, ts=None, question="q"):
    return store.record(
        question=question, mode=mode, top_k=4, model="claude-haiku-4-5",
        answer="I don't have enough information to answer that." if refused else "An answer [c001].",
        refused=refused, latency_ms={"retrieval": 5.0, "generation": total_ms - 5.0, "total": total_ms},
        sources=[{"chunk_id": "c001", "score": 0.03, "ranks": {"bm25": 2, "vector": 1}}], ts=ts,
    )


def test_trace_round_trips_with_nested_sources_and_utc_time(store):
    doc = store.get(record(store))
    assert doc["sources"][0] == {"chunk_id": "c001", "score": 0.03, "ranks": {"bm25": 2, "vector": 1}}
    assert doc["ts"].tzinfo is not None and abs((now() - doc["ts"]).total_seconds()) < 10
    assert "text" not in doc["sources"][0]  # passages are referenced by id only


def test_indexes_support_the_queries_and_expire_old_traces(store):
    info = store.col.index_information()
    assert info["mode_ts"]["key"] == [("mode", 1), ("ts", 1)]
    assert info["ttl_ts"]["expireAfterSeconds"] == RETENTION_DAYS * 24 * 3600


def test_window_queries_by_mode_use_the_compound_index(store):
    record(store)
    plan = store.col.find({"mode": "hybrid", "ts": {"$gte": now() - timedelta(hours=1)}}).explain()
    assert "mode_ts" in str(plan["queryPlanner"]["winningPlan"])


def test_feedback_is_attached_and_can_be_changed(store):
    tid = record(store)
    assert store.add_feedback(tid, 1, "helpful")
    assert store.add_feedback(tid, -1, "actually wrong")
    fb = store.get(tid)["feedback"]
    assert (fb["rating"], fb["comment"]) == (-1, "actually wrong")


def test_feedback_for_unknown_trace_or_bad_rating(store):
    assert store.add_feedback("does-not-exist", 1) is False
    with pytest.raises(ValueError):
        store.add_feedback(record(store), 5)


def test_stats_per_mode(store):
    for ms in (100, 200, 300, 400, 1000):
        record(store, mode="hybrid", total_ms=ms)
    record(store, mode="bm25", refused=True)
    record(store, mode="bm25")
    store.add_feedback(record(store, mode="vector"), 1)
    by_mode = {r["mode"]: r for r in store.stats(now() - timedelta(hours=1))}
    assert by_mode["hybrid"]["requests"] == 5 and by_mode["bm25"]["refusal_rate"] == 0.5
    assert by_mode["hybrid"]["latency_p50_ms"] <= by_mode["hybrid"]["latency_p95_ms"]
    assert (by_mode["vector"]["thumbs_up"], by_mode["vector"]["thumbs_down"]) == (1, 0)


def test_a_trace_written_just_before_the_query_is_counted(store):
    # Regression: an "until now" default lost a trace written in the same
    # millisecond as the query (MongoDB stores milliseconds).
    for _ in range(30):
        record(store)
    assert sum(r["requests"] for r in store.stats(now() - timedelta(hours=1))) == 30


def test_stats_respect_the_time_window(store):
    record(store, ts=now() - timedelta(days=2))
    record(store, ts=now() - timedelta(hours=2))
    record(store)
    assert sum(r["requests"] for r in store.stats(now() - timedelta(hours=24))) == 2
    assert sum(r["requests"] for r in store.stats(now() - timedelta(days=3), until=now() - timedelta(hours=1))) == 2


def test_recent_refusals_newest_first(store):
    record(store, refused=True, question="old", ts=now() - timedelta(hours=3))
    record(store, refused=True, question="new")
    record(store, refused=False, question="answered")
    got = store.recent_refusals(now() - timedelta(hours=24))
    assert [r["question"] for r in got] == ["new", "old"]

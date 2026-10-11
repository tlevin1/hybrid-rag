"""Request traces in MongoDB: what each /ask did, for monitoring and feedback.

One document per request. The shape is naturally nested (a variable list of
sources, each with a rank per retriever), which is why it's a document store
rather than a table:

    {
      "_id": "3f2a...",                         # trace id, returned to the client
      "ts": ISODate(...),                        # UTC
      "question": "...", "mode": "hybrid", "top_k": 4, "model": "claude-haiku-4-5",
      "answer": "...", "refused": false,
      "latency_ms": {"retrieval": 41.2, "generation": 812.5, "total": 853.7},
      "sources": [{"chunk_id": "c012", "score": 0.031, "ranks": {"bm25": 2, "vector": 1}}],
      "feedback": {"rating": 1, "comment": "...", "ts": ISODate(...)}   # added later
    }

Passage text is not stored, only chunk ids: it's in the corpus already and
would make every trace ~3 KB larger.

Indexes:
- (mode, ts): the monitoring queries filter by time window and group by mode.
- ts with a TTL: traces expire after RETENTION_DAYS, so the collection can't
  grow without bound and old questions aren't kept forever.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from pymongo import ASCENDING, MongoClient
from pymongo.collection import Collection

RETENTION_DAYS = 30


def now() -> datetime:
    return datetime.now(timezone.utc)


class TraceStore:
    def __init__(self, uri: str = "mongodb://localhost:27017", db: str = "hybrid_rag",
                 collection: str = "traces", client: MongoClient | None = None):
        # tz_aware so datetimes come back as UTC-aware, matching what was stored.
        self.client = client or MongoClient(uri, serverSelectionTimeoutMS=3000, tz_aware=True)
        self.col: Collection = self.client[db][collection]
        self.col.create_index([("mode", ASCENDING), ("ts", ASCENDING)], name="mode_ts")
        self.col.create_index("ts", name="ttl_ts", expireAfterSeconds=RETENTION_DAYS * 24 * 3600)

    def record(self, *, question: str, mode: str, top_k: int, model: str, answer: str,
               refused: bool, sources: list[dict], latency_ms: dict, ts: datetime | None = None) -> str:
        trace_id = uuid.uuid4().hex
        self.col.insert_one({
            "_id": trace_id, "ts": ts or now(), "question": question, "mode": mode, "top_k": top_k,
            "model": model, "answer": answer, "refused": refused, "latency_ms": latency_ms,
            "sources": sources,
        })
        return trace_id

    def add_feedback(self, trace_id: str, rating: int, comment: str | None = None) -> bool:
        """Attach a thumbs up (1) or down (-1). Re-rating replaces the earlier
        rating. Returns False if the trace doesn't exist (or has expired)."""
        if rating not in (1, -1):
            raise ValueError("rating must be 1 or -1")
        res = self.col.update_one(
            {"_id": trace_id},
            {"$set": {"feedback": {"rating": rating, "comment": comment, "ts": now()}}},
        )
        return res.matched_count == 1

    def get(self, trace_id: str) -> dict | None:
        return self.col.find_one({"_id": trace_id})

    def stats(self, since: datetime, until: datetime | None = None) -> list[dict]:
        """Per retrieval mode, over a time window: volume, refusal rate, latency
        percentiles and feedback. One aggregation, served by the (mode, ts) index.

        No upper bound unless `until` is given: MongoDB stores times to the
        millisecond, so a default of "until now" excluded a trace written in the
        same millisecond as the query (a test caught 29 of 30 being counted)."""
        window = {"$gte": since, **({"$lt": until} if until else {})}
        pipeline = [
            {"$match": {"ts": window}},
            {"$group": {
                "_id": "$mode",
                "requests": {"$sum": 1},
                "refusal_rate": {"$avg": {"$cond": ["$refused", 1, 0]}},
                "latency_p50_ms": {"$percentile": {"input": "$latency_ms.total", "p": [0.5], "method": "approximate"}},
                "latency_p95_ms": {"$percentile": {"input": "$latency_ms.total", "p": [0.95], "method": "approximate"}},
                "thumbs_up": {"$sum": {"$cond": [{"$eq": ["$feedback.rating", 1]}, 1, 0]}},
                "thumbs_down": {"$sum": {"$cond": [{"$eq": ["$feedback.rating", -1]}, 1, 0]}},
            }},
            {"$project": {
                "_id": 0, "mode": "$_id", "requests": 1,
                "refusal_rate": {"$round": ["$refusal_rate", 3]},
                "latency_p50_ms": {"$round": [{"$first": "$latency_p50_ms"}, 1]},
                "latency_p95_ms": {"$round": [{"$first": "$latency_p95_ms"}, 1]},
                "thumbs_up": 1, "thumbs_down": 1,
            }},
            {"$sort": {"mode": 1}},
        ]
        return list(self.col.aggregate(pipeline))

    def recent_refusals(self, since: datetime, limit: int = 20) -> list[dict]:
        """The questions the system declined, newest first: the list to read when
        refusal rate rises, to tell retrieval misses from over-cautious answers."""
        cursor = self.col.find(
            {"ts": {"$gte": since}, "refused": True},
            {"question": 1, "mode": 1, "ts": 1, "sources.chunk_id": 1},
        ).sort("ts", -1).limit(limit)
        return list(cursor)


def default_window(hours: int = 24) -> datetime:
    return now() - timedelta(hours=hours)

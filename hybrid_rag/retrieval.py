"""Keyword (BM25), vector (Chroma) and hybrid (reciprocal rank fusion) retrieval.

All three return ranked chunk ids behind one interface, so the API and the
evaluation can swap them with a single parameter.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

from .corpus import Chunk

Mode = Literal["bm25", "vector", "hybrid"]
MODES: tuple[Mode, ...] = ("bm25", "vector", "hybrid")

# A small stopword list: BM25's IDF already discounts common words, but very
# frequent function words still add noise to short queries.
STOPWORDS = frozenset(
    "a an and are as at be but by did do does for from had has have he her his how i if in "
    "into is it its me my of on or our she so than that the their them then there these they "
    "this to was we were what when where which who why will with would you your".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


@dataclass
class Hit:
    chunk_id: str
    text: str
    score: float
    # Rank of this chunk in each underlying retriever (1-based), for transparency.
    ranks: dict[str, int] = field(default_factory=dict)


class BM25:
    """Okapi BM25.

    score(q, d) = sum over query terms t of
        idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))

    k1 caps how much repeating a term helps (term-frequency saturation); b
    controls how strongly long chunks are penalized. 1.5 / 0.75 are the
    standard defaults.
    """

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(c.text)) for c in chunks]
        self.lengths = [sum(tf.values()) for tf in self.tfs]
        self.avgdl = sum(self.lengths) / len(self.lengths) if chunks else 0.0
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(chunks)
        # The "+1" form of BM25 idf keeps every weight positive.
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out = []
        for tf, length in zip(self.tfs, self.lengths):
            norm = self.k1 * (1 - self.b + self.b * length / self.avgdl)
            out.append(sum(
                self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm)
                for t in terms if t in tf
            ))
        return out

    def search(self, query: str, k: int) -> list[Hit]:
        scored = sorted(zip(self.scores(query), self.chunks), key=lambda p: -p[0])
        return [Hit(c.id, c.text, s, {"bm25": i}) for i, (s, c) in enumerate(scored[:k], 1) if s > 0]


class VectorIndex:
    """Chroma collection with its default embedding model (all-MiniLM-L6-v2,
    run through ONNX Runtime: no PyTorch, ~80 MB cached on first use)."""

    def __init__(
        self,
        chunks: list[Chunk],
        persist_dir: str | None = None,
        collection: str = "chunks",
        embedding_function=None,  # tests pass a tiny deterministic one; default is MiniLM
    ):
        import chromadb

        client = chromadb.PersistentClient(path=persist_dir) if persist_dir else chromadb.EphemeralClient()
        kwargs = {"embedding_function": embedding_function} if embedding_function else {}
        # Cosine distance to match how the embedding model was trained.
        self.col = client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"}, **kwargs)
        ids = [c.id for c in chunks]
        existing = set(self.col.get(ids=ids, include=[])["ids"]) if ids else set()
        missing = [c for c in chunks if c.id not in existing]
        if missing:  # embed once; a persisted index is reused across runs
            self.col.add(ids=[c.id for c in missing], documents=[c.text for c in missing])

    def search(self, query: str, k: int) -> list[Hit]:
        res = self.col.query(query_texts=[query], n_results=k)
        return [
            # Chroma returns cosine distance; 1 - distance is cosine similarity.
            Hit(cid, doc, 1.0 - dist, {"vector": i})
            for i, (cid, doc, dist) in enumerate(zip(res["ids"][0], res["documents"][0], res["distances"][0]), 1)
        ]


def reciprocal_rank_fusion(rankings: dict[str, list[Hit]], k: int = 60) -> list[Hit]:
    """Merge rankings by position, not score: score(d) = sum 1 / (k + rank).

    BM25 scores are unbounded and cosine similarities sit in [-1, 1], so adding
    raw scores would let one retriever dominate. RRF only uses ranks, needs no
    normalization or weights, and rewards chunks both retrievers agree on. k=60
    is the constant from the original paper (Cormack et al., 2009); it damps the
    gap between rank 1 and rank 2 so one retriever's top hit can't win alone.
    """
    fused: dict[str, Hit] = {}
    for name, hits in rankings.items():
        for rank, h in enumerate(hits, 1):
            hit = fused.setdefault(h.chunk_id, Hit(h.chunk_id, h.text, 0.0))
            hit.score += 1.0 / (k + rank)
            hit.ranks[name] = rank
    return sorted(fused.values(), key=lambda h: -h.score)


class Retriever:
    def __init__(
        self,
        chunks: list[Chunk],
        persist_dir: str | None = None,
        candidates: int = 20,
        embedding_function=None,
        collection: str = "chunks",
    ):
        self.bm25 = BM25(chunks)
        self.vector = VectorIndex(chunks, persist_dir, collection, embedding_function)
        # How deep each retriever looks before fusion; the final top-k comes after.
        self.candidates = candidates

    def search(self, query: str, k: int = 4, mode: Mode = "hybrid") -> list[Hit]:
        if mode == "bm25":
            return self.bm25.search(query, k)
        if mode == "vector":
            return self.vector.search(query, k)
        if mode == "hybrid":
            return self.fused_candidates(query)[:k]
        raise ValueError(f"unknown retrieval mode {mode!r}; expected one of {MODES}")

    def fused_candidates(self, query: str) -> list[Hit]:
        """The full hybrid candidate list, before cutting to k: what a reranker reorders."""
        return reciprocal_rank_fusion({
            "bm25": self.bm25.search(query, self.candidates),
            "vector": self.vector.search(query, self.candidates),
        })[: self.candidates]

    def search_reranked(self, query: str, reranker, k: int = 4) -> list[Hit]:
        """Hybrid candidates (top 20) reordered by a cross-encoder, then cut to k."""
        return reranker.rerank(query, self.fused_candidates(query), k)

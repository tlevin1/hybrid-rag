import pytest

from hybrid_rag.corpus import Chunk, chunk_text
from hybrid_rag.retrieval import BM25, MODES, Hit, reciprocal_rank_fusion, tokenize

from .conftest import CHUNKS

# Chunking


def test_chunks_overlap_and_cover_the_text():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = chunk_text(text, size=300, overlap=50)
    assert chunks[0].text.startswith("Sentence number 0")
    assert chunks[-1].text.endswith("Sentence number 199 is here.")
    assert all(len(c.text) <= 300 for c in chunks)
    # Consecutive chunks share text, so a fact on a boundary appears whole once.
    assert chunks[1].text[:20] in chunks[0].text


def test_chunks_end_on_sentence_boundaries_when_possible():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    assert all(c.text.endswith(".") for c in chunk_text(text, size=300, overlap=50))


def test_chunk_ids_are_stable_and_ordered():
    ids = [c.id for c in chunk_text("x. " * 1000, size=200, overlap=20)]
    assert ids[:3] == ["c000", "c001", "c002"] and ids == sorted(ids)


# BM25


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("What did the IBM 1401 run?") == ["ibm", "1401", "run"]


def test_bm25_ranks_the_chunk_with_the_rare_term_first():
    assert BM25(CHUNKS).search("Accademia painting", k=1)[0].chunk_id == "c001"


def test_bm25_returns_no_hits_for_unknown_words():
    assert BM25(CHUNKS).search("quantum chromodynamics", k=3) == []


def test_bm25_rare_terms_outweigh_common_ones():
    bm = BM25(CHUNKS)
    assert bm.idf["accademia"] > bm.idf["viaweb"]  # 1 chunk vs 2 chunks


def test_bm25_term_frequency_saturates():
    one = BM25([Chunk("a", "lisp"), Chunk("b", "other words here")])
    many = BM25([Chunk("a", "lisp " * 50), Chunk("b", "other words here")])
    # 50 repetitions help, but nowhere near 50x: that is what k1 controls.
    assert many.scores("lisp")[0] < 3 * one.scores("lisp")[0]


# Reciprocal rank fusion


def _hits(*ids):
    return [Hit(i, i, 0.0) for i in ids]


def test_rrf_rewards_agreement_over_one_strong_vote():
    fused = reciprocal_rank_fusion({"bm25": _hits("a", "b", "c"), "vector": _hits("x", "b", "y")})
    assert fused[0].chunk_id == "b"  # rank 2 in both beats rank 1 in one
    assert fused[0].ranks == {"bm25": 2, "vector": 2}


def test_rrf_ignores_raw_score_scales():
    huge = [Hit("a", "a", 1000.0), Hit("b", "b", 999.0)]
    tiny = [Hit("b", "b", 0.02), Hit("a", "a", 0.01)]
    scores = {h.chunk_id: h.score for h in reciprocal_rank_fusion({"bm25": huge, "vector": tiny})}
    assert scores["a"] == pytest.approx(scores["b"])  # symmetric ranks -> tie


# Retriever


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_finds_the_obvious_chunk(retriever, mode):
    assert retriever.search("Yahoo bought Viaweb online stores", k=2, mode=mode)[0].chunk_id == "c002"


def test_hybrid_reports_ranks_from_both_retrievers(retriever):
    top = retriever.search("Yahoo bought Viaweb online stores", k=1, mode="hybrid")[0]
    assert set(top.ranks) == {"bm25", "vector"}


def test_unknown_mode_is_rejected(retriever):
    with pytest.raises(ValueError, match="unknown retrieval mode"):
        retriever.search("x", mode="sparse")

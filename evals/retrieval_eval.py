"""Exact retrieval metrics: did each retriever find the chunk a question was
written from, and how high? No LLM involved, so it's free and deterministic.

- hit@1 / hit@4: the source chunk is ranked first / in the top 4 (what the LLM sees)
- MRR@20: mean of 1 / rank of the source chunk (0 if not in the top 20)

This undercounts slightly: another chunk can also answer a question (the
LLM-judged relevance in phoenix_eval.py catches that).

    python -m evals.retrieval_eval              # writes reports/retrieval.json
    python -m evals.retrieval_eval --rerank     # also hybrid + each cross-encoder (needs requirements-train.txt)
    python -m evals.retrieval_eval --azure      # also Azure AI Search keyword / vector / hybrid (needs .env)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hybrid_rag.corpus import load_chunks
from hybrid_rag.retrieval import MODES, Retriever

QUESTIONS = Path(__file__).with_name("questions.jsonl")
REPORT = Path(__file__).resolve().parent.parent / "reports" / "retrieval.json"
DEPTH = 20
FINETUNED = Path(__file__).resolve().parent.parent / "models" / "reranker-ft"


def load_questions() -> list[dict]:
    return [json.loads(line) for line in QUESTIONS.read_text().splitlines()]


def source_rank(hits, source: str) -> int | None:
    return next((i for i, h in enumerate(hits, 1) if h.chunk_id == source), None)


def score(ranks: list[int | None]) -> dict:
    n = len(ranks)
    return {
        "n": n,
        "hit@1": round(sum(r == 1 for r in ranks) / n, 3),
        "hit@4": round(sum(r is not None and r <= 4 for r in ranks) / n, 3),
        "mrr@20": round(sum(1 / r for r in ranks if r) / n, 3),
    }


def rerankers() -> dict:
    """Cross-encoders to compare, keyed by mode name. Loaded lazily so the base
    eval doesn't need PyTorch."""
    from hybrid_rag.rerank import PRETRAINED, CrossEncoder

    found = {"hybrid+ms-marco": CrossEncoder(PRETRAINED)}
    if FINETUNED.exists():
        found["hybrid+finetuned"] = CrossEncoder(str(FINETUNED))
    return found


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rerank", action="store_true", help="also evaluate cross-encoder reranking")
    p.add_argument("--azure", action="store_true", help="also evaluate Azure AI Search")
    a = p.parse_args()
    retriever = Retriever(load_chunks(), persist_dir="data/chroma")
    questions = load_questions()
    searchers = {mode: (lambda q, m=mode: retriever.search(q, DEPTH, m)) for mode in MODES}
    if a.rerank:
        for name, ce in rerankers().items():
            searchers[name] = lambda q, ce=ce: retriever.search_reranked(q, ce, DEPTH)
    if a.azure:
        from hybrid_rag.azure_search import AzureSearchIndex

        az = AzureSearchIndex(load_chunks())  # English analyzer
        # Same chunks under Azure's default analyzer, to show what configuration costs.
        az_std = AzureSearchIndex(load_chunks(), index_name="hybrid-rag-chunks", analyzer=None)
        searchers["azure-keyword-std"] = lambda q: az_std.search(q, DEPTH, "keyword")
        for mode in ("keyword", "vector", "hybrid"):
            searchers[f"azure-{mode}"] = lambda q, m=mode: az.search(q, DEPTH, m)
    report: dict = {}
    for style in ("direct", "paraphrased", "all"):
        qs = [q for q in questions if style == "all" or q["style"] == style]
        report[style] = {
            name: score([source_rank(search(q["question"]), q["source_chunk"]) for q in qs])
            for name, search in searchers.items()
        }
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))

    print(f"{'questions':17} {'retriever':17} {'hit@1':>6} {'hit@4':>6} {'MRR@20':>7}")
    for style, by_mode in report.items():
        for mode, m in by_mode.items():
            print(f"{style + ' (' + str(m['n']) + ')':17} {mode:17} {m['hit@1']:>6} {m['hit@4']:>6} {m['mrr@20']:>7}")


if __name__ == "__main__":
    main()

"""Exact retrieval metrics: did each retriever find the chunk a question was
written from, and how high? No LLM involved, so it's free and deterministic.

- hit@1 / hit@4: the source chunk is ranked first / in the top 4 (what the LLM sees)
- MRR@20: mean of 1 / rank of the source chunk (0 if not in the top 20)

This undercounts slightly: another chunk can also answer a question (the
LLM-judged relevance in phoenix_eval.py catches that).

    python -m evals.retrieval_eval        # writes reports/retrieval.json
"""

from __future__ import annotations

import json
from pathlib import Path

from hybrid_rag.corpus import load_chunks
from hybrid_rag.retrieval import MODES, Retriever

QUESTIONS = Path(__file__).with_name("questions.jsonl")
REPORT = Path(__file__).resolve().parent.parent / "reports" / "retrieval.json"
DEPTH = 20


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


def main() -> None:
    retriever = Retriever(load_chunks(), persist_dir="data/chroma")
    questions = load_questions()
    report: dict = {}
    for style in ("direct", "paraphrased", "all"):
        qs = [q for q in questions if style == "all" or q["style"] == style]
        report[style] = {
            mode: score([source_rank(retriever.search(q["question"], DEPTH, mode), q["source_chunk"]) for q in qs])
            for mode in MODES
        }
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))

    print(f"{'questions':12} {'retriever':8} {'hit@1':>6} {'hit@4':>6} {'MRR@20':>7}")
    for style, by_mode in report.items():
        for mode, m in by_mode.items():
            print(f"{style + ' (' + str(m['n']) + ')':12} {mode:8} {m['hit@1']:>6} {m['hit@4']:>6} {m['mrr@20']:>7}")


if __name__ == "__main__":
    main()

"""Answer quality per retriever, judged with Arize Phoenix evaluators.

Adapted from the Phoenix "Evaluate RAG" cookbook pipeline (rag-chatbot's
eval.py), run on all three retrievers over the same saved questions:

1. Answer every question with each retriever (bm25, vector, hybrid).
2. Retrieval: Phoenix's DocumentRelevanceEvaluator judges each retrieved
   passage -> precision@4, NDCG@4 and hit rate per question.
3. Answers: a reference-based correctness judge (built with Phoenix's
   create_classifier) and Phoenix's FaithfulnessEvaluator (hallucination).
   Refusals count as not correct and are excluded from faithfulness.

Why reference-based: Phoenix's built-in CorrectnessEvaluator has no answer key
and judges from its own knowledge. In a smoke test it marked one of three
near-identical, verbatim-quoted answers "incorrect": noise larger than the
differences between retrievers. Each question has a saved reference answer.

Scores are computed from the evaluator outputs directly, not read back out of
Phoenix traces, so the numbers don't depend on a running Phoenix server.

    python -m evals.phoenix_eval                   # answer + judge (~$1-3 on Haiku)
    python -m evals.phoenix_eval --reuse-answers   # re-judge saved answers only
"""

from __future__ import annotations

import argparse
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
from phoenix.evals import LLM, async_evaluate_dataframe, create_classifier
from phoenix.evals.metrics import DocumentRelevanceEvaluator, FaithfulnessEvaluator

from hybrid_rag.corpus import load_chunks
from hybrid_rag.generate import NO_ANSWER, answer
from hybrid_rag.retrieval import MODES, Hit, Retriever

from .metrics import ndcg_at_k, phoenix_scores, summarize
from .retrieval_eval import load_questions

REPORTS = Path(__file__).resolve().parent.parent / "reports"
ANSWERS = REPORTS / "answers.jsonl"
REPORT = REPORTS / "answer_quality.json"
TOP_K = 4
JUDGE_MODEL = "claude-haiku-4-5"

REFERENCE_CORRECTNESS_TEMPLATE = """You are grading an answer to a question about an essay against a reference answer.

Question: {input}
Reference answer: {reference}
Answer to grade: {output}

The answer is "correct" if it contains the key fact(s) of the reference answer and
does not contradict it. Extra correct detail, different wording and citations like
[c012] are fine. It is "incorrect" if it misses or contradicts the key fact, or
declines to answer."""


# 1. Answer


def answer_all(questions: list[dict]) -> list[dict]:
    retriever = Retriever(load_chunks(), persist_dir="data/chroma")
    client = anthropic.Anthropic()
    # Retrieve serially (fast, and it keeps Chroma single-threaded), then
    # parallelize only the LLM calls.
    rows = []
    for q in questions:
        for mode in MODES:
            hits = retriever.search(q["question"], k=TOP_K, mode=mode)
            rows.append({
                "qid": q["id"], "style": q["style"], "mode": mode, "question": q["question"],
                "reference": q["reference"], "source_chunk": q["source_chunk"],
                "retrieved": [{"chunk_id": h.chunk_id, "text": h.text} for h in hits],
            })

    def run(row):
        hits = [Hit(d["chunk_id"], d["text"], 0.0) for d in row["retrieved"]]
        return {**row, "answer": answer(client, row["question"], hits)}

    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(run, rows))


# 2-3. Judge


def judge(rows: list[dict]) -> pd.DataFrame:
    llm = LLM(provider="anthropic", model=JUDGE_MODEL)
    reference_correctness = create_classifier(
        name="reference_correctness", prompt_template=REFERENCE_CORRECTNESS_TEMPLATE,
        llm=llm, choices={"correct": 1, "incorrect": 0},
    )

    docs = pd.DataFrame([
        {"row": i, "rank": r, "input": row["question"], "document_text": d["text"]}
        for i, row in enumerate(rows) for r, d in enumerate(row["retrieved"], 1)
    ])
    answers = pd.DataFrame([
        {
            "input": r["question"], "output": r["answer"], "reference": r["reference"],
            "context": "\n\n".join(d["text"] for d in r["retrieved"]),
        }
        for r in rows
    ])

    async def run_all():
        # One event loop for every judge: the LLM's async client is bound to the
        # loop it first ran on, and a fresh asyncio.run() per evaluator produced
        # connection errors. Evaluators still run one at a time, since running
        # several in one call deadlocked in the original script.
        kw = {"concurrency": 8, "max_retries": 5}
        rel = await async_evaluate_dataframe(dataframe=docs, evaluators=[DocumentRelevanceEvaluator(llm=llm)], **kw)
        corr = await async_evaluate_dataframe(dataframe=answers, evaluators=[reference_correctness], **kw)
        faith = await async_evaluate_dataframe(dataframe=answers, evaluators=[FaithfulnessEvaluator(llm=llm)], **kw)
        return rel, corr, faith

    rel, corr, faith = asyncio.run(run_all())
    docs["relevant"] = phoenix_scores(rel, "document_relevance_score")

    out = pd.DataFrame([{k: r[k] for k in ("qid", "style", "mode")} for r in rows])
    refused = np.array([r["answer"] == NO_ANSWER for r in rows])
    out["refused"] = refused
    # A refusal gives the user nothing: not correct. It also has no claims to be
    # faithful or unfaithful to, so it's excluded from the faithfulness rate.
    out["correct"] = np.where(refused, 0.0, phoenix_scores(corr, "reference_correctness_score").to_numpy())
    out["faithful"] = np.where(refused, np.nan, phoenix_scores(faith, "faithfulness_score").to_numpy())
    by_row = docs.sort_values("rank").groupby("row")["relevant"].apply(list)
    out["precision@4"] = [np.nanmean(by_row.get(i, [np.nan])) for i in range(len(rows))]
    out["ndcg@4"] = [ndcg_at_k(by_row.get(i, []), TOP_K) for i in range(len(rows))]
    out["hit"] = [float(np.nansum(by_row.get(i, [0])) > 0) for i in range(len(rows))]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reuse-answers", action="store_true", help="re-judge reports/answers.jsonl")
    p.add_argument("--limit", type=int, help="only the first N questions (a cheap smoke test)")
    a = p.parse_args()
    REPORTS.mkdir(exist_ok=True)

    if a.reuse_answers:
        rows = [json.loads(line) for line in ANSWERS.read_text().splitlines()]
    else:
        rows = answer_all(load_questions()[: a.limit])
        ANSWERS.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"Answered {len(rows)} question x retriever pairs -> {ANSWERS}")

    scored = judge(rows)
    scored.to_csv(REPORTS / "answer_scores.csv", index=False)
    report = summarize(scored)
    REPORT.write_text(json.dumps(report, indent=2))

    print(f"\n{'questions':18} {'retriever':8} {'prec@4':>7} {'ndcg@4':>7} {'hit':>5} {'correct':>8} {'faithful':>9} {'refused':>8}")
    for style, by_mode in report.items():
        for mode in MODES:
            m = by_mode[mode]
            print(f"{style + ' (' + str(m['n']) + ')':18} {mode:8} {m['precision@4']:>7} {m['ndcg@4']:>7} "
                  f"{m['hit']:>5} {m['correct']:>8} {m['faithful']:>9} {m['refused']:>8}")


if __name__ == "__main__":
    main()

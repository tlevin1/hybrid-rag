"""Training data for the reranker.

Leakage rules:
- Training questions come only from chunks that no eval question was written
  from, so the model never sees an eval question or its answer passage as a
  positive.
- Train and validation are split by chunk, not by question: both questions
  about a chunk land on the same side, so validation measures generalization
  to unseen passages.

Each training example is a group: one question, its source chunk (positive),
and hard negatives: chunks hybrid search ranks highly that are not the source.
Chunks adjacent to the source are never negatives: chunks overlap by 100
characters, so a neighbor may contain the answer, and calling it irrelevant
would teach the model something false.

    python -m training.data      # writes training/questions.jsonl (~$0.10 on Haiku)
"""

from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from evals.retrieval_eval import load_questions as load_eval_questions
from hybrid_rag.corpus import Chunk

QUESTIONS = Path(__file__).with_name("questions.jsonl")
N_NEGATIVES = 7
VAL_CHUNKS = 15
SEED = 0


@dataclass
class Group:
    question: str
    positive: str
    negatives: list[str]
    source_chunk: str


def chunk_index(chunk_id: str) -> int:
    return int(chunk_id[1:])


def forbidden_negatives(source_chunk: str) -> set[str]:
    """The source and its overlapping neighbors."""
    i = chunk_index(source_chunk)
    return {f"c{j:03d}" for j in (i - 1, i, i + 1)}


def training_chunks(chunks: list[Chunk]) -> list[Chunk]:
    eval_sources = {q["source_chunk"] for q in load_eval_questions()}
    return [c for c in chunks if c.id not in eval_sources]


def split_by_chunk(questions: list[dict], val_chunks: int = VAL_CHUNKS, seed: int = SEED):
    ids = sorted({q["source_chunk"] for q in questions})
    val_ids = set(random.Random(seed).sample(ids, val_chunks))
    train = [q for q in questions if q["source_chunk"] not in val_ids]
    val = [q for q in questions if q["source_chunk"] in val_ids]
    return train, val


def build_group(question: dict, candidates: list, by_id: dict[str, Chunk], n_neg: int = N_NEGATIVES,
                rng: random.Random | None = None) -> Group:
    """`candidates`: hybrid retrieval hits for the question, best first."""
    rng = rng or random.Random(SEED)
    banned = forbidden_negatives(question["source_chunk"])
    hard = [h.chunk_id for h in candidates if h.chunk_id not in banned][:n_neg]
    if len(hard) < n_neg:  # top up with random chunks if retrieval returned too few
        pool = [cid for cid in by_id if cid not in banned and cid not in hard]
        hard += rng.sample(pool, n_neg - len(hard))
    return Group(
        question=question["question"],
        positive=by_id[question["source_chunk"]].text,
        negatives=[by_id[cid].text for cid in hard],
        source_chunk=question["source_chunk"],
    )


def build_groups(questions: list[dict], retriever, chunks: list[Chunk], n_neg: int = N_NEGATIVES) -> list[Group]:
    by_id = {c.id: c for c in chunks}
    rng = random.Random(SEED)
    return [build_group(q, retriever.fused_candidates(q["question"]), by_id, n_neg, rng) for q in questions]


def load_questions() -> list[dict]:
    return [json.loads(line) for line in QUESTIONS.read_text().splitlines()]


def main() -> None:
    import anthropic

    from evals.generate_questions import generate
    from hybrid_rag.corpus import load_chunks

    pool = training_chunks(load_chunks())
    client = anthropic.Anthropic()
    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = [r for rs in ex.map(lambda c: generate(client, c), pool) for r in rs]
    with QUESTIONS.open("w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"id": f"t{i:03d}", **r}) + "\n")
    print(f"Wrote {len(rows)} training questions from {len(pool)} chunks (none used by the eval) to {QUESTIONS}")


if __name__ == "__main__":
    main()

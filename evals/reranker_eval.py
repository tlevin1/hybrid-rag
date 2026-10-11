"""Is the fine-tuned reranker actually better than the pretrained one?

For every eval question, compare where the source passage lands after
reranking with the pretrained model vs each fine-tuned seed:

- per-seed hit@1 / hit@4 / MRR, so a gain that only one seed shows is visible
- paired wins/losses: questions where fine-tuning moved the source up vs down
- a two-sided sign test on those wins/losses (ties dropped)
- a paired bootstrap 95% CI on the MRR difference

    python -m evals.reranker_eval      # needs models from python -m training.train
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from hybrid_rag.corpus import load_chunks
from hybrid_rag.rerank import PRETRAINED, CrossEncoder
from hybrid_rag.retrieval import Retriever

from .retrieval_eval import DEPTH, load_questions, source_rank

ROOT = Path(__file__).resolve().parent.parent
MODELS = {"seed0": ROOT / "models" / "reranker-ft", "seed1": ROOT / "models" / "reranker-ft-seed1",
          "seed2": ROOT / "models" / "reranker-ft-seed2"}
REPORT = ROOT / "reports" / "reranker.json"


def metrics(ranks: list[int | None]) -> dict:
    n = len(ranks)
    return {
        "hit@1": round(sum(r == 1 for r in ranks) / n, 3),
        "hit@4": round(sum(r is not None and r <= 4 for r in ranks) / n, 3),
        "mrr": round(sum(1 / r for r in ranks if r) / n, 3),
    }


def rr(rank: int | None) -> float:
    return 1 / rank if rank else 0.0


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact binomial test that wins and losses are equally likely."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired_bootstrap_ci(a: list[float], b: list[float], n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """95% CI of mean(b - a), resampling questions with replacement."""
    rng = random.Random(seed)
    diffs = [y - x for x, y in zip(a, b)]
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n))
    return round(means[int(0.025 * n)], 3), round(means[int(0.975 * n)], 3)


def compare(base_ranks: list, ft_ranks: list) -> dict:
    wins = sum(rr(f) > rr(b) for b, f in zip(base_ranks, ft_ranks))
    losses = sum(rr(f) < rr(b) for b, f in zip(base_ranks, ft_ranks))
    return {
        "wins": wins, "losses": losses, "ties": len(base_ranks) - wins - losses,
        "sign_test_p": round(sign_test(wins, losses), 4),
        "mrr_diff_95ci": paired_bootstrap_ci([rr(r) for r in base_ranks], [rr(r) for r in ft_ranks]),
    }


def main() -> None:
    retriever = Retriever(load_chunks(), persist_dir=str(ROOT / "data" / "chroma"))
    questions = load_questions()
    models = {"pretrained": CrossEncoder(PRETRAINED)}
    models.update({name: CrossEncoder(str(path)) for name, path in MODELS.items() if path.exists()})

    ranks = {
        name: [source_rank(retriever.search_reranked(q["question"], ce, DEPTH), q["source_chunk"]) for q in questions]
        for name, ce in models.items()
    }
    report: dict = {"models": {}, "vs_pretrained": {}}
    for style in ("direct", "paraphrased", "all"):
        idx = [i for i, q in enumerate(questions) if style == "all" or q["style"] == style]
        report["models"][style] = {name: metrics([r[i] for i in idx]) for name, r in ranks.items()}
        report["vs_pretrained"][style] = {
            name: compare([ranks["pretrained"][i] for i in idx], [r[i] for i in idx])
            for name, r in ranks.items() if name != "pretrained"
        }
    REPORT.write_text(json.dumps(report, indent=2))

    for style in ("direct", "paraphrased", "all"):
        print(f"\n{style}")
        for name, m in report["models"][style].items():
            line = f"  {name:10} hit@1 {m['hit@1']:.3f}  hit@4 {m['hit@4']:.3f}  MRR {m['mrr']:.3f}"
            if name != "pretrained":
                c = report["vs_pretrained"][style][name]
                line += f"   vs pretrained: {c['wins']} up / {c['losses']} down / {c['ties']} same, " \
                        f"sign test p={c['sign_test_p']}, MRR diff 95% CI {c['mrr_diff_95ci']}"
            print(line)


if __name__ == "__main__":
    main()

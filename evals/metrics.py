"""Scoring helpers for the answer-quality eval. Pure numpy/pandas/sklearn, so
they're unit-tested in CI without installing Phoenix."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score


def phoenix_scores(df: pd.DataFrame, column: str) -> pd.Series:
    """Phoenix puts each result in a dict ({score, label, explanation, ...})."""
    return df[column].map(lambda r: r.get("score") if isinstance(r, dict) else np.nan).astype(float)


def ndcg_at_k(relevance: list[float], k: int) -> float:
    """NDCG of judged relevance (in retrieval order) against the ideal order.
    1.0 means every relevant passage was ranked above every irrelevant one."""
    rel = np.nan_to_num(np.array(relevance, dtype=float))
    if len(rel) < 2:
        rel = np.pad(rel, (0, 2 - len(rel)))
    order = np.arange(len(rel), 0, -1)  # retrieval order as the predicted ranking
    return float(ndcg_score([rel], [order], k=k)) if rel.any() else 0.0


def summarize(scored: pd.DataFrame) -> dict:
    cols = ["precision@4", "ndcg@4", "hit", "correct", "faithful", "refused"]
    report = {}
    for style in ("direct", "paraphrased", "all"):
        sub = scored if style == "all" else scored[scored["style"] == style]
        report[style] = {
            # mean() skips NaN, so faithfulness is over answered questions only.
            mode: {"n": int(len(g)), **{c: round(float(g[c].astype(float).mean()), 3) for c in cols}}
            for mode, g in sub.groupby("mode")
        }
    return report

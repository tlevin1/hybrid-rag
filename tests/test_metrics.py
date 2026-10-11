import math

import numpy as np
import pandas as pd
import pytest

from evals.metrics import ndcg_at_k, phoenix_scores, summarize


def test_ndcg_is_one_when_relevant_passages_come_first():
    assert ndcg_at_k([1, 1, 0, 0], k=4) == pytest.approx(1.0)


def test_ndcg_penalizes_relevant_passages_ranked_low():
    assert ndcg_at_k([0, 0, 0, 1], k=4) < ndcg_at_k([0, 1, 0, 0], k=4) < 1.0


def test_ndcg_is_zero_when_nothing_is_relevant_and_handles_one_passage():
    assert ndcg_at_k([0, 0, 0, 0], k=4) == 0.0
    assert ndcg_at_k([1], k=4) == pytest.approx(1.0)


def test_phoenix_scores_unwraps_result_dicts_and_failed_rows():
    df = pd.DataFrame({"x_score": [{"score": 1.0, "label": "relevant"}, {"score": 0.0}, None]})
    out = phoenix_scores(df, "x_score")
    assert out.iloc[0] == 1.0 and out.iloc[1] == 0.0 and math.isnan(out.iloc[2])


def test_summary_excludes_refusals_from_faithfulness_but_not_correctness():
    scored = pd.DataFrame([
        # answered, correct and faithful
        {"style": "direct", "mode": "hybrid", "refused": False, "correct": 1.0, "faithful": 1.0,
         "precision@4": 0.5, "ndcg@4": 1.0, "hit": 1.0},
        # refused: counts as not correct, has no faithfulness score
        {"style": "direct", "mode": "hybrid", "refused": True, "correct": 0.0, "faithful": np.nan,
         "precision@4": 0.0, "ndcg@4": 0.0, "hit": 0.0},
    ])
    m = summarize(scored)["direct"]["hybrid"]
    assert m["n"] == 2 and m["correct"] == 0.5 and m["refused"] == 0.5
    assert m["faithful"] == 1.0  # the refusal neither helps nor hurts

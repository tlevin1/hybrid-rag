"""Reranker training and inference, on a tiny randomly initialized BERT built in
the test: no model download, runs on CPU in seconds."""

import random

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from evals.reranker_eval import paired_bootstrap_ci, sign_test  # noqa: E402
from hybrid_rag.corpus import Chunk  # noqa: E402
from hybrid_rag.rerank import CrossEncoder  # noqa: E402
from hybrid_rag.retrieval import Hit  # noqa: E402
from training.data import build_group, forbidden_negatives, split_by_chunk  # noqa: E402
from training.train import GroupDataset, listwise_loss, make_collate, train_step  # noqa: E402

WORDS = "what who when the author painting lisp viaweb yahoo florence startup essay wrote bought studied".split()


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny-ce")
    (d / "vocab.txt").write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]))
    tokenizer = transformers.BertTokenizer(str(d / "vocab.txt"))
    config = transformers.BertConfig(
        vocab_size=len(WORDS) + 5, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=64, max_position_embeddings=128, num_labels=1,
        # No dropout: on a 32-dim model its noise swamps 30 steps of signal, and
        # the test should be deterministic.
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
    )
    torch.manual_seed(0)
    transformers.BertForSequenceClassification(config).save_pretrained(d)
    tokenizer.save_pretrained(d)
    return d


# Loss


def test_listwise_loss_is_cross_entropy_with_positive_first():
    logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    expected = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 0]))
    assert torch.isclose(listwise_loss(logits), expected)


def test_listwise_loss_is_lower_when_positive_ranks_first():
    good, bad = torch.tensor([[5.0, 0.0, 0.0]]), torch.tensor([[0.0, 5.0, 0.0]])
    assert listwise_loss(good) < listwise_loss(bad)


# Batching and training


def test_collate_flattens_groups_into_question_passage_pairs(tiny_model_dir):
    tok = transformers.AutoTokenizer.from_pretrained(tiny_model_dir)
    batch = [("who bought viaweb", ["yahoo bought viaweb", "lisp", "florence"])] * 2
    enc, group_size = make_collate(tok)(batch)
    assert group_size == 3 and enc["input_ids"].shape[0] == 6


def test_training_steps_reduce_the_loss(tiny_model_dir):
    torch.manual_seed(0)
    tok = transformers.AutoTokenizer.from_pretrained(tiny_model_dir)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(tiny_model_dir)
    groups = [("who bought viaweb", ["yahoo bought viaweb", "painting florence", "lisp essay"]),
              ("where studied painting", ["studied painting florence", "yahoo bought", "startup essay"])]
    enc, g = make_collate(tok)(groups)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = [train_step(model, enc, g, opt) for _ in range(30)]
    assert losses[-1] < losses[0] * 0.5


# Inference


def test_rerank_orders_by_model_score_and_records_the_new_rank(tiny_model_dir):
    ce = CrossEncoder(str(tiny_model_dir), device=torch.device("cpu"))
    hits = [Hit(f"c{i:03d}", text, 0.0, {"bm25": i + 1}) for i, text in
            enumerate(["yahoo bought viaweb", "painting florence", "lisp essay", "startup"])]
    out = ce.rerank("who bought viaweb", hits, k=3)
    scores = ce.score("who bought viaweb", [h.text for h in hits])
    assert [h.chunk_id for h in out] == [hits[i].chunk_id for i in sorted(range(4), key=lambda i: -scores[i])][:3]
    assert [h.ranks["rerank"] for h in out] == [1, 2, 3]
    assert all("bm25" in h.ranks for h in out)  # earlier ranks are kept


# Training data


def test_neighbors_of_the_source_are_never_negatives():
    assert forbidden_negatives("c010") == {"c009", "c010", "c011"}


def test_hard_negatives_come_from_retrieval_and_skip_the_source_and_neighbors():
    by_id = {f"c{i:03d}": Chunk(f"c{i:03d}", f"text {i}") for i in range(30)}
    candidates = [Hit(cid, "", 0.0) for cid in ["c010", "c011", "c020", "c009", "c021", "c022"]]
    g = build_group({"question": "q", "source_chunk": "c010"}, candidates, by_id, n_neg=5, rng=random.Random(0))
    assert g.positive == "text 10"
    assert g.negatives[:3] == ["text 20", "text 21", "text 22"]  # retrieval order kept
    assert len(g.negatives) == 5  # topped up with random chunks
    assert not {"text 9", "text 10", "text 11"} & set(g.negatives)


def test_train_and_validation_never_share_a_chunk():
    qs = [{"source_chunk": f"c{i % 40:03d}", "style": s} for i in range(40) for s in ("direct", "paraphrased")]
    train, val = split_by_chunk(qs, val_chunks=8)
    assert {q["source_chunk"] for q in train}.isdisjoint({q["source_chunk"] for q in val})
    assert len({q["source_chunk"] for q in val}) == 8


def test_group_dataset_puts_the_positive_first():
    from training.data import Group

    ds = GroupDataset([Group("q", "pos", ["n1", "n2"], "c000")])
    assert ds[0] == ("q", ["pos", "n1", "n2"])


# Significance helpers


def test_sign_test_matches_exact_binomial():
    assert sign_test(5, 5) == 1.0
    assert sign_test(10, 0) == pytest.approx(2 / 1024)


def test_bootstrap_ci_brackets_a_clear_improvement():
    lo, hi = paired_bootstrap_ci([0.0] * 50, [1.0] * 40 + [0.0] * 10)
    assert 0 < lo <= 0.8 <= hi

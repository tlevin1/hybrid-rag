"""Fine-tune the cross-encoder reranker on this corpus, in PyTorch.

Starts from cross-encoder/ms-marco-MiniLM-L-6-v2 and trains with a listwise
loss: for each question, softmax over [positive, 7 hard negatives] and
cross-entropy with the positive as the target. That optimizes the ranking
directly, rather than scoring each pair in isolation.

Validation runs the real pipeline on held-out chunks (rerank hybrid's top 20,
is the source passage first?) and the best epoch is kept. Epoch 0 is the
untrained model, so if fine-tuning never beats it, the saved model is the
pretrained one and the report says so.

    python -m training.train      # ~2-4 min on Apple MPS; writes models/reranker-ft and reports/training.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from hybrid_rag.corpus import load_chunks
from hybrid_rag.rerank import MAX_LENGTH, PRETRAINED, best_device
from hybrid_rag.retrieval import Retriever

from .data import Group, build_groups, load_questions, split_by_chunk

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "models" / "reranker-ft"
REPORT = ROOT / "reports" / "training.json"

CONFIG = {
    "base_model": PRETRAINED,
    "epochs": 4,
    "batch_groups": 8,  # 8 questions x (1 positive + 7 negatives) = 64 pairs per step
    "lr": 2e-5,
    "weight_decay": 0.01,
    "warmup_fraction": 0.1,
    "max_grad_norm": 1.0,
    "seed": 0,
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class GroupDataset(Dataset):
    def __init__(self, groups: list[Group]):
        self.groups = groups

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, i: int) -> tuple[str, list[str]]:
        g = self.groups[i]
        return g.question, [g.positive, *g.negatives]  # positive is always index 0


def make_collate(tokenizer):
    def collate(batch: list[tuple[str, list[str]]]):
        group_size = len(batch[0][1])
        questions = [q for q, docs in batch for _ in docs]
        docs = [d for _, ds in batch for d in ds]
        enc = tokenizer(questions, docs, padding=True, truncation="only_second",
                        max_length=MAX_LENGTH, return_tensors="pt")
        return enc, group_size

    return collate


def listwise_loss(logits: torch.Tensor) -> torch.Tensor:
    """logits: (batch, group_size) with the positive at column 0."""
    target = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return torch.nn.functional.cross_entropy(logits, target)


def train_step(model, enc, group_size: int, optimizer, scheduler=None, max_grad_norm: float = 1.0) -> float:
    model.train()
    logits = model(**enc).logits.view(-1, group_size)
    loss = listwise_loss(logits)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return loss.item()


@torch.no_grad()
def validate(model, tokenizer, retriever, val_questions: list[dict], device) -> dict:
    """The real use: rerank hybrid's 20 candidates; where does the source land?"""
    model.eval()
    ranks = []
    for q in val_questions:
        cands = retriever.fused_candidates(q["question"])
        enc = tokenizer([q["question"]] * len(cands), [h.text for h in cands], padding=True,
                        truncation="only_second", max_length=MAX_LENGTH, return_tensors="pt").to(device)
        scores = model(**enc).logits.squeeze(-1).float().cpu().numpy()
        order = [cands[i].chunk_id for i in np.argsort(-scores)]
        ranks.append(order.index(q["source_chunk"]) + 1 if q["source_chunk"] in order else None)
    n = len(ranks)
    return {
        "hit@1": round(sum(r == 1 for r in ranks) / n, 3),
        "hit@4": round(sum(r is not None and r <= 4 for r in ranks) / n, 3),
        "mrr": round(sum(1 / r for r in ranks if r) / n, 4),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=CONFIG["seed"])
    p.add_argument("--out", type=Path, default=OUT_DIR)
    p.add_argument("--report", type=Path, default=REPORT)
    a = p.parse_args()
    cfg = {**CONFIG, "seed": a.seed}
    out_dir, report = a.out, a.report
    seed_everything(cfg["seed"])
    device = best_device()
    chunks = load_chunks()
    retriever = Retriever(chunks, persist_dir=str(ROOT / "data" / "chroma"))
    train_q, val_q = split_by_chunk(load_questions())
    train_groups = build_groups(train_q, retriever, chunks)

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = AutoModelForSequenceClassification.from_pretrained(cfg["base_model"]).to(device)
    loader = DataLoader(GroupDataset(train_groups), batch_size=cfg["batch_groups"], shuffle=True,
                        collate_fn=make_collate(tokenizer), generator=torch.Generator().manual_seed(cfg["seed"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total = cfg["epochs"] * len(loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg["warmup_fraction"] * total), total)

    history = [{"epoch": 0, "train_loss": None, "val": validate(model, tokenizer, retriever, val_q, device)}]
    print(f"epoch 0 (pretrained): val {history[0]['val']}")
    best = (history[0]["val"]["mrr"], 0)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    start = time.perf_counter()
    for epoch in range(1, cfg["epochs"] + 1):
        losses = []
        for enc, group_size in loader:
            losses.append(train_step(model, enc.to(device), group_size, optimizer, scheduler, cfg["max_grad_norm"]))
        val = validate(model, tokenizer, retriever, val_q, device)
        history.append({"epoch": epoch, "train_loss": round(float(np.mean(losses)), 4), "val": val})
        print(f"epoch {epoch}: train loss {history[-1]['train_loss']}  val {val}", flush=True)
        if val["mrr"] > best[0]:  # strictly better, or the pretrained model stays
            best = (val["mrr"], epoch)
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)

    report.parent.mkdir(exist_ok=True)
    report.write_text(json.dumps({
        "config": cfg, "device": str(device),
        "train_questions": len(train_q), "val_questions": len(val_q),
        "train_minutes": round((time.perf_counter() - start) / 60, 2),
        "best_epoch": best[1], "history": history,
    }, indent=2))
    print(f"best epoch {best[1]} (val MRR {best[0]}) saved to {out_dir}")


if __name__ == "__main__":
    main()

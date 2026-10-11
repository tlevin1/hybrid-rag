"""Cross-encoder reranking.

Retrievers embed the question and each passage separately, so they can't
compare them word by word. A cross-encoder reads the question and one passage
together through a transformer and outputs a single relevance score: far more
accurate, but too slow to run over the whole corpus. So hybrid search narrows
123 chunks to 20 candidates, and the cross-encoder reorders those 20.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .retrieval import Hit

PRETRAINED = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # MS MARCO web-search relevance, 22.7M params
MAX_LENGTH = 256  # question + ~800-character chunk fits comfortably


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class CrossEncoder:
    def __init__(self, model_name_or_path: str = PRETRAINED, device: torch.device | None = None):
        self.device = device or best_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path).to(self.device).eval()

    @torch.no_grad()
    def score(self, query: str, texts: list[str], batch_size: int = 32) -> list[float]:
        scores: list[float] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                [query] * len(batch), batch, padding=True, truncation="only_second",
                max_length=MAX_LENGTH, return_tensors="pt",
            ).to(self.device)
            scores.extend(self.model(**enc).logits.squeeze(-1).float().cpu().tolist())
        return scores

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        scored = sorted(zip(self.score(query, [h.text for h in hits]), hits), key=lambda p: -p[0])
        out = []
        for new_rank, (s, h) in enumerate(scored[:k], 1):
            out.append(Hit(h.chunk_id, h.text, s, {**h.ranks, "rerank": new_rank}))
        return out

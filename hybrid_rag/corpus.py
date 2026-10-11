"""Corpus download and chunking.

The evaluation corpus is Paul Graham's "What I Worked On", the essay used in the
LlamaIndex and Arize Phoenix RAG cookbooks. It is downloaded rather than
committed (it's his copyright), and pinned by checksum: saved eval questions
refer to chunk ids, so a silently changed file would invalidate them.
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ESSAY_URL = (
    "https://raw.githubusercontent.com/run-llama/llama_index/main/"
    "docs/examples/data/paul_graham/paul_graham_essay.txt"
)
ESSAY_SHA256 = "0adaa4726b030c37ca4aaba899094d3806bc6fffe436643deb0e1b13c03f1d80"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ESSAY_PATH = DATA_DIR / "paul_graham_essay.txt"

CHUNK_SIZE = 800  # characters
CHUNK_OVERLAP = 100


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str


def fetch_essay(path: Path = ESSAY_PATH) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(ESSAY_URL, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != ESSAY_SHA256:
        raise RuntimeError(
            f"{path} has sha256 {digest[:12]}…, expected {ESSAY_SHA256[:12]}…. "
            "The saved eval questions refer to chunks of the pinned version."
        )
    return path


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    """Fixed-size windows that end on a sentence boundary when one falls in the
    second half of the window, overlapping so a fact split across a boundary
    still appears whole in one chunk. Ids are positional and deterministic."""
    pieces = []
    start = 0
    while start < len(text):
        window = text[start : start + size]
        if start + size < len(text):
            cut = window.rfind(". ")
            if cut > size // 2:
                window = window[: cut + 1]
        if window.strip():
            pieces.append(window.strip())
        advance = len(window) - overlap
        if advance <= 0 or start + len(window) >= len(text):
            break
        start += advance
    return [Chunk(id=f"c{i:03d}", text=t) for i, t in enumerate(pieces)]


def load_chunks(path: Path | None = None) -> list[Chunk]:
    path = path or fetch_essay()
    return chunk_text(path.read_text(encoding="utf-8"))

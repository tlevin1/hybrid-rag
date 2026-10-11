import hashlib
import uuid

import numpy as np
import pytest
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from hybrid_rag.corpus import Chunk
from hybrid_rag.retrieval import Retriever, tokenize


class HashedBagOfWords(EmbeddingFunction[Documents]):
    """Deterministic stand-in for the real embedding model, so tests run offline
    in milliseconds without downloading it. Similar wording -> similar vectors."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def __call__(self, input: Documents) -> Embeddings:
        out = []
        for text in input:
            v = np.zeros(self.dim, dtype=np.float32)
            for tok in tokenize(text):
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
            n = np.linalg.norm(v)
            out.append(v / n if n else v)
        return out

    @staticmethod
    def name() -> str:
        return "hashed-bag-of-words"

    def get_config(self) -> dict:
        return {"dim": self.dim}

    @staticmethod
    def build_from_config(config: dict) -> "HashedBagOfWords":
        return HashedBagOfWords(config["dim"])


CHUNKS = [
    Chunk("c000", "The IBM 1401 used punched cards and Fortran was the language we tried to use."),
    Chunk("c001", "Painting classes at the Accademia in Florence taught still life."),
    Chunk("c002", "Viaweb let merchants build online stores; Yahoo bought it in 1998."),
    Chunk("c003", "Y Combinator funded startups in batches, starting with the summer founders program."),
    Chunk("c004", "Lisp macros let programs write programs, which made Viaweb fast to change."),
]


@pytest.fixture
def retriever():
    # A unique collection name: EphemeralClient shares state within a process.
    return Retriever(CHUNKS, embedding_function=HashedBagOfWords(), collection=f"t{uuid.uuid4().hex}")

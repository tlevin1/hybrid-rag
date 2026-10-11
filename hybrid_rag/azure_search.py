"""Azure AI Search as a retrieval backend: keyword (BM25), vector (HNSW), and
its built-in hybrid search, which fuses the two with reciprocal rank fusion.

The index holds the same 123 chunks with the same MiniLM vectors as Chroma
(computed locally with Chroma's embedding function), so differences against
the local retrievers come from the search engine, not the embeddings.

Credentials come from AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY, read from
the environment or a gitignored .env file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from .corpus import Chunk
from .retrieval import Hit

AzureMode = Literal["keyword", "vector", "hybrid"]
INDEX_NAME = "hybrid-rag-chunks-en"
# English analyzer: stopword removal + stemming. Measured on the eval set,
# Azure's default "standard" analyzer (no stopword removal) cost 9 points of
# keyword hit@1; en.lucene matches the hand-written BM25.
DEFAULT_ANALYZER = "en.lucene"
DIMENSIONS = 384  # all-MiniLM-L6-v2
CANDIDATES = 20  # same candidate depth the local hybrid fuses


def credentials() -> tuple[str, str]:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    endpoint, key = os.getenv("AZURE_SEARCH_ENDPOINT"), os.getenv("AZURE_SEARCH_KEY")
    if not (endpoint and key):
        raise RuntimeError("Set AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY (environment or .env).")
    return endpoint, key


class AzureSearchIndex:
    def __init__(self, chunks: list[Chunk], endpoint: str | None = None, key: str | None = None,
                 index_name: str = INDEX_NAME, embedding_function=None, analyzer: str | None = DEFAULT_ANALYZER):
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient
        from azure.search.documents.indexes import SearchIndexClient

        if not (endpoint and key):
            endpoint, key = credentials()
        cred = AzureKeyCredential(key)
        if embedding_function is None:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

            embedding_function = DefaultEmbeddingFunction()
        self.embed = embedding_function
        self.index_client = SearchIndexClient(endpoint, cred)
        self.client = SearchClient(endpoint, index_name, cred)
        self.index_name = index_name
        # None = Azure's default "standard" analyzer (lowercase + word split, no
        # stopword removal); "en.lucene" adds English stopwords and stemming.
        self.analyzer = analyzer
        self._ensure_index()
        if self.client.get_document_count() != len(chunks):
            self._upload(chunks)

    def _ensure_index(self) -> None:
        from azure.search.documents.indexes.models import (
            HnswAlgorithmConfiguration,
            HnswParameters,
            SearchableField,
            SearchField,
            SearchFieldDataType,
            SearchIndex,
            SimpleField,
            VectorSearch,
            VectorSearchAlgorithmMetric,
            VectorSearchProfile,
        )

        index = SearchIndex(
            name=self.index_name,
            fields=[
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                # Default analyzer: lowercasing and word splitting, BM25 scoring.
                SearchableField(name="text", type=SearchFieldDataType.String, analyzer_name=self.analyzer),
                SearchField(
                    name="embedding",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=DIMENSIONS,
                    vector_search_profile_name="hnsw-cosine",
                ),
            ],
            vector_search=VectorSearch(
                algorithms=[HnswAlgorithmConfiguration(
                    name="hnsw", parameters=HnswParameters(metric=VectorSearchAlgorithmMetric.COSINE),
                )],
                profiles=[VectorSearchProfile(name="hnsw-cosine", algorithm_configuration_name="hnsw")],
            ),
        )
        self.index_client.create_or_update_index(index)

    def _upload(self, chunks: list[Chunk]) -> None:
        vectors = self.embed([c.text for c in chunks])
        docs = [{"id": c.id, "text": c.text, "embedding": [float(x) for x in v]} for c, v in zip(chunks, vectors)]
        results = self.client.merge_or_upload_documents(docs)
        failed = [r.key for r in results if not r.succeeded]
        if failed:
            raise RuntimeError(f"Azure rejected {len(failed)} chunks: {failed[:5]}")
        self._wait_until_indexed(len(chunks))

    def _wait_until_indexed(self, expected: int, timeout_s: float = 60.0) -> None:
        """Uploads are indexed asynchronously: for a few seconds, searches see a
        partial index (an early test got zero keyword hits). Block until every
        document is searchable so no query runs against a half-built index."""
        import time

        deadline = time.monotonic() + timeout_s
        while self.client.get_document_count() < expected:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Azure indexed {self.client.get_document_count()}/{expected} chunks in {timeout_s}s")
            time.sleep(1.0)

    def search(self, query: str, k: int = 4, mode: AzureMode = "hybrid") -> list[Hit]:
        from azure.search.documents.models import VectorizedQuery

        kwargs = {"top": k, "select": ["id", "text"]}
        if mode in ("keyword", "hybrid"):
            kwargs["search_text"] = query
        if mode in ("vector", "hybrid"):
            vector = [float(x) for x in self.embed([query])[0]]
            kwargs["vector_queries"] = [VectorizedQuery(vector=vector, k_nearest_neighbors=max(k, CANDIDATES),
                                                        fields="embedding")]
        if mode not in ("keyword", "vector", "hybrid"):
            raise ValueError(f"unknown Azure mode {mode!r}")
        return [
            Hit(r["id"], r["text"], float(r["@search.score"]), {f"azure-{mode}": i})
            for i, r in enumerate(self.client.search(**kwargs), 1)
        ]

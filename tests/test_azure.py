"""Azure AI Search retriever against a fake client: no network, no credentials."""

import pytest

pytest.importorskip("azure.search.documents")

from hybrid_rag import azure_search  # noqa: E402
from hybrid_rag.azure_search import AzureSearchIndex  # noqa: E402


class FakeSearchClient:
    def __init__(self, results=None, counts=None):
        self.results = results or [{"id": "c001", "text": "one", "@search.score": 2.0},
                                   {"id": "c002", "text": "two", "@search.score": 1.0}]
        self.counts = list(counts or [])
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return iter(self.results)

    def get_document_count(self):
        return self.counts.pop(0) if len(self.counts) > 1 else self.counts[0]


def index_with(client):
    """An AzureSearchIndex wired to a fake client, skipping the network setup."""
    idx = object.__new__(AzureSearchIndex)
    idx.client = client
    idx.embed = lambda texts: [[0.1, 0.2, 0.3] for _ in texts]
    return idx


def test_keyword_mode_sends_text_only():
    client = FakeSearchClient()
    index_with(client).search("who bought viaweb", k=2, mode="keyword")
    call = client.calls[0]
    assert call["search_text"] == "who bought viaweb" and "vector_queries" not in call


def test_vector_mode_sends_the_embedding_only():
    client = FakeSearchClient()
    index_with(client).search("who bought viaweb", k=2, mode="vector")
    call = client.calls[0]
    assert "search_text" not in call
    (vq,) = call["vector_queries"]
    assert vq.vector == [0.1, 0.2, 0.3] and vq.k_nearest_neighbors == azure_search.CANDIDATES


def test_hybrid_mode_sends_both_so_azure_fuses_them():
    client = FakeSearchClient()
    index_with(client).search("q", k=4, mode="hybrid")
    call = client.calls[0]
    assert call["search_text"] == "q" and len(call["vector_queries"]) == 1 and call["top"] == 4


def test_results_become_hits_with_azure_ranks():
    hits = index_with(FakeSearchClient()).search("q", k=2, mode="hybrid")
    assert [h.chunk_id for h in hits] == ["c001", "c002"]
    assert hits[0].ranks == {"azure-hybrid": 1} and hits[1].score == 1.0


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown Azure mode"):
        index_with(FakeSearchClient()).search("q", mode="semantic")


def test_waits_until_every_chunk_is_searchable(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    client = FakeSearchClient(counts=[0, 50, 123])
    index_with(client)._wait_until_indexed(123)
    assert client.counts == [123]  # polled until the full count appeared


def test_gives_up_if_indexing_never_finishes(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(TimeoutError, match="0/123"):
        index_with(FakeSearchClient(counts=[0]))._wait_until_indexed(123, timeout_s=0.0)


def test_missing_credentials_fail_with_a_clear_message(monkeypatch, tmp_path):
    monkeypatch.delenv("AZURE_SEARCH_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_SEARCH_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    with pytest.raises(RuntimeError, match="AZURE_SEARCH_ENDPOINT"):
        azure_search.credentials()

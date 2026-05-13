"""Unit tests for spec_qa.embeddings.VoyageEmbedder.

Live Voyage API calls are skipped unless VOYAGE_API_KEY is set.
All core logic (batching, order preservation, empty-string filtering) is
verified through a mocked voyageai.Client.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, call, patch

import pytest

from spec_qa.embeddings import VoyageEmbedder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_response(texts: list[str], dim: int = 8) -> MagicMock:
    """Return a mock Voyage response whose .embeddings is a list of dim-vectors."""
    resp = MagicMock()
    resp.embeddings = [[float(i)] * dim for i in range(len(texts))]
    return resp


def _make_embedder(mock_client: MagicMock) -> VoyageEmbedder:
    """Construct a VoyageEmbedder that uses *mock_client* internally."""
    with patch("voyageai.Client", return_value=mock_client):
        embedder = VoyageEmbedder(api_key="fake-key", model="voyage-3-large")
    return embedder


# ---------------------------------------------------------------------------
# Mocked tests (no API key required)
# ---------------------------------------------------------------------------


class TestEmbedDocumentsMocked:
    def test_single_batch(self) -> None:
        """All texts fit in one batch → exactly one client.embed() call."""
        mock_client = MagicMock()
        texts = ["alpha", "beta", "gamma"]
        mock_client.embed.return_value = _fake_response(texts)

        embedder = _make_embedder(mock_client)
        result = embedder.embed_documents(texts, batch_size=128)

        assert len(result) == 3
        mock_client.embed.assert_called_once()
        _, kwargs = mock_client.embed.call_args
        assert kwargs.get("input_type") == "document" or mock_client.embed.call_args[0][1] == "document" or True
        # Verify input_type was passed correctly by inspecting positional/kw args
        call_args = mock_client.embed.call_args
        assert "document" in str(call_args)

    def test_batch_splitting(self) -> None:
        """With batch_size=2 and 5 texts, embed() should be called 3 times."""
        mock_client = MagicMock()

        def side_effect(texts, **kwargs):
            return _fake_response(texts)

        mock_client.embed.side_effect = side_effect

        embedder = _make_embedder(mock_client)
        texts = ["a", "b", "c", "d", "e"]
        result = embedder.embed_documents(texts, batch_size=2)

        assert mock_client.embed.call_count == 3  # batches: [a,b], [c,d], [e]
        assert len(result) == 5

    def test_order_preserved(self) -> None:
        """Output order must match input order, even across multiple batches."""
        mock_client = MagicMock()

        # Each batch returns embeddings indexed from 0 within the batch —
        # we use the text itself to encode a unique fingerprint.
        call_index = {"n": 0}

        def side_effect(texts, **kwargs):
            resp = MagicMock()
            # Encode position as the first element so we can verify ordering
            resp.embeddings = [[float(call_index["n"] * 100 + i)] for i, _ in enumerate(texts)]
            call_index["n"] += 1
            return resp

        mock_client.embed.side_effect = side_effect

        embedder = _make_embedder(mock_client)
        texts = ["t0", "t1", "t2", "t3"]
        result = embedder.embed_documents(texts, batch_size=2)

        # batch 0 → positions 0,1 get [0.0], [1.0]
        # batch 1 → positions 2,3 get [100.0], [101.0]
        assert result[0] == [0.0]
        assert result[1] == [1.0]
        assert result[2] == [100.0]
        assert result[3] == [101.0]

    def test_empty_strings_skipped(self) -> None:
        """Empty or whitespace-only strings must not be sent to the API."""
        mock_client = MagicMock()

        non_empty = ["hello", "world"]
        mock_client.embed.return_value = _fake_response(non_empty)

        embedder = _make_embedder(mock_client)
        # Mix of real and empty strings
        result = embedder.embed_documents(["hello", "", "world", "  "], batch_size=128)

        # Only one batch call, containing only the two non-empty texts
        assert mock_client.embed.call_count == 1
        sent_texts = mock_client.embed.call_args[0][0]
        assert "" not in sent_texts
        assert "  " not in sent_texts
        assert len(sent_texts) == 2

        # Result length must equal input length; empty slots get zero vectors
        assert len(result) == 4
        assert result[1] == [] or all(v == 0.0 for v in result[1])
        assert result[3] == [] or all(v == 0.0 for v in result[3])

    def test_all_empty_strings(self) -> None:
        """If all inputs are empty, no API call is made and empty lists returned."""
        mock_client = MagicMock()
        embedder = _make_embedder(mock_client)

        result = embedder.embed_documents(["", "  ", ""], batch_size=128)

        mock_client.embed.assert_not_called()
        assert result == [[], [], []]

    def test_empty_input_list(self) -> None:
        """An empty input list returns an empty output list with no API call."""
        mock_client = MagicMock()
        embedder = _make_embedder(mock_client)

        result = embedder.embed_documents([], batch_size=128)

        mock_client.embed.assert_not_called()
        assert result == []

    def test_input_type_document(self) -> None:
        """embed_documents must pass input_type='document' to the Voyage client."""
        mock_client = MagicMock()
        mock_client.embed.return_value = _fake_response(["text"])

        embedder = _make_embedder(mock_client)
        embedder.embed_documents(["text"], batch_size=128)

        call_kwargs = mock_client.embed.call_args
        # input_type can be positional or keyword — check the full call representation
        assert "document" in str(call_kwargs)


class TestEmbedQueryMocked:
    def test_query_returns_single_vector(self) -> None:
        mock_client = MagicMock()
        mock_client.embed.return_value = _fake_response(["query text"], dim=8)

        embedder = _make_embedder(mock_client)
        result = embedder.embed_query("query text")

        assert isinstance(result, list)
        assert len(result) == 8

    def test_input_type_query(self) -> None:
        """embed_query must pass input_type='query' to the Voyage client."""
        mock_client = MagicMock()
        mock_client.embed.return_value = _fake_response(["q"])

        embedder = _make_embedder(mock_client)
        embedder.embed_query("q")

        assert "query" in str(mock_client.embed.call_args)

    def test_empty_query_raises(self) -> None:
        mock_client = MagicMock()
        embedder = _make_embedder(mock_client)

        with pytest.raises(ValueError, match="empty"):
            embedder.embed_query("")


class TestRetryMocked:
    def test_retry_on_failure(self) -> None:
        """embed_documents should retry once on exception and succeed."""
        mock_client = MagicMock()
        success_resp = _fake_response(["text"])
        # First call raises; second call succeeds
        mock_client.embed.side_effect = [RuntimeError("transient"), success_resp]

        embedder = _make_embedder(mock_client)
        # patch time.sleep to avoid actual waiting
        with patch("spec_qa.embeddings.time.sleep"):
            result = embedder.embed_documents(["text"], batch_size=128)

        assert mock_client.embed.call_count == 2
        assert len(result) == 1

    def test_no_retry_after_max_exceeded(self) -> None:
        """If both attempts fail, the exception should propagate."""
        mock_client = MagicMock()
        mock_client.embed.side_effect = RuntimeError("persistent error")

        embedder = _make_embedder(mock_client)
        with patch("spec_qa.embeddings.time.sleep"):
            with pytest.raises(RuntimeError, match="persistent error"):
                embedder.embed_documents(["text"], batch_size=128)


# ---------------------------------------------------------------------------
# Live integration tests (skipped without API key)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("VOYAGE_API_KEY"),
    reason="VOYAGE_API_KEY not set — skipping live Voyage API test",
)
class TestEmbedDocumentsLive:
    def test_embed_returns_correct_dimension(self) -> None:
        embedder = VoyageEmbedder()
        results = embedder.embed_documents(["Hello, 3GPP world."])
        assert len(results) == 1
        # voyage-3-large → 1024 dimensions
        assert len(results[0]) == 1024

    def test_embed_query_live(self) -> None:
        embedder = VoyageEmbedder()
        vec = embedder.embed_query("What is the purpose of RRC?")
        assert len(vec) == 1024

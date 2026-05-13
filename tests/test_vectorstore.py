"""Unit tests for spec_qa.vectorstore.ChromaSpecStore.

Uses a temporary directory so each test run gets a fresh ChromaDB instance.
No Voyage API key is required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spec_qa.parser import Chunk
from spec_qa.vectorstore import ChromaSpecStore, QueryResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)  # fixed seed for reproducibility


def _random_vec(dim: int = 1024) -> list[float]:
    """Return a random unit-normalised float vector."""
    v = _RNG.standard_normal(dim).astype(float)
    v /= np.linalg.norm(v)
    return v.tolist()


def _make_chunk(
    chunk_id: str,
    spec_no: str = "38.331",
    section_no: str = "5.1",
    text: str = "Sample body text.",
) -> Chunk:
    return Chunk(
        spec_no=spec_no,
        version="18.5.0",
        section_no=section_no,
        section_title="Sample Section",
        heading_path=["5", section_no],
        text=text,
        token_count=4,
        chunk_id=chunk_id,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path: Path) -> ChromaSpecStore:
    """Fresh ChromaSpecStore backed by a temporary directory."""
    return ChromaSpecStore(db_path=tmp_path / "chroma_db", collection_name="test_specs")


@pytest.fixture()
def three_chunks() -> tuple[list[Chunk], list[list[float]]]:
    """Three dummy chunks with random 1024-dim embeddings."""
    chunks = [
        _make_chunk("chunk-1", spec_no="38.331", section_no="5.1"),
        _make_chunk("chunk-2", spec_no="38.331", section_no="5.2"),
        _make_chunk("chunk-3", spec_no="36.331", section_no="6.1"),
    ]
    embeddings = [_random_vec() for _ in chunks]
    return chunks, embeddings


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestUpsertAndCount:
    def test_count_after_upsert(
        self,
        store: ChromaSpecStore,
        three_chunks: tuple[list[Chunk], list[list[float]]],
    ) -> None:
        chunks, embeddings = three_chunks
        store.upsert_chunks(chunks, embeddings)
        assert store.count() == 3

    def test_empty_upsert_is_noop(self, store: ChromaSpecStore) -> None:
        store.upsert_chunks([], [])
        assert store.count() == 0

    def test_mismatched_lengths_raises(self, store: ChromaSpecStore) -> None:
        chunk = _make_chunk("chunk-x")
        with pytest.raises(ValueError, match="same length"):
            store.upsert_chunks([chunk], [])

    def test_upsert_is_idempotent(
        self,
        store: ChromaSpecStore,
        three_chunks: tuple[list[Chunk], list[list[float]]],
    ) -> None:
        """Upserting the same chunks twice should not increase count."""
        chunks, embeddings = three_chunks
        store.upsert_chunks(chunks, embeddings)
        store.upsert_chunks(chunks, embeddings)
        assert store.count() == 3


class TestQuery:
    def test_top_k_result_length(
        self,
        store: ChromaSpecStore,
        three_chunks: tuple[list[Chunk], list[list[float]]],
    ) -> None:
        chunks, embeddings = three_chunks
        store.upsert_chunks(chunks, embeddings)

        query_vec = _random_vec()
        results = store.query(query_vec, top_k=2)
        assert len(results) == 2

    def test_result_type(
        self,
        store: ChromaSpecStore,
        three_chunks: tuple[list[Chunk], list[list[float]]],
    ) -> None:
        chunks, embeddings = three_chunks
        store.upsert_chunks(chunks, embeddings)

        results = store.query(_random_vec(), top_k=1)
        assert isinstance(results[0], QueryResult)
        assert isinstance(results[0].chunk_id, str)
        assert isinstance(results[0].text, str)
        assert isinstance(results[0].metadata, dict)
        assert isinstance(results[0].distance, float)

    def test_exact_match_is_top_result(
        self,
        store: ChromaSpecStore,
        three_chunks: tuple[list[Chunk], list[list[float]]],
    ) -> None:
        """Querying with an exact stored embedding should retrieve that chunk first."""
        chunks, embeddings = three_chunks
        store.upsert_chunks(chunks, embeddings)

        # Use the embedding of chunk-2 as the query
        results = store.query(embeddings[1], top_k=3)
        assert results[0].chunk_id == "chunk-2"

    def test_where_filter(
        self,
        store: ChromaSpecStore,
        three_chunks: tuple[list[Chunk], list[list[float]]],
    ) -> None:
        """A where filter on spec_no should restrict results to matching chunks."""
        chunks, embeddings = three_chunks
        store.upsert_chunks(chunks, embeddings)

        # Only 2 chunks have spec_no="38.331"; request top_k=3 but expect 2 back
        results = store.query(
            _random_vec(),
            top_k=3,
            where={"spec_no": "38.331"},
        )
        assert len(results) == 2
        for r in results:
            assert r.metadata["spec_no"] == "38.331"

    def test_top_k_capped_by_collection_size(
        self,
        store: ChromaSpecStore,
        three_chunks: tuple[list[Chunk], list[list[float]]],
    ) -> None:
        """Requesting more results than stored documents returns all documents."""
        chunks, embeddings = three_chunks
        store.upsert_chunks(chunks, embeddings)

        results = store.query(_random_vec(), top_k=100)
        assert len(results) == 3


class TestReset:
    def test_reset_empties_collection(
        self,
        store: ChromaSpecStore,
        three_chunks: tuple[list[Chunk], list[list[float]]],
    ) -> None:
        chunks, embeddings = three_chunks
        store.upsert_chunks(chunks, embeddings)
        assert store.count() == 3

        store.reset()
        assert store.count() == 0

    def test_reset_empty_collection_is_safe(self, store: ChromaSpecStore) -> None:
        """Calling reset on an empty collection must not raise."""
        store.reset()
        assert store.count() == 0

    def test_upsert_after_reset(
        self,
        store: ChromaSpecStore,
        three_chunks: tuple[list[Chunk], list[list[float]]],
    ) -> None:
        chunks, embeddings = three_chunks
        store.upsert_chunks(chunks, embeddings)
        store.reset()

        # Should be able to re-insert after reset
        store.upsert_chunks(chunks[:1], embeddings[:1])
        assert store.count() == 1

"""Unit tests for spec_qa.rag.SpecRAG and spec_qa.cli.

Tests use unittest.mock to avoid real Anthropic / Voyage / ChromaDB API calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from spec_qa.rag import Citation, RagAnswer, SpecRAG, _extract_usage, estimate_cost
from spec_qa.vectorstore import QueryResult


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


def _make_query_result(
    chunk_id: str = "chunk-1",
    spec_no: str = "38.331",
    version: str = "18.5.0",
    section_no: str = "5.1",
    section_title: str = "Test Section",
    text: str = "Sample spec text about PDCCH monitoring.",
    distance: float = 0.1,
) -> QueryResult:
    return QueryResult(
        chunk_id=chunk_id,
        text=text,
        metadata={
            "spec_no": spec_no,
            "version": version,
            "section_no": section_no,
            "section_title": section_title,
            "heading_path_str": section_no,
            "token_count": 10,
            "chunk_id": chunk_id,
        },
        distance=distance,
    )


def _make_mock_response(
    answer_text: str = "답변 본문입니다.",
    citation_text: str = "Sample spec text about PDCCH monitoring.",
    document_index: int = 0,
    input_tokens: int = 1000,
    output_tokens: int = 200,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 800,
) -> MagicMock:
    """Build a mock Anthropic Messages response with one text block containing a citation."""
    mock_usage = MagicMock()
    mock_usage.input_tokens = input_tokens
    mock_usage.output_tokens = output_tokens
    mock_usage.cache_creation_input_tokens = cache_creation_input_tokens
    mock_usage.cache_read_input_tokens = cache_read_input_tokens

    # Simulate a CitationContentBlockLocation object
    mock_citation = MagicMock()
    mock_citation.type = "content_block_location"
    mock_citation.cited_text = citation_text
    mock_citation.document_index = document_index

    mock_block = MagicMock()
    mock_block.type = "text"
    mock_block.text = answer_text
    mock_block.citations = [mock_citation]

    mock_response = MagicMock()
    mock_response.content = [mock_block]
    mock_response.usage = mock_usage

    return mock_response


@pytest.fixture()
def mock_store() -> MagicMock:
    store = MagicMock(spec=["query", "count", "upsert_chunks", "reset", "_collection"])
    store.query.return_value = [_make_query_result()]
    return store


@pytest.fixture()
def mock_embedder() -> MagicMock:
    embedder = MagicMock(spec=["embed_query", "embed_documents"])
    embedder.embed_query.return_value = [0.1] * 1024
    return embedder


@pytest.fixture()
def rag(mock_store: MagicMock, mock_embedder: MagicMock) -> SpecRAG:
    """SpecRAG instance with mocked store, embedder, and Anthropic client."""
    with patch("anthropic.Anthropic") as mock_anthropic_cls:
        instance = SpecRAG(
            store=mock_store,
            embedder=mock_embedder,
            anthropic_api_key="test-key-not-real",
        )
        # Attach the mock client for later configuration in tests
        instance._mock_anthropic_cls = mock_anthropic_cls
    return instance


# ---------------------------------------------------------------------------
# Tests: SpecRAG.answer — core flow
# ---------------------------------------------------------------------------


class TestSpecRAGAnswer:
    def test_returns_rag_answer(self, rag: SpecRAG, mock_store: MagicMock) -> None:
        """answer() should return a RagAnswer with all fields populated."""
        mock_response = _make_mock_response()
        rag._client.messages.create.return_value = mock_response

        result = rag.answer("What is PDCCH monitoring occasion?")

        assert isinstance(result, RagAnswer)
        assert result.question == "What is PDCCH monitoring occasion?"
        assert result.answer == "답변 본문입니다."
        assert isinstance(result.citations, list)
        assert isinstance(result.retrieved_chunks, list)
        assert isinstance(result.usage, dict)

    def test_embed_query_called(self, rag: SpecRAG, mock_embedder: MagicMock) -> None:
        """answer() must embed the question before querying the store."""
        rag._client.messages.create.return_value = _make_mock_response()
        question = "NR PDCCH monitoring?"
        rag.answer(question)
        mock_embedder.embed_query.assert_called_once_with(question)

    def test_store_query_called_with_embedding(
        self, rag: SpecRAG, mock_store: MagicMock, mock_embedder: MagicMock
    ) -> None:
        """The embedding from embed_query must be passed to store.query."""
        fake_vec = [0.5] * 1024
        mock_embedder.embed_query.return_value = fake_vec
        rag._client.messages.create.return_value = _make_mock_response()

        rag.answer("test question")

        call_args = mock_store.query.call_args
        # store.query is called positionally: query(query_vec, top_k=..., where=...)
        positional_vec = call_args[0][0] if call_args[0] else call_args[1].get("query_embedding")
        assert positional_vec == fake_vec

    def test_top_k_override(self, rag: SpecRAG, mock_store: MagicMock) -> None:
        """Passing top_k= should override the instance default."""
        rag._client.messages.create.return_value = _make_mock_response()
        rag.answer("test", top_k=3)
        # top_k must be forwarded to store.query
        call_kwargs = mock_store.query.call_args
        assert call_kwargs[1].get("top_k") == 3 or call_kwargs[0][1] == 3

    def test_spec_filter_passed_as_where(self, rag: SpecRAG, mock_store: MagicMock) -> None:
        """spec_filter list must be translated to a Chroma $in where clause."""
        rag._client.messages.create.return_value = _make_mock_response()
        rag.answer("test", spec_filter=["38.331", "36.331"])

        call_kwargs = mock_store.query.call_args
        # where kwarg should be {"spec_no": {"$in": [...]}}
        where = call_kwargs[1].get("where") or call_kwargs[0][2]
        assert where == {"spec_no": {"$in": ["38.331", "36.331"]}}

    def test_no_spec_filter_means_no_where(self, rag: SpecRAG, mock_store: MagicMock) -> None:
        """When spec_filter is None, where must be None (no filter applied)."""
        rag._client.messages.create.return_value = _make_mock_response()
        rag.answer("test", spec_filter=None)

        call_kwargs = mock_store.query.call_args
        where = call_kwargs[1].get("where")
        assert where is None


# ---------------------------------------------------------------------------
# Tests: Document blocks construction
# ---------------------------------------------------------------------------


class TestDocumentBlocks:
    def _capture_messages_create_kwargs(self, rag: SpecRAG, mock_store: MagicMock) -> dict:
        """Call answer() and return the kwargs passed to messages.create."""
        rag._client.messages.create.return_value = _make_mock_response()
        rag.answer("PDCCH test")
        return rag._client.messages.create.call_args[1]

    def test_document_blocks_count_matches_chunks(
        self, rag: SpecRAG, mock_store: MagicMock
    ) -> None:
        """One document block must be created per retrieved chunk."""
        two_chunks = [_make_query_result("c1"), _make_query_result("c2")]
        mock_store.query.return_value = two_chunks

        kwargs = self._capture_messages_create_kwargs(rag, mock_store)
        messages = kwargs["messages"]
        user_content = messages[0]["content"]

        doc_blocks = [b for b in user_content if b.get("type") == "document"]
        assert len(doc_blocks) == 2

    def test_document_block_structure(self, rag: SpecRAG, mock_store: MagicMock) -> None:
        """Each document block must have type, source, title, context, and citations fields."""
        kwargs = self._capture_messages_create_kwargs(rag, mock_store)
        messages = kwargs["messages"]
        user_content = messages[0]["content"]

        doc_block = next(b for b in user_content if b.get("type") == "document")
        assert doc_block["source"]["type"] == "text"
        assert doc_block["source"]["media_type"] == "text/plain"
        assert "spec_no" in doc_block["title"] or "38.331" in doc_block["title"]
        assert doc_block["citations"] == {"enabled": True}

    def test_question_text_block_is_last(self, rag: SpecRAG, mock_store: MagicMock) -> None:
        """The user question text block must come after all document blocks."""
        kwargs = self._capture_messages_create_kwargs(rag, mock_store)
        user_content = kwargs["messages"][0]["content"]
        last_block = user_content[-1]
        assert last_block["type"] == "text"
        assert last_block["text"] == "PDCCH test"

    def test_system_has_cache_control(self, rag: SpecRAG, mock_store: MagicMock) -> None:
        """System prompt must carry cache_control={'type': 'ephemeral'}."""
        kwargs = self._capture_messages_create_kwargs(rag, mock_store)
        system = kwargs["system"]
        # system is a list of blocks
        assert isinstance(system, list)
        first_block = system[0]
        assert first_block.get("cache_control") == {"type": "ephemeral"}


# ---------------------------------------------------------------------------
# Tests: Citation parsing
# ---------------------------------------------------------------------------


class TestCitationParsing:
    def test_citation_parsed_correctly(self, rag: SpecRAG, mock_store: MagicMock) -> None:
        """Parsed Citation must contain spec metadata from the matched chunk."""
        chunk = _make_query_result(
            spec_no="38.213",
            version="18.5.0",
            section_no="10.1",
            section_title="Search space sets",
            text="slot offset and pattern are configured via searchSpacesToAddModList",
        )
        mock_store.query.return_value = [chunk]
        mock_response = _make_mock_response(
            citation_text="slot offset and pattern are configured via searchSpacesToAddModList",
            document_index=0,
        )
        rag._client.messages.create.return_value = mock_response

        result = rag.answer("How is PDCCH monitoring configured?")

        assert len(result.citations) == 1
        cit = result.citations[0]
        assert cit.spec_no == "38.213"
        assert cit.version == "18.5.0"
        assert cit.section_no == "10.1"
        assert cit.section_title == "Search space sets"
        assert "searchSpacesToAddModList" in cit.cited_text

    def test_no_citations_when_empty(self, rag: SpecRAG) -> None:
        """If the response block has no citations, result.citations must be empty."""
        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = "답변"
        mock_block.citations = []

        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_response.usage = MagicMock(
            input_tokens=100,
            output_tokens=20,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
        rag._client.messages.create.return_value = mock_response

        result = rag.answer("test")
        assert result.citations == []

    def test_duplicate_cited_text_deduplicated(
        self, rag: SpecRAG, mock_store: MagicMock
    ) -> None:
        """Same cited_text appearing twice should produce only one Citation entry."""
        chunk = _make_query_result()
        mock_store.query.return_value = [chunk]

        dup_cit = MagicMock()
        dup_cit.type = "content_block_location"
        dup_cit.cited_text = "duplicate text"
        dup_cit.document_index = 0

        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = "Answer."
        mock_block.citations = [dup_cit, dup_cit]  # same citation twice

        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_response.usage = MagicMock(
            input_tokens=100,
            output_tokens=20,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
        rag._client.messages.create.return_value = mock_response

        result = rag.answer("test")
        assert len(result.citations) == 1

    def test_out_of_range_document_index_skipped(
        self, rag: SpecRAG, mock_store: MagicMock
    ) -> None:
        """A citation with document_index out of range must be silently ignored."""
        chunk = _make_query_result()
        mock_store.query.return_value = [chunk]

        bad_cit = MagicMock()
        bad_cit.type = "content_block_location"
        bad_cit.cited_text = "some text"
        bad_cit.document_index = 999  # out of range

        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = "Answer."
        mock_block.citations = [bad_cit]

        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_response.usage = MagicMock(
            input_tokens=100,
            output_tokens=20,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
        rag._client.messages.create.return_value = mock_response

        result = rag.answer("test")
        assert result.citations == []


# ---------------------------------------------------------------------------
# Tests: Usage extraction
# ---------------------------------------------------------------------------


class TestUsageExtraction:
    def test_usage_dict_keys(self, rag: SpecRAG) -> None:
        """Usage dict must contain the four standard token fields."""
        mock_response = _make_mock_response(
            input_tokens=1500, output_tokens=300, cache_read_input_tokens=900
        )
        rag._client.messages.create.return_value = mock_response
        result = rag.answer("test")

        assert "input_tokens" in result.usage
        assert "output_tokens" in result.usage
        assert "cache_read_input_tokens" in result.usage
        assert "cache_creation_input_tokens" in result.usage

    def test_usage_values(self, rag: SpecRAG) -> None:
        """Usage values must match what the mock response carries."""
        mock_response = _make_mock_response(
            input_tokens=1500,
            output_tokens=300,
            cache_creation_input_tokens=50,
            cache_read_input_tokens=900,
        )
        rag._client.messages.create.return_value = mock_response
        result = rag.answer("test")

        assert result.usage["input_tokens"] == 1500
        assert result.usage["output_tokens"] == 300
        assert result.usage["cache_read_input_tokens"] == 900
        assert result.usage["cache_creation_input_tokens"] == 50

    def test_estimate_cost_is_positive(self) -> None:
        """estimate_cost must return a non-negative float."""
        usage = {
            "input_tokens": 5000,
            "output_tokens": 400,
            "cache_read_input_tokens": 3000,
            "cache_creation_input_tokens": 0,
        }
        cost = estimate_cost(usage)
        assert cost > 0.0
        assert isinstance(cost, float)


# ---------------------------------------------------------------------------
# Tests: CLI — info command (no API calls)
# ---------------------------------------------------------------------------


class TestCliInfo:
    def test_info_no_db(self, tmp_path) -> None:
        """info command should not crash even if the DB is empty."""
        from typer.testing import CliRunner

        from spec_qa.cli import app

        runner = CliRunner()

        # Patch ChromaSpecStore so no real Chroma DB is created
        mock_store = MagicMock()
        mock_store.count.return_value = 0
        mock_store._collection.get.return_value = {"metadatas": []}

        with patch("spec_qa.cli.ChromaSpecStore", return_value=mock_store):
            result = runner.invoke(app, ["info"])

        assert result.exit_code == 0
        assert "0" in result.output  # total count shown

    def test_info_with_data(self) -> None:
        """info command should show spec distribution table when data exists."""
        from typer.testing import CliRunner

        from spec_qa.cli import app

        runner = CliRunner()

        mock_store = MagicMock()
        mock_store.count.return_value = 5
        mock_store._collection.get.return_value = {
            "metadatas": [
                {"spec_no": "38.331", "version": "18.5.0"},
                {"spec_no": "38.331", "version": "18.5.0"},
                {"spec_no": "38.213", "version": "18.5.0"},
                {"spec_no": "38.213", "version": "18.5.0"},
                {"spec_no": "36.331", "version": "17.0.0"},
            ]
        }

        with patch("spec_qa.cli.ChromaSpecStore", return_value=mock_store):
            result = runner.invoke(app, ["info"])

        assert result.exit_code == 0
        assert "38.331" in result.output
        assert "38.213" in result.output
        assert "36.331" in result.output

"""RAG (Retrieval-Augmented Generation) engine for 3GPP spec Q&A.

Orchestrates:
1. Query embedding via VoyageEmbedder
2. Nearest-chunk retrieval from ChromaSpecStore
3. Claude API call with Citations enabled
4. Parsing of citation blocks into structured Citation dataclasses
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

from spec_qa.embeddings import VoyageEmbedder
from spec_qa.vectorstore import ChromaSpecStore, QueryResult


# ---------------------------------------------------------------------------
# Load .env once
# ---------------------------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------------------------
# Domain dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Citation:
    """A single citation extracted from a Claude Citations-API response."""

    spec_no: str
    version: str
    section_no: str
    section_title: str
    cited_text: str  # Exact text Claude cited from the source document


@dataclass(frozen=True)
class RagAnswer:
    """Complete answer produced by SpecRAG.answer()."""

    question: str
    answer: str                          # Answer body (may contain citation markers)
    citations: list[Citation]            # Parsed Citation objects
    retrieved_chunks: list[QueryResult]  # Raw retrieval results (for debugging)
    usage: dict                          # Token usage from response.usage


# ---------------------------------------------------------------------------
# SpecRAG
# ---------------------------------------------------------------------------

# Sonnet 4.6 pricing per million tokens (approximate, for cost estimate only)
_PRICE_INPUT = 3.0 / 1_000_000
_PRICE_OUTPUT = 15.0 / 1_000_000
_PRICE_CACHE_READ = 0.30 / 1_000_000


class SpecRAG:
    """RAG pipeline that answers questions about 3GPP TS 36/38 specifications.

    Parameters
    ----------
    store:
        ChromaSpecStore instance. Created with default settings if None.
    embedder:
        VoyageEmbedder instance. Created with default settings if None.
    anthropic_api_key:
        Anthropic API key. Falls back to ``ANTHROPIC_API_KEY`` env var.
    model:
        Claude model ID. Falls back to ``ANTHROPIC_MODEL`` env var, then
        ``claude-sonnet-4-6``.
    top_k:
        Default number of chunks to retrieve per query.
    """

    def __init__(
        self,
        store: ChromaSpecStore | None = None,
        embedder: VoyageEmbedder | None = None,
        anthropic_api_key: str | None = None,
        model: str = "claude-sonnet-4-6",
        top_k: int = 8,
    ) -> None:
        self._store = store or ChromaSpecStore()
        self._embedder = embedder or VoyageEmbedder()

        api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key not found. "
                "Set ANTHROPIC_API_KEY env var or pass anthropic_api_key=."
            )

        import anthropic  # noqa: PLC0415 — lazy import; anthropic is optional at module load

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = os.environ.get("ANTHROPIC_MODEL", model)
        self._top_k = top_k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        spec_filter: list[str] | None = None,
    ) -> RagAnswer:
        """Answer *question* using retrieved spec chunks as context.

        Parameters
        ----------
        question:
            Natural language question about 3GPP specifications.
        top_k:
            Number of chunks to retrieve. Overrides the instance default.
        spec_filter:
            If given, restrict retrieval to these spec numbers
            (e.g. ``["38.331", "36.331"]``).

        Returns
        -------
        RagAnswer
            Structured answer with citations and token-usage statistics.
        """
        k = top_k if top_k is not None else self._top_k

        # 1. Embed the query
        query_vec = self._embedder.embed_query(question)

        # 2. Build optional Chroma metadata filter
        where: dict | None = None
        if spec_filter:
            where = {"spec_no": {"$in": spec_filter}}

        # 3. Retrieve nearest chunks
        chunks = self._store.query(query_vec, top_k=k, where=where)

        # 4. Build the messages payload and call Claude
        response = self._call_claude(question, chunks)

        # 5. Parse the response into structured output
        answer_text, citations = self._parse_response(response, chunks)

        # 6. Collect token usage
        usage = _extract_usage(response.usage)

        return RagAnswer(
            question=question,
            answer=answer_text,
            citations=citations,
            retrieved_chunks=chunks,
            usage=usage,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_claude(self, question: str, chunks: list[QueryResult]):  # type: ignore[return]
        """Build and send the Claude API request with Citations enabled.

        System prompt is cached with cache_control=ephemeral because it never
        changes between calls in the same session (↑ cache-hit rate, ↓ cost).

        Each retrieved chunk becomes a separate ``document`` content block so
        that Claude's Citations API can emit ``content_block_location``
        citations pointing back to a specific document_index.
        """
        system_prompt = (
            "당신은 3GPP 무선 통신 표준(TS 36 시리즈, TS 38 시리즈)에 특화된 "
            "기술 전문가입니다.\n\n"
            "규칙:\n"
            "1. 제공된 컨텍스트(문서 블록)만을 근거로 답변하십시오. "
            "컨텍스트에 없는 내용은 추측하지 마십시오.\n"
            "2. 답변에 인용(citation)을 최대한 활용하여 출처를 명확히 하십시오.\n"
            "3. 컨텍스트에서 답을 찾을 수 없으면 '제공된 문서에서 해당 정보를 "
            "찾을 수 없습니다'라고 명시하십시오.\n"
            "4. 기술 용어는 원문 그대로 사용하고, 설명은 한국어로 제공하십시오."
        )

        # Build document blocks from retrieved chunks
        doc_blocks: list[dict] = [
            {
                "type": "document",
                "source": {
                    "type": "text",
                    "media_type": "text/plain",
                    "data": chunk.text,
                },
                "title": (
                    f"{chunk.metadata['spec_no']} "
                    f"v{chunk.metadata['version']} "
                    f"§{chunk.metadata['section_no']} "
                    f"{chunk.metadata['section_title']}"
                ),
                "context": (
                    f"3GPP TS {chunk.metadata['spec_no']} "
                    f"section {chunk.metadata['section_no']}"
                ),
                # Enable citations for this document block
                "citations": {"enabled": True},
            }
            for chunk in chunks
        ]

        # User message: all document blocks first, then the question
        user_content: list[dict] = [
            *doc_blocks,
            {"type": "text", "text": question},
        ]

        return self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    # Prompt cache: system rarely changes → high cache-hit probability
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_content}],
        )

    @staticmethod
    def _parse_response(response, chunks: list[QueryResult]) -> tuple[str, list[Citation]]:
        """Extract answer text and citations from a Claude API response.

        Claude returns a list of content blocks. Each ``text`` block may carry
        a ``citations`` list containing ``content_block_location`` objects (one
        per cited document). We extract ``cited_text`` and map back to the
        original chunk via ``document_index``.

        Returns
        -------
        tuple[str, list[Citation]]
            (joined answer text, deduplicated citation list)
        """
        text_parts: list[str] = []
        seen_cited_texts: set[str] = set()
        citations: list[Citation] = []

        for block in response.content:
            if block.type != "text":
                continue

            text_parts.append(block.text)

            # Each text block may carry zero or more citation references
            if not block.citations:
                continue

            for raw_cit in block.citations:
                # We use the Citations API with plain-text documents, so the
                # citation type will be "content_block_location"
                cited_text: str = getattr(raw_cit, "cited_text", "")
                doc_index: int = getattr(raw_cit, "document_index", -1)

                if not cited_text or doc_index < 0 or doc_index >= len(chunks):
                    continue

                # Deduplicate by exact cited_text to avoid repeated citations
                if cited_text in seen_cited_texts:
                    continue
                seen_cited_texts.add(cited_text)

                chunk = chunks[doc_index]
                citations.append(
                    Citation(
                        spec_no=str(chunk.metadata.get("spec_no", "")),
                        version=str(chunk.metadata.get("version", "")),
                        section_no=str(chunk.metadata.get("section_no", "")),
                        section_title=str(chunk.metadata.get("section_title", "")),
                        cited_text=cited_text,
                    )
                )

        return "".join(text_parts), citations


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _extract_usage(usage) -> dict:
    """Convert a Claude Usage object into a plain dict.

    Includes cache_creation_input_tokens and cache_read_input_tokens if
    present so callers can compute accurate cost estimates.
    """
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_creation_input_tokens": usage.cache_creation_input_tokens or 0,
        "cache_read_input_tokens": usage.cache_read_input_tokens or 0,
    }


def estimate_cost(usage: dict) -> float:
    """Return an approximate cost in USD for the given usage dict."""
    return (
        usage.get("input_tokens", 0) * _PRICE_INPUT
        + usage.get("output_tokens", 0) * _PRICE_OUTPUT
        + usage.get("cache_read_input_tokens", 0) * _PRICE_CACHE_READ
    )

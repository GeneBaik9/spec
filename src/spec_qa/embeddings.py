"""Voyage AI embedding client wrapper for 3GPP spec chunks.

Provides document-level and query-level embeddings using the Voyage AI SDK.
Handles batching, empty-text filtering, and single retry on failure.
"""

from __future__ import annotations

import os
import time
from functools import lru_cache

from dotenv import load_dotenv
from tqdm import tqdm

# ---------------------------------------------------------------------------
# .env loading — happens once at module import
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_env() -> None:
    """Load .env into os.environ exactly once."""
    load_dotenv()


# ---------------------------------------------------------------------------
# VoyageEmbedder
# ---------------------------------------------------------------------------


class VoyageEmbedder:
    """Thin wrapper around the Voyage AI SDK for embedding 3GPP spec text.

    Parameters
    ----------
    api_key:
        Voyage AI API key. If ``None``, falls back to the ``VOYAGE_API_KEY``
        environment variable (loaded from ``.env`` if present).
    model:
        Voyage model name. Defaults to ``voyage-3-large`` (1024-dim, 32K ctx).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        _load_env()

        self._api_key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Voyage API key not found. Set VOYAGE_API_KEY env var or pass api_key=."
            )

        self.model: str = model or os.environ.get("VOYAGE_MODEL", "voyage-3-large")

        # Lazy import — voyageai is optional at import time
        import voyageai  # noqa: PLC0415

        self._client = voyageai.Client(api_key=self._api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_documents(
        self,
        texts: list[str],
        batch_size: int = 128,
    ) -> list[list[float]]:
        """Embed a list of document strings (``input_type='document'``).

        Empty strings are silently removed before sending to Voyage (the API
        rejects them). The output list preserves the original input order;
        positions that corresponded to empty strings receive a zero vector of
        the correct dimension.

        Parameters
        ----------
        texts:
            Document strings to embed.
        batch_size:
            Number of texts to send per Voyage API call (default 128, per
            Voyage recommendation).

        Returns
        -------
        list[list[float]]
            One embedding vector per input text, in the same order.
        """
        # Collect (original_index, text) for non-empty inputs
        indexed: list[tuple[int, str]] = [
            (i, t) for i, t in enumerate(texts) if t.strip()
        ]

        # Pre-fill result with zero vectors; dimension filled in after first call
        result: list[list[float] | None] = [None] * len(texts)

        if not indexed:
            return [[] for _ in texts]

        # Split into batches — Voyage recommends ≤128 texts per request
        batches: list[list[tuple[int, str]]] = [
            indexed[start : start + batch_size]
            for start in range(0, len(indexed), batch_size)
        ]

        dim: int | None = None

        for batch in tqdm(batches, desc="Embedding batches", unit="batch", leave=False):
            batch_texts = [t for _, t in batch]
            embeddings = self._embed_with_retry(batch_texts, input_type="document")

            # Record dimension from first successful call
            if dim is None and embeddings:
                dim = len(embeddings[0])

            for (orig_idx, _), vec in zip(batch, embeddings):
                result[orig_idx] = vec

        # Replace None slots (empty-string inputs) with zero vectors
        zero: list[float] = [0.0] * (dim or 0)
        return [v if v is not None else list(zero) for v in result]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string (``input_type='query'``).

        Parameters
        ----------
        text:
            Query string.

        Returns
        -------
        list[float]
            Embedding vector.
        """
        if not text.strip():
            raise ValueError("Query text must not be empty.")
        vecs = self._embed_with_retry([text], input_type="query")
        return vecs[0]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_with_retry(
        self,
        texts: list[str],
        input_type: str,
        max_retries: int = 1,
    ) -> list[list[float]]:
        """Call Voyage embed API with one retry on failure.

        Rate-limit errors (HTTP 429) trigger a 60-second back-off before the
        single retry. All other errors are re-raised after the retry attempt.
        """
        for attempt in range(max_retries + 1):
            try:
                response = self._client.embed(
                    texts,
                    model=self.model,
                    input_type=input_type,
                )
                return response.embeddings
            except Exception as exc:
                if attempt < max_retries:
                    # Heuristic: rate-limit messages typically contain "429" or "rate"
                    if "429" in str(exc) or "rate" in str(exc).lower():
                        time.sleep(60)
                    # Other transient errors: short wait
                    else:
                        time.sleep(2)
                else:
                    raise
        # Unreachable — kept for type checker
        raise RuntimeError("embed retry exhausted")

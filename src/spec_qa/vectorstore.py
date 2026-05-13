"""Chroma persistent vector store wrapper for 3GPP spec chunks.

Stores and queries Chunk embeddings in a local ChromaDB database using cosine
similarity.  Metadata values are constrained to str/int/float/bool because
ChromaDB does not accept list or dict types in metadata fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb

from spec_qa.parser import Chunk

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryResult:
    """A single result returned by :meth:`ChromaSpecStore.query`."""

    chunk_id: str
    text: str
    metadata: dict
    distance: float


# ---------------------------------------------------------------------------
# ChromaSpecStore
# ---------------------------------------------------------------------------


class ChromaSpecStore:
    """Chroma-backed vector store for 3GPP spec chunks.

    Parameters
    ----------
    db_path:
        Directory where ChromaDB will persist its data.
    collection_name:
        Name of the Chroma collection to use (created if absent).
    """

    # Chroma recommends batches ≤ 5 000 to avoid memory pressure
    _UPSERT_BATCH = 5_000

    def __init__(
        self,
        db_path: Path = Path("./chroma_db"),
        collection_name: str = "specs_3gpp",
    ) -> None:
        self._client = chromadb.PersistentClient(path=str(db_path))
        # cosine distance is more appropriate for normalised text embeddings
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Upsert *chunks* and their corresponding *embeddings* into Chroma.

        Parameters
        ----------
        chunks:
            Chunk objects to store. Their ``chunk_id`` is used as the Chroma
            document ID.
        embeddings:
            One embedding vector per chunk (must be the same length as
            *chunks*).

        Raises
        ------
        ValueError
            If the lengths of *chunks* and *embeddings* differ.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must have the same length."
            )

        if not chunks:
            return

        # Build flat lists for Chroma; process in batches to cap memory usage
        for start in range(0, len(chunks), self._UPSERT_BATCH):
            batch_chunks = chunks[start : start + self._UPSERT_BATCH]
            batch_embeddings = embeddings[start : start + self._UPSERT_BATCH]

            ids: list[str] = []
            documents: list[str] = []
            metadatas: list[dict] = []

            for chunk in batch_chunks:
                ids.append(chunk.chunk_id)
                documents.append(chunk.text)
                # chunk.metadata() already returns only str/int/float/bool values;
                # heading_path is pre-joined as "heading_path_str" (see parser.py).
                # We still sanitise to be safe: convert None → "" and any other
                # unsupported type → str.
                raw_meta = chunk.metadata()
                metadatas.append(_sanitise_metadata(raw_meta))

            self._collection.upsert(
                ids=ids,
                embeddings=batch_embeddings,
                documents=documents,
                metadatas=metadatas,
            )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 8,
        where: dict | None = None,
    ) -> list[QueryResult]:
        """Return the *top_k* nearest chunks to *query_embedding*.

        Parameters
        ----------
        query_embedding:
            Dense vector (same dimension as stored embeddings).
        top_k:
            Number of results to return.
        where:
            Optional Chroma metadata filter, e.g.
            ``{"spec_no": "38.331"}`` or ``{"$and": [...]}``.

        Returns
        -------
        list[QueryResult]
            Ordered from most to least similar (lowest cosine distance first).
        """
        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        raw = self._collection.query(**kwargs)

        results: list[QueryResult] = []
        # Chroma returns nested lists (one list per query); we sent exactly one
        for chunk_id, doc, meta, dist in zip(
            raw["ids"][0],
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ):
            results.append(
                QueryResult(
                    chunk_id=chunk_id,
                    text=doc,
                    metadata=meta,
                    distance=dist,
                )
            )
        return results

    def count(self) -> int:
        """Return the total number of stored chunks."""
        return self._collection.count()

    def reset(self) -> None:
        """Delete all documents from the collection (collection itself is kept)."""
        # Retrieve all IDs and delete them; get() with no args returns everything
        all_ids = self._collection.get(include=[])["ids"]
        if all_ids:
            self._collection.delete(ids=all_ids)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sanitise_metadata(meta: dict) -> dict:
    """Ensure all metadata values are Chroma-compatible (str/int/float/bool).

    Chroma raises a validation error for None, list, or dict values.
    """
    clean: dict = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif v is None:
            clean[k] = ""
        elif isinstance(v, list):
            # Encode list as a delimited string; caller can split on "||" to recover
            clean[k] = "||".join(str(item) for item in v)
        else:
            clean[k] = str(v)
    return clean

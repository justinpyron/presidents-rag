"""Shared API request/response models for Presidents RAG."""

from pydantic import BaseModel

CHUNK_KEY_DELIMITER = "$"


class RetrievedChunk(BaseModel):
    """A single document chunk returned by retrieval/reranking.

    Identity is the ``(source, document, start_index)`` triple. ``chunk_id`` is
    the DB primary key, carried along for eval joins/debugging but not relied
    upon.
    """

    source: str
    document: str
    start_index: int
    text: str
    retrieval_score: float | None = None
    rerank_score: float | None = None
    chunk_id: int | None = None


def chunk_key(chunk: "RetrievedChunk") -> str:
    """Stable, embedding-model-independent identifier for a logical chunk.

    Joins the ``(source, document, start_index)`` identity with
    ``CHUNK_KEY_DELIMITER`` so eval gold labels remain valid across embedding
    models, whose DB primary keys differ for the same logical chunk.
    """
    return CHUNK_KEY_DELIMITER.join(
        [chunk.source, chunk.document, str(chunk.start_index)]
    )


class RetrieveRequest(BaseModel):
    query: str
    top_k: int
    sources: list[str] | None = None


class RerankRequest(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
    top_k: int

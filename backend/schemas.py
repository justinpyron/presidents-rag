"""Shared API request/response models for Presidents RAG."""

from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    """A single document chunk returned by retrieval/reranking.

    Identity is the ``(source, start_index)`` pair. ``chunk_id`` is the DB
    primary key, carried along for eval joins/debugging but not relied upon.
    """

    source: str
    start_index: int
    text: str
    retrieval_score: float | None = None
    rerank_score: float | None = None
    chunk_id: int | None = None


class RetrieveRequest(BaseModel):
    query: str
    vector_store_config_id: int
    top_k: int
    source: str | None = None


class RerankRequest(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
    top_k: int

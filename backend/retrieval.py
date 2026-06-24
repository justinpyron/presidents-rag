"""Shared retrieval logic for Presidents RAG.

These functions are intentionally pure: every heavy dependency (DB session,
embedding model, cross-encoder) is passed in rather than loaded internally, so
the same code path runs in the Modal server and in the eval harness.

Heavy imports (torch, sqlalchemy, db models) are deferred into the function
bodies so this module can be imported cheaply -- e.g. the Streamlit client only
needs ``RetrievedChunk`` for deserialization, not torch or SQLAlchemy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from sqlalchemy.orm import Session


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


class QueryRequest(BaseModel):
    query: str
    vector_store_config_id: int
    top_k: int
    source: str | None = None


class RerankRequest(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
    top_k: int


def retrieve(
    session: "Session",
    embedder: "SentenceTransformer",
    vector_store_config_id: int,
    query: str,
    top_k: int,
    source: str | None = None,
) -> list[RetrievedChunk]:
    """Embed ``query`` and return the ``top_k`` nearest chunks by cosine distance.

    ``retrieval_score`` is reported as cosine similarity (``1 - distance``), so
    higher means more similar.
    """
    import torch
    from sqlalchemy import select

    from db.models import ChunkDim384

    with torch.no_grad():
        query_embedding = embedder.encode([query])[0]
    if hasattr(query_embedding, "tolist"):
        query_embedding = query_embedding.tolist()

    distance = ChunkDim384.embedding.cosine_distance(query_embedding)
    statement = select(ChunkDim384, distance.label("distance")).where(
        ChunkDim384.vector_store_config_id == vector_store_config_id
    )
    if source is not None:
        statement = statement.where(ChunkDim384.source == source)
    statement = statement.order_by(distance).limit(top_k)

    rows = session.execute(statement).all()
    return [
        RetrievedChunk(
            source=chunk.source,
            start_index=chunk.start_index,
            text=chunk.text,
            retrieval_score=1.0 - float(distance_value),
            chunk_id=chunk.id,
        )
        for chunk, distance_value in rows
    ]


def rerank(
    cross_encoder: "CrossEncoder",
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int,
) -> list[RetrievedChunk]:
    """Reorder ``chunks`` by cross-encoder relevance and keep the top ``top_k``.

    Operates on whole ``RetrievedChunk`` objects so text/scores can never
    desync, and records each chunk's ``rerank_score``.
    """
    if not chunks:
        return []

    ranks = cross_encoder.rank(query, [chunk.text for chunk in chunks])
    return [
        chunks[row["corpus_id"]].model_copy(
            update={"rerank_score": float(row["score"])}
        )
        for row in ranks[:top_k]
    ]

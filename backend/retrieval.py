"""Shared retrieval logic for Presidents RAG.

These functions are intentionally pure: every heavy dependency (DB session,
embedding model, cross-encoder) is passed in rather than loaded internally, so
the same code path runs in the Modal server and in the eval harness.

torch (a heavy dependency) is imported lazily inside ``retrieve`` since it is
only needed when actually embedding a query.
"""

from sentence_transformers import CrossEncoder, SentenceTransformer
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.schemas import RetrievedChunk
from db.models import ChunkDim384


def retrieve(
    session: Session,
    embedder: SentenceTransformer,
    vector_store_config_id: int,
    query: str,
    top_k: int,
    sources: list[str] | None = None,
) -> list[RetrievedChunk]:
    """Embed ``query`` and return the ``top_k`` nearest chunks by cosine distance.

    ``retrieval_score`` is reported as cosine similarity (``1 - distance``), so
    higher means more similar. When ``sources`` is set, only chunks from those
    collections are considered; ``None`` searches the full knowledge base.
    """
    import torch

    if sources is not None and not sources:
        return []

    with torch.no_grad():
        q_emb = embedder.encode([query])[0].tolist()

    stmt = select(
        ChunkDim384,
        ChunkDim384.embedding.cosine_distance(q_emb).label("distance"),
    ).where(ChunkDim384.vector_store_config_id == vector_store_config_id)
    if sources is not None:
        stmt = stmt.where(ChunkDim384.source.in_(sources))
    stmt = stmt.order_by("distance").limit(top_k)

    rows = session.execute(stmt).all()
    return [
        RetrievedChunk(
            source=chunk.source,
            document=chunk.document,
            start_index=chunk.start_index,
            text=chunk.text,
            retrieval_score=1.0 - float(distance),
            chunk_id=chunk.id,
        )
        for chunk, distance in rows
    ]


def rerank(
    cross_encoder: CrossEncoder,
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

    ranks = cross_encoder.rank(query, [c.text for c in chunks])
    return [
        chunks[row["corpus_id"]].model_copy(
            update={"rerank_score": float(row["score"])}
        )
        for row in ranks[:top_k]
    ]

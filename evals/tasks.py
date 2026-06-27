"""Task factories for Pydantic Evals experiments."""

from collections.abc import Callable

from backend.schemas import RetrievedChunk
from frontend.client import RAGClient


def make_retrieval_task(
    top_k_retrieval: int,
    top_k_rerank: int,
) -> Callable[[str], list[RetrievedChunk]]:
    """Return a task that runs retrieve → rerank for a single query."""
    rag_client = RAGClient()

    def task(query: str) -> list[RetrievedChunk]:
        chunks = rag_client.retrieve(query, top_k=top_k_retrieval)
        return rag_client.rerank(query, chunks, top_k_rerank)

    return task

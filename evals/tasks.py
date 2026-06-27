"""Task factories for Pydantic Evals experiments."""

from collections.abc import Callable

from pydantic_ai.usage import UsageLimits

from backend.schemas import RetrievedChunk
from evals.schemas import GenerationResult
from frontend.agent import REQUEST_LIMIT, AgentDeps, agent, parse_agent_run
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


def make_generation_task(
    top_k_retrieval: int,
    top_k_rerank: int,
) -> Callable[[str], GenerationResult]:
    """Return a task that runs the agentic RAG loop for a single query."""
    rag_client = RAGClient()

    def task(query: str) -> GenerationResult:
        deps = AgentDeps(
            client=rag_client,
            top_k_retrieval=top_k_retrieval,
            top_k_rerank=top_k_rerank,
        )
        result = agent.run_sync(
            query,
            deps=deps,
            usage_limits=UsageLimits(request_limit=REQUEST_LIMIT),
        )
        answer, cited_chunks = parse_agent_run(result, deps)
        return GenerationResult(
            answer=answer,
            retrieved_chunks=list(deps.seen.values()),
            cited_chunks=cited_chunks,
        )

    return task

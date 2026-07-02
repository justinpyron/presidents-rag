"""Task factories for Pydantic Evals experiments."""

from collections.abc import Callable

from pydantic_ai.usage import UsageLimits

from backend.schemas import chunk_key
from evals.schemas import GenerationResult
from frontend.agent import AgentDeps, agent, parse_agent_run, request_limit_for
from frontend.client import RAGClient


def make_retrieval_task(
    top_k_retrieval: int,
    top_k_rerank: int,
) -> Callable[[str], list[str]]:
    """Return a task that runs retrieve → rerank for a single query.

    Chunks are identified by a stable ``source$document$start_index`` key so
    gold labels stay valid across embedding models (whose DB primary keys
    differ for the same logical chunk).
    """
    rag_client = RAGClient()

    def task(query: str) -> list[str]:
        chunks = rag_client.retrieve(query, top_k=top_k_retrieval)
        ranked = rag_client.rerank(query, chunks, top_k_rerank)
        return [chunk_key(chunk) for chunk in ranked]

    return task


def make_generation_task(
    model: str,
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
            model=model,
            usage_limits=UsageLimits(
                request_limit=request_limit_for(deps.max_searches)
            ),
        )
        answer, cited_chunks = parse_agent_run(result, deps)
        return GenerationResult(
            answer=answer,
            retrieved_chunks=list(deps.seen.values()),
            cited_chunks=cited_chunks,
        )

    return task

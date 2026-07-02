"""Agentic RAG loop for Presidents RAG, built on Pydantic AI.

Where ``RAGClient.ask`` is a fixed ``retrieve -> rerank -> generate`` pipeline,
this module hands control to the model: it decides when to search, what to
search for, and how many searches to issue, using a single retrieval tool.

Design notes:
- Sync throughout (``run_sync``); parallel tool calls are disabled so the model
  issues one retrieval at a time against the vector DB.
- Every chunk the tool returns is accumulated on the run's ``deps`` (keyed by
  ``chunk_id``), giving a faithful record of everything seen regardless of what
  the model echoes back.
- The structured output cites supporting chunks by ``chunk_id`` (the
  ``ChunkDim384.id`` primary key) rather than re-emitting text, so citations
  can't drift from the source. ``parse_agent_run`` resolves those ids back to
  full ``RetrievedChunk`` objects for the UI.
"""

from dataclasses import dataclass, field

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.run import AgentRunResult
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

from backend.schemas import RetrievedChunk
from frontend.client import RAGClient

DEFAULT_MODEL = "openai:gpt-5.4-mini"
REQUEST_LIMIT = 8
DEFAULT_TOP_K_RETRIEVAL = 50
DEFAULT_TOP_K_RERANK = 10

SYSTEM_PROMPT = """
You are a research assistant answering questions about US Presidents and
Secretaries of State. You have one tool, `search_knowledge_base`, which takes a
search query and returns relevant document chunks from a knowledge
base. Each chunk has a `chunk_id` and a `source` collection label.

How to work:
- Decide how much effort the question needs. A simple, single-fact question may
  only require one search; a harder question may require several.
- For multi-hop questions, search step by step: use what you learn from one
  search to form the next query.
- For questions that compare or aggregate independent facts, search for each
  fact separately, then combine the results.
- Inspect what you get back. If the chunks don't actually support an answer,
  rewrite your query and search again, but only up to 3 times. If after reasonable
  effort (up to 3 searches using the search_knowledge_base tool), the knowledge
  base still doesn't contain the answer, stop searching and say you don't
  know rather than guessing.

Final answer:
- Ground your answer strictly in the retrieved chunks. Hallucinating answers
  is strictly forbidden.
- Set `supporting_chunk_ids` to the `chunk_id`s of the chunks that actually
  support your answer (omit chunks you retrieved but didn't rely on).
""".strip()


class ToolChunk(BaseModel):
    """A retrieved chunk as exposed to the model.

    Carries ``chunk_id`` (so the model can cite it) and ``text`` (so it can
    read it). Full metadata/scores stay on the server-side ``RetrievedChunk``.
    """

    chunk_id: int
    source: str
    text: str


class AgentResponse(BaseModel):
    """Structured output of the agentic RAG run."""

    answer: str = Field(
        description=(
            "The final natural-language answer to the user's question, "
            "grounded strictly in the retrieved chunks. If the knowledge base "
            "does not contain enough information to answer, say so here instead "
            "of guessing."
        ),
    )
    supporting_chunk_ids: list[int] = Field(
        description=(
            "The chunk_id values of the retrieved chunks that actually support "
            "the answer. Include only chunks you relied on, omitting any that "
            "were retrieved but unused. Leave empty if no chunk supports the "
            "answer."
        ),
    )


@dataclass
class AgentDeps:
    """Per-run dependencies and state.

    ``sources`` is the per-turn retrieval policy (set before each run).
    ``seen`` accumulates every chunk returned by the tool across the whole
    conversation, keyed by ``chunk_id`` so repeats dedupe naturally.
    """

    client: RAGClient
    sources: list[str] | None = None
    top_k_retrieval: int = DEFAULT_TOP_K_RETRIEVAL
    top_k_rerank: int = DEFAULT_TOP_K_RERANK
    seen: dict[int, RetrievedChunk] = field(default_factory=dict)


agent = Agent(
    model=DEFAULT_MODEL,
    deps_type=AgentDeps,
    output_type=AgentResponse,
    model_settings=ModelSettings(
        parallel_tool_calls=False,  # Agent is synchronous atm
        thinking="medium",
        max_tokens=20_000,  # must exceed Anthropic thinking budget
    ),
    system_prompt=SYSTEM_PROMPT,
    capabilities=[Instrumentation()],
)


@agent.tool
def search_knowledge_base(
    ctx: RunContext[AgentDeps],
    query: str,
) -> list[ToolChunk]:
    """Search the knowledge base and return the most relevant chunks.

    Args:
        query: A focused natural-language search query.
    """
    deps = ctx.deps
    chunks = deps.client.retrieve(
        query,
        top_k=deps.top_k_retrieval,
        sources=deps.sources,
    )
    top = deps.client.rerank(query, chunks, deps.top_k_rerank)
    for chunk in top:
        if chunk.chunk_id is not None:
            deps.seen[chunk.chunk_id] = chunk
    return [
        ToolChunk(chunk_id=c.chunk_id, source=c.source, text=c.text)
        for c in top
        if c.chunk_id is not None
    ]


def parse_agent_run(
    result: AgentRunResult[AgentResponse],
    deps: AgentDeps,
) -> tuple[str, list[RetrievedChunk]]:
    """Resolve the answer and cited chunks for display."""
    response = result.output
    chunks = [
        deps.seen[chunk_id]
        for chunk_id in response.supporting_chunk_ids
        if chunk_id in deps.seen
    ]
    return response.answer, chunks

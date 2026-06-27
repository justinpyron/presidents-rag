"""Shared types for Pydantic Evals experiments."""

from pydantic import BaseModel

from backend.schemas import RetrievedChunk


class GenerationResult(BaseModel):
    """Output of a single agentic RAG eval run."""

    answer: str
    retrieved_chunks: list[RetrievedChunk]
    cited_chunks: list[RetrievedChunk]

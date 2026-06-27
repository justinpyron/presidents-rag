"""Evaluators for retrieval and generation experiments."""

from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from backend.schemas import RetrievedChunk


def _hits_at_k(
    ranked: list[RetrievedChunk],
    relevant_chunk_ids: list[int] | None,
    k: int,
) -> tuple[int, int]:
    """Return the number of relevant hits and the size of the gold set."""
    relevant = set(relevant_chunk_ids or [])
    top_k_ids = {
        chunk.chunk_id for chunk in ranked[:k] if chunk.chunk_id is not None
    }
    return len(relevant & top_k_ids), len(relevant)


@dataclass
class RecallAtK(Evaluator[str, list[RetrievedChunk]]):
    k: int

    def get_default_evaluation_name(self) -> str:
        return f"recall@{self.k}"

    def evaluate(
        self, ctx: EvaluatorContext[str, list[RetrievedChunk]]
    ) -> float:
        hits, num_relevant = _hits_at_k(
            ctx.output, ctx.expected_output, self.k
        )
        if num_relevant == 0:
            return 0.0
        return hits / num_relevant


@dataclass
class PrecisionAtK(Evaluator[str, list[RetrievedChunk]]):
    k: int

    def get_default_evaluation_name(self) -> str:
        return f"precision@{self.k}"

    def evaluate(
        self, ctx: EvaluatorContext[str, list[RetrievedChunk]]
    ) -> float:
        hits, _ = _hits_at_k(ctx.output, ctx.expected_output, self.k)
        return hits / self.k


@dataclass
class HitAtK(Evaluator[str, list[RetrievedChunk]]):
    k: int

    def get_default_evaluation_name(self) -> str:
        return f"hit@{self.k}"

    def evaluate(
        self, ctx: EvaluatorContext[str, list[RetrievedChunk]]
    ) -> bool:
        hits, num_relevant = _hits_at_k(
            ctx.output, ctx.expected_output, self.k
        )
        if num_relevant == 0:
            return False
        return hits > 0

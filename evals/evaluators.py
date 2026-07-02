"""Evaluators for retrieval and generation experiments."""

from dataclasses import dataclass, field

from pydantic_ai import models
from pydantic_ai.settings import ModelSettings
from pydantic_evals.evaluators import (
    EvaluationReason,
    Evaluator,
    EvaluatorContext,
)
from pydantic_evals.evaluators.llm_as_a_judge import (
    GradingOutput,
    judge_input_output,
    judge_input_output_expected,
)

from backend.schemas import RetrievedChunk
from evals.schemas import GenerationResult

# ===========================================================================
# Retrieval
# ===========================================================================


def _hits_at_k(
    ranked_chunk_keys: list[str],
    relevant_chunk_keys: list[str] | None,
    k: int,
) -> tuple[int, int]:
    """Return the number of relevant hits and the size of the gold set."""
    relevant = set(relevant_chunk_keys or [])
    top_k_keys = set(ranked_chunk_keys[:k])
    return len(relevant & top_k_keys), len(relevant)


@dataclass
class RecallAtK(Evaluator[str, list[str]]):
    k: int

    def get_default_evaluation_name(self) -> str:
        return f"recall@{self.k}"

    def evaluate(self, ctx: EvaluatorContext[str, list[str]]) -> float:
        hits, num_relevant = _hits_at_k(
            ctx.output, ctx.expected_output, self.k
        )
        if num_relevant == 0:
            return 0.0
        return hits / num_relevant


@dataclass
class PrecisionAtK(Evaluator[str, list[str]]):
    k: int

    def get_default_evaluation_name(self) -> str:
        return f"precision@{self.k}"

    def evaluate(self, ctx: EvaluatorContext[str, list[str]]) -> float:
        hits, _ = _hits_at_k(ctx.output, ctx.expected_output, self.k)
        return hits / self.k


@dataclass
class HitAtK(Evaluator[str, list[str]]):
    k: int

    def get_default_evaluation_name(self) -> str:
        return f"hit@{self.k}"

    def evaluate(self, ctx: EvaluatorContext[str, list[str]]) -> bool:
        hits, num_relevant = _hits_at_k(
            ctx.output, ctx.expected_output, self.k
        )
        if num_relevant == 0:
            return False
        return hits > 0


# ===========================================================================
# Generation
# ===========================================================================


def _faithfulness_inputs(question: str, chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        documents = "No documents retrieved"
    else:
        documents = "\n\n".join(
            f"### Document {chunk.chunk_id}\n{chunk.text}" for chunk in chunks
        )
    return f"## Question\n{question}\n\n## Retrieved documents\n{documents}"


def _grading_to_reason(grading: GradingOutput) -> EvaluationReason:
    return EvaluationReason(value=grading.pass_, reason=grading.reason)


DEFAULT_RELEVANCE_RUBRIC = (
    "The response directly addresses the question that was asked. "
    "If the response states that the available information is insufficient to answer "
    "(e.g. it does not know, or the knowledge base does not contain the answer), that "
    "counts as addressing the question."
)
DEFAULT_CORRECTNESS_RUBRIC = (
    "The response is correct because it is consistent with the expected answer. "
    "Minor wording differences are acceptable."
)
DEFAULT_FAITHFULNESS_RUBRIC = (
    "Every factual claim in the response must be supported by the retrieved documents. "
    "Claims drawn from general knowledge or speculation beyond the retrieved documents "
    "fail. If the response declines to answer because the retrieved documents do not support "
    "an answer, pass when it makes no unsupported factual claims about the subject. "
    "Meta-statements about insufficient evidence (e.g. 'the documents don't say') do "
    "not need to be literally quoted from a chunk."
)


@dataclass
class Relevance(Evaluator[str, GenerationResult]):
    """On-topic? Does the answer address the question that was asked?"""

    rubric: str = DEFAULT_RELEVANCE_RUBRIC
    model: models.Model | models.KnownModelName | str | None = None
    model_settings: ModelSettings | None = None

    async def evaluate(
        self, ctx: EvaluatorContext[str, GenerationResult]
    ) -> EvaluationReason:
        grading = await judge_input_output(
            ctx.inputs,
            ctx.output.answer,
            self.rubric,
            self.model,
            self.model_settings,
        )
        return _grading_to_reason(grading)


@dataclass
class Correctness(Evaluator[str, GenerationResult]):
    """Right vs. gold? Is the answer consistent with the expected answer?"""

    rubric: str = DEFAULT_CORRECTNESS_RUBRIC
    model: models.Model | models.KnownModelName | str | None = None
    model_settings: ModelSettings | None = None

    async def evaluate(
        self, ctx: EvaluatorContext[str, GenerationResult]
    ) -> EvaluationReason:
        if ctx.expected_output is None:
            return EvaluationReason(
                value=False,
                reason="No expected output provided for this case.",
            )
        grading = await judge_input_output_expected(
            ctx.inputs,
            ctx.output.answer,
            ctx.expected_output,
            self.rubric,
            self.model,
            self.model_settings,
        )
        return _grading_to_reason(grading)


@dataclass
class Faithfulness(Evaluator[str, GenerationResult]):
    """Grounded in retrieved context? Is every claim supported by the docs?"""

    rubric: str = DEFAULT_FAITHFULNESS_RUBRIC
    model: models.Model | models.KnownModelName | str | None = None
    model_settings: ModelSettings | None = field(default=None)

    async def evaluate(
        self, ctx: EvaluatorContext[str, GenerationResult]
    ) -> EvaluationReason:
        grading = await judge_input_output(
            _faithfulness_inputs(ctx.inputs, ctx.output.retrieved_chunks),
            ctx.output.answer,
            self.rubric,
            self.model,
            self.model_settings,
        )
        return _grading_to_reason(grading)

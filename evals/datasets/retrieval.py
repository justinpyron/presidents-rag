"""Retrieval eval cases for retrieve → rerank."""

from pydantic_evals import Case, Dataset

from backend.schemas import RetrievedChunk

retrieval_dataset = Dataset[str, list[RetrievedChunk]](
    name="retrieval",
    cases=[
        Case(
            name="case_01",
            inputs="Who did Lincoln promote after learning he had no presidential ambitions?",
            expected_output=[108],
        ),
        Case(
            name="case_02",
            inputs="How did Andrew Jackson's wife die?",
            expected_output=[275],
        ),
    ],
)

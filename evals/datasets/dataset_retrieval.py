"""Retrieval eval cases for retrieve → rerank."""

from pydantic_evals import Case, Dataset

from evals.evaluators import HitAtK, PrecisionAtK, RecallAtK

retrieval_dataset = Dataset[str, list[str]](
    name="retrieval",
    cases=[
        Case(
            name="case_01",
            inputs="Who did Lincoln promote after learning he had no presidential ambitions?",
            expected_output=[
                "wikipedia$abraham_lincoln.txt$55825"
            ],  # TODO: Verify this output
        ),
        Case(
            name="case_02",
            inputs="How did Andrew Jackson's wife die?",
            expected_output=[
                "wikipedia$andrew_jackson.txt$28233"
            ],  # TODO: Verify this output
        ),
    ],
    evaluators=[
        RecallAtK(k=10),
        PrecisionAtK(k=10),
        HitAtK(k=10),
    ],
)

# TODO: Expand this stubbed dataset with more cases.

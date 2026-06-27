"""Generation eval cases for the agentic RAG loop."""

from pydantic_ai.settings import ModelSettings
from pydantic_evals import Case, Dataset

from evals.evaluators import Correctness, Faithfulness, Relevance
from evals.schemas import GenerationResult

JUDGE_MODEL = "openai:gpt-5.4-mini"
JUDGE_MODEL_SETTINGS = ModelSettings(temperature=0)

generation_dataset = Dataset[str, GenerationResult](
    name="generation",
    cases=[
        Case(
            name="case_01",
            inputs="Who did Lincoln promote after learning he had no presidential ambitions?",
            expected_output="Lincoln promoted Ulysses S. Grant.",
        ),
        Case(
            name="case_02",
            inputs="How did Andrew Jackson's wife die?",
            expected_output="Rachel Jackson died of a stroke or heart attack.",
        ),
    ],
    evaluators=[
        Relevance(model=JUDGE_MODEL, model_settings=JUDGE_MODEL_SETTINGS),
        Correctness(model=JUDGE_MODEL, model_settings=JUDGE_MODEL_SETTINGS),
        Faithfulness(model=JUDGE_MODEL, model_settings=JUDGE_MODEL_SETTINGS),
    ],
)

# TODO: Expand this stubbed dataset with more cases.

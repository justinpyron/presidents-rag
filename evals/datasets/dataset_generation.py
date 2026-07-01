"""Generation eval cases for the agentic RAG loop."""

from pydantic_ai.settings import ModelSettings
from pydantic_evals import Case, Dataset

from evals.evaluators import Correctness, Faithfulness, Relevance
from evals.schemas import GenerationResult

# TODO: Use a stronger model for evaluation (e.g. gpt-5.4)
JUDGE_MODEL = "openai:gpt-5.4-mini"
JUDGE_MODEL_SETTINGS = ModelSettings(temperature=0)

CASES_SINGLEHOP = [
    Case(
        name="singlehop_01",
        inputs="Who did Lincoln promote after learning he had no presidential ambitions?",
        expected_output="Lincoln promoted Ulysses S. Grant.",
    ),
    Case(
        name="singlehop_02",
        inputs="How did Andrew Jackson's wife die?",
        expected_output="Rachel Jackson died of either a stroke or a heart attack.",
    ),
    Case(
        name="singlehop_03",
        inputs="How did Nixon pay for law school?",
        expected_output="Nixon received a scholarship from the Duke University School of Law.",
    ),
    Case(
        name="singlehop_04",
        inputs="How did Kennedy prepare for his inaugural address?",
        expected_output="Kennedy carefully studied famous American speeches, such as the Gettysburg Address, and copied their terse, vivid style.",
    ),
    Case(
        name="singlehop_05",
        inputs="How did Gerald Ford signal his openness to running with Reagan on the 1980 presidential ticket?",
        expected_output="Ford discussed the idea of running with Reagan on the 1980 ticket during an interview with Walter Cronkite.",
    ),
]

CASES_MULTIHOP = [
    Case(
        name="multihop_01",
        inputs="On what date was the president that Mark Twain helped via a generous book deal born?",
        expected_output="April 27, 1822",
    ),
    Case(
        name="multihop_02",
        inputs="What physical activities were typically included in the routine (while he was in office) of the president who ordered the atomic bombing of Japan?",
        expected_output="While in the White House, Truman's routine included a one or two mile walk and swimming laps in the White House pool.",
    ),
    Case(
        name="multihop_03",
        inputs="Which presidents have been governor of the state where Nixon was born?",
        expected_output="Ronald Reagan",
    ),
    Case(
        name="multihop_04",
        inputs="On which dates did the presidents who graduated from Princeton University die?",
        expected_output="James Madison died on June 28, 1836, and Woodrow Wilson died on February 3, 1924.",
    ),
    Case(
        name="multihop_05",
        inputs="Who was the last president born in Ohio?",
        expected_output="Warren G. Harding was the last president born in Ohio.",
    ),
]

CASES_UNANSWERABLE = [
    Case(
        name="unanswerable_01",
        inputs="How old was Millard Fillmore when he said his first words?",
        expected_output="The knowledge base does not contain information about Millard Fillmore's first words.",
    ),
    Case(
        name="unanswerable_02",
        inputs="What was the name of George W. Bush's first dog?",
        expected_output="The knowledge base does not contain information about George W. Bush's first dog.",
    ),
    Case(
        name="unanswerable_03",
        inputs="What was Zachary Taylor's favorite food as a teenager?",
        expected_output="The knowledge base does not contain information about Zachary Taylor's favorite food as a teenager.",
    ),
    Case(
        name="unanswerable_04",
        inputs="What was Jimmy Carter's view of the Boston Red Sox?",
        expected_output="The knowledge base does not contain information about Jimmy Carter's view of the Boston Red Sox.",
    ),
    Case(
        name="unanswerable_05",
        inputs="Which novel did Taft read most during his college years?",
        expected_output="The knowledge base does not contain information about which novel Taft read most during his college years.",
    ),
]


generation_dataset = Dataset[str, GenerationResult](
    name="generation",
    cases=CASES_SINGLEHOP + CASES_MULTIHOP + CASES_UNANSWERABLE,
    evaluators=[
        Relevance(model=JUDGE_MODEL, model_settings=JUDGE_MODEL_SETTINGS),
        Correctness(model=JUDGE_MODEL, model_settings=JUDGE_MODEL_SETTINGS),
        Faithfulness(model=JUDGE_MODEL, model_settings=JUDGE_MODEL_SETTINGS),
    ],
)

# TODO: Add efficiency metrics
# TODO: Add an evaluator that measures the number of tool calls made
# TODO: Add an evaluator that measures the number of tokens used / cost
# TODO: Add an evaluator that measures latency

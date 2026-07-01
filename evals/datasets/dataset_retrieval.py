"""Retrieval eval cases for retrieve → rerank."""

from pydantic_evals import Case, Dataset

from evals.evaluators import HitAtK, PrecisionAtK, RecallAtK

CASES_SEMANTIC = [
    Case(
        name="semantic_01",
        # Who did Lincoln promote after learning he had no presidential ambitions?
        inputs="lincoln who promoted no presidential ambitions",
        expected_output=[
            "wikipedia$abraham_lincoln.txt$55825",
        ],
    ),
    Case(
        name="semantic_02",
        # How did Andrew Jackson's wife die?
        inputs="andrew jackson wife death",
        expected_output=[
            "wikipedia$andrew_jackson.txt$28233",
        ],
    ),
    Case(
        name="semantic_03",
        # Which president was married to a woman whose death was potentially caused by rumors that she cheated on her previous husband?
        inputs="president wife died rumors cheating previous husband",
        expected_output=[
            "wikipedia$andrew_jackson.txt$28233",
        ],
    ),
    Case(
        name="semantic_04",
        # Who was almost ousted from the presidency in the middle of the nineteenth century due to his cabinet members all quitting in a staged, organized manner?
        inputs="19th century president cabinet all resigned sequentially",
        expected_output=[
            "wikipedia$john_tyler.txt$38479",
        ],
    ),
    Case(
        name="semantic_05",
        # Who did not actively (but perhaps did passively) seek a 2nd term because he knew he had a terminal illness?
        inputs="president terminal illness second term reelection",
        expected_output=[
            "miller_center$21_chester_a_arthur_5_life-after-the-presidency.txt$0",
            "miller_center$21_chester_a_arthur_2_campaigns-and-elections.txt$3476",
        ],
    ),
    Case(
        name="semantic_06",
        # Who sent a military force to China to save diplomats from a violent group of rebels who had killed Christians?
        inputs="military force china save diplomats rebels killed christians",
        expected_output=[
            "miller_center$25_william_mckinley_4_foreign-affairs.txt$8642",
            "miller_center$25_william_mckinley_0_life-in-brief.txt$3081",
            "wikipedia$william_mckinley.txt$49967",
        ],
    ),
    Case(
        name="semantic_07",
        # Which action taken by the USSR regarding a commercial jet from an east asian nation resulted in the deaths of nearly three hundred civilians?
        inputs="ussr commercial jet east asia 300 civilians killed",
        expected_output=[
            "miller_center$40_ronald_reagan_4_foreign-affairs.txt$11546",
        ],
    ),
    Case(
        name="semantic_08",
        # What was the act that Adams signed into law whose passage was facilitated by anti-French feelings after a foreign controversy?
        inputs="john adams act signed anti-french foreign controversy",
        expected_output=[
            "miller_center$02_john_adams_3_domestic-affairs.txt$505",
            "wikipedia$john_adams.txt$51038",
        ],
    ),
    Case(
        name="semantic_09",
        # Who advocated for the citizenship of native americans during his inaugural address?
        inputs="president native american citizenship inaugural address",
        expected_output=[
            "miller_center$18_ulysses_s_grant_3_domestic-affairs.txt$8749",
            "wikipedia$ulysses_s_grant.txt$45911",
        ],
    ),
    Case(
        name="semantic_10",
        # Which president was accused of being elitist due to his son traveling to Europe and meeting with royalty?
        inputs="president elitist son europe royalty",
        expected_output=[
            "miller_center$08_martin_van_buren_6_family-life.txt$747",
        ],
    ),
]


CASES_LEXICAL = [
    Case(
        name="lexical_01",
        inputs="Which president went to Plymouth elementary school?",
        expected_output=[
            "miller_center$30_calvin_coolidge_1_life-before-the-presidency.txt$392",
        ],
    ),
    Case(
        name="lexical_02",
        inputs="Which president signed the Kellogg-Briand Pact?",
        expected_output=[
            "miller_center$30_calvin_coolidge_4_foreign-affairs.txt$1062",
            "miller_center$30_calvin_coolidge_0_life-in-brief.txt$4045",
            "miller_center$30_calvin_coolidge_8_impact-and-legacy.txt$720",
        ],
    ),
    Case(
        name="lexical_03",
        inputs="Who was almost forced to resign on September 11, 1841?",
        expected_output=[
            "wikipedia$john_tyler.txt$38479",
        ],
    ),
    Case(
        name="lexical_04",
        inputs="Which organization that opposed overseas territorial expansion of the United States did Mark Twain and Andrew Carnegie belong to?",
        expected_output=[
            "miller_center$25_william_mckinley_4_foreign-affairs.txt$1042",
            "wikipedia$william_jennings_bryan.txt$18241",
        ],
    ),
    Case(
        name="lexical_05",
        inputs="Who signed the Surface Mining Control and Reclamation Act?",
        expected_output=[
            "miller_center$39_jimmy_carter_3_domestic-affairs.txt$4322",
        ],
    ),
    Case(
        name="lexical_06",
        inputs="Who did Jefferson write a letter to on September 23, 1800?",
        expected_output=[
            "wikipedia$thomas_jefferson.txt$97469",
        ],
    ),
    Case(
        name="lexical_07",
        inputs="What did the Organization of Eastern Caribbean States ask Reagan to do?",
        expected_output=[
            "miller_center$40_ronald_reagan_4_foreign-affairs.txt$15751",
            "miller_center$40_ronald_reagan_4_foreign-affairs.txt$16349",
        ],
    ),
    Case(
        name="lexical_08",
        inputs="Who wrote 'Maine and her soil, or BLOOD!'?",
        expected_output=[
            "wikipedia$martin_van_buren.txt$41446",
        ],
    ),
    Case(
        name="lexical_09",
        inputs="Who believed the ghost of assassinated president William McKinley had directed him to kill Roosevelt?",
        expected_output=[
            "wikipedia$theodore_roosevelt.txt$69119",
        ],
    ),
    Case(
        name="lexical_10",
        inputs="Which president had a brother Vivian and sister Mary Jane?",
        expected_output=[
            "miller_center$33_harry_s_truman_1_life-before-the-presidency.txt$0",
            "miller_center$33_harry_s_truman_1_life-before-the-presidency.txt$2028",
            "miller_center$33_harry_s_truman_6_family-life.txt$727",
            "wikipedia$harry_s_truman.txt$35",
            "wikipedia$harry_s_truman.txt$2451",
            "wikipedia$harry_s_truman.txt$12339",
        ],
    ),
]

retrieval_dataset = Dataset[str, list[str]](
    name="retrieval",
    # cases=CASES_SEMANTIC + CASES_LEXICAL,
    cases=CASES_SEMANTIC,
    evaluators=[
        RecallAtK(k=10),
        PrecisionAtK(k=10),
        HitAtK(k=10),
    ],
)

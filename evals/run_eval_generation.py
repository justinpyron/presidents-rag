"""Run generation evals against the generation dataset."""

import argparse
import sys

import logfire
from dotenv import load_dotenv
from pydantic_ai import Agent

from evals.datasets.dataset_generation import generation_dataset
from evals.tasks import make_generation_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run generation evals.")
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Pydantic AI model string for the agent (e.g. openai:gpt-5.4-mini).",
    )
    parser.add_argument(
        "--top-k-retrieval",
        "-k1",
        type=int,
        dest="top_k_retrieval",
        help="Number of chunks to retrieve from the vector database.",
    )
    parser.add_argument(
        "--top-k-rerank",
        "-k2",
        type=int,
        dest="top_k_rerank",
        help="Number of chunks to keep after reranking.",
    )
    parser.add_argument(
        "--max-concurrency",
        "-c",
        type=int,
        default=50,
        dest="max_concurrency",
        help="Maximum number of eval cases to run concurrently.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    logfire.configure(
        service_name="presidents-rag-evals",
        environment="dev",
    )
    # Instrument every pydantic_ai Agent, including the internal judge agents
    # used by pydantic_evals' LLM-as-a-judge helpers, so their token usage and
    # cost show up in the Logfire UI.
    Agent.instrument_all()

    args = parse_args()
    task = make_generation_task(
        args.model,
        args.top_k_retrieval,
        args.top_k_rerank,
    )
    report = generation_dataset.evaluate_sync(
        task,
        max_concurrency=args.max_concurrency,
        metadata={
            "model": args.model,
            "top_k_retrieval": args.top_k_retrieval,
            "top_k_rerank": args.top_k_rerank,
        },
    )
    report.print()


if __name__ == "__main__":
    sys.exit(main() or 0)

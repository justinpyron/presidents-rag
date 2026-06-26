"""Run retrieval evals against the retrieval dataset."""

import argparse
import sys

import logfire
from dotenv import load_dotenv

from evals.datasets.retrieval import retrieval_dataset
from evals.tasks import make_retrieval_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval evals.")
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
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    logfire.configure(
        service_name="presidents-rag-evals",
        environment="dev",
    )

    args = parse_args()
    task = make_retrieval_task(args.top_k_retrieval, args.top_k_rerank)
    report = retrieval_dataset.evaluate_sync(
        task,
        metadata={
            "top_k_retrieval": args.top_k_retrieval,
            "top_k_rerank": args.top_k_rerank,
        },
    )
    report.print()


if __name__ == "__main__":
    sys.exit(main() or 0)

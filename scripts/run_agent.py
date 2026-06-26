"""Run the agentic RAG agent interactively from the terminal."""

import os
import sys

import logfire
from dotenv import load_dotenv
from pydantic_ai.usage import UsageLimits

from frontend.agent import REQUEST_LIMIT, AgentDeps, agent, parse_agent_run
from frontend.client import RAGClient

load_dotenv()

logfire.configure(
    service_name="presidents-rag",
    environment=os.getenv("LOGFIRE_ENV", "dev"),
)


def main() -> None:
    try:
        query = input("Ask a question: ").strip()
        print()
    except (EOFError, KeyboardInterrupt):
        print()
        return

    if not query:
        return

    deps = AgentDeps(
        client=RAGClient(),
        top_k_retrieval=50,
        top_k_rerank=10,
    )
    result = agent.run_sync(
        query,
        deps=deps,
        usage_limits=UsageLimits(request_limit=REQUEST_LIMIT),
    )
    answer, chunks = parse_agent_run(result, deps)
    print(answer)

    if chunks:
        print(f"\nSources ({len(chunks)}):")
        for i, chunk in enumerate(chunks, start=1):
            print(f"\n[{i}] {chunk.source} (char {chunk.start_index})")
            print(chunk.text)


if __name__ == "__main__":
    sys.exit(main() or 0)

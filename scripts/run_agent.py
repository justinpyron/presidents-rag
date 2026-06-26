"""Run the agentic RAG agent interactively from the terminal."""

import os
import sys

import logfire
from dotenv import load_dotenv
from pydantic_ai.usage import UsageLimits

from backend.schemas import RetrievedChunk
from frontend.agent import REQUEST_LIMIT, AgentDeps, agent, parse_agent_run
from frontend.client import RAGClient

load_dotenv()

logfire.configure(
    service_name="presidents-rag",
    environment=os.getenv("LOGFIRE_ENV", "dev"),
)

INDENT = " " * 4
DIVIDER = "─" * 80


def _print_role(emoji: str, role: str, text: str) -> None:
    print(f"{emoji} {role}")
    for line in text.splitlines():
        print(f"{INDENT}{line}")


def _print_sources(chunks: list[RetrievedChunk]) -> None:
    print(f"📚 Sources ({len(chunks)})")
    for i, chunk in enumerate(chunks, start=1):
        header = f"[{i}] {chunk.source} (char {chunk.start_index})"
        print(f"{INDENT}{header}")
        for line in chunk.text.splitlines():
            print(f"{INDENT * 2}{line}")


def main() -> None:
    deps = AgentDeps(
        client=RAGClient(),
        top_k_retrieval=50,
        top_k_rerank=10,
    )
    messages = None

    while True:
        try:
            query = input("Ask a question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query:
            break

        print()
        _print_role("👤", "User", query)
        print()

        result = agent.run_sync(
            query,
            deps=deps,
            message_history=messages,
            usage_limits=UsageLimits(request_limit=REQUEST_LIMIT),
        )
        messages = result.all_messages()
        answer, chunks = parse_agent_run(result, deps)

        _print_role("🤖", "AI", answer)

        if chunks:
            print()
            _print_sources(chunks)

        print()
        print(DIVIDER)
        print()


if __name__ == "__main__":
    sys.exit(main() or 0)

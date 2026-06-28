"""Server-side conversation state and the bridge to the agentic RAG loop.

Each browser conversation maps to a :class:`Conversation` holding the Pydantic
AI message history and the run's ``AgentDeps`` (which accumulates every chunk
the agent has seen). Keeping this server-side means the rich, non-JSON state
— message history and ``RetrievedChunk`` objects — never has to round-trip
through the browser; the client only carries an opaque conversation id.

A new user message launches a fresh agentic loop via ``agent.run_sync``,
mirroring ``scripts/run_agent.py`` but with the chosen model and prior history.
"""

import threading
from dataclasses import dataclass, field

from pydantic_ai.usage import UsageLimits

from backend.schemas import RetrievedChunk
from frontend.agent import REQUEST_LIMIT, AgentDeps, agent, parse_agent_run
from frontend.client import RAGClient
from frontend.dash_app.config import (
    ABSTAIN_TEXT,
    ERROR_TEXT,
    source_collection_name,
)


def prettify_source(source: str) -> str:
    """Turn a ``source`` slug into a display name.

    e.g. ``andrew_johnson.txt`` -> ``Andrew Johnson``.
    """
    name = source[:-4] if source.lower().endswith(".txt") else source
    return name.replace("_", " ").strip().title()


@dataclass
class SourceView:
    """A retrieved chunk prepared for display under an answer."""

    n: int
    name: str
    collection: str
    excerpt: str

    @classmethod
    def from_chunk(cls, n: int, chunk: RetrievedChunk) -> "SourceView":
        return cls(
            n=n,
            name=prettify_source(chunk.document),
            collection=source_collection_name(chunk.source),
            excerpt=chunk.text.strip(),
        )


@dataclass
class DisplayMessage:
    """A single rendered turn in the transcript."""

    id: int
    role: str  # "user" | "assistant"
    text: str = ""
    sources: list[SourceView] = field(default_factory=list)
    abstain: bool = False
    error: bool = False


@dataclass
class Conversation:
    """All state for one chat thread."""

    deps: AgentDeps
    messages: list = field(default_factory=list)  # Pydantic AI message history
    display: list[DisplayMessage] = field(default_factory=list)
    _next_id: int = 0

    def next_id(self) -> int:
        self._next_id += 1
        return self._next_id


class ConversationStore:
    """In-memory registry of conversations, keyed by an opaque id.

    The :class:`RAGClient` is shared across conversations (it only holds HTTP
    and model clients); each conversation gets its own ``AgentDeps`` so the
    accumulated ``seen`` chunks never leak between threads.
    """

    def __init__(self) -> None:
        self._client = RAGClient()
        self._conversations: dict[str, Conversation] = {}
        self._lock = threading.Lock()

    def _create(self) -> Conversation:
        return Conversation(deps=AgentDeps(client=self._client))

    def get(self, conv_id: str) -> Conversation:
        """Return the conversation for ``conv_id``, creating it if needed."""
        with self._lock:
            conv = self._conversations.get(conv_id)
            if conv is None:
                conv = self._create()
                self._conversations[conv_id] = conv
            return conv

    def has_messages(self, conv_id: str | None) -> bool:
        """Whether a conversation exists and has any turns (without creating it)."""
        conv = self._conversations.get(conv_id) if conv_id else None
        return bool(conv and conv.display)

    def reset(self, conv_id: str) -> None:
        """Forget a conversation so the next message starts a clean thread."""
        with self._lock:
            self._conversations.pop(conv_id, None)

    def add_user_message(self, conv_id: str, text: str) -> None:
        conv = self.get(conv_id)
        conv.display.append(
            DisplayMessage(id=conv.next_id(), role="user", text=text)
        )

    def run_assistant(
        self,
        conv_id: str,
        query: str,
        model_id: str,
        sources: list[str],
    ) -> None:
        """Run one agentic loop and append the assistant's turn.

        Failures (e.g. a missing provider API key) are caught and surfaced as a
        friendly error message rather than crashing the callback.
        """
        conv = self.get(conv_id)
        conv.deps.sources = sources
        try:
            result = agent.run_sync(
                query,
                deps=conv.deps,
                message_history=conv.messages,
                model=model_id,
                usage_limits=UsageLimits(request_limit=REQUEST_LIMIT),
            )
            conv.messages = result.all_messages()
            answer, chunks = parse_agent_run(result, conv.deps)
            sources = [
                SourceView.from_chunk(i, c) for i, c in enumerate(chunks, 1)
            ]
            conv.display.append(
                DisplayMessage(
                    id=conv.next_id(),
                    role="assistant",
                    text=answer or ABSTAIN_TEXT,
                    sources=sources,
                    abstain=not sources,
                )
            )
        except Exception:
            conv.display.append(
                DisplayMessage(
                    id=conv.next_id(),
                    role="assistant",
                    text=ERROR_TEXT,
                    error=True,
                )
            )


# A single store instance backs the whole app.
store = ConversationStore()

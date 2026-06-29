"""Server-side chat state and the bridge to the agentic RAG loop.

Each browser chat maps to a :class:`Chat` holding the Pydantic
AI message history and the run's ``AgentDeps`` (which accumulates every chunk
the agent has seen). Keeping this server-side means the rich, non-JSON state
— message history and ``RetrievedChunk`` objects — never has to round-trip
through the browser; the client only carries an opaque chat id.

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


def _prettify_miller_center_document(name: str) -> str:
    """Format a Miller Center filename stem as a readable title.

    Mirrors ``build_filename`` in ``scripts/scrape_miller_center.py``:
    ``{number}_{president}_{subpage_index}_{subpage}``
    """
    parts = name.split("_")
    if len(parts) < 4 or not parts[0].isdigit() or not parts[-2].isdigit():
        return name.replace("_", " ").replace("-", " ").strip().title()

    president = "_".join(parts[1:-2])
    subpage = parts[-1]
    text = f"{president} {subpage}".replace("_", " ").replace("-", " ")
    return text.strip().title()


def prettify_source(document: str, collection: str | None = None) -> str:
    """Turn a document filename into a display name.

    Wikipedia: ``andrew_johnson.txt`` -> ``Andrew Johnson``.

    Miller Center: ``10_john_tyler_5_life-after-the-presidency.txt`` ->
    ``John Tyler Life After The Presidency``.
    """
    name = document[:-4] if document.lower().endswith(".txt") else document
    if collection == "miller_center":
        return _prettify_miller_center_document(name)
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
            name=prettify_source(chunk.document, chunk.source),
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
class Chat:
    """All state for one chat thread."""

    deps: AgentDeps
    messages: list = field(default_factory=list)  # Pydantic AI message history
    display: list[DisplayMessage] = field(default_factory=list)
    _next_id: int = 0

    def next_id(self) -> int:
        self._next_id += 1
        return self._next_id


class ChatStore:
    """In-memory registry of chats, keyed by an opaque id.

    The :class:`RAGClient` is shared across chats (it only holds HTTP
    and model clients); each chat gets its own ``AgentDeps`` so the
    accumulated ``seen`` chunks never leak between threads.
    """

    def __init__(self) -> None:
        self._client = RAGClient()
        self._chats: dict[str, Chat] = {}
        self._lock = threading.Lock()

    def _create(self) -> Chat:
        return Chat(deps=AgentDeps(client=self._client))

    def get(self, chat_id: str) -> Chat:
        """Return the chat for ``chat_id``, creating it if needed."""
        with self._lock:
            chat = self._chats.get(chat_id)
            if chat is None:
                chat = self._create()
                self._chats[chat_id] = chat
            return chat

    def has_messages(self, chat_id: str | None) -> bool:
        """Whether a chat exists and has any turns (without creating it)."""
        chat = self._chats.get(chat_id) if chat_id else None
        return bool(chat and chat.display)

    def is_server_healthy(self) -> bool:
        """Whether the retrieval server is warm and ready to serve requests."""
        return self._client.is_healthy()

    def reset(self, chat_id: str) -> None:
        """Forget a chat so the next message starts a clean thread."""
        with self._lock:
            self._chats.pop(chat_id, None)

    def add_user_message(self, chat_id: str, text: str) -> None:
        chat = self.get(chat_id)
        chat.display.append(
            DisplayMessage(id=chat.next_id(), role="user", text=text)
        )

    def run_assistant(
        self,
        chat_id: str,
        query: str,
        model_id: str,
        sources: list[str],
    ) -> None:
        """Run one agentic loop and append the assistant's turn.

        Failures (e.g. a missing provider API key) are caught and surfaced as a
        friendly error message rather than crashing the callback.
        """
        chat = self.get(chat_id)
        chat.deps.sources = sources
        try:
            result = agent.run_sync(
                query,
                deps=chat.deps,
                message_history=chat.messages,
                model=model_id,
                usage_limits=UsageLimits(request_limit=REQUEST_LIMIT),
            )
            chat.messages = result.all_messages()
            answer, chunks = parse_agent_run(result, chat.deps)
            sources = [
                SourceView.from_chunk(i, c) for i, c in enumerate(chunks, 1)
            ]
            chat.display.append(
                DisplayMessage(
                    id=chat.next_id(),
                    role="assistant",
                    text=answer or ABSTAIN_TEXT,
                    sources=sources,
                    abstain=not sources,
                )
            )
        except Exception:
            chat.display.append(
                DisplayMessage(
                    id=chat.next_id(),
                    role="assistant",
                    text=ERROR_TEXT,
                    error=True,
                )
            )


# A single store instance backs the whole app.
store = ChatStore()

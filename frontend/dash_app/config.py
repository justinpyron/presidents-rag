"""Static configuration for the Presidential Archive app.

Copy, the model registry, the source collections and all element ids live
here so components and callbacks share one vocabulary.
"""

# --- Copy --------------------------------------------------------------------
APP_NAME = "Presidential Archive"
GITHUB_URL = "https://github.com/justinpyron/presidents-rag"

ABOUT_LEAD = (
    "Presidential Archive is an agentic RAG system. "
    "Ask a question and an AI agent composes its own searches over a knowledge base of "
    "the U.S. presidents, reads what it finds, and writes an answer you can trace back to its sources."
)
ABOUT_BUILT_WITH = "Built with Pydantic AI & Dash."

WELCOME_HEADING = "A reference desk for American presidents"
WELCOME_SUBTEXT = "Ask anything about the presidents. An answer grounded in reliable sources is returned."
EXAMPLE_PROMPTS = [
    "Who succeeded the president who purchased Alaska?",
    "Which U.S. president was the tallest?",
    "Compare the Monroe and Truman doctrines.",
]

COMPOSER_PLACEHOLDER = "Ask a question…"
LOADING_TEXT = "Searching the archive"
ABSTAIN_TEXT = "No grounded sources matched this question."
ERROR_TEXT = (
    "Something went wrong while searching the archive. Please try again — and "
    "if you chose a non-OpenAI model, make sure its API key is configured."
)

# --- Models ------------------------------------------------------------------
# The picker overrides the model per run. ``id`` is the Pydantic AI model
# string passed to ``agent.run_sync(model=...)``.
DEFAULT_MODEL_ID = "openai:gpt-5.4-mini"

MODELS = [
    {
        "id": "anthropic:claude-haiku-4-5",
        "name": "Claude Haiku 4.5",
        "desc": "Anthropic",
    },
    {
        "id": DEFAULT_MODEL_ID,
        "name": "GPT-5.4 mini",
        "desc": "OpenAI",
    },
    {
        "id": "google:gemini-3.5-flash",
        "name": "Gemini 3.5 Flash",
        "desc": "Google",
    },
]


def model_name(model_id: str) -> str:
    """Human-facing name for a model id (falls back to the id itself)."""
    return next((m["name"] for m in MODELS if m["id"] == model_id), model_id)


# --- Source collections ------------------------------------------------------
SOURCES = [
    {
        "id": "wikipedia",
        "name": "Wikipedia",
        "desc": "Crowd-sourced encyclopedia articles",
    },
    {
        "id": "miller_center",
        "name": "Miller Center",
        "desc": "Scholarly essays from the University of Virginia",
    },
]
DEFAULT_SOURCE_IDS = [s["id"] for s in SOURCES]
SOURCE_NAMES = {s["id"]: s["name"] for s in SOURCES}


def source_collection_name(source_id: str) -> str:
    """Human-facing name for a collection id."""
    return SOURCE_NAMES.get(source_id, source_id.replace("_", " ").title())


def sources_chip_label(selected_ids: list[str]) -> str:
    """Composer chip text for the current source selection."""
    if len(selected_ids) == len(SOURCES):
        return "Sources · All"
    if len(selected_ids) == 1:
        return f"Sources · {source_collection_name(selected_ids[0])}"
    return f"Sources · {len(selected_ids)}"


# --- Element ids -------------------------------------------------------------
class ID:
    # stores
    CHAT = "chat-store"  # active chat id
    MODEL = "model-store"  # selected model id
    SOURCES = "sources-store"  # selected source collection ids
    PENDING = "pending-store"  # in-flight query (drives the agent run)
    POPOVER = "popover-store"  # name of the open popover, or None
    SCROLL_DUMMY = "scroll-dummy"  # autoscroll callback sink
    RESIZE_DUMMY = "resize-dummy"  # textarea autosize callback sink
    SERVER_POLL = "server-poll"  # interval that polls server health

    # layout
    CHAT_SCROLL = "chat-scroll"
    CHAT_CONTENT = "chat-content"
    BACKDROP = "backdrop"

    # header
    SERVER_STATUS = "server-status"  # warm-up indicator container
    ABOUT_TRIGGER = "about-trigger"
    ABOUT_POP = "about-pop"
    NEW_TRIGGER = "new-trigger"
    NEWCONFIRM_POP = "newconfirm-pop"
    CANCEL_NEW = "cancel-new"
    CONFIRM_NEW = "confirm-new"

    # composer
    INPUT = "composer-input"
    SEND = "send-btn"
    MODEL_TRIGGER = "model-trigger"
    MODEL_CHIP_LABEL = "model-chip-label"
    MODEL_POP = "model-pop"
    MODEL_MENU = "model-menu"
    SOURCES_TRIGGER = "sources-trigger"
    SOURCES_CHIP_LABEL = "sources-chip-label"
    SOURCES_POP = "sources-pop"
    SOURCES_MENU = "sources-menu"


# Popover names (values stored in ID.POPOVER)
POP_ABOUT = "about"
POP_NEWCONFIRM = "newconfirm"
POP_MODEL = "model"
POP_SOURCES = "sources"

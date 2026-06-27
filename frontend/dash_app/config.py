"""Static configuration for the Presidential Archive app.

Copy, the model registry, the source collections and all element ids live
here so components and callbacks share one vocabulary.
"""

from frontend.agent import MODEL

# --- Copy --------------------------------------------------------------------
APP_NAME = "Presidential Archive"
GITHUB_URL = "https://github.com/justinpyron/presidents-rag"

ABOUT_LEAD = (
    "Presidential Archive is a personal project exploring agentic "
    "retrieval-augmented generation. Ask a question and the agent composes its "
    "own searches over an encyclopedia of the U.S. presidents, reads what it "
    "finds, and writes an answer you can trace back to its sources."
)
ABOUT_BUILT_WITH = "Built with Pydantic AI & Dash."

WELCOME_HEADING = "A reference desk for the U.S. presidents"
WELCOME_SUBTEXT = (
    "Ask anything about the presidents. Every answer is drawn from encyclopedia "
    "articles, with its sources kept close at hand."
)
EXAMPLE_PROMPTS = [
    "Who succeeded the president who purchased Alaska?",
    "Which U.S. president was the tallest?",
    "Compare the Monroe and Truman doctrines.",
]

COMPOSER_PLACEHOLDER = "Ask about any U.S. president…"
DISCLAIMER = (
    "Answers are drawn from encyclopedia articles. Verify important facts."
)
LOADING_TEXT = "Searching the archive"
ABSTAIN_TEXT = "No grounded sources matched this question."
ERROR_TEXT = (
    "Something went wrong while searching the archive. Please try again — and "
    "if you chose a non-OpenAI model, make sure its API key is configured."
)

# --- Models ------------------------------------------------------------------
# The picker overrides the model per run. ``id`` is the Pydantic AI model
# string passed to ``agent.run_sync(model=...)``. The default tracks the model
# the agent is configured with, so the app works out of the box with only an
# OpenAI key; Claude and Gemini require ANTHROPIC_API_KEY / GEMINI_API_KEY.
MODELS = [
    {
        "id": "anthropic:claude-sonnet-4-5",
        "name": "Claude Sonnet 4.5",
        "desc": "Balanced reasoning & speed",
    },
    {
        "id": MODEL,  # "openai:gpt-5.4-mini"
        "name": "GPT-5.4 mini",
        "desc": "OpenAI flagship, fast & grounded",
    },
    {
        "id": "google-gla:gemini-2.5-pro",
        "name": "Gemini 2.5 Pro",
        "desc": "Long-context reasoning",
    },
]
DEFAULT_MODEL_ID = MODEL


def model_name(model_id: str) -> str:
    """Human-facing name for a model id (falls back to the id itself)."""
    return next((m["name"] for m in MODELS if m["id"] == model_id), model_id)


# --- Source collections ------------------------------------------------------
# The knowledge base is Wikipedia-only, so there is a single, always-on
# collection. The picker exists to mirror the design and make the scope of the
# archive explicit.
SOURCES = [
    {
        "id": "wikipedia",
        "name": "Wikipedia",
        "desc": "Crowd-sourced encyclopedia articles",
    },
]
SOURCE_LABEL = "Wikipedia"  # shown on each source card's collection tag


# --- Element ids -------------------------------------------------------------
class ID:
    # stores
    CONVERSATION = "conv-store"  # active conversation id
    MODEL = "model-store"  # selected model id
    PENDING = "pending-store"  # in-flight query (drives the agent run)
    POPOVER = "popover-store"  # name of the open popover, or None
    SCROLL_DUMMY = "scroll-dummy"  # autoscroll callback sink
    RESIZE_DUMMY = "resize-dummy"  # textarea autosize callback sink

    # layout
    CHAT_SCROLL = "chat-scroll"
    CHAT_CONTENT = "chat-content"
    BACKDROP = "backdrop"

    # header
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
    SOURCES_POP = "sources-pop"


# Popover names (values stored in ID.POPOVER)
POP_ABOUT = "about"
POP_NEWCONFIRM = "newconfirm"
POP_MODEL = "model"
POP_SOURCES = "sources"

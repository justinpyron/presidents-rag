"""Entry point for the Presidential Archive Dash app.

Run locally with::

    uv run python -m frontend.dash_app.main

The agent talks to the retrieval server over HTTP (``SERVER_URL``) and to the
model providers directly, so the usual environment variables must be set (see
``.env``): ``OPENAI_API_KEY`` / ``SERVER_URL`` are required, and Claude/Gemini
in the model picker additionally need ``ANTHROPIC_API_KEY`` / ``GEMINI_API_KEY``.
"""

import os
from pathlib import Path

import logfire
from dash import Dash, dcc, html
from dotenv import load_dotenv

from frontend.dash_app import theme as t
from frontend.dash_app.components.composer import composer
from frontend.dash_app.components.header import header
from frontend.dash_app.components.welcome import welcome
from frontend.dash_app.config import (
    APP_NAME,
    DEFAULT_MODEL_ID,
    DEFAULT_SOURCE_IDS,
    ID,
)

# Configure logfire before the app is built so the agent's instrumentation
# (frontend.agent's Instrumentation capability) is exported on every run.
# Runs at import time so it applies under both `python -m` and a WSGI server.
load_dotenv()
logfire.configure(
    service_name="presidents-rag",
    environment=os.getenv("LOGFIRE_ENV", "dev"),
)

FONTS = (
    "https://fonts.googleapis.com/css2?"
    "family=Public+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400"
    "&family=Spectral:wght@400;500;600;700&display=swap"
)

app = Dash(
    __name__,
    title=APP_NAME,
    assets_folder=str(Path(__file__).parent / "assets"),
    external_stylesheets=[FONTS],
    suppress_callback_exceptions=True,
)
server = app.server  # for WSGI deployment (e.g. gunicorn)


def _stores() -> list:
    return [
        dcc.Store(id=ID.CONVERSATION, data=None),
        dcc.Store(id=ID.MODEL, data=DEFAULT_MODEL_ID),
        dcc.Store(id=ID.SOURCES, data=DEFAULT_SOURCE_IDS),
        dcc.Store(id=ID.PENDING, data=None),
        dcc.Store(id=ID.POPOVER, data=None),
        dcc.Store(id=ID.SCROLL_DUMMY),
        dcc.Store(id=ID.RESIZE_DUMMY),
    ]


app.layout = html.Div(
    style={
        "height": "100vh",
        "display": "flex",
        "flexDirection": "column",
        "background": t.PAPER,
        "fontFamily": t.SANS,
        "color": t.INK,
    },
    children=[
        header(),
        html.Div(
            id=ID.CHAT_SCROLL,
            style={"flex": "1", "overflow": "auto"},
            children=html.Div(id=ID.CHAT_CONTENT, children=welcome()),
        ),
        composer(DEFAULT_MODEL_ID, DEFAULT_SOURCE_IDS),
        # Full-screen click target that closes any open popover.
        html.Div(
            id=ID.BACKDROP,
            n_clicks=0,
            className="hidden",
            style={"position": "fixed", "inset": "0", "zIndex": 30},
        ),
        *_stores(),
    ],
)

# Importing the callbacks module registers every callback on the app.
from frontend.dash_app import callbacks  # noqa: E402,F401

if __name__ == "__main__":
    app.run(
        host=os.getenv("DASH_HOST", "127.0.0.1"),
        port=int(os.getenv("DASH_PORT", "8050")),
        debug=bool(os.getenv("DASH_DEBUG", "1") == "1"),
    )

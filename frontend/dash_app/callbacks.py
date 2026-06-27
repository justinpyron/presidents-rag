"""All Dash callbacks for the Presidential Archive app.

The chat round-trip is deliberately split into two server callbacks so the
loading indicator can render between them:

1. ``submit`` appends the user's message, shows the "Searching the archive"
   bubble and stores the query in ``pending-store``.
2. ``run_agent`` fires off the change to ``pending-store``, runs the agentic
   loop (blocking), then swaps in the answer and clears the pending flag.

Pure presentation — popovers, source expand/collapse, autoscroll and textarea
autosize — is handled clientside so it stays instant.
"""

import uuid

from dash import (
    ALL,
    MATCH,
    Input,
    Output,
    State,
    callback,
    clientside_callback,
    ctx,
)
from dash.exceptions import PreventUpdate

from frontend.dash_app.components.composer import model_menu
from frontend.dash_app.components.conversation import render_conversation
from frontend.dash_app.components.welcome import welcome
from frontend.dash_app.config import (
    DEFAULT_MODEL_ID,
    EXAMPLE_PROMPTS,
    ID,
    POP_ABOUT,
    POP_MODEL,
    POP_NEWCONFIRM,
    POP_SOURCES,
    model_name,
)
from frontend.dash_app.services.conversation import store


# --- Sending a message -------------------------------------------------------
@callback(
    Output(ID.CHAT_CONTENT, "children"),
    Output(ID.INPUT, "value"),
    Output(ID.PENDING, "data"),
    Output(ID.CONVERSATION, "data"),
    Output(ID.POPOVER, "data", allow_duplicate=True),
    Input(ID.SEND, "n_clicks"),
    Input({"type": "example-prompt", "index": ALL}, "n_clicks"),
    State(ID.INPUT, "value"),
    State(ID.CONVERSATION, "data"),
    State(ID.MODEL, "data"),
    State(ID.PENDING, "data"),
    prevent_initial_call=True,
)
def submit(send_clicks, _example_clicks, text, conv_id, model_id, pending):
    if pending:  # a run is already in flight; ignore further submits
        raise PreventUpdate

    trigger = ctx.triggered_id
    if isinstance(trigger, dict) and trigger.get("type") == "example-prompt":
        if not ctx.triggered[0]["value"]:  # phantom fire from a re-render
            raise PreventUpdate
        query = EXAMPLE_PROMPTS[trigger["index"]]
    elif trigger == ID.SEND:
        if not send_clicks:
            raise PreventUpdate
        query = text or ""
    else:
        raise PreventUpdate

    query = query.strip()
    if not query:
        raise PreventUpdate

    conv_id = conv_id or uuid.uuid4().hex
    model_id = model_id or DEFAULT_MODEL_ID
    store.add_user_message(conv_id, query)

    conv = store.get(conv_id)
    children = render_conversation(conv, loading=True)
    pending = {"conv_id": conv_id, "query": query, "model": model_id}
    return children, "", pending, conv_id, None


@callback(
    Output(ID.CHAT_CONTENT, "children", allow_duplicate=True),
    Output(ID.PENDING, "data", allow_duplicate=True),
    Input(ID.PENDING, "data"),
    prevent_initial_call=True,
)
def run_agent(pending):
    if not pending:
        raise PreventUpdate
    store.run_assistant(pending["conv_id"], pending["query"], pending["model"])
    conv = store.get(pending["conv_id"])
    return render_conversation(conv, loading=False), None


# --- New conversation --------------------------------------------------------
@callback(
    Output(ID.CHAT_CONTENT, "children", allow_duplicate=True),
    Output(ID.CONVERSATION, "data", allow_duplicate=True),
    Output(ID.PENDING, "data", allow_duplicate=True),
    Output(ID.POPOVER, "data", allow_duplicate=True),
    Input(ID.CONFIRM_NEW, "n_clicks"),
    State(ID.CONVERSATION, "data"),
    prevent_initial_call=True,
)
def confirm_new(n_clicks, conv_id):
    if not n_clicks:
        raise PreventUpdate
    if conv_id:
        store.reset(conv_id)
    return welcome(), uuid.uuid4().hex, None, None


# --- Model selection ---------------------------------------------------------
@callback(
    Output(ID.MODEL, "data"),
    Output(ID.MODEL_CHIP_LABEL, "children"),
    Output(ID.MODEL_MENU, "children"),
    Output(ID.POPOVER, "data", allow_duplicate=True),
    Input({"type": "model-option", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def select_model(_clicks):
    trigger = ctx.triggered_id
    if not isinstance(trigger, dict) or not ctx.triggered[0]["value"]:
        raise PreventUpdate
    model_id = trigger["index"]
    return model_id, model_name(model_id), model_menu(model_id), None


# --- Popover open/close (which popover is open) -------------------------------
@callback(
    Output(ID.POPOVER, "data"),
    Input(ID.ABOUT_TRIGGER, "n_clicks"),
    Input(ID.NEW_TRIGGER, "n_clicks"),
    Input(ID.MODEL_TRIGGER, "n_clicks"),
    Input(ID.SOURCES_TRIGGER, "n_clicks"),
    Input(ID.BACKDROP, "n_clicks"),
    Input(ID.CANCEL_NEW, "n_clicks"),
    State(ID.POPOVER, "data"),
    State(ID.CONVERSATION, "data"),
    prevent_initial_call=True,
)
def toggle_popover(_a, _n, _m, _s, _bd, _cancel, current, conv_id):
    trigger = ctx.triggered_id
    if trigger in (ID.BACKDROP, ID.CANCEL_NEW):
        return None

    if trigger == ID.NEW_TRIGGER:
        # Nothing to clear → no confirmation needed.
        target = POP_NEWCONFIRM if store.has_messages(conv_id) else None
    else:
        target = {
            ID.ABOUT_TRIGGER: POP_ABOUT,
            ID.MODEL_TRIGGER: POP_MODEL,
            ID.SOURCES_TRIGGER: POP_SOURCES,
        }.get(trigger)

    return None if current == target else target


# --- Clientside: popover visibility ------------------------------------------
clientside_callback(
    """
    function(open) {
        const show = (name) => (open === name ? "" : "hidden");
        return [
            show("about"), show("newconfirm"), show("model"),
            show("sources"), open ? "" : "hidden"
        ];
    }
    """,
    Output(ID.ABOUT_POP, "className"),
    Output(ID.NEWCONFIRM_POP, "className"),
    Output(ID.MODEL_POP, "className"),
    Output(ID.SOURCES_POP, "className"),
    Output(ID.BACKDROP, "className"),
    Input(ID.POPOVER, "data"),
)

# --- Clientside: expand / collapse a message's sources -----------------------
clientside_callback(
    """
    function(n) {
        const open = ((n || 0) % 2) === 1;
        return [
            open ? "src-panel" : "src-panel collapsed",
            open ? "caret open" : "caret"
        ];
    }
    """,
    Output({"type": "src-panel", "index": MATCH}, "className"),
    Output({"type": "src-caret", "index": MATCH}, "className"),
    Input({"type": "src-toggle", "index": MATCH}, "n_clicks"),
    prevent_initial_call=True,
)

# --- Clientside: keep the transcript scrolled to the latest turn -------------
clientside_callback(
    """
    function(_children) {
        const el = document.getElementById("chat-scroll");
        if (el) { el.scrollTop = el.scrollHeight; }
        return "";
    }
    """,
    Output(ID.SCROLL_DUMMY, "data"),
    Input(ID.CHAT_CONTENT, "children"),
    prevent_initial_call=True,
)

# --- Clientside: autosize the composer textarea ------------------------------
clientside_callback(
    """
    function(_value) {
        const el = document.getElementById("composer-input");
        if (el) {
            el.style.height = "auto";
            el.style.height = Math.min(el.scrollHeight, 140) + "px";
        }
        return "";
    }
    """,
    Output(ID.RESIZE_DUMMY, "data"),
    Input(ID.INPUT, "value"),
    prevent_initial_call=True,
)

# --- Clientside: send button reflects whether the composer has input ---------
clientside_callback(
    """
    function(value) {
        const hasText = !!(value && value.trim().length > 0);
        return hasText ? "send-btn active" : "send-btn";
    }
    """,
    Output(ID.SEND, "className"),
    Input(ID.INPUT, "value"),
    prevent_initial_call=True,
)

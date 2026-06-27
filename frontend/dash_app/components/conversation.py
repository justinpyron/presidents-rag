"""Renders a conversation transcript: user bubbles, answers and sources."""

from dash import dcc, html

from frontend.dash_app import theme as t
from frontend.dash_app.config import ABSTAIN_TEXT, LOADING_TEXT
from frontend.dash_app.services.conversation import (
    Conversation,
    DisplayMessage,
    SourceView,
)


def _avatar() -> html.Div:
    return html.Div("P", style={**t.logo(27, 7, 14), "marginTop": "2px"})


def _user_message(msg: DisplayMessage) -> html.Div:
    return html.Div(
        style={
            "display": "flex",
            "justifyContent": "flex-end",
            "marginBottom": "26px",
        },
        children=html.Div(
            msg.text,
            style={
                "maxWidth": "78%",
                "background": t.CARD_ALT,
                "border": f"1px solid {t.BORDER}",
                "borderRadius": "16px 16px 5px 16px",
                "padding": "12px 16px",
                "fontFamily": t.SANS,
                "fontSize": "15px",
                "lineHeight": 1.5,
                "color": t.INK,
                "whiteSpace": "pre-wrap",
            },
        ),
    )


def _source_card(src: SourceView) -> html.Div:
    badge = html.Span(
        str(src.n),
        style={
            "flex": "none",
            "width": "18px",
            "height": "18px",
            "borderRadius": "50%",
            "border": "1px solid #cdaaa4",
            "color": t.ACCENT,
            "fontFamily": t.SANS,
            "fontWeight": 600,
            "fontSize": "10px",
            "display": "inline-flex",
            "alignItems": "center",
            "justifyContent": "center",
        },
    )
    return html.Div(
        style={
            "border": f"1px solid {t.BORDER_SOURCE}",
            "borderRadius": "10px",
            "background": t.SOURCE_BG,
            "padding": "13px 15px",
        },
        children=[
            html.Div(
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "9px",
                    "marginBottom": "7px",
                },
                children=[
                    badge,
                    html.Span(
                        src.name,
                        style={
                            "fontFamily": t.SANS,
                            "fontWeight": 700,
                            "fontSize": "11px",
                            "letterSpacing": ".06em",
                            "textTransform": "uppercase",
                            "color": t.ACCENT,
                        },
                    ),
                    html.Span(
                        src.collection,
                        style={
                            "fontFamily": t.SANS,
                            "fontSize": "12px",
                            "color": t.MUTED,
                        },
                    ),
                ],
            ),
            html.Div(
                src.excerpt,
                style={
                    "fontFamily": t.SERIF,
                    "fontSize": "14.5px",
                    "lineHeight": 1.6,
                    "color": t.TAUPE,
                    "whiteSpace": "pre-wrap",
                },
            ),
        ],
    )


def _sources_block(msg: DisplayMessage) -> html.Div:
    caret = html.Span(
        "▸",
        id={"type": "src-caret", "index": msg.id},
        className="caret",
    )
    toggle = html.Div(
        [caret, f" Sources · {len(msg.sources)}"],
        id={"type": "src-toggle", "index": msg.id},
        n_clicks=0,
        style={
            "display": "inline-flex",
            "alignItems": "center",
            "gap": "8px",
            "fontFamily": t.SANS,
            "fontWeight": 600,
            "fontSize": "12.5px",
            "color": t.ACCENT,
            "cursor": "pointer",
        },
    )
    panel = html.Div(
        id={"type": "src-panel", "index": msg.id},
        className="src-panel collapsed",
        children=[_source_card(src) for src in msg.sources],
    )
    return html.Div(
        style={
            "marginTop": "20px",
            "borderTop": f"1px solid {t.BORDER}",
            "paddingTop": "13px",
        },
        children=[toggle, panel],
    )


def _abstain_note() -> html.Div:
    return html.Div(
        [
            html.Span(
                style={
                    "width": "5px",
                    "height": "5px",
                    "borderRadius": "50%",
                    "background": "#c9b89a",
                }
            ),
            f" {ABSTAIN_TEXT}",
        ],
        style={
            "marginTop": "14px",
            "display": "inline-flex",
            "alignItems": "center",
            "gap": "7px",
            "fontFamily": t.SANS,
            "fontSize": "12.5px",
            "color": "#a2937a",
        },
    )


def _assistant_message(msg: DisplayMessage) -> html.Div:
    if msg.error:
        body: list = [
            html.Div(
                msg.text,
                style={
                    "fontFamily": t.SANS,
                    "fontSize": "14.5px",
                    "lineHeight": 1.6,
                    "color": t.MUTED_WARM,
                },
            )
        ]
    else:
        body = [dcc.Markdown(msg.text, className="answer-md")]
        if msg.sources:
            body.append(_sources_block(msg))
        elif msg.abstain:
            body.append(_abstain_note())

    return html.Div(
        style={"display": "flex", "gap": "13px", "marginBottom": "32px"},
        children=[
            _avatar(),
            html.Div(style={"flex": "1", "minWidth": "0"}, children=body),
        ],
    )


def _loading() -> html.Div:
    dots = html.Span(
        style={"display": "inline-flex", "gap": "3px"},
        children=[
            html.Span(
                className="archive-dot",
                style={"animationDelay": f"{delay}s"},
            )
            for delay in (0, 0.15, 0.3)
        ],
    )
    return html.Div(
        style={"display": "flex", "gap": "13px", "marginBottom": "32px"},
        children=[
            _avatar(),
            html.Div(
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "9px",
                    "paddingTop": "4px",
                },
                children=[
                    html.Span(
                        LOADING_TEXT,
                        style={
                            "fontFamily": t.SANS,
                            "fontSize": "15px",
                            "color": t.MUTED,
                        },
                    ),
                    dots,
                ],
            ),
        ],
    )


def render_conversation(conv: Conversation, loading: bool) -> html.Div:
    """Render the full transcript, with a loading bubble when a run is in flight."""
    children = []
    for msg in conv.display:
        if msg.role == "user":
            children.append(_user_message(msg))
        else:
            children.append(_assistant_message(msg))
    if loading:
        children.append(_loading())

    return html.Div(
        style={
            "maxWidth": "760px",
            "margin": "0 auto",
            "padding": "34px 24px 16px",
        },
        children=children,
    )

"""Bottom composer: textarea, model picker, sources picker and send button."""

from dash import dcc, html

from frontend.dash_app import theme as t
from frontend.dash_app.config import (
    COMPOSER_PLACEHOLDER,
    ID,
    MODELS,
    SOURCES,
    model_name,
)


def _up_popover(width: int) -> dict:
    """A popover that opens upward from a composer chip.

    Visibility is controlled by the ``hidden`` class, toggled by the popover
    clientside callback.
    """
    return {
        "position": "absolute",
        "bottom": "calc(100% + 9px)",
        "right": 0,
        "width": f"{width}px",
        "background": t.CARD,
        "border": f"1px solid {t.BORDER_SOFT}",
        "borderRadius": "14px",
        "boxShadow": t.SHADOW_MENU,
        "padding": "7px",
        "zIndex": 60,
    }


def _menu_label(text: str) -> html.Div:
    return html.Div(
        text,
        style={
            "fontFamily": t.SANS,
            "fontWeight": 600,
            "fontSize": "10.5px",
            "letterSpacing": ".08em",
            "textTransform": "uppercase",
            "color": t.MUTED_SOFT,
            "padding": "8px 10px 6px",
        },
    )


def _chip(label_id: str, trigger_id: str, label: str) -> html.Div:
    return html.Div(
        [
            html.Span(label, id=label_id),
            html.Span(" ▾", style={"fontSize": "9px", "color": t.MUTED}),
        ],
        id=trigger_id,
        n_clicks=0,
        style={
            "display": "inline-flex",
            "alignItems": "center",
            "gap": "7px",
            "background": t.CHIP_BG,
            "border": f"1px solid {t.BORDER_SOFT}",
            "borderRadius": "9px",
            "padding": "6px 10px",
            "fontFamily": t.SANS,
            "fontWeight": 500,
            "fontSize": "12.5px",
            "color": t.TAUPE_DEEP,
            "cursor": "pointer",
        },
    )


def model_menu(selected_id: str) -> list:
    """Rows for the model picker; re-rendered on selection to update the tick."""
    rows = []
    for m in MODELS:
        active = m["id"] == selected_id
        rows.append(
            html.Div(
                id={"type": "model-option", "index": m["id"]},
                n_clicks=0,
                className="menu-row",
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "10px",
                    "padding": "9px 10px",
                    "borderRadius": "9px",
                    "cursor": "pointer",
                    "background": "#f0e7d6" if active else "transparent",
                },
                children=[
                    html.Div(
                        style={"flex": "1"},
                        children=[
                            html.Div(
                                m["name"],
                                style={
                                    "fontFamily": t.SANS,
                                    "fontWeight": 600,
                                    "fontSize": "13.5px",
                                    "color": t.INK,
                                },
                            ),
                            html.Div(
                                m["desc"],
                                style={
                                    "fontFamily": t.SANS,
                                    "fontSize": "12px",
                                    "color": t.MUTED,
                                    "marginTop": "1px",
                                },
                            ),
                        ],
                    ),
                    html.Span(
                        "✓" if active else "",
                        style={"color": t.ACCENT, "fontSize": "14px"},
                    ),
                ],
            )
        )
    return rows


def _model_picker(selected_id: str) -> html.Div:
    popover = html.Div(
        id=ID.MODEL_POP,
        className="hidden",
        style={**_up_popover(300)},
        children=[
            _menu_label("Model"),
            html.Div(id=ID.MODEL_MENU, children=model_menu(selected_id)),
        ],
    )
    return html.Div(
        style={"position": "relative"},
        children=[
            _chip(
                ID.MODEL_CHIP_LABEL, ID.MODEL_TRIGGER, model_name(selected_id)
            ),
            popover,
        ],
    )


def _sources_picker() -> html.Div:
    # The knowledge base is Wikipedia-only: one collection, always selected.
    rows = [
        html.Div(
            style={
                "display": "flex",
                "alignItems": "flex-start",
                "gap": "11px",
                "padding": "9px 10px",
                "borderRadius": "9px",
            },
            children=[
                html.Span(
                    "✓",
                    style={
                        "flex": "none",
                        "width": "18px",
                        "height": "18px",
                        "borderRadius": "5px",
                        "background": t.ACCENT,
                        "color": "#fff",
                        "display": "inline-flex",
                        "alignItems": "center",
                        "justifyContent": "center",
                        "fontSize": "11px",
                        "marginTop": "1px",
                    },
                ),
                html.Div(
                    [
                        html.Div(
                            s["name"],
                            style={
                                "fontFamily": t.SANS,
                                "fontWeight": 600,
                                "fontSize": "13.5px",
                                "color": t.INK,
                            },
                        ),
                        html.Div(
                            s["desc"],
                            style={
                                "fontFamily": t.SANS,
                                "fontSize": "12px",
                                "color": t.MUTED,
                                "marginTop": "1px",
                            },
                        ),
                    ]
                ),
            ],
        )
        for s in SOURCES
    ]
    popover = html.Div(
        id=ID.SOURCES_POP,
        className="hidden",
        style={**_up_popover(330)},
        children=[
            _menu_label("Sources"),
            html.Div(
                "Choose which collections the agent searches.",
                style={
                    "fontFamily": t.SANS,
                    "fontSize": "12px",
                    "lineHeight": 1.4,
                    "color": t.MUTED,
                    "padding": "0 10px 8px",
                },
            ),
            *rows,
            html.Div(
                html.Span(
                    f"{len(SOURCES)} of {len(SOURCES)} selected",
                    style={
                        "fontFamily": t.SANS,
                        "fontSize": "12px",
                        "color": t.MUTED,
                    },
                ),
                style={
                    "padding": "8px 10px 4px",
                    "marginTop": "3px",
                    "borderTop": f"1px solid {t.CHIP_BG}",
                },
            ),
        ],
    )
    return html.Div(
        style={"position": "relative"},
        children=[
            _chip(
                f"{ID.SOURCES_TRIGGER}-label",
                ID.SOURCES_TRIGGER,
                "Sources · All",
            ),
            popover,
        ],
    )


def _send_button() -> html.Div:
    return html.Div(
        "↑",
        id=ID.SEND,
        n_clicks=0,
        className="send-btn",
        style={
            "flex": "none",
            "width": "34px",
            "height": "34px",
            "borderRadius": "50%",
            "background": t.ACCENT,
            "color": t.ON_ACCENT,
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center",
            "fontSize": "16px",
            "cursor": "pointer",
        },
    )


def composer(selected_model_id: str) -> html.Div:
    box = html.Div(
        style={
            "position": "relative",
            "background": t.CARD,
            "border": f"1px solid {t.BORDER_SOFT}",
            "borderRadius": "20px",
            "boxShadow": t.SHADOW_COMPOSER,
            "padding": "12px 14px 10px",
        },
        children=[
            dcc.Textarea(
                id=ID.INPUT,
                value="",
                rows=1,
                placeholder=COMPOSER_PLACEHOLDER,
                className="composer-input",
                style={
                    "display": "block",
                    "width": "100%",
                    "border": "none",
                    "outline": "none",
                    "background": "transparent",
                    "resize": "none",
                    "fontFamily": t.SANS,
                    "fontSize": "15px",
                    "lineHeight": 1.5,
                    "color": t.INK,
                    "padding": "3px 4px 0",
                    "minHeight": "24px",
                    "maxHeight": "140px",
                },
            ),
            html.Div(
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "flex-end",
                    "gap": "8px",
                    "marginTop": "10px",
                },
                children=[
                    _model_picker(selected_model_id),
                    _sources_picker(),
                    _send_button(),
                ],
            ),
        ],
    )

    return html.Div(
        style={
            "flex": "none",
            "padding": "22px 24px 18px",
            "background": t.PAPER,
        },
        children=html.Div(
            style={"maxWidth": "760px", "margin": "0 auto"},
            children=box,
        ),
    )

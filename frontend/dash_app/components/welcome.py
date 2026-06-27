"""Empty-state hero shown before the first message."""

from dash import html

from frontend.dash_app import theme as t
from frontend.dash_app.config import (
    EXAMPLE_PROMPTS,
    WELCOME_HEADING,
    WELCOME_SUBTEXT,
)


def _example_card(index: int, prompt: str) -> html.Div:
    return html.Div(
        id={"type": "example-prompt", "index": index},
        n_clicks=0,
        className="example-card",
        style={
            "display": "flex",
            "alignItems": "center",
            "gap": "12px",
            "border": f"1px solid {t.BORDER_SOFT}",
            "background": t.CARD,
            "borderRadius": "12px",
            "padding": "14px 16px",
            "cursor": "pointer",
        },
        children=[
            html.Span(
                prompt,
                style={
                    "flex": "1",
                    "fontFamily": t.SANS,
                    "fontSize": "14.5px",
                    "lineHeight": 1.4,
                    "color": "#4f4534",
                },
            ),
            html.Span("↗", style={"color": t.ACCENT_SOFT, "fontSize": "15px"}),
        ],
    )


def welcome() -> html.Div:
    return html.Div(
        style={
            "minHeight": "100%",
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center",
            "padding": "24px",
        },
        children=html.Div(
            style={
                "width": "100%",
                "maxWidth": "600px",
                "textAlign": "center",
            },
            children=[
                html.Div(
                    "P",
                    style={**t.logo(48, 11, 25), "display": "inline-flex"},
                ),
                html.Div(
                    WELCOME_HEADING,
                    style={
                        "fontFamily": t.SERIF,
                        "fontWeight": 600,
                        "fontSize": "31px",
                        "lineHeight": 1.25,
                        "color": t.INK_STRONG,
                        "marginTop": "18px",
                    },
                ),
                html.Div(
                    WELCOME_SUBTEXT,
                    style={
                        "fontFamily": t.SANS,
                        "fontSize": "15px",
                        "lineHeight": 1.6,
                        "color": t.MUTED_WARM,
                        "margin": "10px auto 0",
                        "maxWidth": "470px",
                    },
                ),
                html.Div(
                    "Try asking",
                    style={
                        "fontFamily": t.SANS,
                        "fontWeight": 600,
                        "fontSize": "11px",
                        "letterSpacing": ".1em",
                        "textTransform": "uppercase",
                        "color": t.MUTED_SOFT,
                        "margin": "34px 0 14px",
                    },
                ),
                html.Div(
                    style={
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "9px",
                        "textAlign": "left",
                    },
                    children=[
                        _example_card(i, prompt)
                        for i, prompt in enumerate(EXAMPLE_PROMPTS)
                    ],
                ),
            ],
        ),
    )

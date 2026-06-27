"""Top bar: brand mark, About popover and New-conversation control."""

from dash import dcc, html

from frontend.dash_app import theme as t
from frontend.dash_app.config import (
    ABOUT_BUILT_WITH,
    ABOUT_LEAD,
    APP_NAME,
    GITHUB_URL,
    ID,
)


def _about_popover() -> html.Div:
    return html.Div(
        id=ID.ABOUT_POP,
        className="hidden",
        style=t.popover(360),
        children=[
            html.Div(
                "About this project",
                style={
                    "fontFamily": t.SERIF,
                    "fontWeight": 600,
                    "fontSize": "16px",
                    "color": t.INK,
                },
            ),
            html.Div(
                ABOUT_LEAD,
                style={
                    "fontFamily": t.SANS,
                    "fontSize": "13.5px",
                    "lineHeight": 1.62,
                    "color": t.TAUPE,
                    "marginTop": "8px",
                },
            ),
            html.Div(
                ABOUT_BUILT_WITH,
                style={
                    "fontFamily": t.SANS,
                    "fontSize": "13px",
                    "color": t.MUTED,
                    "marginTop": "8px",
                },
            ),
            html.A(
                [
                    "View source on GitHub ",
                    html.Span("↗", style={"fontSize": "12px"}),
                ],
                href=GITHUB_URL,
                target="_blank",
                rel="noopener noreferrer",
                style={
                    "display": "inline-flex",
                    "alignItems": "center",
                    "gap": "8px",
                    "marginTop": "15px",
                    "background": t.INK,
                    "color": t.ON_ACCENT,
                    "borderRadius": "9px",
                    "padding": "9px 14px",
                    "fontFamily": t.SANS,
                    "fontWeight": 600,
                    "fontSize": "13px",
                    "textDecoration": "none",
                },
            ),
        ],
    )


def _newconfirm_popover() -> html.Div:
    return html.Div(
        id=ID.NEWCONFIRM_POP,
        className="hidden",
        style={**t.popover(290), "borderRadius": "13px", "padding": "16px"},
        children=[
            html.Div(
                "Start a new conversation?",
                style={
                    "fontFamily": t.SANS,
                    "fontWeight": 600,
                    "fontSize": "14px",
                    "color": t.INK,
                },
            ),
            html.Div(
                "This clears the current thread and begins fresh.",
                style={
                    "fontFamily": t.SANS,
                    "fontSize": "13px",
                    "lineHeight": 1.5,
                    "color": t.MUTED,
                    "marginTop": "5px",
                },
            ),
            html.Div(
                style={"display": "flex", "gap": "8px", "marginTop": "14px"},
                children=[
                    html.Div(
                        "Cancel",
                        id=ID.CANCEL_NEW,
                        n_clicks=0,
                        style={
                            "flex": "1",
                            "textAlign": "center",
                            "border": f"1px solid {t.BORDER_SOFT}",
                            "borderRadius": "9px",
                            "padding": "8px",
                            "fontFamily": t.SANS,
                            "fontWeight": 600,
                            "fontSize": "13px",
                            "color": t.TAUPE,
                            "cursor": "pointer",
                        },
                    ),
                    html.Div(
                        "New conversation",
                        id=ID.CONFIRM_NEW,
                        n_clicks=0,
                        style={
                            "flex": "1",
                            "textAlign": "center",
                            "background": t.ACCENT,
                            "color": t.ON_ACCENT,
                            "borderRadius": "9px",
                            "padding": "8px",
                            "fontFamily": t.SANS,
                            "fontWeight": 600,
                            "fontSize": "13px",
                            "cursor": "pointer",
                        },
                    ),
                ],
            ),
        ],
    )


def header() -> html.Div:
    info_badge = html.Span(
        "i",
        style={
            "width": "16px",
            "height": "16px",
            "borderRadius": "50%",
            "border": "1.5px solid #b6a587",
            "color": t.MUTED,
            "fontFamily": "Georgia, serif",
            "fontStyle": "italic",
            "fontWeight": 700,
            "fontSize": "10px",
            "display": "inline-flex",
            "alignItems": "center",
            "justifyContent": "center",
        },
    )

    return html.Div(
        style={
            "flex": "none",
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "space-between",
            "padding": "14px 30px",
            "background": t.PANEL,
            "borderBottom": f"1px solid {t.BORDER}",
        },
        children=[
            # brand
            html.Div(
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "11px",
                },
                children=[
                    html.Div("P", style=t.logo(30, 7, 17)),
                    html.Div(
                        APP_NAME,
                        style={
                            "fontFamily": t.SERIF,
                            "fontWeight": 600,
                            "fontSize": "18px",
                            "color": t.INK,
                            "letterSpacing": ".2px",
                        },
                    ),
                ],
            ),
            # actions
            html.Div(
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "14px",
                },
                children=[
                    html.Div(
                        style={"position": "relative"},
                        children=[
                            html.Div(
                                [info_badge, " About"],
                                id=ID.ABOUT_TRIGGER,
                                n_clicks=0,
                                className="hover-ink",
                                style={
                                    "display": "inline-flex",
                                    "alignItems": "center",
                                    "gap": "6px",
                                    "fontFamily": t.SANS,
                                    "fontWeight": 500,
                                    "fontSize": "13px",
                                    "color": t.MUTED,
                                    "cursor": "pointer",
                                },
                            ),
                            _about_popover(),
                        ],
                    ),
                    html.Div(
                        style={"position": "relative"},
                        children=[
                            html.Div(
                                [
                                    html.Span(
                                        "+",
                                        style={
                                            "fontSize": "15px",
                                            "lineHeight": "1",
                                            "color": t.ACCENT,
                                        },
                                    ),
                                    " New conversation",
                                ],
                                id=ID.NEW_TRIGGER,
                                n_clicks=0,
                                className="hover-border",
                                style={
                                    "display": "inline-flex",
                                    "alignItems": "center",
                                    "gap": "7px",
                                    "background": t.CARD,
                                    "border": f"1px solid {t.BORDER_SOFT}",
                                    "borderRadius": "8px",
                                    "padding": "7px 13px",
                                    "fontFamily": t.SANS,
                                    "fontWeight": 500,
                                    "fontSize": "13px",
                                    "color": t.TAUPE,
                                    "cursor": "pointer",
                                },
                            ),
                            _newconfirm_popover(),
                        ],
                    ),
                ],
            ),
        ],
    )

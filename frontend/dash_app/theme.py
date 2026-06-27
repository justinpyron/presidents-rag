"""Visual language for the Presidential Archive app.

A single source of truth for the palette and typography so every component
reads from the same vocabulary. Structural styles live inline in the
components (to mirror the designer's prototype closely); hover states,
animations, fonts and markdown styling live in ``assets/app.css``.
"""

# --- Palette -----------------------------------------------------------------
PAPER = "#f4efe4"  # app background
PANEL = "#f9f4ea"  # header background
CARD = "#fdfbf5"  # cards, popovers, composer
CARD_ALT = "#fbf7ee"  # user message bubble
SOURCE_BG = "#faf5e9"  # source card background
CHIP_BG = "#efe7d6"  # composer chips

BORDER = "#e6ddc9"
BORDER_SOFT = "#e3dac8"
BORDER_SOURCE = "#e8dfca"

INK = "#33291d"  # primary text
INK_STRONG = "#2e251a"  # headings / answer text
TAUPE = "#5e5141"
TAUPE_DEEP = "#5b4a39"
MUTED = "#8a7c66"
MUTED_SOFT = "#a99c84"
MUTED_WARM = "#7c6f5b"

ACCENT = "#7c3b34"  # maroon
ACCENT_HOVER = "#6b332d"
ACCENT_MUTED = "#b8958c"  # empty-composer send button (see app.css)
ACCENT_SOFT = "#b58c84"  # subtle arrows
ON_ACCENT = "#f6efe2"  # text on maroon

# --- Typography --------------------------------------------------------------
SANS = "'Public Sans', system-ui, sans-serif"
SERIF = "'Spectral', Georgia, serif"

# --- Shared building blocks --------------------------------------------------
SHADOW_POPOVER = "0 18px 44px rgba(60,40,15,.2)"
SHADOW_MENU = "0 16px 40px rgba(60,40,15,.18)"
SHADOW_COMPOSER = "0 2px 12px rgba(60,40,15,.06)"


def logo(size: int, radius: int, font_size: int) -> dict:
    """Square maroon 'P' mark used in the header, welcome screen and avatars."""
    return {
        "flex": "none",
        "width": f"{size}px",
        "height": f"{size}px",
        "borderRadius": f"{radius}px",
        "background": ACCENT,
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "center",
        "fontFamily": SERIF,
        "fontWeight": 600,
        "fontSize": f"{font_size}px",
        "color": ON_ACCENT,
    }


def popover(width: int) -> dict:
    """Base style for an absolutely-positioned popover card.

    Visibility is controlled by the ``hidden`` class (see ``app.css``), toggled
    by the popover clientside callback — not by an inline ``display``.
    """
    return {
        "position": "absolute",
        "top": "calc(100% + 12px)",
        "right": 0,
        "width": f"{width}px",
        "background": CARD,
        "border": f"1px solid {BORDER_SOFT}",
        "borderRadius": "14px",
        "boxShadow": SHADOW_POPOVER,
        "padding": "18px",
        "zIndex": 60,
    }

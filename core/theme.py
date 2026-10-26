"""Apple-style design tokens and global stylesheet.

Single source of truth for colour. Widgets should read tokens from `palette()`
instead of hard-coding hex values, so light/dark stay consistent everywhere
(including matplotlib figures).
"""

# Font stack: real SF Pro when present, otherwise Windows' UI font.
FONT_UI = '"SF Pro Text", "SF Pro Display", "Segoe UI Variable Text", "Segoe UI", "PingFang SC", "Microsoft YaHei UI", sans-serif'
FONT_MONO = '"SF Mono", "JetBrains Mono", "Cascadia Mono", Consolas, monospace'

_LIGHT = {
    "bg":          "#f5f5f7",
    "surface":     "#ffffff",
    "surface_alt": "#fbfbfd",
    "sidebar":     "#f0f0f2",
    "separator":   "#d2d2d7",
    "border":      "#d2d2d7",
    "border_soft": "#e8e8ed",
    "text":        "#1d1d1f",
    "text_2":      "#6e6e73",
    "text_3":      "#8e8e93",
    "accent":      "#0071e3",
    "accent_hi":   "#0077ed",
    "accent_soft": "rgba(0, 113, 227, 0.10)",
    "on_accent":   "#ffffff",
    "success":     "#248a3d",
    "warning":     "#b25000",
    "danger":      "#d70015",
    "control":     "#ffffff",
    "control_alt": "#ececef",
    "hover":       "#ebebef",
    "track":       "#e3e3e8",
    # matplotlib
    "fig_bg":      "#ffffff",
    "axes_bg":     "#fbfbfd",
    "grid":        "#e8e8ed",
    "series_pos":  "#0071e3",
    "series_neg":  "#ff9500",
    "series_alt":  "#5856d6",
}

_DARK = {
    "bg":          "#1c1c1e",
    "surface":     "#2c2c2e",
    "surface_alt": "#242426",
    "sidebar":     "#242426",
    "separator":   "#38383a",
    "border":      "#3a3a3c",
    "border_soft": "#2f2f31",
    "text":        "#f5f5f7",
    "text_2":      "#a1a1a6",
    "text_3":      "#8e8e93",
    "accent":      "#0a84ff",
    "accent_hi":   "#3b9bff",
    "accent_soft": "rgba(10, 132, 255, 0.18)",
    "on_accent":   "#ffffff",
    "success":     "#30d158",
    "warning":     "#ff9f0a",
    "danger":      "#ff453a",
    "control":     "#3a3a3c",
    "control_alt": "#2c2c2e",
    "hover":       "#3a3a3c",
    "track":       "#48484a",
    # matplotlib
    "fig_bg":      "#1c1c1e",
    "axes_bg":     "#2c2c2e",
    "grid":        "#3a3a3c",
    "series_pos":  "#0a84ff",
    "series_neg":  "#ff9f0a",
    "series_alt":  "#5e5ce6",
}


def palette(dark: bool) -> dict:
    """Colour tokens for the active appearance."""
    return dict(_DARK if dark else _LIGHT)


def stylesheet(dark: bool) -> str:
    """Global QSS. Covers every standard control so pages need little of their own."""
    p = palette(dark)
    return f"""
    * {{
        font-family: {FONT_UI};
        font-size: 14px;
    }}

    QMainWindow, QDialog {{ background: {p['bg']}; }}
    QWidget#pageStack, QScrollArea {{ background: {p['bg']}; border: none; }}
    QScrollArea > QWidget > QWidget {{ background: transparent; }}

    QLabel {{ color: {p['text']}; background: transparent; }}
    QLabel#pageTitle {{
        font-size: 26px; font-weight: 600; color: {p['text']};
        letter-spacing: -0.4px; padding: 2px 0 0 0;
    }}
    QLabel#pageSubtitle {{ font-size: 14px; color: {p['text_2']}; padding-bottom: 4px; }}
    QLabel#sectionTitle {{
        font-size: 15px; font-weight: 600; color: {p['text']};
        padding-top: 8px; letter-spacing: -0.2px;
    }}
    QLabel#hint {{ font-size: 12px; color: {p['text_3']}; }}
    QLabel#statusOk {{ font-size: 13px; color: {p['success']}; }}
    QLabel#statusWarn {{ font-size: 13px; color: {p['warning']}; }}
    QLabel#statusErr {{ font-size: 13px; color: {p['danger']}; }}

    /* ── Sidebar ─────────────────────────────────────────── */
    QWidget#navPanel {{
        background: {p['sidebar']};
        border-right: 1px solid {p['separator']};
    }}
    QLabel#navTitle {{
        font-size: 15px; font-weight: 600; color: {p['text']};
        letter-spacing: -0.2px; padding: 2px 0;
    }}
    QLabel#navFooter {{
        font-size: 11px; color: {p['text_3']}; padding: 10px 0 2px 0;
        border-top: 1px solid {p['separator']};
    }}
    QLabel#navStatus {{
        font-size: 12px; color: {p['text_2']}; padding: 10px 2px;
        border-top: 1px solid {p['separator']};
    }}
    QListWidget#navList {{
        background: transparent; border: none; outline: none;
        font-size: 14px; color: {p['text']};
    }}
    QListWidget#navList::item {{
        padding: 8px 12px; border-radius: 7px; margin: 1px 0; color: {p['text']};
    }}
    QListWidget#navList::item:selected {{
        background: {p['accent_soft']}; color: {p['accent']}; font-weight: 600;
    }}
    QListWidget#navList::item:hover:!selected {{ background: {p['hover']}; }}

    /* ── Cards / group boxes ─────────────────────────────── */
    QGroupBox {{
        background: {p['surface']};
        border: 1px solid {p['border_soft']};
        border-radius: 12px;
        margin-top: 24px; padding: 16px;
        font-size: 14px; font-weight: 600; color: {p['text']};
    }}
    QGroupBox::title {{
        /* Section header sitting above the card, iOS grouped-list style */
        subcontrol-origin: margin; subcontrol-position: top left;
        left: 2px; padding: 0 2px 7px 2px;
        color: {p['text_2']}; font-size: 13px; font-weight: 600;
    }}
    QFrame#card {{
        background: {p['surface']};
        border: 1px solid {p['border_soft']};
        border-radius: 12px;
    }}
    QFrame#separator {{ background: {p['separator']}; border: none; max-height: 1px; }}

    /* ── Error / warning card ────────────────────────────── */
    QFrame#errorCard {{
        background: {p['surface']};
        border: 1px solid {p['danger']};
        border-left: 4px solid {p['danger']};
        border-radius: 12px;
    }}
    QFrame#errorCard[level="warning"] {{
        border: 1px solid {p['warning']};
        border-left: 4px solid {p['warning']};
    }}
    QLabel#errorCardIcon {{
        background: {p['danger']}; color: #ffffff;
        border-radius: 11px; font-weight: 700; font-size: 13px;
    }}
    QFrame#errorCard[level="warning"] QLabel#errorCardIcon {{
        background: {p['warning']};
    }}
    QLabel#errorCardHeadline {{
        font-size: 14px; font-weight: 600; color: {p['text']};
    }}
    QPushButton#errorCardToggle, QPushButton#errorCardClose {{
        background: transparent; border: 1px solid {p['border_soft']};
        border-radius: 7px; padding: 3px 6px;
        color: {p['text_2']}; font-size: 12px;
    }}
    QPushButton#errorCardToggle:hover, QPushButton#errorCardClose:hover {{
        background: {p['hover']};
    }}
    QTextEdit#errorCardDetail {{
        background: {p['surface_alt']};
        border: 1px solid {p['border_soft']};
        border-radius: 8px;
        color: {p['text_2']};
        font-family: {FONT_MONO};
        font-size: 12px;
    }}

    /* ── Drop zone ───────────────────────────────────────── */
    QFrame#dropZone {{
        background: {p['surface_alt']};
        border: 2px dashed {p['border']};
        border-radius: 12px;
    }}
    QFrame#dropZone:hover {{ border-color: {p['accent']}; }}
    QFrame#dropZone[dragActive="true"] {{
        border-color: {p['accent']}; background: {p['accent_soft']};
    }}
    QFrame#dropZone[dropState="ok"] {{
        border: 1px solid {p['success']}; background: {p['surface']};
    }}
    QFrame#dropZone[dropState="bad"] {{
        border: 1px solid {p['danger']}; background: {p['surface']};
    }}

    /* ── Buttons ─────────────────────────────────────────── */
    QPushButton {{
        background: {p['control']}; color: {p['text']};
        border: 1px solid {p['border']}; border-radius: 8px;
        padding: 7px 16px; font-size: 13px; font-weight: 500;
    }}
    QPushButton:hover {{ background: {p['hover']}; }}
    QPushButton:pressed {{ background: {p['control_alt']}; }}
    QPushButton:disabled {{ color: {p['text_3']}; background: {p['control_alt']};
                            border-color: {p['border_soft']}; }}
    QPushButton#primaryBtn {{
        background: {p['accent']}; color: {p['on_accent']};
        border: none; border-radius: 9px;
        padding: 9px 20px; font-size: 14px; font-weight: 600;
    }}
    QPushButton#primaryBtn:hover {{ background: {p['accent_hi']}; }}
    QPushButton#primaryBtn:disabled {{ background: {p['control_alt']}; color: {p['text_3']}; }}
    QPushButton#ghostBtn {{
        background: transparent; border: 1px solid {p['border']};
        color: {p['text']}; border-radius: 8px; padding: 5px 12px; font-size: 13px;
    }}
    QPushButton#ghostBtn:hover {{ background: {p['hover']}; }}
    QPushButton#dangerBtn {{
        background: {p['danger']}; color: #ffffff; border: none;
        border-radius: 8px; padding: 7px 16px; font-weight: 600;
    }}

    /* ── Inputs ──────────────────────────────────────────── */
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
        background: {p['control']}; color: {p['text']};
        border: 1px solid {p['border']}; border-radius: 8px;
        padding: 7px 10px; font-size: 13px;
        selection-background-color: {p['accent']}; selection-color: #ffffff;
    }}
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
        border: 2px solid {p['accent']}; padding: 6px 9px;
    }}
    QLineEdit::placeholder {{ color: {p['text_3']}; }}
    QComboBox::drop-down {{ border: none; width: 22px; }}

    /* Spin steppers are deliberately left unstyled: styling any sub-control
       makes Qt stop drawing the native arrow glyphs, and CSS border-triangles
       do not render for ::up-arrow here — the result is invisible controls. */
    QComboBox QAbstractItemView {{
        background: {p['surface']}; color: {p['text']};
        border: 1px solid {p['border']}; border-radius: 8px;
        selection-background-color: {p['accent_soft']}; selection-color: {p['accent']};
        outline: none;
    }}

    QTextEdit {{
        background: {p['surface_alt']}; color: {p['text']};
        border: 1px solid {p['border_soft']}; border-radius: 10px;
        font-family: {FONT_MONO}; font-size: 12px; padding: 8px;
        selection-background-color: {p['accent']}; selection-color: #ffffff;
    }}

    QCheckBox {{ color: {p['text']}; font-size: 13px; spacing: 7px; }}
    QCheckBox::indicator {{
        width: 17px; height: 17px; border-radius: 5px;
        border: 1px solid {p['border']}; background: {p['control']};
    }}
    QCheckBox::indicator:checked {{ background: {p['accent']}; border-color: {p['accent']}; }}

    /* ── Tables ──────────────────────────────────────────── */
    QTableWidget {{
        background: {p['surface']}; color: {p['text']};
        gridline-color: {p['border_soft']};
        border: 1px solid {p['border_soft']}; border-radius: 10px;
        font-size: 13px; selection-background-color: {p['accent_soft']};
        selection-color: {p['text']};
    }}
    QHeaderView::section {{
        background: {p['surface_alt']}; color: {p['text_2']};
        border: none; border-bottom: 1px solid {p['border']};
        padding: 7px 6px; font-size: 12px; font-weight: 600;
    }}
    QTableWidget::item {{ padding: 5px; border: none; }}

    QTabWidget::pane {{
        border: 1px solid {p['border_soft']}; border-radius: 10px;
        background: {p['surface']}; top: -1px;
    }}
    QTabBar::tab {{
        background: transparent; color: {p['text_2']};
        padding: 7px 18px; font-size: 13px; font-weight: 500;
        border: none; margin-right: 4px; border-radius: 7px;
    }}
    QTabBar::tab:selected {{ background: {p['accent_soft']}; color: {p['accent']}; }}
    QTabBar::tab:hover:!selected {{ background: {p['hover']}; }}

    /* ── Sliders ─────────────────────────────────────────── */
    QSlider::groove:horizontal {{
        height: 4px; background: {p['track']}; border-radius: 2px;
    }}
    QSlider::sub-page:horizontal {{ background: {p['accent']}; border-radius: 2px; }}
    QSlider::handle:horizontal {{
        width: 16px; height: 16px; margin: -7px 0;
        background: #ffffff; border: 1px solid rgba(0,0,0,0.16);
        border-radius: 8px;
    }}
    QSlider::handle:horizontal:hover {{ background: {p['surface']}; }}

    QProgressBar {{
        border: none; border-radius: 3px; background: {p['track']};
        height: 6px; text-align: center; color: transparent;
    }}
    QProgressBar::chunk {{ background: {p['accent']}; border-radius: 3px; }}

    /* ── Scrollbars ──────────────────────────────────────── */
    QScrollBar:vertical {{
        background: transparent; width: 11px; margin: 2px;
    }}
    QScrollBar::handle:vertical {{
        background: {p['track']}; border-radius: 5px; min-height: 28px;
    }}
    QScrollBar::handle:vertical:hover {{ background: {p['text_3']}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
    QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 2px; }}
    QScrollBar::handle:horizontal {{
        background: {p['track']}; border-radius: 5px; min-width: 28px;
    }}

    QStatusBar {{
        background: {p['sidebar']}; color: {p['text_2']};
        border-top: 1px solid {p['separator']}; font-size: 12px;
    }}
    QToolTip {{
        background: {p['surface']}; color: {p['text']};
        border: 1px solid {p['border']}; border-radius: 6px; padding: 5px 8px;
    }}
    """


# ── matplotlib fonts ──────────────────────────────────────────────────
# Qt reads FONT_UI from the system, so Chinese in the panels always had
# glyphs; matplotlib does not -- it ships DejaVu Sans, which has none, and
# draws every CJK character as an empty box. That is why the dynamic-shim
# current table came out as rows of rectangles where the axis labels should
# be. Put the first CJK face that is actually installed in front of the
# default stack, once, at import time.
_CJK_FONTS = (
    "Microsoft YaHei", "Microsoft YaHei UI", "SimHei", "PingFang SC",
    "Noto Sans CJK SC", "Noto Sans SC", "Source Han Sans SC",
    "Microsoft JhengHei", "Heiti SC", "SimSun",
)

#: The face matplotlib ended up with, or "" if the machine has no CJK font.
MPL_CJK_FONT = ""


def apply_mpl_fonts() -> str:
    """Point matplotlib at an installed CJK font. Returns the face used."""
    global MPL_CJK_FONT
    try:
        import matplotlib
        from matplotlib import font_manager
    except Exception:
        return ""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((f for f in _CJK_FONTS if f in installed), "")
    if chosen:
        stack = list(matplotlib.rcParams["font.sans-serif"])
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = (
            [chosen] + [f for f in stack if f != chosen])
    # SimHei and several other CJK faces have no U+2212, so a negative tick
    # label would be the one box left on an otherwise fixed figure.
    matplotlib.rcParams["axes.unicode_minus"] = False
    MPL_CJK_FONT = chosen
    return chosen


apply_mpl_fonts()


def style_axes(ax, p, *, grid=True):
    """Apply the palette to a matplotlib Axes."""
    ax.set_facecolor(p["axes_bg"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(p["border"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=p["text_2"], labelsize=8, length=3, width=0.8)
    if grid:
        ax.grid(True, color=p["grid"], linewidth=0.6, alpha=0.9)
        ax.set_axisbelow(True)

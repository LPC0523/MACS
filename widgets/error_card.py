"""An inline failure card: one actionable sentence, details on request.

A modal dialog is the wrong shape for this. It has to be dismissed before the
log behind it can be read, it cannot hold a stack trace without becoming
unreadable, and once closed the message is gone -- so the user ends up
describing the error from memory. The card stays on the page next to the log,
says the one thing worth acting on, and keeps the diagnostics one click away.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget,
)


class ErrorCard(QFrame):
    """Red (error) or amber (warning) card with a collapsible detail pane."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("errorCard")
        self.setVisible(False)
        self._level = "error"

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(8)

        top = QHBoxLayout()
        top.setSpacing(10)

        self.icon = QLabel("!")
        self.icon.setObjectName("errorCardIcon")
        self.icon.setFixedSize(22, 22)
        self.icon.setAlignment(Qt.AlignCenter)
        top.addWidget(self.icon, 0, Qt.AlignTop)

        self.headline = QLabel("")
        self.headline.setObjectName("errorCardHeadline")
        self.headline.setWordWrap(True)
        self.headline.setTextInteractionFlags(Qt.TextSelectableByMouse)
        top.addWidget(self.headline, 1)

        self.btn_detail = QPushButton("详情")
        self.btn_detail.setObjectName("errorCardToggle")
        self.btn_detail.setCheckable(True)
        self.btn_detail.setFixedWidth(64)
        self.btn_detail.toggled.connect(self._toggle_detail)
        top.addWidget(self.btn_detail, 0, Qt.AlignTop)

        self.btn_close = QPushButton("✕")
        self.btn_close.setObjectName("errorCardClose")
        self.btn_close.setFixedWidth(28)
        self.btn_close.clicked.connect(self.clear)
        top.addWidget(self.btn_close, 0, Qt.AlignTop)

        outer.addLayout(top)

        self.detail = QTextEdit()
        self.detail.setObjectName("errorCardDetail")
        self.detail.setReadOnly(True)
        self.detail.setVisible(False)
        self.detail.setMaximumHeight(180)
        outer.addWidget(self.detail)

    # ── API ──────────────────────────────────────────────────────────
    def show_error(self, headline: str, detail: str = "", code: str = ""):
        self._set("error", headline, detail, code)

    def show_warning(self, headline: str, detail: str = "", code: str = ""):
        self._set("warning", headline, detail, code)

    def clear(self):
        self.setVisible(False)
        self.btn_detail.setChecked(False)
        self.detail.clear()

    # ── internals ────────────────────────────────────────────────────
    def _set(self, level: str, headline: str, detail: str, code: str):
        self._level = level
        self.setProperty("level", level)
        self.icon.setText("!" if level == "error" else "⚠")
        self.headline.setText(headline)

        body = detail or ""
        if code:
            body = (body + "\n\n" if body else "") + f"错误代码：{code}"
        self.detail.setPlainText(body)
        self.btn_detail.setVisible(bool(body))
        self.btn_detail.setChecked(False)
        self.detail.setVisible(False)

        # Re-evaluate the stylesheet now that `level` changed.
        self.style().unpolish(self)
        self.style().polish(self)
        self.setVisible(True)

    def _toggle_detail(self, on: bool):
        self.detail.setVisible(on)
        self.btn_detail.setText("收起" if on else "详情")

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor, QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class PromptInputDialog(QDialog):
    """Floating prompt input dialog for the Alt+P hotkey.

    Shows near the cursor and lets the user type a prompt that will be
    applied to the previously highlighted text.
    """

    _DIALOG_WIDTH = 420

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
        self.setObjectName("PromptInputDialog")
        self.setFixedWidth(self._DIALOG_WIDTH)

        self._build_ui()
        self._apply_style()

        close_shortcut = QShortcut(QKeySequence("Esc"), self)
        close_shortcut.activated.connect(self.reject)

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(0)

        card = QFrame()
        card.setObjectName("PromptCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(14, 14, 14, 14)
        card_layout.setSpacing(10)

        # Header row
        header_row = QHBoxLayout()
        header_row.setSpacing(8)

        title_label = QLabel("Enter your prompt")
        title_label.setObjectName("PromptTitle")

        close_button = QPushButton("×")
        close_button.setObjectName("PromptCloseButton")
        close_button.setFixedSize(28, 28)
        close_button.clicked.connect(self.reject)

        header_row.addWidget(title_label)
        header_row.addStretch(1)
        header_row.addWidget(close_button)

        # Text input
        self._input = QLineEdit()
        self._input.setObjectName("PromptLineEdit")
        self._input.setPlaceholderText("Type your prompt and press Enter…")
        self._input.returnPressed.connect(self._on_enter)

        # Hint label
        hint_label = QLabel("Enter to send  ·  Esc to cancel")
        hint_label.setObjectName("PromptHint")
        hint_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        card_layout.addLayout(header_row)
        card_layout.addWidget(self._input)
        card_layout.addWidget(hint_label)

        root_layout.addWidget(card)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QDialog#PromptInputDialog {
                background: transparent;
            }
            QFrame#PromptCard {
                background: #ffffff;
                border: 1px solid #dbe3ee;
                border-radius: 14px;
            }
            QLabel#PromptTitle {
                color: #0f172a;
                font-size: 13px;
                font-weight: 700;
            }
            QPushButton#PromptCloseButton {
                background: transparent;
                border: none;
                color: #64748b;
                font-size: 18px;
                font-weight: 400;
                border-radius: 6px;
                padding: 0px;
            }
            QPushButton#PromptCloseButton:hover {
                background: #f1f5f9;
                color: #0f172a;
            }
            QLineEdit#PromptLineEdit {
                background: #f8fafc;
                border: 1px solid #cbd5e1;
                border-radius: 10px;
                padding: 8px 10px;
                color: #0f172a;
                font-size: 13px;
                min-height: 20px;
            }
            QLineEdit#PromptLineEdit:focus {
                border: 1px solid #60a5fa;
                background: #ffffff;
            }
            QLabel#PromptHint {
                color: #94a3b8;
                font-size: 11px;
            }
            """
        )

    def _on_enter(self) -> None:
        if self._input.text().strip():
            self.accept()

    def prompt_text(self) -> str:
        """Return the user-entered prompt text."""
        return self._input.text().strip()

    def show_near_cursor(self) -> None:
        """Position the dialog near the current cursor position."""
        self.adjustSize()

        cursor_pos = QCursor.pos()
        screen = QGuiApplication.screenAt(cursor_pos) or QGuiApplication.primaryScreen()

        if screen is None:
            self.move(cursor_pos)
            return

        geom = screen.availableGeometry()
        margin = 16
        dialog_w = self.width()
        dialog_h = self.height()

        # Prefer placing below and slightly left of cursor; flip sides if near edges
        x = cursor_pos.x() - dialog_w // 2
        y = cursor_pos.y() + 20

        x = max(geom.x() + margin, min(x, geom.x() + geom.width() - dialog_w - margin))
        y = max(geom.y() + margin, min(y, geom.y() + geom.height() - dialog_h - margin))

        self.move(x, y)

from __future__ import annotations

from html import escape

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from session.manager import ConversationMessage


class ChatTranscript(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._messages: list[ConversationMessage] = []
        self._message_index: dict[str, int] = {}
        self._message_widgets: dict[str, QWidget] = {}
        self._expanded_screen_context_message_ids: set[str] = set()
        self.setObjectName("ChatTranscript")
        self.setMinimumHeight(340)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        self._scroll_area = QScrollArea()
        self._scroll_area.setObjectName("ChatTranscriptScrollArea")
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll_area.setFrameShape(QFrame.Shape.NoFrame)

        self._content_widget = QWidget()
        self._content_widget.setObjectName("ChatTranscriptContent")
        self._content_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)

        self._content_layout = QVBoxLayout(self._content_widget)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(14)
        self._content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._scroll_area.setWidget(self._content_widget)
        outer_layout.addWidget(self._scroll_area)

        self.setStyleSheet(
            """
            QWidget#ChatTranscript {
                background: transparent;
            }
            QScrollArea#ChatTranscriptScrollArea {
                background: transparent;
                border: none;
            }
            QFrame#ChatTranscriptCard {
                background: #ffffff;
                border: 1px solid #dbe3ee;
                border-radius: 14px;
            }
            QLabel#ChatTranscriptHeader {
                color: #2563eb;
                font-size: 14px;
                font-weight: 700;
            }
            QLabel#ChatTranscriptMode {
                color: #64748b;
                font-size: 12px;
                font-weight: 600;
            }
            QLabel#ChatTranscriptBody {
                color: #0f172a;
                font-size: 14px;
            }
            QLabel#ChatTranscriptEmptyState {
                color: #64748b;
                font-size: 15px;
            }
            QLabel#ChatTranscriptOcrTitle {
                color: #64748b;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.04em;
            }
            QToolButton#ChatTranscriptOcrToggle {
                background: #eff6ff;
                color: #1d4ed8;
                border: 1px solid #bfdbfe;
                border-radius: 999px;
                padding: 4px 10px;
                font-size: 12px;
                font-weight: 700;
            }
            QToolButton#ChatTranscriptOcrToggle:checked {
                background: #dbeafe;
            }
            QPlainTextEdit#ChatTranscriptOcrContext {
                background: #f8fafc;
                border: 1px solid #dbe3ee;
                border-radius: 12px;
                color: #334155;
                font-size: 13px;
                padding: 8px 10px;
            }
            QPlainTextEdit#ChatTranscriptOcrContext:focus {
                border: 1px solid #60a5fa;
            }
            """
        )
        self._render()

    def reset_messages(self, messages: list[ConversationMessage]) -> None:
        self._messages = list(messages)
        self._reindex()
        self._render()

    def upsert_message(self, message: ConversationMessage) -> None:
        if message.id in self._message_index:
            self._messages[self._message_index[message.id]] = message
            self._replace_message_widget(message)
        else:
            self._message_index[message.id] = len(self._messages)
            self._messages.append(message)
            self._insert_message_widget(message)
        self.scroll_to_bottom()

    def _toggle_screen_context(self, message_id: str) -> None:
        if message_id in self._expanded_screen_context_message_ids:
            self._expanded_screen_context_message_ids.remove(message_id)
        else:
            self._expanded_screen_context_message_ids.add(message_id)

        message_index = self._message_index.get(message_id)
        if message_index is None:
            self._render()
            return

        scrollbar = self._scroll_area.verticalScrollBar()
        previous_value = scrollbar.value()
        was_at_bottom = previous_value >= scrollbar.maximum() - 1

        self._replace_message_widget(self._messages[message_index])

        if was_at_bottom:
            self.scroll_to_bottom()
        else:
            scrollbar.setValue(min(previous_value, scrollbar.maximum()))

    def scroll_to_bottom(self) -> None:
        scrollbar = self._scroll_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _reindex(self) -> None:
        self._message_index = {message.id: index for index, message in enumerate(self._messages)}

    def _clear_content(self) -> None:
        self._message_widgets.clear()
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _render_empty_state(self) -> QWidget:
        empty_state = QLabel(
            "No conversation yet. Highlight text in another app and use Alt+D, Alt+E, or Alt+S. "
            "Press Alt+H to hide or show the window. "
            "Press Alt+L to switch between your chosen language and English."
        )
        empty_state.setObjectName("ChatTranscriptEmptyState")
        empty_state.setWordWrap(True)
        empty_state.setMargin(2)
        return empty_state

    def _render_body_label(self, message: ConversationMessage) -> QLabel:
        body_label = QLabel()
        body_label.setObjectName("ChatTranscriptBody")
        body_label.setWordWrap(True)
        body_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        body_label.setTextFormat(Qt.TextFormat.RichText)

        content = escape(message.content).replace("\n", "<br />")
        if not content.strip() and message.role == "assistant":
            content = "<span style='opacity: 0.7; font-style: italic; color: #64748b;'>Generating response...</span>"
        elif not content.strip():
            content = "<span style='opacity: 0.7; color: #64748b;'>Empty message</span>"

        body_label.setText(content)
        return body_label

    def _render_screen_context_box(self, screen_context: str) -> QPlainTextEdit:
        context_box = QPlainTextEdit()
        context_box.setObjectName("ChatTranscriptOcrContext")
        context_box.setReadOnly(True)
        context_box.setPlainText(screen_context)
        context_box.setMinimumHeight(72)
        context_box.setMaximumHeight(120)
        context_box.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        context_box.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        context_box.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        context_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        return context_box

    def _render_message_card(self, message: ConversationMessage) -> QWidget:
        role = message.role.capitalize()
        mode_label = message.mode.replace("_", " ").title() if message.mode else ""
        screen_context = message.screen_context.strip()
        has_screen_context = bool(screen_context)
        expanded = self._is_screen_context_expanded(message.id)

        card = QFrame()
        card.setObjectName("ChatTranscriptCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(14, 14, 14, 14)
        card_layout.setSpacing(10)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(8)

        header_label = QLabel(role)
        header_label.setObjectName("ChatTranscriptHeader")
        header_label.setStyleSheet(
            "color: #2563eb; font-size: 14px; font-weight: 700;"
            if message.role == "user"
            else "color: #059669; font-size: 14px; font-weight: 700;"
        )
        header_row.addWidget(header_label)

        if mode_label:
            mode_widget = QLabel(mode_label)
            mode_widget.setObjectName("ChatTranscriptMode")
            header_row.addWidget(mode_widget)

        header_row.addStretch(1)

        if message.role == "user" and has_screen_context:
            toggle_button = QToolButton()
            toggle_button.setObjectName("ChatTranscriptOcrToggle")
            toggle_button.setCheckable(True)
            toggle_button.setChecked(expanded)
            toggle_button.setText("Hide OCR context" if expanded else "Show OCR context")
            toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
            toggle_button.clicked.connect(lambda _checked=False, message_id=message.id: self._toggle_screen_context(message_id))
            header_row.addWidget(toggle_button)

        card_layout.addLayout(header_row)

        if message.role == "user" and has_screen_context and expanded:
            context_title = QLabel("OCR context")
            context_title.setObjectName("ChatTranscriptOcrTitle")
            card_layout.addWidget(context_title)
            card_layout.addWidget(self._render_screen_context_box(screen_context))

        card_layout.addWidget(self._render_body_label(message))
        return card

    def _insert_message_widget(self, message: ConversationMessage) -> None:
        if len(self._messages) == 1:
            self._clear_content()

        message_widget = self._render_message_card(message)
        self._message_widgets[message.id] = message_widget

        index = self._message_index[message.id]
        if index >= self._content_layout.count():
            self._content_layout.addWidget(message_widget)
        else:
            self._content_layout.insertWidget(index, message_widget)

        self._content_widget.adjustSize()

    def _replace_message_widget(self, message: ConversationMessage) -> None:
        existing_widget = self._message_widgets.get(message.id)
        if existing_widget is None:
            self._render()
            return

        index = self._message_index[message.id]
        self._content_layout.removeWidget(existing_widget)
        existing_widget.deleteLater()

        replacement_widget = self._render_message_card(message)
        self._message_widgets[message.id] = replacement_widget

        if index >= self._content_layout.count():
            self._content_layout.addWidget(replacement_widget)
        else:
            self._content_layout.insertWidget(index, replacement_widget)

        self._content_widget.adjustSize()

    def _render(self) -> None:
        self._clear_content()

        if not self._messages:
            self._content_layout.addWidget(self._render_empty_state())
            self._content_widget.adjustSize()
            self.scroll_to_bottom()
            return

        for message in self._messages:
            message_widget = self._render_message_card(message)
            self._message_widgets[message.id] = message_widget
            self._content_layout.addWidget(message_widget)

        self._content_widget.adjustSize()
        self.scroll_to_bottom()

    def _is_screen_context_expanded(self, message_id: str) -> bool:
        return message_id in self._expanded_screen_context_message_ids

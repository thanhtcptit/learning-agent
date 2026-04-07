from __future__ import annotations

from typing import Any, cast

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel, QPlainTextEdit, QToolButton, QWidget

from session.manager import ConversationMessage
import ui.chat_widget as chat_widget_module


_APP = QApplication.instance() or QApplication([])


def _card_widget(transcript: chat_widget_module.ChatTranscript) -> QWidget:
    item = transcript._content_layout.itemAt(0)
    assert item is not None
    widget = item.widget()
    assert widget is not None
    return cast(QWidget, widget)


def _card_layout(card: QWidget) -> Any:
    layout = card.layout()
    assert layout is not None
    return layout


def _toggle_button(card: QWidget) -> QToolButton | None:
    return card.findChild(QToolButton, "ChatTranscriptOcrToggle")


def _context_box(card: QWidget) -> QPlainTextEdit | None:
    return card.findChild(QPlainTextEdit, "ChatTranscriptOcrContext")


def _body_label(card: QWidget) -> QLabel | None:
    return card.findChild(QLabel, "ChatTranscriptBody")


def test_chat_transcript_render_message_shows_collapsed_ocr_toggle() -> None:
    message = ConversationMessage(role="user", content="Selected text", screen_context="Nearby label: Figure 2")
    transcript = chat_widget_module.ChatTranscript()
    transcript.reset_messages([message])

    card = _card_widget(transcript)
    toggle = _toggle_button(card)
    body = _body_label(card)

    assert toggle is not None
    assert toggle.text() == "Show OCR context"
    assert _context_box(card) is None
    assert body is not None
    layout = _card_layout(card)
    assert layout.count() == 2
    assert layout.itemAt(1).widget() is body


def test_chat_transcript_first_message_replaces_empty_state() -> None:
    transcript = chat_widget_module.ChatTranscript()

    message = ConversationMessage(role="user", content="Selected text")
    transcript.upsert_message(message)

    assert transcript._content_layout.count() == 1
    assert transcript._content_layout.itemAt(0).widget().objectName() == "ChatTranscriptCard"


def test_chat_transcript_render_message_shows_expanded_ocr_context_above_query() -> None:
    message = ConversationMessage(role="user", content="Selected text", screen_context="Nearby label: Figure 2")
    transcript = chat_widget_module.ChatTranscript()
    transcript.reset_messages([message])

    toggle = _toggle_button(_card_widget(transcript))
    assert toggle is not None
    toggle.click()

    card = _card_widget(transcript)
    toggle = _toggle_button(card)
    layout = _card_layout(card)
    context_title = cast(QLabel, layout.itemAt(1).widget())
    context_box = _context_box(card)
    body = _body_label(card)

    assert toggle is not None
    assert toggle.text() == "Hide OCR context"
    assert context_title is not None
    assert context_title.text() == "OCR context"
    assert context_box is not None
    assert context_box.toPlainText() == "Nearby label: Figure 2"
    assert context_box.maximumHeight() == 120
    assert context_box.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded
    assert body is not None
    assert layout.count() == 4
    assert layout.itemAt(2).widget() is context_box
    assert layout.itemAt(3).widget() is body


def test_chat_transcript_toggle_button_hides_and_shows_ocr_context() -> None:
    message = ConversationMessage(role="user", content="Selected text", screen_context="Nearby label: Figure 2")
    transcript = chat_widget_module.ChatTranscript()
    transcript.reset_messages([message])

    toggle = _toggle_button(_card_widget(transcript))
    assert toggle is not None

    toggle.click()
    card = _card_widget(transcript)
    assert _context_box(card) is not None
    toggled_button = _toggle_button(card)
    assert toggled_button is not None
    assert toggled_button.text() == "Hide OCR context"

    toggle = _toggle_button(card)
    assert toggle is not None
    toggle.click()

    card = _card_widget(transcript)
    assert _context_box(card) is None
    final_button = _toggle_button(card)
    assert final_button is not None
    assert final_button.text() == "Show OCR context"


def test_chat_transcript_toggle_preserves_scroll_position(monkeypatch) -> None:
    transcript = chat_widget_module.ChatTranscript()
    messages = []

    for index in range(1, 24):
        messages.append(
            ConversationMessage(
                role="user" if index % 2 else "assistant",
                content=(f"Message {index} " * 12).strip(),
                screen_context="Nearby label: Figure 2" if index == 5 else "",
            )
        )

    transcript.reset_messages(messages)
    transcript.resize(420, 220)
    transcript.show()
    _APP.processEvents()

    scrollbar = transcript._scroll_area.verticalScrollBar()
    assert scrollbar.maximum() > 0
    scrollbar.setValue(scrollbar.maximum() // 2)
    original_value = scrollbar.value()

    target_message = messages[4]
    card = transcript._message_widgets[target_message.id]
    toggle = _toggle_button(card)
    assert toggle is not None

    def fail_render() -> None:
        raise AssertionError("toggle should not rerender the full transcript")

    monkeypatch.setattr(transcript, "_render", fail_render)

    toggle.click()
    _APP.processEvents()

    assert scrollbar.value() != 0
    assert abs(scrollbar.value() - original_value) <= 1

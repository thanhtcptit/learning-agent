from __future__ import annotations

from html import escape

from PySide6.QtWidgets import QTextBrowser

from session.manager import ConversationMessage


class ChatTranscript(QTextBrowser):
    def __init__(self) -> None:
        super().__init__()
        self._messages: list[ConversationMessage] = []
        self._message_index: dict[str, int] = {}
        self.setReadOnly(True)
        self.setOpenExternalLinks(True)
        self.setObjectName("ChatTranscript")
        self.setMinimumHeight(340)
        self.setStyleSheet(
            """
            QTextBrowser#ChatTranscript {
                background: #ffffff;
                border: 1px solid #dbe3ee;
                border-radius: 16px;
                color: #0f172a;
                padding: 16px;
                font-size: 14px;
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
        else:
            self._message_index[message.id] = len(self._messages)
            self._messages.append(message)
        self._render()

    def scroll_to_bottom(self) -> None:
        scrollbar = self.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _reindex(self) -> None:
        self._message_index = {message.id: index for index, message in enumerate(self._messages)}

    def _render(self) -> None:
        if not self._messages:
            html = self._wrap_document(
                "<div style='opacity: 0.75; font-size: 15px; color: #64748b;'>"
                "No conversation yet. Highlight text in another app and use Alt+D, Alt+E, or Alt+S. "
                "Press Alt+H to hide or show the window. "
                "Press Alt+L to switch between your chosen language and English."
                "</div>"
            )
            self.setHtml(html)
            return

        blocks = [self._render_message(message) for message in self._messages]
        self.setHtml(self._wrap_document("".join(blocks)))
        self.scroll_to_bottom()

    def _render_message(self, message: ConversationMessage) -> str:
        role = message.role.capitalize()
        mode_label = message.mode.replace("_", " ").title() if message.mode else ""
        content = escape(message.content).replace("\n", "<br />")

        if not content.strip() and message.role == "assistant":
            content = "<span style='opacity: 0.7; font-style: italic; color: #64748b;'>Generating response...</span>"
        elif not content.strip():
            content = "<span style='opacity: 0.7; color: #64748b;'>Empty message</span>"

        accent = "#2563eb" if message.role == "user" else "#059669" if message.role == "assistant" else "#64748b"
        background = "#eff6ff" if message.role == "user" else "#ecfdf5" if message.role == "assistant" else "#f8fafc"

        label_html = escape(role)
        if mode_label:
            label_html = f"{label_html} <span style='opacity: 0.7; font-size: 12px; color: #64748b;'>{escape(mode_label)}</span>"
        return (
            f"<article style='margin-bottom: 14px; padding: 14px 16px; border-radius: 14px; background: {background}; border: 1px solid #dbe3ee;'>"
            f"<div style='margin-bottom: 8px; font-weight: 700; color: {accent}; letter-spacing: 0.02em;'>{label_html}</div>"
            f"<div style='white-space: pre-wrap; line-height: 1.55; color: #0f172a;'>{content}</div>"
            f"</article>"
        )

    def _wrap_document(self, body: str) -> str:
        return (
            "<html><head><style>"
            "body { margin: 0; padding: 0; background: #ffffff; color: #0f172a; font-family: 'Segoe UI', sans-serif; }"
            "</style></head><body>"
            f"{body}"
            "</body></html>"
        )

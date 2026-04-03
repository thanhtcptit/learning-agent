from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLineEdit


class MessageInput(QLineEdit):
    submitted = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setPlaceholderText("Type a message or note, then press Enter")
        self.returnPressed.connect(self._emit_submission)

    def _emit_submission(self) -> None:
        text = self.text().strip()
        if not text:
            return
        self.submitted.emit(text)

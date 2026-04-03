from __future__ import annotations

import uuid
import time

import pyperclip
from pynput.keyboard import Controller, Key


class ClipboardService:
    def __init__(
        self,
        trigger_delay: float = 0.2,
        settle_delay: float = 0.15,
        retry_delay: float = 0.1,
        attempts: int = 3,
    ) -> None:
        self._trigger_delay = trigger_delay
        self._settle_delay = settle_delay
        self._retry_delay = retry_delay
        self._attempts = max(1, attempts)
        self._controller = Controller()

    def copy_selection(self) -> None:
        with self._controller.pressed(Key.ctrl):
            self._controller.press("c")
            self._controller.release("c")

    def read_text(self) -> str:
        try:
            return pyperclip.paste().strip()
        except pyperclip.PyperclipException:
            return ""

    def capture_selection(self) -> str:
        original_text = self.read_text()
        sentinel = f"learning-agent-capture-{uuid.uuid4().hex}"

        try:
            pyperclip.copy(sentinel)
        except pyperclip.PyperclipException:
            pass

        if self._trigger_delay:
            time.sleep(self._trigger_delay)

        for attempt in range(self._attempts):
            self.copy_selection()
            captured = self._wait_for_clipboard_change(sentinel)
            if captured is not None:
                return captured

            if attempt < self._attempts - 1 and self._retry_delay:
                time.sleep(self._retry_delay)

        try:
            pyperclip.copy(original_text)
        except pyperclip.PyperclipException:
            pass
        return ""

    def _wait_for_clipboard_change(self, sentinel: str) -> str | None:
        deadline = time.monotonic() + self._settle_delay

        while True:
            current_text = self.read_text()
            if current_text != sentinel:
                return current_text

            if time.monotonic() >= deadline:
                return None

            time.sleep(0.01)

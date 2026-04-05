from __future__ import annotations

from main import HotkeyActionRouter
from prompts.templates import PromptMode


class FakeController:
    def __init__(self, busy: bool = False) -> None:
        self.is_busy = busy
        self.create_session_calls = 0
        self.handle_hotkey_calls: list[PromptMode] = []
        self.toggle_language_calls = 0

    def create_session(self) -> None:
        self.create_session_calls += 1

    def handle_hotkey(self, mode: PromptMode) -> None:
        self.handle_hotkey_calls.append(mode)

    def toggle_target_language(self) -> None:
        self.toggle_language_calls += 1


class FakeWindow:
    def __init__(self, pending_new_session: bool = False) -> None:
        self.pending_new_session = pending_new_session
        self.request_exit_calls = 0

    def consume_new_session_request(self) -> bool:
        pending = self.pending_new_session
        self.pending_new_session = False
        return pending

    def request_exit(self) -> None:
        self.request_exit_calls += 1


def test_hotkey_router_creates_new_session_after_minimize() -> None:
    controller = FakeController()
    window = FakeWindow(pending_new_session=True)
    router = HotkeyActionRouter(controller, window)

    router.handle_action(PromptMode.EXPLAIN)

    assert controller.create_session_calls == 1
    assert controller.handle_hotkey_calls == [PromptMode.EXPLAIN]
    assert window.pending_new_session is False


def test_hotkey_router_does_not_create_session_when_window_not_minimized() -> None:
    controller = FakeController()
    window = FakeWindow(pending_new_session=False)
    router = HotkeyActionRouter(controller, window)

    router.handle_action(PromptMode.DEFINITION)

    assert controller.create_session_calls == 0
    assert controller.handle_hotkey_calls == [PromptMode.DEFINITION]


def test_hotkey_router_leaves_pending_session_when_busy() -> None:
    controller = FakeController(busy=True)
    window = FakeWindow(pending_new_session=True)
    router = HotkeyActionRouter(controller, window)

    router.handle_action(PromptMode.SUMMARY)

    assert controller.create_session_calls == 0
    assert controller.handle_hotkey_calls == [PromptMode.SUMMARY]
    assert window.pending_new_session is True

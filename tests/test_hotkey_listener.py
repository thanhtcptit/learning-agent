from __future__ import annotations

import core.hotkey as hotkey_module

from prompts.templates import PromptMode

from core.hotkey import EXIT_HOTKEY_ACTION, TOGGLE_LANGUAGE_HOTKEY_ACTION, GlobalHotkeyListener


def test_windows_hotkey_backend_initializes_message_queue_before_registration(monkeypatch) -> None:
    events: list[str] = []

    class FakeUser32:
        def PeekMessageW(self, *args) -> int:
            events.append("peek")
            return 1

        def RegisterHotKey(self, *args) -> int:
            events.append("register")
            return 1

        def UnregisterHotKey(self, *args) -> int:
            events.append("unregister")
            return 1

        def PostThreadMessageW(self, *args) -> int:
            events.append("post")
            return 1

        def GetMessageW(self, *args) -> int:
            events.append("get-message")
            return 0

    class FakeKernel32:
        def GetCurrentThreadId(self) -> int:
            events.append("thread-id")
            return 123

    class FakeWindll:
        user32 = FakeUser32()
        kernel32 = FakeKernel32()

    monkeypatch.setattr(hotkey_module.ctypes, "windll", FakeWindll())

    backend = hotkey_module._WindowsHotkeyBackend({"<alt>+d": PromptMode.DEFINITION}, lambda _mode: None)
    backend._stop_event.set()

    backend._run()

    assert events == ["thread-id", "peek", "register", "unregister"]


def test_global_hotkey_listener_toggles_running_state(monkeypatch) -> None:
    events: list[str] = []

    class FakeBackend:
        def __init__(self, hotkey_map, trigger) -> None:
            self.hotkey_map = dict(hotkey_map)
            self.trigger = trigger
            self.start_calls = 0
            self.stop_calls = 0

        def start(self) -> None:
            self.start_calls += 1
            events.append("start")

        def stop(self) -> None:
            self.stop_calls += 1
            events.append("stop")

        def fire(self, action) -> None:
            self.trigger(action)

    backend_holder: dict[str, FakeBackend] = {}

    def fake_backend_factory(hotkey_map, trigger):
        backend = FakeBackend(hotkey_map, trigger)
        backend_holder["backend"] = backend
        return backend

    monkeypatch.setattr("core.hotkey._create_hotkey_backend", fake_backend_factory)

    listener = GlobalHotkeyListener()
    statuses: list[str] = []
    triggered: list[object] = []
    listener.status_changed.connect(statuses.append)
    listener.hotkey_triggered.connect(triggered.append)

    assert listener.is_running is False

    listener.start()
    assert listener.is_running is True

    backend = backend_holder["backend"]
    assert backend.start_calls == 1
    assert backend.hotkey_map["<alt>+d"] == PromptMode.DEFINITION
    assert backend.hotkey_map["<alt>+e"] == PromptMode.EXPLAIN
    assert backend.hotkey_map["<alt>+s"] == PromptMode.SUMMARY
    assert backend.hotkey_map["<alt>+l"] == TOGGLE_LANGUAGE_HOTKEY_ACTION
    assert backend.hotkey_map["<alt>+x"] == EXIT_HOTKEY_ACTION

    backend.trigger(PromptMode.EXPLAIN)
    backend.fire(TOGGLE_LANGUAGE_HOTKEY_ACTION)
    backend.fire(EXIT_HOTKEY_ACTION)

    listener.stop()
    assert listener.is_running is False

    assert backend.stop_calls == 1

    assert events == ["start", "stop"]
    assert triggered == [PromptMode.EXPLAIN, TOGGLE_LANGUAGE_HOTKEY_ACTION, EXIT_HOTKEY_ACTION]
    assert statuses == ["Hotkeys active", "Hotkeys stopped"]

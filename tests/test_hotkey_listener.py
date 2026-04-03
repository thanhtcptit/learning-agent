from __future__ import annotations

from prompts.templates import PromptMode

from core.hotkey import GlobalHotkeyListener


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

    backend_holder: dict[str, FakeBackend] = {}

    def fake_backend_factory(hotkey_map, trigger):
        backend = FakeBackend(hotkey_map, trigger)
        backend_holder["backend"] = backend
        return backend

    monkeypatch.setattr("core.hotkey._create_hotkey_backend", fake_backend_factory)

    listener = GlobalHotkeyListener()
    statuses: list[str] = []
    triggered: list[PromptMode] = []
    listener.status_changed.connect(statuses.append)
    listener.hotkey_triggered.connect(triggered.append)

    assert listener.is_running is False

    listener.start()
    assert listener.is_running is True

    backend = backend_holder["backend"]
    assert backend.start_calls == 1
    backend.trigger(PromptMode.SIMPLE)

    listener.stop()
    assert listener.is_running is False

    assert backend.stop_calls == 1

    assert events == ["start", "stop"]
    assert triggered == [PromptMode.SIMPLE]
    assert statuses == ["Hotkeys active", "Hotkeys stopped"]

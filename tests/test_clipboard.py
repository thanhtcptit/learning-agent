from __future__ import annotations

from core.clipboard import ClipboardService


class DummyController:
    pass


def test_capture_selection_reads_changed_clipboard(monkeypatch) -> None:
    monkeypatch.setattr("core.clipboard.Controller", DummyController)

    clipboard_state = {"text": "clipboard text"}
    events: list[tuple[str, object]] = []

    def fake_sleep(delay: float) -> None:
        events.append(("sleep", delay))

    def fake_paste() -> str:
        events.append(("paste", ""))
        return clipboard_state["text"]

    def fake_copy(value: str) -> None:
        events.append(("copy", value))
        clipboard_state["text"] = value

    def fake_copy_selection() -> None:
        events.append(("copy_selection", ""))
        clipboard_state["text"] = "highlighted text"

    monkeypatch.setattr("core.clipboard.time.sleep", fake_sleep)
    monkeypatch.setattr("core.clipboard.pyperclip.paste", fake_paste)
    monkeypatch.setattr("core.clipboard.pyperclip.copy", fake_copy)

    service = ClipboardService(trigger_delay=0.1, settle_delay=0.0, retry_delay=0.0, attempts=1)
    monkeypatch.setattr(service, "copy_selection", fake_copy_selection)

    assert service.capture_selection() == "highlighted text"
    assert clipboard_state["text"] == "highlighted text"
    assert events[0] == ("paste", "")
    assert events[1][0] == "copy"
    assert events[2] == ("sleep", 0.1)
    assert ("copy_selection", "") in events


def test_capture_selection_restores_original_clipboard_when_copy_fails(monkeypatch) -> None:
    monkeypatch.setattr("core.clipboard.Controller", DummyController)

    clipboard_state = {"text": "clipboard text"}
    events: list[tuple[str, object]] = []

    def fake_sleep(delay: float) -> None:
        events.append(("sleep", delay))

    def fake_paste() -> str:
        events.append(("paste", ""))
        return clipboard_state["text"]

    def fake_copy(value: str) -> None:
        events.append(("copy", value))
        clipboard_state["text"] = value

    def fake_copy_selection() -> None:
        events.append(("copy_selection", ""))

    monkeypatch.setattr("core.clipboard.time.sleep", fake_sleep)
    monkeypatch.setattr("core.clipboard.pyperclip.paste", fake_paste)
    monkeypatch.setattr("core.clipboard.pyperclip.copy", fake_copy)

    service = ClipboardService(trigger_delay=0.0, settle_delay=0.0, retry_delay=0.0, attempts=1)
    monkeypatch.setattr(service, "copy_selection", fake_copy_selection)

    assert service.capture_selection() == ""
    assert clipboard_state["text"] == "clipboard text"
    assert events.count(("copy_selection", "")) == 1

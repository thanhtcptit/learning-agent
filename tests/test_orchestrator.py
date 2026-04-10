from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from core.orchestrator import AppController
from core.config import ProviderConfig
from prompts.templates import PromptMode


@dataclass
class ImmediateThread:
    target: object
    args: tuple = ()
    kwargs: dict | None = None
    daemon: bool = False

    def start(self) -> None:
        self.target(*self.args, **(self.kwargs or {}))


class FakeProvider:
    model = "fake-model"

    def __init__(self) -> None:
        self.requests = []

    def stream_chat(self, messages, *, temperature=None, cancel_event=None):
        self.requests.append((list(messages), temperature))
        yield "Hel"
        yield "lo"


class CancellableProvider:
    model = "fake-model"

    def __init__(self) -> None:
        self.requests = []
        self.cancel_calls = 0
        self.started = threading.Event()
        self.stopped = threading.Event()

    def stream_chat(self, messages, *, temperature=None, cancel_event=None):
        self.requests.append((list(messages), temperature, cancel_event))
        self.started.set()

        while cancel_event is not None and not cancel_event.is_set():
            cancel_event.wait(0.01)
            yield from ()

        self.stopped.set()

    def cancel_current_request(self) -> None:
        self.cancel_calls += 1


class FakeClipboardService:
    def __init__(self, text: str) -> None:
        self.text = text
        self.capture_calls = 0

    def capture_selection(self) -> str:
        self.capture_calls += 1
        return self.text


class FakeScreenOcrService:
    def __init__(self, text: str) -> None:
        self.text = text
        self.capture_calls: list[str | None] = []

    def capture_screen_text(self, selection_text: str | None = None) -> str:
        self.capture_calls.append(selection_text)
        return self.text


class FailingScreenOcrService:
    def __init__(self) -> None:
        self.capture_calls: list[str | None] = []

    def capture_screen_text(self, selection_text: str | None = None) -> str:
        self.capture_calls.append(selection_text)
        raise RuntimeError("OCR failed")


def test_submit_text_appends_user_and_streamed_assistant_message(monkeypatch) -> None:
    monkeypatch.setattr("core.orchestrator.threading.Thread", ImmediateThread)

    provider = FakeProvider()
    controller = AppController(provider)
    observed_messages = []
    controller.message_upserted.connect(observed_messages.append)

    controller.submit_text("Explain vectors", PromptMode.EXPLAIN, "English")

    session = controller.current_session

    assert session.messages[0].role == "user"
    assert session.messages[0].content == "Explain vectors"
    assert session.messages[1].role == "assistant"
    assert session.messages[1].content == "Hello"
    assert provider.requests[0][0][0].content == (
        "You are a learning assistant. Explain the meaning of the following text in English. "
        "Use clear language, intuition, and examples."
    )
    assert provider.requests[0][0][-1].content == "Explain the following text:\n\nExplain vectors"
    assert observed_messages[-1].content == "Hello"


def test_submit_chat_text_uses_normal_chat_mode_and_limits_context(monkeypatch) -> None:
    monkeypatch.setattr("core.orchestrator.threading.Thread", ImmediateThread)

    provider = FakeProvider()
    controller = AppController(provider)

    for index in range(1, 26):
        controller.current_session.append_message("user" if index % 2 else "assistant", f"prior-{index}")

    controller.submit_chat_text("What should I do next?", "English")

    request_messages, _temperature = provider.requests[0]

    assert len(request_messages) == 22
    assert [message.content for message in request_messages[:20]] == [f"prior-{index}" for index in range(6, 26)]
    assert request_messages[20].role == "system"
    assert "helpful conversational assistant" in request_messages[20].content.lower()
    assert request_messages[21].role == "user"
    assert request_messages[21].content == "What should I do next?"


def test_handle_hotkey_captures_clipboard_text(monkeypatch) -> None:
    monkeypatch.setattr("core.orchestrator.threading.Thread", ImmediateThread)

    provider = FakeProvider()
    clipboard = FakeClipboardService("Selection from another app")
    screen_ocr = FakeScreenOcrService("Nearby screen text")
    controller = AppController(provider, clipboard_service=clipboard, screen_ocr_service=screen_ocr)

    controller.handle_hotkey(PromptMode.DEFINITION)

    session = controller.current_session

    assert clipboard.capture_calls == 1
    assert screen_ocr.capture_calls == []
    assert session.messages[0].role == "user"
    assert session.messages[0].content == "Selection from another app"
    assert session.messages[0].screen_context == ""
    assert session.title == "Definition: Selection from another app"
    assert session.messages[1].content == "Hello"
    assert provider.requests[0][0][0].content.startswith("You are a dictionary and language-learning assistant.")
    assert "Vietnamese" in provider.requests[0][0][0].content
    assert provider.requests[0][0][-1].content == "Define the following word or term:\n\nSelection from another app"


def test_handle_hotkey_truncates_session_title_from_long_selection(monkeypatch) -> None:
    monkeypatch.setattr("core.orchestrator.threading.Thread", ImmediateThread)

    provider = FakeProvider()
    clipboard = FakeClipboardService("x" * 100)
    controller = AppController(provider, clipboard_service=clipboard)

    controller.handle_hotkey(PromptMode.SUMMARY)

    assert controller.current_session.title == "Summary: " + ("x" * 45) + "..."


def test_handle_hotkey_includes_screen_ocr_context_for_definition_mode(monkeypatch) -> None:
    monkeypatch.setattr("core.orchestrator.threading.Thread", ImmediateThread)

    provider = FakeProvider()
    clipboard = FakeClipboardService("Highlighted text")
    screen_ocr = FakeScreenOcrService("Toolbar label: Linear algebra")
    controller = AppController(
        provider,
        clipboard_service=clipboard,
        screen_ocr_service=screen_ocr,
        screen_ocr_enabled=True,
    )

    controller.handle_hotkey(PromptMode.DEFINITION)

    assert clipboard.capture_calls == 1
    assert screen_ocr.capture_calls == ["Highlighted text"]
    assert controller.current_session.messages[0].screen_context == "Toolbar label: Linear algebra"
    user_prompt = provider.requests[0][0][-1].content
    assert "Screen OCR context from the current screen" in user_prompt
    assert "Toolbar label: Linear algebra" in user_prompt
    assert user_prompt.endswith("Define the following word or term:\n\nHighlighted text")


def test_handle_hotkey_skips_screen_ocr_for_explain_mode(monkeypatch) -> None:
    monkeypatch.setattr("core.orchestrator.threading.Thread", ImmediateThread)

    provider = FakeProvider()
    clipboard = FakeClipboardService("Highlighted text")
    screen_ocr = FakeScreenOcrService("Toolbar label: Linear algebra")
    controller = AppController(
        provider,
        clipboard_service=clipboard,
        screen_ocr_service=screen_ocr,
        screen_ocr_enabled=True,
    )

    controller.handle_hotkey(PromptMode.EXPLAIN)

    assert clipboard.capture_calls == 1
    assert screen_ocr.capture_calls == []
    assert controller.current_session.messages[0].screen_context == ""
    user_prompt = provider.requests[0][0][-1].content
    assert "Screen OCR context from the current screen" not in user_prompt
    assert user_prompt.endswith("Explain the following text:\n\nHighlighted text")


def test_handle_hotkey_skips_screen_ocr_for_summary_mode(monkeypatch) -> None:
    monkeypatch.setattr("core.orchestrator.threading.Thread", ImmediateThread)

    provider = FakeProvider()
    clipboard = FakeClipboardService("Highlighted text")
    screen_ocr = FailingScreenOcrService()
    controller = AppController(
        provider,
        clipboard_service=clipboard,
        screen_ocr_service=screen_ocr,
        screen_ocr_enabled=True,
    )

    controller.handle_hotkey(PromptMode.SUMMARY)

    assert clipboard.capture_calls == 1
    assert screen_ocr.capture_calls == []
    assert controller.current_session.messages[0].screen_context == ""
    user_prompt = provider.requests[0][0][-1].content
    assert "Screen OCR context from the current screen" not in user_prompt
    assert user_prompt.endswith("Summarize the following text:\n\nHighlighted text")


def test_delete_session_updates_current_session(monkeypatch) -> None:
    monkeypatch.setattr("core.orchestrator.threading.Thread", ImmediateThread)

    provider = FakeProvider()
    controller = AppController(provider)
    first_session = controller.current_session
    second_session = controller.create_session("Second")
    third_session = controller.create_session("Third")

    observed_current_sessions = []
    controller.current_session_changed.connect(observed_current_sessions.append)

    deleted_session = controller.delete_session(second_session.id)

    assert deleted_session.id == second_session.id
    assert [session.title for session in controller.sessions] == [first_session.title, "Third"]
    assert controller.current_session.id == third_session.id
    assert observed_current_sessions[-1].id == third_session.id


def test_set_provider_switches_active_config() -> None:
    initial_config = ProviderConfig(provider="openrouter", model="qwen/qwen3.6-plus:free", family="qwen", name="qwen3.6-plus")
    next_config = ProviderConfig(provider="openai", model="gpt-4.1", family="gpt", name="gpt-4.1")

    created_configs = []
    statuses: list[str] = []

    def factory(config: ProviderConfig) -> FakeProvider:
        created_configs.append(config)
        return FakeProvider()

    controller = AppController(FakeProvider(), provider_config=initial_config, provider_factory=factory)
    controller.status_changed.connect(statuses.append)
    controller.set_provider(next_config)

    assert controller.provider_config == next_config
    assert created_configs == [next_config]
    assert statuses[-1] == "LLM set to gpt-4.1 (openai)"


def test_toggle_target_language_switches_between_preferred_and_english() -> None:
    controller = AppController(FakeProvider(), target_language="Vietnamese")
    current_languages: list[str] = []
    preferred_languages: list[str] = []

    controller.current_language_changed.connect(current_languages.append)
    controller.preferred_language_changed.connect(preferred_languages.append)

    assert controller.preferred_language == "Vietnamese"
    assert controller.target_language == "Vietnamese"

    controller.toggle_target_language()
    assert controller.preferred_language == "Vietnamese"
    assert controller.target_language == "English"

    controller.toggle_target_language()
    assert controller.target_language == "Vietnamese"
    assert current_languages == ["English", "Vietnamese"]
    assert preferred_languages == []


def test_set_target_language_updates_preferred_and_current_language() -> None:
    controller = AppController(FakeProvider(), target_language="Vietnamese")
    current_languages: list[str] = []
    preferred_languages: list[str] = []

    controller.current_language_changed.connect(current_languages.append)
    controller.preferred_language_changed.connect(preferred_languages.append)

    controller.set_target_language("French")

    assert controller.preferred_language == "French"
    assert controller.target_language == "French"
    assert current_languages == ["French"]
    assert preferred_languages == ["French"]


def test_stop_current_request_cancels_active_stream() -> None:
    provider = CancellableProvider()
    controller = AppController(provider)

    controller.submit_text("Explain vectors", PromptMode.EXPLAIN, "English")

    assert provider.started.wait(timeout=1.0)
    assert controller.is_busy is True

    controller.stop_current_request()

    assert provider.cancel_calls == 1
    assert provider.stopped.wait(timeout=1.0)

    deadline = time.monotonic() + 1.0
    while controller.is_busy and time.monotonic() < deadline:
        time.sleep(0.01)

    assert controller.is_busy is False


def test_stop_current_request_excludes_cancelled_turn_from_followup_context() -> None:
    provider = CancellableProvider()
    controller = AppController(provider)

    controller.submit_chat_text("First question", "English")

    assert provider.started.wait(timeout=1.0)

    controller.stop_current_request()

    assert provider.stopped.wait(timeout=1.0)

    deadline = time.monotonic() + 1.0
    while controller.is_busy and time.monotonic() < deadline:
        time.sleep(0.01)

    assert controller.is_busy is False

    followup_provider = FakeProvider()
    followup_controller = AppController(followup_provider, session_manager=controller.session_manager)
    followup_controller.submit_chat_text("Second question", "English")

    request_messages, _temperature = followup_provider.requests[0]

    assert len(request_messages) == 2
    assert request_messages[0].role == "system"
    assert request_messages[1].role == "user"
    assert request_messages[1].content == "Second question"

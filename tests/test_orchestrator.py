from __future__ import annotations

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

    def stream_chat(self, messages, *, temperature=None):
        self.requests.append((list(messages), temperature))
        yield "Hel"
        yield "lo"


class FakeClipboardService:
    def __init__(self, text: str) -> None:
        self.text = text
        self.capture_calls = 0

    def capture_selection(self) -> str:
        self.capture_calls += 1
        return self.text


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


def test_handle_hotkey_captures_clipboard_text(monkeypatch) -> None:
    monkeypatch.setattr("core.orchestrator.threading.Thread", ImmediateThread)

    provider = FakeProvider()
    clipboard = FakeClipboardService("Selection from another app")
    controller = AppController(provider, clipboard_service=clipboard)

    controller.handle_hotkey(PromptMode.DEFINITION)

    session = controller.current_session

    assert clipboard.capture_calls == 1
    assert session.messages[0].role == "user"
    assert session.messages[0].content == "Selection from another app"
    assert session.messages[1].content == "Hello"
    assert provider.requests[0][0][0].content.startswith("You are a dictionary and language-learning assistant.")
    assert "Vietnamese" in provider.requests[0][0][0].content
    assert provider.requests[0][0][-1].content == "Define the following word or term:\n\nSelection from another app"


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

    def factory(config: ProviderConfig) -> FakeProvider:
        created_configs.append(config)
        return FakeProvider()

    controller = AppController(FakeProvider(), provider_config=initial_config, provider_factory=factory)
    controller.set_provider(next_config)

    assert controller.provider_config == next_config
    assert created_configs == [next_config]


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

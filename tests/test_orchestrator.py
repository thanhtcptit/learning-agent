from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from core.audio_recorder import RecordedAudio, VoiceCaptureError
from core.orchestrator import AppController
from core.config import ProviderConfig
from core.voice_catalog import DEFAULT_VIETNAMESE_STT_MODEL_ID, DEFAULT_VIETNAMESE_TTS_MODEL_ID, DEFAULT_VIETNAMESE_TTS_VOICE_NAME, F5_VIETNAMESE_TTS_MODEL_ID, PHOWHISPER_MEDIUM_STT_MODEL_ID, VIETNAMESE_TTS_VOICE_CHOICES
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


class BrowserActionProvider:
    model = "fake-model"

    def __init__(self) -> None:
        self.requests = []

    def stream_chat(self, messages, *, temperature=None, cancel_event=None):
        self.requests.append((list(messages), temperature))
        yield (
            "Playing the first YouTube result.\n"
            '<<<BROWSER_ACTION>>>{"action": "search_and_play", "query": "piano music"}<<<END_ACTION>>>'
        )


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


class FakeVoiceRecorder:
    def __init__(self, recordings: RecordedAudio | list[RecordedAudio | Exception]) -> None:
        if isinstance(recordings, list):
            self._recordings = list(recordings)
        else:
            self._recordings = [recordings]
        self.record_calls = 0
        self.record_kwargs: list[dict[str, object | None]] = []
        self.cancel_calls = 0

    def record_until_silence(self, cancel_event=None, *, initial_timeout_seconds=None) -> RecordedAudio:
        self.record_calls += 1
        self.last_cancel_event = cancel_event
        self.record_kwargs.append(
            {
                "cancel_event": cancel_event,
                "initial_timeout_seconds": initial_timeout_seconds,
            }
        )

        if not self._recordings:
            raise VoiceCaptureError("No speech detected.")

        next_result = self._recordings.pop(0)
        if isinstance(next_result, Exception):
            raise next_result

        return next_result

    def cancel_current_request(self) -> None:
        self.cancel_calls += 1


class FakeSttService:
    def __init__(self, transcript: str) -> None:
        self.transcript = transcript
        self.calls: list[tuple[RecordedAudio, object | None, str | None]] = []

    def transcribe(self, recording: RecordedAudio, *, cancel_event=None, language=None) -> str:
        self.calls.append((recording, cancel_event, language))
        return self.transcript


class FakeTtsService:
    def __init__(self) -> None:
        self.speak_calls: list[tuple[str, object | None, str | None]] = []
        self.stop_calls = 0
        self.selected_vietnamese_model_id = DEFAULT_VIETNAMESE_TTS_MODEL_ID
        self.selected_vietnamese_voice_name = DEFAULT_VIETNAMESE_TTS_VOICE_NAME
        self.set_calls: list[str] = []
        self.voice_name_calls: list[str] = []

    def speak(self, text: str, *, cancel_event=None, language=None) -> None:
        self.speak_calls.append((text, cancel_event, language))

    def stop(self) -> None:
        self.stop_calls += 1

    def set_selected_vietnamese_model_id(self, model_id: str) -> None:
        self.selected_vietnamese_model_id = model_id
        self.set_calls.append(model_id)

    def set_selected_vietnamese_voice_name(self, voice_name: str) -> None:
        self.selected_vietnamese_voice_name = voice_name
        self.voice_name_calls.append(voice_name)


class FakeVoiceSttService(FakeSttService):
    def __init__(self, transcript: str) -> None:
        super().__init__(transcript)
        self.selected_vietnamese_model_id = DEFAULT_VIETNAMESE_STT_MODEL_ID
        self.set_calls: list[str] = []

    def set_selected_vietnamese_model_id(self, model_id: str) -> None:
        self.selected_vietnamese_model_id = model_id
        self.set_calls.append(model_id)


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


def test_handle_hotkey_includes_screen_ocr_context_for_explain_mode(monkeypatch) -> None:
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
    assert screen_ocr.capture_calls == ["Highlighted text"]
    assert controller.current_session.messages[0].screen_context == "Toolbar label: Linear algebra"
    user_prompt = provider.requests[0][0][-1].content
    assert "Screen OCR context from the current screen" in user_prompt
    assert "Toolbar label: Linear algebra" in user_prompt
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


def test_handle_voice_hotkey_records_transcribes_streams_and_speaks(monkeypatch) -> None:
    monkeypatch.setattr("core.orchestrator.threading.Thread", ImmediateThread)

    provider = FakeProvider()
    recording = RecordedAudio(samples=np.ones(1600, dtype=np.float32), sample_rate=16000)
    voice_recorder = FakeVoiceRecorder([recording, VoiceCaptureError("No speech detected.")])
    stt_service = FakeSttService("Hello from voice")
    tts_service = FakeTtsService()
    controller = AppController(
        provider,
        voice_recorder=voice_recorder,
        stt_service=stt_service,
        tts_service=tts_service,
    )

    statuses: list[str] = []
    controller.status_changed.connect(statuses.append)

    controller.handle_voice_hotkey()

    session = controller.current_session

    assert voice_recorder.record_calls == 2
    assert voice_recorder.record_kwargs[0]["initial_timeout_seconds"] is None
    assert voice_recorder.record_kwargs[1]["initial_timeout_seconds"] == controller.VOICE_FOLLOW_UP_IDLE_TIMEOUT_SECONDS
    assert len(stt_service.calls) == 1
    assert stt_service.calls[0][0].sample_rate == 16000
    assert stt_service.calls[0][2] == "Vietnamese"
    assert tts_service.speak_calls == [("Hello", stt_service.calls[0][1], "Vietnamese")]
    assert session.messages[0].role == "user"
    assert session.messages[0].mode == "voice"
    assert session.messages[0].content == "Hello from voice"
    assert session.title.startswith("Voice: ")
    assert session.messages[1].role == "assistant"
    assert session.messages[1].mode == "voice"
    assert session.messages[1].content == "Hello"
    assert provider.requests[0][0][0].content.startswith("You are a voice assistant.")
    assert "plain natural speech" in provider.requests[0][0][0].content.lower()
    assert "avoid markdown" in provider.requests[0][0][0].content.lower()
    assert provider.requests[0][0][-1].content == "Hello from voice"
    assert "Listening..." in statuses
    assert statuses.count("Listening...") == 2
    assert "Transcribing..." in statuses
    assert "Thinking..." in statuses
    assert "Speaking..." in statuses
    assert statuses[-1] == "Ready"


def test_handle_voice_hotkey_skips_tts_for_youtube_play_action(monkeypatch) -> None:
    monkeypatch.setattr("core.orchestrator.threading.Thread", ImmediateThread)
    monkeypatch.setattr("core.browser_service._fetch_first_youtube_video_url", lambda query: "https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    monkeypatch.setattr("core.browser_service.webbrowser.open", lambda url: None)

    provider = BrowserActionProvider()
    recording = RecordedAudio(samples=np.ones(1600, dtype=np.float32), sample_rate=16000)
    voice_recorder = FakeVoiceRecorder([recording])
    stt_service = FakeSttService("Play piano music on YouTube")
    tts_service = FakeTtsService()
    controller = AppController(
        provider,
        voice_recorder=voice_recorder,
        stt_service=stt_service,
        tts_service=tts_service,
    )

    statuses: list[str] = []
    controller.status_changed.connect(statuses.append)

    controller.handle_voice_hotkey()

    session = controller.current_session

    assert voice_recorder.record_calls == 1
    assert tts_service.speak_calls == []
    assert session.messages[1].role == "assistant"
    assert session.messages[1].content == "Playing the first YouTube result."
    assert "Speaking..." not in statuses
    assert statuses[-1] == "Ready"


def test_set_voice_model_ids_updates_selected_services() -> None:
    provider = FakeProvider()
    stt_service = FakeVoiceSttService("Hello from voice")
    tts_service = FakeTtsService()
    controller = AppController(provider, stt_service=stt_service, tts_service=tts_service)

    statuses: list[str] = []
    controller.status_changed.connect(statuses.append)

    controller.set_voice_stt_model_id(PHOWHISPER_MEDIUM_STT_MODEL_ID)
    controller.set_voice_tts_model_id(F5_VIETNAMESE_TTS_MODEL_ID)

    assert controller.voice_stt_model_id == PHOWHISPER_MEDIUM_STT_MODEL_ID
    assert controller.voice_tts_model_id == F5_VIETNAMESE_TTS_MODEL_ID
    assert stt_service.set_calls == [PHOWHISPER_MEDIUM_STT_MODEL_ID]
    assert tts_service.set_calls == [F5_VIETNAMESE_TTS_MODEL_ID]
    assert statuses[-2] == "Vietnamese STT set to PhoWhisper medium"
    assert statuses[-1] == "Vietnamese TTS set to F5-TTS Vietnamese ViVoice"


def test_set_voice_tts_voice_name_updates_selected_services() -> None:
    provider = FakeProvider()
    stt_service = FakeVoiceSttService("Hello from voice")
    tts_service = FakeTtsService()
    controller = AppController(provider, stt_service=stt_service, tts_service=tts_service)

    statuses: list[str] = []
    controller.status_changed.connect(statuses.append)

    controller.set_voice_tts_voice_name(VIETNAMESE_TTS_VOICE_CHOICES[1].voice_name)

    assert controller.voice_tts_voice_name == VIETNAMESE_TTS_VOICE_CHOICES[1].voice_name
    assert tts_service.voice_name_calls == [VIETNAMESE_TTS_VOICE_CHOICES[1].voice_name]
    assert statuses[-1] == f"Voice preset set to {VIETNAMESE_TTS_VOICE_CHOICES[1].voice_name}"


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


# ---------------------------------------------------------------------------
# Wake-word integration
# ---------------------------------------------------------------------------


class FakeWakeWordService:
    def __init__(self) -> None:
        self.start_calls = 0
        self.stop_calls = 0
        self.pause_calls = 0
        self.resume_calls = 0
        self._listening = False
        self._stop_event = threading.Event()

    class _Signal:
        def __init__(self):
            self._cb = None

        def connect(self, cb):
            self._cb = cb

        def emit(self):
            if self._cb:
                self._cb()

    wake_word_detected = _Signal()
    listening_state_changed = _Signal()

    @property
    def wake_word(self) -> str:
        return "Mario"

    @property
    def is_listening(self) -> bool:
        return self._listening

    def start_listening(self) -> None:
        self.start_calls += 1
        self._listening = True
        self._stop_event.clear()

    def stop_listening(self) -> None:
        self.stop_calls += 1
        self._listening = False
        self._stop_event.set()

    def pause(self) -> None:
        self.pause_calls += 1

    def resume(self) -> None:
        self.resume_calls += 1


def test_toggle_wake_word_starts_and_stops_service() -> None:
    provider = FakeProvider()
    wake_svc = FakeWakeWordService()
    controller = AppController(provider, wake_word_service=wake_svc)

    statuses: list[str] = []
    controller.status_changed.connect(statuses.append)
    active_states: list[bool] = []
    controller.wake_word_active_changed.connect(active_states.append)

    controller.toggle_wake_word()
    assert wake_svc.start_calls == 1
    assert active_states == [True]
    assert any("Mario" in s for s in statuses)

    controller.toggle_wake_word()
    assert wake_svc.stop_calls == 1
    assert active_states == [True, False]


def test_toggle_wake_word_without_service_emits_status() -> None:
    provider = FakeProvider()
    controller = AppController(provider)

    statuses: list[str] = []
    controller.status_changed.connect(statuses.append)

    controller.toggle_wake_word()
    assert any("not available" in s for s in statuses)


def test_wake_word_detected_triggers_voice_mode(monkeypatch) -> None:
    monkeypatch.setattr("core.orchestrator.threading.Thread", ImmediateThread)

    recording = RecordedAudio(samples=np.zeros(16000, dtype=np.float32), sample_rate=16000)
    provider = FakeProvider()
    voice_recorder = FakeVoiceRecorder(recording)
    stt_service = FakeSttService("Hello from wake word")
    tts_service = FakeTtsService()
    wake_svc = FakeWakeWordService()

    controller = AppController(
        provider,
        voice_recorder=voice_recorder,
        stt_service=stt_service,
        tts_service=tts_service,
        wake_word_service=wake_svc,
    )

    # Simulate wake word detection
    controller._on_wake_word_detected()

    assert voice_recorder.record_calls >= 1
    assert stt_service.calls[0][0] == recording
    assert tts_service.speak_calls[0][0] == "Hello"
    # resume should have been called in the finally block
    assert wake_svc.resume_calls >= 1


def test_shutdown_stops_wake_word_service() -> None:
    provider = FakeProvider()
    wake_svc = FakeWakeWordService()
    wake_svc.start_listening()
    controller = AppController(provider, wake_word_service=wake_svc)

    controller.shutdown()
    assert wake_svc.stop_calls >= 1

from __future__ import annotations

from pathlib import Path

import core.voice_services as voice_services
from core.voice_catalog import (
    DEFAULT_VIETNAMESE_STT_MODEL_ID,
    DEFAULT_VIETNAMESE_TTS_MODEL_ID,
    DEFAULT_VIETNAMESE_TTS_VOICE_NAME,
    F5_VIETNAMESE_TTS_MODEL_ID,
    PHOWHISPER_MEDIUM_STT_MODEL_ID,
    VIENEU_TTS_03B_MODEL_ID,
    VIENEU_TTS_03B_Q4_GGUF_MODEL_ID,
    VIETNAMESE_TTS_VOICE_CHOICES,
)
from core.voice_services import LanguageAwareSttService, LanguageAwareTtsService


class FakeSttService:
    def __init__(self, label: str) -> None:
        self.label = label
        self.calls: list[tuple[object, object | None, str | None]] = []

    def transcribe(self, recording, *, cancel_event=None, language=None) -> str:
        self.calls.append((recording, cancel_event, language))
        return self.label


class RecordingSttService:
    instances: list["RecordingSttService"] = []

    def __init__(self, config) -> None:
        self.config = config
        self.model_id = config.model_name_or_path
        self.calls: list[tuple[object, object | None, str | None]] = []
        RecordingSttService.instances.append(self)

    def transcribe(self, recording, *, cancel_event=None, language=None) -> str:
        self.calls.append((recording, cancel_event, language))
        return "zipformer"


class FakeTtsService:
    def __init__(self, label: str) -> None:
        self.label = label
        self.calls: list[tuple[str, object | None]] = []
        self.stop_calls = 0
        self.selected_vietnamese_voice_name = DEFAULT_VIETNAMESE_TTS_VOICE_NAME
        self.voice_name_calls: list[str] = []

    def speak(self, text: str, *, cancel_event=None, language=None) -> None:
        self.calls.append((text, cancel_event))

    def stop(self) -> None:
        self.stop_calls += 1

    def set_selected_vietnamese_voice_name(self, voice_name: str) -> None:
        self.selected_vietnamese_voice_name = voice_name
        self.voice_name_calls.append(voice_name)


class RecordingService:
    instances: list["RecordingService"] = []

    def __init__(self, config) -> None:
        self.config = config
        self.kind = config.__class__.__name__
        self.model_id = getattr(config, "model_repo_id", getattr(config, "backbone_repo", self.kind))
        self.calls: list[tuple[str, object | None]] = []
        self.stop_calls = 0
        RecordingService.instances.append(self)

    def speak(self, text: str, *, cancel_event=None, language=None) -> None:
        self.calls.append((text, cancel_event))

    def stop(self) -> None:
        self.stop_calls += 1


def test_language_aware_stt_uses_vietnamese_engine_for_vietnamese_language() -> None:
    default_service = FakeSttService("default")
    vietnamese_service = FakeSttService("vietnamese")
    service = LanguageAwareSttService(default_service, vietnamese_service)

    result = service.transcribe("audio", language="Vietnamese")

    assert result == "vietnamese"
    assert default_service.calls == []
    assert vietnamese_service.calls == [("audio", None, "vi")]


def test_language_aware_stt_uses_default_engine_for_other_languages() -> None:
    default_service = FakeSttService("default")
    vietnamese_service = FakeSttService("vietnamese")
    service = LanguageAwareSttService(default_service, vietnamese_service)

    result = service.transcribe("audio", language="English")

    assert result == "default"
    assert default_service.calls == [("audio", None, "en")]


def test_language_aware_tts_replaces_urls_and_links_with_web_link_before_speaking() -> None:
    default_service = FakeTtsService("default")
    vietnamese_service = FakeTtsService("vietnamese")
    service = LanguageAwareTtsService(default_service, vietnamese_service)

    service.speak("See https://example.com and [Docs](https://docs.example.com).", language="Vietnamese")

    assert default_service.calls == []
    assert vietnamese_service.calls == [("See web link and web link.", None)]


def test_language_aware_tts_turns_url_only_responses_into_web_link() -> None:
    default_service = FakeTtsService("default")
    vietnamese_service = FakeTtsService("vietnamese")
    service = LanguageAwareTtsService(default_service, vietnamese_service)

    service.speak("https://example.com", language="Vietnamese")

    assert default_service.calls == []
    assert vietnamese_service.calls == [("web link", None)]


def test_language_aware_tts_uses_vietnamese_engine_for_vietnamese_language() -> None:
    default_service = FakeTtsService("default")
    vietnamese_service = FakeTtsService("vietnamese")
    service = LanguageAwareTtsService(default_service, vietnamese_service)

    service.speak("Hello", language="Vietnamese")
    service.stop()

    assert default_service.calls == []
    assert vietnamese_service.calls == [("Hello", None)]
    assert default_service.stop_calls == 1
    assert vietnamese_service.stop_calls == 1


def test_language_aware_tts_uses_default_engine_for_other_languages() -> None:
    default_service = FakeTtsService("default")
    vietnamese_service = FakeTtsService("vietnamese")
    service = LanguageAwareTtsService(default_service, vietnamese_service)

    service.speak("Hello", language="English")

    assert default_service.calls == [("Hello", None)]
    assert vietnamese_service.calls == []


def test_language_aware_stt_switches_between_vietnamese_models() -> None:
    default_service = FakeSttService("default")
    zipformer_service = FakeSttService("zipformer")
    phowhisper_service = FakeSttService("phowhisper")
    service = LanguageAwareSttService(
        default_service,
        {
            DEFAULT_VIETNAMESE_STT_MODEL_ID: zipformer_service,
            PHOWHISPER_MEDIUM_STT_MODEL_ID: phowhisper_service,
        },
        selected_vietnamese_model_id=PHOWHISPER_MEDIUM_STT_MODEL_ID,
    )

    first_result = service.transcribe("audio-1", language="Vietnamese")
    service.set_selected_vietnamese_model_id(DEFAULT_VIETNAMESE_STT_MODEL_ID)
    second_result = service.transcribe("audio-2", language="Vietnamese")

    assert first_result == "phowhisper"
    assert second_result == "zipformer"
    assert zipformer_service.calls == [("audio-2", None, "vi")]
    assert phowhisper_service.calls == [("audio-1", None, "vi")]
    assert default_service.calls == []


def test_language_aware_tts_switches_between_vietnamese_models() -> None:
    default_service = FakeTtsService("default")
    vieneu_service = FakeTtsService("vieneu")
    vieneu_03b_q4_service = FakeTtsService("vieneu-03b-q4")
    vieneu_03b_service = FakeTtsService("vieneu-03b")
    f5_service = FakeTtsService("f5")
    service = LanguageAwareTtsService(
        default_service,
        {
            DEFAULT_VIETNAMESE_TTS_MODEL_ID: vieneu_service,
            VIENEU_TTS_03B_Q4_GGUF_MODEL_ID: vieneu_03b_q4_service,
            VIENEU_TTS_03B_MODEL_ID: vieneu_03b_service,
            F5_VIETNAMESE_TTS_MODEL_ID: f5_service,
        },
        selected_vietnamese_model_id=F5_VIETNAMESE_TTS_MODEL_ID,
    )

    service.speak("Xin chao", language="Vietnamese")
    service.set_selected_vietnamese_model_id(DEFAULT_VIETNAMESE_TTS_MODEL_ID)
    service.speak("Tam biet", language="Vietnamese")

    assert f5_service.calls == [("Xin chao", None)]
    assert vieneu_service.calls == [("Tam biet", None)]
    assert vieneu_03b_q4_service.calls == []
    assert vieneu_03b_service.calls == []
    assert default_service.calls == []


def test_language_aware_tts_switches_vietnamese_voice_name() -> None:
    default_service = FakeTtsService("default")
    vieneu_service = FakeTtsService("vieneu")
    service = LanguageAwareTtsService(default_service, {DEFAULT_VIETNAMESE_TTS_MODEL_ID: vieneu_service})

    service.speak("Xin chao", language="Vietnamese")
    service.set_selected_vietnamese_voice_name(VIETNAMESE_TTS_VOICE_CHOICES[1].voice_name)
    service.speak("Tam biet", language="Vietnamese")

    assert vieneu_service.voice_name_calls[0] == DEFAULT_VIETNAMESE_TTS_VOICE_NAME
    assert vieneu_service.voice_name_calls[-1] == VIETNAMESE_TTS_VOICE_CHOICES[1].voice_name
    assert default_service.calls == []


def test_language_aware_tts_uses_lazy_vietnamese_factory_once() -> None:
    default_service = FakeTtsService("default")
    created_services: list[FakeTtsService] = []

    def factory() -> FakeTtsService:
        service = FakeTtsService("vietnamese")
        created_services.append(service)
        return service

    service = LanguageAwareTtsService(default_service, vietnamese_service_factory=factory)

    service.speak("Xin chao", language="Vietnamese")
    service.speak("Tam biet", language="Vietnamese")

    assert len(created_services) == 1
    assert default_service.calls == []
    assert created_services[0].calls == [("Xin chao", None), ("Tam biet", None)]


def test_build_default_voice_services_uses_requested_voice_models(monkeypatch, tmp_path: Path) -> None:
    RecordingService.instances.clear()
    RecordingSttService.instances.clear()
    monkeypatch.setattr(voice_services, "ChatterboxTtsService", RecordingService)
    monkeypatch.setattr(voice_services, "ZipformerTransducerSttService", RecordingSttService)
    monkeypatch.setattr(voice_services, "PhoWhisperSttService", RecordingSttService)
    monkeypatch.setattr(voice_services, "VieneuTtsService", RecordingService)
    monkeypatch.setattr(voice_services, "F5TtsService", RecordingService)
    monkeypatch.setattr(voice_services, "_voice_models_root", lambda: tmp_path)

    _recorder, stt_service, tts_service = voice_services.build_default_voice_services(
        voice_stt_model_id=PHOWHISPER_MEDIUM_STT_MODEL_ID,
        voice_tts_model_id=F5_VIETNAMESE_TTS_MODEL_ID,
        voice_tts_voice_name=VIETNAMESE_TTS_VOICE_CHOICES[2].voice_name,
    )

    assert stt_service.selected_vietnamese_model_id == PHOWHISPER_MEDIUM_STT_MODEL_ID
    assert tts_service.selected_vietnamese_model_id == F5_VIETNAMESE_TTS_MODEL_ID
    assert tts_service.selected_vietnamese_voice_name == VIETNAMESE_TTS_VOICE_CHOICES[2].voice_name

    stt_service.transcribe("audio", language="Vietnamese")
    tts_service.speak("Xin chao", language="Vietnamese")

    assert len(RecordingSttService.instances) == 3
    english_service = next(service for service in RecordingSttService.instances if service.model_id == voice_services.DEFAULT_ENGLISH_STT_MODEL_ID)
    zipformer_service = next(service for service in RecordingSttService.instances if service.model_id == DEFAULT_VIETNAMESE_STT_MODEL_ID)
    phowhisper_service = next(service for service in RecordingSttService.instances if service.model_id == PHOWHISPER_MEDIUM_STT_MODEL_ID)
    assert english_service.calls == []
    assert english_service.config.language == "en"
    assert zipformer_service.calls == []
    assert phowhisper_service.calls == [("audio", None, "vi")]

    assert len(RecordingService.instances) == 5
    chatterbox_service = next(service for service in RecordingService.instances if service.model_id == "ChatterboxTtsConfig")
    vieneu_service = next(service for service in RecordingService.instances if service.model_id == DEFAULT_VIETNAMESE_TTS_MODEL_ID)
    vieneu_03b_q4_service = next(service for service in RecordingService.instances if service.model_id == VIENEU_TTS_03B_Q4_GGUF_MODEL_ID)
    vieneu_03b_service = next(service for service in RecordingService.instances if service.model_id == VIENEU_TTS_03B_MODEL_ID)
    f5_service = next(service for service in RecordingService.instances if service.model_id == F5_VIETNAMESE_TTS_MODEL_ID)
    assert chatterbox_service.calls == []
    assert vieneu_service.calls == []
    assert vieneu_03b_q4_service.calls == []
    assert vieneu_03b_service.calls == []
    assert f5_service.calls == [("Xin chao", None)]
    assert vieneu_service.config.voice_name == VIETNAMESE_TTS_VOICE_CHOICES[2].voice_name
    assert vieneu_03b_q4_service.config.backbone_repo == VIENEU_TTS_03B_Q4_GGUF_MODEL_ID
    assert vieneu_03b_service.config.backbone_repo == VIENEU_TTS_03B_MODEL_ID


def test_build_default_voice_services_prefers_cuda_for_vietnamese_tts_when_available(monkeypatch, tmp_path: Path) -> None:
    RecordingService.instances.clear()
    monkeypatch.setattr(voice_services, "ChatterboxTtsService", RecordingService)
    monkeypatch.setattr(voice_services, "VieneuTtsService", RecordingService)
    monkeypatch.setattr(voice_services, "F5TtsService", RecordingService)
    monkeypatch.setattr(voice_services, "_voice_models_root", lambda: tmp_path)
    monkeypatch.setattr(voice_services, "torch", type("FakeTorch", (), {"cuda": type("FakeCuda", (), {"is_available": staticmethod(lambda: True)})()})())

    _recorder, _stt_service, tts_service = voice_services.build_default_voice_services()

    assert tts_service.selected_vietnamese_model_id == DEFAULT_VIETNAMESE_TTS_MODEL_ID
    assert len(RecordingService.instances) == 5

    chatterbox_service = next(service for service in RecordingService.instances if service.model_id == "ChatterboxTtsConfig")
    vienteu_service = next(service for service in RecordingService.instances if service.model_id == DEFAULT_VIETNAMESE_TTS_MODEL_ID)
    vienteu_03b_q4_service = next(service for service in RecordingService.instances if service.model_id == VIENEU_TTS_03B_Q4_GGUF_MODEL_ID)
    vienteu_03b_service = next(service for service in RecordingService.instances if service.model_id == VIENEU_TTS_03B_MODEL_ID)
    f5_service = next(service for service in RecordingService.instances if service.model_id == F5_VIETNAMESE_TTS_MODEL_ID)

    assert chatterbox_service.config.device == "cuda"
    assert vienteu_service.config.backbone_device == "cuda"
    assert vienteu_service.config.codec_device == "cuda"
    assert vienteu_03b_q4_service.config.backbone_device == "cuda"
    assert vienteu_03b_q4_service.config.codec_device == "cuda"
    assert vienteu_03b_service.config.backbone_device == "cuda"
    assert vienteu_03b_service.config.codec_device == "cuda"
    assert f5_service.config.device == "cuda"
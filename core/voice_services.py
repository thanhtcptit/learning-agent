from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable, Mapping

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency during bootstrap
    torch = None

from core.audio_recorder import AudioRecorder, AudioRecorderConfig, RecordedAudio
from core.stt_service import (
    PhoWhisperSttConfig,
    PhoWhisperSttService,
    SttService,
    WhisperSttConfig,
    WhisperSttService,
    ZipformerTransducerSttConfig,
    ZipformerTransducerSttService,
)
from core.tts_service import ChatterboxTtsConfig, ChatterboxTtsService, F5TtsConfig, F5TtsService, TtsService, VieneuTtsConfig, VieneuTtsService
from core.voice_catalog import (
    DEFAULT_VIETNAMESE_STT_MODEL_ID,
    DEFAULT_VIETNAMESE_TTS_MODEL_ID,
    DEFAULT_VIETNAMESE_TTS_VOICE_NAME,
    F5_VIETNAMESE_TTS_MODEL_ID,
    PHOWHISPER_MEDIUM_STT_MODEL_ID,
    VIENEU_TTS_03B_MODEL_ID,
    VIENEU_TTS_03B_Q4_GGUF_MODEL_ID,
    VIETNAMESE_STT_MODEL_CHOICES,
    VIETNAMESE_TTS_MODEL_CHOICES,
    VIETNAMESE_TTS_VOICE_CHOICES,
    resolve_voice_model_id,
    resolve_vietnamese_tts_voice_name,
)


DEFAULT_VIETNAMESE_STT_LANGUAGE = "vi"
DEFAULT_VIETNAMESE_TTS_CODEC_REPO = "neuphonic/distill-neucodec"


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False

    if value is None:
        return default

    return bool(value)


def _coerce_int(value: Any, default: int) -> int:
    if isinstance(value, int):
        return value

    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default

    if value is None:
        return default

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_language(language: str | None) -> str:
    return (language or "").strip().lower()


def _is_vietnamese_language(language: str | None) -> bool:
    normalized = _normalize_language(language)
    return normalized in {"vi", "vi-vn", "vietnamese", "tiếng việt", "tieng viet"}


def _normalize_whisper_language(language: str | None) -> str | None:
    if not _is_vietnamese_language(language):
        normalized = _normalize_language(language)
        if normalized in {"en", "english"}:
            return "en"
        return None

    return DEFAULT_VIETNAMESE_STT_LANGUAGE


_LINK_OR_URL_RE = re.compile(
    r"\[([^\]]+)\]\((?:https?://|www\.)[^\s)]+\)|<(?:https?://|www\.)[^>\s]+>|(?:https?://|www\.)[^\s<>\]]+",
    re.IGNORECASE,
)


def _replace_urls_and_links_with_web_link_for_tts(text: str) -> str:
    cleaned_text = text.strip()
    if not cleaned_text:
        return ""

    if not _LINK_OR_URL_RE.search(cleaned_text):
        return cleaned_text

    def replace_link_or_url(match: re.Match[str]) -> str:
        value = match.group(0)
        if value.startswith("[") or value.startswith("<"):
            return "web link"

        trailing_punctuation = ""
        while value and value[-1] in ".,;:!?)]\"'":
            trailing_punctuation = value[-1] + trailing_punctuation
            value = value[:-1]
        return f"web link{trailing_punctuation}"

    cleaned_text = _LINK_OR_URL_RE.sub(replace_link_or_url, cleaned_text)
    cleaned_text = " ".join(cleaned_text.split())
    cleaned_text = re.sub(r"\s+([,.;:!?])", r"\1", cleaned_text)
    return cleaned_text.strip()


def _voice_models_root() -> Path:
    roaming_root = Path(os.getenv("APPDATA") or (Path.home() / "AppData" / "Roaming"))
    root = roaming_root / "learning-agent" / "voice-models"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_repo_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def _voice_model_cache_dir(model_name_or_path: str) -> Path | None:
    candidate_path = Path(model_name_or_path)
    if candidate_path.exists():
        return None

    cache_dir = _voice_models_root() / _safe_repo_name(model_name_or_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _preferred_tts_device() -> str:
    explicit_device = (os.getenv("LEARNING_AGENT_VOICE_TTS_VI_DEVICE") or os.getenv("LEARNING_AGENT_VOICE_TTS_DEVICE") or "").strip()
    if explicit_device:
        if explicit_device.startswith("gpu"):
            explicit_device = "cuda" + explicit_device[3:]
        return explicit_device

    if torch is not None and torch.cuda.is_available():
        return "cuda"

    return "cpu"


class LanguageAwareSttService:
    def __init__(
        self,
        default_service: SttService,
        vietnamese_service: SttService | Mapping[str, SttService] | None = None,
        *,
        selected_vietnamese_model_id: str = DEFAULT_VIETNAMESE_STT_MODEL_ID,
    ) -> None:
        self._default_service = default_service
        self._vietnamese_services: dict[str, SttService] = self._normalize_vietnamese_services(vietnamese_service)
        self._selected_vietnamese_model_id = resolve_voice_model_id(
            selected_vietnamese_model_id,
            VIETNAMESE_STT_MODEL_CHOICES,
            DEFAULT_VIETNAMESE_STT_MODEL_ID,
        )

    @property
    def selected_vietnamese_model_id(self) -> str:
        return self._selected_vietnamese_model_id

    def set_selected_vietnamese_model_id(self, model_id: str) -> None:
        self._selected_vietnamese_model_id = resolve_voice_model_id(
            model_id,
            VIETNAMESE_STT_MODEL_CHOICES,
            DEFAULT_VIETNAMESE_STT_MODEL_ID,
        )

    def transcribe(self, recording: RecordedAudio | Any, *, cancel_event: Any = None, language: str | None = None) -> str:
        service = self._select_service(language)
        normalized_language = _normalize_whisper_language(language)
        return service.transcribe(recording, cancel_event=cancel_event, language=normalized_language)

    def _select_service(self, language: str | None) -> SttService:
        if _is_vietnamese_language(language):
            service = self._vietnamese_services.get(self._selected_vietnamese_model_id)
            if service is None:
                service = self._vietnamese_services.get(DEFAULT_VIETNAMESE_STT_MODEL_ID)
            if service is None and self._vietnamese_services:
                service = next(iter(self._vietnamese_services.values()))
            if service is not None:
                return service
        return self._default_service

    @staticmethod
    def _normalize_vietnamese_services(
        vietnamese_service: SttService | Mapping[str, SttService] | None,
    ) -> dict[str, SttService]:
        if vietnamese_service is None:
            return {}

        if isinstance(vietnamese_service, Mapping):
            return {str(model_id): service for model_id, service in vietnamese_service.items()}

        return {DEFAULT_VIETNAMESE_STT_MODEL_ID: vietnamese_service}


class LanguageAwareTtsService:
    def __init__(
        self,
        default_service: TtsService,
        vietnamese_service: TtsService | Mapping[str, TtsService] | None = None,
        *,
        selected_vietnamese_model_id: str = DEFAULT_VIETNAMESE_TTS_MODEL_ID,
        selected_vietnamese_voice_name: str | None = None,
        vietnamese_service_factory: Callable[[], TtsService] | None = None,
    ) -> None:
        self._default_service = default_service
        self._vietnamese_services: dict[str, TtsService] = self._normalize_vietnamese_services(vietnamese_service)
        self._selected_vietnamese_model_id = resolve_voice_model_id(
            selected_vietnamese_model_id,
            VIETNAMESE_TTS_MODEL_CHOICES,
            DEFAULT_VIETNAMESE_TTS_MODEL_ID,
        )
        inferred_voice_name = selected_vietnamese_voice_name
        if inferred_voice_name is None:
            inferred_voice_name = self._infer_vietnamese_voice_name(vietnamese_service)
        self._selected_vietnamese_voice_name = resolve_vietnamese_tts_voice_name(
            inferred_voice_name,
            VIETNAMESE_TTS_VOICE_CHOICES,
            DEFAULT_VIETNAMESE_TTS_VOICE_NAME,
        )
        self._vietnamese_service_factory = vietnamese_service_factory

    @property
    def selected_vietnamese_model_id(self) -> str:
        return self._selected_vietnamese_model_id

    @property
    def selected_vietnamese_voice_name(self) -> str:
        return self._selected_vietnamese_voice_name

    def set_selected_vietnamese_model_id(self, model_id: str) -> None:
        self._selected_vietnamese_model_id = resolve_voice_model_id(
            model_id,
            VIETNAMESE_TTS_MODEL_CHOICES,
            DEFAULT_VIETNAMESE_TTS_MODEL_ID,
        )
        service = self._resolve_vietnamese_service(create_if_missing=False)
        if service is not None:
            self._apply_selected_vietnamese_voice_name(service)

    def set_selected_vietnamese_voice_name(self, voice_name: str | None) -> None:
        self._selected_vietnamese_voice_name = resolve_vietnamese_tts_voice_name(
            voice_name,
            VIETNAMESE_TTS_VOICE_CHOICES,
            DEFAULT_VIETNAMESE_TTS_VOICE_NAME,
        )
        service = self._resolve_vietnamese_service(create_if_missing=False)
        if service is not None:
            self._apply_selected_vietnamese_voice_name(service)

    def speak(self, text: str, *, cancel_event: Any = None, language: str | None = None) -> None:
        cleaned_text = _replace_urls_and_links_with_web_link_for_tts(text)
        if not cleaned_text:
            return

        service = self._select_service(language)
        service.speak(cleaned_text, cancel_event=cancel_event)

    def stop(self) -> None:
        for service in self._unique_services():
            try:
                service.stop()
            except Exception:
                pass

    def _select_service(self, language: str | None) -> TtsService:
        if _is_vietnamese_language(language):
            service = self._select_vietnamese_service()
            if service is not None:
                return service
        return self._default_service

    def _select_vietnamese_service(self) -> TtsService | None:
        service = self._resolve_vietnamese_service(create_if_missing=True)
        if service is not None:
            self._apply_selected_vietnamese_voice_name(service)
        return service

    def _resolve_vietnamese_service(self, *, create_if_missing: bool) -> TtsService | None:
        if not self._vietnamese_services and self._vietnamese_service_factory is not None:
            if create_if_missing:
                created_service = self._vietnamese_service_factory()
                self._vietnamese_services[DEFAULT_VIETNAMESE_TTS_MODEL_ID] = created_service

        service = self._vietnamese_services.get(self._selected_vietnamese_model_id)
        if service is None:
            service = self._vietnamese_services.get(DEFAULT_VIETNAMESE_TTS_MODEL_ID)
        if service is None and self._vietnamese_services:
            service = next(iter(self._vietnamese_services.values()))
        return service

    def _apply_selected_vietnamese_voice_name(self, service: TtsService) -> None:
        setter = getattr(service, "set_selected_vietnamese_voice_name", None)
        if not callable(setter):
            setter = getattr(service, "set_voice_name", None)
        if callable(setter):
            setter(self._selected_vietnamese_voice_name)

    @staticmethod
    def _infer_vietnamese_voice_name(vietnamese_service: TtsService | Mapping[str, TtsService] | None) -> str | None:
        if vietnamese_service is None:
            return None

        if isinstance(vietnamese_service, Mapping):
            services = list(vietnamese_service.values())
        else:
            services = [vietnamese_service]

        for service in services:
            candidate = getattr(service, "selected_vietnamese_voice_name", None)
            if isinstance(candidate, str) and candidate.strip():
                return candidate

            candidate = getattr(service, "voice_name", None)
            if isinstance(candidate, str) and candidate.strip():
                return candidate

        return None

    def _unique_services(self) -> list[TtsService]:
        services = [self._default_service, *self._vietnamese_services.values()]
        unique_services: list[TtsService] = []
        seen_ids: set[int] = set()
        for service in services:
            service_id = id(service)
            if service_id in seen_ids:
                continue
            seen_ids.add(service_id)
            unique_services.append(service)
        return unique_services

    @staticmethod
    def _normalize_vietnamese_services(
        vietnamese_service: TtsService | Mapping[str, TtsService] | None,
    ) -> dict[str, TtsService]:
        if vietnamese_service is None:
            return {}

        if isinstance(vietnamese_service, Mapping):
            return {str(model_id): service for model_id, service in vietnamese_service.items()}

        return {DEFAULT_VIETNAMESE_TTS_MODEL_ID: vietnamese_service}


def build_default_voice_services(
    voice_stt_model_id: str | None = None,
    voice_tts_model_id: str | None = None,
    voice_tts_voice_name: str | None = None,
) -> tuple[AudioRecorder, LanguageAwareSttService, LanguageAwareTtsService]:
    recorder = AudioRecorder(AudioRecorderConfig())
    tts_device = _preferred_tts_device()

    default_stt_service = WhisperSttService(
        WhisperSttConfig(
            model_size_or_path=os.getenv("LEARNING_AGENT_VOICE_STT_MODEL", "base"),
            device=os.getenv("LEARNING_AGENT_VOICE_STT_DEVICE", "cpu"),
            compute_type=os.getenv("LEARNING_AGENT_VOICE_STT_COMPUTE_TYPE", "int8"),
        )
    )

    selected_stt_model_id = resolve_voice_model_id(
        voice_stt_model_id or os.getenv("LEARNING_AGENT_VOICE_STT_VI_MODEL"),
        VIETNAMESE_STT_MODEL_CHOICES,
        DEFAULT_VIETNAMESE_STT_MODEL_ID,
    )
    selected_voice_name = resolve_vietnamese_tts_voice_name(
        voice_tts_voice_name or os.getenv("LEARNING_AGENT_VOICE_TTS_VI_VOICE_NAME"),
        VIETNAMESE_TTS_VOICE_CHOICES,
        DEFAULT_VIETNAMESE_TTS_VOICE_NAME,
    )

    zipformer_cache_dir = _voice_model_cache_dir(DEFAULT_VIETNAMESE_STT_MODEL_ID)
    pho_whisper_cache_dir = _voice_model_cache_dir(PHOWHISPER_MEDIUM_STT_MODEL_ID)

    vietnamese_stt_services: dict[str, SttService] = {
        DEFAULT_VIETNAMESE_STT_MODEL_ID: ZipformerTransducerSttService(
            ZipformerTransducerSttConfig(
                model_name_or_path=DEFAULT_VIETNAMESE_STT_MODEL_ID,
                provider=os.getenv(
                    "LEARNING_AGENT_VOICE_STT_VI_PROVIDER",
                    os.getenv("LEARNING_AGENT_VOICE_STT_VI_DEVICE", os.getenv("LEARNING_AGENT_VOICE_STT_DEVICE", "cpu")),
                ),
                num_threads=_coerce_int(os.getenv("LEARNING_AGENT_VOICE_STT_VI_NUM_THREADS"), 1),
                cache_dir=str(zipformer_cache_dir) if zipformer_cache_dir is not None else None,
                local_files_only=_coerce_bool(os.getenv("LEARNING_AGENT_VOICE_STT_VI_LOCAL_FILES_ONLY"), default=False),
                language=DEFAULT_VIETNAMESE_STT_LANGUAGE,
            )
        ),
        PHOWHISPER_MEDIUM_STT_MODEL_ID: PhoWhisperSttService(
            PhoWhisperSttConfig(
                model_name_or_path=PHOWHISPER_MEDIUM_STT_MODEL_ID,
                device=os.getenv("LEARNING_AGENT_VOICE_STT_VI_DEVICE", os.getenv("LEARNING_AGENT_VOICE_STT_DEVICE", "cpu")),
                beam_size=_coerce_int(os.getenv("LEARNING_AGENT_VOICE_STT_VI_BEAM_SIZE"), 5),
                max_new_tokens=_coerce_int(os.getenv("LEARNING_AGENT_VOICE_STT_VI_MAX_NEW_TOKENS"), 256),
                revision=os.getenv("LEARNING_AGENT_VOICE_STT_VI_REVISION"),
                cache_dir=str(pho_whisper_cache_dir) if pho_whisper_cache_dir is not None else None,
                local_files_only=_coerce_bool(os.getenv("LEARNING_AGENT_VOICE_STT_VI_LOCAL_FILES_ONLY"), default=False),
                language=DEFAULT_VIETNAMESE_STT_LANGUAGE,
            )
        ),
    }

    default_tts_service = _build_default_chatterbox_tts_service(tts_device)

    vietnamese_tts_services: dict[str, TtsService] = {
        DEFAULT_VIETNAMESE_TTS_MODEL_ID: _build_vietnamese_vieneu_tts_service(tts_device, selected_voice_name),
        VIENEU_TTS_03B_Q4_GGUF_MODEL_ID: _build_vietnamese_vieneu_tts_service(tts_device, selected_voice_name, backbone_repo=VIENEU_TTS_03B_Q4_GGUF_MODEL_ID),
        VIENEU_TTS_03B_MODEL_ID: _build_vietnamese_vieneu_tts_service(tts_device, selected_voice_name, backbone_repo=VIENEU_TTS_03B_MODEL_ID),
        F5_VIETNAMESE_TTS_MODEL_ID: _build_vietnamese_f5_tts_service(tts_device),
    }

    selected_tts_model_id = resolve_voice_model_id(
        voice_tts_model_id or os.getenv("LEARNING_AGENT_VOICE_TTS_VI_MODEL"),
        VIETNAMESE_TTS_MODEL_CHOICES,
        DEFAULT_VIETNAMESE_TTS_MODEL_ID,
    )
    return (
        recorder,
        LanguageAwareSttService(default_stt_service, vietnamese_stt_services, selected_vietnamese_model_id=selected_stt_model_id),
        LanguageAwareTtsService(
            default_tts_service,
            vietnamese_tts_services,
            selected_vietnamese_model_id=selected_tts_model_id,
            selected_vietnamese_voice_name=selected_voice_name,
        ),
    )


def _build_vietnamese_vieneu_tts_service(
    device: str | None = None,
    voice_name: str | None = None,
    *,
    backbone_repo: str | None = None,
) -> VieneuTtsService:
    resolved_backbone_repo = backbone_repo or os.getenv("LEARNING_AGENT_VOICE_TTS_VI_REPO_ID") or DEFAULT_VIETNAMESE_TTS_MODEL_ID
    resolved_device = (device or _preferred_tts_device()).strip() or "cpu"
    hf_token = os.getenv("LEARNING_AGENT_VOICE_TTS_HF_TOKEN") or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    resolved_voice_name = resolve_vietnamese_tts_voice_name(
        voice_name or os.getenv("LEARNING_AGENT_VOICE_TTS_VI_VOICE_NAME"),
        VIETNAMESE_TTS_VOICE_CHOICES,
        DEFAULT_VIETNAMESE_TTS_VOICE_NAME,
    )

    return VieneuTtsService(
        VieneuTtsConfig(
            backbone_repo=resolved_backbone_repo,
            backbone_device=resolved_device,
            codec_repo=os.getenv("LEARNING_AGENT_VOICE_TTS_VI_CODEC_REPO") or DEFAULT_VIETNAMESE_TTS_CODEC_REPO,
            codec_device=resolved_device,
            hf_token=hf_token,
            voice_name=resolved_voice_name,
        )
    )


def _build_vietnamese_f5_tts_service(device: str | None = None) -> F5TtsService:
    resolved_device = (device or _preferred_tts_device()).strip() or "cpu"
    hf_token = os.getenv("LEARNING_AGENT_VOICE_TTS_HF_TOKEN") or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    cache_dir = _voice_model_cache_dir(F5_VIETNAMESE_TTS_MODEL_ID)

    return F5TtsService(
        F5TtsConfig(
            model_repo_id=F5_VIETNAMESE_TTS_MODEL_ID,
            device=resolved_device,
            hf_cache_dir=cache_dir,
            hf_token=hf_token,
            reference_text=os.getenv("LEARNING_AGENT_VOICE_TTS_F5_REFERENCE_TEXT", "xin chào, tôi là trợ lý học tập."),
            reference_voice_repo_id=os.getenv("LEARNING_AGENT_VOICE_TTS_F5_REFERENCE_VOICE_REPO_ID") or DEFAULT_VIETNAMESE_TTS_MODEL_ID,
            reference_voice_name=os.getenv("LEARNING_AGENT_VOICE_TTS_F5_REFERENCE_VOICE_NAME"),
            revision=os.getenv("LEARNING_AGENT_VOICE_TTS_F5_REVISION"),
            local_files_only=_coerce_bool(os.getenv("LEARNING_AGENT_VOICE_TTS_F5_LOCAL_FILES_ONLY"), default=False),
        )
    )


def _build_default_chatterbox_tts_service(device: str | None = None) -> ChatterboxTtsService:
    resolved_device = (device or _preferred_tts_device()).strip() or "cpu"
    return ChatterboxTtsService(ChatterboxTtsConfig(device=resolved_device))


def _optional_path(value: str | None) -> Path | None:
    if not value:
        return None

    return Path(value)

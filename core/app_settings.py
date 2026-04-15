from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from core.config import ProviderConfig
from core.voice_catalog import (
    DEFAULT_VIETNAMESE_STT_MODEL_ID,
    DEFAULT_VIETNAMESE_TTS_MODEL_ID,
    DEFAULT_VIETNAMESE_TTS_VOICE_NAME,
    VIETNAMESE_STT_MODEL_CHOICES,
    VIETNAMESE_TTS_MODEL_CHOICES,
    VIETNAMESE_TTS_VOICE_CHOICES,
    resolve_voice_model_id,
    resolve_vietnamese_tts_voice_name,
)
from prompts.templates import DEFAULT_TARGET_LANGUAGE


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


@dataclass(frozen=True)
class AppSettings:
    preferred_language: str = DEFAULT_TARGET_LANGUAGE
    screen_ocr_enabled: bool = False
    selected_provider_config: ProviderConfig | None = None
    voice_stt_model_id: str = DEFAULT_VIETNAMESE_STT_MODEL_ID
    voice_tts_model_id: str = DEFAULT_VIETNAMESE_TTS_MODEL_ID
    voice_tts_voice_name: str = DEFAULT_VIETNAMESE_TTS_VOICE_NAME

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AppSettings":
        preferred_language = str(payload.get("preferred_language") or DEFAULT_TARGET_LANGUAGE).strip()
        selected_provider_config_payload = payload.get("selected_provider_config")
        selected_provider_config = None
        if isinstance(selected_provider_config_payload, Mapping):
            try:
                selected_provider_config = ProviderConfig.from_mapping(selected_provider_config_payload)
            except Exception:
                selected_provider_config = None
        return cls(
            preferred_language=preferred_language or DEFAULT_TARGET_LANGUAGE,
            screen_ocr_enabled=_coerce_bool(payload.get("screen_ocr_enabled"), default=False),
            selected_provider_config=selected_provider_config,
            voice_stt_model_id=resolve_voice_model_id(
                str(payload.get("voice_stt_model_id") or "").strip() or None,
                VIETNAMESE_STT_MODEL_CHOICES,
                DEFAULT_VIETNAMESE_STT_MODEL_ID,
            ),
            voice_tts_model_id=resolve_voice_model_id(
                str(payload.get("voice_tts_model_id") or "").strip() or None,
                VIETNAMESE_TTS_MODEL_CHOICES,
                DEFAULT_VIETNAMESE_TTS_MODEL_ID,
            ),
            voice_tts_voice_name=resolve_vietnamese_tts_voice_name(
                str(payload.get("voice_tts_voice_name") or "").strip() or None,
                VIETNAMESE_TTS_VOICE_CHOICES,
                DEFAULT_VIETNAMESE_TTS_VOICE_NAME,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "preferred_language": self.preferred_language,
            "screen_ocr_enabled": self.screen_ocr_enabled,
            "selected_provider_config": self.selected_provider_config.to_dict() if self.selected_provider_config is not None else None,
            "voice_stt_model_id": self.voice_stt_model_id,
            "voice_tts_model_id": self.voice_tts_model_id,
            "voice_tts_voice_name": self.voice_tts_voice_name,
        }


def load_app_settings(path: Path | str) -> AppSettings:
    file_path = Path(path)
    if not file_path.exists():
        return AppSettings()

    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Settings file must contain a JSON object.")
    return AppSettings.from_mapping(payload)


def save_app_settings(path: Path | str, settings: AppSettings) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(settings.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
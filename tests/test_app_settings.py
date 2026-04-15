from __future__ import annotations

from core.app_settings import AppSettings, load_app_settings, save_app_settings
from core.config import ProviderConfig
from core.voice_catalog import DEFAULT_VIETNAMESE_TTS_VOICE_NAME, PHOWHISPER_MEDIUM_STT_MODEL_ID, F5_VIETNAMESE_TTS_MODEL_ID


def test_app_settings_round_trip_preferred_language(tmp_path) -> None:
    settings_path = tmp_path / "settings.json"

    selected_provider_config = ProviderConfig.from_mapping(
        {
            "provider": "openai",
            "model": "gpt-5.4",
            "family": "gpt",
            "name": "gpt-5.4",
            "reasoning_effort": "medium",
            "web_search_enabled": True,
        }
    )

    save_app_settings(
        settings_path,
        AppSettings(
            preferred_language="Vietnamese",
            screen_ocr_enabled=True,
            selected_provider_config=selected_provider_config,
            voice_stt_model_id=PHOWHISPER_MEDIUM_STT_MODEL_ID,
            voice_tts_model_id=F5_VIETNAMESE_TTS_MODEL_ID,
            voice_tts_voice_name=DEFAULT_VIETNAMESE_TTS_VOICE_NAME,
        ),
    )

    restored = load_app_settings(settings_path)

    assert restored.preferred_language == "Vietnamese"
    assert restored.screen_ocr_enabled is True
    assert restored.selected_provider_config == selected_provider_config
    assert restored.voice_stt_model_id == PHOWHISPER_MEDIUM_STT_MODEL_ID
    assert restored.voice_tts_model_id == F5_VIETNAMESE_TTS_MODEL_ID
    assert restored.voice_tts_voice_name == DEFAULT_VIETNAMESE_TTS_VOICE_NAME


def test_app_settings_defaults_to_vietnamese_when_file_missing(tmp_path) -> None:
    settings_path = tmp_path / "missing.json"

    restored = load_app_settings(settings_path)

    assert restored.preferred_language == "Vietnamese"
    assert restored.screen_ocr_enabled is False
    assert restored.selected_provider_config is None
    assert restored.voice_stt_model_id == "hynt/Zipformer-30M-RNNT-6000h"
    assert restored.voice_tts_model_id == "pnnbao-ump/VieNeu-TTS"
    assert restored.voice_tts_voice_name == DEFAULT_VIETNAMESE_TTS_VOICE_NAME
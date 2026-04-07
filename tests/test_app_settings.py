from __future__ import annotations

from core.app_settings import AppSettings, load_app_settings, save_app_settings
from core.config import ProviderConfig


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
        ),
    )

    restored = load_app_settings(settings_path)

    assert restored.preferred_language == "Vietnamese"
    assert restored.screen_ocr_enabled is True
    assert restored.selected_provider_config == selected_provider_config


def test_app_settings_defaults_to_vietnamese_when_file_missing(tmp_path) -> None:
    settings_path = tmp_path / "missing.json"

    restored = load_app_settings(settings_path)

    assert restored.preferred_language == "Vietnamese"
    assert restored.screen_ocr_enabled is False
    assert restored.selected_provider_config is None
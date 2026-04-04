from __future__ import annotations

from core.app_settings import AppSettings, load_app_settings, save_app_settings


def test_app_settings_round_trip_preferred_language(tmp_path) -> None:
    settings_path = tmp_path / "settings.json"

    save_app_settings(settings_path, AppSettings(preferred_language="Vietnamese"))

    restored = load_app_settings(settings_path)

    assert restored.preferred_language == "Vietnamese"


def test_app_settings_defaults_to_vietnamese_when_file_missing(tmp_path) -> None:
    settings_path = tmp_path / "missing.json"

    restored = load_app_settings(settings_path)

    assert restored.preferred_language == "Vietnamese"
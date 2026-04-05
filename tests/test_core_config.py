from __future__ import annotations

import json

import pytest

from core.config import load_provider_config


def test_load_provider_config_reads_first_entry(tmp_path) -> None:
    config_path = tmp_path / "provider.json"
    config_path.write_text(
        json.dumps([
            {
                "provider": "openrouter",
                "model": "qwen/qwen3.6-plus:free",
                "base_url": "https://example.com/api/v1",
                "temperature": 0.75,
            }
        ]),
        encoding="utf-8",
    )

    config = load_provider_config(config_path)

    assert config.provider == "openrouter"
    assert config.model == "qwen/qwen3.6-plus:free"
    assert config.base_url == "https://example.com/api/v1"
    assert config.temperature == pytest.approx(0.75)


def test_load_provider_config_rejects_missing_model(tmp_path) -> None:
    config_path = tmp_path / "provider.json"
    config_path.write_text(json.dumps([{"provider": "openrouter"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="missing a model name"):
        load_provider_config(config_path)

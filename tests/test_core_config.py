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


def test_load_provider_config_reads_openai_tooling_options(tmp_path) -> None:
    config_path = tmp_path / "provider.json"
    config_path.write_text(
        json.dumps([
            {
                "provider": "openai",
                "model": "gpt-5.4",
                "reasoning_effort": "medium",
                "web_search_enabled": True,
                "web_search_external_web_access": False,
                "web_search_allowed_domains": ["openai.com", "developers.openai.com"],
                "max_output_tokens": 512,
            }
        ]),
        encoding="utf-8",
    )

    config = load_provider_config(config_path)

    assert config.provider == "openai"
    assert config.model == "gpt-5.4"
    assert config.reasoning_effort == "medium"
    assert config.web_search_enabled is True
    assert config.web_search_external_web_access is False
    assert config.web_search_allowed_domains == ("openai.com", "developers.openai.com")
    assert config.max_output_tokens == 512

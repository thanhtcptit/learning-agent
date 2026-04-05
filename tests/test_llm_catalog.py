from __future__ import annotations

import json

from core.config import discover_llm_catalog, load_provider_config


def test_discover_llm_catalog_groups_families_models_and_providers(tmp_path) -> None:
    qwen_dir = tmp_path / "qwen"
    qwen_dir.mkdir()
    (qwen_dir / "qwen3.6-plus.json").write_text(
        json.dumps([
            {"provider": "openrouter", "model": "qwen/qwen3.6-plus:free"},
        ]),
        encoding="utf-8",
    )

    gpt_dir = tmp_path / "gpt"
    gpt_dir.mkdir()
    (gpt_dir / "gpt-4.1.json").write_text(
        json.dumps([
            {"provider": "openai", "model": "gpt-4.1"},
            {"provider": "openrouter", "model": "openai/gpt-4.1"},
        ]),
        encoding="utf-8",
    )

    catalog = discover_llm_catalog(tmp_path)

    assert [entry.family for entry in catalog] == ["gpt", "qwen"]
    assert [entry.display_name for entry in catalog] == ["gpt-4.1", "qwen3.6-plus"]
    assert [entry.name for entry in catalog if entry.family == "gpt"] == ["gpt-4.1"]
    assert [provider.provider for provider in catalog[0].providers] == ["openai", "openrouter"]
    assert catalog[1].providers[0].provider == "openrouter"


def test_load_provider_config_infers_family_and_name(tmp_path) -> None:
    qwen_dir = tmp_path / "qwen"
    qwen_dir.mkdir()
    config_path = qwen_dir / "qwen3.6-plus.json"
    config_path.write_text(
        json.dumps({"provider": "openrouter", "model": "qwen/qwen3.6-plus:free"}),
        encoding="utf-8",
    )

    config = load_provider_config(config_path)

    assert config.family == "qwen"
    assert config.name == "qwen3.6-plus"

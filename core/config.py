from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


DEFAULT_LLM_API_ROOT = Path(__file__).resolve().parents[1] / "configs" / "llm_api"
DEFAULT_PROVIDER_CONFIG_PATH = (
    DEFAULT_LLM_API_ROOT / "qwen" / "qwen3.6-plus.json"
)
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


def _display_name_from_model(model: str) -> str:
    return model.split(":", 1)[0].strip() or model


@dataclass(frozen=True)
class LLMModelEntry:
    display_name: str
    family: str
    name: str
    providers: tuple["ProviderConfig", ...]


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    model: str
    display_name: str = ""
    family: str = ""
    name: str = ""
    base_url: str = DEFAULT_OPENROUTER_BASE_URL
    api_key_env: str = "OPENROUTER_API_KEY"
    site_url_env: str = "OPENROUTER_SITE_URL"
    app_name_env: str = "OPENROUTER_APP_NAME"
    temperature: float = 0.2


def _load_json(config_path: Path) -> Any:
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _default_provider_metadata(provider: str) -> dict[str, str]:
    provider_key = provider.strip().lower()
    if provider_key == "openai":
        return {
            "base_url": DEFAULT_OPENAI_BASE_URL,
            "api_key_env": "OPENAI_API_KEY",
            "site_url_env": "",
            "app_name_env": "",
        }

    return {
        "base_url": DEFAULT_OPENROUTER_BASE_URL,
        "api_key_env": "OPENROUTER_API_KEY",
        "site_url_env": "OPENROUTER_SITE_URL",
        "app_name_env": "OPENROUTER_APP_NAME",
    }


def _coerce_provider_entry(entry: dict[str, Any], *, family: str = "", name: str = "") -> ProviderConfig:
    provider = str(entry.get("provider", "openrouter")).strip() or "openrouter"
    model = str(entry.get("model", "")).strip()
    if not model:
        raise ValueError("Provider config is missing a model name.")

    defaults = _default_provider_metadata(provider)
    base_url = str(entry.get("base_url", defaults["base_url"])).strip() or defaults["base_url"]
    temperature = float(entry.get("temperature", 0.2))
    display_name = str(entry.get("display_name", _display_name_from_model(model))).strip() or _display_name_from_model(model)

    return ProviderConfig(
        provider=provider,
        model=model,
        display_name=display_name,
        family=str(entry.get("family", family)).strip() or family,
        name=str(entry.get("name", name)).strip() or name,
        base_url=base_url,
        api_key_env=str(entry.get("api_key_env", defaults["api_key_env"])).strip() or defaults["api_key_env"],
        site_url_env=str(entry.get("site_url_env", defaults["site_url_env"])).strip() or defaults["site_url_env"],
        app_name_env=str(entry.get("app_name_env", defaults["app_name_env"])).strip() or defaults["app_name_env"],
        temperature=temperature,
    )


def _iter_provider_entries(
    config_path: Path | str,
    *,
    family: str = "",
    name: str = "",
) -> Iterator[ProviderConfig]:
    path = Path(config_path)
    raw_config = _load_json(path)

    if isinstance(raw_config, list):
        if not raw_config:
            raise ValueError(f"Provider config file is empty: {path}")
        raw_entries = raw_config
    else:
        raw_entries = [raw_config]

    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            raise ValueError(f"Provider config must be an object or a non-empty list of objects: {path}")
        yield _coerce_provider_entry(raw_entry, family=family or path.parent.name, name=name or path.stem)


def load_provider_configs(config_path: Path | str = DEFAULT_PROVIDER_CONFIG_PATH) -> list[ProviderConfig]:
    return list(_iter_provider_entries(config_path))


def load_provider_config(config_path: Path | str = DEFAULT_PROVIDER_CONFIG_PATH) -> ProviderConfig:
    return load_provider_configs(config_path)[0]


def discover_llm_catalog(root_path: Path | str = DEFAULT_LLM_API_ROOT) -> list[LLMModelEntry]:
    root = Path(root_path)
    if not root.exists():
        return []

    catalog: list[LLMModelEntry] = []
    family_dirs = [path for path in root.iterdir() if path.is_dir()]
    for family_dir in sorted(family_dirs, key=lambda path: path.name.lower()):
        for config_path in sorted(family_dir.glob("*.json"), key=lambda path: path.stem.lower()):
            provider_configs = tuple(
                sorted(
                    load_provider_configs(config_path),
                    key=lambda config: (config.provider.lower(), config.name.lower(), config.model.lower()),
                )
            )
            catalog.append(
                LLMModelEntry(
                    display_name=config_path.stem,
                    family=family_dir.name,
                    name=config_path.stem,
                    providers=provider_configs,
                )
            )

    return sorted(catalog, key=lambda entry: entry.display_name.lower())


def build_provider(config: ProviderConfig):
    from llm.openrouter_provider import OpenRouterProvider

    return OpenRouterProvider.from_config(config)

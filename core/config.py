from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from core.runtime_paths import get_bundle_data_root


DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


def get_llm_api_root() -> Path:
    return get_bundle_data_root() / "configs" / "llm_api"


def get_default_provider_config_path() -> Path:
    return get_llm_api_root() / "gpt" / "gpt-4.1-mini.json"


DEFAULT_LLM_API_ROOT = get_llm_api_root()
DEFAULT_PROVIDER_CONFIG_PATH = get_default_provider_config_path()


def _display_name_from_model(model: str) -> str:
    return model.split(":", 1)[0].strip() or model


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
    reasoning_effort: str | None = None
    web_search_enabled: bool = False
    web_search_external_web_access: bool = True
    web_search_allowed_domains: tuple[str, ...] = ()
    max_output_tokens: int | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, family: str = "", name: str = "") -> "ProviderConfig":
        provider = str(payload.get("provider", "openrouter")).strip() or "openrouter"
        model = str(payload.get("model", "")).strip()
        if not model:
            raise ValueError("Provider config is missing a model name.")

        defaults = _default_provider_metadata(provider)
        base_url = str(payload.get("base_url", defaults["base_url"])).strip() or defaults["base_url"]
        temperature = float(payload.get("temperature", 0.2))
        display_name = str(payload.get("display_name", _display_name_from_model(model))).strip() or _display_name_from_model(model)
        reasoning_effort = str(payload.get("reasoning_effort") or "").strip() or None
        web_search_enabled = _coerce_bool(payload.get("web_search_enabled"), default=False)
        web_search_external_web_access = _coerce_bool(payload.get("web_search_external_web_access"), default=True)

        raw_allowed_domains = payload.get("web_search_allowed_domains", ())
        if isinstance(raw_allowed_domains, str):
            allowed_domains = tuple(
                domain.strip() for domain in raw_allowed_domains.split(",") if domain.strip()
            )
        elif isinstance(raw_allowed_domains, list):
            allowed_domains = tuple(
                str(domain).strip() for domain in raw_allowed_domains if str(domain).strip()
            )
        else:
            allowed_domains = ()

        raw_max_output_tokens = payload.get("max_output_tokens")
        max_output_tokens = int(raw_max_output_tokens) if raw_max_output_tokens not in (None, "") else None

        return cls(
            provider=provider,
            model=model,
            display_name=display_name,
            family=str(payload.get("family", family)).strip() or family,
            name=str(payload.get("name", name)).strip() or name,
            base_url=base_url,
            api_key_env=str(payload.get("api_key_env", defaults["api_key_env"])).strip() or defaults["api_key_env"],
            site_url_env=str(payload.get("site_url_env", defaults["site_url_env"])).strip() or defaults["site_url_env"],
            app_name_env=str(payload.get("app_name_env", defaults["app_name_env"])).strip() or defaults["app_name_env"],
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            web_search_enabled=web_search_enabled,
            web_search_external_web_access=web_search_external_web_access,
            web_search_allowed_domains=allowed_domains,
            max_output_tokens=max_output_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "display_name": self.display_name,
            "family": self.family,
            "name": self.name,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "site_url_env": self.site_url_env,
            "app_name_env": self.app_name_env,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
            "web_search_enabled": self.web_search_enabled,
            "web_search_external_web_access": self.web_search_external_web_access,
            "web_search_allowed_domains": list(self.web_search_allowed_domains),
            "max_output_tokens": self.max_output_tokens,
        }


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
    return ProviderConfig.from_mapping(entry, family=family, name=name)


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


def load_provider_configs(config_path: Path | str | None = None) -> list[ProviderConfig]:
    resolved_config_path = config_path if config_path is not None else get_default_provider_config_path()
    return list(_iter_provider_entries(resolved_config_path))


def load_provider_config(config_path: Path | str | None = None) -> ProviderConfig:
    return load_provider_configs(config_path)[0]


def discover_llm_catalog(root_path: Path | str | None = None) -> list[LLMModelEntry]:
    root = Path(root_path) if root_path is not None else get_llm_api_root()
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
    provider_key = config.provider.strip().lower()

    if provider_key == "openai":
        from llm.openai_provider import OpenAIProvider

        return OpenAIProvider.from_config(config)

    if provider_key == "openrouter":
        from llm.openrouter_provider import OpenRouterProvider

        return OpenRouterProvider.from_config(config)

    raise ValueError(f"Unsupported provider: {config.provider!r}")

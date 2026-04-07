from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

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

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AppSettings":
        preferred_language = str(payload.get("preferred_language") or DEFAULT_TARGET_LANGUAGE).strip()
        return cls(
            preferred_language=preferred_language or DEFAULT_TARGET_LANGUAGE,
            screen_ocr_enabled=_coerce_bool(payload.get("screen_ocr_enabled"), default=False),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "preferred_language": self.preferred_language,
            "screen_ocr_enabled": self.screen_ocr_enabled,
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
from __future__ import annotations

import json
import os
from typing import Iterator, Sequence

import httpx

from core.config import ProviderConfig
from llm.base import LLMMessage


class OpenRouterProvider:
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float = 0.2,
        site_url: str | None = None,
        app_name: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._site_url = site_url
        self._app_name = app_name
        self._timeout = timeout

    @classmethod
    def from_config(cls, config: ProviderConfig) -> "OpenRouterProvider":
        api_key = os.getenv(config.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"Missing API key. Set the {config.api_key_env} environment variable.")

        site_url = os.getenv(config.site_url_env, "").strip() or None
        app_name = os.getenv(config.app_name_env, "").strip() or None
        return cls(
            model=config.model,
            api_key=api_key,
            base_url=config.base_url,
            temperature=config.temperature,
            site_url=site_url,
            app_name=app_name,
        )

    def stream_chat(self, messages: Sequence[LLMMessage], *, temperature: float | None = None) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": [{"role": message.role, "content": message.content} for message in messages],
            "stream": True,
            "temperature": self._temperature if temperature is None else temperature,
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if self._site_url:
            headers["HTTP-Referer"] = self._site_url
        if self._app_name:
            headers["X-Title"] = self._app_name

        url = f"{self._base_url}/chat/completions"

        try:
            with httpx.Client(timeout=self._timeout) as client:
                with client.stream("POST", url, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line:
                            continue
                        if line.startswith("data:"):
                            data = line.removeprefix("data:").strip()
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                            except json.JSONDecodeError:
                                continue

                            choices = chunk.get("choices") or []
                            if not choices:
                                continue
                            delta = choices[0].get("delta") or {}
                            content = delta.get("content")
                            if content:
                                yield str(content)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

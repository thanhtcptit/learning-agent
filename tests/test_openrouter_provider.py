from __future__ import annotations

from typing import Any

import pytest

from core.config import ProviderConfig
from llm.base import LLMMessage
from llm.openrouter_provider import OpenRouterProvider


def test_openrouter_provider_from_config_reads_environment(monkeypatch) -> None:
    config = ProviderConfig(
        provider="openrouter",
        model="qwen/qwen3.6-plus:free",
        api_key_env="TEST_OPENROUTER_API_KEY",
        site_url_env="TEST_OPENROUTER_SITE_URL",
        app_name_env="TEST_OPENROUTER_APP_NAME",
    )
    monkeypatch.setenv("TEST_OPENROUTER_API_KEY", "secret-key")
    monkeypatch.setenv("TEST_OPENROUTER_SITE_URL", "https://example.com")
    monkeypatch.setenv("TEST_OPENROUTER_APP_NAME", "Learning Agent")

    provider = OpenRouterProvider.from_config(config)

    assert provider.model == "qwen/qwen3.6-plus:free"


def test_openrouter_provider_stream_chat_parses_sse_chunks(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeResponse:
        def __init__(self, lines: list[str]) -> None:
            self._lines = lines

        def raise_for_status(self) -> None:
            return None

        def iter_lines(self):
            yield from self._lines

    class FakeStreamContext:
        def __init__(self, response: FakeResponse) -> None:
            self._response = response

        def __enter__(self) -> FakeResponse:
            return self._response

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    class FakeClient:
        def __init__(self, timeout: float) -> None:
            captured["timeout"] = timeout

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def stream(self, method: str, url: str, *, json: dict[str, Any], headers: dict[str, str]):
            captured["method"] = method
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return FakeStreamContext(
                FakeResponse(
                    [
                        'data: {"choices":[{"delta":{"content":"Hel"}}]}',
                        'data: {"choices":[{"delta":{"content":"lo"}}]}',
                        "data: [DONE]",
                    ]
                )
            )

    monkeypatch.setattr("llm.openrouter_provider.httpx.Client", FakeClient)

    provider = OpenRouterProvider(
        model="qwen/qwen3.6-plus:free",
        api_key="secret-key",
        base_url="https://openrouter.ai/api/v1",
        temperature=0.4,
        site_url="https://example.com",
        app_name="Learning Agent",
    )

    chunks = list(provider.stream_chat([LLMMessage(role="user", content="hello")]))

    assert chunks == ["Hel", "lo"]
    assert captured["method"] == "POST"
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["json"]["stream"] is True
    assert captured["json"]["messages"] == [{"role": "user", "content": "hello"}]
    assert captured["headers"]["Authorization"] == "Bearer secret-key"
    assert captured["headers"]["HTTP-Referer"] == "https://example.com"
    assert captured["headers"]["X-Title"] == "Learning Agent"


def test_openrouter_provider_requires_api_key(monkeypatch) -> None:
    config = ProviderConfig(
        provider="openrouter",
        model="qwen/qwen3.6-plus:free",
        api_key_env="TEST_OPENROUTER_API_KEY_MISSING",
    )
    monkeypatch.delenv("TEST_OPENROUTER_API_KEY_MISSING", raising=False)

    with pytest.raises(RuntimeError, match="Missing API key"):
        OpenRouterProvider.from_config(config)

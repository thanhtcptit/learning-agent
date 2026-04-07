from __future__ import annotations

from typing import Any

import pytest

from core.config import ProviderConfig
from llm.base import LLMMessage
from llm.openai_provider import OpenAIProvider


def test_openai_provider_from_config_reads_environment(monkeypatch) -> None:
    config = ProviderConfig(
        provider="openai",
        model="gpt-5.4",
        api_key_env="TEST_OPENAI_API_KEY",
        reasoning_effort="medium",
        web_search_enabled=True,
    )
    monkeypatch.setenv("TEST_OPENAI_API_KEY", "secret-key")

    provider = OpenAIProvider.from_config(config)

    assert provider.model == "gpt-5.4"


def test_openai_provider_stream_chat_uses_responses_api_and_parses_text_chunks(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeResponse:
        def __init__(self, lines: list[str]) -> None:
            self._lines = lines

        def raise_for_status(self) -> None:
            return None

        def iter_lines(self):
            yield from self._lines

        def close(self) -> None:
            return None

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
                        'data: {"type":"response.output_text.delta","delta":"Hel"}',
                        '',
                        'event: response.output_text.delta',
                        'data: {"type":"response.output_text.delta","delta":"lo"}',
                        '',
                        'data: {"type":"response.completed","response":{"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello"}]}]}}',
                        '',
                    ]
                )
            )

    monkeypatch.setattr("llm.openai_provider.httpx.Client", FakeClient)

    provider = OpenAIProvider(
        model="gpt-5.4",
        api_key="secret-key",
        base_url="https://api.openai.com/v1",
        temperature=0.4,
        reasoning_effort="medium",
        web_search_enabled=True,
        web_search_external_web_access=False,
        web_search_allowed_domains=("openai.com",),
        max_output_tokens=512,
    )

    chunks = list(provider.stream_chat([LLMMessage(role="system", content="Be precise."), LLMMessage(role="user", content="hello")]))

    assert chunks == ["Hel", "lo"]
    assert captured["method"] == "POST"
    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert captured["json"]["stream"] is True
    assert captured["json"]["store"] is False
    assert captured["json"]["temperature"] == 0.4
    assert captured["json"]["reasoning"] == {"effort": "medium"}
    assert captured["json"]["max_output_tokens"] == 512
    assert captured["json"]["tools"] == [
        {
            "type": "web_search",
            "external_web_access": False,
            "filters": {"allowed_domains": ["openai.com"]},
        }
    ]
    assert captured["json"]["include"] == ["web_search_call.action.sources"]
    assert captured["json"]["input"] == [
        {"role": "system", "content": "Be precise."},
        {"role": "user", "content": "hello"},
    ]
    assert captured["headers"]["Authorization"] == "Bearer secret-key"


def test_openai_provider_requires_api_key(monkeypatch) -> None:
    config = ProviderConfig(
        provider="openai",
        model="gpt-5.4",
        api_key_env="TEST_OPENAI_API_KEY_MISSING",
    )
    monkeypatch.delenv("TEST_OPENAI_API_KEY_MISSING", raising=False)

    with pytest.raises(RuntimeError, match="Missing API key"):
        OpenAIProvider.from_config(config)
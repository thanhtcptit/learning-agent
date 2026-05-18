from __future__ import annotations

import threading
from typing import Any, Callable, Iterator

from core.config import DEFAULT_PROVIDER_CONFIG_PATH, ProviderConfig, discover_llm_catalog, load_provider_config
from llm.base import LLMMessage

_SENTINEL = object()


class FreeLLMManager:
    """Manages round-robin selection of free LLM providers with fallback.

    On each request it tries free providers starting from the last known-good
    one.  A provider is accepted as soon as it yields its first response chunk.
    If all free providers fail, the request falls back to gpt-4.1-mini.
    """

    def __init__(
        self,
        provider_factory: Callable[[ProviderConfig], Any],
        fallback_config_path: Any = None,
    ) -> None:
        self._provider_factory = provider_factory
        self._lock = threading.Lock()

        fallback_path = fallback_config_path if fallback_config_path is not None else DEFAULT_PROVIDER_CONFIG_PATH
        try:
            self._fallback_config: ProviderConfig = load_provider_config(fallback_path)
        except Exception:
            self._fallback_config = ProviderConfig(provider="openai", model="gpt-4.1-mini")

        self._providers: list[ProviderConfig] = self._collect_free_providers()
        self._working_index: int | None = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def providers(self) -> list[ProviderConfig]:
        return list(self._providers)

    def mark_working(self, config: ProviderConfig) -> None:
        """Record a provider that successfully returned a response."""
        with self._lock:
            try:
                self._working_index = self._providers.index(config)
            except ValueError:
                pass

    def stream_with_fallback(
        self,
        messages: list[LLMMessage],
        cancel_event: threading.Event,
        *,
        emergency_fallback: Any = None,
        on_provider_selected: Callable[[ProviderConfig], None] | None = None,
        on_provider_attempt: Callable[[ProviderConfig], None] | None = None,
    ) -> Iterator[str]:
        """Yield response chunks, trying free providers in round-robin order.

        Falls back to gpt-4.1-mini if all free providers fail before yielding
        their first chunk.  If gpt-4.1-mini also fails, falls back to
        ``emergency_fallback`` (the caller's currently-configured provider).
        ``on_provider_attempt`` is called immediately before each provider is
        tried so the caller can show which provider is currently in progress.
        ``on_provider_selected`` is called with the winning ProviderConfig the
        first time a provider successfully yields a chunk.
        """
        ordered = self._get_ordered_providers()

        for cfg in ordered:
            if cancel_event.is_set():
                return

            if on_provider_attempt is not None:
                on_provider_attempt(cfg)

            try:
                provider = self._provider_factory(cfg)
                gen = provider.stream_chat(messages, cancel_event=cancel_event)
                # Use two-arg next() to avoid PEP 479: StopIteration inside a
                # generator is converted to RuntimeError in Python 3.7+.
                first_chunk = next(gen, _SENTINEL)
                if first_chunk is _SENTINEL:
                    # Provider returned an empty stream — skip to next.
                    continue
            except Exception:
                continue

            # First chunk succeeded — commit to this provider.
            self.mark_working(cfg)
            if on_provider_selected is not None:
                on_provider_selected(cfg)
            yield first_chunk
            yield from gen
            return

        # All free providers failed — try gpt-4.1-mini as primary fallback.
        if on_provider_attempt is not None:
            on_provider_attempt(self._fallback_config)
        try:
            fallback_provider = self._provider_factory(self._fallback_config)
            gen = fallback_provider.stream_chat(messages, cancel_event=cancel_event)
            first_chunk = next(gen, _SENTINEL)
            if first_chunk is not _SENTINEL:
                if on_provider_selected is not None:
                    on_provider_selected(self._fallback_config)
                yield first_chunk
                yield from gen
                return
        except Exception:
            pass

        # gpt-4.1-mini also unavailable — use the caller's configured provider.
        if emergency_fallback is not None:
            yield from emergency_fallback.stream_chat(messages, cancel_event=cancel_event)
            return

        # No fallback available at all — raise so the caller surfaces an error.
        raise RuntimeError(
            "All free LLM providers failed and no fallback is available. "
            "Set OPENROUTER_API_KEY or ensure gpt-4.1-mini (OPENAI_API_KEY) is configured."
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_free_providers() -> list[ProviderConfig]:
        """Scan the LLM catalog and return all ProviderConfigs with is_free=True."""
        result: list[ProviderConfig] = []
        try:
            for entry in discover_llm_catalog():
                for cfg in entry.providers:
                    if cfg.is_free:
                        result.append(cfg)
        except Exception:
            pass
        return result

    def _get_ordered_providers(self) -> list[ProviderConfig]:
        """Return providers starting from _working_index (wrapping around)."""
        with self._lock:
            start = self._working_index

        if not self._providers:
            return []

        if start is None:
            return list(self._providers)

        idx = start % len(self._providers)
        return self._providers[idx:] + self._providers[:idx]

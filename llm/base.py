from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol, Sequence


@dataclass(frozen=True)
class LLMMessage:
    role: str
    content: str


class LLMProvider(Protocol):
    model: str

    def stream_chat(self, messages: Sequence[LLMMessage], *, temperature: float | None = None) -> Iterator[str]:
        """Yield incremental content chunks for a chat completion."""

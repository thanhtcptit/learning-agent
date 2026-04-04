from __future__ import annotations

from enum import Enum

from llm.base import LLMMessage


DEFAULT_TARGET_LANGUAGE = "Vietnamese"


class PromptMode(str, Enum):
    DEFINITION = "definition"
    EXPLAIN = "explain"
    SUMMARY = "summary"

    @property
    def label(self) -> str:
        if self is PromptMode.DEFINITION:
            return "Definition"
        if self is PromptMode.EXPLAIN:
            return "Explain"
        return "Summary"


def build_messages(text: str, mode: PromptMode, target_language: str = DEFAULT_TARGET_LANGUAGE) -> list[LLMMessage]:
    if mode is PromptMode.DEFINITION:
        system_prompt = (
            "You are a dictionary and language-learning assistant. Define the target word or term in "
            f"{target_language}. Keep the definition concise, and include 1-2 example usages in the word's original language."
        )
        user_prompt = f"Define the following word or term:\n\n{text}"
    elif mode is PromptMode.EXPLAIN:
        system_prompt = (
            "You are a learning assistant. Explain the meaning of the following text in "
            f"{target_language}. Use clear language, intuition, and examples."
        )
        user_prompt = f"Explain the following text:\n\n{text}"
    elif mode is PromptMode.SUMMARY:
        system_prompt = (
            "You are a summarization assistant. Summarize the following text in "
            f"{target_language}. Keep the summary concise and focused on the main ideas."
        )
        user_prompt = f"Summarize the following text:\n\n{text}"
    else:
        raise ValueError(f"Unsupported prompt mode: {mode!r}")

    return [
        LLMMessage(role="system", content=system_prompt),
        LLMMessage(role="user", content=user_prompt),
    ]

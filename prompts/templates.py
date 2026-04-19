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


def _build_screen_context_prefix(screen_context: str | None) -> str:
    if not screen_context:
        return ""

    cleaned_context = screen_context.strip()
    if not cleaned_context:
        return ""

    return (
        "Screen OCR context from the current screen. Use this only as surrounding context for the highlighted text, not as the main answer target.\n\n"
        f"{cleaned_context}\n\n"
    )


def build_messages(
    text: str,
    mode: PromptMode,
    target_language: str = DEFAULT_TARGET_LANGUAGE,
    *,
    screen_context: str | None = None,
) -> list[LLMMessage]:
    if mode is PromptMode.DEFINITION:
        system_prompt = (
            "You are a dictionary and language-learning assistant. Define the target word or term in "
            f"{target_language}. Keep the definition concise, and include 1-2 example usages in the word's original language."
        )
        user_prompt = f"{_build_screen_context_prefix(screen_context)}Define the following word or term:\n\n{text}"
    elif mode is PromptMode.EXPLAIN:
        system_prompt = (
            "You are a learning assistant. Explain the meaning of the following text in "
            f"{target_language}. Use clear language, intuition, and examples."
        )
        user_prompt = f"{_build_screen_context_prefix(screen_context)}Explain the following text:\n\n{text}"
    elif mode is PromptMode.SUMMARY:
        system_prompt = (
            "You are a summarization assistant. Summarize the following text in "
            f"{target_language}. Keep the summary concise and focused on the main ideas."
        )
        user_prompt = f"{_build_screen_context_prefix(screen_context)}Summarize the following text:\n\n{text}"
    else:
        raise ValueError(f"Unsupported prompt mode: {mode!r}")

    return [
        LLMMessage(role="system", content=system_prompt),
        LLMMessage(role="user", content=user_prompt),
    ]


def build_chat_messages(text: str, target_language: str = DEFAULT_TARGET_LANGUAGE) -> list[LLMMessage]:
    system_prompt = (
        "You are a helpful conversational assistant. Continue the conversation naturally and answer in "
        f"{target_language}. Be concise when possible and use the prior conversation as context."
    )

    return [
        LLMMessage(role="system", content=system_prompt),
        LLMMessage(role="user", content=text),
    ]


def build_voice_messages(text: str, target_language: str = DEFAULT_TARGET_LANGUAGE) -> list[LLMMessage]:
    system_prompt = (
        "You are a voice assistant. Answer in "
        f"{target_language} using plain natural speech that sounds good when spoken aloud. "
        "Keep the reply short, warm, and conversational. Use complete sentences and avoid markdown, list markers, code blocks, tables, and other formatting symbols. "
        "Do not use special characters or formatting. Return only the spoken answer."
    )

    return [
        LLMMessage(role="system", content=system_prompt),
        LLMMessage(role="user", content=text),
    ]

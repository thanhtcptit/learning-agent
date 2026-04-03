from __future__ import annotations

from enum import Enum

from llm.base import LLMMessage


DEFAULT_TARGET_LANGUAGE = "English"


class PromptMode(str, Enum):
    SIMPLE = "simple_explain"
    DETAILED = "detailed_explain"
    TRANSLATE = "translate"

    @property
    def label(self) -> str:
        if self is PromptMode.SIMPLE:
            return "Explain"
        if self is PromptMode.DETAILED:
            return "Explain deeply"
        return "Translate"


def build_messages(text: str, mode: PromptMode, target_language: str = DEFAULT_TARGET_LANGUAGE) -> list[LLMMessage]:
    if mode is PromptMode.DETAILED:
        system_prompt = (
            "You are a learning assistant. Explain the selected text clearly and in depth. "
            "Use examples, intuition, and concise structure."
        )
        user_prompt = f"Provide a detailed explanation of the following text:\n\n{text}"
    elif mode is PromptMode.TRANSLATE:
        system_prompt = (
            "You are a translation assistant. Preserve meaning, formatting, and tone. "
            "Return only the translated text unless clarification is required."
        )
        user_prompt = f"Translate the following text to {target_language}:\n\n{text}"
    else:
        system_prompt = (
            "You are a learning assistant. Explain the selected text in simple terms for a beginner. "
            "Use plain language and keep the answer focused."
        )
        user_prompt = f"Explain the following text in simple terms:\n\n{text}"

    return [
        LLMMessage(role="system", content=system_prompt),
        LLMMessage(role="user", content=user_prompt),
    ]

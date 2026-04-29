from __future__ import annotations

from enum import Enum

from llm.base import LLMMessage


DEFAULT_TARGET_LANGUAGE = "Vietnamese"

_BROWSER_ACTION_INSTRUCTIONS = (
    "\n\nYou can open a YouTube video for the user. When the user asks you to play media on YouTube, include a browser action block in your reply.\n"
    "Format:\n"
    "<<<BROWSER_ACTION>>>{\"action\": \"<action_type>\", ...}<<<END_ACTION>>>\n\n"
    "Available actions:\n"
    "- search_and_play: Search YouTube and auto-play the first video result. Fields: {\"action\": \"search_and_play\", \"query\": \"search terms\"}\n"
    "\n"
    "When the user asks to play, listen to, or watch something on YouTube, prefer search_and_play.\n\n"
    "Examples:\n"
    "- User: 'play piano music on youtube' → Include <<<BROWSER_ACTION>>>{\"action\": \"search_and_play\", \"query\": \"piano music\"}<<<END_ACTION>>> in your reply.\n"
    "Place the action block at the end of your reply. You may include a short conversational message before it."
)


class PromptMode(str, Enum):
    DEFINITION = "definition"
    EXPLAIN = "explain"
    SUMMARY = "summary"
    REWRITE = "rewrite"

    @property
    def label(self) -> str:
        if self is PromptMode.DEFINITION:
            return "Definition"
        if self is PromptMode.EXPLAIN:
            return "Explain"
        if self is PromptMode.SUMMARY:
            return "Summary"
        return "Rewrite"


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


def build_rewrite_messages(text: str) -> list[LLMMessage]:
    system_prompt = (
        "You are a professional editor. Fix any typos, spelling mistakes, and grammar errors in the text the user provides. "
        "If the text would benefit from it, also improve fluency, clarity, and structural coherence. "
        "Preserve the original language and tone. "
        "Output ONLY the rewritten text — no explanations, no commentary, no quotation marks around the result."
    )
    return [
        LLMMessage(role="system", content=system_prompt),
        LLMMessage(role="user", content=text),
    ]


def build_chat_messages(text: str, target_language: str = DEFAULT_TARGET_LANGUAGE) -> list[LLMMessage]:
    system_prompt = (
        "You are a helpful conversational assistant. Continue the conversation naturally and answer in "
        f"{target_language}. Be concise when possible and use the prior conversation as context."
        + _BROWSER_ACTION_INSTRUCTIONS
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
        + _BROWSER_ACTION_INSTRUCTIONS
    )

    return [
        LLMMessage(role="system", content=system_prompt),
        LLMMessage(role="user", content=text),
    ]

from __future__ import annotations

from prompts.templates import DEFAULT_TARGET_LANGUAGE, PromptMode, build_chat_messages, build_messages


def test_build_messages_definition_mode_uses_target_language() -> None:
    messages = build_messages("vector", PromptMode.DEFINITION)

    assert len(messages) == 2
    assert messages[0].role == "system"
    assert "dictionary" in messages[0].content.lower()
    assert "Vietnamese" in messages[0].content
    assert messages[1].role == "user"
    assert messages[1].content == "Define the following word or term:\n\nvector"


def test_build_messages_explain_mode_uses_target_language() -> None:
    messages = build_messages("vector", PromptMode.EXPLAIN, target_language="French")

    assert messages[0].role == "system"
    assert "French" in messages[0].content
    assert messages[1].content == "Explain the following text:\n\nvector"


def test_build_messages_summary_mode_uses_target_language() -> None:
    messages = build_messages("hola", PromptMode.SUMMARY, target_language="French")

    assert messages[0].role == "system"
    assert "summarization assistant" in messages[0].content.lower()
    assert messages[1].content == "Summarize the following text:\n\nhola"


def test_build_chat_messages_uses_target_language() -> None:
    messages = build_chat_messages("How does this work?", target_language="French")

    assert messages[0].role == "system"
    assert "helpful conversational assistant" in messages[0].content.lower()
    assert "French" in messages[0].content
    assert messages[1].role == "user"
    assert messages[1].content == "How does this work?"


def test_prompt_mode_labels_are_stable() -> None:
    assert PromptMode.DEFINITION.label == "Definition"
    assert PromptMode.EXPLAIN.label == "Explain"
    assert PromptMode.SUMMARY.label == "Summary"
    assert DEFAULT_TARGET_LANGUAGE == "Vietnamese"

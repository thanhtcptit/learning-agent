from __future__ import annotations

from prompts.templates import DEFAULT_TARGET_LANGUAGE, PromptMode, build_messages


def test_build_messages_simple_mode() -> None:
    messages = build_messages("vector", PromptMode.SIMPLE)

    assert len(messages) == 2
    assert messages[0].role == "system"
    assert "simple terms" in messages[0].content.lower()
    assert messages[1].role == "user"
    assert messages[1].content.endswith("vector")


def test_build_messages_translate_mode_uses_target_language() -> None:
    messages = build_messages("hola", PromptMode.TRANSLATE, target_language="French")

    assert messages[0].role == "system"
    assert "translation assistant" in messages[0].content.lower()
    assert messages[1].content == "Translate the following text to French:\n\nhola"


def test_prompt_mode_labels_are_stable() -> None:
    assert PromptMode.SIMPLE.label == "Explain"
    assert PromptMode.DETAILED.label == "Explain deeply"
    assert PromptMode.TRANSLATE.label == "Translate"
    assert DEFAULT_TARGET_LANGUAGE == "English"

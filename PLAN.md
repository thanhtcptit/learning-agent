# Implementation Plan

## Goal
Build a Windows-only learning assistant that captures selected text, sends it to an OpenRouter-backed LLM, and shows streaming answers in a PySide6 chat UI.

## Current Scope
- UI-first implementation
- One provider backend, but behind an abstraction
- In-memory session management first
- Windows hotkeys for quick capture from any app

## Milestones
1. Bootstrap packaging and startup wiring.
2. Implement hotkey capture, clipboard read, and prompt construction.
3. Add the OpenRouter streaming client and provider interface.
4. Build in-memory sessions and conversation history.
5. Add the PySide6 chat window, input box, and session controls.
6. Validate behavior and update the README to match the shipped code.

## Notes
- Use the existing `configs/llm_api/qwen/qwen3.6-plus.json` file as the default model config.
- Keep the first pass focused on a working vertical slice rather than exhaustive settings.

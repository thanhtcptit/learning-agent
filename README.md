# 🧠 AI Learning Assistant (Windows)

A lightweight AI-powered desktop assistant that helps users quickly understand unfamiliar terms or text while using their computer.

This tool runs in the background and allows users to:

* Select any text on screen
* Press a keyboard shortcut
* Instantly get an AI-generated explanation in a chat interface

The goal is to create a **system-wide learning copilot** that works across all applications (browser, IDE, PDF reader, etc.).

---

# ✨ Features
* Global keyboard shortcut listener
* Capture selected text via clipboard
* Send text to an LLM for explanation
* Return response via a chat window UI
* Session management (reuse last session, multiple sessions support, saved locally between runs)
* Display conversation history
* Settings popup (cascading LLM name/provider selection, language preference, modes, session switching, delete session, hotkey toggle)
* Streaming responses (real-time token output)

---

# 📌 Requirements

## System

* Windows OS

## Python

* Python 3.9+

## Environment Management

This project uses uv for dependency and environment management.

Install `uv`:

```bash
pip install uv
```

Create and activate environment:

```bash
uv venv
uv sync
```

---

# 🔧 How It Works

1. User highlights text anywhere on screen
2. Presses a shortcut (e.g., `Alt + E`)
3. Program:

   * Simulates `Ctrl + C`
   * Reads clipboard content
   * Builds a prompt based on mode
   * Sends request to LLM
4. Response is shown in the chat UI with streaming token updates

---

# 🧠 Prompt Modes

### Explain (Simple)

```
Explain the following text in simple terms (like for a beginner):

{text}
```

### Explain (Detailed)

```
Provide a detailed explanation of the following text.
Include examples and intuition:

{text}
```

### Translate

```
Translate the following text to {target_language}:

{text}
```

---

# 📁 Project Structure

```
learning-agent/
│
├── configs/
│   ├── llm-api/              # Store LLM API JSON configs by family and model
│
├── core/
│   ├── hotkey.py             # Global keyboard listener
│   ├── clipboard.py          # Clipboard handling (copy/paste)
│   ├── orchestrator.py       # Pipeline: hotkey → clipboard → prompt → LLM → UI
├── llm/
│   ├── base.py               # LLM interface
│   ├── openai_provider.py    # Compatibility alias
│   ├── openrouter_provider.py
│
├── prompts/
│   ├── templates.py          # Prompt templates
│
├── session/
│   ├── manager.py            # Session & conversation state

├── ui/
│   ├── main_window.py        # Main chat window (PySide6)
│   ├── chat_widget.py        # Chat message display
│   ├── input_box.py          # User input component
│   ├── settings_popup.py     # Compact settings popup
│
├── main.py                   # Entry point
│
└── README.md
```

---

# 🚀 Getting Started

### 1. Run the app

```bash
python main.py
```

---

### 2. Use it

* Highlight any text
* Press:

  * `Alt + E` → simple explanation
  * `Alt + D` → deep explanation
  * `Alt + T` → translation
  * `Alt + X` → exit the program
  * `Esc` → minimize the chat window to tray
* Check chat window output
* Press the settings icon to open the popup and disable hotkeys if needed

---

# 🧪 Testing

Run the unit tests with:

```bash
pytest
```

---

# 📄 License

MIT License (recommended for open-source projects)

---

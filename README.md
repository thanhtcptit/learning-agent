# 🧠 AI Learning Assistant (Windows)

A lightweight AI-powered desktop assistant that helps users quickly understand unfamiliar terms or text while using their computer.

This tool runs in the background and allows users to:

* Select any text on screen
* Press a keyboard shortcut
* Instantly get an AI-generated explanation in a chat interface
* Optionally scan the current screen with lightweight CPU OCR to add surrounding context for Definition mode

The goal is to create a **system-wide learning copilot** that works across all applications (browser, IDE, PDF reader, etc.).

---

# ✨ Features
* Global keyboard shortcut listener
* Capture selected text via clipboard
* Send text to an LLM for explanation
* Return response via a chat window UI
* Session management (reuse last session, multiple sessions support, saved locally between runs)
* Display conversation history
* Settings popup (single LLM selector with provider-qualified names, language preference, session switching, delete session, hotkey toggle, screen OCR toggle)
* Remembers the last selected LLM configuration across restarts
* OpenAI Responses API support for web search and reasoning-heavy models
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
2. Presses a shortcut (e.g., `Alt + D`, `Alt + E`, or `Alt + S`)
3. Program:

   * Simulates `Ctrl + C`
   * Reads clipboard content
    * If enabled for Definition mode, captures the current monitor, finds the ROI around the highlighted text, and OCRs that cropped area
  * If the selected provider is OpenAI, it can use web search and reasoning before answering
   * Builds a prompt based on mode
   * Sends request to LLM
4. Response is shown in the chat UI with streaming token updates

---

# 🧠 Prompt Modes

### Definition

```
Define the following word or term in {target_language}:

{text}
```

### Explain

```
Explain the meaning of the following text in {target_language}.
Use clear language, intuition, and examples:

{text}
```
### Summary

```
Summarize the following text in {target_language}:

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

### 2. Build a portable executable

The app can be bundled into a single Windows `.exe` with PyInstaller. Runtime paths now resolve bundled provider configs from the packaged data directory and look for an optional `.env` next to the executable.

```powershell
.\scripts\build_exe.ps1
```

The script calls `learning-agent.spec`, which bundles `configs\llm_api` into the executable and keeps the build definition checked into the repo.

The build produces `dist\learning-agent.exe`. Keep API keys external by placing a `.env` file next to the executable or by setting the required environment variables on the target machine. Session and settings data still live under `%APPDATA%\learning-agent`.

---

### 3. Use it

* Highlight any text
* Press:

  * `Alt + D` → definition mode
  * `Alt + E` → explain mode
  * `Alt + S` → summary mode
  * `Alt + H` → hide or show the chat window
  * `Alt + L` → toggle between the chosen language and English
  * `Alt + X` → exit the program
  * `Esc` → minimize the chat window to tray
  * `Stop` button → cancel the current request
* Check chat window output
* Press the settings icon to open the popup and disable hotkeys if needed
* If OCR is enabled, click the OCR context toggle on a user query to reveal the captured screen text above that query

The chosen language defaults to Vietnamese and is saved between runs. Screen OCR is off by default and can be enabled from the settings popup. When enabled, the app uses the selected text to localize a smaller OCR region before scanning it, but only for Definition mode.

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

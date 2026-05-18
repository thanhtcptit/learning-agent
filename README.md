# 🧠 AI Learning Assistant (Windows)

A lightweight AI-powered desktop assistant that helps users quickly understand unfamiliar terms or text while using their computer.

This tool runs in the background and allows users to:

* Select any text on screen
* Press a keyboard shortcut
* Instantly get an AI-generated explanation in a chat interface
* Optionally scan the current screen with lightweight CPU OCR to add surrounding context for Definition mode
* Record microphone input with a hotkey, transcribe it locally, and play back the reply with a local TTS voice model

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
* Rewrite mode — rewrites selected text in the target language
* Custom prompt mode — floating input dialog to apply an ad-hoc instruction to the selected text
* Wake word detection — say the wake word (default: "Mario") to trigger voice mode hands-free
* Free LLM mode — automatically round-robins across all `is_free` providers with gpt-4.1-mini as fallback
* Browser integration — the assistant can open YouTube videos on request

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
2. Presses the leader shortcut `Alt + Q` followed by a mode key (see hotkeys below)
3. Program:

   * Simulates `Ctrl + C`
   * Reads clipboard content
  * If enabled for Definition mode, captures the current monitor, finds the ROI around the highlighted text, and OCRs that cropped area
  * If the selected provider is OpenAI, it can use web search and reasoning before answering
  * Builds a prompt based on mode
  * Sends request to LLM
4. Response is shown in the chat UI with streaming token updates

Voice mode follows the same controller pipeline, but starts with microphone capture instead of clipboard selection:

1. User presses `Alt + Q → V`
2. Program records microphone audio until 2 seconds of silence
3. The audio is transcribed locally with the selected STT backend
4. The transcript is sent to the selected LLM
5. The response is synthesized locally with Chatterbox for non-Vietnamese speech or VieNeu for Vietnamese speech, then played back sentence by sentence so audio can start sooner

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

### Rewrite

```
Rewrite the following text in {target_language}. Keep the meaning but improve clarity and style:

{text}
```

### Custom Prompt

User types a free-form instruction in the floating `Alt + Q → P` dialog. The instruction is applied to the currently highlighted text.

---

# 📁 Project Structure

```
learning-agent/
│
├── configs/
│   └── llm_api/              # LLM API JSON configs by family and model
│       ├── gpt/              # GPT models (paid)
│       ├── qwen/             # Qwen models (free tier available)
│       └── ...               # Other families
│
├── core/
│   ├── hotkey.py             # Leader-key global hotkey listener (Alt+Q chord)
│   ├── orchestrator.py       # Pipeline: hotkey → clipboard/voice → prompt → LLM → UI
│   ├── free_llm_manager.py   # Round-robin free-provider routing with fallback
│   ├── wake_word_service.py  # Continuous wake word detection via faster-whisper
│   ├── browser_service.py    # YouTube search-and-play browser integration
│   └── ...                   # clipboard, OCR, STT, TTS, settings, …
│
├── llm/
│   ├── base.py               # LLM interface
│   ├── openai_provider.py    # OpenAI / OpenAI-compatible provider
│   └── openrouter_provider.py
│
├── prompts/
│   └── templates.py          # Prompt templates (Definition, Explain, Summary, Rewrite, Custom)
│
├── session/
│   └── manager.py            # Session & conversation state
│
├── ui/
│   ├── main_window.py        # Main chat window (PySide6)
│   ├── chat_widget.py        # Chat message display
│   ├── input_box.py          # User input component
│   ├── prompt_input_dialog.py # Floating custom-prompt dialog (Alt+Q → P)
│   └── settings_popup.py     # Compact settings popup
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
* Press `Alt + Q` (the leader key) then a mode key:

  | Second key | Action |
  |---|---|
  | `D` | Definition mode |
  | `E` | Explain mode |
  | `S` | Summary mode |
  | `R` | Rewrite mode |
  | `P` | Custom prompt (opens floating input dialog) |
  | `V` | Voice mode (mic capture) |
  | `W` | Toggle wake word listener |
  | `H` | Hide / show the chat window |
  | `L` | Toggle between the chosen language and English |
  | `X` | Exit the program |

  * `Esc` → minimize the chat window to tray
  * `Stop` button → cancel the current request
* Check chat window output
* Press the settings icon to open the popup and disable hotkeys if needed
* If OCR is enabled, click the OCR context toggle on a user query to reveal the captured screen text above that query

The chosen language defaults to Vietnamese and is saved between runs. Screen OCR is off by default and can be enabled from the settings popup. When enabled, the app uses the selected text to localize a smaller OCR region before scanning it, but only for Definition mode.

**Wake word mode** (`Alt + Q → W`) toggles the continuous wake word listener. When active, the app listens in the background via `faster-whisper` and triggers voice mode automatically when the configured wake word is detected. The default wake word is `Mario` and the default model is `small`. Override them with `LEARNING_AGENT_WAKE_WORD` and `LEARNING_AGENT_WAKE_WORD_DEVICE`.

**Free LLM mode** can be selected from the settings popup. In this mode the app automatically picks an available `is_free` provider from `configs/llm_api/` (round-robin, starting from the last known-good one) and falls back to `gpt-4.1-mini` if all free providers fail.

**Browser integration**: the assistant can open YouTube videos on request. When the selected LLM emits a `<<<BROWSER_ACTION>>>` block, the app parses it and opens the matching video in the default browser.

Voice mode uses `sounddevice` for microphone capture, `openai/whisper-large-v3-turbo` through the Transformers Whisper loader for the default English/non-Vietnamese STT path, `faster-whisper` for the fallback Whisper STT path, `sherpa-onnx` for the Vietnamese Zipformer STT path, and Chatterbox for the default non-Vietnamese TTS path. For Vietnamese, the app now defaults to `hynt/Zipformer-30M-RNNT-6000h` for STT and VieNeu TTS v1 for speech. The settings popup includes a VieNeu model dropdown with VieNeu TTS v1, VieNeu TTS 0.3B Q4 GGUF, and VieNeu TTS 0.3B. The voice preset dropdown refreshes when you switch engines and currently follows VieNeu's published preset catalog: Vinh, Binh, Tuyen, Doan, Ly, and Ngoc. The selected model and preset are saved between runs. The Vietnamese reply is fed to VieNeu sentence by sentence so playback can begin sooner. If you need to force a device override, prefer `cuda` or `cuda:0`; the TTS backends normalize `gpu` to CUDA when the installed PyTorch build exposes it. The current VieNeu models are loaded through the SDK's standard backend, which handles both transformer and GGUF backbones, and the codec path depends on `torchao` so the local VieNeu loader can initialize correctly. If the non-Vietnamese TTS model cannot be loaded, the app falls back to the local Windows speech engine so the response is still spoken. Chatterbox uses CUDA automatically when the installed PyTorch build exposes it.

If you want to override the English model, set `LEARNING_AGENT_VOICE_STT_MODEL`; to override the Vietnamese models, set `LEARNING_AGENT_VOICE_STT_VI_MODEL`, `LEARNING_AGENT_VOICE_TTS_VI_REPO_ID`, or `LEARNING_AGENT_VOICE_TTS_VI_VOICE_NAME` as needed. The app will use the Vietnamese engine when the selected language is Vietnamese and the default engines for other languages.

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

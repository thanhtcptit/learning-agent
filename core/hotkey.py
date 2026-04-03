from __future__ import annotations

import ctypes
import os
import threading
from ctypes import wintypes
from typing import Any, Mapping

from PySide6.QtCore import QObject, Signal

from prompts.templates import PromptMode


_WM_HOTKEY = 0x0312
_WM_QUIT = 0x0012
_MOD_ALT = 0x0001
_MOD_CONTROL = 0x0002
_MOD_SHIFT = 0x0004
_MOD_WIN = 0x0008

_MODIFIER_TOKENS = {
    "<alt>": _MOD_ALT,
    "alt": _MOD_ALT,
    "<ctrl>": _MOD_CONTROL,
    "<control>": _MOD_CONTROL,
    "ctrl": _MOD_CONTROL,
    "control": _MOD_CONTROL,
    "<shift>": _MOD_SHIFT,
    "shift": _MOD_SHIFT,
    "<win>": _MOD_WIN,
    "<cmd>": _MOD_WIN,
    "win": _MOD_WIN,
    "cmd": _MOD_WIN,
    "meta": _MOD_WIN,
}

_SPECIAL_KEYS = {
    "backspace": 0x08,
    "tab": 0x09,
    "enter": 0x0D,
    "return": 0x0D,
    "escape": 0x1B,
    "esc": 0x1B,
    "space": 0x20,
    "pageup": 0x21,
    "pagedown": 0x22,
    "end": 0x23,
    "home": 0x24,
    "left": 0x25,
    "up": 0x26,
    "right": 0x27,
    "down": 0x28,
    "insert": 0x2D,
    "delete": 0x2E,
    "del": 0x2E,
}


class _HotkeyBackend:
    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError


class _PynputHotkeyBackend(_HotkeyBackend):
    def __init__(self, hotkey_map: Mapping[str, PromptMode], trigger) -> None:
        self._hotkey_map = dict(hotkey_map)
        self._trigger = trigger
        self._listener: Any | None = None

    def start(self) -> None:
        if self._listener is not None:
            return

        from pynput.keyboard import GlobalHotKeys

        handlers = {combo: self._make_callback(mode) for combo, mode in self._hotkey_map.items()}
        self._listener = GlobalHotKeys(handlers)
        self._listener.start()

    def stop(self) -> None:
        listener = self._listener
        self._listener = None
        if listener is not None:
            listener.stop()

    def _make_callback(self, mode: PromptMode):
        def callback() -> None:
            self._trigger(mode)

        return callback


class _WindowsHotkeyBackend(_HotkeyBackend):
    def __init__(self, hotkey_map: Mapping[str, PromptMode], trigger) -> None:
        self._hotkey_map = dict(hotkey_map)
        self._trigger = trigger
        self._thread: threading.Thread | None = None
        self._thread_id: int | None = None
        self._stop_event = threading.Event()
        self._started_event = threading.Event()
        self._error: Exception | None = None
        self._id_to_mode: dict[int, PromptMode] = {}

    def start(self) -> None:
        if self._thread is not None:
            return

        self._stop_event.clear()
        self._started_event.clear()
        self._error = None
        self._id_to_mode.clear()

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        if not self._started_event.wait(timeout=2.0):
            self.stop()
            raise TimeoutError("Timed out starting the Windows hotkey listener")

        if self._error is not None:
            self.stop()
            raise self._error

    def stop(self) -> None:
        thread = self._thread
        thread_id = self._thread_id

        self._stop_event.set()
        if thread_id is not None:
            ctypes.windll.user32.PostThreadMessageW(thread_id, _WM_QUIT, 0, 0)

        if thread is not None:
            thread.join(timeout=2.0)

        self._thread = None
        self._thread_id = None

    def _run(self) -> None:
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        message = wintypes.MSG()
        self._thread_id = kernel32.GetCurrentThreadId()

        try:
            for hotkey_id, (combo, mode) in enumerate(self._hotkey_map.items(), start=1):
                modifiers, virtual_key = _parse_hotkey(combo)
                if not user32.RegisterHotKey(None, hotkey_id, modifiers, virtual_key):
                    raise OSError(f"Failed to register hotkey {combo!r}: {ctypes.WinError()}")
                self._id_to_mode[hotkey_id] = mode

            self._started_event.set()

            while not self._stop_event.is_set():
                result = user32.GetMessageW(ctypes.byref(message), None, 0, 0)
                if result == -1:
                    raise OSError(f"Hotkey listener message loop failed: {ctypes.WinError()}")
                if result == 0:
                    break
                if message.message == _WM_HOTKEY:
                    mode = self._id_to_mode.get(int(message.wParam))
                    if mode is not None:
                        self._trigger(mode)
        except Exception as exc:  # noqa: BLE001 - start() surfaces startup failures to the UI
            self._error = exc
            self._started_event.set()
        finally:
            for hotkey_id in list(self._id_to_mode):
                user32.UnregisterHotKey(None, hotkey_id)
            self._id_to_mode.clear()
            self._thread_id = None


def _create_hotkey_backend(hotkey_map: Mapping[str, PromptMode], trigger) -> _HotkeyBackend:
    if os.name == "nt":
        return _WindowsHotkeyBackend(hotkey_map, trigger)
    return _PynputHotkeyBackend(hotkey_map, trigger)


def _parse_hotkey(combo: str) -> tuple[int, int]:
    modifiers = 0
    virtual_key: int | None = None

    for raw_token in combo.split("+"):
        token = raw_token.strip().lower()
        if not token:
            continue

        modifier = _MODIFIER_TOKENS.get(token)
        if modifier is not None:
            modifiers |= modifier
            continue

        if token.startswith("<") and token.endswith(">"):
            token = token[1:-1]

        if virtual_key is not None:
            raise ValueError(f"Hotkey {combo!r} may only contain one non-modifier key")
        virtual_key = _virtual_key_code(token)

    if virtual_key is None:
        raise ValueError(f"Hotkey {combo!r} does not include a key")

    return modifiers, virtual_key


def _virtual_key_code(token: str) -> int:
    if len(token) == 1:
        return ord(token.upper())

    special_key = _SPECIAL_KEYS.get(token)
    if special_key is not None:
        return special_key

    if token.startswith("f") and token[1:].isdigit():
        function_key = int(token[1:])
        if 1 <= function_key <= 24:
            return 0x6F + function_key

    raise ValueError(f"Unsupported hotkey key: {token!r}")


class GlobalHotkeyListener(QObject):
    hotkey_triggered = Signal(object)
    status_changed = Signal(str)

    def __init__(self, hotkey_map: Mapping[str, PromptMode] | None = None) -> None:
        super().__init__()
        self._hotkey_map = dict(
            hotkey_map
            or {
                "<alt>+e": PromptMode.SIMPLE,
                "<alt>+d": PromptMode.DETAILED,
                "<alt>+t": PromptMode.TRANSLATE,
            }
        )
        self._backend: _HotkeyBackend | None = None

    @property
    def is_running(self) -> bool:
        return self._backend is not None

    def start(self) -> None:
        if self._backend is not None:
            return

        backend = _create_hotkey_backend(self._hotkey_map, self.hotkey_triggered.emit)
        backend.start()
        self._backend = backend
        self.status_changed.emit("Hotkeys active")

    def stop(self) -> None:
        backend = self._backend
        self._backend = None
        if backend is not None:
            backend.stop()
            self.status_changed.emit("Hotkeys stopped")

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from PySide6.QtCore import QObject, Signal

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover - optional dependency during bootstrap
    sd = None

try:
    from faster_whisper import WhisperModel
except ImportError:  # pragma: no cover - optional dependency during bootstrap
    WhisperModel = None


DEFAULT_WAKE_WORD = "Mario"
DEFAULT_WAKE_WORD_MODEL = "small"

_ENV_WAKE_WORD = "LEARNING_AGENT_WAKE_WORD"
_ENV_WAKE_WORD_DEVICE = "LEARNING_AGENT_WAKE_WORD_DEVICE"


@dataclass(frozen=True)
class WakeWordConfig:
    wake_word: str = DEFAULT_WAKE_WORD
    model_size_or_path: str = DEFAULT_WAKE_WORD_MODEL
    device: str = "cpu"
    compute_type: str = "int8"
    listen_duration_seconds: float = 3.0
    sample_rate: int = 16000
    channels: int = 1
    silence_threshold: float = 0.015
    audio_device: int | str | None = None


class WakeWordService(QObject):
    wake_word_detected = Signal()
    listening_state_changed = Signal(bool)

    def __init__(
        self,
        config: WakeWordConfig | None = None,
        *,
        stream_factory: Callable[..., Any] | None = None,
        model_factory: Callable[..., Any] | None = None,
        on_wake_word: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self._config = config or WakeWordConfig(
            wake_word=os.environ.get(_ENV_WAKE_WORD, DEFAULT_WAKE_WORD),
            device=os.environ.get(_ENV_WAKE_WORD_DEVICE, "cpu"),
        )
        self._stream_factory = stream_factory or self._default_stream_factory
        self._model_factory = model_factory
        self._on_wake_word = on_wake_word
        self._model: Any | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._lock = threading.Lock()

    @property
    def is_listening(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    @property
    def is_paused(self) -> bool:
        return self._pause_event.is_set()

    @property
    def wake_word(self) -> str:
        return self._config.wake_word

    def start_listening(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._pause_event.clear()
            self._thread = threading.Thread(target=self._listen_loop, daemon=True)
            self._thread.start()
        self.listening_state_changed.emit(True)

    def stop_listening(self) -> None:
        self._stop_event.set()
        with self._lock:
            thread = self._thread
            self._thread = None
        if thread is not None:
            thread.join(timeout=5.0)
        self.listening_state_changed.emit(False)

    def pause(self) -> None:
        self._pause_event.set()

    def resume(self) -> None:
        if self._stop_event.is_set():
            return
        self._pause_event.clear()

    def _listen_loop(self) -> None:
        blocksize = max(1, int(self._config.sample_rate * self._config.listen_duration_seconds))

        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                self._stop_event.wait(0.2)
                continue

            try:
                samples = self._record_chunk(blocksize)
            except Exception:
                if self._stop_event.is_set():
                    break
                self._stop_event.wait(0.5)
                continue

            if samples is None or samples.size == 0:
                continue

            if not self._has_speech(samples):
                continue

            try:
                text = self._transcribe_chunk(samples)
            except Exception:
                continue

            if self._contains_wake_word(text):
                self._pause_event.set()
                self.wake_word_detected.emit()
                if self._on_wake_word is not None:
                    self._on_wake_word()

    def _record_chunk(self, blocksize: int) -> np.ndarray | None:
        if sd is None:
            raise RuntimeError("sounddevice is required for wake-word detection.")

        with self._stream_factory(
            samplerate=self._config.sample_rate,
            blocksize=blocksize,
            device=self._config.audio_device,
            channels=self._config.channels,
            dtype="float32",
        ) as stream:
            frames, _overflowed = stream.read(blocksize)

        return np.asarray(frames, dtype=np.float32).reshape(-1)

    def _has_speech(self, samples: np.ndarray) -> bool:
        amplitude = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
        return amplitude >= self._config.silence_threshold

    def _transcribe_chunk(self, samples: np.ndarray) -> str:
        model = self._load_model()
        segments, _info = model.transcribe(
            samples,
            beam_size=1,
            language="en",
            vad_filter=False,
            condition_on_previous_text=False,
        )
        parts: list[str] = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                parts.append(text)
        return " ".join(parts).strip()

    def _contains_wake_word(self, text: str) -> bool:
        return self._config.wake_word.lower() in text.lower()

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        if self._model_factory is not None:
            self._model = self._model_factory()
            return self._model

        if WhisperModel is None:
            raise RuntimeError("faster-whisper is required for wake-word detection.")

        self._model = WhisperModel(
            self._config.model_size_or_path,
            device=self._config.device,
            compute_type=self._config.compute_type,
        )
        return self._model

    @staticmethod
    def _default_stream_factory(**kwargs: Any) -> Any:
        if sd is None:
            raise RuntimeError("sounddevice is required for wake-word detection.")
        return sd.InputStream(**kwargs)

from __future__ import annotations

import builtins
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from core.wake_word_service import (
    DEFAULT_WAKE_WORD,
    DEFAULT_WAKE_WORD_MODEL,
    WakeWordConfig,
    WakeWordService,
    _ENV_WAKE_WORD,
    _ENV_WAKE_WORD_DEVICE,
)


class FakeInputStream:
    """Simulates sounddevice.InputStream as a context manager."""

    def __init__(self, samples: np.ndarray) -> None:
        self._samples = samples

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, blocksize: int):
        return self._samples.reshape(-1, 1)[:blocksize], False


def _make_stream_factory(samples: np.ndarray):
    """Returns a stream factory that yields *samples* once."""

    @contextmanager
    def factory(**_kwargs):
        yield FakeInputStream(samples)

    return factory


class FakeSegment:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeWhisperModel:
    """Mimics faster_whisper.WhisperModel enough for wake-word tests."""

    def __init__(self, transcription: str = "") -> None:
        self.transcription = transcription
        self.transcribe_calls: list[tuple[np.ndarray, dict]] = []

    def transcribe(self, audio, **kwargs):
        self.transcribe_calls.append((audio, kwargs))
        segments = [FakeSegment(self.transcription)] if self.transcription else []
        info = SimpleNamespace(language="en", language_probability=0.99)
        return iter(segments), info


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = WakeWordConfig()
    assert cfg.wake_word == DEFAULT_WAKE_WORD
    assert cfg.model_size_or_path == DEFAULT_WAKE_WORD_MODEL
    assert cfg.device == "cpu"
    assert cfg.compute_type == "int8"
    assert cfg.listen_duration_seconds == 3.0
    assert cfg.sample_rate == 16000
    assert cfg.silence_threshold == 0.015


def test_config_from_env(monkeypatch):
    monkeypatch.setenv(_ENV_WAKE_WORD, "Jarvis")
    monkeypatch.setenv(_ENV_WAKE_WORD_DEVICE, "cuda")
    svc = WakeWordService()
    assert svc.wake_word == "Jarvis"
    assert svc._config.device == "cuda"


# ---------------------------------------------------------------------------
# Wake-word detection logic
# ---------------------------------------------------------------------------

def test_contains_wake_word():
    svc = WakeWordService(config=WakeWordConfig(wake_word="Mario"))
    assert svc._contains_wake_word("Hey Mario, what's up?")
    assert svc._contains_wake_word("MARIO")
    assert svc._contains_wake_word("mario")
    assert not svc._contains_wake_word("Hey Luigi")
    assert not svc._contains_wake_word("")


def test_transcribe_chunk_prints_stt_output(monkeypatch):
    loud_samples = np.full(1600, 0.1, dtype=np.float32)
    printed: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_print(*args, **kwargs) -> None:
        printed.append((args, kwargs))

    monkeypatch.setattr(builtins, "print", fake_print)

    svc = WakeWordService(
        config=WakeWordConfig(wake_word="Mario"),
        model_factory=lambda: FakeWhisperModel(transcription="Hey Mario"),
    )

    transcription = svc._transcribe_chunk(loud_samples)

    assert transcription == "Hey Mario"
    assert printed == [(("Wake-word STT output: 'Hey Mario'",), {"flush": True})]


def test_has_speech_above_threshold():
    svc = WakeWordService(config=WakeWordConfig(silence_threshold=0.01))
    loud = np.full(1600, 0.1, dtype=np.float32)
    silent = np.full(1600, 0.001, dtype=np.float32)
    assert svc._has_speech(loud)
    assert not svc._has_speech(silent)
    assert not svc._has_speech(np.array([], dtype=np.float32))


# ---------------------------------------------------------------------------
# Full listen loop — wake word detected
# ---------------------------------------------------------------------------

def test_wake_word_detected_emits_signal_and_pauses():
    loud_samples = np.full(48000, 0.1, dtype=np.float32)
    model = FakeWhisperModel(transcription="Hey Mario how are you")
    detected = threading.Event()

    call_count = 0

    @contextmanager
    def counting_stream_factory(**_kwargs):
        nonlocal call_count
        call_count += 1
        yield FakeInputStream(loud_samples)

    svc = WakeWordService(
        config=WakeWordConfig(wake_word="Mario", listen_duration_seconds=0.1),
        stream_factory=counting_stream_factory,
        model_factory=lambda: model,
        on_wake_word=detected.set,
    )

    svc.start_listening()
    assert detected.wait(timeout=5.0), "wake_word_detected was not emitted"
    assert svc.is_paused
    assert svc.is_listening  # thread still alive, just paused

    svc.stop_listening()
    assert not svc.is_listening
    assert len(model.transcribe_calls) >= 1


# ---------------------------------------------------------------------------
# Silence skips transcription
# ---------------------------------------------------------------------------

def test_silence_skips_transcription():
    silent_samples = np.full(48000, 0.001, dtype=np.float32)
    model = FakeWhisperModel(transcription="Mario")

    iterations = 0

    @contextmanager
    def counting_factory(**_kwargs):
        nonlocal iterations
        iterations += 1
        yield FakeInputStream(silent_samples)

    svc = WakeWordService(
        config=WakeWordConfig(wake_word="Mario", listen_duration_seconds=0.1, silence_threshold=0.01),
        stream_factory=counting_factory,
        model_factory=lambda: model,
    )

    svc.start_listening()
    # Let the loop run a few iterations
    import time
    time.sleep(0.5)
    svc.stop_listening()

    assert iterations >= 1
    assert len(model.transcribe_calls) == 0, "transcribe should not be called for silent audio"


# ---------------------------------------------------------------------------
# Pause / Resume lifecycle
# ---------------------------------------------------------------------------

def test_pause_resume_lifecycle():
    loud_samples = np.full(48000, 0.1, dtype=np.float32)
    model = FakeWhisperModel(transcription="Mario")
    detected_count = 0

    @contextmanager
    def stream_factory(**_kwargs):
        yield FakeInputStream(loud_samples)

    def on_detected():
        nonlocal detected_count
        detected_count += 1

    svc = WakeWordService(
        config=WakeWordConfig(wake_word="Mario", listen_duration_seconds=0.1),
        stream_factory=stream_factory,
        model_factory=lambda: model,
        on_wake_word=on_detected,
    )

    svc.start_listening()
    import time
    time.sleep(0.5)

    # Should have detected and auto-paused
    assert svc.is_paused
    first_count = detected_count
    assert first_count >= 1

    # Resume — should detect again
    model.transcribe_calls.clear()
    svc.resume()
    time.sleep(0.5)
    assert detected_count > first_count

    svc.stop_listening()


# ---------------------------------------------------------------------------
# No speech in transcription — loop continues
# ---------------------------------------------------------------------------

def test_no_wake_word_in_transcription_continues_loop():
    loud_samples = np.full(48000, 0.1, dtype=np.float32)
    model = FakeWhisperModel(transcription="Hello world")
    detected = threading.Event()

    @contextmanager
    def stream_factory(**_kwargs):
        yield FakeInputStream(loud_samples)

    svc = WakeWordService(
        config=WakeWordConfig(wake_word="Mario", listen_duration_seconds=0.1),
        stream_factory=stream_factory,
        model_factory=lambda: model,
        on_wake_word=lambda: detected.set(),
    )

    svc.start_listening()
    import time
    time.sleep(0.5)

    assert not detected.is_set(), "wake_word_detected should NOT be emitted"
    assert not svc.is_paused
    assert len(model.transcribe_calls) >= 2  # multiple iterations ran

    svc.stop_listening()


# ---------------------------------------------------------------------------
# start_listening is idempotent
# ---------------------------------------------------------------------------

def test_start_listening_idempotent():
    svc = WakeWordService(
        config=WakeWordConfig(listen_duration_seconds=0.1),
        stream_factory=_make_stream_factory(np.full(1600, 0.001, dtype=np.float32)),
        model_factory=lambda: FakeWhisperModel(),
    )
    svc.start_listening()
    svc.start_listening()  # should not start a second thread
    assert svc.is_listening
    svc.stop_listening()


def test_stop_listening_when_not_started():
    svc = WakeWordService()
    svc.stop_listening()  # should not raise
    assert not svc.is_listening


# ---------------------------------------------------------------------------
# model_factory injects the model
# ---------------------------------------------------------------------------

def test_model_factory_is_used():
    model = FakeWhisperModel(transcription="Mario")
    loud = np.full(48000, 0.1, dtype=np.float32)
    detected = threading.Event()

    svc = WakeWordService(
        config=WakeWordConfig(wake_word="Mario", listen_duration_seconds=0.1),
        stream_factory=_make_stream_factory(loud),
        model_factory=lambda: model,
        on_wake_word=detected.set,
    )

    svc.start_listening()
    assert detected.wait(timeout=5.0)
    svc.stop_listening()

    assert len(model.transcribe_calls) >= 1
    kwargs = model.transcribe_calls[0][1]
    assert kwargs["beam_size"] == 1
    assert kwargs["language"] == "en"

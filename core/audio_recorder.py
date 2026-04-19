from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Event
from typing import Any, Callable

import numpy as np

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover - optional dependency during bootstrap
    sd = None


class VoiceCaptureCancelled(RuntimeError):
    pass


class VoiceCaptureError(RuntimeError):
    pass


@dataclass(frozen=True)
class RecordedAudio:
    samples: np.ndarray
    sample_rate: int

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate <= 0 or self.samples.size == 0:
            return 0.0
        return float(self.samples.shape[0]) / float(self.sample_rate)

    @property
    def is_empty(self) -> bool:
        return self.samples.size == 0


@dataclass(frozen=True)
class AudioRecorderConfig:
    sample_rate: int = 16000
    channels: int = 1
    block_duration_seconds: float = 0.1
    silence_duration_seconds: float = 2.0
    silence_threshold: float = 0.015
    initial_timeout_seconds: float = 8.0
    max_record_seconds: float = 60.0
    device: int | str | None = None


class AudioRecorder:
    def __init__(self, config: AudioRecorderConfig | None = None, *, stream_factory: Callable[..., Any] | None = None) -> None:
        self._config = config or AudioRecorderConfig()
        self._stream_factory = stream_factory or self._default_stream_factory

    def record_until_silence(
        self,
        cancel_event: Event | None = None,
        *,
        initial_timeout_seconds: float | None = None,
    ) -> RecordedAudio:
        if sd is None:
            raise VoiceCaptureError("sounddevice is required for voice recording.")

        if cancel_event is not None and cancel_event.is_set():
            raise VoiceCaptureCancelled("Voice capture was cancelled.")

        blocksize = max(1, int(self._config.sample_rate * self._config.block_duration_seconds))
        silent_block_limit = max(1, int(round(self._config.silence_duration_seconds / self._config.block_duration_seconds)))
        initial_timeout = self._config.initial_timeout_seconds if initial_timeout_seconds is None else max(0.0, float(initial_timeout_seconds))
        recorded_blocks: list[np.ndarray] = []
        silence_blocks = 0
        speech_started = False
        start_time = time.monotonic()

        with self._stream_factory(
            samplerate=self._config.sample_rate,
            blocksize=blocksize,
            device=self._config.device,
            channels=self._config.channels,
            dtype="float32",
        ) as stream:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise VoiceCaptureCancelled("Voice capture was cancelled.")

                frames, _overflowed = stream.read(blocksize)
                samples = np.asarray(frames, dtype=np.float32).reshape(-1)
                if samples.size == 0:
                    continue

                amplitude = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
                elapsed = time.monotonic() - start_time

                if not speech_started:
                    if amplitude >= self._config.silence_threshold:
                        speech_started = True
                        recorded_blocks.append(samples.copy())
                        silence_blocks = 0
                    elif elapsed >= initial_timeout:
                        raise VoiceCaptureError("No speech detected.")
                    continue

                recorded_blocks.append(samples.copy())
                if amplitude < self._config.silence_threshold:
                    silence_blocks += 1
                else:
                    silence_blocks = 0

                if silence_blocks >= silent_block_limit:
                    if silence_blocks:
                        recorded_blocks = recorded_blocks[:-silence_blocks]
                    break

                if elapsed >= self._config.max_record_seconds:
                    break

        if not speech_started or not recorded_blocks:
            raise VoiceCaptureError("No speech detected.")

        samples = np.concatenate(recorded_blocks).astype(np.float32, copy=False)
        return RecordedAudio(samples=samples, sample_rate=self._config.sample_rate)

    def cancel_current_request(self) -> None:
        if sd is None:
            return

        try:
            sd.stop()
        except Exception:
            pass

    @staticmethod
    def _default_stream_factory(**kwargs: Any) -> Any:
        if sd is None:
            raise VoiceCaptureError("sounddevice is required for voice recording.")

        return sd.InputStream(**kwargs)
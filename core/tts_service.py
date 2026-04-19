from __future__ import annotations

import os
import re
import tempfile
import time
import wave
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event
from typing import Any, Protocol
import unicodedata

import numpy as np

try:
    import winsound
except ImportError:  # pragma: no cover - Windows-only dependency
    winsound = None

try:
    import pyttsx3
except ImportError:  # pragma: no cover - optional fallback during bootstrap
    pyttsx3 = None

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency during bootstrap
    torch = None

try:
    from huggingface_hub import hf_hub_download
except ImportError:  # pragma: no cover - optional dependency during bootstrap
    hf_hub_download = None

try:
    from transformers import AutoTokenizer, VitsModel
except ImportError:  # pragma: no cover - optional dependency during bootstrap
    AutoTokenizer = None
    VitsModel = None

from piper import PiperVoice

from core.voice_catalog import DEFAULT_VIETNAMESE_TTS_VOICE_NAME, VIETNAMESE_TTS_VOICE_CHOICES, resolve_vietnamese_tts_voice_name

try:
    from vieneu import Vieneu
except ImportError:  # pragma: no cover - optional dependency during bootstrap
    Vieneu = None

try:
    from f5_tts.api import F5TTS
except ImportError:  # pragma: no cover - optional dependency during bootstrap
    F5TTS = None

try:
    from chatterbox.tts import ChatterboxTTS
except Exception:  # pragma: no cover - optional dependency during bootstrap
    ChatterboxTTS = None


class TtsService(Protocol):
    def speak(self, text: str, *, cancel_event: Event | None = None, language: str | None = None) -> None:
        ...

    def stop(self) -> None:
        ...


@dataclass(frozen=True)
class PiperTtsConfig:
    model_path: Path | None = None
    config_path: Path | None = None
    download_dir: Path | None = None
    use_cuda: bool = False
    espeak_data_dir: Path | None = None


@dataclass(frozen=True)
class ChatterboxTtsConfig:
    device: str = "cpu"


@dataclass(frozen=True)
class MmsTtsConfig:
    model_name_or_path: str = "facebook/mms-tts-vie"
    device: str = "cpu"
    revision: str | None = None
    cache_dir: Path | None = None
    local_files_only: bool = False


@dataclass(frozen=True)
class VieneuTtsConfig:
    backbone_repo: str = "pnnbao-ump/VieNeu-TTS"
    backbone_device: str = "cpu"
    codec_repo: str = "neuphonic/distill-neucodec"
    codec_device: str = "cpu"
    hf_token: str | None = None
    voice_name: str | None = DEFAULT_VIETNAMESE_TTS_VOICE_NAME


@dataclass(frozen=True)
class F5TtsConfig:
    model_repo_id: str = "hynt/F5-TTS-Vietnamese-ViVoice"
    model_name: str = "F5TTS_v1_Base"
    ckpt_filename: str = "model_last.pt"
    vocab_filename: str = "config.json"
    device: str = "cpu"
    hf_cache_dir: Path | None = None
    hf_token: str | None = None
    reference_text: str = "xin chào, tôi là trợ lý học tập."
    reference_voice_repo_id: str = "pnnbao-ump/VieNeu-TTS"
    reference_voice_name: str | None = None
    revision: str | None = None
    local_files_only: bool = False


def _voice_models_root() -> Path:
    roaming_root = Path(os.getenv("APPDATA") or (Path.home() / "AppData" / "Roaming"))
    root = roaming_root / "learning-agent" / "voice-models"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_repo_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def _write_waveform_to_wav(wav_path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    clipped_waveform = np.clip(np.asarray(waveform, dtype=np.float32).reshape(-1), -1.0, 1.0)
    if clipped_waveform.size == 0:
        raise RuntimeError("Audio waveform was empty.")

    pcm16 = (clipped_waveform * np.iinfo(np.int16).max).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm16.tobytes())


def _split_sentence_chunks(text: str) -> list[str]:
    normalized_text = " ".join(text.strip().split())
    if not normalized_text:
        return []

    chunks = [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", normalized_text) if chunk.strip()]
    return chunks or [normalized_text]


class PiperTtsService:
    def __init__(self, config: PiperTtsConfig | None = None) -> None:
        self._config = config or PiperTtsConfig()
        self._voice: PiperVoice | None = None
        self._fallback_engine: Any | None = None

    def speak(self, text: str, *, cancel_event: Event | None = None, language: str | None = None) -> None:
        cleaned = text.strip()
        if not cleaned:
            return

        wav_path: Path | None = None
        try:
            try:
                voice = self._load_voice()
                wav_path = self._synthesize_to_wav(voice, cleaned)
                self._play_wav(wav_path, cancel_event=cancel_event)
                return
            except Exception:
                if not self._can_use_fallback_voice():
                    raise

            self._speak_with_fallback_voice(cleaned, cancel_event=cancel_event)
        finally:
            if wav_path is not None:
                with suppress(FileNotFoundError):
                    wav_path.unlink()
            self._fallback_engine = None

    def stop(self) -> None:
        if winsound is None:
            self._stop_fallback_voice()
            return

        try:
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass

        self._stop_fallback_voice()

    def _load_voice(self) -> PiperVoice:
        if self._voice is not None:
            return self._voice

        model_path = self._config.model_path
        if model_path is None:
            raise RuntimeError(
                "No Piper voice model is configured. Set LEARNING_AGENT_VOICE_TTS_MODEL_PATH to a local .onnx file."
            )

        if not model_path.exists():
            raise RuntimeError(f"Piper voice model not found: {model_path}")

        config_path = self._config.config_path
        if config_path is not None and not config_path.exists():
            raise RuntimeError(f"Piper voice config not found: {config_path}")

        try:
            load_kwargs = {
                "model_path": model_path,
                "config_path": config_path,
                "use_cuda": self._config.use_cuda,
                "download_dir": self._config.download_dir,
            }
            if self._config.espeak_data_dir is not None:
                load_kwargs["espeak_data_dir"] = self._config.espeak_data_dir

            self._voice = PiperVoice.load(**load_kwargs)
        except Exception as exc:  # noqa: BLE001 - surfaced to the controller for user feedback
            raise RuntimeError(f"Failed to load Piper voice model: {exc}") from exc

        return self._voice

    def _can_use_fallback_voice(self) -> bool:
        return pyttsx3 is not None

    def _get_fallback_engine(self) -> Any:
        if pyttsx3 is None:
            raise RuntimeError("pyttsx3 is required for the fallback voice engine.")

        if self._fallback_engine is None:
            self._fallback_engine = pyttsx3.init()
        return self._fallback_engine

    def _speak_with_fallback_voice(self, text: str, *, cancel_event: Event | None = None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            return

        engine = self._create_fallback_engine()
        self._fallback_engine = engine
        try:
            if cancel_event is not None and cancel_event.is_set():
                return

            engine.say(text)
            engine.runAndWait()
        finally:
            try:
                engine.stop()
            except Exception:
                pass

    def _stop_fallback_voice(self) -> None:
        if self._fallback_engine is None:
            return

        try:
            self._fallback_engine.stop()
        except Exception:
            pass

    def _create_fallback_engine(self) -> Any:
        if pyttsx3 is None:
            raise RuntimeError("pyttsx3 is required for the fallback voice engine.")

        try:
            return pyttsx3.init()
        except Exception as exc:  # noqa: BLE001 - surfaced to the controller for user feedback
            raise RuntimeError(f"Failed to initialize the fallback voice engine: {exc}") from exc

    def _synthesize_to_wav(self, voice: PiperVoice, text: str) -> Path:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file.close()
        wav_path = Path(temp_file.name)

        try:
            with wave.open(str(wav_path), "wb") as wav_file:
                voice.synthesize_wav(text, wav_file)
        except Exception:
            with suppress(FileNotFoundError):
                wav_path.unlink()
            raise

        return wav_path

    def _play_wav(self, wav_path: Path, *, cancel_event: Event | None = None) -> None:
        if winsound is None:
            raise RuntimeError("winsound is required for audio playback on Windows.")

        with wave.open(str(wav_path), "rb") as wav_file:
            frame_count = wav_file.getnframes()
            frame_rate = wav_file.getframerate()
            duration_seconds = frame_count / float(frame_rate) if frame_rate else 0.0

        winsound.PlaySound(str(wav_path), winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)

        deadline = time.monotonic() + max(duration_seconds, 0.0) + 0.25
        try:
            while time.monotonic() < deadline:
                if cancel_event is not None and cancel_event.is_set():
                    self.stop()
                    return
                time.sleep(0.05)
        finally:
            self.stop()


class ChatterboxTtsService:
    def __init__(self, config: ChatterboxTtsConfig | None = None) -> None:
        self._config = config or ChatterboxTtsConfig()
        self._model: Any | None = None
        self._fallback_engine: Any | None = None

    def speak(self, text: str, *, cancel_event: Event | None = None, language: str | None = None) -> None:
        cleaned = text.strip()
        if not cleaned:
            return

        wav_path: Path | None = None
        try:
            try:
                model = self._load_model()
                if cancel_event is not None and cancel_event.is_set():
                    return

                wav_path = self._synthesize_to_wav(model, cleaned)
                self._play_wav(wav_path, cancel_event=cancel_event)
                return
            except Exception:
                if not self._can_use_fallback_voice():
                    raise

            self._speak_with_fallback_voice(cleaned, cancel_event=cancel_event)
        finally:
            if wav_path is not None:
                with suppress(FileNotFoundError):
                    wav_path.unlink()
            self._fallback_engine = None

    def stop(self) -> None:
        if winsound is None:
            self._stop_fallback_voice()
            return

        try:
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass

        self._stop_fallback_voice()

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        if ChatterboxTTS is None:
            raise RuntimeError("chatterbox-tts is required for non-Vietnamese text-to-speech.")

        device = self._resolve_device(self._config.device)
        try:
            self._model = ChatterboxTTS.from_pretrained(device=device)
        except Exception as exc:  # noqa: BLE001 - surfaced to the controller for user feedback
            raise RuntimeError(f"Failed to load Chatterbox TTS model: {exc}") from exc

        return self._model

    def _synthesize_to_wav(self, model: Any, text: str) -> Path:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file.close()
        wav_path = Path(temp_file.name)

        try:
            waveform = model.generate(text)
            if isinstance(waveform, (list, tuple)):
                waveform = waveform[0]

            if hasattr(waveform, "squeeze"):
                waveform = waveform.squeeze()
            if hasattr(waveform, "detach"):
                waveform = waveform.detach()
            if hasattr(waveform, "cpu"):
                waveform = waveform.cpu()
            if hasattr(waveform, "numpy"):
                waveform = waveform.numpy()

            waveform_array = np.asarray(waveform, dtype=np.float32).reshape(-1)
            if waveform_array.size == 0:
                raise RuntimeError("Chatterbox TTS model produced an empty waveform.")

            sample_rate = int(getattr(model, "sr", 24000))
            _write_waveform_to_wav(wav_path, waveform_array, sample_rate)
        except Exception:
            with suppress(FileNotFoundError):
                wav_path.unlink()
            raise

        return wav_path

    def _play_wav(self, wav_path: Path, *, cancel_event: Event | None = None) -> None:
        if winsound is None:
            raise RuntimeError("winsound is required for audio playback on Windows.")

        with wave.open(str(wav_path), "rb") as wav_file:
            frame_count = wav_file.getnframes()
            frame_rate = wav_file.getframerate()
            duration_seconds = frame_count / float(frame_rate) if frame_rate else 0.0

        winsound.PlaySound(str(wav_path), winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)

        deadline = time.monotonic() + max(duration_seconds, 0.0) + 0.25
        try:
            while time.monotonic() < deadline:
                if cancel_event is not None and cancel_event.is_set():
                    self.stop()
                    return
                time.sleep(0.05)
        finally:
            self.stop()

    def _can_use_fallback_voice(self) -> bool:
        return pyttsx3 is not None

    def _get_fallback_engine(self) -> Any:
        if pyttsx3 is None:
            raise RuntimeError("pyttsx3 is required for the fallback voice engine.")

        if self._fallback_engine is None:
            self._fallback_engine = pyttsx3.init()
        return self._fallback_engine

    def _speak_with_fallback_voice(self, text: str, *, cancel_event: Event | None = None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            return

        engine = self._create_fallback_engine()
        self._fallback_engine = engine
        try:
            if cancel_event is not None and cancel_event.is_set():
                return

            engine.say(text)
            engine.runAndWait()
        finally:
            try:
                engine.stop()
            except Exception:
                pass

    def _stop_fallback_voice(self) -> None:
        if self._fallback_engine is None:
            return

        try:
            self._fallback_engine.stop()
        except Exception:
            pass

    def _create_fallback_engine(self) -> Any:
        if pyttsx3 is None:
            raise RuntimeError("pyttsx3 is required for the fallback voice engine.")

        try:
            return pyttsx3.init()
        except Exception as exc:  # noqa: BLE001 - surfaced to the controller for user feedback
            raise RuntimeError(f"Failed to initialize the fallback voice engine: {exc}") from exc

    @staticmethod
    def _resolve_device(device: str) -> str:
        requested_device = device.strip() or "cpu"
        if requested_device.startswith("gpu"):
            requested_device = "cuda" + requested_device[3:]
        if requested_device.startswith("cuda") and torch is not None and not torch.cuda.is_available():
            return "cpu"

        if requested_device == "mps" and torch is not None:
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend is not None and not mps_backend.is_available():
                return "cpu"

        return requested_device


class VieneuTtsService:
    def __init__(self, config: VieneuTtsConfig | None = None) -> None:
        self._config = config or VieneuTtsConfig()
        self._tts: Any | None = None
        self._voice: Any | None = None
        self._fallback_engine: Any | None = None

    @property
    def selected_vietnamese_voice_name(self) -> str:
        return self._config.voice_name or DEFAULT_VIETNAMESE_TTS_VOICE_NAME

    def set_voice_name(self, voice_name: str | None) -> None:
        resolved_voice_name = resolve_vietnamese_tts_voice_name(
            voice_name,
            VIETNAMESE_TTS_VOICE_CHOICES,
            DEFAULT_VIETNAMESE_TTS_VOICE_NAME,
        )
        if resolved_voice_name == self.selected_vietnamese_voice_name:
            return

        self._config = replace(self._config, voice_name=resolved_voice_name)
        self._voice = None

    def speak(self, text: str, *, cancel_event: Event | None = None, language: str | None = None) -> None:
        cleaned = text.strip()
        if not cleaned:
            return

        try:
            tts = self._load_tts()
            voice = self._load_voice(tts)
        except Exception:
            if not self._can_use_fallback_voice():
                raise

            try:
                self._speak_with_fallback_voice(cleaned, cancel_event=cancel_event)
            finally:
                self._fallback_engine = None
            return

        spoken_any = False
        try:
            for chunk in _split_sentence_chunks(cleaned):
                if cancel_event is not None and cancel_event.is_set():
                    return

                self._speak_chunk(tts, voice, chunk, cancel_event=cancel_event)
                spoken_any = True
        except Exception:
            if not spoken_any and self._can_use_fallback_voice():
                self._speak_with_fallback_voice(cleaned, cancel_event=cancel_event)
            else:
                raise
        finally:
            self._fallback_engine = None

    def stop(self) -> None:
        if winsound is None:
            self._stop_fallback_voice()
            return

        try:
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass

        self._stop_fallback_voice()

    def _load_tts(self) -> Any:
        if self._tts is not None:
            return self._tts

        if Vieneu is None:
            raise RuntimeError("vieneu is required for Vietnamese text-to-speech.")

        backbone_device = self._resolve_device(self._config.backbone_device)
        try:
            if self._should_use_standard_backend():
                self._tts = Vieneu(
                    mode="standard",
                    backbone_repo=self._config.backbone_repo,
                    backbone_device=backbone_device,
                    codec_repo=self._config.codec_repo,
                    codec_device=self._config.codec_device,
                    hf_token=self._config.hf_token,
                )
            else:
                self._tts = Vieneu(
                    mode="turbo",
                    backbone_repo=self._config.backbone_repo,
                    device=backbone_device,
                    hf_token=self._config.hf_token,
                )
        except Exception as exc:  # noqa: BLE001 - surfaced to the controller for user feedback
            raise RuntimeError(f"Failed to load VieNeu TTS model: {exc}") from exc

        return self._tts

    def _should_use_standard_backend(self) -> bool:
        repo_id = self._config.backbone_repo.strip().lower()
        return "turbo" not in repo_id

    def _load_voice(self, tts: Any) -> Any:
        if self._voice is not None:
            return self._voice

        requested_voice_name = self._normalize_voice_name(self.selected_vietnamese_voice_name)
        if requested_voice_name:
            matched_voice_name = self._find_matching_voice_name(tts, requested_voice_name)
            if matched_voice_name is None:
                matched_voice_name = self._find_female_voice_name(tts)
            if matched_voice_name is None:
                raise RuntimeError(f"Could not find VieNeu voice preset matching '{self._config.voice_name}'.")
        else:
            matched_voice_name = self._find_female_voice_name(tts)
            if matched_voice_name is None:
                raise RuntimeError("Could not find a female VieNeu voice preset.")

        try:
            self._voice = tts.get_preset_voice(matched_voice_name)
        except Exception as exc:  # noqa: BLE001 - surfaced to the controller for user feedback
            raise RuntimeError(f"Failed to load VieNeu voice preset '{matched_voice_name}': {exc}") from exc

        if self._voice is None:
            raise RuntimeError(f"VieNeu voice preset '{matched_voice_name}' was empty.")

        return self._voice

    def _speak_chunk(self, tts: Any, voice: Any, text: str, *, cancel_event: Event | None = None) -> None:
        infer_kwargs = {"text": text}
        if voice is not None:
            infer_kwargs["voice"] = voice

        audio = tts.infer(**infer_kwargs)
        if cancel_event is not None and cancel_event.is_set():
            return

        wav_path = self._synthesize_to_wav(tts, audio)
        try:
            self._play_wav(wav_path, cancel_event=cancel_event)
        finally:
            with suppress(FileNotFoundError):
                wav_path.unlink()

    def render_waveform(self, text: str) -> tuple[np.ndarray, int]:
        cleaned = text.strip()
        if not cleaned:
            return np.array([], dtype=np.float32), 24000

        tts = self._load_tts()
        voice = self._load_voice(tts)
        infer_kwargs = {"text": cleaned}
        if voice is not None:
            infer_kwargs["voice"] = voice

        audio = tts.infer(**infer_kwargs)
        waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
        if waveform.size == 0:
            raise RuntimeError("VieNeu TTS model produced an empty waveform.")

        sample_rate = int(getattr(tts, "sample_rate", 24000))
        return waveform, sample_rate

    def _find_female_voice_name(self, tts: Any) -> str | None:
        for description, voice_name in self._iter_preset_voice_names(tts):
            normalized_description = self._normalize_voice_name(description)
            if "(nu" in normalized_description or " female" in normalized_description:
                return voice_name

        return None

    def _find_matching_voice_name(self, tts: Any, requested_voice_name: str) -> str | None:
        for description, voice_name in self._iter_preset_voice_names(tts):
            normalized_description = self._normalize_voice_name(description)
            normalized_voice_name = self._normalize_voice_name(voice_name)
            if requested_voice_name in normalized_description or requested_voice_name in normalized_voice_name:
                return voice_name

        return None

    @staticmethod
    def _iter_preset_voice_names(tts: Any) -> list[tuple[str, str]]:
        try:
            voices = tts.list_preset_voices()
        except Exception as exc:  # noqa: BLE001 - surfaced to the controller for user feedback
            raise RuntimeError(f"Failed to list VieNeu voice presets: {exc}") from exc

        return [(str(description), str(voice_name)) for description, voice_name in voices]

    @staticmethod
    def _normalize_voice_name(value: str | None) -> str:
        if not value:
            return ""

        normalized = unicodedata.normalize("NFKD", value)
        ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
        return " ".join(ascii_text.lower().split())

    def _synthesize_to_wav(self, tts: Any, audio: Any) -> Path:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file.close()
        wav_path = Path(temp_file.name)

        try:
            waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
            if waveform.size == 0:
                raise RuntimeError("VieNeu TTS model produced an empty waveform.")

            waveform = np.clip(waveform, -1.0, 1.0)
            pcm16 = (waveform * np.iinfo(np.int16).max).astype(np.int16)
            sample_rate = int(getattr(tts, "sample_rate", 24000))

            with wave.open(str(wav_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm16.tobytes())
        except Exception:
            with suppress(FileNotFoundError):
                wav_path.unlink()
            raise

        return wav_path

    def _play_wav(self, wav_path: Path, *, cancel_event: Event | None = None) -> None:
        if winsound is None:
            raise RuntimeError("winsound is required for audio playback on Windows.")

        with wave.open(str(wav_path), "rb") as wav_file:
            frame_count = wav_file.getnframes()
            frame_rate = wav_file.getframerate()
            duration_seconds = frame_count / float(frame_rate) if frame_rate else 0.0

        winsound.PlaySound(str(wav_path), winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)

        deadline = time.monotonic() + max(duration_seconds, 0.0) + 0.25
        try:
            while time.monotonic() < deadline:
                if cancel_event is not None and cancel_event.is_set():
                    self.stop()
                    return
                time.sleep(0.05)
        finally:
            self.stop()

    def _can_use_fallback_voice(self) -> bool:
        return pyttsx3 is not None

    def _get_fallback_engine(self) -> Any:
        if pyttsx3 is None:
            raise RuntimeError("pyttsx3 is required for the fallback voice engine.")

        if self._fallback_engine is None:
            self._fallback_engine = pyttsx3.init()
        return self._fallback_engine

    def _speak_with_fallback_voice(self, text: str, *, cancel_event: Event | None = None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            return

        engine = self._create_fallback_engine()
        self._fallback_engine = engine
        try:
            if cancel_event is not None and cancel_event.is_set():
                return

            engine.say(text)
            engine.runAndWait()
        finally:
            try:
                engine.stop()
            except Exception:
                pass

    def _stop_fallback_voice(self) -> None:
        if self._fallback_engine is None:
            return

        try:
            self._fallback_engine.stop()
        except Exception:
            pass

    def _create_fallback_engine(self) -> Any:
        if pyttsx3 is None:
            raise RuntimeError("pyttsx3 is required for the fallback voice engine.")

        try:
            return pyttsx3.init()
        except Exception as exc:  # noqa: BLE001 - surfaced to the controller for user feedback
            raise RuntimeError(f"Failed to initialize the fallback voice engine: {exc}") from exc

    @staticmethod
    def _resolve_device(device: str) -> str:
        requested_device = device.strip() or "cpu"
        if requested_device.startswith("gpu"):
            requested_device = "cuda" + requested_device[3:]
        if requested_device.startswith("cuda") and torch is not None and not torch.cuda.is_available():
            return "cpu"

        if requested_device == "mps" and torch is not None:
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend is not None and not mps_backend.is_available():
                return "cpu"

        return requested_device


class F5TtsService:
    def __init__(self, config: F5TtsConfig | None = None) -> None:
        self._config = config or F5TtsConfig()
        self._tts: Any | None = None
        self._reference_tts_service: VieneuTtsService | None = None
        self._reference_audio_path: Path | None = None
        self._fallback_engine: Any | None = None

    def speak(self, text: str, *, cancel_event: Event | None = None, language: str | None = None) -> None:
        cleaned = text.strip()
        if not cleaned:
            return

        wav_path: Path | None = None
        try:
            try:
                tts = self._load_tts()
                ref_file = self._load_reference_audio()
                wav_path = self._synthesize_to_wav(tts, ref_file, cleaned)
                self._play_wav(wav_path, cancel_event=cancel_event)
                return
            except Exception:
                if not self._can_use_fallback_voice():
                    raise

            self._speak_with_fallback_voice(cleaned, cancel_event=cancel_event)
        finally:
            if wav_path is not None:
                with suppress(FileNotFoundError):
                    wav_path.unlink()
            self._fallback_engine = None

    def stop(self) -> None:
        if winsound is None:
            self._stop_fallback_voice()
            return

        try:
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass

        self._stop_fallback_voice()

    def _load_tts(self) -> Any:
        if self._tts is not None:
            return self._tts

        if F5TTS is None:
            raise RuntimeError("f5-tts is required for Vietnamese text-to-speech.")

        device = self._resolve_device(self._config.device)
        cache_root = self._cache_root()
        ckpt_file = self._resolve_repo_file(self._config.ckpt_filename, cache_root)
        vocab_file = self._resolve_repo_file(self._config.vocab_filename, cache_root)
        try:
            self._tts = F5TTS(
                model=self._config.model_name,
                ckpt_file=str(ckpt_file),
                vocab_file=str(vocab_file),
                device=device,
                hf_cache_dir=str(cache_root),
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the controller for user feedback
            raise RuntimeError(f"Failed to load F5-TTS model: {exc}") from exc

        return self._tts

    def _load_reference_audio(self) -> Path:
        if self._reference_audio_path is not None and self._reference_audio_path.exists():
            return self._reference_audio_path

        reference_audio_path = self._cache_root() / "reference.wav"
        if reference_audio_path.exists():
            self._reference_audio_path = reference_audio_path
            return reference_audio_path

        reference_tts_service = self._load_reference_tts_service()
        waveform, sample_rate = reference_tts_service.render_waveform(self._config.reference_text)

        try:
            _write_waveform_to_wav(reference_audio_path, waveform, sample_rate)
        except Exception as exc:  # noqa: BLE001 - surfaced to the controller for user feedback
            raise RuntimeError(f"Failed to create F5-TTS reference audio: {exc}") from exc

        self._reference_audio_path = reference_audio_path
        return reference_audio_path

    def _load_reference_tts_service(self) -> VieneuTtsService:
        if self._reference_tts_service is not None:
            return self._reference_tts_service

        self._reference_tts_service = VieneuTtsService(
            VieneuTtsConfig(
                backbone_repo=self._config.reference_voice_repo_id,
                backbone_device=self._config.device,
                codec_repo="neuphonic/distill-neucodec",
                codec_device=self._config.device,
                hf_token=self._config.hf_token,
                voice_name=self._config.reference_voice_name,
            )
        )
        return self._reference_tts_service

    def _synthesize_to_wav(self, tts: Any, ref_file: Path, text: str) -> Path:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file.close()
        wav_path = Path(temp_file.name)

        try:
            tts.infer(
                str(ref_file),
                self._config.reference_text,
                text,
                file_wave=str(wav_path),
                show_info=lambda *args, **kwargs: None,
            )
        except Exception:
            with suppress(FileNotFoundError):
                wav_path.unlink()
            raise

        return wav_path

    def _play_wav(self, wav_path: Path, *, cancel_event: Event | None = None) -> None:
        if winsound is None:
            raise RuntimeError("winsound is required for audio playback on Windows.")

        with wave.open(str(wav_path), "rb") as wav_file:
            frame_count = wav_file.getnframes()
            frame_rate = wav_file.getframerate()
            duration_seconds = frame_count / float(frame_rate) if frame_rate else 0.0

        winsound.PlaySound(str(wav_path), winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)

        deadline = time.monotonic() + max(duration_seconds, 0.0) + 0.25
        try:
            while time.monotonic() < deadline:
                if cancel_event is not None and cancel_event.is_set():
                    self.stop()
                    return
                time.sleep(0.05)
        finally:
            self.stop()

    def _can_use_fallback_voice(self) -> bool:
        return pyttsx3 is not None

    def _get_fallback_engine(self) -> Any:
        if pyttsx3 is None:
            raise RuntimeError("pyttsx3 is required for the fallback voice engine.")

        if self._fallback_engine is None:
            self._fallback_engine = pyttsx3.init()
        return self._fallback_engine

    def _speak_with_fallback_voice(self, text: str, *, cancel_event: Event | None = None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            return

        engine = self._create_fallback_engine()
        self._fallback_engine = engine
        try:
            if cancel_event is not None and cancel_event.is_set():
                return

            engine.say(text)
            engine.runAndWait()
        finally:
            try:
                engine.stop()
            except Exception:
                pass

    def _stop_fallback_voice(self) -> None:
        if self._fallback_engine is None:
            return

        try:
            self._fallback_engine.stop()
        except Exception:
            pass

    def _create_fallback_engine(self) -> Any:
        if pyttsx3 is None:
            raise RuntimeError("pyttsx3 is required for the fallback voice engine.")

        try:
            return pyttsx3.init()
        except Exception as exc:  # noqa: BLE001 - surfaced to the controller for user feedback
            raise RuntimeError(f"Failed to initialize the fallback voice engine: {exc}") from exc

    def _cache_root(self) -> Path:
        if self._config.hf_cache_dir is not None:
            cache_root = self._config.hf_cache_dir
        else:
            cache_root = _voice_models_root() / _safe_repo_name(self._config.model_repo_id)

        cache_root.mkdir(parents=True, exist_ok=True)
        return cache_root

    def _resolve_repo_file(self, filename: str, cache_root: Path) -> Path:
        candidate_path = Path(self._config.model_repo_id)
        if candidate_path.exists():
            local_file = candidate_path / filename
            if local_file.exists():
                return local_file
            raise FileNotFoundError(f"Could not find '{filename}' in local F5-TTS model directory: {candidate_path}")

        if hf_hub_download is None:
            raise RuntimeError("huggingface_hub is required to download the F5-TTS model files.")

        try:
            downloaded_path = hf_hub_download(
                repo_id=self._config.model_repo_id,
                filename=filename,
                cache_dir=str(cache_root),
                revision=self._config.revision,
                token=self._config.hf_token,
                local_files_only=self._config.local_files_only,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the controller for user feedback
            raise RuntimeError(f"Failed to download F5-TTS file '{filename}': {exc}") from exc

        return Path(downloaded_path)

    @staticmethod
    def _resolve_device(device: str) -> str:
        requested_device = device.strip() or "cpu"
        if requested_device.startswith("gpu"):
            requested_device = "cuda" + requested_device[3:]
        if requested_device.startswith("cuda") and torch is not None and not torch.cuda.is_available():
            return "cpu"

        if requested_device == "mps" and torch is not None:
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend is not None and not mps_backend.is_available():
                return "cpu"

        return requested_device


class MmsTtsService:
    def __init__(self, config: MmsTtsConfig | None = None) -> None:
        self._config = config or MmsTtsConfig()
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._device: str | None = None
        self._fallback_engine: Any | None = None

    def speak(self, text: str, *, cancel_event: Event | None = None, language: str | None = None) -> None:
        cleaned = text.strip()
        if not cleaned:
            return

        wav_path: Path | None = None
        try:
            try:
                model = self._load_model()
                tokenizer = self._load_tokenizer()
                wav_path = self._synthesize_to_wav(model, tokenizer, cleaned)
                self._play_wav(wav_path, cancel_event=cancel_event)
                return
            except Exception:
                if not self._can_use_fallback_voice():
                    raise

            self._speak_with_fallback_voice(cleaned, cancel_event=cancel_event)
        finally:
            if wav_path is not None:
                with suppress(FileNotFoundError):
                    wav_path.unlink()
            self._fallback_engine = None

    def stop(self) -> None:
        if winsound is None:
            self._stop_fallback_voice()
            return

        try:
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass

        self._stop_fallback_voice()

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        _, _, vits_model = self._require_transformers_dependencies()
        self._device = self._resolve_device()
        pretrained_kwargs: dict[str, Any] = {
            "cache_dir": self._config.cache_dir,
            "local_files_only": self._config.local_files_only,
        }
        if self._config.revision is not None:
            pretrained_kwargs["revision"] = self._config.revision

        model = vits_model.from_pretrained(self._config.model_name_or_path, **pretrained_kwargs)
        if model is None:
            raise RuntimeError("MMS TTS model could not be loaded.")

        model.to(self._device)
        model.eval()
        self._model = model
        return model

    def _load_tokenizer(self) -> Any:
        if self._tokenizer is not None:
            return self._tokenizer

        _, auto_tokenizer, _ = self._require_transformers_dependencies()
        pretrained_kwargs: dict[str, Any] = {
            "cache_dir": self._config.cache_dir,
            "local_files_only": self._config.local_files_only,
        }
        if self._config.revision is not None:
            pretrained_kwargs["revision"] = self._config.revision

        tokenizer = auto_tokenizer.from_pretrained(self._config.model_name_or_path, **pretrained_kwargs)
        self._tokenizer = tokenizer
        return tokenizer

    def _synthesize_to_wav(self, model: Any, tokenizer: Any, text: str) -> Path:
        torch_module, _, _ = self._require_transformers_dependencies()
        inputs = tokenizer(text, return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(self._device)
        elif isinstance(inputs, dict):
            inputs = {
                key: value.to(self._device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file.close()
        wav_path = Path(temp_file.name)
        try:
            with torch_module.no_grad():
                output = model(**inputs)

            waveform = getattr(output, "waveform", None)
            if waveform is None:
                raise RuntimeError("MMS TTS model did not return a waveform.")

            waveform_array = waveform
            if hasattr(waveform_array, "squeeze"):
                waveform_array = waveform_array.squeeze()
            if hasattr(waveform_array, "detach"):
                waveform_array = waveform_array.detach()
            if hasattr(waveform_array, "cpu"):
                waveform_array = waveform_array.cpu()
            if hasattr(waveform_array, "numpy"):
                waveform_array = waveform_array.numpy()

            waveform_array = np.asarray(waveform_array, dtype=np.float32).reshape(-1)
            if waveform_array.size == 0:
                raise RuntimeError("MMS TTS model produced an empty waveform.")

            waveform_array = np.clip(waveform_array, -1.0, 1.0)
            pcm16 = (waveform_array * np.iinfo(np.int16).max).astype(np.int16)
            sample_rate = int(getattr(getattr(model, "config", None), "sampling_rate", 16000))

            with wave.open(str(wav_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm16.tobytes())
        except Exception:
            with suppress(FileNotFoundError):
                wav_path.unlink()
            raise

        return wav_path

    def _play_wav(self, wav_path: Path, *, cancel_event: Event | None = None) -> None:
        if winsound is None:
            raise RuntimeError("winsound is required for audio playback on Windows.")

        with wave.open(str(wav_path), "rb") as wav_file:
            frame_count = wav_file.getnframes()
            frame_rate = wav_file.getframerate()
            duration_seconds = frame_count / float(frame_rate) if frame_rate else 0.0

        winsound.PlaySound(str(wav_path), winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)

        deadline = time.monotonic() + max(duration_seconds, 0.0) + 0.25
        try:
            while time.monotonic() < deadline:
                if cancel_event is not None and cancel_event.is_set():
                    self.stop()
                    return
                time.sleep(0.05)
        finally:
            self.stop()

    def _can_use_fallback_voice(self) -> bool:
        return pyttsx3 is not None

    def _get_fallback_engine(self) -> Any:
        if pyttsx3 is None:
            raise RuntimeError("pyttsx3 is required for the fallback voice engine.")

        if self._fallback_engine is None:
            self._fallback_engine = pyttsx3.init()
        return self._fallback_engine

    def _speak_with_fallback_voice(self, text: str, *, cancel_event: Event | None = None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            return

        engine = self._create_fallback_engine()
        self._fallback_engine = engine
        try:
            if cancel_event is not None and cancel_event.is_set():
                return

            engine.say(text)
            engine.runAndWait()
        finally:
            try:
                engine.stop()
            except Exception:
                pass

    def _stop_fallback_voice(self) -> None:
        if self._fallback_engine is None:
            return

        try:
            self._fallback_engine.stop()
        except Exception:
            pass

    def _create_fallback_engine(self) -> Any:
        if pyttsx3 is None:
            raise RuntimeError("pyttsx3 is required for the fallback voice engine.")

        try:
            return pyttsx3.init()
        except Exception as exc:  # noqa: BLE001 - surfaced to the controller for user feedback
            raise RuntimeError(f"Failed to initialize the fallback voice engine: {exc}") from exc

    def _resolve_device(self) -> str:
        requested_device = self._config.device.strip() or "cpu"
        if requested_device.startswith("gpu"):
            requested_device = "cuda" + requested_device[3:]
        if requested_device.startswith("cuda") and torch is not None and not torch.cuda.is_available():
            return "cpu"

        return requested_device

    @staticmethod
    def _require_transformers_dependencies() -> tuple[Any, Any, Any]:
        if torch is None or AutoTokenizer is None or VitsModel is None:
            raise RuntimeError("transformers and torch are required for MMS text-to-speech.")

        return torch, AutoTokenizer, VitsModel
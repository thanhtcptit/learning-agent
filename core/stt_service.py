from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any, Protocol

import numpy as np

from core.audio_recorder import RecordedAudio

try:
    from huggingface_hub import snapshot_download
except ImportError:  # pragma: no cover - optional dependency during bootstrap
    snapshot_download = None

try:
    from faster_whisper import WhisperModel
except ImportError:  # pragma: no cover - optional dependency during bootstrap
    WhisperModel = None

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency during bootstrap
    torch = None

try:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
except ImportError:  # pragma: no cover - optional dependency during bootstrap
    AutoModelForSpeechSeq2Seq = None
    AutoProcessor = None

try:
    import sherpa_onnx
except ImportError:  # pragma: no cover - optional dependency during bootstrap
    sherpa_onnx = None


class SttService(Protocol):
    def transcribe(
        self,
        recording: RecordedAudio | np.ndarray,
        *,
        cancel_event: Event | None = None,
        language: str | None = None,
    ) -> str:
        ...


DEFAULT_WHISPER_STT_MODEL_ID = "base"


@dataclass(frozen=True)
class WhisperSttConfig:
    model_size_or_path: str = DEFAULT_WHISPER_STT_MODEL_ID
    device: str = "cpu"
    compute_type: str = "int8"
    cpu_threads: int = 0
    num_workers: int = 1
    download_root: str | None = None
    revision: str | None = None
    local_files_only: bool = False
    beam_size: int = 5
    language: str | None = None


@dataclass(frozen=True)
class PhoWhisperSttConfig:
    model_name_or_path: str = "vinai/PhoWhisper-large"
    device: str = "cpu"
    beam_size: int = 5
    max_new_tokens: int = 256
    revision: str | None = None
    cache_dir: str | None = None
    local_files_only: bool = False
    language: str = "vi"


@dataclass(frozen=True)
class ZipformerTransducerSttConfig:
    model_name_or_path: str = "hynt/Zipformer-30M-RNNT-6000h"
    provider: str = "cpu"
    num_threads: int = 1
    sample_rate: int = 16000
    feature_dim: int = 80
    dither: float = 0.0
    decoding_method: str = "greedy_search"
    max_active_paths: int = 4
    hotwords_file: str = ""
    hotwords_score: float = 1.5
    blank_penalty: float = 0.0
    revision: str | None = None
    cache_dir: str | None = None
    local_files_only: bool = False
    debug: bool = False
    language: str = "vi"


_ZIPFORMER_MODEL_FILE_PATTERNS = ("*.onnx", "bpe.model", "config.json", "tokens.txt")
_ZIPFORMER_ENCODER_FILENAMES = (
    "encoder-epoch-20-avg-10.int8.onnx",
    "encoder-epoch-20-avg-10.onnx",
    "encoder.int8.onnx",
    "encoder.onnx",
)
_ZIPFORMER_DECODER_FILENAMES = (
    "decoder-epoch-20-avg-10.int8.onnx",
    "decoder-epoch-20-avg-10.onnx",
    "decoder.int8.onnx",
    "decoder.onnx",
)
_ZIPFORMER_JOINER_FILENAMES = (
    "joiner-epoch-20-avg-10.int8.onnx",
    "joiner-epoch-20-avg-10.onnx",
    "joiner.int8.onnx",
    "joiner.onnx",
)
_ZIPFORMER_TOKENS_FILENAMES = ("tokens.txt", "config.json")


class PhoWhisperSttService:
    def __init__(self, config: PhoWhisperSttConfig | None = None) -> None:
        self._config = config or PhoWhisperSttConfig()
        self._model: Any | None = None
        self._processor: Any | None = None
        self._device: str | None = None
        self._torch_dtype: Any | None = None

    def transcribe(
        self,
        recording: RecordedAudio | np.ndarray,
        *,
        cancel_event: Event | None = None,
        language: str | None = None,
    ) -> str:
        if cancel_event is not None and cancel_event.is_set():
            return ""

        samples = recording.samples if isinstance(recording, RecordedAudio) else np.asarray(recording, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return ""

        model = self._load_model()
        processor = self._load_processor()
        requested_language = language.strip().lower() if isinstance(language, str) and language.strip() else self._config.language

        inputs = processor(samples, sampling_rate=self._sample_rate(recording), return_tensors="pt")
        input_features = getattr(inputs, "input_features", None)
        if input_features is None and isinstance(inputs, dict):
            input_features = inputs.get("input_features")
        if input_features is None:
            raise RuntimeError("PhoWhisper processor did not return input features.")

        input_features = input_features.to(self._device, dtype=self._torch_dtype)
        generation_kwargs = {
            "forced_decoder_ids": processor.get_decoder_prompt_ids(language=requested_language, task="transcribe"),
            "num_beams": self._config.beam_size,
            "max_new_tokens": self._config.max_new_tokens,
        }

        with torch.no_grad():
            predicted_ids = model.generate(input_features, **generation_kwargs)

        if cancel_event is not None and cancel_event.is_set():
            return ""

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        if not transcription:
            return ""

        return transcription[0].strip()

    def _load_processor(self) -> Any:
        if self._processor is not None:
            return self._processor

        self._require_transformers_dependencies()
        self._processor = AutoProcessor.from_pretrained(
            self._config.model_name_or_path,
            cache_dir=self._config.cache_dir,
            revision=self._config.revision,
            local_files_only=self._config.local_files_only,
        )
        return self._processor

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        self._require_transformers_dependencies()
        self._device = self._resolve_device()
        self._torch_dtype = torch.float16 if self._device.startswith("cuda") else torch.float32
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self._config.model_name_or_path,
            torch_dtype=self._torch_dtype,
            cache_dir=self._config.cache_dir,
            revision=self._config.revision,
            local_files_only=self._config.local_files_only,
        )
        self._model.to(self._device)
        self._model.eval()
        return self._model

    def _resolve_device(self) -> str:
        requested_device = self._config.device.strip() or "cpu"
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"

        return requested_device

    @staticmethod
    def _sample_rate(recording: RecordedAudio | np.ndarray) -> int:
        if isinstance(recording, RecordedAudio):
            return recording.sample_rate

        return 16000

    @staticmethod
    def _require_transformers_dependencies() -> None:
        if torch is None or AutoModelForSpeechSeq2Seq is None or AutoProcessor is None:
            raise RuntimeError("transformers and torch are required for PhoWhisper speech-to-text.")


class ZipformerTransducerSttService:
    def __init__(self, config: ZipformerTransducerSttConfig | None = None) -> None:
        self._config = config or ZipformerTransducerSttConfig()
        self._recognizer: Any | None = None
        self._model_dir: Path | None = None

    def transcribe(
        self,
        recording: RecordedAudio | np.ndarray,
        *,
        cancel_event: Event | None = None,
        language: str | None = None,
    ) -> str:
        if cancel_event is not None and cancel_event.is_set():
            return ""

        samples = np.asarray(recording.samples if isinstance(recording, RecordedAudio) else recording, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return ""

        recognizer = self._load_recognizer()
        stream = recognizer.create_stream()
        stream.accept_waveform(self._sample_rate(recording), samples.tolist())
        recognizer.decode_stream(stream)

        if cancel_event is not None and cancel_event.is_set():
            return ""

        result = stream.result
        if callable(result):
            result = result()

        return self._extract_text(result)

    def _load_recognizer(self) -> Any:
        if self._recognizer is not None:
            return self._recognizer

        self._require_sherpa_onnx_dependencies()
        model_dir = self._resolve_model_directory()

        encoder_filename = self._find_model_file(model_dir, _ZIPFORMER_ENCODER_FILENAMES, "encoder")
        decoder_filename = self._find_model_file(model_dir, _ZIPFORMER_DECODER_FILENAMES, "decoder")
        joiner_filename = self._find_model_file(model_dir, _ZIPFORMER_JOINER_FILENAMES, "joiner")
        tokens_filename = self._find_model_file(model_dir, _ZIPFORMER_TOKENS_FILENAMES, "tokens")

        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(encoder_filename),
            decoder=str(decoder_filename),
            joiner=str(joiner_filename),
            tokens=str(tokens_filename),
            num_threads=self._config.num_threads,
            sample_rate=self._config.sample_rate,
            feature_dim=self._config.feature_dim,
            dither=self._config.dither,
            decoding_method=self._config.decoding_method,
            max_active_paths=self._config.max_active_paths,
            hotwords_file=self._config.hotwords_file,
            hotwords_score=self._config.hotwords_score,
            blank_penalty=self._config.blank_penalty,
            debug=self._config.debug,
            provider=self._config.provider,
        )
        return self._recognizer

    def _resolve_model_directory(self) -> Path:
        if self._model_dir is not None:
            return self._model_dir

        candidate_path = Path(self._config.model_name_or_path)
        if candidate_path.exists():
            self._model_dir = candidate_path
            return candidate_path

        self._require_huggingface_dependencies()
        downloaded_path = snapshot_download(
            repo_id=self._config.model_name_or_path,
            cache_dir=self._config.cache_dir,
            revision=self._config.revision,
            local_files_only=self._config.local_files_only,
            allow_patterns=_ZIPFORMER_MODEL_FILE_PATTERNS,
        )
        self._model_dir = Path(downloaded_path)
        return self._model_dir

    @staticmethod
    def _find_model_file(directory: Path, filenames: tuple[str, ...], description: str) -> Path:
        for filename in filenames:
            candidate = directory / filename
            if candidate.exists():
                return candidate

        expected = ", ".join(filenames)
        raise FileNotFoundError(f"Could not find {description} file in {directory}. Expected one of: {expected}")

    @staticmethod
    def _extract_text(result: Any) -> str:
        if result is None:
            return ""

        text = getattr(result, "text", None)
        if text is None and isinstance(result, dict):
            text = result.get("text")
        if text is None:
            text = str(result)

        return str(text).strip()

    @staticmethod
    def _sample_rate(recording: RecordedAudio | np.ndarray) -> int:
        if isinstance(recording, RecordedAudio):
            return recording.sample_rate

        return 16000

    @staticmethod
    def _require_huggingface_dependencies() -> None:
        if snapshot_download is None:
            raise RuntimeError("huggingface_hub is required to download the Vietnamese sherpa-onnx model.")

    @staticmethod
    def _require_sherpa_onnx_dependencies() -> None:
        if sherpa_onnx is None:
            raise RuntimeError("sherpa-onnx is required for Vietnamese speech-to-text.")


class WhisperSttService:
    def __init__(self, config: WhisperSttConfig | None = None) -> None:
        self._config = config or WhisperSttConfig()
        self._model: Any | None = None

    def transcribe(
        self,
        recording: RecordedAudio | np.ndarray,
        *,
        cancel_event: Event | None = None,
        language: str | None = None,
    ) -> str:
        if cancel_event is not None and cancel_event.is_set():
            return ""

        samples = recording.samples if isinstance(recording, RecordedAudio) else np.asarray(recording, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return ""

        model = self._load_model()
        requested_language = language.strip() if isinstance(language, str) else None
        if not requested_language:
            requested_language = self._config.language

        segments, _info = model.transcribe(
            samples,
            beam_size=self._config.beam_size,
            language=requested_language,
            vad_filter=False,
            condition_on_previous_text=False,
        )

        parts: list[str] = []
        for segment in segments:
            if cancel_event is not None and cancel_event.is_set():
                break

            text = segment.text.strip()
            if text:
                parts.append(text)

        return " ".join(parts).strip()

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        if WhisperModel is None:
            raise RuntimeError("faster-whisper is required for speech-to-text.")

        self._model = WhisperModel(
            self._config.model_size_or_path,
            device=self._config.device,
            compute_type=self._config.compute_type,
            cpu_threads=self._config.cpu_threads,
            num_workers=self._config.num_workers,
            download_root=self._config.download_root,
            revision=self._config.revision,
            local_files_only=self._config.local_files_only,
        )
        return self._model
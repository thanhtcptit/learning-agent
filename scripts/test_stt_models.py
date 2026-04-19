from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional during bootstrap
    def load_dotenv(*args, **kwargs) -> bool:
        return False

from core.audio_recorder import AudioRecorder, AudioRecorderConfig, VoiceCaptureError
from core.stt_service import (
    PhoWhisperSttConfig,
    PhoWhisperSttService,
    WhisperSttConfig,
    WhisperSttService,
    ZipformerTransducerSttConfig,
    ZipformerTransducerSttService,
)


@dataclass(frozen=True)
class SttModelChoice:
    key: str
    label: str
    backend: str
    model_id: str


MODEL_CHOICES: tuple[SttModelChoice, ...] = (
    SttModelChoice("whisper-base", "Whisper base", "whisper", "base"),
    SttModelChoice("whisper-small", "Whisper small", "whisper", "small"),
    SttModelChoice("whisper-medium", "Whisper medium", "whisper", "medium"),
    SttModelChoice("whisper-large-v3", "Whisper large-v3", "whisper", "large-v3"),
    SttModelChoice("whisper-large-v3-turbo", "Whisper large-v3 turbo", "whisper", "large-v3-turbo"),
    SttModelChoice("pho-whisper-medium", "PhoWhisper medium", "pho", "vinai/PhoWhisper-medium"),
    SttModelChoice("pho-whisper-large", "PhoWhisper large", "pho", "vinai/PhoWhisper-large"),
    SttModelChoice("zipformer", "Zipformer RNNT", "zipformer", "hynt/Zipformer-30M-RNNT-6000h"),
)

MODEL_BY_KEY = {choice.key: choice for choice in MODEL_CHOICES}
DEFAULT_MODEL_KEY = "whisper-small"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record microphone audio and print the transcription from a selected STT model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_BY_KEY),
        default=DEFAULT_MODEL_KEY,
        help="STT model to use.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print the available STT model choices and exit.",
    )
    parser.add_argument("--language", default=None, help="Optional language hint for the model. Use auto to let Whisper auto-detect.")
    parser.add_argument("--device", default="cpu", help="Compute device for Whisper and PhoWhisper models.")
    parser.add_argument("--compute-type", default="int8", help="Compute type for Whisper models.")
    parser.add_argument("--provider", default="cpu", help="Provider for the Zipformer model.")
    parser.add_argument("--cpu-threads", type=int, default=0, help="CPU thread count for Whisper.")
    parser.add_argument("--num-workers", type=int, default=1, help="Worker count for Whisper.")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size for Whisper and PhoWhisper.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum new tokens for PhoWhisper.")
    parser.add_argument("--num-threads", type=int, default=1, help="Thread count for Zipformer.")
    parser.add_argument("--input-device", default=None, help="Microphone input device index or name.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate.")
    parser.add_argument("--channels", type=int, default=1, help="Number of audio channels.")
    parser.add_argument("--block-duration", type=float, default=0.1, help="Audio block duration in seconds.")
    parser.add_argument("--silence-duration", type=float, default=2.0, help="Silence duration that ends a recording.")
    parser.add_argument("--silence-threshold", type=float, default=0.015, help="RMS threshold used to detect speech.")
    parser.add_argument("--initial-timeout", type=float, default=8.0, help="Timeout waiting for the first speech segment.")
    parser.add_argument("--max-record-seconds", type=float, default=60.0, help="Maximum recording length.")
    parser.add_argument("--download-root", default=None, help="Optional cache directory for faster-whisper downloads.")
    parser.add_argument("--revision", default=None, help="Optional model revision to load.")
    parser.add_argument("--local-files-only", action="store_true", help="Fail instead of downloading model files.")
    parser.add_argument("--cache-dir", default=None, help="Optional HF cache directory for PhoWhisper/Zipformer downloads.")
    parser.add_argument("--hotwords-file", default="", help="Optional hotwords file for Zipformer.")
    parser.add_argument("--hotwords-score", type=float, default=1.5, help="Hotwords score for Zipformer.")
    parser.add_argument("--blank-penalty", type=float, default=0.0, help="Blank penalty for Zipformer.")
    parser.add_argument("--debug", action="store_true", help="Enable verbose Zipformer debugging.")
    return parser


def _normalize_language_hint(language: str | None) -> str | None:
    if not isinstance(language, str):
        return None

    candidate = language.strip()
    if not candidate or candidate.lower() in {"auto", "none", "detect"}:
        return None

    return candidate


def load_environment() -> None:
    env_path = ROOT_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def print_model_choices() -> None:
    print("Available STT models:")
    for choice in MODEL_CHOICES:
        print(f"  {choice.key:<24} {choice.label}")


def build_audio_recorder(args: argparse.Namespace) -> AudioRecorder:
    config = AudioRecorderConfig(
        sample_rate=args.sample_rate,
        channels=args.channels,
        block_duration_seconds=args.block_duration,
        silence_duration_seconds=args.silence_duration,
        silence_threshold=args.silence_threshold,
        initial_timeout_seconds=args.initial_timeout,
        max_record_seconds=args.max_record_seconds,
        device=args.input_device,
    )
    return AudioRecorder(config)


def build_stt_service(args: argparse.Namespace):
    choice = MODEL_BY_KEY[args.model]
    language = _normalize_language_hint(args.language)

    if choice.backend == "whisper":
        config = WhisperSttConfig(
            model_size_or_path=choice.model_id,
            device=args.device,
            compute_type=args.compute_type,
            cpu_threads=args.cpu_threads,
            num_workers=args.num_workers,
            download_root=args.download_root,
            revision=args.revision,
            local_files_only=args.local_files_only,
            beam_size=args.beam_size,
            language=language,
        )
        return WhisperSttService(config)

    if choice.backend == "pho":
        config = PhoWhisperSttConfig(
            model_name_or_path=choice.model_id,
            device=args.device,
            beam_size=args.beam_size,
            max_new_tokens=args.max_new_tokens,
            revision=args.revision,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
            language=language or "vi",
        )
        return PhoWhisperSttService(config)

    config = ZipformerTransducerSttConfig(
        model_name_or_path=choice.model_id,
        provider=args.provider,
        num_threads=args.num_threads,
        hotwords_file=args.hotwords_file,
        hotwords_score=args.hotwords_score,
        blank_penalty=args.blank_penalty,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        debug=args.debug,
        language=language or "vi",
    )
    return ZipformerTransducerSttService(config)


def transcribe_once(args: argparse.Namespace) -> str:
    recorder = build_audio_recorder(args)
    service = build_stt_service(args)
    choice = MODEL_BY_KEY[args.model]

    print(f"Selected model: {choice.label} ({choice.key})")
    print("Speak into the microphone. Recording stops after silence.")

    recording = recorder.record_until_silence()
    transcript = service.transcribe(recording, language=_normalize_language_hint(args.language))
    return transcript.strip()


def main(argv: list[str] | None = None) -> int:
    load_environment()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_models:
        print_model_choices()
        return 0

    try:
        transcript = transcribe_once(args)
    except VoiceCaptureError as exc:
        print(f"Audio capture failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - print model/runtime errors directly for a test utility
        print(f"STT failed: {exc}", file=sys.stderr)
        return 1

    if transcript:
        print(transcript)
    else:
        print("(no transcript)")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

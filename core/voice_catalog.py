from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VoiceModelChoice:
    model_id: str
    label: str


@dataclass(frozen=True)
class VoicePresetChoice:
    voice_name: str
    label: str


DEFAULT_VIETNAMESE_STT_MODEL_ID = "hynt/Zipformer-30M-RNNT-6000h"
PHOWHISPER_MEDIUM_STT_MODEL_ID = "vinai/PhoWhisper-medium"

DEFAULT_VIETNAMESE_TTS_MODEL_ID = "pnnbao-ump/VieNeu-TTS"
VIENEU_TTS_03B_Q4_GGUF_MODEL_ID = "pnnbao-ump/VieNeu-TTS-0.3B-q4-gguf"
VIENEU_TTS_03B_MODEL_ID = "pnnbao-ump/VieNeu-TTS-0.3B"
F5_VIETNAMESE_TTS_MODEL_ID = "hynt/F5-TTS-Vietnamese-ViVoice"
DEFAULT_VIETNAMESE_TTS_VOICE_NAME = "Ly"

VIENEU_TTS_MODEL_IDS: tuple[str, ...] = (
    DEFAULT_VIETNAMESE_TTS_MODEL_ID,
    VIENEU_TTS_03B_Q4_GGUF_MODEL_ID,
    VIENEU_TTS_03B_MODEL_ID,
)


VIETNAMESE_STT_MODEL_CHOICES: tuple[VoiceModelChoice, ...] = (
    VoiceModelChoice(DEFAULT_VIETNAMESE_STT_MODEL_ID, "Zipformer RNNT (default)"),
    VoiceModelChoice(PHOWHISPER_MEDIUM_STT_MODEL_ID, "PhoWhisper medium"),
)

VIETNAMESE_TTS_MODEL_CHOICES: tuple[VoiceModelChoice, ...] = (
    VoiceModelChoice(DEFAULT_VIETNAMESE_TTS_MODEL_ID, "VieNeu TTS v1"),
    VoiceModelChoice(VIENEU_TTS_03B_Q4_GGUF_MODEL_ID, "VieNeu TTS 0.3B Q4 GGUF (CPU)"),
    VoiceModelChoice(VIENEU_TTS_03B_MODEL_ID, "VieNeu TTS 0.3B"),
    VoiceModelChoice(F5_VIETNAMESE_TTS_MODEL_ID, "F5-TTS Vietnamese ViVoice"),
)

VIETNAMESE_TTS_MODEL_CHOICES_FOR_SETTINGS: tuple[VoiceModelChoice, ...] = (
    VoiceModelChoice(DEFAULT_VIETNAMESE_TTS_MODEL_ID, "VieNeu TTS v1"),
    VoiceModelChoice(VIENEU_TTS_03B_Q4_GGUF_MODEL_ID, "VieNeu TTS 0.3B Q4 GGUF (CPU)"),
    VoiceModelChoice(VIENEU_TTS_03B_MODEL_ID, "VieNeu TTS 0.3B"),
)

VIETNAMESE_TTS_VOICE_CHOICES: tuple[VoicePresetChoice, ...] = (
    VoicePresetChoice("Vinh", "Vĩnh (nam miền Nam)"),
    VoicePresetChoice("Binh", "Bình (nam miền Bắc)"),
    VoicePresetChoice("Tuyen", "Tuyên (nam miền Bắc)"),
    VoicePresetChoice("Doan", "Đoan (nữ miền Nam)"),
    VoicePresetChoice("Ly", "Ly (nữ miền Bắc)"),
    VoicePresetChoice("Ngoc", "Ngọc (nữ miền Bắc)"),
)

VIETNAMESE_TTS_VOICE_CHOICES_BY_MODEL: dict[str, tuple[VoicePresetChoice, ...]] = {
    model_id: VIETNAMESE_TTS_VOICE_CHOICES for model_id in VIENEU_TTS_MODEL_IDS
}


def resolve_voice_model_id(model_id: str | None, choices: tuple[VoiceModelChoice, ...], default_model_id: str) -> str:
    candidate = (model_id or "").strip()
    if not candidate:
        return default_model_id

    valid_model_ids = {choice.model_id for choice in choices}
    return candidate if candidate in valid_model_ids else default_model_id


def voice_model_label(model_id: str | None, choices: tuple[VoiceModelChoice, ...], default_label: str | None = None) -> str:
    candidate = (model_id or "").strip()
    for choice in choices:
        if choice.model_id == candidate:
            return choice.label

    if default_label is not None:
        return default_label

    return candidate or "Unknown"


def resolve_vietnamese_tts_voice_name(
    voice_name: str | None,
    choices: tuple[VoicePresetChoice, ...],
    default_voice_name: str,
) -> str:
    candidate = (voice_name or "").strip()
    if not candidate:
        return default_voice_name

    valid_voice_names = {choice.voice_name for choice in choices}
    return candidate if candidate in valid_voice_names else default_voice_name


def vietnamese_tts_voice_choices_for_model(model_id: str | None) -> tuple[VoicePresetChoice, ...]:
    candidate = (model_id or "").strip()
    return VIETNAMESE_TTS_VOICE_CHOICES_BY_MODEL.get(candidate, VIETNAMESE_TTS_VOICE_CHOICES)
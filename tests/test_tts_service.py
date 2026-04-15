from __future__ import annotations

import wave
from types import SimpleNamespace

import numpy as np

import core.tts_service as tts_module
from core.voice_catalog import DEFAULT_VIETNAMESE_TTS_VOICE_NAME, VIENEU_TTS_03B_MODEL_ID, VIENEU_TTS_03B_Q4_GGUF_MODEL_ID, VIETNAMESE_TTS_VOICE_CHOICES
from core.tts_service import ChatterboxTtsConfig, ChatterboxTtsService, F5TtsConfig, F5TtsService, MmsTtsConfig, MmsTtsService, PiperTtsConfig, PiperTtsService, VieneuTtsConfig, VieneuTtsService


def test_piper_tts_service_falls_back_to_system_voice_when_no_model(monkeypatch) -> None:
    events: list[object] = []
    init_calls = 0

    class FakeEngine:
        def stop(self) -> None:
            events.append("stop")

        def say(self, text: str) -> None:
            events.append(("say", text))

        def runAndWait(self) -> None:
            events.append("run")

    def init() -> FakeEngine:
        nonlocal init_calls
        init_calls += 1
        return FakeEngine()

    monkeypatch.setattr(tts_module, "pyttsx3", SimpleNamespace(init=init))

    service = PiperTtsService(PiperTtsConfig(model_path=None))
    service.speak("Hello voice")
    service.speak("Second voice")
    service.stop()

    assert init_calls == 2
    assert events == [("say", "Hello voice"), "run", "stop", ("say", "Second voice"), "run", "stop"]


class FakeChatterboxTTS:
    loaded: list[tuple[str, "FakeChatterboxTTS"]] = []

    @classmethod
    def from_pretrained(cls, device):
        model = cls(device)
        cls.loaded.append((device, model))
        return model

    def __init__(self, device) -> None:
        self.device = device
        self.sr = 22050
        self.generate_calls: list[tuple[str, dict[str, object]]] = []

    def generate(self, text: str, **kwargs):
        self.generate_calls.append((text, kwargs))
        return np.array([[0.0, 0.5, -0.5, 0.25]], dtype=np.float32)


class FakeWaveform:
    def __init__(self, values: np.ndarray) -> None:
        self._values = values

    def squeeze(self):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class FakeTokenizerBatch(dict):
    def __init__(self) -> None:
        super().__init__(input_ids=np.array([1, 2, 3], dtype=np.int64))
        self.to_calls: list[object] = []

    def to(self, device):
        self.to_calls.append(device)
        return self


class FakeTokenizer:
    loaded: list[tuple[str, dict[str, object], "FakeTokenizer"]] = []

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        tokenizer = cls()
        cls.loaded.append((model_name_or_path, kwargs, tokenizer))
        return tokenizer

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.batch = FakeTokenizerBatch()

    def __call__(self, text: str, return_tensors: str):
        self.calls.append((text, return_tensors))
        return self.batch


class FakeMmsOutput:
    def __init__(self, waveform: FakeWaveform) -> None:
        self.waveform = waveform


class FakeMmsModel:
    loaded: list[tuple[str, dict[str, object], "FakeMmsModel"]] = []

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        model = cls()
        cls.loaded.append((model_name_or_path, kwargs, model))
        return model

    def __init__(self) -> None:
        self.to_calls: list[object] = []
        self.eval_calls = 0
        self.calls: list[dict[str, object]] = []
        self.config = SimpleNamespace(sampling_rate=22050)

    def to(self, device):
        self.to_calls.append(device)
        return self

    def eval(self):
        self.eval_calls += 1
        return self

    def __call__(self, **inputs):
        self.calls.append(inputs)
        return FakeMmsOutput(FakeWaveform(np.array([0.0, 0.5, -0.5, 0.25], dtype=np.float32)))


class FakeVieneu:
    loaded: list[tuple[dict[str, object], "FakeVieneu"]] = []
    preset_voices = [
        ("Vĩnh (nam miền Nam)", "Vinh"),
        ("Bình (nam miền Bắc)", "Binh"),
        ("Tuyên (nam miền Bắc)", "Tuyen"),
        ("Đoan (nữ miền Nam)", "Doan"),
        ("Ly (nữ miền Bắc)", "Ly"),
        ("Ngọc (nữ miền Bắc)", "Ngoc"),
    ]

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.calls: list[tuple[str, object | None]] = []
        self.preset_requests: list[str] = []
        self.sample_rate = 24000
        FakeVieneu.loaded.append((kwargs, self))

    def list_preset_voices(self):
        return self.preset_voices

    def get_preset_voice(self, voice_name: str):
        self.preset_requests.append(voice_name)
        return {"voice_name": voice_name}

    def infer(self, text: str, voice=None):
        self.calls.append((text, voice))
        return np.array([0.0, 0.5, -0.5, 0.25], dtype=np.float32)


class FakeReferenceVieneuService:
    instances: list["FakeReferenceVieneuService"] = []

    def __init__(self, config) -> None:
        self.config = config
        self.calls: list[str] = []
        FakeReferenceVieneuService.instances.append(self)

    def render_waveform(self, text: str):
        self.calls.append(text)
        return np.array([0.0, 0.2, -0.2, 0.1], dtype=np.float32), 24000


class FakeF5TTS:
    instances: list["FakeF5TTS"] = []

    def __init__(self, model, ckpt_file, vocab_file, device, hf_cache_dir) -> None:
        self.init_args = {
            "model": model,
            "ckpt_file": ckpt_file,
            "vocab_file": vocab_file,
            "device": device,
            "hf_cache_dir": hf_cache_dir,
        }
        self.infer_calls: list[tuple[str, str, str, str | None]] = []
        FakeF5TTS.instances.append(self)

    def infer(self, ref_file, ref_text, gen_text, *, file_wave=None, show_info=None, **_kwargs):
        self.infer_calls.append((str(ref_file), ref_text, gen_text, str(file_wave) if file_wave is not None else None))
        assert file_wave is not None
        with wave.open(str(file_wave), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            wav_file.writeframes((np.array([0.0, 0.5, -0.5, 0.25], dtype=np.float32) * np.iinfo(np.int16).max).astype(np.int16).tobytes())

        return np.array([0.0, 0.5, -0.5, 0.25], dtype=np.float32), 24000, None


class FakeTorch:
    class cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    def no_grad(self):
        return self._NoGrad()


def test_chatterbox_tts_service_uses_pretrained_model_and_writes_wav(monkeypatch) -> None:
    FakeChatterboxTTS.loaded.clear()
    monkeypatch.setattr(tts_module, "torch", FakeTorch())
    monkeypatch.setattr(tts_module, "ChatterboxTTS", FakeChatterboxTTS)

    service = ChatterboxTtsService(ChatterboxTtsConfig(device="cuda"))
    played_paths: list[str] = []

    def fake_play_wav(path, *, cancel_event=None) -> None:
        played_paths.append(str(path))
        with wave.open(str(path), "rb") as wav_file:
            assert wav_file.getnchannels() == 1
            assert wav_file.getsampwidth() == 2
            assert wav_file.getframerate() == 22050
            assert wav_file.getnframes() > 0

    monkeypatch.setattr(service, "_play_wav", fake_play_wav)

    service.speak("Hello from Chatterbox")

    assert played_paths
    assert FakeChatterboxTTS.loaded[0][0] == "cpu"
    chatterbox = FakeChatterboxTTS.loaded[0][1]
    assert chatterbox.generate_calls == [("Hello from Chatterbox", {})]


def test_mms_tts_service_uses_transformers_model_and_writes_wav(monkeypatch) -> None:
    FakeTokenizer.loaded.clear()
    FakeMmsModel.loaded.clear()
    monkeypatch.setattr(tts_module, "torch", FakeTorch())
    monkeypatch.setattr(tts_module, "AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr(tts_module, "VitsModel", FakeMmsModel)

    service = MmsTtsService(MmsTtsConfig(model_name_or_path="facebook/mms-tts-vie"))
    played_paths: list[str] = []

    def fake_play_wav(path, *, cancel_event=None) -> None:
        played_paths.append(str(path))
        with wave.open(str(path), "rb") as wav_file:
            assert wav_file.getnchannels() == 1
            assert wav_file.getsampwidth() == 2
            assert wav_file.getframerate() == 22050
            assert wav_file.getnframes() > 0

    monkeypatch.setattr(service, "_play_wav", fake_play_wav)

    service.speak("Xin chao")

    assert played_paths
    assert FakeTokenizer.loaded[0][0] == "facebook/mms-tts-vie"
    assert FakeMmsModel.loaded[0][0] == "facebook/mms-tts-vie"

    tokenizer = FakeTokenizer.loaded[0][2]
    model = FakeMmsModel.loaded[0][2]

    assert tokenizer.calls == [("Xin chao", "pt")]
    assert tokenizer.batch.to_calls == ["cpu"]
    assert model.to_calls == ["cpu"]
    assert model.eval_calls == 1
    assert model.calls[0]["input_ids"].tolist() == [1, 2, 3]


def test_vieneu_tts_service_uses_vieneu_model_and_writes_wav(monkeypatch) -> None:
    FakeVieneu.loaded.clear()
    monkeypatch.setattr(tts_module, "Vieneu", FakeVieneu)

    service = VieneuTtsService(VieneuTtsConfig(backbone_repo="pnnbao-ump/VieNeu-TTS"))
    played_paths: list[str] = []

    def fake_play_wav(path, *, cancel_event=None) -> None:
        played_paths.append(str(path))
        with wave.open(str(path), "rb") as wav_file:
            assert wav_file.getnchannels() == 1
            assert wav_file.getsampwidth() == 2
            assert wav_file.getframerate() == 24000
            assert wav_file.getnframes() > 0

    monkeypatch.setattr(service, "_play_wav", fake_play_wav)

    service.speak("Xin chao")

    assert played_paths
    assert FakeVieneu.loaded[0][0] == {
        "mode": "standard",
        "backbone_repo": "pnnbao-ump/VieNeu-TTS",
        "backbone_device": "cpu",
        "codec_repo": "neuphonic/distill-neucodec",
        "codec_device": "cpu",
        "hf_token": None,
    }

    vieneu = FakeVieneu.loaded[0][1]

    assert vieneu.preset_requests == [DEFAULT_VIETNAMESE_TTS_VOICE_NAME]
    assert vieneu.calls == [("Xin chao", {"voice_name": DEFAULT_VIETNAMESE_TTS_VOICE_NAME})]


def test_vieneu_tts_service_uses_standard_backend_for_0_3b_models(monkeypatch) -> None:
    FakeVieneu.loaded.clear()
    monkeypatch.setattr(tts_module, "Vieneu", FakeVieneu)

    service = VieneuTtsService(VieneuTtsConfig(backbone_repo=VIENEU_TTS_03B_Q4_GGUF_MODEL_ID))
    monkeypatch.setattr(service, "_play_wav", lambda path, *, cancel_event=None: None)

    service.speak("Xin chao")

    assert FakeVieneu.loaded[0][0]["mode"] == "standard"
    assert FakeVieneu.loaded[0][0]["backbone_repo"] == VIENEU_TTS_03B_Q4_GGUF_MODEL_ID
    assert FakeVieneu.loaded[0][0]["backbone_device"] == "cpu"
    assert FakeVieneu.loaded[0][0]["codec_device"] == "cpu"


def test_vieneu_tts_service_uses_standard_backend_for_0_3b_transformer_model(monkeypatch) -> None:
    FakeVieneu.loaded.clear()
    monkeypatch.setattr(tts_module, "Vieneu", FakeVieneu)

    service = VieneuTtsService(VieneuTtsConfig(backbone_repo=VIENEU_TTS_03B_MODEL_ID))
    monkeypatch.setattr(service, "_play_wav", lambda path, *, cancel_event=None: None)

    service.speak("Xin chao")

    assert FakeVieneu.loaded[0][0]["mode"] == "standard"
    assert FakeVieneu.loaded[0][0]["backbone_repo"] == VIENEU_TTS_03B_MODEL_ID
    assert FakeVieneu.loaded[0][0]["backbone_device"] == "cpu"
    assert FakeVieneu.loaded[0][0]["codec_device"] == "cpu"


def test_vieneu_tts_service_speaks_sentence_by_sentence(monkeypatch) -> None:
    FakeVieneu.loaded.clear()
    monkeypatch.setattr(tts_module, "Vieneu", FakeVieneu)

    service = VieneuTtsService(VieneuTtsConfig(backbone_repo="pnnbao-ump/VieNeu-TTS"))
    played_paths: list[str] = []

    def fake_play_wav(path, *, cancel_event=None) -> None:
        played_paths.append(str(path))

    monkeypatch.setattr(service, "_play_wav", fake_play_wav)

    service.speak("Xin chao. Tam biet! Hen gap lai?")

    vieneu = FakeVieneu.loaded[0][1]

    assert len(played_paths) == 3
    assert vieneu.preset_requests == [DEFAULT_VIETNAMESE_TTS_VOICE_NAME]
    assert [call[0] for call in vieneu.calls] == ["Xin chao.", "Tam biet!", "Hen gap lai?"]


def test_vieneu_tts_service_updates_voice_name_and_reloads_cached_voice(monkeypatch) -> None:
    FakeVieneu.loaded.clear()
    monkeypatch.setattr(tts_module, "Vieneu", FakeVieneu)

    service = VieneuTtsService(VieneuTtsConfig(backbone_repo="pnnbao-ump/VieNeu-TTS"))

    monkeypatch.setattr(service, "_play_wav", lambda path, *, cancel_event=None: None)

    service.speak("Xin chao")
    service.set_voice_name(VIETNAMESE_TTS_VOICE_CHOICES[1].voice_name)
    service.speak("Tam biet")

    vieneu = FakeVieneu.loaded[0][1]

    assert vieneu.preset_requests == [DEFAULT_VIETNAMESE_TTS_VOICE_NAME, VIETNAMESE_TTS_VOICE_CHOICES[1].voice_name]
    assert vieneu.calls == [
        ("Xin chao", {"voice_name": DEFAULT_VIETNAMESE_TTS_VOICE_NAME}),
        ("Tam biet", {"voice_name": VIETNAMESE_TTS_VOICE_CHOICES[1].voice_name}),
    ]


def test_vieneu_tts_service_normalizes_gpu_device_alias(monkeypatch) -> None:
    FakeVieneu.loaded.clear()
    monkeypatch.setattr(tts_module, "Vieneu", FakeVieneu)

    service = VieneuTtsService(VieneuTtsConfig(backbone_device="gpu"))

    monkeypatch.setattr(service, "_play_wav", lambda path, *, cancel_event=None: None)

    service.speak("Xin chao")

    assert FakeVieneu.loaded[0][0]["backbone_device"] == "cuda"


def test_f5_tts_service_uses_cached_reference_audio_and_writes_wav(monkeypatch, tmp_path) -> None:
    FakeReferenceVieneuService.instances.clear()
    FakeF5TTS.instances.clear()
    monkeypatch.setattr(tts_module, "F5TTS", FakeF5TTS)
    monkeypatch.setattr(tts_module, "VieneuTtsService", FakeReferenceVieneuService)

    model_dir = tmp_path / "f5-model"
    model_dir.mkdir()
    (model_dir / "model_last.pt").write_text("stub", encoding="utf-8")
    (model_dir / "config.json").write_text("stub", encoding="utf-8")

    service = F5TtsService(
        F5TtsConfig(
            model_repo_id=str(model_dir),
            reference_text="xin chào, tôi là trợ lý học tập.",
            hf_cache_dir=tmp_path / "cache",
        )
    )
    played_paths: list[str] = []

    def fake_play_wav(path, *, cancel_event=None) -> None:
        played_paths.append(str(path))
        with wave.open(str(path), "rb") as wav_file:
            assert wav_file.getnchannels() == 1
            assert wav_file.getsampwidth() == 2
            assert wav_file.getframerate() == 24000
            assert wav_file.getnframes() > 0

    monkeypatch.setattr(service, "_play_wav", fake_play_wav)

    service.speak("Xin chao")

    assert played_paths
    assert len(FakeReferenceVieneuService.instances) == 1
    reference_service = FakeReferenceVieneuService.instances[0]
    assert reference_service.calls == ["xin chào, tôi là trợ lý học tập."]

    assert len(FakeF5TTS.instances) == 1
    f5_instance = FakeF5TTS.instances[0]
    assert f5_instance.init_args["model"] == "F5TTS_v1_Base"
    assert f5_instance.init_args["ckpt_file"] == str(model_dir / "model_last.pt")
    assert f5_instance.init_args["vocab_file"] == str(model_dir / "config.json")
    assert f5_instance.infer_calls[0][1] == "xin chào, tôi là trợ lý học tập."
    assert f5_instance.infer_calls[0][2] == "Xin chao"
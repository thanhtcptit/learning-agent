from __future__ import annotations

import numpy as np
from types import SimpleNamespace

import core.stt_service as stt_module
from core.audio_recorder import RecordedAudio
from core.stt_service import PhoWhisperSttConfig, PhoWhisperSttService, ZipformerTransducerSttConfig, ZipformerTransducerSttService


class FakeTensor:
    def __init__(self) -> None:
        self.to_calls: list[tuple[object, object | None]] = []

    def to(self, device, dtype=None):
        self.to_calls.append((device, dtype))
        return self


class FakeInputs:
    def __init__(self, input_features: FakeTensor) -> None:
        self.input_features = input_features


class FakeProcessor:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, int, str]] = []
        self.prompt_calls: list[tuple[str, str]] = []
        self.decode_calls: list[tuple[object, bool]] = []
        self.input_features = FakeTensor()

    def __call__(self, samples, *, sampling_rate, return_tensors):
        self.calls.append((np.asarray(samples, dtype=np.float32).copy(), sampling_rate, return_tensors))
        return FakeInputs(self.input_features)

    def get_decoder_prompt_ids(self, *, language, task):
        self.prompt_calls.append((language, task))
        return [(0, 1)]

    def batch_decode(self, predicted_ids, skip_special_tokens):
        self.decode_calls.append((predicted_ids, skip_special_tokens))
        return ["  Xin chao  "]


class FakeModel:
    def __init__(self) -> None:
        self.to_calls: list[object] = []
        self.eval_calls = 0
        self.generate_calls: list[tuple[object, dict[str, object]]] = []

    def to(self, device):
        self.to_calls.append(device)
        return self

    def eval(self):
        self.eval_calls += 1
        return self

    def generate(self, input_features, **kwargs):
        self.generate_calls.append((input_features, kwargs))
        return [[1, 2, 3]]


class FakeOfflineStream:
    def __init__(self) -> None:
        self.accept_calls: list[tuple[object, np.ndarray]] = []
        self.result_calls = 0

    def accept_waveform(self, sample_rate, waveform):
        self.accept_calls.append((sample_rate, np.asarray(waveform, dtype=np.float32).copy()))

    @property
    def result(self):
        self.result_calls += 1
        return SimpleNamespace(text="  Xin chao tu Zipformer  ")


class FakeOfflineRecognizer:
    loaded: list[dict[str, object]] = []
    instances: list["FakeOfflineRecognizer"] = []

    def __init__(self, kwargs: dict[str, object]) -> None:
        self.kwargs = kwargs
        self.stream = FakeOfflineStream()
        self.decode_calls = 0

    @classmethod
    def from_transducer(cls, **kwargs):
        recognizer = cls(kwargs)
        cls.loaded.append(kwargs)
        cls.instances.append(recognizer)
        return recognizer

    def create_stream(self):
        return self.stream

    def decode_stream(self, stream):
        self.decode_calls += 1
        self.decoded_stream = stream


class FakeTorch:
    float16 = "float16"
    float32 = "float32"

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


class FakeAutoProcessor:
    loaded: list[tuple[str, dict[str, object], FakeProcessor]] = []

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        processor = FakeProcessor()
        cls.loaded.append((model_name_or_path, kwargs, processor))
        return processor


class FakeAutoModelForSpeechSeq2Seq:
    loaded: list[tuple[str, dict[str, object], FakeModel]] = []

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        model = FakeModel()
        cls.loaded.append((model_name_or_path, kwargs, model))
        return model


def test_phowhisper_stt_service_uses_transformers_model(monkeypatch) -> None:
    FakeAutoProcessor.loaded.clear()
    FakeAutoModelForSpeechSeq2Seq.loaded.clear()
    monkeypatch.setattr(stt_module, "torch", FakeTorch())
    monkeypatch.setattr(stt_module, "AutoProcessor", FakeAutoProcessor)
    monkeypatch.setattr(stt_module, "AutoModelForSpeechSeq2Seq", FakeAutoModelForSpeechSeq2Seq)

    recording = RecordedAudio(samples=np.array([0.1, -0.2, 0.3], dtype=np.float32), sample_rate=16000)
    service = PhoWhisperSttService(PhoWhisperSttConfig(model_name_or_path="vinai/PhoWhisper-large"))

    result = service.transcribe(recording, language="vi")

    assert result == "Xin chao"
    assert FakeAutoProcessor.loaded[0][0] == "vinai/PhoWhisper-large"
    assert FakeAutoModelForSpeechSeq2Seq.loaded[0][0] == "vinai/PhoWhisper-large"

    processor = FakeAutoProcessor.loaded[0][2]
    model = FakeAutoModelForSpeechSeq2Seq.loaded[0][2]

    assert processor.calls[0][1:] == (16000, "pt")
    np.testing.assert_array_equal(processor.calls[0][0], recording.samples)
    assert processor.prompt_calls == [("vi", "transcribe")]
    assert processor.input_features.to_calls == [("cpu", "float32")]
    assert model.to_calls == ["cpu"]
    assert model.eval_calls == 1
    assert model.generate_calls[0][1]["num_beams"] == 5
    assert model.generate_calls[0][1]["max_new_tokens"] == 256


def test_zipformer_transducer_stt_service_uses_sherpa_onnx_transducer(monkeypatch, tmp_path) -> None:
    FakeOfflineRecognizer.loaded.clear()
    FakeOfflineRecognizer.instances.clear()

    model_dir = tmp_path / "zipformer-model"
    model_dir.mkdir()
    for filename in (
        "encoder-epoch-20-avg-10.int8.onnx",
        "decoder-epoch-20-avg-10.int8.onnx",
        "joiner-epoch-20-avg-10.int8.onnx",
        "config.json",
    ):
        (model_dir / filename).write_text("stub", encoding="utf-8")

    snapshot_calls: list[dict[str, object]] = []

    def fake_snapshot_download(**kwargs):
        snapshot_calls.append(kwargs)
        return str(model_dir)

    monkeypatch.setattr(stt_module, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(stt_module, "sherpa_onnx", SimpleNamespace(OfflineRecognizer=FakeOfflineRecognizer))

    service = ZipformerTransducerSttService(
        ZipformerTransducerSttConfig(
            model_name_or_path="hynt/Zipformer-30M-RNNT-6000h",
            provider="cpu",
            num_threads=2,
            cache_dir=str(tmp_path / "cache"),
            local_files_only=True,
        )
    )

    recording = RecordedAudio(samples=np.array([0.1, -0.2, 0.3], dtype=np.float32), sample_rate=22050)

    result = service.transcribe(recording, language="vi")

    assert result == "Xin chao tu Zipformer"
    assert snapshot_calls == [
        {
            "repo_id": "hynt/Zipformer-30M-RNNT-6000h",
            "cache_dir": str(tmp_path / "cache"),
            "revision": None,
            "local_files_only": True,
            "allow_patterns": ("*.onnx", "bpe.model", "config.json", "tokens.txt"),
        }
    ]

    loaded_kwargs = FakeOfflineRecognizer.loaded[0]
    assert loaded_kwargs["encoder"] == str(model_dir / "encoder-epoch-20-avg-10.int8.onnx")
    assert loaded_kwargs["decoder"] == str(model_dir / "decoder-epoch-20-avg-10.int8.onnx")
    assert loaded_kwargs["joiner"] == str(model_dir / "joiner-epoch-20-avg-10.int8.onnx")
    assert loaded_kwargs["tokens"] == str(model_dir / "config.json")
    assert loaded_kwargs["num_threads"] == 2
    assert loaded_kwargs["sample_rate"] == 16000
    assert loaded_kwargs["feature_dim"] == 80
    assert loaded_kwargs["dither"] == 0.0
    assert loaded_kwargs["decoding_method"] == "greedy_search"
    assert loaded_kwargs["provider"] == "cpu"

    recognizer = FakeOfflineRecognizer.instances[0]
    assert recognizer.decode_calls == 1
    assert recognizer.stream.accept_calls[0][0] == 22050
    np.testing.assert_array_equal(recognizer.stream.accept_calls[0][1], recording.samples)
    assert recognizer.stream.result_calls == 1

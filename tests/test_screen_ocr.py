from __future__ import annotations

import sys
import types

import core.screen_ocr as screen_ocr_module


def test_screen_ocr_service_captures_monitor_under_cursor_and_extracts_text(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeShot:
        rgb = b"fake-rgb"
        size = (100, 100)

    class FakeMssContext:
        def __init__(self) -> None:
            self.monitors = [
                {"left": 0, "top": 0, "width": 200, "height": 100},
                {"left": 0, "top": 0, "width": 100, "height": 100},
                {"left": 100, "top": 0, "width": 100, "height": 100},
            ]
            self.grabbed: list[dict[str, int]] = []

        def __enter__(self) -> "FakeMssContext":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def grab(self, monitor: dict[str, int]) -> FakeShot:
            self.grabbed.append(monitor)
            captured["monitor"] = monitor
            return FakeShot()

    class FakeRapidOCR:
        def __init__(self, **kwargs) -> None:
            captured["ocr_init_kwargs"] = kwargs

        def __call__(self, img_content, **kwargs):
            captured["ocr_input"] = img_content
            captured["ocr_kwargs"] = kwargs
            return (
                [
                    [[[0.0, 0.0], [20.0, 0.0], [20.0, 10.0], [0.0, 10.0]], "Alpha Beta", "0.99"],
                    [[[0.0, 20.0], [20.0, 20.0], [20.0, 30.0], [0.0, 30.0]], "Gamma", "0.95"],
                ],
                [0.1, 0.2, 0.3],
            )

    fake_mss_module = types.ModuleType("mss")
    fake_mss_module.mss = lambda: FakeMssContext()

    fake_tools_module = types.ModuleType("mss.tools")
    fake_tools_module.to_png = lambda data, size: b"png-bytes"

    fake_rapidocr_module = types.ModuleType("rapidocr_onnxruntime")
    fake_rapidocr_module.RapidOCR = FakeRapidOCR

    monkeypatch.setitem(sys.modules, "mss", fake_mss_module)
    monkeypatch.setitem(sys.modules, "mss.tools", fake_tools_module)
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", fake_rapidocr_module)

    service = screen_ocr_module.ScreenOcrService()
    monkeypatch.setattr(service, "_cursor_position", lambda: (150, 50))

    text = service.capture_screen_text()

    assert text == "Alpha Beta\nGamma"
    assert captured["monitor"] == {"left": 100, "top": 0, "width": 100, "height": 100}
    assert captured["ocr_input"] == b"png-bytes"


def test_screen_ocr_service_filters_to_relevant_text_when_selection_is_provided(monkeypatch) -> None:
    captured: dict[str, object] = {"ocr_inputs": []}

    class FakeShot:
        rgb = b"\x00" * (100 * 100 * 3)
        size = (100, 100)

    class FakeMssContext:
        def __init__(self) -> None:
            self.monitors = [{"left": 0, "top": 0, "width": 100, "height": 100}]

        def __enter__(self) -> "FakeMssContext":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def grab(self, monitor: dict[str, int]) -> FakeShot:
            return FakeShot()

    class FakeRapidOCR:
        def __init__(self, **kwargs) -> None:
            return None

        def __call__(self, img_content, **kwargs):
            captured["ocr_inputs"].append(img_content)
            if img_content == b"full-png":
                return (
                    [
                        [[[0.0, 0.0], [20.0, 0.0], [20.0, 10.0], [0.0, 10.0]], "Noise header", "0.91"],
                        [[[0.0, 20.0], [50.0, 20.0], [50.0, 30.0], [0.0, 30.0]], "Linear algebra", "0.99"],
                        [[[0.0, 40.0], [70.0, 40.0], [70.0, 50.0], [0.0, 50.0]], "Vector spaces and matrices", "0.97"],
                    ],
                    [0.1, 0.2, 0.3],
                )

            return (
                [
                    [[[0.0, 0.0], [50.0, 0.0], [50.0, 10.0], [0.0, 10.0]], "Linear algebra", "0.99"],
                    [[[0.0, 20.0], [70.0, 20.0], [70.0, 30.0], [0.0, 30.0]], "Vector spaces and matrices", "0.97"],
                ],
                [0.1, 0.2, 0.3],
            )

    fake_mss_module = types.ModuleType("mss")
    fake_mss_module.mss = lambda: FakeMssContext()

    fake_tools_module = types.ModuleType("mss.tools")
    fake_tools_module.to_png = lambda data, size: b"full-png" if size == (100, 100) else b"crop-png"

    fake_rapidocr_module = types.ModuleType("rapidocr_onnxruntime")
    fake_rapidocr_module.RapidOCR = FakeRapidOCR

    monkeypatch.setitem(sys.modules, "mss", fake_mss_module)
    monkeypatch.setitem(sys.modules, "mss.tools", fake_tools_module)
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", fake_rapidocr_module)

    service = screen_ocr_module.ScreenOcrService()

    text = service.capture_screen_text("linear algebra")

    assert text == "linear algebra\nNoise header\nVector spaces and matrices"
    assert captured["ocr_inputs"] == [b"full-png", b"crop-png"]
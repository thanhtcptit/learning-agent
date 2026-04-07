from __future__ import annotations

import ctypes
import re
import os
from ctypes import wintypes
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any


@dataclass(frozen=True)
class _OcrBlock:
    text: str
    top: float
    left: float
    score: float


class ScreenOcrService:
    def __init__(self) -> None:
        self._engine: Any | None = None

    def capture_screen_text(self, selection_text: str | None = None) -> str:
        if os.name != "nt":
            raise RuntimeError("Screen OCR is only supported on Windows.")

        image_bytes = self._capture_screen_png_bytes()
        if not image_bytes:
            return ""

        engine = self._ensure_engine()
        result = engine(image_bytes)
        blocks = self._extract_blocks(result)
        if selection_text is not None and selection_text.strip():
            blocks = self._filter_relevant_blocks(blocks, selection_text)
        return self._join_blocks(blocks)

    def _ensure_engine(self) -> Any:
        if self._engine is not None:
            return self._engine

        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:  # pragma: no cover - dependency availability is validated at runtime
            raise RuntimeError("Screen OCR requires rapidocr-onnxruntime.") from exc

        self._engine = RapidOCR(print_verbose=False)
        return self._engine

    def _capture_screen_png_bytes(self) -> bytes:
        try:
            from mss import mss
            from mss.tools import to_png
        except ImportError as exc:  # pragma: no cover - dependency availability is validated at runtime
            raise RuntimeError("Screen OCR requires mss.") from exc

        with mss() as screen_source:
            monitor = self._select_monitor(screen_source.monitors)
            screenshot = screen_source.grab(monitor)
            png_bytes = to_png(screenshot.rgb, screenshot.size)
        return png_bytes or b""

    def _select_monitor(self, monitors: list[dict[str, int]]) -> dict[str, int]:
        if not monitors:
            raise RuntimeError("No monitors are available for screen capture.")

        if len(monitors) == 1:
            return monitors[0]

        cursor_x, cursor_y = self._cursor_position()
        for monitor in monitors[1:]:
            if self._point_in_monitor(cursor_x, cursor_y, monitor):
                return monitor

        return monitors[1]

    def _cursor_position(self) -> tuple[int, int]:
        point = wintypes.POINT()
        if not ctypes.windll.user32.GetCursorPos(ctypes.byref(point)):
            raise RuntimeError("Failed to read the cursor position.")
        return point.x, point.y

    def _point_in_monitor(self, x: int, y: int, monitor: dict[str, int]) -> bool:
        left = int(monitor.get("left", 0))
        top = int(monitor.get("top", 0))
        width = int(monitor.get("width", 0))
        height = int(monitor.get("height", 0))
        return left <= x < left + width and top <= y < top + height

    def _extract_blocks(self, result: object) -> list[_OcrBlock]:
        candidate = result[0] if isinstance(result, tuple) and result else result
        if not isinstance(candidate, list):
            return []

        blocks: list[_OcrBlock] = []
        for item in candidate:
            block = self._coerce_block(item)
            if block is not None:
                blocks.append(block)

        blocks.sort(key=lambda block: (block.top, block.left, block.text.lower()))
        return blocks

    def _coerce_block(self, item: object) -> _OcrBlock | None:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            return None

        text = self._collapse_whitespace(str(item[1]))
        if not text:
            return None

        bbox = item[0]
        left, top = self._extract_position(bbox)
        score = self._extract_score(item)
        return _OcrBlock(text=text, top=top, left=left, score=score)

    def _extract_position(self, bbox: object) -> tuple[float, float]:
        points: list[tuple[float, float]] = []

        if isinstance(bbox, (list, tuple)):
            for point in bbox:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue
                try:
                    x_value = float(point[0])
                    y_value = float(point[1])
                except (TypeError, ValueError):
                    continue
                points.append((x_value, y_value))

        if not points:
            return 0.0, 0.0

        left = min(point[0] for point in points)
        top = min(point[1] for point in points)
        return left, top

    def _extract_score(self, item: object) -> float:
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            try:
                return float(item[2])
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    def _filter_relevant_blocks(self, blocks: list[_OcrBlock], selection_text: str) -> list[_OcrBlock]:
        normalized_anchor = self._normalize_for_match(selection_text)
        if not normalized_anchor:
            return blocks

        scored_blocks: list[tuple[float, int, _OcrBlock]] = []
        for index, block in enumerate(blocks):
            relevance = self._relevance_score(normalized_anchor, block.text)
            if relevance > 0.0:
                scored_blocks.append((relevance, index, block))

        if not scored_blocks:
            return []

        best_score = max(score for score, _, _ in scored_blocks)
        threshold = 0.35

        selected_blocks = [block for score, _, block in scored_blocks if score >= threshold]
        if selected_blocks:
            selected_set = set(selected_blocks)
            return [block for block in blocks if block in selected_set]

        if best_score >= 0.25:
            best_block = max(scored_blocks, key=lambda item: (item[0], -item[1]))[2]
            return [best_block]

        return []

    def _relevance_score(self, anchor: str, candidate: str) -> float:
        normalized_candidate = self._normalize_for_match(candidate)
        if not normalized_candidate:
            return 0.0

        if anchor in normalized_candidate or normalized_candidate in anchor:
            return 1.0

        anchor_tokens = set(anchor.split())
        candidate_tokens = set(normalized_candidate.split())
        token_overlap = 0.0
        if anchor_tokens and candidate_tokens:
            token_overlap = len(anchor_tokens & candidate_tokens) / max(len(anchor_tokens), len(candidate_tokens))

        similarity = SequenceMatcher(None, anchor, normalized_candidate).ratio()
        return max(similarity, token_overlap)

    def _normalize_for_match(self, text: str) -> str:
        collapsed = self._collapse_whitespace(text).lower()
        if not collapsed:
            return ""

        stripped = re.sub(r"[\W_]+", " ", collapsed, flags=re.UNICODE)
        return self._collapse_whitespace(stripped)

    def _join_blocks(self, blocks: list[_OcrBlock]) -> str:
        return "\n".join(block.text for block in blocks).strip()

    def _collapse_whitespace(self, text: str) -> str:
        return " ".join(text.split())
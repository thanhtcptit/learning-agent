from __future__ import annotations

import ctypes
import math
import re
import os
from ctypes import wintypes
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any


@dataclass(frozen=True)
class _CapturedScreen:
    rgb: bytes
    width: int
    height: int


@dataclass(frozen=True)
class _ScreenRegion:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


@dataclass(frozen=True)
class _OcrBlock:
    text: str
    top: float
    left: float
    right: float
    bottom: float
    score: float


class ScreenOcrService:
    def __init__(self) -> None:
        self._engine: Any | None = None

    def capture_screen_text(self, selection_text: str | None = None) -> str:
        if os.name != "nt":
            raise RuntimeError("Screen OCR is only supported on Windows.")

        capture = self._capture_screen()
        if not capture.rgb or capture.width <= 0 or capture.height <= 0:
            return ""

        engine = self._ensure_engine()
        blocks = self._run_ocr(engine, capture.rgb, capture.width, capture.height)
        if selection_text is not None and selection_text.strip():
            selected_text = self._collapse_whitespace(selection_text)
            focus_block = self._find_focus_block(blocks, selection_text)
            if focus_block is None:
                return selected_text

            focus_blocks = self._collect_line_blocks(blocks, focus_block)
            region = self._build_region_from_blocks(focus_blocks, capture.width, capture.height)
            context_blocks = list(focus_blocks)
            if region is not None and (region.width < capture.width or region.height < capture.height):
                cropped_rgb, cropped_width, cropped_height = self._crop_rgb_bytes(
                    capture.rgb,
                    capture.width,
                    capture.height,
                    region,
                )
                if cropped_rgb and cropped_width > 0 and cropped_height > 0:
                    cropped_blocks = self._run_ocr(engine, cropped_rgb, cropped_width, cropped_height)
                    if cropped_blocks:
                        cropped_focus_block = self._find_focus_block(cropped_blocks, selection_text)
                        if cropped_focus_block is not None:
                            cropped_focus_blocks = self._collect_line_blocks(cropped_blocks, cropped_focus_block)
                            if cropped_focus_blocks:
                                pass

            context_text = self._join_blocks(context_blocks)
            return self._combine_selection_and_context(selected_text, context_text)
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

    def _capture_screen(self) -> _CapturedScreen:
        try:
            from mss import mss
        except ImportError as exc:  # pragma: no cover - dependency availability is validated at runtime
            raise RuntimeError("Screen OCR requires mss.") from exc

        with mss() as screen_source:
            monitor = self._select_monitor(screen_source.monitors)
            screenshot = screen_source.grab(monitor)
        width = int(screenshot.size[0])
        height = int(screenshot.size[1])
        return _CapturedScreen(rgb=screenshot.rgb or b"", width=width, height=height)

    def _run_ocr(self, engine: Any, rgb_bytes: bytes, width: int, height: int) -> list[_OcrBlock]:
        if not rgb_bytes or width <= 0 or height <= 0:
            return []

        try:
            from mss.tools import to_png
        except ImportError as exc:  # pragma: no cover - dependency availability is validated at runtime
            raise RuntimeError("Screen OCR requires mss.") from exc

        png_bytes = to_png(rgb_bytes, (width, height))
        if not png_bytes:
            return []

        return self._extract_blocks(engine(png_bytes))

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
        left, top, right, bottom = self._extract_bounds(bbox)
        score = self._extract_score(item)
        return _OcrBlock(text=text, top=top, left=left, right=right, bottom=bottom, score=score)

    def _extract_bounds(self, bbox: object) -> tuple[float, float, float, float]:
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
            return 0.0, 0.0, 0.0, 0.0

        left = min(point[0] for point in points)
        top = min(point[1] for point in points)
        right = max(point[0] for point in points)
        bottom = max(point[1] for point in points)
        return left, top, right, bottom

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

    def _find_focus_block(self, blocks: list[_OcrBlock], selection_text: str) -> _OcrBlock | None:
        normalized_anchor = self._normalize_for_match(selection_text)
        if not normalized_anchor:
            return None

        scored_blocks: list[tuple[float, int, _OcrBlock]] = []
        for index, block in enumerate(blocks):
            relevance = self._relevance_score(normalized_anchor, block.text)
            if relevance > 0.0:
                scored_blocks.append((relevance, index, block))

        if not scored_blocks:
            return None

        best_score = max(score for score, _, _ in scored_blocks)
        if best_score < 0.25:
            return None

        return max(scored_blocks, key=lambda item: (item[0], -item[1]))[2]

    def _collect_line_blocks(self, blocks: list[_OcrBlock], focus_block: _OcrBlock) -> list[_OcrBlock]:
        if not blocks:
            return []

        line_groups = self._group_blocks_into_lines(blocks)
        focus_line_index = next(
            (index for index, line in enumerate(line_groups) if focus_block in line),
            None,
        )

        if focus_line_index is None:
            return [focus_block]

        start_index = max(0, focus_line_index - 1)
        end_index = min(len(line_groups), focus_line_index + 2)

        context_blocks: list[_OcrBlock] = []
        for line in line_groups[start_index:end_index]:
            context_blocks.extend(line)

        return context_blocks

    def _group_blocks_into_lines(self, blocks: list[_OcrBlock]) -> list[list[_OcrBlock]]:
        grouped_lines: list[list[_OcrBlock]] = []
        line_centers: list[float] = []
        line_heights: list[float] = []

        for block in sorted(blocks, key=lambda block: (block.top, block.left, block.text.lower())):
            block_center_y = (block.top + block.bottom) / 2.0
            block_height = max(block.bottom - block.top, 1.0)

            if not grouped_lines:
                grouped_lines.append([block])
                line_centers.append(block_center_y)
                line_heights.append(block_height)
                continue

            current_line_center = line_centers[-1]
            current_line_height = max(line_heights[-1], block_height)
            if abs(block_center_y - current_line_center) <= max(8.0, current_line_height * 0.5):
                previous_count = len(grouped_lines[-1])
                grouped_lines[-1].append(block)
                line_centers[-1] = ((current_line_center * previous_count) + block_center_y) / (previous_count + 1)
                line_heights[-1] = current_line_height
            else:
                grouped_lines.append([block])
                line_centers.append(block_center_y)
                line_heights.append(block_height)

        return grouped_lines

    def _build_region_from_blocks(self, blocks: list[_OcrBlock], image_width: int, image_height: int) -> _ScreenRegion | None:
        if not blocks or image_width <= 0 or image_height <= 0:
            return None

        min_left = min(block.left for block in blocks)
        min_top = min(block.top for block in blocks)
        max_right = max(block.right for block in blocks)
        max_bottom = max(block.bottom for block in blocks)

        content_width = max(max_right - min_left, 1.0)
        content_height = max(max_bottom - min_top, 1.0)

        padding_x = max(10, int(content_width * 0.08))
        padding_y = max(10, int(content_height * 0.25))

        left = max(0, math.floor(min_left - padding_x))
        top = max(0, math.floor(min_top - padding_y))
        right = min(image_width, math.ceil(max_right + padding_x))
        bottom = min(image_height, math.ceil(max_bottom + padding_y))

        if right <= left or bottom <= top:
            return None

        return _ScreenRegion(left=left, top=top, right=right, bottom=bottom)

    def _crop_rgb_bytes(
        self,
        rgb_bytes: bytes,
        image_width: int,
        image_height: int,
        region: _ScreenRegion,
    ) -> tuple[bytes, int, int]:
        if not rgb_bytes or image_width <= 0 or image_height <= 0:
            return b"", 0, 0

        left = max(0, min(region.left, image_width - 1))
        top = max(0, min(region.top, image_height - 1))
        right = max(left + 1, min(region.right, image_width))
        bottom = max(top + 1, min(region.bottom, image_height))

        crop_width = right - left
        crop_height = bottom - top
        if crop_width <= 0 or crop_height <= 0:
            return b"", 0, 0

        bytes_per_pixel = 3
        row_stride = image_width * bytes_per_pixel
        crop_row_size = crop_width * bytes_per_pixel
        left_offset = left * bytes_per_pixel
        source = memoryview(rgb_bytes)

        cropped_rows: list[bytes] = []
        for row in range(top, bottom):
            row_start = row * row_stride + left_offset
            row_end = row_start + crop_row_size
            cropped_rows.append(source[row_start:row_end].tobytes())

        return b"".join(cropped_rows), crop_width, crop_height

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
        grouped_lines = self._group_blocks_into_lines(blocks)
        return "\n".join(self._join_line_blocks(line) for line in grouped_lines).strip()

    def _join_line_blocks(self, blocks: list[_OcrBlock]) -> str:
        return self._collapse_whitespace(" ".join(block.text for block in blocks))

    def _merge_context_blocks(self, primary_blocks: list[_OcrBlock], secondary_blocks: list[_OcrBlock]) -> list[_OcrBlock]:
        merged: list[_OcrBlock] = []
        seen_keys: set[str] = set()

        for block in sorted([*primary_blocks, *secondary_blocks], key=lambda block: (block.top, block.left, block.text.lower())):
            normalized_text = self._normalize_for_match(block.text)
            if not normalized_text or normalized_text in seen_keys:
                continue

            seen_keys.add(normalized_text)
            merged.append(block)

        return merged

    def _combine_selection_and_context(self, selection_text: str, context_text: str) -> str:
        cleaned_selection = self._collapse_whitespace(selection_text)
        cleaned_context = self._collapse_whitespace(context_text)

        if not cleaned_selection:
            return cleaned_context

        if not cleaned_context:
            return cleaned_selection

        selection_key = self._normalize_for_match(cleaned_selection)
        context_lines: list[str] = []
        for raw_line in context_text.splitlines():
            cleaned_line = self._collapse_whitespace(raw_line)
            if not cleaned_line:
                continue
            if self._normalize_for_match(cleaned_line) == selection_key:
                continue
            context_lines.append(cleaned_line)

        if not context_lines:
            return cleaned_selection

        return "\n".join([cleaned_selection, *context_lines]).strip()

    def _collapse_whitespace(self, text: str) -> str:
        return " ".join(text.split())
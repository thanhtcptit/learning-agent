from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QSize

from ui.main_window import _calculate_hotkey_window_position


def test_calculate_hotkey_window_position_places_window_on_opposite_side_of_cursor() -> None:
    geometry = QRect(0, 0, 1200, 800)
    window_size = QSize(480, 752)

    left_side_position = _calculate_hotkey_window_position(geometry, QPoint(180, 400), window_size)
    right_side_position = _calculate_hotkey_window_position(geometry, QPoint(980, 400), window_size)

    assert left_side_position.x() == 1200 - 480 - 16
    assert right_side_position.x() == 16
    assert left_side_position.y() == 24
    assert right_side_position.y() == 24


def test_calculate_hotkey_window_position_clamps_to_screen_bounds() -> None:
    geometry = QRect(100, 100, 700, 500)
    window_size = QSize(480, 420)

    position = _calculate_hotkey_window_position(geometry, QPoint(110, 120), window_size)

    assert position.x() == 100 + 700 - 480 - 16
    assert position.y() == 100 + 16
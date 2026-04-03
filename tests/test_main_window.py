from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QSize

import ui.main_window as main_window_module

from ui.main_window import _calculate_hotkey_window_position
from session.manager import ConversationMessage


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


def test_hotkey_window_waits_for_captured_user_message(monkeypatch) -> None:
    class DummyWindow:
        def __init__(self) -> None:
            self._hotkey_pending = False
            self.show_calls = 0
            self.show_normal_calls = 0
            self.raise_calls = 0
            self.activate_calls = 0
            self.present_calls = 0

        def _show_for_hotkey(self) -> None:
            self.show_calls += 1

        def _focus_for_hotkey(self) -> None:
            self.present_calls += 1

    window = DummyWindow()

    main_window_module.MainWindow._queue_hotkey_presentation(window)
    assert window._hotkey_pending is True
    assert window.show_calls == 1

    main_window_module.MainWindow._maybe_present_for_hotkey(
        window,
        ConversationMessage(role="assistant", content="reply"),
    )
    assert window.present_calls == 0
    assert window._hotkey_pending is True
    assert window.show_calls == 1

    main_window_module.MainWindow._maybe_present_for_hotkey(
        window,
        ConversationMessage(role="user", content="captured text"),
    )

    assert window.present_calls == 1
    assert window._hotkey_pending is False
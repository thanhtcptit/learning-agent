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
            self.visible = True
            self.show_calls = 0
            self.show_normal_calls = 0
            self.raise_calls = 0
            self.activate_calls = 0
            self.present_calls = 0

        def isVisible(self) -> bool:
            return self.visible

        def _show_for_hotkey(self) -> None:
            self.show_calls += 1
            self.visible = True

        def _focus_for_hotkey(self) -> None:
            if not self.visible:
                self._show_for_hotkey()
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


def test_hotkey_window_waits_to_restore_until_after_capture_when_hidden(monkeypatch) -> None:
    class DummyWindow:
        def __init__(self) -> None:
            self._hotkey_pending = False
            self.visible = False
            self.show_calls = 0
            self.present_calls = 0

        def isVisible(self) -> bool:
            return self.visible

        def _show_for_hotkey(self) -> None:
            self.show_calls += 1
            self.visible = True

        def _focus_for_hotkey(self) -> None:
            if not self.visible:
                self._show_for_hotkey()
            self.present_calls += 1

    window = DummyWindow()

    main_window_module.MainWindow._queue_hotkey_presentation(window)
    assert window._hotkey_pending is True
    assert window.show_calls == 0

    main_window_module.MainWindow._maybe_present_for_hotkey(
        window,
        ConversationMessage(role="user", content="captured text"),
    )

    assert window.show_calls == 1
    assert window.present_calls == 1
    assert window._hotkey_pending is False


def test_main_window_close_event_hides_to_tray_when_available() -> None:
    class DummyEvent:
        def __init__(self) -> None:
            self.ignored = False
            self.accepted = False

        def ignore(self) -> None:
            self.ignored = True

        def accept(self) -> None:
            self.accepted = True

    class DummyWindow:
        def __init__(self) -> None:
            self._tray_icon = object()
            self._allow_close = False
            self.hide_calls = 0

        def hide(self) -> None:
            self.hide_calls += 1

        def request_minimize_to_tray(self) -> None:
            self.hide()

    window = DummyWindow()
    event = DummyEvent()

    main_window_module.MainWindow.closeEvent(window, event)

    assert event.ignored is True
    assert event.accepted is False
    assert window.hide_calls == 1


def test_main_window_close_event_allows_exit_without_tray() -> None:
    class DummyEvent:
        def __init__(self) -> None:
            self.ignored = False
            self.accepted = False

        def ignore(self) -> None:
            self.ignored = True

        def accept(self) -> None:
            self.accepted = True

    class DummyWindow:
        def __init__(self) -> None:
            self._tray_icon = None
            self._allow_close = False
            self.hide_calls = 0

        def hide(self) -> None:
            self.hide_calls += 1

    window = DummyWindow()
    event = DummyEvent()

    main_window_module.MainWindow.closeEvent(window, event)

    assert event.ignored is False
    assert event.accepted is True
    assert window.hide_calls == 0


def test_main_window_restores_from_tray() -> None:
    class DummyWindow:
        def __init__(self) -> None:
            self.minimized = True
            self.visible = False
            self.show_calls = 0
            self.show_normal_calls = 0
            self.raise_calls = 0
            self.activate_calls = 0

        def isMinimized(self) -> bool:
            return self.minimized

        def isVisible(self) -> bool:
            return self.visible

        def showNormal(self) -> None:
            self.show_normal_calls += 1
            self.minimized = False
            self.visible = True

        def show(self) -> None:
            self.show_calls += 1
            self.visible = True

        def raise_(self) -> None:
            self.raise_calls += 1

        def activateWindow(self) -> None:
            self.activate_calls += 1

    window = DummyWindow()

    main_window_module.MainWindow._restore_from_tray(window)

    assert window.show_normal_calls == 1
    assert window.show_calls == 0
    assert window.raise_calls == 1
    assert window.activate_calls == 1


def test_main_window_request_minimize_to_tray_hides_window_when_tray_exists() -> None:
    class DummyWindow:
        def __init__(self) -> None:
            self._tray_icon = object()
            self.hide_calls = 0
            self.show_minimized_calls = 0

        def hide(self) -> None:
            self.hide_calls += 1

        def showMinimized(self) -> None:
            self.show_minimized_calls += 1

    window = DummyWindow()

    main_window_module.MainWindow.request_minimize_to_tray(window)

    assert window.hide_calls == 1
    assert window.show_minimized_calls == 0


def test_main_window_request_minimize_to_tray_falls_back_to_taskbar_minimize() -> None:
    class DummyWindow:
        def __init__(self) -> None:
            self._tray_icon = None
            self.hide_calls = 0
            self.show_minimized_calls = 0

        def hide(self) -> None:
            self.hide_calls += 1

        def showMinimized(self) -> None:
            self.show_minimized_calls += 1

    window = DummyWindow()

    main_window_module.MainWindow.request_minimize_to_tray(window)

    assert window.hide_calls == 0
    assert window.show_minimized_calls == 1
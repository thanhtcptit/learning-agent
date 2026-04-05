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
    class DummyTranscript:
        def __init__(self) -> None:
            self.scroll_calls = 0

        def scroll_to_bottom(self) -> None:
            self.scroll_calls += 1

    class DummyWindow:
        def __init__(self) -> None:
            self.minimized = True
            self.visible = False
            self.show_calls = 0
            self.show_normal_calls = 0
            self.raise_calls = 0
            self.activate_calls = 0
            self.transcript = DummyTranscript()

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

        def _scroll_transcript_to_bottom(self) -> None:
            self.transcript.scroll_to_bottom()

    window = DummyWindow()

    main_window_module.MainWindow._restore_from_tray(window)

    assert window.show_normal_calls == 1
    assert window.show_calls == 0
    assert window.raise_calls == 1
    assert window.activate_calls == 1
    assert window.transcript.scroll_calls == 1


def test_main_window_scrolls_transcript_to_bottom_when_hotkey_popup_occurs(monkeypatch) -> None:
    class DummyTranscript:
        def __init__(self) -> None:
            self.scroll_calls = 0

        def scroll_to_bottom(self) -> None:
            self.scroll_calls += 1

    class DummyWindow:
        def __init__(self) -> None:
            self.minimized = False
            self.visible = True
            self.show_calls = 0
            self.raise_calls = 0
            self.move_calls = 0
            self.transcript = DummyTranscript()

        def isMinimized(self) -> bool:
            return self.minimized

        def isVisible(self) -> bool:
            return self.visible

        def size(self) -> QSize:
            return QSize(480, 652)

        def showNormal(self) -> None:
            self.show_calls += 1
            self.minimized = False
            self.visible = True

        def show(self) -> None:
            self.show_calls += 1
            self.visible = True

        def move(self, _position) -> None:
            self.move_calls += 1

        def raise_(self) -> None:
            self.raise_calls += 1

        def _scroll_transcript_to_bottom(self) -> None:
            self.transcript.scroll_to_bottom()

    window = DummyWindow()

    monkeypatch.setattr(main_window_module.QCursor, "pos", staticmethod(lambda: QPoint(100, 100)))
    monkeypatch.setattr(main_window_module.QGuiApplication, "screenAt", staticmethod(lambda _point: None))
    monkeypatch.setattr(main_window_module.QGuiApplication, "primaryScreen", staticmethod(lambda: None))

    main_window_module.MainWindow._show_for_hotkey(window)

    assert window.show_calls == 0
    assert window.raise_calls == 1
    assert window.move_calls == 0
    assert window.transcript.scroll_calls == 1


def test_main_window_repositions_only_when_minimized(monkeypatch) -> None:
    class DummyTranscript:
        def __init__(self) -> None:
            self.scroll_calls = 0

        def scroll_to_bottom(self) -> None:
            self.scroll_calls += 1

    class DummyWindow:
        def __init__(self) -> None:
            self.minimized = True
            self.visible = False
            self.show_calls = 0
            self.show_normal_calls = 0
            self.raise_calls = 0
            self.move_positions: list[object] = []
            self.transcript = DummyTranscript()

        def isMinimized(self) -> bool:
            return self.minimized

        def isVisible(self) -> bool:
            return self.visible

        def size(self) -> QSize:
            return QSize(480, 652)

        def showNormal(self) -> None:
            self.show_normal_calls += 1
            self.minimized = False
            self.visible = True

        def show(self) -> None:
            self.show_calls += 1
            self.visible = True

        def move(self, position) -> None:
            self.move_positions.append(position)

        def raise_(self) -> None:
            self.raise_calls += 1

        def _scroll_transcript_to_bottom(self) -> None:
            self.transcript.scroll_to_bottom()

    class DummyScreen:
        def availableGeometry(self) -> QRect:
            return QRect(0, 0, 1200, 800)

    window = DummyWindow()

    monkeypatch.setattr(main_window_module.QCursor, "pos", staticmethod(lambda: QPoint(100, 100)))
    monkeypatch.setattr(main_window_module.QGuiApplication, "screenAt", staticmethod(lambda _point: DummyScreen()))
    monkeypatch.setattr(main_window_module.QGuiApplication, "primaryScreen", staticmethod(lambda: DummyScreen()))

    main_window_module.MainWindow._show_for_hotkey(window)

    assert window.show_normal_calls == 1
    assert window.show_calls == 0
    assert window.raise_calls == 1
    assert len(window.move_positions) == 1
    assert window.transcript.scroll_calls == 1


def test_main_window_repositions_when_restoring_from_tray(monkeypatch) -> None:
    class DummyTranscript:
        def __init__(self) -> None:
            self.scroll_calls = 0

        def scroll_to_bottom(self) -> None:
            self.scroll_calls += 1

    class DummyWindow:
        def __init__(self) -> None:
            self._tray_hidden = True
            self.minimized = False
            self.visible = False
            self.show_calls = 0
            self.show_normal_calls = 0
            self.raise_calls = 0
            self.move_positions: list[object] = []
            self.transcript = DummyTranscript()

        def isMinimized(self) -> bool:
            return self.minimized

        def isVisible(self) -> bool:
            return self.visible

        def size(self) -> QSize:
            return QSize(480, 652)

        def showNormal(self) -> None:
            self.show_normal_calls += 1
            self.minimized = False
            self.visible = True

        def show(self) -> None:
            self.show_calls += 1
            self.visible = True

        def move(self, position) -> None:
            self.move_positions.append(position)

        def raise_(self) -> None:
            self.raise_calls += 1

        def _scroll_transcript_to_bottom(self) -> None:
            self.transcript.scroll_to_bottom()

    class DummyScreen:
        def availableGeometry(self) -> QRect:
            return QRect(0, 0, 1200, 800)

    window = DummyWindow()

    monkeypatch.setattr(main_window_module.QCursor, "pos", staticmethod(lambda: QPoint(100, 100)))
    monkeypatch.setattr(main_window_module.QGuiApplication, "screenAt", staticmethod(lambda _point: DummyScreen()))
    monkeypatch.setattr(main_window_module.QGuiApplication, "primaryScreen", staticmethod(lambda: DummyScreen()))

    main_window_module.MainWindow._show_for_hotkey(window)

    assert window.show_normal_calls == 0
    assert window.show_calls == 1
    assert window.raise_calls == 1
    assert len(window.move_positions) == 1
    assert window.transcript.scroll_calls == 1


def test_main_window_marks_next_mode_for_new_session_when_minimized() -> None:
    class DummyWindow:
        def __init__(self) -> None:
            self._tray_icon = object()
            self.hide_calls = 0

        def hide(self) -> None:
            self.hide_calls += 1

    window = DummyWindow()

    main_window_module.MainWindow.request_minimize_to_tray(window)

    assert main_window_module.MainWindow.consume_new_session_request(window) is True
    assert main_window_module.MainWindow.consume_new_session_request(window) is False
    assert window.hide_calls == 1


def test_main_window_updates_language_indicator() -> None:
    class DummyLabel:
        def __init__(self) -> None:
            self.text = ""

        def setText(self, value: str) -> None:
            self.text = value

    class DummyWindow:
        def __init__(self) -> None:
            self.language_indicator = DummyLabel()

    window = DummyWindow()

    main_window_module.MainWindow._set_current_language(window, "English")

    assert window.language_indicator.text == "Language: English"


def test_main_window_defers_initial_scroll_until_first_show(monkeypatch) -> None:
    class DummyTranscript:
        def __init__(self) -> None:
            self.scroll_calls = 0

        def scroll_to_bottom(self) -> None:
            self.scroll_calls += 1

    class DummyWindow:
        def __init__(self) -> None:
            self._initial_show_scroll_pending = True
            self.transcript = DummyTranscript()

        def _scroll_transcript_to_bottom(self) -> None:
            self.transcript.scroll_to_bottom()

    scheduled_calls: list[tuple[int, object]] = []
    monkeypatch.setattr(
        main_window_module.QTimer,
        "singleShot",
        staticmethod(lambda delay, callback: scheduled_calls.append((delay, callback))),
    )

    window = DummyWindow()

    main_window_module.MainWindow._scroll_latest_messages_after_initial_show(window)

    assert window._initial_show_scroll_pending is False
    assert len(scheduled_calls) == 1
    assert scheduled_calls[0][0] == 0
    assert window.transcript.scroll_calls == 0

    scheduled_calls[0][1]()

    assert window.transcript.scroll_calls == 1


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
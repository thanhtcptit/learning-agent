from __future__ import annotations

from typing import Callable, cast

from PySide6.QtCore import QEvent, QPoint, QPointF, QRect, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QCloseEvent, QCursor, QGuiApplication, QIcon, QKeySequence, QPainter, QPen, QPolygonF, QScreen, QShortcut, QShowEvent
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QPushButton,
    QStyle,
    QSystemTrayIcon,
    QToolButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.hotkey import GlobalHotkeyListener, VOICE_HOTKEY_ACTION
from core.orchestrator import AppController
from core.runtime_paths import get_bundle_data_root
from session.manager import ConversationMessage, ConversationSession
from prompts.templates import PromptMode
from ui.chat_widget import ChatTranscript
from ui.input_box import MessageInput
from ui.settings_popup import SettingsPopup


def _clamp(value: int, minimum: int, maximum: int) -> int:
    if maximum < minimum:
        return minimum
    return max(minimum, min(value, maximum))


def _call_optional_method(target: object, method_name: str) -> None:
    method = getattr(target, method_name, None)
    if callable(method):
        method()


def _create_mascot_icon() -> QIcon:
    icon_path = get_bundle_data_root() / "assets" / "icon.jpg"
    if not icon_path.exists():
        raise FileNotFoundError(f"Robot mascot icon asset is missing: {icon_path}")

    icon = QIcon(str(icon_path))
    if icon.isNull():
        raise RuntimeError(f"Robot mascot icon asset could not be loaded: {icon_path}")

    return icon


def _calculate_hotkey_window_position(available_geometry: QRect, cursor_position: QPoint, window_size: QSize, margin: int = 16) -> QPoint:
    screen_center_x = available_geometry.x() + available_geometry.width() // 2
    if cursor_position.x() < screen_center_x:
        target_x = available_geometry.x() + available_geometry.width() - window_size.width() - margin
    else:
        target_x = available_geometry.x() + margin

    target_y = cursor_position.y() - window_size.height() // 2

    min_x = available_geometry.x() + margin
    max_x = available_geometry.x() + available_geometry.width() - window_size.width() - margin
    min_y = available_geometry.y() + margin
    max_y = available_geometry.y() + available_geometry.height() - window_size.height() - margin

    return QPoint(_clamp(target_x, min_x, max_x), _clamp(target_y, min_y, max_y))


def _constrain_floating_helper_position(available_geometry: QRect, helper_size: QSize, position: QPoint, margin: int = 16) -> QPoint:
    min_x = available_geometry.x() + margin
    max_x = available_geometry.x() + available_geometry.width() - helper_size.width() - margin
    min_y = available_geometry.y() + margin
    max_y = available_geometry.y() + available_geometry.height() - helper_size.height() - margin

    return QPoint(_clamp(position.x(), min_x, max_x), _clamp(position.y(), min_y, max_y))


def _calculate_floating_helper_position(available_geometry: QRect, helper_size: QSize, margin: int = 16) -> QPoint:
    target_x = available_geometry.x() + available_geometry.width() - helper_size.width() - margin
    target_y = available_geometry.y() + available_geometry.height() - helper_size.height() - margin

    return _constrain_floating_helper_position(available_geometry, helper_size, QPoint(target_x, target_y), margin)


_STATUS_COLORS: dict[str, tuple[str, str, str]] = {
    "Thinking": ("#7c3aed", "#a78bfa", "#f5f3ff"),
    "Listening": ("#2563eb", "#60a5fa", "#eff6ff"),
    "Speaking": ("#059669", "#34d399", "#ecfdf5"),
}

_LANGUAGE_STATUS_PREFIXES = ("language switched to ", "language set to ")
_WAKE_WORD_STATUS_PREFIXES = ("wake word enabled", "wake word disabled")
_MODE_STATUS_DISPLAY_TEXT: dict[str, str] = {
    PromptMode.DEFINITION.label.lower(): "Defining",
    PromptMode.EXPLAIN.label.lower(): "Explaining",
    PromptMode.SUMMARY.label.lower(): "Summarizing",
}


def _extract_language_status_text(text: str) -> str | None:
    stripped = text.strip()
    lowered = stripped.lower()
    for prefix in _LANGUAGE_STATUS_PREFIXES:
        if lowered.startswith(prefix):
            language = stripped[len(prefix):].strip()
            return language or None

    return None


def _extract_wake_word_status_text(text: str) -> str | None:
    lowered = text.strip().lower()
    for prefix in _WAKE_WORD_STATUS_PREFIXES:
        if lowered.startswith(prefix):
            if prefix.endswith("enabled"):
                return "Wake word enabled"
            return "Wake word disabled"

    return None


def _extract_mode_status_text(text: str) -> str | None:
    lowered = text.strip().lower()
    for prefix in ("capturing selection for ", "scanning screen for ocr context for "):
        if not lowered.startswith(prefix):
            continue

        mode_label = lowered[len(prefix):].strip()
        return _MODE_STATUS_DISPLAY_TEXT.get(mode_label)

    return None


def _extract_temporary_floating_status_text(text: str) -> str | None:
    language_text = _extract_language_status_text(text)
    if language_text is not None:
        return language_text

    wake_word_text = _extract_wake_word_status_text(text)
    if wake_word_text is not None:
        return wake_word_text

    return _extract_mode_status_text(text)


def _normalize_floating_status_text(text: str) -> str | None:
    stripped = text.strip()
    lowered = stripped.lower()
    if "think" in lowered:
        return "Thinking"

    if "listen" in lowered:
        return "Listening"

    if "speak" in lowered:
        return "Speaking"

    temporary_text = _extract_temporary_floating_status_text(text)
    if temporary_text is not None:
        return temporary_text

    return None


class StatusPill(QFrame):
    _PILL_HEIGHT = 24
    _NOTCH_SIZE = 5
    _H_PADDING = 10

    def __init__(self, text: str = "Ready", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("StatusPill")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self._dot = QLabel("\u2022", self)
        self._dot.setObjectName("StatusPillDot")
        self._dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._dot.setFixedSize(7, self._PILL_HEIGHT)

        self._text_label = QLabel(text, self)
        self._text_label.setObjectName("StatusPillLabel")
        self._text_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)

        row = QHBoxLayout(self)
        row.setContentsMargins(self._H_PADDING, 0, self._H_PADDING, self._NOTCH_SIZE)
        row.setSpacing(4)
        row.addWidget(self._dot)
        row.addWidget(self._text_label)
        self.setFixedHeight(self._PILL_HEIGHT + self._NOTCH_SIZE)

        self._status_key: str | None = None
        self.setText(text)

    def setText(self, text: str) -> None:
        display_text = _normalize_floating_status_text(text)
        if display_text is None:
            self._text_label.clear()
            self._status_key = None
            self.setToolTip("")
            self.hide()
        else:
            self._status_key = display_text
            self._text_label.setText(display_text)
            self.setToolTip(display_text)
            self._apply_colors()
            self.show()

        self.adjustSize()
        self.update()

    def text(self) -> str:
        return self._text_label.text()

    def _apply_colors(self) -> None:
        if self._status_key is None:
            return
        primary, _, _ = _STATUS_COLORS.get(self._status_key, ("#64748b", "#94a3b8", "#f8fafc"))
        self.setStyleSheet(
            f"""
            QLabel#StatusPillDot {{
                background: transparent;
                color: {primary};
                font-size: 16px;
                font-weight: 900;
            }}
            QLabel#StatusPillLabel {{
                background: transparent;
                color: #ffffff;
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 0.5px;
            }}
            """
        )

    def paintEvent(self, event) -> None:
        del event
        if self._status_key is None:
            return

        primary_hex, _, _ = _STATUS_COLORS.get(self._status_key, ("#64748b", "#94a3b8", "#f8fafc"))

        w = max(self.width(), 1)
        pill_h = self._PILL_HEIGHT
        radius = pill_h / 2.0

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        shadow_color = QColor(0, 0, 0, 30)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(shadow_color)
        painter.drawRoundedRect(QRectF(1, 2, w - 2, pill_h), radius, radius)

        pill_color = QColor(primary_hex)
        painter.setBrush(pill_color)
        painter.drawRoundedRect(QRectF(0, 0, w, pill_h), radius, radius)

        notch_cx = w / 2.0
        notch = QPolygonF(
            [
                QPointF(notch_cx - self._NOTCH_SIZE, pill_h - 1),
                QPointF(notch_cx + self._NOTCH_SIZE, pill_h - 1),
                QPointF(notch_cx, pill_h + self._NOTCH_SIZE - 1),
            ]
        )
        painter.drawPolygon(notch)


class FloatingStatusWidget(QWidget):
    TEMPORARY_BUBBLE_AUTO_HIDE_MS = 3000

    def __init__(
        self,
        restore_callback: Callable[[], None],
        moved_callback: Callable[[], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("FloatingHelperWindow")
        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.status_bubble = StatusPill("Ready", self)
        self._status_bubble_hide_token = 0
        self.icon_button = FloatingHelperButton(restore_callback, moved_callback, self)
        self.icon_button.setObjectName("FloatingHelperButton")
        self.icon_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.icon_button.setAutoRaise(True)
        self.icon_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.icon_button.setFixedSize(60, 60)
        self.icon_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.icon_button.setIcon(_create_mascot_icon())
        self.icon_button.setIconSize(QSize(30, 30))
        self.icon_button.setToolTip("Restore AI Learning Assistant")
        self.icon_button.setStyleSheet(
            """
            QToolButton#FloatingHelperButton {
                background: #ffffff;
                border: 1px solid #dbe3ee;
                border-radius: 30px;
                padding: 0px;
            }
            QToolButton#FloatingHelperButton:hover {
                background: #eff6ff;
                border-color: #93c5fd;
            }
            QToolButton#FloatingHelperButton:pressed {
                background: #dbeafe;
                border-color: #60a5fa;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.status_bubble, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.icon_button, 0, Qt.AlignmentFlag.AlignHCenter)

        self._reserve_size_for_status_pill()
        self.adjustSize()

    def _reserve_size_for_status_pill(self) -> None:
        reserved_width = self.icon_button.width()
        reserved_height = self.icon_button.height()
        reserved_pill_width = 0
        reserved_pill_height = StatusPill._PILL_HEIGHT + StatusPill._NOTCH_SIZE

        for sample_text in ("Thinking", "Listening", "Speaking", "Wake word disabled", "Defining", "Explaining", "Summarizing"):
            self.status_bubble.setText(sample_text)
            reserved_pill_width = max(reserved_pill_width, self.status_bubble.sizeHint().width())

        self.status_bubble.setText("Ready")
        layout = self.layout()
        layout_spacing = layout.spacing() if layout is not None else 0

        reserved_width = max(reserved_width, reserved_pill_width)
        reserved_height = reserved_height + layout_spacing + reserved_pill_height

        self.setMinimumSize(reserved_width, reserved_height)
        self.resize(reserved_width, reserved_height)

    def set_status_text(self, text: str) -> None:
        self.status_bubble.setText(text)
        self._status_bubble_hide_token += 1
        hide_token = self._status_bubble_hide_token
        if _extract_temporary_floating_status_text(text) is not None:
            QTimer.singleShot(
                self.TEMPORARY_BUBBLE_AUTO_HIDE_MS,
                lambda: self._hide_status_bubble_if_current(hide_token),
            )
        self.adjustSize()

    def _hide_status_bubble_if_current(self, hide_token: int) -> None:
        if hide_token != self._status_bubble_hide_token:
            return

        self.status_bubble.setText("Ready")
        self.adjustSize()

    def status_text(self) -> str:
        return self.status_bubble.text()


class FloatingHelperButton(QToolButton):
    def __init__(
        self,
        restore_callback: Callable[[], None],
        moved_callback: Callable[[], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._restore_callback = restore_callback
        self._moved_callback = moved_callback
        self._drag_start_global_position: QPoint | None = None
        self._drag_start_window_position: QPoint | None = None
        self._dragging = False

    def _drag_target_widget(self) -> QWidget:
        target = self.window()
        if isinstance(target, QWidget):
            return target

        return self

    def begin_drag(self, global_position: QPoint) -> None:
        drag_target = self._drag_target_widget()
        self._drag_start_global_position = global_position
        self._drag_start_window_position = drag_target.pos()
        self._dragging = False

    def update_drag(self, global_position: QPoint) -> None:
        if self._drag_start_global_position is None or self._drag_start_window_position is None:
            return

        delta = global_position - self._drag_start_global_position
        if not self._dragging and delta.manhattanLength() < QApplication.startDragDistance():
            return

        if not self._dragging:
            self._dragging = True
            self._moved_callback()

        desired_position = self._drag_start_window_position + delta
        self._drag_target_widget().move(self._constrain_to_screen(desired_position))

    def end_drag(self) -> None:
        self._drag_start_global_position = None
        self._drag_start_window_position = None
        self._dragging = False

    def _constrain_to_screen(self, desired_position: QPoint) -> QPoint:
        drag_target = self._drag_target_widget()
        screen = cast(QScreen | None, drag_target.screen()) or QGuiApplication.screenAt(desired_position) or QGuiApplication.primaryScreen()
        if screen is None:
            return desired_position

        return _constrain_floating_helper_position(screen.availableGeometry(), drag_target.size(), desired_position)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.begin_drag(event.globalPosition().toPoint())

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.update_drag(event.globalPosition().toPoint())

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mouseReleaseEvent(event)
            return

        was_dragging = self._dragging
        self.end_drag()
        if was_dragging:
            event.accept()
            return

        super().mouseReleaseEvent(event)
        self._restore_callback()


class MainWindow(QMainWindow):
    HOTKEY_PRESENTATION_DELAY_MS = 250

    def __init__(self, controller: AppController, hotkey_listener: GlobalHotkeyListener) -> None:
        super().__init__()
        self._controller = controller
        self._hotkey_listener = hotkey_listener
        self._hotkey_pending = False
        self._allow_close = False
        self._tray_icon: QSystemTrayIcon | None = None
        self._floating_helper: QWidget | None = None
        self._floating_helper_needs_default_position = True
        self._escape_shortcut: QShortcut | None = None
        self._initial_show_scroll_pending = True
        self._tray_hidden = False
        self._start_new_session_on_next_mode = False
        self._build_ui()
        self._create_tray_icon()
        self._create_floating_helper()
        self._connect_signals()
        self._apply_initial_state()

    def _build_ui(self) -> None:
        self.setWindowTitle("AI Learning Assistant")
        self.resize(480, 652)
        self.setMinimumSize(420, 620)

        self.setObjectName("RootSurface")
        self.transcript = ChatTranscript()
        self.input_box = MessageInput()
        self.send_button = QPushButton("Send")
        self.send_button.setObjectName("PrimaryButton")

        self.settings_button = QToolButton()
        self.settings_button.setObjectName("SettingsButton")
        self.settings_button.setText("⚙")
        self.settings_button.setToolTip("Settings")
        self.settings_button.setAutoRaise(True)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("StopButton")
        self.stop_button.setToolTip("Stop the current request")
        self.stop_button.setFixedWidth(72)
        self.stop_button.setEnabled(False)

        self.settings_popup = SettingsPopup(self._controller, self._hotkey_listener, self)

        header_frame = QFrame()
        header_frame.setObjectName("HeaderBar")
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(14, 10, 14, 10)
        header_layout.setSpacing(8)

        self.language_indicator = QLabel()
        self.language_indicator.setObjectName("LanguageBadge")

        self.status_badge = QLabel("Ready")
        self.status_badge.setObjectName("StatusBadge")

        header_layout.addWidget(self.settings_button)
        header_layout.addWidget(self.language_indicator)
        header_layout.addWidget(self.status_badge)
        header_layout.addStretch(1)

        content_frame = QFrame()
        content_frame.setObjectName("ContentPanel")
        content_layout = QVBoxLayout(content_frame)
        content_layout.setContentsMargins(14, 14, 14, 14)
        content_layout.setSpacing(12)
        content_layout.addWidget(self.transcript, 1)

        composer_layout = QHBoxLayout()
        composer_layout.setSpacing(10)
        composer_layout.addWidget(self.input_box, 1)
        composer_layout.addWidget(self.send_button)
        composer_layout.addWidget(self.stop_button)
        content_layout.addLayout(composer_layout)

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(12)
        root_layout.addWidget(header_frame)
        root_layout.addWidget(content_frame, 1)

        self.setCentralWidget(root)
        self.setStyleSheet(
            """
            QMainWindow {
                background: #f5f7fb;
                color: #0f172a;
            }
            QWidget#RootSurface {
                background: #f5f7fb;
            }
            QFrame#HeaderBar {
                background: #ffffff;
                border: 1px solid #dbe3ee;
                border-radius: 14px;
            }
            QLabel#AppTitle {
                color: #0f172a;
                font-size: 16px;
                font-weight: 700;
            }
            QLabel#LanguageBadge {
                background: #eff6ff;
                color: #1d4ed8;
                border: 1px solid #bfdbfe;
                border-radius: 999px;
                padding: 4px 10px;
                font-size: 12px;
                font-weight: 700;
            }
            QLabel#StatusBadge {
                background: #f8fafc;
                color: #475569;
                border: 1px solid #dbe3ee;
                border-radius: 999px;
                padding: 4px 10px;
                font-size: 12px;
                font-weight: 700;
            }
            QFrame#ContentPanel {
                background: #ffffff;
                border: 1px solid #dbe3ee;
                border-radius: 18px;
            }
            QPushButton {
                border: none;
                border-radius: 10px;
                padding: 10px 14px;
                font-weight: 600;
            }
            QPushButton#PrimaryButton {
                background: #2563eb;
                color: white;
            }
            QPushButton#PrimaryButton:disabled {
                background: #93c5fd;
                color: #eff6ff;
            }
            QPushButton#StopButton {
                background: #dc2626;
                color: white;
            }
            QPushButton#StopButton:disabled {
                background: #fca5a5;
                color: #fff5f5;
            }
            QToolButton#SettingsButton {
                background: #f1f5f9;
                color: #0f172a;
                border: 1px solid #dbe3ee;
                border-radius: 10px;
                padding: 4px 10px;
                font-size: 18px;
            }
            QLineEdit, QComboBox {
                background: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 10px;
                padding: 8px 10px;
                color: #0f172a;
                min-height: 20px;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 1px solid #60a5fa;
            }
            QComboBox::drop-down {
                border: none;
                width: 24px;
            }
            QComboBox QAbstractItemView {
                background: #ffffff;
                border: 1px solid #dbe3ee;
                selection-background-color: #dbeafe;
                selection-color: #0f172a;
                outline: none;
            }
            """
        )

    def _connect_signals(self) -> None:
        self._controller.current_session_changed.connect(self._load_session)
        self._controller.message_upserted.connect(self.transcript.upsert_message)
        self._controller.message_upserted.connect(self._maybe_present_for_hotkey)
        self._controller.busy_changed.connect(self._set_busy)
        self._controller.current_language_changed.connect(self._set_current_language)
        self._controller.status_changed.connect(self._set_status)
        self._hotkey_listener.hotkey_triggered.connect(self._queue_hotkey_presentation)

        self._escape_shortcut = QShortcut(QKeySequence("Esc"), self)
        self._escape_shortcut.activated.connect(self.request_minimize_to_tray)

        self.settings_button.clicked.connect(self._toggle_settings_popup)
        self.stop_button.clicked.connect(self._request_stop_current_request)
        self.send_button.clicked.connect(self._on_send_clicked)
        self.input_box.submitted.connect(self._on_send_text)

    def _create_tray_icon(self) -> None:
        tray_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
        self.setWindowIcon(tray_icon)

        if not QSystemTrayIcon.isSystemTrayAvailable():
            return

        if QApplication.instance() is None:
            return

        system_tray = QSystemTrayIcon(tray_icon, self)
        system_tray.setToolTip("AI Learning Assistant")

        tray_menu = QMenu(self)
        show_action = tray_menu.addAction("Show")
        show_action.triggered.connect(self._restore_from_tray)

        quit_action = tray_menu.addAction("Exit")
        quit_action.triggered.connect(self.request_exit)

        system_tray.setContextMenu(tray_menu)
        system_tray.activated.connect(self._on_tray_activated)
        system_tray.show()

        self._tray_icon = system_tray

    def _create_floating_helper(self) -> None:
        helper = FloatingStatusWidget(self._restore_from_tray, self._mark_floating_helper_moved)
        helper.hide()
        self._floating_helper = helper

    def _mark_floating_helper_moved(self) -> None:
        self._floating_helper_needs_default_position = False

    def _position_floating_helper(self) -> None:
        helper = getattr(self, "_floating_helper", None)
        if helper is None:
            return

        helper_adjust = getattr(helper, "adjustSize", None)
        if callable(helper_adjust):
            helper_adjust()

        screen = cast(QScreen | None, helper.screen())
        if screen is None:
            screen = QGuiApplication.primaryScreen()

        if screen is None:
            return

        helper.move(_calculate_floating_helper_position(screen.availableGeometry(), helper.size()))
        self._floating_helper_needs_default_position = False

    def _show_floating_helper(self) -> None:
        helper = getattr(self, "_floating_helper", None)
        if helper is None:
            return

        if self._floating_helper_needs_default_position:
            self._position_floating_helper()

        helper.show()
        helper.raise_()

    def _update_floating_helper_status(self, text: str) -> None:
        helper = getattr(self, "_floating_helper", None)
        if helper is None:
            return

        set_status_text = getattr(helper, "set_status_text", None)
        if callable(set_status_text):
            set_status_text(text)

        helper_is_visible = getattr(helper, "isVisible", None)
        if (
            self._floating_helper_needs_default_position
            and callable(helper_is_visible)
            and helper_is_visible()
        ):
            self._position_floating_helper()

    def _hide_floating_helper(self) -> None:
        helper = getattr(self, "_floating_helper", None)
        if helper is None:
            return

        helper.hide()

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self._restore_from_tray()

    def _restore_from_tray(self) -> None:
        _call_optional_method(self, "_hide_floating_helper")
        tray_hidden = getattr(self, "_tray_hidden", False)
        if self.isMinimized() or tray_hidden:
            cursor_position = QCursor.pos()
            screen = QGuiApplication.screenAt(cursor_position) or QGuiApplication.primaryScreen()
            if screen is not None:
                target_position = _calculate_hotkey_window_position(screen.availableGeometry(), cursor_position, self.size())
                self.move(target_position)

        if self.isMinimized():
            self.showNormal()
        elif not self.isVisible():
            self.show()

        self.raise_()
        self.activateWindow()
        self._scroll_transcript_to_bottom()
        self._tray_hidden = False

    def request_minimize_to_tray(self) -> None:
        self._start_new_session_on_next_mode = True
        if self._tray_icon is not None:
            self._tray_hidden = True
            self.hide()
            _call_optional_method(self, "_show_floating_helper")
            return

        self.showMinimized()
        _call_optional_method(self, "_show_floating_helper")

    def toggle_window_visibility(self) -> None:
        if self.isMinimized() or self._tray_hidden or not self.isVisible():
            self._restore_from_tray()
            return

        self.request_minimize_to_tray()

    def request_exit(self) -> None:
        _call_optional_method(self, "_hide_floating_helper")
        self._allow_close = True
        app = QApplication.instance()
        if app is not None:
            app.quit()
            return

        self.close()

    def _apply_initial_state(self) -> None:
        self._load_session(self._controller.current_session)
        self._set_current_language(self._controller.target_language)
        self._set_status("Ready")

    def start_minimized(self) -> None:
        if self._tray_icon is not None:
            self._tray_hidden = True
            self.hide()
            _call_optional_method(self, "_show_floating_helper")
            return

        self.showMinimized()
        _call_optional_method(self, "_show_floating_helper")

    def _scroll_latest_messages_after_initial_show(self) -> None:
        if not self._initial_show_scroll_pending:
            return

        self._initial_show_scroll_pending = False
        QTimer.singleShot(0, self._scroll_transcript_to_bottom)

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self._scroll_latest_messages_after_initial_show()

    def _toggle_settings_popup(self) -> None:
        if self.settings_popup.isVisible():
            self.settings_popup.close()
            return

        self.settings_popup.show_near(self.settings_button)

    def _queue_hotkey_presentation(self, mode: object | None = None) -> None:
        if mode == VOICE_HOTKEY_ACTION:
            if self.isMinimized() or getattr(self, "_tray_hidden", False):
                return

            delay_ms = getattr(self, "HOTKEY_PRESENTATION_DELAY_MS", MainWindow.HOTKEY_PRESENTATION_DELAY_MS)
            QTimer.singleShot(delay_ms, self._show_for_hotkey)
            return

        if not isinstance(mode, PromptMode) and mode is not None:
            return

        self._hotkey_pending = True
        if mode is PromptMode.DEFINITION and getattr(self._controller, "screen_ocr_enabled", False):
            return

        delay_ms = getattr(self, "HOTKEY_PRESENTATION_DELAY_MS", MainWindow.HOTKEY_PRESENTATION_DELAY_MS)
        QTimer.singleShot(delay_ms, self._show_for_hotkey)

    def present_prompt_hotkey(self) -> None:
        self._queue_hotkey_presentation()

    def _maybe_present_for_hotkey(self, message: object) -> None:
        if not self._hotkey_pending:
            return

        if not isinstance(message, ConversationMessage):
            return

        screen_ocr_enabled = getattr(self._controller, "screen_ocr_enabled", False)
        is_definition_mode = message.mode == PromptMode.DEFINITION.value
        if message.role == "user":
            if not message.content.strip():
                return

            if is_definition_mode and screen_ocr_enabled and not message.screen_context.strip():
                return

            self._hotkey_pending = False
            self._focus_for_hotkey()
            return

        if is_definition_mode and screen_ocr_enabled and message.role == "assistant":
            self._hotkey_pending = False
            self._focus_for_hotkey()
            return

    def _show_for_hotkey(self) -> None:
        _call_optional_method(self, "_hide_floating_helper")
        reposition = self.isMinimized() or getattr(self, "_tray_hidden", False)
        if reposition:
            cursor_position = QCursor.pos()
            screen = QGuiApplication.screenAt(cursor_position) or QGuiApplication.primaryScreen()
            if screen is not None:
                target_position = _calculate_hotkey_window_position(screen.availableGeometry(), cursor_position, self.size())
                self.move(target_position)

        if self.isMinimized():
            self.showNormal()
        elif not self.isVisible():
            self.show()

        self.raise_()
        self._scroll_transcript_to_bottom()
        self._tray_hidden = False

    def _focus_for_hotkey(self) -> None:
        if not self.isVisible():
            self._show_for_hotkey()

        self.activateWindow()
        self.input_box.setFocus(Qt.FocusReason.ShortcutFocusReason)

    def _load_session(self, session: ConversationSession) -> None:
        self.transcript.reset_messages(session.messages)

    def _scroll_transcript_to_bottom(self) -> None:
        self.transcript.scroll_to_bottom()

    def _set_status(self, text: str) -> None:
        self.status_badge.setText(text)
        self.status_badge.setToolTip(text)
        update_floating_helper_status = getattr(self, "_update_floating_helper_status", None)
        if callable(update_floating_helper_status):
            update_floating_helper_status(text)

    def consume_new_session_request(self) -> bool:
        pending = getattr(self, "_start_new_session_on_next_mode", False)
        self._start_new_session_on_next_mode = False
        return pending

    def _set_current_language(self, language: str) -> None:
        self.language_indicator.setText(language)
        self.language_indicator.setToolTip(f"Current language: {language}")

    def _on_send_clicked(self, _checked: bool = False) -> None:
        self._on_send_text(self.input_box.text())

    def _request_stop_current_request(self) -> None:
        stop_current_request = getattr(self._controller, "stop_current_request", None)
        if callable(stop_current_request):
            stop_current_request()

    def _on_send_text(self, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        submit_chat_text = getattr(self._controller, "submit_chat_text", None)
        if callable(submit_chat_text):
            submit_chat_text(cleaned)
        else:
            self._controller.submit_text(cleaned)
        self.input_box.clear()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._tray_icon is not None and not self._allow_close:
            event.ignore()
            self.request_minimize_to_tray()
            return

        _call_optional_method(self, "_hide_floating_helper")
        event.accept()

    def changeEvent(self, event: QEvent) -> None:
        super().changeEvent(event)
        if event.type() != QEvent.Type.WindowStateChange:
            return

        if self.isMinimized():
            self._start_new_session_on_next_mode = True
            _call_optional_method(self, "_show_floating_helper")
            return

        _call_optional_method(self, "_hide_floating_helper")

    def _set_busy(self, busy: bool) -> None:
        self.send_button.setEnabled(not busy)
        self.send_button.setText("Thinking..." if busy else "Send")
        self.stop_button.setEnabled(busy)
        self.input_box.setEnabled(not busy)

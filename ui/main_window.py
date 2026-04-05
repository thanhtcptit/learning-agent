from __future__ import annotations

from PySide6.QtCore import QEvent, QPoint, QRect, QSize, Qt, QTimer
from PySide6.QtGui import QCloseEvent, QCursor, QGuiApplication, QKeySequence, QShortcut, QShowEvent
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
    QVBoxLayout,
    QWidget,
)

from core.hotkey import GlobalHotkeyListener
from core.orchestrator import AppController
from session.manager import ConversationMessage, ConversationSession
from ui.chat_widget import ChatTranscript
from ui.input_box import MessageInput
from ui.settings_popup import SettingsPopup


def _clamp(value: int, minimum: int, maximum: int) -> int:
    if maximum < minimum:
        return minimum
    return max(minimum, min(value, maximum))


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


class MainWindow(QMainWindow):
    def __init__(self, controller: AppController, hotkey_listener: GlobalHotkeyListener) -> None:
        super().__init__()
        self._controller = controller
        self._hotkey_listener = hotkey_listener
        self._hotkey_pending = False
        self._allow_close = False
        self._tray_icon: QSystemTrayIcon | None = None
        self._escape_shortcut: QShortcut | None = None
        self._initial_show_scroll_pending = True
        self._tray_hidden = False
        self._start_new_session_on_next_mode = False
        self._build_ui()
        self._create_tray_icon()
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

        self.settings_popup = SettingsPopup(self._controller, self._hotkey_listener, self)

        header_frame = QFrame()
        header_frame.setObjectName("HeaderBar")
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(14, 10, 14, 10)
        header_layout.setSpacing(8)

        title_label = QLabel("AI Learning Assistant")
        title_label.setObjectName("AppTitle")

        self.language_indicator = QLabel()
        self.language_indicator.setObjectName("LanguageBadge")

        header_layout.addWidget(title_label)
        header_layout.addStretch(1)
        header_layout.addWidget(self.language_indicator)
        header_layout.addWidget(self.settings_button)

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
        self._hotkey_listener.hotkey_triggered.connect(self._queue_hotkey_presentation)

        self._escape_shortcut = QShortcut(QKeySequence("Esc"), self)
        self._escape_shortcut.activated.connect(self.request_minimize_to_tray)

        self.settings_button.clicked.connect(self._toggle_settings_popup)
        self.send_button.clicked.connect(self._on_send_clicked)
        self.input_box.submitted.connect(self._on_send_text)

    def _create_tray_icon(self) -> None:
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return

        if QApplication.instance() is None:
            return

        tray_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
        self.setWindowIcon(tray_icon)

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

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self._restore_from_tray()

    def _restore_from_tray(self) -> None:
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
            return

        self.showMinimized()

    def request_exit(self) -> None:
        self._allow_close = True
        app = QApplication.instance()
        if app is not None:
            app.quit()
            return

        self.close()

    def _apply_initial_state(self) -> None:
        self._load_session(self._controller.current_session)
        self._set_current_language(self._controller.target_language)

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

    def _queue_hotkey_presentation(self, _mode: object | None = None) -> None:
        self._hotkey_pending = True
        if self.isVisible():
            self._show_for_hotkey()

    def _maybe_present_for_hotkey(self, message: object) -> None:
        if not self._hotkey_pending:
            return

        if not isinstance(message, ConversationMessage):
            return

        if message.role != "user":
            return

        if not message.content.strip():
            return

        self._hotkey_pending = False
        self._focus_for_hotkey()

    def _show_for_hotkey(self) -> None:
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

    def consume_new_session_request(self) -> bool:
        pending = getattr(self, "_start_new_session_on_next_mode", False)
        self._start_new_session_on_next_mode = False
        return pending

    def _set_current_language(self, language: str) -> None:
        self.language_indicator.setText(f"Language: {language}")

    def _on_send_clicked(self, _checked: bool = False) -> None:
        self._on_send_text(self.input_box.text())

    def _on_send_text(self, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        self._controller.submit_text(cleaned)
        self.input_box.clear()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._tray_icon is not None and not self._allow_close:
            event.ignore()
            self.request_minimize_to_tray()
            return

        event.accept()

    def changeEvent(self, event: QEvent) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange and self.isMinimized():
            self._start_new_session_on_next_mode = True

    def _set_busy(self, busy: bool) -> None:
        self.send_button.setEnabled(not busy)
        self.send_button.setText("Thinking..." if busy else "Send")
        self.input_box.setEnabled(not busy)

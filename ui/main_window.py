from __future__ import annotations

from core.hotkey import GlobalHotkeyListener
from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtGui import QCursor, QGuiApplication
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.orchestrator import AppController
from session.manager import ConversationSession
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
        self._build_ui()
        self._connect_signals()
        self._apply_initial_state()

    def _build_ui(self) -> None:
        self.setWindowTitle("AI Learning Assistant")
        self.resize(480, 752)
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

        header_layout.addWidget(title_label)
        header_layout.addStretch(1)
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
        self._controller.busy_changed.connect(self._set_busy)
        self._hotkey_listener.hotkey_triggered.connect(self._present_for_hotkey)

        self.settings_button.clicked.connect(self._toggle_settings_popup)
        self.send_button.clicked.connect(self._on_send_clicked)
        self.input_box.submitted.connect(self._on_send_text)

    def _apply_initial_state(self) -> None:
        self._load_session(self._controller.current_session)

    def _toggle_settings_popup(self) -> None:
        if self.settings_popup.isVisible():
            self.settings_popup.close()
            return

        self.settings_popup.show_near(self.settings_button)

    def _present_for_hotkey(self, _mode: object | None = None) -> None:
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
        self.input_box.setFocus(Qt.FocusReason.ShortcutFocusReason)

    def _load_session(self, session: ConversationSession) -> None:
        self.transcript.reset_messages(session.messages)

    def _on_send_clicked(self, _checked: bool = False) -> None:
        self._on_send_text(self.input_box.text())

    def _on_send_text(self, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        self._controller.submit_text(cleaned)
        self.input_box.clear()

    def _set_busy(self, busy: bool) -> None:
        self.send_button.setEnabled(not busy)
        self.send_button.setText("Thinking..." if busy else "Send")
        self.input_box.setEnabled(not busy)

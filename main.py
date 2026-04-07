from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Slot
from PySide6.QtWidgets import QApplication, QMessageBox

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional during bootstrap
    def load_dotenv(*args, **kwargs) -> bool:
        return False

from core.app_settings import AppSettings, load_app_settings, save_app_settings
from core.config import DEFAULT_PROVIDER_CONFIG_PATH, build_provider, load_provider_config
from core.hotkey import (
    EXIT_HOTKEY_ACTION,
    TOGGLE_LANGUAGE_HOTKEY_ACTION,
    TOGGLE_WINDOW_VISIBILITY_HOTKEY_ACTION,
    GlobalHotkeyListener,
)
from core.orchestrator import AppController
from core.screen_ocr import ScreenOcrService
from prompts.templates import PromptMode
from session.manager import SessionManager
from ui.main_window import MainWindow


class HotkeyActionRouter(QObject):
    def __init__(self, controller: Any, window: Any) -> None:
        super().__init__()
        self._controller = controller
        self._window = window

    @Slot(object)
    def handle_action(self, action: object) -> None:
        if action == EXIT_HOTKEY_ACTION:
            self._window.request_exit()
            return

        if action == TOGGLE_LANGUAGE_HOTKEY_ACTION:
            self._controller.toggle_target_language()
            return

        if action == TOGGLE_WINDOW_VISIBILITY_HOTKEY_ACTION:
            self._window.toggle_window_visibility()
            return

        if isinstance(action, PromptMode):
            if not self._controller.is_busy and self._window.consume_new_session_request():
                self._controller.create_session()
            self._controller.handle_hotkey(action)


def _default_session_state_path() -> Path:
    roaming_root = Path(os.getenv("APPDATA") or (Path.home() / "AppData" / "Roaming"))
    return roaming_root / "learning-agent" / "sessions.json"


def _default_app_settings_path() -> Path:
    roaming_root = Path(os.getenv("APPDATA") or (Path.home() / "AppData" / "Roaming"))
    return roaming_root / "learning-agent" / "settings.json"


def _load_session_manager(session_state_path: Path) -> SessionManager:
    if session_state_path.exists():
        try:
            return SessionManager.load_from_file(session_state_path)
        except Exception:
            pass
    return SessionManager()


def _load_app_settings(settings_path: Path) -> AppSettings:
    if settings_path.exists():
        try:
            return load_app_settings(settings_path)
        except Exception:
            pass
    return AppSettings()


def main() -> int:
    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env")

    app = QApplication(sys.argv)
    app.setApplicationName("AI Learning Assistant")
    app.setOrganizationName("learning-agent")
    app.setStyle("Fusion")

    try:
        provider_config = load_provider_config(DEFAULT_PROVIDER_CONFIG_PATH)
        provider = build_provider(provider_config)
    except Exception as exc:  # noqa: BLE001 - startup errors should be surfaced clearly
        QMessageBox.critical(None, "Startup failed", str(exc))
        return 1

    session_state_path = _default_session_state_path()
    app_settings_path = _default_app_settings_path()
    session_manager = _load_session_manager(session_state_path)
    app_settings = _load_app_settings(app_settings_path)

    controller = AppController(
        provider,
        provider_config=provider_config,
        provider_factory=build_provider,
        default_mode=PromptMode.EXPLAIN,
        target_language=app_settings.preferred_language,
        screen_ocr_enabled=app_settings.screen_ocr_enabled,
        screen_ocr_service=ScreenOcrService(),
        session_manager=session_manager,
    )

    hotkey_listener = GlobalHotkeyListener()

    window = MainWindow(controller, hotkey_listener)
    hotkey_router = HotkeyActionRouter(controller, window)

    hotkey_listener.hotkey_triggered.connect(hotkey_router.handle_action)
    hotkey_listener.status_changed.connect(controller.status_changed.emit)
    window.start_minimized()

    def save_settings() -> None:
        try:
            save_app_settings(
                app_settings_path,
                AppSettings(
                    preferred_language=controller.preferred_language,
                    screen_ocr_enabled=controller.screen_ocr_enabled,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - settings persistence should not block exit
            controller.status_changed.emit(f"Failed to save settings: {exc}")

    controller.preferred_language_changed.connect(lambda _language: save_settings())
    controller.screen_ocr_enabled_changed.connect(lambda _enabled: save_settings())

    try:
        hotkey_listener.start()
    except Exception as exc:  # noqa: BLE001 - allow the UI to run without global hotkeys
        controller.status_changed.emit(f"Hotkeys unavailable: {exc}")

    def save_sessions() -> None:
        try:
            session_manager.save_to_file(session_state_path)
        except Exception as exc:  # noqa: BLE001 - session persistence should not block exit
            controller.status_changed.emit(f"Failed to save sessions: {exc}")

    app.aboutToQuit.connect(save_sessions)
    app.aboutToQuit.connect(save_settings)
    app.aboutToQuit.connect(hotkey_listener.stop)
    app.aboutToQuit.connect(controller.shutdown)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())


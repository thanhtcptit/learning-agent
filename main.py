from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMessageBox

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional during bootstrap
    def load_dotenv(*args, **kwargs) -> bool:
        return False

from core.config import DEFAULT_PROVIDER_CONFIG_PATH, build_provider, load_provider_config
from core.hotkey import EXIT_HOTKEY_ACTION, GlobalHotkeyListener
from core.orchestrator import AppController
from prompts.templates import DEFAULT_TARGET_LANGUAGE, PromptMode
from session.manager import SessionManager
from ui.main_window import MainWindow


def _default_session_state_path() -> Path:
    roaming_root = Path(os.getenv("APPDATA") or (Path.home() / "AppData" / "Roaming"))
    return roaming_root / "learning-agent" / "sessions.json"


def _load_session_manager(session_state_path: Path) -> SessionManager:
    if session_state_path.exists():
        try:
            return SessionManager.load_from_file(session_state_path)
        except Exception:
            pass
    return SessionManager()


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
    session_manager = _load_session_manager(session_state_path)

    controller = AppController(
        provider,
        provider_config=provider_config,
        provider_factory=build_provider,
        default_mode=PromptMode.SIMPLE,
        target_language=DEFAULT_TARGET_LANGUAGE,
        session_manager=session_manager,
    )

    hotkey_listener = GlobalHotkeyListener()

    window = MainWindow(controller, hotkey_listener)

    def handle_hotkey(action: object) -> None:
        if action == EXIT_HOTKEY_ACTION:
            window.request_exit()
            return

        if isinstance(action, PromptMode):
            controller.handle_hotkey(action)

    hotkey_listener.hotkey_triggered.connect(handle_hotkey)
    hotkey_listener.status_changed.connect(controller.status_changed.emit)
    window.show()

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
    app.aboutToQuit.connect(hotkey_listener.stop)
    app.aboutToQuit.connect(controller.shutdown)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())


from __future__ import annotations

import threading
from typing import Callable, Sequence

from PySide6.QtCore import QObject, Signal

from core.clipboard import ClipboardService
from core.config import ProviderConfig, build_provider
from llm.base import LLMMessage, LLMProvider
from prompts.templates import DEFAULT_TARGET_LANGUAGE, PromptMode, build_messages
from session.manager import ConversationSession, SessionManager


class AppController(QObject):
    sessions_changed = Signal(object)
    current_session_changed = Signal(object)
    message_upserted = Signal(object)
    preferred_language_changed = Signal(str)
    current_language_changed = Signal(str)
    status_changed = Signal(str)
    error_occurred = Signal(str)
    busy_changed = Signal(bool)

    def __init__(
        self,
        provider: LLMProvider,
        *,
        provider_config: ProviderConfig | None = None,
        provider_factory: Callable[[ProviderConfig], LLMProvider] | None = None,
        default_mode: PromptMode = PromptMode.EXPLAIN,
        target_language: str = DEFAULT_TARGET_LANGUAGE,
        clipboard_service: ClipboardService | None = None,
        session_manager: SessionManager | None = None,
    ) -> None:
        super().__init__()
        self._provider = provider
        self._provider_config = provider_config or ProviderConfig(provider="unknown", model="unknown")
        self._provider_factory = provider_factory or build_provider
        self._clipboard_service = clipboard_service or ClipboardService()
        self._session_manager = session_manager or SessionManager()
        self._default_mode = default_mode
        self._preferred_language = target_language.strip() or DEFAULT_TARGET_LANGUAGE
        self._target_language = self._preferred_language
        self._generation_lock = threading.Lock()
        self._active_request = False

        self._emit_sessions_changed()

    @property
    def default_mode(self) -> PromptMode:
        return self._default_mode

    @property
    def target_language(self) -> str:
        return self._target_language

    @property
    def preferred_language(self) -> str:
        return self._preferred_language

    @property
    def session_manager(self) -> SessionManager:
        return self._session_manager

    @property
    def provider_config(self) -> ProviderConfig:
        return self._provider_config

    @property
    def current_session(self) -> ConversationSession:
        return self._session_manager.current_session()

    @property
    def sessions(self) -> list[ConversationSession]:
        return self._session_manager.list_sessions()

    def set_default_mode(self, mode: PromptMode) -> None:
        self._default_mode = mode
        self.status_changed.emit(f"Mode set to {mode.label}")

    def set_target_language(self, language: str) -> None:
        cleaned = language.strip() or DEFAULT_TARGET_LANGUAGE
        self._preferred_language = cleaned
        self._target_language = cleaned
        self.preferred_language_changed.emit(cleaned)
        self.status_changed.emit(f"Language set to {cleaned}")
        self.current_language_changed.emit(cleaned)

    def toggle_target_language(self) -> None:
        current_language = self._target_language.strip().lower()
        next_language = "English" if current_language != "english" else self._preferred_language
        self._target_language = next_language
        self.status_changed.emit(f"Language switched to {next_language}")
        self.current_language_changed.emit(next_language)

    def create_session(self, title: str | None = None) -> ConversationSession:
        session = self._session_manager.create_session(title)
        self._emit_sessions_changed()
        self.current_session_changed.emit(session)
        return session

    def delete_session(self, session_id: str) -> ConversationSession:
        session = self._session_manager.delete_session(session_id)
        self._emit_sessions_changed()
        self.current_session_changed.emit(self._session_manager.current_session())
        return session

    def set_provider(self, provider_config: ProviderConfig) -> None:
        if self._is_busy():
            self.status_changed.emit("Wait for the current response before changing the LLM")
            return

        self._provider = self._provider_factory(provider_config)
        self._provider_config = provider_config
        self.status_changed.emit(
            f"LLM set to {provider_config.name or provider_config.display_name or provider_config.model}"
        )

    def select_session(self, session_id: str) -> ConversationSession:
        session = self._session_manager.select_session(session_id)
        self._emit_sessions_changed()
        self.current_session_changed.emit(session)
        return session

    def submit_text(self, text: str, mode: PromptMode | None = None, target_language: str | None = None) -> None:
        cleaned = text.strip()
        if not cleaned:
            self.status_changed.emit("No text to send")
            return
        self._dispatch_request(cleaned, mode or self._default_mode, target_language or self._target_language)

    def handle_hotkey(self, mode: PromptMode) -> None:
        if self._is_busy():
            self.status_changed.emit("A response is already in progress")
            return

        if not self._begin_request():
            self.status_changed.emit("A response is already in progress")
            return

        self.busy_changed.emit(True)
        self.status_changed.emit(f"Capturing selection for {mode.label}")

        def worker() -> None:
            try:
                text = self._clipboard_service.capture_selection()
                if not text.strip():
                    self.status_changed.emit("Clipboard capture returned no text")
                    self._end_request()
                    return
                self._dispatch_reserved_request(text, mode, self._target_language)
            except Exception as exc:  # noqa: BLE001 - capture failures should be surfaced to the UI
                self.error_occurred.emit(f"Clipboard capture failed: {exc}")
                self.status_changed.emit("Error")
                self._end_request()

        threading.Thread(target=worker, daemon=True).start()

    def shutdown(self) -> None:
        self.status_changed.emit("Shutting down")

    def _dispatch_request(self, text: str, mode: PromptMode, target_language: str) -> None:
        if self._begin_request() is False:
            self.status_changed.emit("A response is already in progress")
            return

        self.busy_changed.emit(True)
        self._dispatch_reserved_request(text, mode, target_language)

    def _dispatch_reserved_request(self, text: str, mode: PromptMode, target_language: str) -> None:

        session = self._session_manager.current_session()
        user_message = session.append_message("user", text, mode=mode.value)
        self.message_upserted.emit(user_message)
        self._emit_sessions_changed()

        history_messages = session.llm_history(exclude_last=1)
        assistant_message = session.append_message("assistant", "", mode=mode.value)
        self.message_upserted.emit(assistant_message)

        request_messages = [*history_messages, *build_messages(text, mode, target_language)]
        self.status_changed.emit("Thinking...")
        self.busy_changed.emit(True)

        thread = threading.Thread(
            target=self._stream_response,
            args=(assistant_message.id, request_messages),
            daemon=True,
        )
        thread.start()

    def _stream_response(self, assistant_message_id: str, messages: Sequence[LLMMessage]) -> None:
        response_text = ""

        try:
            for chunk in self._provider.stream_chat(messages):
                response_text += chunk
                updated_message = self._session_manager.update_message(assistant_message_id, response_text)
                self.message_upserted.emit(updated_message)

            if not response_text.strip():
                response_text = "No response received."
                updated_message = self._session_manager.update_message(assistant_message_id, response_text)
                self.message_upserted.emit(updated_message)

            self.status_changed.emit("Ready")
        except Exception as exc:  # noqa: BLE001 - surface provider failures to the UI
            error_text = f"LLM request failed: {exc}"
            try:
                updated_message = self._session_manager.update_message(assistant_message_id, error_text)
                self.message_upserted.emit(updated_message)
            except KeyError:
                pass
            self.error_occurred.emit(error_text)
            self.status_changed.emit("Error")
        finally:
            self._end_request()

    def _begin_request(self) -> bool:
        with self._generation_lock:
            if self._active_request:
                return False
            self._active_request = True
            return True

    def _end_request(self) -> None:
        with self._generation_lock:
            self._active_request = False
        self.busy_changed.emit(False)

    def _is_busy(self) -> bool:
        with self._generation_lock:
            return self._active_request

    def _emit_sessions_changed(self) -> None:
        self.sessions_changed.emit(self._session_manager.list_sessions())

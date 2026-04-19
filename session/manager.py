from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Mapping
from uuid import uuid4

from llm.base import LLMMessage


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


SESSION_RETENTION = timedelta(days=5)


def _summarize_title(text: str, limit: int = 48) -> str:
    normalized = " ".join(text.split())
    if not normalized:
        return "New Session"
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3].rstrip()}..."


def _format_title(title_prefix: str | None, text: str, limit: int = 48) -> str:
    summarized_text = _summarize_title(text, limit)
    cleaned_prefix = (title_prefix or "").strip()
    if not cleaned_prefix:
        return summarized_text
    return f"{cleaned_prefix}: {summarized_text}"


def _serialize_datetime(value: datetime) -> str:
    return value.isoformat()


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)

    if not isinstance(value, str):
        raise ValueError(f"Expected an ISO 8601 datetime string, got {type(value).__name__}.")

    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


@dataclass
class ConversationMessage:
    id: str = field(default_factory=lambda: uuid4().hex)
    role: str = "user"
    content: str = ""
    mode: str | None = None
    screen_context: str = ""
    include_in_context: bool = True
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "mode": self.mode,
            "screen_context": self.screen_context,
            "include_in_context": self.include_in_context,
            "created_at": _serialize_datetime(self.created_at),
            "updated_at": _serialize_datetime(self.updated_at),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConversationMessage":
        return cls(
            id=str(payload.get("id") or uuid4().hex),
            role=str(payload.get("role") or "user"),
            content=str(payload.get("content") or ""),
            mode=str(payload.get("mode")) if payload.get("mode") is not None else None,
            screen_context=str(payload.get("screen_context") or ""),
            include_in_context=bool(payload.get("include_in_context", True)),
            created_at=_parse_datetime(payload.get("created_at") or _utcnow().isoformat()),
            updated_at=_parse_datetime(payload.get("updated_at") or _utcnow().isoformat()),
        )


@dataclass
class ConversationSession:
    id: str = field(default_factory=lambda: uuid4().hex)
    title: str = "New Session"
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)
    messages: list[ConversationMessage] = field(default_factory=list)

    def touch(self) -> None:
        self.updated_at = _utcnow()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": _serialize_datetime(self.created_at),
            "updated_at": _serialize_datetime(self.updated_at),
            "messages": [message.to_dict() for message in self.messages],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConversationSession":
        raw_messages = payload.get("messages") or []
        if not isinstance(raw_messages, list):
            raise ValueError("Session messages must be a list.")

        messages: list[ConversationMessage] = []
        for raw_message in raw_messages:
            if not isinstance(raw_message, Mapping):
                raise ValueError("Each session message must be an object.")
            messages.append(ConversationMessage.from_dict(raw_message))

        return cls(
            id=str(payload.get("id") or uuid4().hex),
            title=str(payload.get("title") or "New Session"),
            created_at=_parse_datetime(payload.get("created_at") or _utcnow().isoformat()),
            updated_at=_parse_datetime(payload.get("updated_at") or _utcnow().isoformat()),
            messages=messages,
        )

    def append_message(
        self,
        role: str,
        content: str,
        *,
        mode: str | None = None,
        screen_context: str | None = None,
        include_in_context: bool = True,
        title_prefix: str | None = None,
        message_id: str | None = None,
    ) -> ConversationMessage:
        cleaned_screen_context = (screen_context or "").strip()
        message = ConversationMessage(
            id=message_id or uuid4().hex,
            role=role,
            content=content,
            mode=mode,
            screen_context=cleaned_screen_context,
            include_in_context=include_in_context,
        )
        self.messages.append(message)
        self.touch()

        if role == "user" and self.title == "New Session" and content.strip():
            self.title = _format_title(title_prefix, content)

        return message

    def set_message_include_in_context(self, message_id: str, include_in_context: bool) -> ConversationMessage:
        for message in self.messages:
            if message.id == message_id:
                message.include_in_context = include_in_context
                message.updated_at = _utcnow()
                self.touch()
                return message
        raise KeyError(f"Unknown message id: {message_id}")

    def update_message(self, message_id: str, content: str) -> ConversationMessage:
        for message in self.messages:
            if message.id == message_id:
                message.content = content
                message.updated_at = _utcnow()
                self.touch()
                return message
        raise KeyError(f"Unknown message id: {message_id}")

    def update_message_screen_context(self, message_id: str, screen_context: str) -> ConversationMessage:
        for message in self.messages:
            if message.id == message_id:
                message.screen_context = screen_context.strip()
                message.updated_at = _utcnow()
                self.touch()
                return message
        raise KeyError(f"Unknown message id: {message_id}")

    def llm_history(self, *, exclude_last: int = 0, limit: int | None = None) -> list[LLMMessage]:
        end_index = len(self.messages) - max(exclude_last, 0)
        if end_index < 0:
            end_index = 0

        messages_to_process = [
            message
            for message in self.messages[:end_index]
            if message.include_in_context and message.content.strip()
        ]

        if limit is not None:
            if limit <= 0:
                messages_to_process = []
            else:
                messages_to_process = messages_to_process[-limit:]

        return [
            LLMMessage(role=message.role, content=message.content)
            for message in messages_to_process
        ]


class SessionManager:
    def __init__(self) -> None:
        self._initialize_storage()
        self.create_session()

    def _initialize_storage(self) -> None:
        self._lock = RLock()
        self._sessions: list[ConversationSession] = []
        self._current_session: ConversationSession | None = None

    def _prune_stale_sessions_locked(self) -> None:
        cutoff = _utcnow() - SESSION_RETENTION
        retained_sessions = [session for session in self._sessions if session.updated_at >= cutoff]
        if len(retained_sessions) == len(self._sessions):
            return

        current_session_id = self._current_session.id if self._current_session else None
        self._sessions = retained_sessions

        if current_session_id is not None:
            self._current_session = next((session for session in retained_sessions if session.id == current_session_id), None)

        if self._current_session is None and self._sessions:
            self._current_session = max(self._sessions, key=lambda session: session.updated_at)

    def _ensure_current_session_locked(self) -> ConversationSession:
        if self._current_session is None:
            self._current_session = ConversationSession()
            self._sessions.append(self._current_session)
        return self._current_session

    def create_session(self, title: str | None = None) -> ConversationSession:
        with self._lock:
            self._prune_stale_sessions_locked()
            session = ConversationSession(title=title or "New Session")
            self._sessions.append(session)
            self._current_session = session
            return session

    def delete_session(self, session_id: str) -> ConversationSession:
        with self._lock:
            self._prune_stale_sessions_locked()
            for index, session in enumerate(self._sessions):
                if session.id != session_id:
                    continue

                deleted_session = self._sessions.pop(index)

                if self._current_session and self._current_session.id == session_id:
                    if self._sessions:
                        next_index = min(index, len(self._sessions) - 1)
                        self._current_session = self._sessions[next_index]
                    else:
                        self._current_session = self.create_session()

                return deleted_session

        raise KeyError(f"Unknown session id: {session_id}")

    def select_session(self, session_id: str) -> ConversationSession:
        with self._lock:
            self._prune_stale_sessions_locked()
            for session in self._sessions:
                if session.id == session_id:
                    self._current_session = session
                    return session
        raise KeyError(f"Unknown session id: {session_id}")

    def current_session(self) -> ConversationSession:
        with self._lock:
            self._prune_stale_sessions_locked()
            return self._ensure_current_session_locked()

    def list_sessions(self) -> list[ConversationSession]:
        with self._lock:
            self._prune_stale_sessions_locked()
            self._ensure_current_session_locked()
            return list(self._sessions)

    def append_message(
        self,
        role: str,
        content: str,
        *,
        mode: str | None = None,
        screen_context: str | None = None,
        include_in_context: bool = True,
        title_prefix: str | None = None,
    ) -> ConversationMessage:
        with self._lock:
            return self.current_session().append_message(
                role,
                content,
                mode=mode,
                screen_context=screen_context,
                include_in_context=include_in_context,
                title_prefix=title_prefix,
            )

    def set_message_include_in_context(self, message_id: str, include_in_context: bool) -> ConversationMessage:
        with self._lock:
            for session in self._sessions:
                try:
                    return session.set_message_include_in_context(message_id, include_in_context)
                except KeyError:
                    continue
        raise KeyError(f"Unknown message id: {message_id}")

    def update_message(self, message_id: str, content: str) -> ConversationMessage:
        with self._lock:
            for session in self._sessions:
                try:
                    return session.update_message(message_id, content)
                except KeyError:
                    continue
        raise KeyError(f"Unknown message id: {message_id}")

    def update_message_screen_context(self, message_id: str, screen_context: str) -> ConversationMessage:
        with self._lock:
            for session in self._sessions:
                try:
                    return session.update_message_screen_context(message_id, screen_context)
                except KeyError:
                    continue
        raise KeyError(f"Unknown message id: {message_id}")

    def export_state(self) -> dict[str, Any]:
        with self._lock:
            self._prune_stale_sessions_locked()
            current_session = self._ensure_current_session_locked()
            return {
                "current_session_id": current_session.id,
                "sessions": [session.to_dict() for session in self._sessions],
            }

    def import_state(self, state: Mapping[str, Any]) -> None:
        sessions_payload = state.get("sessions") or []
        if not isinstance(sessions_payload, list):
            raise ValueError("Session state must contain a list under 'sessions'.")

        sessions: list[ConversationSession] = []
        for raw_session in sessions_payload:
            if not isinstance(raw_session, Mapping):
                raise ValueError("Each session entry must be an object.")
            sessions.append(ConversationSession.from_dict(raw_session))

        if not sessions:
            sessions = [ConversationSession()]

        current_session_id = state.get("current_session_id")

        with self._lock:
            self._sessions = sessions
            self._current_session = next((session for session in sessions if session.id == current_session_id), sessions[-1])
            self._prune_stale_sessions_locked()
            if self._current_session is None:
                self._current_session = max(self._sessions, key=lambda session: session.updated_at) if self._sessions else self._ensure_current_session_locked()

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "SessionManager":
        manager = cls.__new__(cls)
        manager._initialize_storage()
        manager.import_state(state)
        return manager

    def save_to_file(self, path: Path | str) -> None:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(self.export_state(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load_from_file(cls, path: Path | str) -> "SessionManager":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("Session file must contain a JSON object.")
        return cls.from_state(payload)

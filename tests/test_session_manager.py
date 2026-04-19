from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from prompts.templates import PromptMode
import session.manager as session_manager_module
from session.manager import SessionManager


def test_session_manager_creates_session_and_updates_title() -> None:
    manager = SessionManager()

    current = manager.current_session()
    message = manager.append_message("user", "Explain transformers in machine learning.", mode=PromptMode.EXPLAIN.value)

    assert current.id == manager.current_session().id
    assert current.title == "Explain transformers in machine learning."
    assert message.role == "user"
    assert message.mode == PromptMode.EXPLAIN.value


def test_session_manager_selects_session_and_updates_messages() -> None:
    manager = SessionManager()
    first_session = manager.current_session()
    second_session = manager.create_session("Second")

    manager.select_session(first_session.id)
    first_message = manager.append_message("user", "first")
    updated = manager.update_message(first_message.id, "first updated")

    manager.select_session(second_session.id)
    second_message = manager.append_message("assistant", "reply")

    assert updated.content == "first updated"
    assert first_session.messages[0].content == "first updated"
    assert second_message.content == "reply"
    assert manager.current_session().id == second_session.id


def test_llm_history_skips_empty_content_and_excludes_recent_messages() -> None:
    session = SessionManager().current_session()
    session.append_message("user", "alpha")
    session.append_message("assistant", "")
    session.append_message("assistant", "beta")

    history = session.llm_history(exclude_last=1)

    assert [message.content for message in history] == ["alpha"]


def test_llm_history_skips_messages_excluded_from_context() -> None:
    session = SessionManager().current_session()
    session.append_message("user", "keep me")
    session.append_message("assistant", "skip me", include_in_context=False)
    session.append_message("assistant", "keep me too")

    history = session.llm_history()

    assert [message.content for message in history] == ["keep me", "keep me too"]


def test_llm_history_limits_to_latest_messages() -> None:
    session = SessionManager().current_session()
    for index in range(1, 7):
        session.append_message("user" if index % 2 else "assistant", f"message-{index}")

    history = session.llm_history(limit=3)

    assert [message.content for message in history] == ["message-4", "message-5", "message-6"]


def test_session_manager_round_trips_through_json_file(tmp_path) -> None:
    manager = SessionManager()
    first_session = manager.current_session()
    manager.append_message(
        "user",
        "persist me",
        mode=PromptMode.EXPLAIN.value,
        screen_context="OCR context",
    )
    second_session = manager.create_session("Second")
    manager.append_message("assistant", "reply", include_in_context=False)

    state_path = tmp_path / "sessions.json"
    manager.save_to_file(state_path)

    restored = SessionManager.load_from_file(state_path)

    assert [session.title for session in restored.list_sessions()] == [first_session.title, "Second"]
    assert restored.current_session().id == second_session.id
    assert restored.list_sessions()[0].messages[0].content == "persist me"
    assert restored.list_sessions()[0].messages[0].screen_context == "OCR context"
    assert restored.list_sessions()[1].messages[0].content == "reply"
    assert restored.list_sessions()[1].messages[0].include_in_context is False


def test_session_manager_updates_screen_context() -> None:
    manager = SessionManager()
    message = manager.append_message("user", "persist me")

    updated = manager.update_message_screen_context(message.id, "OCR context")

    assert updated.screen_context == "OCR context"
    assert manager.current_session().messages[0].screen_context == "OCR context"


def test_session_manager_deletes_current_session_and_keeps_valid_selection() -> None:
    manager = SessionManager()
    first_session = manager.current_session()
    second_session = manager.create_session("Second")
    third_session = manager.create_session("Third")

    manager.select_session(second_session.id)
    deleted_session = manager.delete_session(second_session.id)

    assert deleted_session.id == second_session.id
    assert [session.title for session in manager.list_sessions()] == [first_session.title, "Third"]
    assert manager.current_session().id == third_session.id

    deleted_last_session = manager.delete_session(third_session.id)

    assert deleted_last_session.id == third_session.id
    assert [session.title for session in manager.list_sessions()] == [first_session.title]
    assert manager.current_session().id == first_session.id


def test_session_manager_deletes_only_session_and_creates_replacement() -> None:
    manager = SessionManager()
    session = manager.current_session()

    deleted_session = manager.delete_session(session.id)

    assert deleted_session.id == session.id
    assert len(manager.list_sessions()) == 1
    assert manager.current_session().id != session.id
    assert manager.current_session().title == "New Session"


def test_session_manager_prunes_sessions_older_than_five_days_on_load(monkeypatch) -> None:
    now = datetime(2026, 4, 15, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(session_manager_module, "_utcnow", lambda: now)

    stale_session = session_manager_module.ConversationSession(
        title="Stale",
        created_at=now - timedelta(days=6),
        updated_at=now - timedelta(days=6),
    )
    boundary_session = session_manager_module.ConversationSession(
        title="Boundary",
        created_at=now - timedelta(days=5),
        updated_at=now - timedelta(days=5),
    )
    recent_session = session_manager_module.ConversationSession(
        title="Recent",
        created_at=now - timedelta(days=2),
        updated_at=now - timedelta(days=2),
    )

    manager = SessionManager.from_state(
        {
            "current_session_id": stale_session.id,
            "sessions": [stale_session.to_dict(), boundary_session.to_dict(), recent_session.to_dict()],
        }
    )

    assert [session.title for session in manager.list_sessions()] == ["Boundary", "Recent"]
    assert manager.current_session().title == "Recent"


def test_session_manager_prunes_sessions_older_than_five_days_before_save(tmp_path, monkeypatch) -> None:
    now = datetime(2026, 4, 15, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(session_manager_module, "_utcnow", lambda: now)

    manager = SessionManager()
    current_session = manager.current_session()
    current_session.title = "Recent"
    stale_session = session_manager_module.ConversationSession(
        title="Stale",
        created_at=now - timedelta(days=6),
        updated_at=now - timedelta(days=6),
    )
    manager._sessions.insert(0, stale_session)

    state_path = tmp_path / "sessions.json"
    manager.save_to_file(state_path)

    payload = json.loads(state_path.read_text(encoding="utf-8"))

    assert [session["title"] for session in payload["sessions"]] == ["Recent"]
    assert payload["current_session_id"] == current_session.id

from __future__ import annotations

from prompts.templates import PromptMode
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


def test_session_manager_round_trips_through_json_file(tmp_path) -> None:
    manager = SessionManager()
    first_session = manager.current_session()
    manager.append_message("user", "persist me", mode=PromptMode.EXPLAIN.value)
    second_session = manager.create_session("Second")
    manager.append_message("assistant", "reply")

    state_path = tmp_path / "sessions.json"
    manager.save_to_file(state_path)

    restored = SessionManager.load_from_file(state_path)

    assert [session.title for session in restored.list_sessions()] == [first_session.title, "Second"]
    assert restored.current_session().id == second_session.id
    assert restored.list_sessions()[0].messages[0].content == "persist me"
    assert restored.list_sessions()[1].messages[0].content == "reply"


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

"""Tests for ChatSession and ChatSessionManager."""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

import pytest

from handlers.chat_sessions import ChatSession, ChatSessionManager


class TestChatSession:
    """Tests for ChatSession message handling."""

    def test_message_accumulation(self) -> None:
        """Messages are appended in order."""
        session = ChatSession()
        session.add_user_message([{"type": "text", "text": "Hello"}])
        session.add_assistant_message("Hi there!")
        session.add_user_message([{"type": "text", "text": "How are you?"}])

        assert len(session.messages) == 3
        assert session.messages[0]["role"] == "user"
        assert session.messages[1]["role"] == "assistant"
        assert session.messages[2]["role"] == "user"

    def test_turn_counting(self) -> None:
        """turn_count only counts user messages."""
        session = ChatSession()
        assert session.turn_count == 0

        session.add_user_message([{"type": "text", "text": "Hello"}])
        assert session.turn_count == 1

        session.add_assistant_message("Hi!")
        assert session.turn_count == 1

        session.add_user_message([{"type": "text", "text": "More"}])
        assert session.turn_count == 2

    def test_trim_preserves_image_turn(self) -> None:
        """When trimming, the first turn is preserved if it has an image."""
        session = ChatSession(max_turns=2)

        # First turn with image
        session.add_user_message([
            {"type": "image", "image": "fake_image_data"},
            {"type": "text", "text": "What's this?"},
        ])
        session.add_assistant_message("It's an X-ray.")

        # Second turn
        session.add_user_message([{"type": "text", "text": "Tell me more."}])
        session.add_assistant_message("More details here.")

        # Third turn triggers trim
        session.add_user_message([{"type": "text", "text": "And more?"}])

        assert session.turn_count == 2
        # Image turn preserved
        first_content = session.messages[0]["content"]
        assert any(item.get("type") == "image" for item in first_content)
        # The second (text-only) turn was dropped, third remains
        assert session.messages[-1]["content"] == [{"type": "text", "text": "And more?"}]

    def test_trim_drops_oldest_without_image(self) -> None:
        """When first turn has no image, it gets dropped on trim."""
        session = ChatSession(max_turns=2)

        session.add_user_message([{"type": "text", "text": "Hello"}])
        session.add_assistant_message("Hi!")

        session.add_user_message([{"type": "text", "text": "More"}])
        session.add_assistant_message("More!")

        # Third turn triggers trim — first turn should be dropped
        session.add_user_message([{"type": "text", "text": "Even more"}])

        assert session.turn_count == 2
        # First remaining message should be the second turn
        assert session.messages[0]["content"] == [{"type": "text", "text": "More"}]

    def test_expiry(self) -> None:
        """Sessions expire after TTL."""
        session = ChatSession()
        assert not session.is_expired(ttl_seconds=3600)

        # Simulate old session
        session.last_active = datetime.now(timezone.utc) - timedelta(hours=2)
        assert session.is_expired(ttl_seconds=3600)


class TestChatSessionManager:
    """Tests for ChatSessionManager."""

    def test_create_and_get(self) -> None:
        """Created sessions are retrievable."""
        manager = ChatSessionManager()
        session_id = manager.create_session()
        session = manager.get_session(session_id)
        assert session is not None
        assert isinstance(session, ChatSession)

    def test_get_nonexistent_returns_none(self) -> None:
        """Getting a non-existent session returns None."""
        manager = ChatSessionManager()
        assert manager.get_session("nonexistent") is None

    def test_delete(self) -> None:
        """Deleted sessions are no longer retrievable."""
        manager = ChatSessionManager()
        session_id = manager.create_session()
        assert manager.delete_session(session_id) is True
        assert manager.get_session(session_id) is None
        assert manager.delete_session(session_id) is False

    def test_expired_returns_none(self) -> None:
        """Expired sessions are auto-removed on get."""
        manager = ChatSessionManager(default_ttl_seconds=1)
        session_id = manager.create_session()

        # Simulate expiry
        manager._sessions[session_id].last_active = (
            datetime.now(timezone.utc) - timedelta(seconds=10)
        )

        assert manager.get_session(session_id) is None

    def test_cleanup_expired(self) -> None:
        """cleanup_expired removes only expired sessions."""
        manager = ChatSessionManager(default_ttl_seconds=1)
        sid1 = manager.create_session()
        sid2 = manager.create_session()

        # Expire only one
        manager._sessions[sid1].last_active = (
            datetime.now(timezone.utc) - timedelta(seconds=10)
        )

        removed = manager.cleanup_expired()
        assert removed == 1
        assert manager.get_session(sid1) is None
        assert manager.get_session(sid2) is not None

    def test_thread_safety(self) -> None:
        """Concurrent session creation produces unique sessions."""
        manager = ChatSessionManager()
        ids: list[str] = []
        ids_lock = threading.Lock()

        def create_sessions() -> None:
            for _ in range(50):
                sid = manager.create_session()
                with ids_lock:
                    ids.append(sid)

        threads = [threading.Thread(target=create_sessions) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(ids) == 200
        assert len(set(ids)) == 200

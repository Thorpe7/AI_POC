"""In-memory chat session management for multi-turn conversations."""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any


class ChatSession:
    """Holds one conversation's message history in chat template format."""

    def __init__(self, max_turns: int = 20) -> None:
        self.messages: list[dict[str, Any]] = []
        self.max_turns = max_turns
        self.created_at = datetime.now(timezone.utc)
        self.last_active = datetime.now(timezone.utc)

    def add_user_message(self, content: list[dict[str, Any]]) -> None:
        """Append a user turn with structured content (text, image, etc.)."""
        self.messages.append({"role": "user", "content": content})
        self._trim_if_needed()
        self.last_active = datetime.now(timezone.utc)

    def add_assistant_message(self, text: str) -> None:
        """Append an assistant response."""
        self.messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
        })
        self.last_active = datetime.now(timezone.utc)

    @property
    def turn_count(self) -> int:
        """Number of user turns in the conversation."""
        return sum(1 for m in self.messages if m["role"] == "user")

    def _trim_if_needed(self) -> None:
        """Drop oldest user+assistant pair when over max_turns.

        Preserves the first turn if it contains an image so visual context
        survives trimming.
        """
        while self.turn_count > self.max_turns:
            # Determine whether the first turn has an image
            start_idx = 0
            if self.messages and self.messages[0]["role"] == "user":
                first_content = self.messages[0]["content"]
                has_image = (
                    isinstance(first_content, list)
                    and any(
                        isinstance(item, dict) and item.get("type") == "image"
                        for item in first_content
                    )
                )
                if has_image and len(self.messages) > 2:
                    start_idx = 2  # Skip first user+assistant pair

            # Find next removable user message
            removed = False
            for i in range(start_idx, len(self.messages)):
                if self.messages[i]["role"] == "user":
                    # Remove user message and its following assistant response
                    if (
                        i + 1 < len(self.messages)
                        and self.messages[i + 1]["role"] == "assistant"
                    ):
                        del self.messages[i : i + 2]
                    else:
                        del self.messages[i]
                    removed = True
                    break

            if not removed:
                break  # Safety: no removable user message found

    def is_expired(self, ttl_seconds: int = 3600) -> bool:
        """Check if session has exceeded its TTL."""
        elapsed = (datetime.now(timezone.utc) - self.last_active).total_seconds()
        return elapsed > ttl_seconds


class ChatSessionManager:
    """Thread-safe session store keyed by session ID."""

    def __init__(self, default_ttl_seconds: int = 3600) -> None:
        self._sessions: dict[str, ChatSession] = {}
        self._lock = threading.Lock()
        self.default_ttl_seconds = default_ttl_seconds

    def create_session(self, max_turns: int = 20) -> str:
        """Create a new session and return its UUID."""
        session_id = str(uuid.uuid4())
        with self._lock:
            self._sessions[session_id] = ChatSession(max_turns=max_turns)
        return session_id

    def get_session(self, session_id: str) -> ChatSession | None:
        """Return session or None. Auto-removes expired sessions."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if session.is_expired(self.default_ttl_seconds):
                del self._sessions[session_id]
                return None
            return session

    def delete_session(self, session_id: str) -> bool:
        """Delete a session. Returns True if it existed."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    def cleanup_expired(self) -> int:
        """Remove all expired sessions. Returns count removed."""
        with self._lock:
            expired = [
                sid
                for sid, s in self._sessions.items()
                if s.is_expired(self.default_ttl_seconds)
            ]
            for sid in expired:
                del self._sessions[sid]
            return len(expired)

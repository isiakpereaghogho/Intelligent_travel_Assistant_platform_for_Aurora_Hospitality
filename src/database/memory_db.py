import sqlite3
import sys
from pathlib import Path
from threading import Lock
from typing import List, Optional, Tuple

from src.exception import CustomException
from src.logger import setup_logger
from src.utils.config import get_config


logger = setup_logger()


class ConversationMemory:
    """
    Manage persistent conversation history using SQLite.
    """

    def __init__(self):
        try:
            logger.info("Initialising Conversation Memory database...")

            config = get_config()

            self.db_path = Path(config["MEMORY_DB_PATH"])

            # Create the parent folder when necessary.
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30)

            self._lock = Lock()

            self._create_table()

            logger.info("Conversation Memory initialised with database: %s", self.db_path)

        except Exception as e:
            logger.exception("Failed to initialise Conversation Memory.")
            raise CustomException(e, sys) from e

    def _create_table(self) -> None:
        """
        Create the conversation-memory table and its index.
        """

        try:
            with self._lock:
                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversation_memory (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        message TEXT NOT NULL,
                        timestamp DATETIME
                            DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )

                self.conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_session_id
                    ON conversation_memory(session_id)
                    """
                )

                self.conn.commit()

            logger.debug("Conversation memory table created successfully.")

        except Exception as e:
            logger.exception("Failed to create the conversation-memory table.")
            raise CustomException(e, sys) from e

    def store_memory(self, session_id: str, role: str, message: str) -> bool:
        """
        Store one user or assistant message.
        """
        try:
            if not session_id:
                raise ValueError("session_id cannot be empty.")

            if not role:
                raise ValueError("role cannot be empty.")

            if message is None:
                raise ValueError("message cannot be None.")

            normalised_role = role.strip().lower()

            if normalised_role not in {"user", "assistant", "system"}:
                raise ValueError("role must be 'user', 'assistant' or 'system'.")

            with self._lock:
                self.conn.execute(
                    """
                    INSERT INTO conversation_memory (
                        session_id,
                        role,
                        message
                    )
                    VALUES (?, ?, ?)
                    """,
                    (str(session_id), normalised_role, str(message)))

                self.conn.commit()

            logger.debug("Stored conversation memory — session: %s, role: %s", session_id, normalised_role)
            return True

        except Exception as e:
            logger.exception("Failed to store conversation memory.")
            raise CustomException(e, sys) from e

    def get_conversations(self, session_id: str, limit: int = 6) -> str:
        """
        Return the most recent messages for a session as formatted text.

        The selected messages are returned in chronological order.
        """

        try:
            if limit < 1:
                raise ValueError("limit must be at least 1.")

            with self._lock:
                rows = self.conn.execute(
                    """
                    SELECT role, message, timestamp
                    FROM (
                        SELECT
                            id,
                            role,
                            message,
                            timestamp
                        FROM conversation_memory
                        WHERE session_id = ?
                        ORDER BY id DESC
                        LIMIT ?
                    )
                    ORDER BY id ASC
                    """,
                    (
                        str(session_id),
                        limit)
                ).fetchall()

            if not rows:
                return "No previous conversation history."

            logger.debug("Retrieved %s messages for session: %s", len(rows), session_id)

            return "\n".join(f"{role}: {message} ({timestamp})" for role, message, timestamp in rows)

        except Exception as error:
            logger.exception("Failed to retrieve conversation memory.")
            raise CustomException(error, sys) from error

    def get_chat_history_tuples(self, session_id: str, limit: int = 6) -> List[Tuple[str, str]]:
        """
        Return recent user/assistant pairs in chronological order.

        Example:
        [("Is breakfast included?", "Yes, it is included."),
            ("What time does it start?", "Breakfast starts at 7 am.")]
        """

        try:
            if limit < 1:
                raise ValueError("limit must be at least 1.")

            with self._lock:
                rows = self.conn.execute(
                    """
                    SELECT role, message
                    FROM (
                        SELECT
                            id,
                            role,
                            message
                        FROM conversation_memory
                        WHERE session_id = ?
                        ORDER BY id DESC
                        LIMIT ?
                    )
                    ORDER BY id ASC
                    """,
                    (str(session_id), limit * 2)
                ).fetchall()

            history: List[Tuple[str, str]] = []
            current_user_message: Optional[str] = None

            for role, message in rows:
                if role == "user":
                    current_user_message = message

                elif (
                    role == "assistant" 
                    and current_user_message is not None):
                    history.append((current_user_message, message))

                    current_user_message = None

            logger.debug("Retrieved %s chat-history pairs for session: %s", len(history), session_id)
            return history[-limit:]

        except Exception as e:
            logger.exception("Failed to retrieve chat-history tuples.")
            raise CustomException(e, sys) from e

    def get_chat_history_text(self, session_id: str, limit: int = 6) -> str:
        """
        Return recent messages as formatted conversation text.

        Here, limit represents the approximate number of conversation
        turns, so up to limit × 2 individual messages are selected.
        """

        try:
            if limit < 1:
                raise ValueError("limit must be at least 1.")

            with self._lock:
                rows = self.conn.execute(
                    """
                    SELECT role, message
                    FROM (
                        SELECT
                            id,
                            role,
                            message
                        FROM conversation_memory
                        WHERE session_id = ?
                        ORDER BY id DESC
                        LIMIT ?
                    )
                    ORDER BY id ASC
                    """,
                    (
                        str(session_id),
                        limit * 2)
                ).fetchall()

            if not rows:
                return ""

            logger.debug(
                "Retrieved chat-history text for session: %s",
                session_id)

            return "\n".join(f"{role}: {message}" for role, message in rows)

        except Exception as e:
            logger.exception("Failed to retrieve chat-history text.")
            raise CustomException(e, sys) from e

    def clear_memory(self, session_id: str) -> bool:
        """
        Delete all conversation history for one session.
        """
        try:
            with self._lock:
                self.conn.execute(
                    """
                    DELETE FROM conversation_memory
                    WHERE session_id = ?
                    """,
                    (str(session_id),)
                )

                self.conn.commit()

            logger.info("Cleared memory for session: %s", session_id)

            return True

        except Exception as e:
            logger.exception("Failed to clear conversation memory.")
            raise CustomException(e, sys) from e

    def clear_all_memory(self) -> bool:
        """
        Delete conversation history for every session.
        """
        try:
            with self._lock:
                self.conn.execute("DELETE FROM conversation_memory")
                self.conn.commit()
            logger.info("Cleared all conversation memory.")
            return True

        except Exception as e:
            logger.exception("Failed to clear all conversation memory.")
            raise CustomException(e, sys) from e

    def close(self) -> None:
        """
        Close the SQLite connection.
        """

        try:
            with self._lock:
                self.conn.close()

            logger.info("Conversation-memory database connection closed.")

        except Exception:
            logger.exception("Failed to close the conversation-memory database.")


# ---------------------------------------------------------------------------
# Shared memory-database instance
# ---------------------------------------------------------------------------

_memory_db: Optional[ConversationMemory] = None
_memory_db_lock = Lock()


def get_memory_db() -> ConversationMemory:
    """
    Create the ConversationMemory object once and reuse it.

    This applies lazy initialisation: the object is not created until
    this function is first called.
    """

    global _memory_db

    if _memory_db is None:
        with _memory_db_lock:
            # Check again because another thread may have created it
            # while this thread was waiting for the lock.
            if _memory_db is None:
                _memory_db = ConversationMemory()

    return _memory_db


def close_memory_db() -> None:
    """
    Close and reset the shared ConversationMemory instance.
    """

    global _memory_db

    with _memory_db_lock:
        if _memory_db is not None:
            _memory_db.close()
            _memory_db = None
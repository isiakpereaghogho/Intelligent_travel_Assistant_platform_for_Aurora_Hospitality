import sqlite3
import sys
from pathlib import Path

from src.exception import CustomException
from src.logger import setup_logger
from src.utils.config import get_config


logger = setup_logger()
config = get_config()

CACHE_DB_PATH = Path(
    config["CACHE_DB_PATH"]
)


def get_cache_connection() -> sqlite3.Connection:
    """
    Create and return a connection to the cache database.
    """
    try:
        CACHE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(CACHE_DB_PATH), timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    except Exception as e:
        logger.error("Failed to connect to cache database.")
        raise CustomException(e, sys) from e


def initialise_cache_database() -> None:
    """
    Create the cache database table if it does not already exist.
    """
    try:
        with get_cache_connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS response_cache (
                    cache_key TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    guest_type TEXT,
                    loyalty TEXT,
                    city TEXT,
                    session_id TEXT,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_accessed_at TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_last_accessed
                ON response_cache(last_accessed_at)
                """
            )
            connection.commit()
        logger.info("Cache database initialised successfully.")

    except Exception as e:
        logger.error("Failed to initialise cache database.")
        raise CustomException(e, sys) from e


if __name__ == "__main__":
    initialise_cache_database()
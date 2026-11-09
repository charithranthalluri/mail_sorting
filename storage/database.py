"""SQLite persistence layer for the mail sorter application.

Provides a thread-safe Database class that tracks email processing state,
classification results, and actions taken. Uses WAL mode for concurrent
read access and parameterized queries to prevent SQL injection.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Optional

from storage.models import ProcessingRecord, ProcessingStatus
from utils.logger import get_logger

logger = get_logger(__name__)


class Database:
    """SQLite-backed persistence for email processing records.

    Attributes:
        db_path: Absolute path to the SQLite database file.
    """

    def __init__(self, db_path: str) -> None:
        """Initialize the database, creating parent directories and schema.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Initializing database at %s", db_path)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager that yields a configured SQLite connection.

        Enables WAL mode, foreign keys, and allows cross-thread usage.

        Yields:
            A sqlite3.Connection with row_factory set to sqlite3.Row.
        """
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Create the processed_emails table if it does not exist."""
        sql = """
        CREATE TABLE IF NOT EXISTS processed_emails (
            message_id TEXT PRIMARY KEY,
            thread_id TEXT,
            processed_at TEXT,
            category TEXT,
            priority TEXT,
            urgency TEXT,
            action_required INTEGER,
            deadline TEXT,
            summary TEXT,
            confidence REAL,
            actions_taken TEXT,
            processing_status TEXT NOT NULL DEFAULT 'discovered',
            error_message TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
        try:
            with self._connect() as conn:
                conn.execute(sql)
                conn.commit()
                logger.debug("Database schema initialized")
        except sqlite3.Error:
            logger.exception("Failed to initialize database schema")
            raise

    def is_processed(self, message_id: str) -> bool:
        """Check whether a message has been fully processed.

        Args:
            message_id: Gmail message ID.

        Returns:
            True if a record exists with status 'complete'.
        """
        sql = (
            "SELECT 1 FROM processed_emails "
            "WHERE message_id = ? AND processing_status = ?"
        )
        try:
            with self._connect() as conn:
                row = conn.execute(sql, (message_id, ProcessingStatus.COMPLETE.value)).fetchone()
                return row is not None
        except sqlite3.Error:
            logger.exception("Error checking processed status for %s", message_id)
            return False

    def has_record(self, message_id: str) -> bool:
        """Check whether any record exists for a given message.

        Args:
            message_id: Gmail message ID.

        Returns:
            True if a record exists regardless of status.
        """
        sql = "SELECT 1 FROM processed_emails WHERE message_id = ?"
        try:
            with self._connect() as conn:
                row = conn.execute(sql, (message_id,)).fetchone()
                return row is not None
        except sqlite3.Error:
            logger.exception("Error checking record existence for %s", message_id)
            return False

    def insert_record(self, record: ProcessingRecord) -> None:
        """Insert a new processing record.

        Args:
            record: The ProcessingRecord to persist.

        Raises:
            sqlite3.IntegrityError: If a record with the same message_id exists.
        """
        sql = """
        INSERT INTO processed_emails (
            message_id, thread_id, processed_at,
            category, priority, urgency, action_required,
            deadline, summary, confidence, actions_taken,
            processing_status, error_message, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._connect() as conn:
                conn.execute(sql, (
                    record.message_id,
                    record.thread_id,
                    record.processed_at.isoformat() if record.processed_at else None,
                    record.category,
                    record.priority,
                    record.urgency,
                    int(record.action_required) if record.action_required is not None else None,
                    record.deadline,
                    record.summary,
                    record.confidence,
                    record.actions_taken,
                    record.processing_status.value,
                    record.error_message,
                    now,
                    now,
                ))
                conn.commit()
                logger.debug("Inserted record for message %s", record.message_id)
        except sqlite3.IntegrityError:
            logger.warning("Duplicate record for message %s", record.message_id)
            raise
        except sqlite3.Error:
            logger.exception("Error inserting record for %s", record.message_id)
            raise

    def update_status(
        self,
        message_id: str,
        status: ProcessingStatus,
        error_message: Optional[str] = None,
    ) -> None:
        """Update the processing status of a record.

        Args:
            message_id: Gmail message ID.
            status: New processing status.
            error_message: Optional error message (typically set with FAILED).
        """
        sql = """
        UPDATE processed_emails
        SET processing_status = ?, error_message = ?, updated_at = ?
        WHERE message_id = ?
        """
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._connect() as conn:
                conn.execute(sql, (status.value, error_message, now, message_id))
                conn.commit()
                logger.debug("Updated status for %s to %s", message_id, status.value)
        except sqlite3.Error:
            logger.exception("Error updating status for %s", message_id)
            raise

    def update_classification(
        self,
        message_id: str,
        category: str,
        priority: str,
        urgency: str,
        action_required: bool,
        deadline: Optional[str],
        summary: str,
        confidence: float,
    ) -> None:
        """Update the classification fields of a record.

        Args:
            message_id: Gmail message ID.
            category: Classified category.
            priority: Classified priority level.
            urgency: Classified urgency level.
            action_required: Whether user action is needed.
            deadline: Optional deadline in ISO 8601 format.
            summary: AI-generated summary.
            confidence: Classification confidence (0.0–1.0).
        """
        sql = """
        UPDATE processed_emails
        SET category = ?, priority = ?, urgency = ?, action_required = ?,
            deadline = ?, summary = ?, confidence = ?, updated_at = ?
        WHERE message_id = ?
        """
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._connect() as conn:
                conn.execute(sql, (
                    category,
                    priority,
                    urgency,
                    int(action_required),
                    deadline,
                    summary,
                    confidence,
                    now,
                    message_id,
                ))
                conn.commit()
                logger.debug("Updated classification for %s", message_id)
        except sqlite3.Error:
            logger.exception("Error updating classification for %s", message_id)
            raise

    def update_actions(self, message_id: str, actions_taken: str) -> None:
        """Update the actions_taken field of a record.

        Args:
            message_id: Gmail message ID.
            actions_taken: JSON-encoded list of actions performed.
        """
        sql = """
        UPDATE processed_emails
        SET actions_taken = ?, updated_at = ?
        WHERE message_id = ?
        """
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._connect() as conn:
                conn.execute(sql, (actions_taken, now, message_id))
                conn.commit()
                logger.debug("Updated actions for %s", message_id)
        except sqlite3.Error:
            logger.exception("Error updating actions for %s", message_id)
            raise

    def mark_complete(self, message_id: str) -> None:
        """Mark a record as fully processed.

        Sets processing_status to COMPLETE and records the completion time.

        Args:
            message_id: Gmail message ID.
        """
        sql = """
        UPDATE processed_emails
        SET processing_status = ?, processed_at = ?, updated_at = ?
        WHERE message_id = ?
        """
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._connect() as conn:
                conn.execute(sql, (
                    ProcessingStatus.COMPLETE.value,
                    now,
                    now,
                    message_id,
                ))
                conn.commit()
                logger.debug("Marked %s as complete", message_id)
        except sqlite3.Error:
            logger.exception("Error marking %s as complete", message_id)
            raise

    def mark_failed(self, message_id: str, error: str) -> None:
        """Mark a record as failed with an error message.

        Args:
            message_id: Gmail message ID.
            error: Description of the failure.
        """
        self.update_status(message_id, ProcessingStatus.FAILED, error_message=error)

    def get_record(self, message_id: str) -> Optional[ProcessingRecord]:
        """Retrieve a processing record by message ID.

        Args:
            message_id: Gmail message ID.

        Returns:
            The ProcessingRecord if found, otherwise None.
        """
        sql = "SELECT * FROM processed_emails WHERE message_id = ?"
        try:
            with self._connect() as conn:
                row = conn.execute(sql, (message_id,)).fetchone()
                if row is None:
                    return None
                return self._row_to_record(row)
        except sqlite3.Error:
            logger.exception("Error retrieving record for %s", message_id)
            return None

    def get_failed_records(self) -> list[ProcessingRecord]:
        """Retrieve all records in FAILED status for retry.

        Returns:
            List of ProcessingRecord instances with FAILED status.
        """
        sql = "SELECT * FROM processed_emails WHERE processing_status = ?"
        try:
            with self._connect() as conn:
                rows = conn.execute(sql, (ProcessingStatus.FAILED.value,)).fetchall()
                return [self._row_to_record(row) for row in rows]
        except sqlite3.Error:
            logger.exception("Error retrieving failed records")
            return []

    def get_stats(self) -> dict:
        """Return counts of records grouped by processing status.

        Returns:
            Dictionary mapping status names to counts.
        """
        sql = "SELECT processing_status, COUNT(*) as cnt FROM processed_emails GROUP BY processing_status"
        try:
            with self._connect() as conn:
                rows = conn.execute(sql).fetchall()
                return {row["processing_status"]: row["cnt"] for row in rows}
        except sqlite3.Error:
            logger.exception("Error retrieving stats")
            return {}

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ProcessingRecord:
        """Convert a sqlite3.Row to a ProcessingRecord.

        Args:
            row: A database row from the processed_emails table.

        Returns:
            A ProcessingRecord populated from the row data.
        """
        return ProcessingRecord(
            message_id=row["message_id"],
            thread_id=row["thread_id"] or "",
            processed_at=(
                datetime.fromisoformat(row["processed_at"])
                if row["processed_at"]
                else None
            ),
            category=row["category"],
            priority=row["priority"],
            urgency=row["urgency"],
            action_required=bool(row["action_required"]) if row["action_required"] is not None else None,
            deadline=row["deadline"],
            summary=row["summary"],
            confidence=row["confidence"],
            actions_taken=row["actions_taken"] or "",
            processing_status=ProcessingStatus(row["processing_status"]),
            error_message=row["error_message"],
            created_at=(
                datetime.fromisoformat(row["created_at"])
                if row["created_at"]
                else datetime.now(timezone.utc)
            ),
            updated_at=(
                datetime.fromisoformat(row["updated_at"])
                if row["updated_at"]
                else datetime.now(timezone.utc)
            ),
        )

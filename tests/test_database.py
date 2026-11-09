"""Tests for the SQLite database layer.

Verifies record insertion, status tracking, classification updates,
failure handling, retry support, statistics, and duplicate prevention.
"""

from __future__ import annotations

import os

import pytest

from storage.database import Database
from storage.models import ProcessingRecord, ProcessingStatus


@pytest.fixture
def db(tmp_path: os.PathLike) -> Database:
    """Create a Database instance using a temporary directory."""
    db_path = str(tmp_path / "test.db")
    return Database(db_path)


def _make_record(
    message_id: str = "msg_001",
    thread_id: str = "thread_001",
) -> ProcessingRecord:
    """Create a ProcessingRecord with sensible defaults."""
    return ProcessingRecord(
        message_id=message_id,
        thread_id=thread_id,
    )


class TestInsertRecord:
    """Test inserting records into the database."""

    def test_insert_and_retrieve(self, db: Database) -> None:
        """Inserted record should be retrievable."""
        record = _make_record()
        db.insert_record(record)

        result = db.get_record("msg_001")
        assert result is not None
        assert result.message_id == "msg_001"
        assert result.thread_id == "thread_001"
        assert result.processing_status == ProcessingStatus.DISCOVERED

    def test_insert_multiple_records(self, db: Database) -> None:
        """Multiple records with different IDs should coexist."""
        db.insert_record(_make_record("msg_001"))
        db.insert_record(_make_record("msg_002"))

        assert db.get_record("msg_001") is not None
        assert db.get_record("msg_002") is not None


class TestIsProcessed:
    """Test the is_processed method."""

    def test_new_record_is_not_processed(self, db: Database) -> None:
        """A newly inserted record should not be marked as processed."""
        db.insert_record(_make_record())

        assert db.is_processed("msg_001") is False

    def test_complete_record_is_processed(self, db: Database) -> None:
        """A record marked as complete should return True."""
        db.insert_record(_make_record())
        db.mark_complete("msg_001")

        assert db.is_processed("msg_001") is True

    def test_nonexistent_is_not_processed(self, db: Database) -> None:
        """A nonexistent message_id should return False."""
        assert db.is_processed("nonexistent") is False


class TestHasRecord:
    """Test the has_record (duplicate prevention) method."""

    def test_has_record_after_insert(self, db: Database) -> None:
        """has_record should return True after insertion."""
        db.insert_record(_make_record())

        assert db.has_record("msg_001") is True

    def test_has_record_nonexistent(self, db: Database) -> None:
        """has_record should return False for unknown IDs."""
        assert db.has_record("nonexistent") is False

    def test_duplicate_insert_raises(self, db: Database) -> None:
        """Inserting a duplicate message_id should raise IntegrityError."""
        import sqlite3

        db.insert_record(_make_record())

        with pytest.raises(sqlite3.IntegrityError):
            db.insert_record(_make_record())


class TestStatusUpdates:
    """Test status update operations."""

    def test_update_status_to_parsed(self, db: Database) -> None:
        """Status should update from DISCOVERED to PARSED."""
        db.insert_record(_make_record())
        db.update_status("msg_001", ProcessingStatus.PARSED)

        record = db.get_record("msg_001")
        assert record is not None
        assert record.processing_status == ProcessingStatus.PARSED

    def test_update_status_chain(self, db: Database) -> None:
        """Status can progress through the pipeline stages."""
        db.insert_record(_make_record())

        for status in [
            ProcessingStatus.PARSED,
            ProcessingStatus.CLASSIFIED,
            ProcessingStatus.VALIDATED,
            ProcessingStatus.ACTIONED,
        ]:
            db.update_status("msg_001", status)

        record = db.get_record("msg_001")
        assert record is not None
        assert record.processing_status == ProcessingStatus.ACTIONED

    def test_update_status_with_error_message(self, db: Database) -> None:
        """Error message should be stored with status update."""
        db.insert_record(_make_record())
        db.update_status("msg_001", ProcessingStatus.FAILED, error_message="LLM timeout")

        record = db.get_record("msg_001")
        assert record is not None
        assert record.processing_status == ProcessingStatus.FAILED
        assert record.error_message == "LLM timeout"


class TestUpdateClassification:
    """Test classification field updates."""

    def test_classification_fields_stored(self, db: Database) -> None:
        """All classification fields should be stored correctly."""
        db.insert_record(_make_record())
        db.update_classification(
            message_id="msg_001",
            category="college",
            priority="high",
            urgency="normal",
            action_required=True,
            deadline="2025-11-01T23:59:00",
            summary="Admissions deadline reminder.",
            confidence=0.95,
        )

        record = db.get_record("msg_001")
        assert record is not None
        assert record.category == "college"
        assert record.priority == "high"
        assert record.urgency == "normal"
        assert record.action_required is True
        assert record.deadline == "2025-11-01T23:59:00"
        assert record.summary == "Admissions deadline reminder."
        assert record.confidence == pytest.approx(0.95)

    def test_classification_null_deadline(self, db: Database) -> None:
        """Null deadline should be stored as None."""
        db.insert_record(_make_record())
        db.update_classification(
            message_id="msg_001",
            category="newsletter",
            priority="low",
            urgency="low",
            action_required=False,
            deadline=None,
            summary="Weekly newsletter.",
            confidence=0.88,
        )

        record = db.get_record("msg_001")
        assert record is not None
        assert record.deadline is None
        assert record.action_required is False


class TestMarkFailed:
    """Test mark_failed method."""

    def test_mark_failed_sets_status_and_error(self, db: Database) -> None:
        """mark_failed should set FAILED status and store error message."""
        db.insert_record(_make_record())
        db.mark_failed("msg_001", "API rate limit exceeded")

        record = db.get_record("msg_001")
        assert record is not None
        assert record.processing_status == ProcessingStatus.FAILED
        assert record.error_message == "API rate limit exceeded"


class TestGetFailedRecords:
    """Test retrieval of failed records for retry."""

    def test_get_failed_records_returns_only_failed(self, db: Database) -> None:
        """get_failed_records should return only records with FAILED status."""
        db.insert_record(_make_record("msg_001"))
        db.insert_record(_make_record("msg_002"))
        db.insert_record(_make_record("msg_003"))

        db.mark_failed("msg_001", "Error 1")
        db.mark_failed("msg_003", "Error 3")
        db.mark_complete("msg_002")

        failed = db.get_failed_records()
        failed_ids = {r.message_id for r in failed}

        assert len(failed) == 2
        assert "msg_001" in failed_ids
        assert "msg_003" in failed_ids
        assert "msg_002" not in failed_ids

    def test_get_failed_records_empty(self, db: Database) -> None:
        """get_failed_records should return empty list when no failures."""
        db.insert_record(_make_record())
        db.mark_complete("msg_001")

        assert db.get_failed_records() == []


class TestGetStats:
    """Test statistics retrieval."""

    def test_stats_counts_by_status(self, db: Database) -> None:
        """get_stats should return correct counts grouped by status."""
        db.insert_record(_make_record("msg_001"))
        db.insert_record(_make_record("msg_002"))
        db.insert_record(_make_record("msg_003"))
        db.insert_record(_make_record("msg_004"))

        db.mark_complete("msg_001")
        db.mark_complete("msg_002")
        db.mark_failed("msg_003", "Error")

        stats = db.get_stats()

        assert stats.get("complete") == 2
        assert stats.get("failed") == 1
        assert stats.get("discovered") == 1

    def test_stats_empty_database(self, db: Database) -> None:
        """get_stats on empty database should return empty dict."""
        stats = db.get_stats()
        assert stats == {}


class TestRetryFlow:
    """Test the retry workflow: fail then reset to discovered."""

    def test_failed_then_rediscovered(self, db: Database) -> None:
        """A failed record can be reset to discovered for retry."""
        db.insert_record(_make_record())
        db.mark_failed("msg_001", "Transient error")

        # Verify it's in the failed list
        assert len(db.get_failed_records()) == 1

        # Reset for retry
        db.update_status("msg_001", ProcessingStatus.DISCOVERED)

        record = db.get_record("msg_001")
        assert record is not None
        assert record.processing_status == ProcessingStatus.DISCOVERED

        # Should no longer appear in failed list
        assert len(db.get_failed_records()) == 0

    def test_retry_to_completion(self, db: Database) -> None:
        """A retried record can proceed to completion."""
        db.insert_record(_make_record())
        db.mark_failed("msg_001", "First attempt failed")
        db.update_status("msg_001", ProcessingStatus.DISCOVERED)
        db.update_status("msg_001", ProcessingStatus.PARSED)
        db.update_status("msg_001", ProcessingStatus.CLASSIFIED)
        db.mark_complete("msg_001")

        assert db.is_processed("msg_001") is True
        assert len(db.get_failed_records()) == 0


class TestUpdateActions:
    """Test actions_taken update."""

    def test_actions_stored(self, db: Database) -> None:
        """actions_taken should be stored as a string."""
        db.insert_record(_make_record())
        actions_json = '["add_label:AI/COLLEGE", "add_star"]'
        db.update_actions("msg_001", actions_json)

        record = db.get_record("msg_001")
        assert record is not None
        assert record.actions_taken == actions_json

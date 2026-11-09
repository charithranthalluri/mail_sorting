"""End-to-end and integration tests for MailSorterApp and main loop."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai.schema import Category, EmailClassification, Priority, Urgency
from config import AppConfig, DatabaseConfig, GmailConfig, LLMConfig
from main import MailSorterApp
from storage.database import Database
from storage.models import ProcessingStatus


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> str:
    return str(tmp_path / "test_sorter.db")


@pytest.fixture
def test_config(tmp_db_path: str) -> AppConfig:
    return AppConfig(
        dry_run=True,
        initial_lookback_hours=2,
        database=DatabaseConfig(path=tmp_db_path),
        llm=LLMConfig(api_key="fake-test-key"),
    )


def _make_mock_gmail_message(msg_id: str, subject: str, body_text: str) -> dict:
    """Create a realistic mock raw Gmail API response dict."""
    encoded_body = base64.urlsafe_b64encode(body_text.encode("utf-8")).decode("utf-8")
    return {
        "id": msg_id,
        "threadId": f"thread_{msg_id}",
        "labelIds": ["INBOX", "UNREAD"],
        "snippet": body_text[:50],
        "payload": {
            "mimeType": "text/plain",
            "headers": [
                {"name": "From", "value": "Prof. Smith <smith@university.edu>"},
                {"name": "To", "value": "student@university.edu"},
                {"name": "Subject", "value": subject},
                {"name": "Date", "value": "Fri, 24 Oct 2025 10:00:00 +0000"},
            ],
            "body": {"data": encoded_body, "size": len(body_text)},
        },
    }


class TestMailSorterAppPipeline:
    """Integration test of the full processing pipeline inside MailSorterApp."""

    @patch("main.MailSorterApp._init_components")
    def test_run_once_processes_and_completes_pipeline(
        self,
        mock_init: MagicMock,
        test_config: AppConfig,
        tmp_db_path: str,
    ) -> None:
        """Single email is retrieved, parsed, cleaned, classified, actioned, and completed."""
        app = MailSorterApp(test_config)
        app._database = Database(tmp_db_path)
        app._reader = MagicMock()
        app._action_executor = MagicMock()
        app._classifier = MagicMock()
        app._parser = __import__("processing.parser", fromlist=["EmailParser"]).EmailParser()
        app._cleaner = __import__("processing.cleaner", fromlist=["EmailCleaner"]).EmailCleaner()
        app._decision_engine = __import__(
            "engine.decision_engine", fromlist=["DecisionEngine"]
        ).DecisionEngine(test_config.labels)
        app._label_id_map = {"AI/COLLEGE": "Label_1", "AI/COLLEGE/HIGH_PRIORITY": "Label_2"}
        app._start_time = datetime.now(timezone.utc)

        # Mock reader
        msg_id = "msg_101"
        app._reader.fetch_new_message_ids.return_value = [msg_id]
        app._reader.get_message.return_value = _make_mock_gmail_message(
            msg_id, "Assignment 3 Deadline", "Please submit your assignment by Friday."
        )

        # Mock classifier
        app._classifier.classify.return_value = EmailClassification(
            category=Category.COLLEGE,
            priority=Priority.HIGH,
            urgency=Urgency.HIGH,
            action_required=True,
            deadline="2025-10-31T23:59:00",
            summary="Assignment 3 due on Friday.",
            confidence=0.95,
        )

        # Process cycle
        count = app.run_once()
        assert count == 1

        # Check DB persistence
        assert app._database.is_processed(msg_id) is True
        record = app._database.get_record(msg_id)
        assert record is not None
        assert record.processing_status == ProcessingStatus.COMPLETE
        assert record.category == "college"
        assert record.priority == "high"
        assert record.action_required is True

        # Second cycle must skip already processed message
        count_second = app.run_once()
        assert count_second == 1
        # Classifier should NOT have been called again for the already processed message
        assert app._classifier.classify.call_count == 1

    @patch("main.MailSorterApp._init_components")
    def test_run_once_recovers_from_single_failure(
        self,
        mock_init: MagicMock,
        test_config: AppConfig,
        tmp_db_path: str,
    ) -> None:
        """A failure on one message records failure and does not crash the loop."""
        app = MailSorterApp(test_config)
        app._database = Database(tmp_db_path)
        app._reader = MagicMock()
        app._reader.fetch_new_message_ids.return_value = ["msg_bad", "msg_good"]
        app._reader.get_message.side_effect = [
            None,  # Simulating fetch failure
            _make_mock_gmail_message("msg_good", "Hello", "Good morning"),
        ]
        app._parser = __import__("processing.parser", fromlist=["EmailParser"]).EmailParser()
        app._cleaner = __import__("processing.cleaner", fromlist=["EmailCleaner"]).EmailCleaner()
        app._classifier = MagicMock()
        app._classifier.classify.return_value = EmailClassification(
            category=Category.PERSONAL,
            priority=Priority.NORMAL,
            urgency=Urgency.NORMAL,
            action_required=False,
            deadline=None,
            summary="Friendly greeting.",
            confidence=0.9,
        )
        app._decision_engine = __import__(
            "engine.decision_engine", fromlist=["DecisionEngine"]
        ).DecisionEngine(test_config.labels)
        app._action_executor = MagicMock()
        app._label_id_map = {}
        app._start_time = datetime.now(timezone.utc)

        count = app.run_once()
        assert count == 2

        # Check msg_bad recorded as failed
        rec_bad = app._database.get_record("msg_bad")
        assert rec_bad is not None
        assert rec_bad.processing_status == ProcessingStatus.FAILED

        # Check msg_good completed
        rec_good = app._database.get_record("msg_good")
        assert rec_good is not None
        assert rec_good.processing_status == ProcessingStatus.COMPLETE

"""Unit tests for the Gmail action executor module.

Tests verify both dry-run and live behavior for all actions, label creation,
caching, and graceful error recovery.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from googleapiclient.errors import HttpError
import httplib2

from config import GmailConfig, LabelConfig
from gmail.actions import GmailActionExecutor


@pytest.fixture
def mock_service() -> MagicMock:
    """Create a mock Gmail API service."""
    service = MagicMock()
    return service


@pytest.fixture
def gmail_config() -> GmailConfig:
    return GmailConfig()


@pytest.fixture
def label_config() -> LabelConfig:
    return LabelConfig()


@pytest.fixture
def live_executor(
    mock_service: MagicMock, gmail_config: GmailConfig, label_config: LabelConfig
) -> GmailActionExecutor:
    """Executor in live execution mode."""
    return GmailActionExecutor(
        service=mock_service,
        config=gmail_config,
        label_config=label_config,
        dry_run=False,
    )


@pytest.fixture
def dry_run_executor(
    mock_service: MagicMock, gmail_config: GmailConfig, label_config: LabelConfig
) -> GmailActionExecutor:
    """Executor in dry-run mode."""
    return GmailActionExecutor(
        service=mock_service,
        config=gmail_config,
        label_config=label_config,
        dry_run=True,
    )


class TestDryRunMode:
    """Verify that dry-run mode never touches the Gmail API."""

    def test_add_label_dry_run(
        self, dry_run_executor: GmailActionExecutor, mock_service: MagicMock
    ) -> None:
        result = dry_run_executor.add_label("msg_123", "Label_1")
        assert result is True
        mock_service.users().messages().modify.assert_not_called()

    def test_star_message_dry_run(
        self, dry_run_executor: GmailActionExecutor, mock_service: MagicMock
    ) -> None:
        result = dry_run_executor.star_message("msg_123")
        assert result is True
        mock_service.users().messages().modify.assert_not_called()

    def test_archive_message_dry_run(
        self, dry_run_executor: GmailActionExecutor, mock_service: MagicMock
    ) -> None:
        result = dry_run_executor.archive_message("msg_123")
        assert result is True
        mock_service.users().messages().modify.assert_not_called()


class TestLiveMessageActions:
    """Verify live actions call modify with proper arguments."""

    def test_add_label_live(
        self, live_executor: GmailActionExecutor, mock_service: MagicMock
    ) -> None:
        mock_service.users().messages().modify().execute.return_value = {"id": "msg_123"}
        result = live_executor.add_label("msg_123", "LABEL_ABC")
        assert result is True
        mock_service.users().messages().modify.assert_called_with(
            userId="me",
            id="msg_123",
            body={"addLabelIds": ["LABEL_ABC"]},
        )

    def test_remove_label_live(
        self, live_executor: GmailActionExecutor, mock_service: MagicMock
    ) -> None:
        mock_service.users().messages().modify().execute.return_value = {"id": "msg_123"}
        result = live_executor.remove_label("msg_123", "LABEL_ABC")
        assert result is True
        mock_service.users().messages().modify.assert_called_with(
            userId="me",
            id="msg_123",
            body={"removeLabelIds": ["LABEL_ABC"]},
        )

    def test_star_message_live(
        self, live_executor: GmailActionExecutor, mock_service: MagicMock
    ) -> None:
        mock_service.users().messages().modify().execute.return_value = {"id": "msg_123"}
        result = live_executor.star_message("msg_123")
        assert result is True
        mock_service.users().messages().modify.assert_called_with(
            userId="me",
            id="msg_123",
            body={"addLabelIds": ["STARRED"]},
        )

    def test_unstar_message_live(
        self, live_executor: GmailActionExecutor, mock_service: MagicMock
    ) -> None:
        mock_service.users().messages().modify().execute.return_value = {"id": "msg_123"}
        result = live_executor.unstar_message("msg_123")
        assert result is True
        mock_service.users().messages().modify.assert_called_with(
            userId="me",
            id="msg_123",
            body={"removeLabelIds": ["STARRED"]},
        )

    def test_archive_message_live(
        self, live_executor: GmailActionExecutor, mock_service: MagicMock
    ) -> None:
        mock_service.users().messages().modify().execute.return_value = {"id": "msg_123"}
        result = live_executor.archive_message("msg_123")
        assert result is True
        mock_service.users().messages().modify.assert_called_with(
            userId="me",
            id="msg_123",
            body={"removeLabelIds": ["INBOX"]},
        )

    def test_mark_read_live(
        self, live_executor: GmailActionExecutor, mock_service: MagicMock
    ) -> None:
        mock_service.users().messages().modify().execute.return_value = {"id": "msg_123"}
        result = live_executor.mark_read("msg_123")
        assert result is True
        mock_service.users().messages().modify.assert_called_with(
            userId="me",
            id="msg_123",
            body={"removeLabelIds": ["UNREAD"]},
        )

    def test_mark_unread_live(
        self, live_executor: GmailActionExecutor, mock_service: MagicMock
    ) -> None:
        mock_service.users().messages().modify().execute.return_value = {"id": "msg_123"}
        result = live_executor.mark_unread("msg_123")
        assert result is True
        mock_service.users().messages().modify.assert_called_with(
            userId="me",
            id="msg_123",
            body={"addLabelIds": ["UNREAD"]},
        )


class TestLabelManagement:
    """Verify label discovery, creation, and caching."""

    def test_ensure_labels_exist_already_present(
        self, live_executor: GmailActionExecutor, mock_service: MagicMock
    ) -> None:
        mock_service.users().labels().list().execute.return_value = {
            "labels": [{"id": "Label_1", "name": "AI/COLLEGE"}]
        }
        res = live_executor.ensure_labels_exist(["AI/COLLEGE"])
        assert res["AI/COLLEGE"] == "Label_1"
        mock_service.users().labels().create.assert_not_called()

    def test_ensure_labels_exist_creates_missing(
        self, live_executor: GmailActionExecutor, mock_service: MagicMock
    ) -> None:
        mock_service.users().labels().list().execute.return_value = {"labels": []}
        mock_service.users().labels().create().execute.return_value = {
            "id": "Label_New",
            "name": "AI/COLLEGE",
        }
        res = live_executor.ensure_labels_exist(["AI/COLLEGE"])
        assert res["AI/COLLEGE"] == "Label_New"
        mock_service.users().labels().create.assert_called()


class TestErrorHandling:
    """Verify API errors return False instead of raising exceptions."""

    def test_modify_error_returns_false(
        self, live_executor: GmailActionExecutor, mock_service: MagicMock
    ) -> None:
        resp = httplib2.Response({"status": 500})
        mock_service.users().messages().modify().execute.side_effect = HttpError(
            resp, b"Internal server error"
        )
        result = live_executor.add_label("msg_err", "Label_1")
        assert result is False

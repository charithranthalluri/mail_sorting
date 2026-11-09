"""Unit tests for the Gmail reader module.

Tests verify query construction, message listing, pagination, message retrieval,
and retry behavior on transient errors.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from googleapiclient.errors import HttpError
import httplib2

from config import GmailConfig
from gmail.reader import GmailReader, GmailReaderError


@pytest.fixture
def mock_service() -> MagicMock:
    return MagicMock()


@pytest.fixture
def reader(mock_service: MagicMock) -> GmailReader:
    config = GmailConfig(max_results_per_fetch=50)
    return GmailReader(service=mock_service, config=config)


class TestQueryBuilder:
    """Test build_query helper function."""

    def test_query_without_timestamp(self) -> None:
        q = GmailReader.build_query("is:unread")
        assert q == "is:unread"

    def test_query_with_timestamp(self) -> None:
        dt = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        epoch = int(dt.timestamp())
        q = GmailReader.build_query("is:unread", dt)
        assert q == f"is:unread after:{epoch}"


class TestFetchNewMessageIds:
    """Test message ID fetching and pagination."""

    def test_fetch_single_page(self, reader: GmailReader, mock_service: MagicMock) -> None:
        mock_service.users().messages().list().execute.return_value = {
            "messages": [{"id": "msg_1"}, {"id": "msg_2"}],
        }
        ids = reader.fetch_new_message_ids()
        assert ids == ["msg_1", "msg_2"]

    def test_fetch_multiple_pages(self, reader: GmailReader, mock_service: MagicMock) -> None:
        mock_service.users().messages().list().execute.side_effect = [
            {"messages": [{"id": "msg_1"}], "nextPageToken": "token_page_2"},
            {"messages": [{"id": "msg_2"}]},
        ]
        ids = reader.fetch_new_message_ids()
        assert ids == ["msg_1", "msg_2"]

    def test_fetch_empty_results(self, reader: GmailReader, mock_service: MagicMock) -> None:
        mock_service.users().messages().list().execute.return_value = {}
        ids = reader.fetch_new_message_ids()
        assert ids == []


class TestGetMessage:
    """Test retrieving full email messages."""

    def test_get_message_success(self, reader: GmailReader, mock_service: MagicMock) -> None:
        expected = {"id": "msg_100", "snippet": "Test email"}
        mock_service.users().messages().get().execute.return_value = expected

        msg = reader.get_message("msg_100")
        assert msg == expected
        mock_service.users().messages().get.assert_called_with(
            userId="me", id="msg_100", format="full"
        )


class TestRetryBehavior:
    """Test retry with backoff on transient errors."""

    def test_retry_on_429_then_succeed(self, reader: GmailReader, mock_service: MagicMock) -> None:
        resp_429 = httplib2.Response({"status": 429})
        error_429 = HttpError(resp_429, b"Rate limit exceeded")

        success_response = {"id": "msg_200", "snippet": "OK"}
        mock_service.users().messages().get().execute.side_effect = [
            error_429,
            success_response,
        ]

        with patch("time.sleep"):  # Avoid delay
            msg = reader.get_message("msg_200")
        assert msg == success_response

    def test_exhausted_retries_raises_reader_error(
        self, reader: GmailReader, mock_service: MagicMock
    ) -> None:
        resp_503 = httplib2.Response({"status": 503})
        error_503 = HttpError(resp_503, b"Service unavailable")
        mock_service.users().messages().get().execute.side_effect = error_503

        with patch("time.sleep"):
            with pytest.raises(GmailReaderError):
                reader.get_message("msg_fail")

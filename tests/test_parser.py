"""Tests for the email parser module.

Verifies that EmailParser correctly extracts fields from Gmail API
message dictionaries, including base64url-encoded bodies, multipart
messages, attachments, and RFC 2047 encoded subjects.
"""

from __future__ import annotations

import base64

import pytest

from processing.parser import EmailParser
from storage.models import EmailMessage


def _b64url(text: str) -> str:
    """Base64url-encode a string, matching Gmail API body encoding."""
    return base64.urlsafe_b64encode(text.encode()).decode()


def _make_message(
    msg_id: str = "msg_001",
    thread_id: str = "thread_001",
    subject: str = "Test Subject",
    sender: str = "alice@example.com",
    to: str = "bob@example.com",
    payload: dict | None = None,
    snippet: str = "Preview snippet",
    label_ids: list[str] | None = None,
) -> dict:
    """Build a realistic Gmail API message dictionary.

    Args:
        msg_id: Message ID.
        thread_id: Thread ID.
        subject: Subject header value.
        sender: From header value.
        to: To header value.
        payload: Optional custom payload dict. If None, an empty body is used.
        snippet: Gmail snippet preview.
        label_ids: Gmail label IDs.

    Returns:
        A dict matching the Gmail API messages.get() response format.
    """
    headers = [
        {"name": "From", "value": sender},
        {"name": "To", "value": to},
        {"name": "Subject", "value": subject},
        {"name": "Date", "value": "Mon, 20 Oct 2025 10:30:00 +0000"},
    ]

    if payload is None:
        payload = {
            "mimeType": "text/plain",
            "headers": headers,
            "body": {"size": 0},
        }
    else:
        # Ensure headers are present in the payload
        if "headers" not in payload:
            payload["headers"] = headers

    return {
        "id": msg_id,
        "threadId": thread_id,
        "snippet": snippet,
        "labelIds": label_ids or ["INBOX"],
        "payload": payload,
    }


class TestPlainTextParsing:
    """Test parsing a plain text email."""

    def test_plain_text_body_is_extracted(self) -> None:
        """Plain text/plain body should be decoded from base64url."""
        body_text = "Hello, this is a plain text email body."
        payload = {
            "mimeType": "text/plain",
            "body": {"data": _b64url(body_text), "size": len(body_text)},
        }
        msg = _make_message(payload=payload)
        result = EmailParser().parse(msg)

        assert isinstance(result, EmailMessage)
        assert result.id == "msg_001"
        assert result.thread_id == "thread_001"
        assert body_text in result.raw_body
        assert result.sender == "alice@example.com"
        assert result.subject == "Test Subject"

    def test_metadata_fields(self) -> None:
        """Verify ID, thread_id, sender, snippet are extracted."""
        body_text = "Some text."
        payload = {
            "mimeType": "text/plain",
            "body": {"data": _b64url(body_text), "size": len(body_text)},
        }
        msg = _make_message(
            msg_id="msg_xyz",
            thread_id="thread_xyz",
            sender="sender@test.com",
            snippet="My snippet",
            payload=payload,
        )
        result = EmailParser().parse(msg)

        assert result.id == "msg_xyz"
        assert result.thread_id == "thread_xyz"
        assert result.sender == "sender@test.com"
        assert result.snippet == "My snippet"


class TestHTMLOnlyParsing:
    """Test parsing an HTML-only email."""

    def test_html_body_is_extracted(self) -> None:
        """text/html body should be decoded and stored."""
        html = "<html><body><p>Hello HTML world</p></body></html>"
        payload = {
            "mimeType": "text/html",
            "body": {"data": _b64url(html), "size": len(html)},
        }
        msg = _make_message(payload=payload)
        result = EmailParser().parse(msg)

        assert "Hello HTML world" in result.raw_body
        assert result.has_html is True


class TestMultipartAlternativeParsing:
    """Test parsing multipart/alternative (text + HTML)."""

    def test_text_preferred_over_html(self) -> None:
        """When both text/plain and text/html are present, text/plain is preferred."""
        text_body = "Plain text version of the email."
        html_body = "<html><body><p>HTML version</p></body></html>"
        payload = {
            "mimeType": "multipart/alternative",
            "parts": [
                {
                    "mimeType": "text/plain",
                    "body": {"data": _b64url(text_body), "size": len(text_body)},
                },
                {
                    "mimeType": "text/html",
                    "body": {"data": _b64url(html_body), "size": len(html_body)},
                },
            ],
            "body": {"size": 0},
        }
        msg = _make_message(payload=payload)
        result = EmailParser().parse(msg)

        assert text_body in result.raw_body


class TestMultipartMixedWithAttachments:
    """Test parsing multipart/mixed with attachments."""

    def test_attachment_metadata_is_extracted(self) -> None:
        """Attachments should have filename, mimeType, and size extracted."""
        text_body = "Email with attachment."
        payload = {
            "mimeType": "multipart/mixed",
            "parts": [
                {
                    "mimeType": "text/plain",
                    "body": {"data": _b64url(text_body), "size": len(text_body)},
                },
                {
                    "mimeType": "application/pdf",
                    "filename": "report.pdf",
                    "body": {
                        "attachmentId": "att_001",
                        "size": 102400,
                    },
                },
                {
                    "mimeType": "image/png",
                    "filename": "chart.png",
                    "body": {
                        "attachmentId": "att_002",
                        "size": 51200,
                    },
                },
            ],
            "body": {"size": 0},
        }
        msg = _make_message(payload=payload)
        result = EmailParser().parse(msg)

        assert text_body in result.raw_body
        assert len(result.attachments) == 2
        assert result.attachments[0].filename == "report.pdf"
        assert result.attachments[0].mime_type == "application/pdf"
        assert result.attachments[0].size == 102400
        assert result.attachments[0].attachment_id == "att_001"
        assert result.attachments[1].filename == "chart.png"


class TestMissingBody:
    """Test handling of emails with no body content."""

    def test_missing_body_returns_empty_string(self) -> None:
        """An email with no body data should still parse with empty raw_body."""
        payload = {
            "mimeType": "text/plain",
            "body": {"size": 0},
        }
        msg = _make_message(payload=payload)
        result = EmailParser().parse(msg)

        assert result.raw_body == ""
        assert result.id == "msg_001"


class TestEncodedSubjectDecoding:
    """Test RFC 2047 encoded subject lines."""

    def test_base64_encoded_subject(self) -> None:
        """Base64-encoded subject (RFC 2047) should be decoded."""
        # RFC 2047: =?charset?encoding?encoded_text?=
        raw_subject = "=?UTF-8?B?SGVsbG8gV29ybGQ=?="
        payload = {
            "mimeType": "text/plain",
            "body": {"data": _b64url("body"), "size": 4},
        }
        msg = _make_message(subject=raw_subject, payload=payload)
        result = EmailParser().parse(msg)

        assert "Hello World" in result.subject

    def test_quoted_printable_encoded_subject(self) -> None:
        """Quoted-printable encoded subject (RFC 2047) should be decoded."""
        raw_subject = "=?UTF-8?Q?Re=3A_Meeting_Tomorrow?="
        payload = {
            "mimeType": "text/plain",
            "body": {"data": _b64url("body"), "size": 4},
        }
        msg = _make_message(subject=raw_subject, payload=payload)
        result = EmailParser().parse(msg)

        assert "Re:" in result.subject
        assert "Meeting Tomorrow" in result.subject


class TestAttachmentMetadataExtraction:
    """Test detailed attachment metadata extraction."""

    def test_attachment_without_id(self) -> None:
        """Inline attachment without attachmentId should still be captured."""
        payload = {
            "mimeType": "multipart/mixed",
            "parts": [
                {
                    "mimeType": "text/plain",
                    "body": {"data": _b64url("text"), "size": 4},
                },
                {
                    "mimeType": "image/jpeg",
                    "filename": "photo.jpg",
                    "body": {"size": 20480},
                },
            ],
            "body": {"size": 0},
        }
        msg = _make_message(payload=payload)
        result = EmailParser().parse(msg)

        assert len(result.attachments) == 1
        assert result.attachments[0].filename == "photo.jpg"
        assert result.attachments[0].mime_type == "image/jpeg"
        assert result.attachments[0].size == 20480

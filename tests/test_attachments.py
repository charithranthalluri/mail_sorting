"""Unit tests for attachment extraction and summarization."""

from __future__ import annotations

import pytest

from processing.attachments import extract_attachments, summarize_attachments, _format_size
from storage.models import AttachmentInfo


class TestExtractAttachments:
    """Test extracting attachment metadata from MIME structures."""

    def test_extract_with_filename(self) -> None:
        payload = {
            "mimeType": "multipart/mixed",
            "parts": [
                {
                    "mimeType": "text/plain",
                    "filename": "",
                    "body": {"data": "dGVzdA=="},
                },
                {
                    "mimeType": "application/pdf",
                    "filename": "syllabus.pdf",
                    "body": {"size": 204800, "attachmentId": "att_123"},
                },
            ],
        }
        attachments = extract_attachments(payload)
        assert len(attachments) == 1
        att = attachments[0]
        assert att.filename == "syllabus.pdf"
        assert att.mime_type == "application/pdf"
        assert att.size == 204800
        assert att.attachment_id == "att_123"

    def test_extract_with_content_disposition_header(self) -> None:
        payload = {
            "mimeType": "multipart/mixed",
            "parts": [
                {
                    "mimeType": "image/png",
                    "filename": "",
                    "headers": [
                        {"name": "Content-Disposition", "value": "attachment; filename=chart.png"}
                    ],
                    "body": {"size": 51200, "attachmentId": "att_456"},
                }
            ],
        }
        attachments = extract_attachments(payload)
        assert len(attachments) == 1
        assert attachments[0].mime_type == "image/png"
        assert attachments[0].size == 51200

    def test_no_attachments(self) -> None:
        payload = {
            "mimeType": "text/plain",
            "filename": "",
            "body": {"data": "aGVsbG8="},
        }
        attachments = extract_attachments(payload)
        assert attachments == []


class TestSummarizeAttachments:
    """Test human-readable summary of attachments."""

    def test_empty_summary(self) -> None:
        summary = summarize_attachments([])
        assert summary == "No attachments"

    def test_single_attachment_summary(self) -> None:
        attachments = [
            AttachmentInfo(
                filename="doc.pdf",
                mime_type="application/pdf",
                size=102400,
                attachment_id="id1",
            )
        ]
        summary = summarize_attachments(attachments)
        assert "1 attachment:" in summary
        assert "doc.pdf (application/pdf, 100KB)" in summary

    def test_multiple_attachments_summary(self) -> None:
        attachments = [
            AttachmentInfo(filename="doc.pdf", mime_type="application/pdf", size=102400),
            AttachmentInfo(filename="data.csv", mime_type="text/csv", size=15360),
        ]
        summary = summarize_attachments(attachments)
        assert "2 attachments:" in summary
        assert "doc.pdf" in summary
        assert "data.csv" in summary


class TestFormatSize:
    """Test size formatting helper."""

    def test_format_size_bytes(self) -> None:
        assert _format_size(500) == "500B"

    def test_format_size_kb(self) -> None:
        assert _format_size(2048) == "2.0KB"
        assert _format_size(102400) == "100KB"

    def test_format_size_mb(self) -> None:
        assert _format_size(2097152) == "2.0MB"

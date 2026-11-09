"""Attachment metadata extraction and summarization.

Provides helpers to recursively walk MIME payloads and collect attachment
metadata without downloading content, plus a human-readable summary
formatter used for logging and LLM context.
"""

from __future__ import annotations

from storage.models import AttachmentInfo
from utils.logger import get_logger

logger = get_logger(__name__)


def extract_attachments(payload: dict) -> list[AttachmentInfo]:
    """Recursively extract attachment metadata from a Gmail MIME payload.

    Walks all nested ``parts`` in the payload tree. A part is considered
    an attachment if it has a non-empty ``filename`` or its
    ``Content-Disposition`` starts with ``attachment``.

    Args:
        payload: The ``payload`` dict from a Gmail API message resource
            (``messages.get(format='full')``).

    Returns:
        A list of :class:`AttachmentInfo` objects with filename, MIME type,
        size, and attachment ID populated.  No attachment content is
        downloaded.
    """
    attachments: list[AttachmentInfo] = []
    _walk_parts(payload, attachments)

    logger.debug("Extracted %d attachment(s) from payload", len(attachments))
    return attachments


def summarize_attachments(attachments: list[AttachmentInfo]) -> str:
    """Return a concise human-readable summary of attachment metadata.

    The summary is intended for logging and for including in LLM context
    so the model is aware of attached files without seeing their content.

    Args:
        attachments: A list of :class:`AttachmentInfo` objects.

    Returns:
        A string such as
        ``"2 attachments: report.pdf (application/pdf, 145KB), data.csv (text/csv, 12KB)"``
        or ``"No attachments"`` if the list is empty.
    """
    if not attachments:
        return "No attachments"

    parts: list[str] = []
    for att in attachments:
        size_str = _format_size(att.size)
        parts.append(f"{att.filename} ({att.mime_type}, {size_str})")

    count = len(attachments)
    noun = "attachment" if count == 1 else "attachments"
    return f"{count} {noun}: {', '.join(parts)}"


# ======================================================================
# Internal helpers
# ======================================================================


def _walk_parts(part: dict, accumulator: list[AttachmentInfo]) -> None:
    """Recursively walk MIME parts and collect attachment metadata.

    A part qualifies as an attachment when:
    - It has a non-empty ``filename`` field, **or**
    - Its headers contain ``Content-Disposition: attachment``

    Args:
        part: A single MIME part dict from the Gmail API payload.
        accumulator: The list to append discovered attachments to.
    """
    filename: str = part.get("filename", "")
    body: dict = part.get("body", {})

    # Check Content-Disposition header as a fallback when filename is empty
    is_attachment_disposition = False
    for header in part.get("headers", []):
        if header.get("name", "").lower() == "content-disposition":
            if header.get("value", "").lower().startswith("attachment"):
                is_attachment_disposition = True
                break

    if filename or is_attachment_disposition:
        attachment = AttachmentInfo(
            filename=filename or "unnamed_attachment",
            mime_type=part.get("mimeType", "application/octet-stream"),
            size=body.get("size", 0),
            attachment_id=body.get("attachmentId"),
        )
        accumulator.append(attachment)
        logger.debug(
            "Found attachment: filename=%s mime_type=%s size=%d",
            attachment.filename,
            attachment.mime_type,
            attachment.size,
        )

    # Recurse into sub-parts (multipart containers)
    for sub_part in part.get("parts", []):
        _walk_parts(sub_part, accumulator)


def _format_size(size_bytes: int) -> str:
    """Format a byte count into a human-friendly string.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Formatted string like ``"145KB"`` or ``"2.3MB"``.
    """
    if size_bytes < 0:
        return "0B"
    if size_bytes < 1024:
        return f"{size_bytes}B"
    if size_bytes < 1024 * 1024:
        kb = size_bytes / 1024
        return f"{kb:.0f}KB" if kb >= 10 else f"{kb:.1f}KB"
    mb = size_bytes / (1024 * 1024)
    return f"{mb:.1f}MB"

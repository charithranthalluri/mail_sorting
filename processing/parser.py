"""MIME email parsing from Gmail API responses to normalized EmailMessage.

Handles the full spectrum of MIME structures returned by Gmail's
``messages.get(format='full')`` endpoint, including nested multipart
payloads, RFC 2047 encoded headers, and base64url-encoded body parts.
"""

from __future__ import annotations

import base64
import email.header
import email.utils
import re
from datetime import datetime, timezone
from typing import Optional

from storage.models import AttachmentInfo, EmailMessage
from utils.logger import get_logger

logger = get_logger(__name__)

# Header names we care about (case-insensitive lookup via _get_header)
_HEADER_FROM = "From"
_HEADER_TO = "To"
_HEADER_CC = "Cc"
_HEADER_BCC = "Bcc"
_HEADER_SUBJECT = "Subject"
_HEADER_DATE = "Date"


class EmailParser:
    """Parse raw Gmail API message dicts into normalized :class:`EmailMessage` objects.

    Usage::

        parser = EmailParser()
        msg = parser.parse(raw_gmail_response)
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, raw_message: dict) -> EmailMessage:
        """Parse a Gmail API message response into a normalized EmailMessage.

        Args:
            raw_message: The full message resource dict returned by
                ``gmail.users().messages().get(format='full')``.

        Returns:
            A populated :class:`EmailMessage` instance.

        Raises:
            ValueError: If ``raw_message`` is missing a required ``id`` field.
        """
        msg_id: str = raw_message.get("id", "")
        if not msg_id:
            raise ValueError("raw_message is missing the 'id' field")

        thread_id: str = raw_message.get("threadId", "")
        label_ids: list[str] = raw_message.get("labelIds", [])
        snippet: str = raw_message.get("snippet", "")

        payload: dict = raw_message.get("payload", {})
        headers: list[dict] = payload.get("headers", [])

        # --- Header extraction ---
        sender = self._decode_header_value(_get_header(headers, _HEADER_FROM))
        recipients = _parse_address_list(
            self._decode_header_value(_get_header(headers, _HEADER_TO))
        )
        cc = _parse_address_list(
            self._decode_header_value(_get_header(headers, _HEADER_CC))
        )
        bcc = _parse_address_list(
            self._decode_header_value(_get_header(headers, _HEADER_BCC))
        )
        subject = self._decode_header_value(_get_header(headers, _HEADER_SUBJECT))
        timestamp = _parse_date(_get_header(headers, _HEADER_DATE))

        # --- Body extraction ---
        body_text, body_html = self._extract_body(payload)

        # Prefer plain text; fall back to HTML
        if body_text:
            raw_body = body_text
            has_html = False
        elif body_html:
            raw_body = body_html
            has_html = True
        else:
            raw_body = ""
            has_html = False

        # --- Attachment extraction ---
        attachments = _extract_attachments_from_payload(payload)

        logger.debug(
            "Parsed message id=%s subject=%.60s attachments=%d",
            msg_id,
            subject,
            len(attachments),
        )

        return EmailMessage(
            id=msg_id,
            thread_id=thread_id,
            sender=sender,
            recipients=recipients,
            cc=cc,
            bcc=bcc,
            subject=subject,
            timestamp=timestamp,
            labels=label_ids,
            raw_body=raw_body,
            has_html=has_html,
            attachments=attachments,
            snippet=snippet,
        )

    # ------------------------------------------------------------------
    # Body extraction (recursive MIME walk)
    # ------------------------------------------------------------------

    def _extract_body(self, payload: dict) -> tuple[str, str]:
        """Walk the MIME tree and return ``(plain_text, html_text)``.

        The method handles:
        - ``text/plain`` and ``text/html`` leaf parts
        - ``multipart/alternative`` (prefer plain, fall back to HTML)
        - ``multipart/mixed`` (body + attachments)
        - Arbitrarily nested multipart structures
        """
        mime_type: str = payload.get("mimeType", "")
        parts: list[dict] = payload.get("parts", [])

        # Leaf node – single-part message
        if not parts:
            return self._decode_leaf(payload, mime_type)

        # Multipart – recurse into sub-parts
        collected_plain = ""
        collected_html = ""
        for part in parts:
            plain, html = self._extract_body(part)
            if plain and not collected_plain:
                collected_plain = plain
            if html and not collected_html:
                collected_html = html

        return collected_plain, collected_html

    def _decode_leaf(self, part: dict, mime_type: str) -> tuple[str, str]:
        """Decode a leaf MIME part and return ``(plain, html)``."""
        # Skip attachment parts (they have a filename)
        filename = part.get("filename", "")
        if filename:
            return "", ""

        body_data: str = part.get("body", {}).get("data", "")
        if not body_data:
            return "", ""

        decoded = _decode_base64url(body_data)

        if mime_type == "text/plain":
            return decoded, ""
        if mime_type == "text/html":
            return "", decoded

        return "", ""

    # ------------------------------------------------------------------
    # Header decoding
    # ------------------------------------------------------------------

    def _decode_header_value(self, raw_value: str) -> str:
        """Decode an RFC 2047 encoded header value to a plain string.

        Handles values like ``=?UTF-8?B?...?=`` and returns clean text.
        """
        if not raw_value:
            return ""

        try:
            decoded_parts = email.header.decode_header(raw_value)
            fragments: list[str] = []
            for data, charset in decoded_parts:
                if isinstance(data, bytes):
                    fragments.append(data.decode(charset or "utf-8", errors="replace"))
                else:
                    fragments.append(data)
            return " ".join(fragments)
        except Exception:
            logger.warning("Failed to decode header, using raw value: %.80s", raw_value)
            return raw_value


# ======================================================================
# Module-level helpers
# ======================================================================


def _get_header(headers: list[dict], name: str) -> str:
    """Return the first header value matching *name* (case-insensitive)."""
    name_lower = name.lower()
    for h in headers:
        if h.get("name", "").lower() == name_lower:
            return h.get("value", "")
    return ""


def _decode_base64url(data: str) -> str:
    """Decode a base64url-encoded string (as used by Gmail API) to UTF-8 text.

    Handles missing padding and encoding errors gracefully.
    """
    try:
        # Gmail uses URL-safe base64 with no padding
        padded = data + "=" * (4 - len(data) % 4) if len(data) % 4 else data
        decoded_bytes = base64.urlsafe_b64decode(padded)
        return decoded_bytes.decode("utf-8", errors="replace")
    except Exception:
        logger.warning("Failed to decode base64url body data (len=%d)", len(data))
        return ""


def _parse_date(date_str: str) -> Optional[datetime]:
    """Parse an RFC 2822 date string into a timezone-aware datetime.

    Returns ``None`` on failure rather than raising.
    """
    if not date_str:
        return None
    try:
        return email.utils.parsedate_to_datetime(date_str)
    except Exception:
        logger.warning("Failed to parse email date: %.60s", date_str)
        return None


def _parse_address_list(raw: str) -> list[str]:
    """Split a comma-separated address header into individual addresses.

    Each address is stripped of leading/trailing whitespace. Empty strings
    are excluded.
    """
    if not raw:
        return []
    return [addr.strip() for addr in raw.split(",") if addr.strip()]


def _extract_attachments_from_payload(payload: dict) -> list[AttachmentInfo]:
    """Recursively extract attachment metadata from a MIME payload tree.

    Args:
        payload: The ``payload`` dict from a Gmail API message resource.

    Returns:
        A list of :class:`AttachmentInfo` with filename, MIME type, size,
        and attachment ID populated.
    """
    attachments: list[AttachmentInfo] = []
    _walk_parts_for_attachments(payload, attachments)
    return attachments


def _walk_parts_for_attachments(
    part: dict, accumulator: list[AttachmentInfo]
) -> None:
    """Recursive helper that walks MIME parts collecting attachment metadata."""
    filename: str = part.get("filename", "")
    body: dict = part.get("body", {})

    # A part is treated as an attachment if it has a non-empty filename
    if filename:
        accumulator.append(
            AttachmentInfo(
                filename=filename,
                mime_type=part.get("mimeType", "application/octet-stream"),
                size=body.get("size", 0),
                attachment_id=body.get("attachmentId"),
            )
        )

    # Recurse into sub-parts (multipart containers)
    for sub_part in part.get("parts", []):
        _walk_parts_for_attachments(sub_part, accumulator)

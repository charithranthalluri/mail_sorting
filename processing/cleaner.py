"""Email body cleaning for LLM consumption.

Strips boilerplate, quoted replies, signatures, tracking pixels, and
HTML markup to produce concise plain text suitable for classification.
When in doubt, content is preserved rather than removed.
"""

from __future__ import annotations

import re
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Pre-compiled patterns
# ---------------------------------------------------------------------------

# Quoted-reply headers ("On Mon, Jan 1, 2024 at 12:00 PM X wrote:")
_RE_QUOTED_HEADER = re.compile(
    r"^On\s+.{10,80}\s+wrote:\s*$", re.MULTILINE | re.IGNORECASE
)

# Outlook-style quoted block ("From: ... Sent: ... To: ...")
_RE_OUTLOOK_QUOTE = re.compile(
    r"^-{2,}\s*(?:Original Message|Forwarded message)\s*-{2,}\s*$"
    r"|^From:\s+.+\nSent:\s+.+\nTo:\s+.+",
    re.MULTILINE | re.IGNORECASE,
)

# Line-level quoting ("> text")
_RE_LINE_QUOTE = re.compile(r"^>+\s?.*$", re.MULTILINE)

# Signature markers
_RE_SIGNATURE_SEPARATOR = re.compile(r"^--\s*$", re.MULTILINE)
_RE_SIGNATURE_PHRASES = re.compile(
    r"^(?:"
    r"Sent from my (?:iPhone|iPad|Galaxy|Android|Pixel|mobile)"
    r"|Get Outlook for (?:iOS|Android|Windows|Mac)"
    r"|Sent from (?:Mail|Yahoo Mail|AOL) for (?:Windows|Mac|iOS|Android)"
    r"|Sent via .+"
    r"|Confidentiality Notice.*"
    r"|DISCLAIMER:.*"
    r")\s*$",
    re.MULTILINE | re.IGNORECASE,
)

# Tracking / boilerplate
_RE_UNSUBSCRIBE = re.compile(
    r"(?:^|\n).*(?:unsubscribe|opt[\-\s]?out|manage\s+(?:your\s+)?(?:preferences|subscription)"
    r"|email\s+preferences|notification\s+settings).*$",
    re.MULTILINE | re.IGNORECASE,
)
_RE_VIEW_IN_BROWSER = re.compile(
    r"(?:^|\n).*(?:view\s+(?:this\s+)?(?:email\s+)?in\s+(?:your\s+)?(?:browser|web)"
    r"|having\s+trouble\s+(?:viewing|reading)).*$",
    re.MULTILINE | re.IGNORECASE,
)
_RE_TRACKING_PIXEL = re.compile(
    r"!\[(?:pixel|tracker|open|img)?\]\([^)]+\)", re.IGNORECASE
)

# Whitespace normalization
_RE_MULTIPLE_BLANK_LINES = re.compile(r"\n{3,}")
_RE_TRAILING_WHITESPACE = re.compile(r"[ \t]+$", re.MULTILINE)
_RE_MULTIPLE_SPACES = re.compile(r"[ \t]{2,}")


class EmailCleaner:
    """Clean raw email bodies into concise text ready for LLM classification.

    Usage::

        cleaner = EmailCleaner()
        clean = cleaner.clean(raw_body, is_html=True)
    """

    def clean(self, raw_body: str, is_html: bool = False) -> str:
        """Clean an email body for downstream LLM consumption.

        Args:
            raw_body: The raw email body text (plain or HTML).
            is_html: Whether *raw_body* contains HTML markup.

        Returns:
            Cleaned plain-text string preserving meaningful content such as
            dates, names, action items, and instructions.
        """
        if not raw_body:
            return ""

        text = raw_body

        # ----- HTML → text conversion -----
        if is_html:
            text = self._html_to_text(text)

        # ----- Quoted replies -----
        text = self._remove_quoted_replies(text)

        # ----- Signatures -----
        text = self._remove_signatures(text)

        # ----- Tracking / boilerplate -----
        text = self._remove_boilerplate(text)

        # ----- Whitespace normalization -----
        text = self._normalize_whitespace(text)

        logger.debug(
            "Cleaned body: original_len=%d cleaned_len=%d is_html=%s",
            len(raw_body),
            len(text),
            is_html,
        )
        return text

    # ------------------------------------------------------------------
    # HTML conversion
    # ------------------------------------------------------------------

    def _html_to_text(self, html: str) -> str:
        """Convert HTML to readable plain text using BeautifulSoup.

        Removes scripts, styles, and non-content elements. Converts
        ``<a>`` tags to ``text (url)`` format.
        """
        try:
            from bs4 import BeautifulSoup, Tag  # type: ignore[import-untyped]
        except ImportError:
            logger.warning(
                "beautifulsoup4 is not installed; stripping HTML with regex fallback"
            )
            return self._html_to_text_fallback(html)

        soup = BeautifulSoup(html, "html.parser")

        # Remove non-content tags entirely
        for tag_name in ("script", "style", "head", "meta", "link"):
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # Convert <a href="url">text</a> → "text (url)"
        for a_tag in soup.find_all("a"):
            if isinstance(a_tag, Tag):
                href = a_tag.get("href", "")
                link_text = a_tag.get_text(strip=True)
                if href and link_text:
                    a_tag.replace_with(f"{link_text} ({href})")
                elif href:
                    a_tag.replace_with(f"({href})")
                else:
                    a_tag.replace_with(link_text)

        return soup.get_text(separator="\n")

    @staticmethod
    def _html_to_text_fallback(html: str) -> str:
        """Minimal regex-based HTML stripping when BeautifulSoup is unavailable."""
        text = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
        text = re.sub(r"</(p|div|tr|li|h[1-6])>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"&nbsp;", " ", text)
        text = re.sub(r"&amp;", "&", text)
        text = re.sub(r"&lt;", "<", text)
        text = re.sub(r"&gt;", ">", text)
        text = re.sub(r"&#\d+;", "", text)
        return text

    # ------------------------------------------------------------------
    # Quoted replies
    # ------------------------------------------------------------------

    def _remove_quoted_replies(self, text: str) -> str:
        """Remove or truncate quoted previous replies.

        Keeps the first quoted-header line as a context indicator
        (e.g. "On Jan 1 wrote:") but removes the quoted body below it.
        """
        # Handle "On … wrote:" style
        match = _RE_QUOTED_HEADER.search(text)
        if match:
            # Keep everything before the quote header, plus the header line itself
            text = text[: match.end()].rstrip() + "\n[previous message quoted]"

        # Handle Outlook-style "--- Original Message ---"
        match = _RE_OUTLOOK_QUOTE.search(text)
        if match:
            text = text[: match.start()].rstrip() + "\n[previous message quoted]"

        # Remove any remaining "> ..." lines (in case they appear before headers)
        # Only remove blocks of 3+ consecutive quoted lines
        lines = text.split("\n")
        result_lines: list[str] = []
        consecutive_quotes = 0
        quote_buffer: list[str] = []

        for line in lines:
            if _RE_LINE_QUOTE.match(line):
                consecutive_quotes += 1
                quote_buffer.append(line)
            else:
                if consecutive_quotes >= 3:
                    # Drop the quote block, keep a marker
                    result_lines.append("[quoted text removed]")
                else:
                    # Fewer than 3 lines – keep them (might be intentional)
                    result_lines.extend(quote_buffer)
                result_lines.append(line)
                consecutive_quotes = 0
                quote_buffer = []

        # Handle trailing quote block
        if consecutive_quotes >= 3:
            result_lines.append("[quoted text removed]")
        else:
            result_lines.extend(quote_buffer)

        return "\n".join(result_lines)

    # ------------------------------------------------------------------
    # Signatures
    # ------------------------------------------------------------------

    def _remove_signatures(self, text: str) -> str:
        """Remove email signatures detected by common markers."""
        # "-- " separator (RFC 3676)
        match = _RE_SIGNATURE_SEPARATOR.search(text)
        if match:
            text = text[: match.start()].rstrip()

        # Device / app signature phrases
        match = _RE_SIGNATURE_PHRASES.search(text)
        if match:
            text = text[: match.start()].rstrip()

        return text

    # ------------------------------------------------------------------
    # Boilerplate & tracking
    # ------------------------------------------------------------------

    def _remove_boilerplate(self, text: str) -> str:
        """Remove tracking pixels, unsubscribe links, and other boilerplate."""
        text = _RE_TRACKING_PIXEL.sub("", text)
        text = _RE_UNSUBSCRIBE.sub("", text)
        text = _RE_VIEW_IN_BROWSER.sub("", text)
        return text

    # ------------------------------------------------------------------
    # Whitespace normalization
    # ------------------------------------------------------------------

    def _normalize_whitespace(self, text: str) -> str:
        """Collapse excessive whitespace while preserving paragraph structure."""
        text = _RE_TRAILING_WHITESPACE.sub("", text)
        text = _RE_MULTIPLE_SPACES.sub(" ", text)
        text = _RE_MULTIPLE_BLANK_LINES.sub("\n\n", text)
        return text.strip()

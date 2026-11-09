"""Tests for the email cleaner module.

Verifies HTML-to-text conversion, whitespace normalization, quoted reply
removal, signature removal, and that important content like dates, deadlines,
and names are preserved.
"""

from __future__ import annotations

import pytest

from processing.cleaner import EmailCleaner


@pytest.fixture
def cleaner() -> EmailCleaner:
    """Return a fresh EmailCleaner instance."""
    return EmailCleaner()


class TestHTMLToTextConversion:
    """Test HTML to plain text conversion."""

    def test_paragraph_extraction(self, cleaner: EmailCleaner) -> None:
        """Paragraphs should be converted to plain text."""
        html = "<p>First paragraph.</p><p>Second paragraph.</p>"
        result = cleaner.clean(html, is_html=True)

        assert "First paragraph." in result
        assert "Second paragraph." in result

    def test_link_text_preserved(self, cleaner: EmailCleaner) -> None:
        """Link anchor text should be preserved."""
        html = '<p>Click <a href="https://example.com">here</a> to visit.</p>'
        result = cleaner.clean(html, is_html=True)

        assert "here" in result
        assert "visit" in result

    def test_list_items_converted(self, cleaner: EmailCleaner) -> None:
        """List items should be converted to text."""
        html = "<ul><li>Item one</li><li>Item two</li><li>Item three</li></ul>"
        result = cleaner.clean(html, is_html=True)

        assert "Item one" in result
        assert "Item two" in result
        assert "Item three" in result

    def test_html_tags_stripped(self, cleaner: EmailCleaner) -> None:
        """HTML tags should be removed from output."""
        html = "<div><b>Bold</b> and <i>italic</i> text.</div>"
        result = cleaner.clean(html, is_html=True)

        assert "<div>" not in result
        assert "<b>" not in result
        assert "<i>" not in result
        assert "Bold" in result
        assert "italic" in result


class TestWhitespaceNormalization:
    """Test excessive whitespace normalization."""

    def test_excessive_newlines_collapsed(self, cleaner: EmailCleaner) -> None:
        """Multiple consecutive newlines should be collapsed."""
        text = "Line one.\n\n\n\n\nLine two.\n\n\n\nLine three."
        result = cleaner.clean(text, is_html=False)

        # Should not have more than 2 consecutive newlines
        assert "\n\n\n" not in result
        assert "Line one." in result
        assert "Line two." in result
        assert "Line three." in result

    def test_excessive_spaces_collapsed(self, cleaner: EmailCleaner) -> None:
        """Multiple consecutive spaces should be collapsed."""
        text = "Word    with     many      spaces."
        result = cleaner.clean(text, is_html=False)

        assert "    " not in result
        assert "Word" in result
        assert "spaces" in result

    def test_leading_trailing_whitespace_stripped(self, cleaner: EmailCleaner) -> None:
        """Leading and trailing whitespace should be stripped."""
        text = "   \n\n  Hello world.  \n\n   "
        result = cleaner.clean(text, is_html=False)

        assert result == result.strip()
        assert "Hello world." in result


class TestQuotedReplyRemoval:
    """Test removal of quoted reply content."""

    def test_on_wrote_pattern(self, cleaner: EmailCleaner) -> None:
        """'On ... wrote:' quoted replies should be removed."""
        text = (
            "Thanks for the update!\n\n"
            "On Mon, Oct 20, 2025 at 10:30 AM Alice <alice@example.com> wrote:\n"
            "> Original message content here\n"
            "> More quoted text"
        )
        result = cleaner.clean(text, is_html=False)

        assert "Thanks for the update!" in result
        assert "Original message content here" not in result

    def test_angle_bracket_quoted_lines(self, cleaner: EmailCleaner) -> None:
        """Lines starting with '>' should be removed."""
        text = (
            "My reply to you.\n\n"
            "> This is quoted\n"
            "> More quoted text\n"
            ">> Double quoted"
        )
        result = cleaner.clean(text, is_html=False)

        assert "My reply to you." in result
        assert "This is quoted" not in result

    def test_from_sent_pattern(self, cleaner: EmailCleaner) -> None:
        """'From: ... Sent: ...' Outlook-style quoted replies should be removed."""
        text = (
            "Acknowledged, will do.\n\n"
            "From: Bob Smith\n"
            "Sent: Monday, October 20, 2025 10:30 AM\n"
            "To: Alice Jones\n"
            "Subject: Re: Project Update\n\n"
            "The original message text."
        )
        result = cleaner.clean(text, is_html=False)

        assert "Acknowledged, will do." in result
        assert "The original message text." not in result


class TestSignatureRemoval:
    """Test email signature removal."""

    def test_double_dash_separator(self, cleaner: EmailCleaner) -> None:
        """Signature after '--' separator should be removed."""
        text = (
            "Please see the attached document.\n\n"
            "--\n"
            "John Doe\n"
            "Software Engineer\n"
            "john@company.com"
        )
        result = cleaner.clean(text, is_html=False)

        assert "Please see the attached document." in result
        assert "Software Engineer" not in result

    def test_sent_from_my_iphone(self, cleaner: EmailCleaner) -> None:
        """'Sent from my iPhone' signature should be removed."""
        text = "Quick reply from mobile.\n\nSent from my iPhone"
        result = cleaner.clean(text, is_html=False)

        assert "Quick reply from mobile." in result
        assert "Sent from my iPhone" not in result


class TestImportantContentPreserved:
    """Test that dates, deadlines, and names survive cleaning."""

    def test_dates_preserved(self, cleaner: EmailCleaner) -> None:
        """Dates in the email body should be preserved."""
        text = "The meeting is scheduled for October 25, 2025 at 3:00 PM."
        result = cleaner.clean(text, is_html=False)

        assert "October 25, 2025" in result
        assert "3:00 PM" in result

    def test_deadlines_preserved(self, cleaner: EmailCleaner) -> None:
        """Deadline mentions should be preserved."""
        text = "Please submit the report by 2025-11-01. The deadline is firm."
        result = cleaner.clean(text, is_html=False)

        assert "2025-11-01" in result
        assert "deadline" in result

    def test_names_preserved(self, cleaner: EmailCleaner) -> None:
        """People's names should be preserved."""
        text = "Dear Professor Smith, I wanted to follow up with Dr. Johnson."
        result = cleaner.clean(text, is_html=False)

        assert "Professor Smith" in result
        assert "Dr. Johnson" in result


class TestPlainTextCleaning:
    """Test cleaning of plain text (no HTML)."""

    def test_plain_text_passthrough(self, cleaner: EmailCleaner) -> None:
        """Simple plain text should pass through with whitespace normalization."""
        text = "This is a simple email with no special formatting."
        result = cleaner.clean(text, is_html=False)

        assert "This is a simple email with no special formatting." in result


class TestEmptyInput:
    """Test handling of empty input."""

    def test_empty_string(self, cleaner: EmailCleaner) -> None:
        """Empty string input should return empty string."""
        result = cleaner.clean("", is_html=False)

        assert result == ""

    def test_whitespace_only(self, cleaner: EmailCleaner) -> None:
        """Whitespace-only input should return empty string."""
        result = cleaner.clean("   \n\n\t  ", is_html=False)

        assert result == ""

    def test_none_like_empty(self, cleaner: EmailCleaner) -> None:
        """HTML with no text content should return empty string."""
        result = cleaner.clean("<html><body></body></html>", is_html=True)

        assert result.strip() == ""

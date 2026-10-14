import imaplib
import email
import os
import re
import html as html_module

from email.header import decode_header
from html.parser import HTMLParser
from dataclasses import dataclass
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

USERNAME = os.getenv("EMAIL")
PASSWORD = os.getenv("PASSWORD")

IMAP_SERVER = "imap.gmail.com"


# ============================================================
# EMAIL DATA MODEL
# ============================================================

@dataclass
class EmailData:
    uid: str
    sender: str
    subject: str
    date: str
    body: str


# ============================================================
# HTML TEXT EXTRACTOR
# ============================================================

class HTMLTextExtractor(HTMLParser):

    def __init__(self):
        super().__init__()

        self.text = []

        self.ignore_tags = {
            "style",
            "script",
            "head",
            "title",
            "meta",
            "link",
            "svg",
            "noscript"
        }

        self.ignore_depth = 0

    def handle_starttag(self, tag, attrs):

        tag = tag.lower()

        if tag in self.ignore_tags:

            self.ignore_depth += 1
            return

        if self.ignore_depth == 0:

            if tag in {
                "br",
                "p",
                "div",
                "section",
                "article",
                "header",
                "footer",
                "main",
                "table",
                "tr",
                "td",
                "th",
                "li",
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6"
            }:

                self.text.append("\n")

    def handle_endtag(self, tag):

        tag = tag.lower()

        if tag in self.ignore_tags:

            if self.ignore_depth > 0:
                self.ignore_depth -= 1

            return

        if self.ignore_depth == 0:

            if tag in {
                "p",
                "div",
                "section",
                "article",
                "header",
                "footer",
                "main",
                "table",
                "tr",
                "td",
                "th",
                "li",
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6"
            }:

                self.text.append("\n")

    def handle_data(self, data):

        if self.ignore_depth > 0:
            return

        if data:
            self.text.append(data)

    def get_text(self):

        return "".join(self.text)


# ============================================================
# HTML → TEXT
# ============================================================

def html_to_text(html):

    if not html:
        return ""

    parser = HTMLTextExtractor()

    try:

        parser.feed(html)
        parser.close()

        text = parser.get_text()

    except Exception:

        # Fallback for badly malformed HTML
        text = re.sub(
            r"<[^>]+>",
            " ",
            html
        )

    text = html_module.unescape(text)

    return text


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if not text:
        return ""

    # --------------------------------------------------------
    # Decode HTML entities
    # --------------------------------------------------------

    text = html_module.unescape(text)

    # --------------------------------------------------------
    # Remove invisible Unicode characters
    # --------------------------------------------------------

    invisible_chars = [
        "\u200b",
        "\u200c",
        "\u200d",
        "\u2060",
        "\ufeff",
        "\u00ad",
        "\u034f"
    ]

    for char in invisible_chars:
        text = text.replace(char, "")

    # --------------------------------------------------------
    # Replace non-breaking spaces
    # --------------------------------------------------------

    text = text.replace("\xa0", " ")

    # --------------------------------------------------------
    # Remove control characters
    # --------------------------------------------------------

    text = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]",
        "",
        text
    )

    # --------------------------------------------------------
    # Convert <URL> → URL
    # --------------------------------------------------------

    text = re.sub(
        r"<(https?://[^>\s]+)>",
        r"\1",
        text
    )

    # --------------------------------------------------------
    # Remove template placeholders
    #
    # Example:
    # <[[${FacebookLink}]]>
    # --------------------------------------------------------

    text = re.sub(
        r"<\[\[.*?\]\]>",
        "",
        text
    )

    # --------------------------------------------------------
    # Normalize spaces and tabs
    # --------------------------------------------------------

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # --------------------------------------------------------
    # Remove spaces before punctuation
    # --------------------------------------------------------

    text = re.sub(
        r"\s+([,.!?;:])",
        r"\1",
        text
    )

    # --------------------------------------------------------
    # Normalize line endings
    # --------------------------------------------------------

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    # --------------------------------------------------------
    # Remove whitespace at beginning of lines
    # --------------------------------------------------------

    text = re.sub(
        r"\n[ \t]+",
        "\n",
        text
    )

    # --------------------------------------------------------
    # Collapse excessive blank lines
    # --------------------------------------------------------

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    # --------------------------------------------------------
    # Remove lines consisting only of symbols
    # --------------------------------------------------------

    text = re.sub(
        r"(?m)^[\s\-_=*|<>]+$",
        "",
        text
    )

    return text.strip()


# ============================================================
# HEADER DECODING
# ============================================================

def decode_header_value(value):

    if not value:
        return ""

    decoded_parts = decode_header(value)

    result = []

    for text, encoding in decoded_parts:

        if isinstance(text, bytes):

            try:

                decoded = text.decode(
                    encoding or "utf-8",
                    errors="replace"
                )

            except (LookupError, UnicodeDecodeError):

                decoded = text.decode(
                    "utf-8",
                    errors="replace"
                )

            result.append(decoded)

        else:

            result.append(text)

    return "".join(result).strip()


# ============================================================
# GMAIL CONNECTION
# ============================================================

def connect_to_mail():

    if not USERNAME:
        raise ValueError(
            "EMAIL is missing from .env"
        )

    if not PASSWORD:
        raise ValueError(
            "PASSWORD is missing from .env"
        )

    mail = imaplib.IMAP4_SSL(
        IMAP_SERVER
    )

    mail.login(
        USERNAME,
        PASSWORD
    )

    status, _ = mail.select(
        "INBOX"
    )

    if status != "OK":

        mail.logout()

        raise RuntimeError(
            "Could not open Gmail inbox"
        )

    return mail


# ============================================================
# PAYLOAD DECODING
# ============================================================

def decode_payload(part):

    payload = part.get_payload(
        decode=True
    )

    if payload is None:
        return ""

    if isinstance(payload, str):
        return payload

    charset = (
        part.get_content_charset()
        or "utf-8"
    )

    try:

        return payload.decode(
            charset,
            errors="replace"
        )

    except (LookupError, UnicodeDecodeError):

        return payload.decode(
            "utf-8",
            errors="replace"
        )


# ============================================================
# ATTACHMENT CHECK
# ============================================================

def is_attachment(part):

    disposition = (
        part.get(
            "Content-Disposition"
        )
        or ""
    ).lower()

    filename = (
        part.get_filename()
        or ""
    )

    return (
        "attachment" in disposition
        or bool(filename)
    )


# ============================================================
# EXTRACT TEXT PARTS
# ============================================================

def collect_body_parts(msg):

    plain_parts = []
    html_parts = []

    # --------------------------------------------------------
    # Multipart message
    # --------------------------------------------------------

    if msg.is_multipart():

        for part in msg.walk():

            content_type = (
                part.get_content_type()
                or ""
            ).lower()

            # Ignore multipart containers
            if content_type.startswith(
                "multipart/"
            ):
                continue

            # Ignore attachments
            if is_attachment(part):
                continue

            # Only text
            if content_type not in {
                "text/plain",
                "text/html"
            }:
                continue

            content = decode_payload(
                part
            )

            if not content:
                continue

            if content_type == "text/plain":

                plain_parts.append(
                    content
                )

            elif content_type == "text/html":

                html_parts.append(
                    content
                )

    # --------------------------------------------------------
    # Non-multipart message
    # --------------------------------------------------------

    else:

        content_type = (
            msg.get_content_type()
            or ""
        ).lower()

        content = decode_payload(
            msg
        )

        if content:

            if content_type == "text/plain":

                plain_parts.append(
                    content
                )

            elif content_type == "text/html":

                html_parts.append(
                    content
                )

    return (
        plain_parts,
        html_parts
    )


# ============================================================
# EMAIL BODY EXTRACTION
# ============================================================

def extract_body(msg):

    plain_parts, html_parts = (
        collect_body_parts(msg)
    )

    # --------------------------------------------------------
    # 1. Prefer plain text
    # --------------------------------------------------------

    if plain_parts:

        plain_text = clean_text(
            "\n".join(
                plain_parts
            )
        )

        useless_plain_text = {
            "to view this email message, open it in a program that understands html!",
            "to view this email, please enable html.",
            "this email requires html."
        }

        if (
            plain_text
            and plain_text.lower()
            not in useless_plain_text
        ):

            return plain_text

    # --------------------------------------------------------
    # 2. HTML fallback
    # --------------------------------------------------------

    if html_parts:

        combined_html = "\n".join(
            html_parts
        )

        html_text = html_to_text(
            combined_html
        )

        html_text = clean_text(
            html_text
        )

        if html_text:
            return html_text

    # --------------------------------------------------------
    # 3. Last-resort raw payload
    #
    # Some badly formatted emails don't advertise their
    # content type correctly. Try to recover text anyway.
    # --------------------------------------------------------

    raw_payload = msg.get_payload(
        decode=True
    )

    if isinstance(
        raw_payload,
        bytes
    ):

        try:

            fallback_text = raw_payload.decode(
                msg.get_content_charset()
                or "utf-8",
                errors="ignore"
            )

        except Exception:

            fallback_text = raw_payload.decode(
                "utf-8",
                errors="ignore"
            )

        # If it looks like HTML, parse it.
        if re.search(
            r"<(html|body|div|p|table|br)\b",
            fallback_text,
            re.IGNORECASE
        ):

            fallback_text = html_to_text(
                fallback_text
            )

        fallback_text = clean_text(
            fallback_text
        )

        if fallback_text:
            return fallback_text

    return ""


# ============================================================
# PARSE ONE EMAIL
# ============================================================

def parse_email(
    email_id,
    data
):

    # Gmail normally returns the RFC822 message
    # as the second item.
    raw_message = data[0][1]

    msg = email.message_from_bytes(
        raw_message
    )

    sender = decode_header_value(
        msg.get("From")
    )

    subject = decode_header_value(
        msg.get("Subject")
    )

    date = msg.get(
        "Date",
        ""
    )

    body = extract_body(
        msg
    )

    return EmailData(
        uid=email_id.decode(),
        sender=sender,
        subject=subject,
        date=date,
        body=body
    )


# ============================================================
# FETCH EMAILS
# ============================================================

def fetch_emails(
    limit=10
):

    mail = connect_to_mail()

    emails = []

    try:

        status, search_data = mail.search(
            None,
            "ALL"
        )

        if status != "OK":

            raise RuntimeError(
                "Failed to search mailbox"
            )

        email_ids = search_data[0].split()

        if not email_ids:
            return []

        # Newest N emails
        selected_ids = email_ids[-limit:]

        # Newest first
        selected_ids = list(
            reversed(
                selected_ids
            )
        )

        for email_id in selected_ids:

            try:

                status, data = mail.fetch(
                    email_id,
                    "(RFC822)"
                )

                if status != "OK":

                    print(
                        f"Could not fetch "
                        f"email {email_id.decode()}"
                    )

                    continue

                parsed_email = parse_email(
                    email_id,
                    data
                )

                emails.append(
                    parsed_email
                )

            except Exception as error:

                print(
                    f"Error processing "
                    f"email {email_id.decode()}: "
                    f"{error}"
                )

        return emails

    finally:

        try:
            mail.close()

        except Exception:
            pass

        try:
            mail.logout()

        except Exception:
            pass


# ============================================================
# DISPLAY EMAILS
# ============================================================

def display_emails(
    emails
):

    for email_data in emails:

        print(
            "-" * 60
        )

        print(
            "UID:",
            email_data.uid
        )

        print(
            "From:",
            email_data.sender
        )

        print(
            "Subject:",
            email_data.subject
        )

        print(
            "Date:",
            email_data.date
        )

        print(
            "Body:"
        )

        if email_data.body:

            print(
                email_data.body
            )

        else:

            print(
                "[NO TEXT CONTENT FOUND]"
            )

        print(
            "-" * 60
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    emails = fetch_emails(
        limit=10
    )

    print(
        f"\nFetched {len(emails)} emails.\n"
    )

    display_emails(
        emails
    )

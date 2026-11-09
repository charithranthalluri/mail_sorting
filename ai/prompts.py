"""LLM prompt management for email classification.

Defines the system prompt, version, and user-prompt builder used by the
classifier to communicate with the LLM.  All instructions are designed to
produce deterministic, schema-conforming JSON output.
"""

from __future__ import annotations

from ai.schema import CLASSIFICATION_SCHEMA_DESCRIPTION

PROMPT_VERSION: str = "1.0.0"

SYSTEM_PROMPT: str = f"""\
You are an email classification engine.  Your ONLY job is to read the
provided email and return a single JSON object that conforms to the schema
below.  Do NOT include any commentary, explanation, markdown formatting,
code fences, or extra text — output ONLY the raw JSON object.

=== REQUIRED JSON SCHEMA ===
{CLASSIFICATION_SCHEMA_DESCRIPTION}

=== ALLOWED ENUM VALUES ===
category : one of "college", "work", "personal", "finance", "shopping",
           "social", "newsletter", "promotion", "notification", "spam",
           "other"
priority : one of "critical", "high", "normal", "low"
urgency  : one of "immediate", "high", "normal", "low"

=== RULES ===
1. Return ONLY valid JSON.  No markdown, no commentary, no extra text.
2. Base your classification ONLY on the provided email content.
3. Do NOT invent facts, deadlines, or details that are not present in the
   email.  Use null for the deadline field when no deadline is mentioned.
4. Keep the summary concise (one or two sentences) and strictly factual.
5. Distinguish between emails that are "important" (high priority, worth
   reading soon) and emails that "require action" (the recipient must DO
   something).  Set action_required to true ONLY when the recipient must
   take a concrete action.
6. If you are uncertain about the category, use category="other" and set
   confidence to a lower value (below 0.6).
7. confidence must be a float between 0.0 and 1.0 reflecting how certain
   you are about the classification.
"""


def build_user_prompt(
    sender: str,
    subject: str,
    body: str,
    timestamp: str,
    attachments_summary: str,
    *,
    max_body_chars: int = 8000,
) -> str:
    """Build the user-facing prompt that presents one email to the LLM.

    Args:
        sender: The ``From`` header value.
        subject: Decoded subject line.
        body: Cleaned email body text.
        timestamp: Email timestamp as an ISO 8601 string.
        attachments_summary: Human-readable summary of attachments, or
            an empty string when there are no attachments.
        max_body_chars: Maximum characters to include from the body.
            Longer bodies are truncated with a notice.

    Returns:
        A formatted string ready to be sent as the user message.
    """
    # Truncate overly long bodies to stay within context limits
    if len(body) > max_body_chars:
        body = body[:max_body_chars] + "\n\n[... body truncated ...]"

    parts: list[str] = [
        "Classify the following email.\n",
        f"From: {sender}",
        f"Subject: {subject}",
        f"Date: {timestamp}",
    ]

    if attachments_summary:
        parts.append(f"Attachments: {attachments_summary}")

    parts.append(f"\n--- EMAIL BODY ---\n{body}\n--- END EMAIL BODY ---")

    return "\n".join(parts)

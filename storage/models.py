"""Data models for the mail sorter application.

These Pydantic models define the normalized internal representations used
throughout the pipeline. They are independent of Gmail API specifics and
LLM provider details.
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class AttachmentInfo(BaseModel):
    """Metadata about an email attachment.

    In the initial version, only metadata is tracked — attachment content
    is not downloaded or analyzed. The structure supports future extension
    for document analysis.
    """

    filename: str = ""
    mime_type: str = ""
    size: int = 0
    attachment_id: Optional[str] = None


class EmailMessage(BaseModel):
    """Normalized representation of an email message.

    This is the canonical internal format passed through the pipeline.
    It is constructed by the parser/cleaner from raw Gmail API data
    and consumed by the classifier and decision engine.
    """

    id: str = Field(..., description="Gmail message ID")
    thread_id: str = Field("", description="Gmail thread ID")
    sender: str = Field("", description="From header value")
    recipients: list[str] = Field(default_factory=list, description="To addresses")
    cc: list[str] = Field(default_factory=list, description="CC addresses")
    bcc: list[str] = Field(default_factory=list, description="BCC addresses")
    subject: str = Field("", description="Decoded subject line")
    timestamp: Optional[datetime] = Field(None, description="Email date/time")
    labels: list[str] = Field(default_factory=list, description="Current Gmail labels")
    clean_body: str = Field("", description="Cleaned body text for LLM")
    raw_body: str = Field("", description="Original body text before cleaning")
    has_html: bool = Field(False, description="Whether the email had HTML content")
    attachments: list[AttachmentInfo] = Field(
        default_factory=list, description="Attachment metadata"
    )
    snippet: str = Field("", description="Gmail snippet preview")

    @property
    def raw_body_available(self) -> bool:
        """Whether raw body content is available."""
        return bool(self.raw_body)


class ProcessingStatus(str, enum.Enum):
    """Processing pipeline state machine.

    Emails progress through these states linearly.
    FAILED is a terminal state that allows retry.
    """

    DISCOVERED = "discovered"
    PARSED = "parsed"
    CLASSIFIED = "classified"
    VALIDATED = "validated"
    ACTIONED = "actioned"
    COMPLETE = "complete"
    FAILED = "failed"


class ProcessingRecord(BaseModel):
    """Persistent record of email processing state and results.

    Stored in the database to track what has been processed,
    prevent duplicates, and support retry on failure.
    """

    message_id: str = Field(..., description="Gmail message ID")
    thread_id: str = Field("", description="Gmail thread ID")
    processed_at: Optional[datetime] = Field(None, description="When processing completed")
    category: Optional[str] = None
    priority: Optional[str] = None
    urgency: Optional[str] = None
    action_required: Optional[bool] = None
    deadline: Optional[str] = None
    summary: Optional[str] = None
    confidence: Optional[float] = None
    actions_taken: str = Field("", description="JSON list of actions performed")
    processing_status: ProcessingStatus = ProcessingStatus.DISCOVERED
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

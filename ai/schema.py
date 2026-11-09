"""AI classification schema and validation.

Defines the strict JSON schema that the LLM must conform to,
and provides validation logic to ensure LLM output is safe before
it reaches the decision engine.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Category(str, enum.Enum):
    """Allowed email categories."""

    COLLEGE = "college"
    WORK = "work"
    PERSONAL = "personal"
    FINANCE = "finance"
    SHOPPING = "shopping"
    SOCIAL = "social"
    NEWSLETTER = "newsletter"
    PROMOTION = "promotion"
    NOTIFICATION = "notification"
    SPAM = "spam"
    OTHER = "other"


class Priority(str, enum.Enum):
    """Allowed priority levels."""

    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class Urgency(str, enum.Enum):
    """Allowed urgency levels."""

    IMMEDIATE = "immediate"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class EmailClassification(BaseModel):
    """Validated classification output from the LLM.

    This model enforces the strict schema contract between the AI layer
    and the decision engine. No unvalidated LLM output passes through.
    """

    category: Category = Field(..., description="Email category")
    priority: Priority = Field(..., description="Priority level")
    urgency: Urgency = Field(..., description="Urgency level")
    action_required: bool = Field(..., description="Whether user action is needed")
    deadline: Optional[str] = Field(
        None, description="Deadline in ISO 8601 format, or null"
    )
    summary: str = Field(..., description="Concise factual summary")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Classification confidence 0-1"
    )

    @field_validator("deadline")
    @classmethod
    def validate_deadline(cls, v: Optional[str]) -> Optional[str]:
        """Validate that deadline is a valid ISO 8601 string if provided."""
        if v is None or v == "" or v == "null":
            return None
        try:
            # Accept various ISO 8601 formats
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            raise ValueError(
                f"Deadline must be in ISO 8601 format or null, got: {v!r}"
            )
        return v

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, v: str) -> str:
        """Ensure summary is not empty."""
        if not v or not v.strip():
            raise ValueError("Summary must not be empty")
        return v.strip()


# JSON schema description for inclusion in LLM prompts
CLASSIFICATION_SCHEMA_DESCRIPTION: str = """\
{
  "category": "college|work|personal|finance|shopping|social|newsletter|promotion|notification|spam|other",
  "priority": "critical|high|normal|low",
  "urgency": "immediate|high|normal|low",
  "action_required": true|false,
  "deadline": "ISO 8601 datetime string or null",
  "summary": "concise factual summary of the email",
  "confidence": 0.0 to 1.0
}"""


def validate_classification(data: dict) -> EmailClassification:
    """Parse and validate a raw dictionary into an EmailClassification.

    Args:
        data: Dictionary parsed from LLM JSON response.

    Returns:
        Validated EmailClassification instance.

    Raises:
        pydantic.ValidationError: If the data does not conform to the schema.
    """
    return EmailClassification.model_validate(data)

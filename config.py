"""Centralized configuration management.

All configuration is loaded from environment variables with sensible defaults.
Use a .env file for local development (loaded automatically if python-dotenv is available).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Attempt to load .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _bool_env(key: str, default: bool = False) -> bool:
    """Parse a boolean from an environment variable."""
    val = os.getenv(key, str(default)).strip().lower()
    return val in ("1", "true", "yes", "on")


def _int_env(key: str, default: int) -> int:
    """Parse an integer from an environment variable."""
    val = os.getenv(key, str(default)).strip()
    return int(val)


def _float_env(key: str, default: float) -> float:
    """Parse a float from an environment variable."""
    val = os.getenv(key, str(default)).strip()
    return float(val)


@dataclass(frozen=True)
class GmailConfig:
    """Gmail API and OAuth2 configuration."""

    credentials_file: str = field(
        default_factory=lambda: os.getenv("GMAIL_CREDENTIALS_FILE", "credentials.json")
    )
    token_file: str = field(
        default_factory=lambda: os.getenv("GMAIL_TOKEN_FILE", "token.json")
    )
    scopes: list[str] = field(default_factory=lambda: [
        "https://www.googleapis.com/auth/gmail.modify",
        "https://www.googleapis.com/auth/gmail.labels",
    ])
    user_id: str = "me"
    max_results_per_fetch: int = field(
        default_factory=lambda: _int_env("MAX_EMAILS_PER_CYCLE", 25)
    )


@dataclass(frozen=True)
class LLMConfig:
    """LLM provider configuration."""

    provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "google")
    )
    model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "gemini-2.0-flash")
    )
    api_key: str = field(
        default_factory=lambda: os.getenv("LLM_API_KEY", "")
    )
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    max_retries: int = field(
        default_factory=lambda: _int_env("MAX_LLM_RETRIES", 3)
    )
    timeout_seconds: int = field(
        default_factory=lambda: _int_env("LLM_TIMEOUT_SECONDS", 60)
    )
    max_body_chars: int = field(
        default_factory=lambda: _int_env("LLM_MAX_BODY_CHARS", 8000)
    )
    temperature: float = field(
        default_factory=lambda: _float_env("LLM_TEMPERATURE", 0.1)
    )


@dataclass(frozen=True)
class DatabaseConfig:
    """SQLite database configuration."""

    path: str = field(
        default_factory=lambda: os.getenv("DATABASE_PATH", "data/mail_sorter.db")
    )


@dataclass(frozen=True)
class LabelConfig:
    """Gmail label name configuration.

    All AI-managed labels are prefixed to avoid collision with user labels.
    """

    prefix: str = field(
        default_factory=lambda: os.getenv("LABEL_PREFIX", "AI")
    )

    @property
    def college(self) -> str:
        return f"{self.prefix}/COLLEGE"

    @property
    def college_high_priority(self) -> str:
        return f"{self.prefix}/COLLEGE/HIGH_PRIORITY"

    @property
    def action_required(self) -> str:
        return f"{self.prefix}/ACTION_REQUIRED"

    @property
    def newsletter(self) -> str:
        return f"{self.prefix}/NEWSLETTER"

    @property
    def promotion(self) -> str:
        return f"{self.prefix}/PROMOTION"

    @property
    def spam_review(self) -> str:
        return f"{self.prefix}/SPAM_REVIEW"

    @property
    def other(self) -> str:
        return f"{self.prefix}/OTHER"

    @property
    def work(self) -> str:
        return f"{self.prefix}/WORK"

    @property
    def personal(self) -> str:
        return f"{self.prefix}/PERSONAL"

    @property
    def finance(self) -> str:
        return f"{self.prefix}/FINANCE"

    @property
    def shopping(self) -> str:
        return f"{self.prefix}/SHOPPING"

    @property
    def social(self) -> str:
        return f"{self.prefix}/SOCIAL"

    @property
    def notification(self) -> str:
        return f"{self.prefix}/NOTIFICATION"

    def all_labels(self) -> list[str]:
        """Return all managed label names."""
        return [
            self.college,
            self.college_high_priority,
            self.action_required,
            self.newsletter,
            self.promotion,
            self.spam_review,
            self.other,
            self.work,
            self.personal,
            self.finance,
            self.shopping,
            self.social,
            self.notification,
        ]


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    poll_interval_seconds: int = field(
        default_factory=lambda: _int_env("POLL_INTERVAL_SECONDS", 300)
    )
    initial_lookback_hours: int = field(
        default_factory=lambda: _int_env("INITIAL_LOOKBACK_HOURS", 0)
    )
    dry_run: bool = field(
        default_factory=lambda: _bool_env("DRY_RUN", True)
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )
    log_file: Optional[str] = field(
        default_factory=lambda: os.getenv("LOG_FILE", "data/mail_sorter.log")
    )

    gmail: GmailConfig = field(default_factory=GmailConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)


def load_config() -> AppConfig:
    """Load and return the application configuration.

    Configuration is read from environment variables (with .env support).
    All values have sensible defaults for safe first-run behavior.
    """
    return AppConfig()

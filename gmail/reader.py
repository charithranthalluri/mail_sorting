"""Email retrieval from the Gmail API.

Provides :class:`GmailReader` for fetching message IDs and full message
payloads with automatic pagination, timestamp-based filtering, and
exponential-backoff retry on transient API errors.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Optional, TYPE_CHECKING

from googleapiclient.errors import HttpError

from config import GmailConfig
from utils.logger import get_logger

if TYPE_CHECKING:
    from googleapiclient.discovery import Resource  # type: ignore[attr-defined]

logger = get_logger(__name__)

# Retry configuration
_MAX_RETRIES: int = 3
_INITIAL_BACKOFF_SECONDS: float = 1.0
_BACKOFF_MULTIPLIER: float = 2.0
# HTTP status codes considered transient / retryable
_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503})


class GmailReaderError(Exception):
    """Raised when an unrecoverable Gmail API error occurs during reading."""


class GmailReader:
    """Fetches emails from a Gmail account via the Gmail API.

    Args:
        service: An authenticated Gmail API ``Resource``.
        config: Gmail configuration (user_id, max_results, etc.).
    """

    def __init__(self, service: "Resource", config: GmailConfig) -> None:
        self._service = service
        self._config = config

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @staticmethod
    def build_query(base_query: str = "is:unread", after_timestamp: Optional[datetime] = None) -> str:
        """Combine a base Gmail search query with an optional ``after:`` filter.

        Args:
            base_query: Gmail search operators (e.g. ``"is:unread"``).
            after_timestamp: If provided, only messages received after this
                time are matched.  Converted to a Unix epoch for the Gmail
                ``after:`` operator.

        Returns:
            The combined query string.
        """
        if after_timestamp is None:
            return base_query

        # Gmail 'after:' uses epoch seconds (UTC)
        if after_timestamp.tzinfo is None:
            after_timestamp = after_timestamp.replace(tzinfo=timezone.utc)
        epoch = int(after_timestamp.timestamp())
        return f"{base_query} after:{epoch}"

    # ------------------------------------------------------------------
    # Core retrieval methods
    # ------------------------------------------------------------------

    def fetch_new_message_ids(
        self,
        query: str = "is:unread",
        after_timestamp: Optional[datetime] = None,
    ) -> list[str]:
        """Return message IDs matching the given query criteria.

        Handles pagination transparently so the caller always receives the
        complete list of matching IDs (bounded by ``max_results_per_fetch``
        from the config).

        Args:
            query: Gmail search query string.
            after_timestamp: Optional lower-bound timestamp filter.

        Returns:
            A list of Gmail message ID strings.

        Raises:
            GmailReaderError: If an unrecoverable API error occurs.
        """
        full_query = self.build_query(query, after_timestamp)
        logger.info("Fetching message IDs with query: %s", full_query)

        message_ids: list[str] = []
        page_token: Optional[str] = None
        remaining = self._config.max_results_per_fetch

        while remaining > 0:
            page_size = min(remaining, 500)  # Gmail API cap per page
            try:
                response: dict[str, Any] = self._execute_with_retry(
                    self._service.users()
                    .messages()
                    .list(
                        userId=self._config.user_id,
                        q=full_query,
                        maxResults=page_size,
                        pageToken=page_token,
                    )
                )
            except HttpError as exc:
                logger.error(
                    "Failed to list messages (status %s): %s",
                    getattr(exc, "status_code", "?"),
                    exc,
                )
                raise GmailReaderError("Failed to list messages from Gmail.") from exc

            messages = response.get("messages", [])
            for msg in messages:
                message_ids.append(msg["id"])
            remaining -= len(messages)

            page_token = response.get("nextPageToken")
            if not page_token:
                break

        logger.info("Found %d message ID(s) matching query", len(message_ids))
        return message_ids

    def get_message(self, message_id: str) -> dict[str, Any]:
        """Retrieve the full payload of a single message.

        Args:
            message_id: The Gmail message ID.

        Returns:
            The raw Gmail API message resource (``format='full'``).

        Raises:
            GmailReaderError: If the message cannot be retrieved after retries.
        """
        logger.debug("Fetching full message id=%s", message_id)
        try:
            message: dict[str, Any] = self._execute_with_retry(
                self._service.users()
                .messages()
                .get(
                    userId=self._config.user_id,
                    id=message_id,
                    format="full",
                )
            )
            logger.debug("Successfully fetched message id=%s", message_id)
            return message
        except HttpError as exc:
            logger.error(
                "Failed to fetch message id=%s (status %s): %s",
                message_id,
                getattr(exc, "status_code", "?"),
                exc,
            )
            raise GmailReaderError(
                f"Failed to retrieve message {message_id}."
            ) from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_with_retry(self, request: Any) -> dict[str, Any]:
        """Execute a Gmail API request with exponential-backoff retry.

        Retries on transient HTTP errors (429, 5xx) up to ``_MAX_RETRIES``
        times.

        Args:
            request: A pending ``googleapiclient`` HTTP request object.

        Returns:
            The parsed JSON response body as a dict.

        Raises:
            HttpError: If all retry attempts are exhausted or the error is
                not retryable.
        """
        backoff = _INITIAL_BACKOFF_SECONDS

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return request.execute()  # type: ignore[no-any-return]
            except HttpError as exc:
                status = exc.status_code  # type: ignore[attr-defined]
                if status not in _RETRYABLE_STATUS_CODES or attempt == _MAX_RETRIES:
                    logger.warning(
                        "Non-retryable or final attempt error (status %s, attempt %d/%d)",
                        status,
                        attempt,
                        _MAX_RETRIES,
                    )
                    raise

                logger.warning(
                    "Retryable API error (status %s, attempt %d/%d). "
                    "Backing off %.1fs",
                    status,
                    attempt,
                    _MAX_RETRIES,
                    backoff,
                )
                time.sleep(backoff)
                backoff *= _BACKOFF_MULTIPLIER

        # Should never reach here, but satisfy the type checker
        raise GmailReaderError("Retry loop exited unexpectedly.")  # pragma: no cover

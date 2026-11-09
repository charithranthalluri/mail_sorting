"""Gmail action executor.

Provides :class:`GmailActionExecutor` for applying label changes, starring,
archiving, and read-state mutations to Gmail messages.  Supports a
``dry_run`` mode that logs intended actions without touching the API.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from googleapiclient.errors import HttpError

from config import GmailConfig, LabelConfig
from utils.logger import get_logger

if TYPE_CHECKING:
    from googleapiclient.discovery import Resource  # type: ignore[attr-defined]

logger = get_logger(__name__)

# Well-known Gmail system label IDs
_LABEL_STARRED: str = "STARRED"
_LABEL_UNREAD: str = "UNREAD"
_LABEL_INBOX: str = "INBOX"


class GmailActionError(Exception):
    """Raised for unrecoverable Gmail action failures."""


class GmailActionExecutor:
    """Execute label, star, archive, and read-state actions on Gmail messages.

    Args:
        service: An authenticated Gmail API ``Resource``.
        config: Gmail configuration (user_id, etc.).
        label_config: Label naming configuration.
        dry_run: When ``True``, actions are logged but **not** executed.
    """

    def __init__(
        self,
        service: "Resource",
        config: GmailConfig,
        label_config: LabelConfig,
        dry_run: bool = True,
    ) -> None:
        self._service = service
        self._config = config
        self._label_config = label_config
        self._dry_run = dry_run
        # Cache: label name -> label id
        self._label_cache: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Label management
    # ------------------------------------------------------------------

    def ensure_labels_exist(self, label_names: list[str]) -> dict[str, str]:
        """Ensure every label in *label_names* exists, creating missing ones.

        Results are cached internally so repeated calls are cheap.

        Args:
            label_names: List of human-readable Gmail label names.

        Returns:
            A mapping of ``label_name -> label_id`` for all requested labels.
        """
        # Seed cache from the API if empty
        if not self._label_cache:
            self._populate_label_cache()

        result: dict[str, str] = {}
        for name in label_names:
            if name in self._label_cache:
                result[name] = self._label_cache[name]
                continue

            # Label is missing — create it
            label_id = self._create_label(name)
            if label_id is not None:
                self._label_cache[name] = label_id
                result[name] = label_id
            else:
                logger.error("Could not create or find label: %s", name)

        logger.info(
            "Label check complete: %d requested, %d resolved",
            len(label_names),
            len(result),
        )
        return result

    # ------------------------------------------------------------------
    # Message actions
    # ------------------------------------------------------------------

    def add_label(self, message_id: str, label_id: str) -> bool:
        """Add a label to a message.

        Args:
            message_id: Gmail message ID.
            label_id: Label ID to add.

        Returns:
            ``True`` if the label was applied (or dry-run logged).
        """
        return self._modify_message(
            message_id,
            add_label_ids=[label_id],
            action_description=f"add_label({label_id})",
        )

    def remove_label(self, message_id: str, label_id: str) -> bool:
        """Remove a label from a message.

        Args:
            message_id: Gmail message ID.
            label_id: Label ID to remove.

        Returns:
            ``True`` if the label was removed (or dry-run logged).
        """
        return self._modify_message(
            message_id,
            remove_label_ids=[label_id],
            action_description=f"remove_label({label_id})",
        )

    def star_message(self, message_id: str) -> bool:
        """Star a message.

        Args:
            message_id: Gmail message ID.

        Returns:
            ``True`` on success (or dry-run logged).
        """
        return self._modify_message(
            message_id,
            add_label_ids=[_LABEL_STARRED],
            action_description="star",
        )

    def unstar_message(self, message_id: str) -> bool:
        """Remove the star from a message.

        Args:
            message_id: Gmail message ID.

        Returns:
            ``True`` on success (or dry-run logged).
        """
        return self._modify_message(
            message_id,
            remove_label_ids=[_LABEL_STARRED],
            action_description="unstar",
        )

    def archive_message(self, message_id: str) -> bool:
        """Archive a message (remove from Inbox).

        Args:
            message_id: Gmail message ID.

        Returns:
            ``True`` on success (or dry-run logged).
        """
        return self._modify_message(
            message_id,
            remove_label_ids=[_LABEL_INBOX],
            action_description="archive",
        )

    def mark_read(self, message_id: str) -> bool:
        """Mark a message as read.

        Args:
            message_id: Gmail message ID.

        Returns:
            ``True`` on success (or dry-run logged).
        """
        return self._modify_message(
            message_id,
            remove_label_ids=[_LABEL_UNREAD],
            action_description="mark_read",
        )

    def mark_unread(self, message_id: str) -> bool:
        """Mark a message as unread.

        Args:
            message_id: Gmail message ID.

        Returns:
            ``True`` on success (or dry-run logged).
        """
        return self._modify_message(
            message_id,
            add_label_ids=[_LABEL_UNREAD],
            action_description="mark_unread",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _modify_message(
        self,
        message_id: str,
        add_label_ids: list[str] | None = None,
        remove_label_ids: list[str] | None = None,
        action_description: str = "modify",
    ) -> bool:
        """Apply a label modification to a message.

        In dry-run mode the intended change is logged and ``True`` is returned
        without calling the API.

        Args:
            message_id: Gmail message ID.
            add_label_ids: Label IDs to add (may be ``None``).
            remove_label_ids: Label IDs to remove (may be ``None``).
            action_description: Human-readable action name for logging.

        Returns:
            ``True`` on success, ``False`` on API error.
        """
        body: dict[str, Any] = {}
        if add_label_ids:
            body["addLabelIds"] = add_label_ids
        if remove_label_ids:
            body["removeLabelIds"] = remove_label_ids

        if self._dry_run:
            logger.info(
                "[DRY RUN] Would %s on message_id=%s (body=%s)",
                action_description,
                message_id,
                body,
            )
            return True

        try:
            self._service.users().messages().modify(
                userId=self._config.user_id,
                id=message_id,
                body=body,
            ).execute()
            logger.info(
                "Action %s succeeded on message_id=%s",
                action_description,
                message_id,
            )
            return True
        except HttpError as exc:
            logger.error(
                "Action %s failed on message_id=%s (status %s): %s",
                action_description,
                message_id,
                getattr(exc, "status_code", "?"),
                exc,
            )
            return False

    def _populate_label_cache(self) -> None:
        """Fetch all labels from the API and populate the internal cache."""
        try:
            response = (
                self._service.users()
                .labels()
                .list(userId=self._config.user_id)
                .execute()
            )
            for label in response.get("labels", []):
                self._label_cache[label["name"]] = label["id"]
            logger.debug(
                "Populated label cache with %d labels", len(self._label_cache)
            )
        except HttpError as exc:
            logger.error("Failed to list labels: %s", exc)

    def _create_label(self, name: str) -> str | None:
        """Create a new Gmail label.

        Args:
            name: The display name for the label (may include ``/`` nesting).

        Returns:
            The new label's ID, or ``None`` on failure.
        """
        if self._dry_run:
            logger.info("[DRY RUN] Would create label: %s", name)
            # Return a placeholder so callers can continue
            return f"dry_run_{name}"

        body: dict[str, Any] = {
            "name": name,
            "labelListVisibility": "labelShow",
            "messageListVisibility": "show",
        }
        try:
            created = (
                self._service.users()
                .labels()
                .create(userId=self._config.user_id, body=body)
                .execute()
            )
            label_id: str = created["id"]
            logger.info("Created label '%s' with id=%s", name, label_id)
            return label_id
        except HttpError as exc:
            # If a 409 Conflict is returned the label was created concurrently
            if getattr(exc, "status_code", None) == 409:
                logger.info(
                    "Label '%s' already exists (409 conflict), refreshing cache",
                    name,
                )
                self._populate_label_cache()
                return self._label_cache.get(name)
            logger.error("Failed to create label '%s': %s", name, exc)
            return None

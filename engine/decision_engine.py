"""Deterministic decision engine.

Maps a validated ``EmailClassification`` to a list of concrete
``Action`` objects that should be applied to the email in Gmail.
No destructive actions (delete, send) are ever produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ai.schema import Category, EmailClassification, Priority, Urgency
from config import LabelConfig
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Action model
# ---------------------------------------------------------------------------

@dataclass
class Action:
    """A single mutation to apply to an email.

    Attributes:
        action_type: The kind of action — one of ``"add_label"``,
            ``"remove_label"``, ``"star"``, ``"unstar"``, ``"archive"``,
            ``"mark_read"``.
        label_name: Gmail label name (required for label actions,
            ``None`` otherwise).
        reason: Human-readable explanation of why the action was chosen.
    """

    action_type: str
    label_name: Optional[str]
    reason: str


class ActionList(list):
    """A list of Action objects with convenience properties for inspection."""

    @property
    def labels(self) -> list[str]:
        """All label names to be added."""
        return [
            a.label_name
            for a in self
            if a.action_type == "add_label" and a.label_name is not None
        ]

    @property
    def star(self) -> bool:
        """Whether a star action is included."""
        return any(a.action_type == "star" for a in self)


# ---------------------------------------------------------------------------
# Decision engine
# ---------------------------------------------------------------------------

class DecisionEngine:
    """Applies deterministic rules to turn a classification into actions.

    Args:
        label_config: The ``LabelConfig`` instance that defines
            label names for each category.
    """

    def __init__(self, label_config: LabelConfig) -> None:
        self._labels = label_config

    # -- public API ---------------------------------------------------------

    def decide(self, classification: EmailClassification) -> ActionList:
        """Decide which actions to take based on *classification*.

        Args:
            classification: Validated classification from the AI layer.

        Returns:
            An ordered list of ``Action`` objects to execute.
        """
        actions: ActionList = ActionList()

        # ---- Category-based labelling ------------------------------------
        category = classification.category
        _cat_label_map: dict[Category, tuple[str, str]] = {
            Category.COLLEGE: (self._labels.college, "Category is college"),
            Category.WORK: (self._labels.work, "Category is work"),
            Category.PERSONAL: (self._labels.personal, "Category is personal"),
            Category.FINANCE: (self._labels.finance, "Category is finance"),
            Category.SHOPPING: (self._labels.shopping, "Category is shopping"),
            Category.SOCIAL: (self._labels.social, "Category is social"),
            Category.NEWSLETTER: (self._labels.newsletter, "Category is newsletter"),
            Category.PROMOTION: (self._labels.promotion, "Category is promotion"),
            Category.NOTIFICATION: (self._labels.notification, "Category is notification"),
            Category.SPAM: (self._labels.spam_review, "Category is spam — flagged for review"),
            Category.OTHER: (self._labels.other, "Category is other"),
        }

        if category in _cat_label_map:
            label, reason = _cat_label_map[category]
            actions.append(Action(
                action_type="add_label",
                label_name=label,
                reason=reason,
            ))

        # College + high/critical priority → extra label
        if category == Category.COLLEGE and classification.priority in (
            Priority.HIGH,
            Priority.CRITICAL,
        ):
            actions.append(Action(
                action_type="add_label",
                label_name=self._labels.college_high_priority,
                reason="College email with high/critical priority",
            ))

        # ---- Action-based rules ------------------------------------------

        if classification.action_required:
            actions.append(Action(
                action_type="add_label",
                label_name=self._labels.action_required,
                reason="Email requires user action",
            ))

        if classification.urgency == Urgency.IMMEDIATE:
            actions.append(Action(
                action_type="star",
                label_name=None,
                reason="Urgency is immediate — starring for visibility",
            ))

        if classification.priority == Priority.CRITICAL:
            # Avoid duplicate star if urgency already triggered one
            if not any(a.action_type == "star" for a in actions):
                actions.append(Action(
                    action_type="star",
                    label_name=None,
                    reason="Priority is critical — starring for visibility",
                ))

        logger.debug(
            "DecisionEngine produced %d action(s) for category=%s, "
            "priority=%s, urgency=%s",
            len(actions),
            classification.category.value,
            classification.priority.value,
            classification.urgency.value,
        )

        return actions

    # -- dry-run reporting --------------------------------------------------

    def format_dry_run_report(
        self,
        email_subject: str,
        classification: EmailClassification,
        actions: list[Action],
    ) -> str:
        """Format a human-readable dry-run report.

        Args:
            email_subject: The subject line of the email being processed.
            classification: The AI classification result.
            actions: Actions the engine would take.

        Returns:
            A multi-line string suitable for logging.
        """
        lines: list[str] = [
            "=" * 60,
            "DRY-RUN REPORT",
            "=" * 60,
            f"Subject   : {email_subject}",
            f"Category  : {classification.category.value}",
            f"Priority  : {classification.priority.value}",
            f"Urgency   : {classification.urgency.value}",
            f"Action?   : {'Yes' if classification.action_required else 'No'}",
            f"Deadline  : {classification.deadline or 'None'}",
            f"Confidence: {classification.confidence:.2f}",
            f"Summary   : {classification.summary}",
            "-" * 60,
            "ACTIONS:",
        ]

        if not actions:
            lines.append("  (none)")
        else:
            for i, action in enumerate(actions, 1):
                label_part = f" [{action.label_name}]" if action.label_name else ""
                lines.append(
                    f"  {i}. {action.action_type}{label_part} — {action.reason}"
                )

        lines.append("=" * 60)
        return "\n".join(lines)

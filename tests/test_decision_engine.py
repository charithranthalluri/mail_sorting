"""Tests for the decision engine module.

Verifies that the DecisionEngine produces the correct labels and star
actions based on email classification results. Tests cover individual
rules and their combinations.
"""

from __future__ import annotations

import pytest

from ai.schema import Category, EmailClassification, Priority, Urgency
from config import LabelConfig
from engine.decision_engine import DecisionEngine


def _make_classification(
    category: Category = Category.OTHER,
    priority: Priority = Priority.NORMAL,
    urgency: Urgency = Urgency.NORMAL,
    action_required: bool = False,
    deadline: str | None = None,
    summary: str = "Test summary.",
    confidence: float = 0.85,
) -> EmailClassification:
    """Create an EmailClassification with sensible defaults."""
    return EmailClassification(
        category=category,
        priority=priority,
        urgency=urgency,
        action_required=action_required,
        deadline=deadline,
        summary=summary,
        confidence=confidence,
    )


@pytest.fixture
def label_config() -> LabelConfig:
    """Return a LabelConfig with default prefix."""
    return LabelConfig()


@pytest.fixture
def engine(label_config: LabelConfig) -> DecisionEngine:
    """Return a DecisionEngine instance."""
    return DecisionEngine(label_config)


class TestCollegeLabel:
    """Test college category label assignment."""

    def test_college_category_gets_college_label(
        self, engine: DecisionEngine, label_config: LabelConfig
    ) -> None:
        """College category should get the college label."""
        classification = _make_classification(category=Category.COLLEGE)
        result = engine.decide(classification)

        assert label_config.college in result.labels

    def test_college_high_priority_gets_both_labels(
        self, engine: DecisionEngine, label_config: LabelConfig
    ) -> None:
        """College + high priority should get both college and college high priority labels."""
        classification = _make_classification(
            category=Category.COLLEGE, priority=Priority.HIGH
        )
        result = engine.decide(classification)

        assert label_config.college in result.labels
        assert label_config.college_high_priority in result.labels


class TestActionRequired:
    """Test action_required flag label assignment."""

    def test_action_required_gets_label(
        self, engine: DecisionEngine, label_config: LabelConfig
    ) -> None:
        """action_required=True should get the ACTION_REQUIRED label."""
        classification = _make_classification(action_required=True)
        result = engine.decide(classification)

        assert label_config.action_required in result.labels

    def test_no_action_required_skips_label(
        self, engine: DecisionEngine, label_config: LabelConfig
    ) -> None:
        """action_required=False should not get the ACTION_REQUIRED label."""
        classification = _make_classification(action_required=False)
        result = engine.decide(classification)

        assert label_config.action_required not in result.labels


class TestUrgencyImmediate:
    """Test immediate urgency star action."""

    def test_immediate_urgency_gets_star(self, engine: DecisionEngine) -> None:
        """Immediate urgency should trigger a star action."""
        classification = _make_classification(urgency=Urgency.IMMEDIATE)
        result = engine.decide(classification)

        assert result.star is True


class TestNewsletter:
    """Test newsletter category label."""

    def test_newsletter_gets_label(
        self, engine: DecisionEngine, label_config: LabelConfig
    ) -> None:
        """Newsletter category should get the NEWSLETTER label."""
        classification = _make_classification(category=Category.NEWSLETTER)
        result = engine.decide(classification)

        assert label_config.newsletter in result.labels


class TestPromotion:
    """Test promotion category label."""

    def test_promotion_gets_label(
        self, engine: DecisionEngine, label_config: LabelConfig
    ) -> None:
        """Promotion category should get the PROMOTION label."""
        classification = _make_classification(category=Category.PROMOTION)
        result = engine.decide(classification)

        assert label_config.promotion in result.labels


class TestSpam:
    """Test spam category label."""

    def test_spam_gets_spam_review_label(
        self, engine: DecisionEngine, label_config: LabelConfig
    ) -> None:
        """Spam category should get the SPAM_REVIEW label."""
        classification = _make_classification(category=Category.SPAM)
        result = engine.decide(classification)

        assert label_config.spam_review in result.labels


class TestCriticalPriority:
    """Test critical priority star action."""

    def test_critical_priority_gets_star(self, engine: DecisionEngine) -> None:
        """Critical priority should trigger a star action."""
        classification = _make_classification(priority=Priority.CRITICAL)
        result = engine.decide(classification)

        assert result.star is True


class TestCombinedRules:
    """Test combinations of classification rules."""

    def test_college_action_required_immediate(
        self, engine: DecisionEngine, label_config: LabelConfig
    ) -> None:
        """College + action_required + immediate urgency should produce multiple labels and star."""
        classification = _make_classification(
            category=Category.COLLEGE,
            priority=Priority.HIGH,
            urgency=Urgency.IMMEDIATE,
            action_required=True,
        )
        result = engine.decide(classification)

        assert label_config.college in result.labels
        assert label_config.college_high_priority in result.labels
        assert label_config.action_required in result.labels
        assert result.star is True

    def test_work_critical_action_required(
        self, engine: DecisionEngine, label_config: LabelConfig
    ) -> None:
        """Work + critical priority + action_required should get work label, action label, and star."""
        classification = _make_classification(
            category=Category.WORK,
            priority=Priority.CRITICAL,
            action_required=True,
        )
        result = engine.decide(classification)

        assert label_config.work in result.labels
        assert label_config.action_required in result.labels
        assert result.star is True

    def test_low_priority_no_action_no_star(
        self, engine: DecisionEngine
    ) -> None:
        """Low priority + no action + low urgency should not star."""
        classification = _make_classification(
            category=Category.OTHER,
            priority=Priority.LOW,
            urgency=Urgency.LOW,
            action_required=False,
        )
        result = engine.decide(classification)

        assert result.star is False

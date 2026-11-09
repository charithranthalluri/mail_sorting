"""Tests for the AI classifier and validation layer.

Verifies that the validate_classification function correctly handles
valid data, malformed JSON, missing fields, invalid enum values,
out-of-range confidence, and invalid deadline formats.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from ai.classifier import (
    ClassificationError,
    GoogleGeminiClassifier,
    OfflineRuleClassifier,
    OllamaClassifier,
    get_classifier,
)
from ai.schema import (
    Category,
    EmailClassification,
    Priority,
    Urgency,
    validate_classification,
)
from config import LLMConfig
from storage.models import EmailMessage


def _valid_classification_data() -> dict:
    """Return a valid classification dictionary."""
    return {
        "category": "college",
        "priority": "high",
        "urgency": "normal",
        "action_required": True,
        "deadline": "2025-11-01T23:59:00",
        "summary": "Application deadline reminder from admissions office.",
        "confidence": 0.92,
    }


class TestValidClassification:
    """Test valid classification responses."""

    def test_valid_data_returns_classification(self) -> None:
        """Valid data should produce an EmailClassification."""
        data = _valid_classification_data()
        result = validate_classification(data)

        assert isinstance(result, EmailClassification)
        assert result.category == Category.COLLEGE
        assert result.priority == Priority.HIGH
        assert result.urgency == Urgency.NORMAL
        assert result.action_required is True
        assert result.deadline == "2025-11-01T23:59:00"
        assert result.summary == "Application deadline reminder from admissions office."
        assert result.confidence == 0.92

    def test_null_deadline_is_valid(self) -> None:
        """Null deadline should be accepted."""
        data = _valid_classification_data()
        data["deadline"] = None
        result = validate_classification(data)

        assert result.deadline is None

    def test_all_categories(self) -> None:
        """All valid category values should be accepted."""
        for cat in Category:
            data = _valid_classification_data()
            data["category"] = cat.value
            result = validate_classification(data)
            assert result.category == cat

    def test_confidence_boundary_zero(self) -> None:
        """Confidence of 0.0 should be valid."""
        data = _valid_classification_data()
        data["confidence"] = 0.0
        result = validate_classification(data)
        assert result.confidence == 0.0

    def test_confidence_boundary_one(self) -> None:
        """Confidence of 1.0 should be valid."""
        data = _valid_classification_data()
        data["confidence"] = 1.0
        result = validate_classification(data)
        assert result.confidence == 1.0


class TestMalformedJSON:
    """Test handling of malformed data (as if JSON parsing already happened but data is garbage)."""

    def test_non_dict_input_raises(self) -> None:
        """Non-dict input should raise ValidationError."""
        with pytest.raises(ValidationError):
            validate_classification("not a dictionary")  # type: ignore[arg-type]

    def test_empty_dict_raises(self) -> None:
        """Empty dict should raise ValidationError for missing required fields."""
        with pytest.raises(ValidationError):
            validate_classification({})

    def test_list_input_raises(self) -> None:
        """List input should raise ValidationError."""
        with pytest.raises(ValidationError):
            validate_classification([1, 2, 3])  # type: ignore[arg-type]


class TestMissingRequiredFields:
    """Test handling of missing required fields."""

    def test_missing_category(self) -> None:
        """Missing category should raise ValidationError."""
        data = _valid_classification_data()
        del data["category"]

        with pytest.raises(ValidationError) as exc_info:
            validate_classification(data)
        assert "category" in str(exc_info.value)

    def test_missing_summary(self) -> None:
        """Missing summary should raise ValidationError."""
        data = _valid_classification_data()
        del data["summary"]

        with pytest.raises(ValidationError) as exc_info:
            validate_classification(data)
        assert "summary" in str(exc_info.value)

    def test_missing_confidence(self) -> None:
        """Missing confidence should raise ValidationError."""
        data = _valid_classification_data()
        del data["confidence"]

        with pytest.raises(ValidationError) as exc_info:
            validate_classification(data)
        assert "confidence" in str(exc_info.value)

    def test_missing_action_required(self) -> None:
        """Missing action_required should raise ValidationError."""
        data = _valid_classification_data()
        del data["action_required"]

        with pytest.raises(ValidationError) as exc_info:
            validate_classification(data)
        assert "action_required" in str(exc_info.value)


class TestInvalidEnumValues:
    """Test handling of invalid enum values."""

    def test_invalid_category(self) -> None:
        """Invalid category value should raise ValidationError."""
        data = _valid_classification_data()
        data["category"] = "invalid_category"

        with pytest.raises(ValidationError) as exc_info:
            validate_classification(data)
        assert "category" in str(exc_info.value)

    def test_invalid_priority(self) -> None:
        """Invalid priority value should raise ValidationError."""
        data = _valid_classification_data()
        data["priority"] = "mega_high"

        with pytest.raises(ValidationError) as exc_info:
            validate_classification(data)
        assert "priority" in str(exc_info.value)

    def test_invalid_urgency(self) -> None:
        """Invalid urgency value should raise ValidationError."""
        data = _valid_classification_data()
        data["urgency"] = "yesterday"

        with pytest.raises(ValidationError) as exc_info:
            validate_classification(data)
        assert "urgency" in str(exc_info.value)


class TestInvalidConfidence:
    """Test handling of out-of-range confidence values."""

    def test_confidence_above_one(self) -> None:
        """Confidence > 1.0 should raise ValidationError."""
        data = _valid_classification_data()
        data["confidence"] = 1.5

        with pytest.raises(ValidationError) as exc_info:
            validate_classification(data)
        assert "confidence" in str(exc_info.value)

    def test_confidence_below_zero(self) -> None:
        """Confidence < 0.0 should raise ValidationError."""
        data = _valid_classification_data()
        data["confidence"] = -0.1

        with pytest.raises(ValidationError) as exc_info:
            validate_classification(data)
        assert "confidence" in str(exc_info.value)

    def test_confidence_non_numeric(self) -> None:
        """Non-numeric confidence should raise ValidationError."""
        data = _valid_classification_data()
        data["confidence"] = "high"

        with pytest.raises(ValidationError):
            validate_classification(data)


class TestInvalidDeadlineFormat:
    """Test handling of invalid deadline formats."""

    def test_random_string_deadline(self) -> None:
        """Non-ISO-8601 string deadline should raise ValidationError."""
        data = _valid_classification_data()
        data["deadline"] = "next Tuesday"

        with pytest.raises(ValidationError) as exc_info:
            validate_classification(data)
        assert "deadline" in str(exc_info.value).lower() or "ISO 8601" in str(exc_info.value)

    def test_numeric_deadline(self) -> None:
        """Numeric deadline should raise ValidationError."""
        data = _valid_classification_data()
        data["deadline"] = 12345

        with pytest.raises(ValidationError):
            validate_classification(data)

    def test_partial_date_deadline(self) -> None:
        """Partial date format should raise ValidationError."""
        data = _valid_classification_data()
        data["deadline"] = "2025-13-45"

        with pytest.raises(ValidationError):
            validate_classification(data)


class TestEmptySummary:
    """Test handling of empty summary."""

    def test_empty_string_summary(self) -> None:
        """Empty string summary should raise ValidationError."""
        data = _valid_classification_data()
        data["summary"] = ""

        with pytest.raises(ValidationError):
            validate_classification(data)

    def test_whitespace_only_summary(self) -> None:
        """Whitespace-only summary should raise ValidationError."""
        data = _valid_classification_data()
        data["summary"] = "   \n\t  "

        with pytest.raises(ValidationError):
            validate_classification(data)


class TestMockedLLMClassifier:
    """Test the full GoogleGeminiClassifier workflow with mocked LLM responses."""

    @pytest.fixture
    def test_email(self) -> EmailMessage:
        return EmailMessage(
            id="msg_test_001",
            thread_id="th_001",
            sender="prof@university.edu",
            recipients=["student@university.edu"],
            subject="Midterm Exam Schedule",
            clean_body="The midterm exam is scheduled for Friday at 10 AM.",
        )

    @pytest.fixture
    def classifier(self) -> GoogleGeminiClassifier:
        config = LLMConfig(api_key="fake-test-key", max_retries=2)
        with patch("google.generativeai.configure"), patch("google.generativeai.GenerativeModel"):
            return GoogleGeminiClassifier(config)

    def test_mocked_llm_valid_response(
        self, classifier: GoogleGeminiClassifier, test_email: EmailMessage
    ) -> None:
        """Classifier correctly parses and validates a valid JSON response from LLM."""
        mock_response = MagicMock()
        mock_response.text = (
            '{"category": "college", "priority": "high", "urgency": "normal", '
            '"action_required": false, "deadline": null, "summary": "Exam on Friday.", "confidence": 0.95}'
        )
        classifier._model.generate_content.return_value = mock_response

        result = classifier.classify(test_email)
        assert isinstance(result, EmailClassification)
        assert result.category == Category.COLLEGE
        assert result.priority == Priority.HIGH
        assert result.action_required is False
        assert result.summary == "Exam on Friday."

    def test_mocked_llm_markdown_fence_stripped(
        self, classifier: GoogleGeminiClassifier, test_email: EmailMessage
    ) -> None:
        """Classifier strips markdown code fences from LLM response."""
        mock_response = MagicMock()
        mock_response.text = (
            '```json\n{"category": "college", "priority": "normal", "urgency": "low", '
            '"action_required": false, "deadline": null, "summary": "Schedule update.", "confidence": 0.8}\n```'
        )
        classifier._model.generate_content.return_value = mock_response

        result = classifier.classify(test_email)
        assert result.category == Category.COLLEGE

    def test_mocked_llm_malformed_json_raises_after_retries(
        self, classifier: GoogleGeminiClassifier, test_email: EmailMessage
    ) -> None:
        """Classifier retries and raises ClassificationError if LLM returns malformed JSON."""
        mock_response = MagicMock()
        mock_response.text = "This is not valid JSON at all."
        classifier._model.generate_content.return_value = mock_response

        with patch("time.sleep"):  # Avoid test delay
            with pytest.raises(ClassificationError):
                classifier.classify(test_email)

    def test_mocked_llm_missing_fields_raises_after_retries(
        self, classifier: GoogleGeminiClassifier, test_email: EmailMessage
    ) -> None:
        """Classifier retries and raises ClassificationError when required fields are missing."""
        mock_response = MagicMock()
        mock_response.text = '{"category": "college"}'
        classifier._model.generate_content.return_value = mock_response

        with patch("time.sleep"):
            with pytest.raises(ClassificationError):
                classifier.classify(test_email)

    def test_mocked_llm_invalid_enum_raises_after_retries(
        self, classifier: GoogleGeminiClassifier, test_email: EmailMessage
    ) -> None:
        """Classifier raises ClassificationError when enum value is invalid."""
        mock_response = MagicMock()
        mock_response.text = (
            '{"category": "unsupported_cat", "priority": "high", "urgency": "low", '
            '"action_required": false, "deadline": null, "summary": "Test", "confidence": 0.8}'
        )
        classifier._model.generate_content.return_value = mock_response

        with patch("time.sleep"):
            with pytest.raises(ClassificationError):
                classifier.classify(test_email)

    def test_mocked_llm_invalid_confidence_raises_after_retries(
        self, classifier: GoogleGeminiClassifier, test_email: EmailMessage
    ) -> None:
        """Classifier raises ClassificationError when confidence is out of range."""
        mock_response = MagicMock()
        mock_response.text = (
            '{"category": "college", "priority": "high", "urgency": "low", '
            '"action_required": false, "deadline": null, "summary": "Test", "confidence": 5.0}'
        )
        classifier._model.generate_content.return_value = mock_response

        with patch("time.sleep"):
            with pytest.raises(ClassificationError):
                classifier.classify(test_email)

    def test_mocked_llm_invalid_deadline_raises_after_retries(
        self, classifier: GoogleGeminiClassifier, test_email: EmailMessage
    ) -> None:
        """Classifier raises ClassificationError when deadline is not ISO 8601."""
        mock_response = MagicMock()
        mock_response.text = (
            '{"category": "college", "priority": "high", "urgency": "low", '
            '"action_required": false, "deadline": "tomorrow evening", "summary": "Test", "confidence": 0.9}'
        )
        classifier._model.generate_content.return_value = mock_response

        with patch("time.sleep"):
            with pytest.raises(ClassificationError):
                classifier.classify(test_email)

    def test_missing_api_key_raises_error(self) -> None:
        """Initializing with missing API key should raise ClassificationError."""
        config = LLMConfig(api_key="")
        with pytest.raises(ClassificationError, match="LLM_API_KEY is not set"):
            GoogleGeminiClassifier(config)


class TestOllamaClassifier:
    """Test the local offline Ollama classifier with mocked HTTP responses."""

    @pytest.fixture
    def test_email(self) -> EmailMessage:
        return EmailMessage(
            id="msg_ollama_001",
            sender="prof@college.edu",
            subject="Class Cancelled",
            clean_body="Class is cancelled today.",
        )

    @patch("requests.post")
    def test_ollama_success(self, mock_post: MagicMock, test_email: EmailMessage) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {
                "content": (
                    '{"category": "college", "priority": "normal", "urgency": "low", '
                    '"action_required": false, "deadline": null, "summary": "Class cancelled.", "confidence": 0.9}'
                )
            }
        }
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        config = LLMConfig(provider="ollama", model="llama3.2")
        classifier = OllamaClassifier(config)
        res = classifier.classify(test_email)

        assert res.category == Category.COLLEGE
        assert res.priority == Priority.NORMAL
        assert res.action_required is False


class TestOfflineRuleClassifier:
    """Test the 100% offline rule-based semantic classifier."""

    @pytest.fixture
    def classifier(self) -> OfflineRuleClassifier:
        return OfflineRuleClassifier()

    def test_college_email(self, classifier: OfflineRuleClassifier) -> None:
        email = EmailMessage(
            id="m1",
            sender="admissions@university.edu",
            subject="Course registration deadline",
            clean_body="Please submit your course selection by tomorrow.",
        )
        res = classifier.classify(email)
        assert res.category == Category.COLLEGE
        assert res.action_required is True
        assert res.priority == Priority.HIGH

    def test_newsletter_email(self, classifier: OfflineRuleClassifier) -> None:
        email = EmailMessage(
            id="m2",
            sender="newsletter@techdigest.com",
            subject="Weekly Digest #42",
            clean_body="Here is your weekly summary. Unsubscribe here.",
        )
        res = classifier.classify(email)
        assert res.category == Category.NEWSLETTER
        assert res.priority == Priority.LOW

    def test_promotion_email(self, classifier: OfflineRuleClassifier) -> None:
        email = EmailMessage(
            id="m3",
            sender="store@deals.com",
            subject="Flash Sale: 50% Off Everything!",
            clean_body="Huge discounts this weekend only. Special promo code SAVE50.",
        )
        res = classifier.classify(email)
        assert res.category == Category.PROMOTION
        assert res.priority == Priority.LOW


class TestClassifierFactory:
    """Test get_classifier factory function."""

    def test_factory_returns_offline(self) -> None:
        config = LLMConfig(provider="offline")
        clf = get_classifier(config)
        assert isinstance(clf, OfflineRuleClassifier)

    def test_factory_returns_ollama(self) -> None:
        config = LLMConfig(provider="ollama")
        clf = get_classifier(config)
        assert isinstance(clf, OllamaClassifier)

    def test_factory_returns_google(self) -> None:
        config = LLMConfig(provider="google", api_key="fake-key")
        with patch("google.generativeai.configure"), patch("google.generativeai.GenerativeModel"):
            clf = get_classifier(config)
            assert isinstance(clf, GoogleGeminiClassifier)



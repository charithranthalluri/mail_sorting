"""LLM classifier with provider abstraction.

Contains the abstract ``BaseClassifier`` interface and the concrete
``GoogleGeminiClassifier`` implementation that uses the Gemini API for
email classification.
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any

from ai.prompts import SYSTEM_PROMPT, build_user_prompt
from ai.schema import EmailClassification, validate_classification
from config import LLMConfig
from processing.attachments import summarize_attachments
from storage.models import EmailMessage
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ClassificationError(Exception):
    """Raised when email classification fails after all retries."""


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseClassifier(ABC):
    """Abstract interface that all LLM classifiers must implement."""

    @abstractmethod
    def classify(self, email: EmailMessage) -> EmailClassification:
        """Classify an email message.

        Args:
            email: The normalized email to classify.

        Returns:
            A validated ``EmailClassification`` object.

        Raises:
            ClassificationError: If classification fails after retries.
        """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable name of the LLM provider."""


# ---------------------------------------------------------------------------
# Google Gemini implementation
# ---------------------------------------------------------------------------

class GoogleGeminiClassifier(BaseClassifier):
    """Classifier that uses the Google Gemini API.

    Args:
        config: LLM configuration including API key, model name,
            retry settings, and temperature.
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

        if not config.api_key:
            raise ClassificationError(
                "LLM_API_KEY is not set.  Please set it in your environment "
                "or .env file."
            )

        import google.generativeai as genai
        genai.configure(api_key=config.api_key)
        self._model = genai.GenerativeModel(
            model_name=config.model,
            system_instruction=SYSTEM_PROMPT,
            generation_config=genai.types.GenerationConfig(
                temperature=config.temperature,
            ),
        )

        logger.info(
            "GoogleGeminiClassifier initialised — model=%s, "
            "max_retries=%d, timeout=%ds",
            config.model,
            config.max_retries,
            config.timeout_seconds,
        )

    # -- public interface ---------------------------------------------------

    @property
    def provider_name(self) -> str:
        """Human-readable name of the LLM provider."""
        return f"Google Gemini ({self._config.model})"

    def classify(self, email: EmailMessage) -> EmailClassification:
        """Classify *email* via the Gemini API with retries.

        The method builds a prompt from the email, calls the API, parses
        the JSON response, validates it against the schema, and returns
        the result.  Up to ``max_retries`` attempts are made with
        exponential back-off on transient errors.

        Args:
            email: Normalised email to classify.

        Returns:
            Validated ``EmailClassification``.

        Raises:
            ClassificationError: When all attempts are exhausted.
        """
        user_prompt = self._build_prompt(email)
        last_error: Exception | None = None

        for attempt in range(1, self._config.max_retries + 1):
            try:
                raw_text = self._call_api(user_prompt)
                parsed = self._parse_json(raw_text)
                classification = validate_classification(parsed)
                logger.info(
                    "Classification succeeded on attempt %d for message_id=%s "
                    "— category=%s, priority=%s",
                    attempt,
                    email.id,
                    classification.category.value,
                    classification.priority.value,
                )
                return classification

            except json.JSONDecodeError as exc:
                last_error = exc
                logger.warning(
                    "Malformed JSON on attempt %d for message_id=%s: %s",
                    attempt, email.id, exc,
                )
            except Exception as exc:  # noqa: BLE001 — broad but logged & retried
                last_error = exc
                logger.warning(
                    "Classification attempt %d failed for message_id=%s: %s",
                    attempt, email.id, exc,
                )

            # Exponential back-off: 2s, 4s, 8s, …
            if attempt < self._config.max_retries:
                backoff = 2 ** attempt
                logger.debug("Retrying in %ds …", backoff)
                time.sleep(backoff)

        raise ClassificationError(
            f"Classification failed after {self._config.max_retries} attempts "
            f"for message_id={email.id}: {last_error}"
        )

    # -- private helpers ----------------------------------------------------

    def _build_prompt(self, email: EmailMessage) -> str:
        """Construct the user prompt from an ``EmailMessage``."""
        timestamp_str = (
            email.timestamp.isoformat() if email.timestamp else "unknown"
        )
        attachments_summary = summarize_attachments(email.attachments)

        return build_user_prompt(
            sender=email.sender,
            subject=email.subject,
            body=email.clean_body,
            timestamp=timestamp_str,
            attachments_summary=attachments_summary,
            max_body_chars=self._config.max_body_chars,
        )

    def _call_api(self, user_prompt: str) -> str:
        """Send the prompt to Gemini and return the raw text response.

        Raises on API errors so the caller can retry.
        """
        response = self._model.generate_content(
            user_prompt,
            request_options={"timeout": self._config.timeout_seconds},
        )

        if not response.text:
            raise ClassificationError(
                "Gemini returned an empty response (possibly blocked)."
            )

        return response.text

    @staticmethod
    def _parse_json(raw_text: str) -> dict[str, Any]:
        """Parse JSON from the LLM response, stripping markdown fences and thinking blocks."""
        text = raw_text.strip()

        # Strip <think>...</think> reasoning blocks (common in Qwen/DeepSeek thinking models)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        # Strip ```json … ``` or ``` … ``` wrappers
        fence_pattern = re.compile(
            r"^```(?:json)?\s*\n?(.*?)\n?\s*```$", re.DOTALL
        )
        match = fence_pattern.match(text)
        if match:
            text = match.group(1).strip()

        # If there's any surrounding text, extract between first '{' and last '}'
        if not text.startswith("{") or not text.endswith("}"):
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                text = text[start : end + 1]

        return json.loads(text)


# ---------------------------------------------------------------------------
# Offline Local Ollama Implementation
# ---------------------------------------------------------------------------

class OllamaClassifier(BaseClassifier):
    """Classifier that uses a locally running Ollama instance (100% offline, zero cloud API keys).

    Args:
        config: LLM configuration containing ollama_base_url, model name, etc.
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._base_url = getattr(config, "ollama_base_url", "http://localhost:11434").rstrip("/")
        # If model is default gemini, use llama3.2 as default local model
        self._model_name = (
            config.model
            if config.model and "gemini" not in config.model
            else "llama3.2"
        )
        logger.info(
            "OllamaClassifier initialized — model=%s, base_url=%s, timeout=%ds",
            self._model_name,
            self._base_url,
            config.timeout_seconds,
        )

    @property
    def provider_name(self) -> str:
        return f"Ollama ({self._model_name})"

    # ------ Schema Healing ------
    # Valid field names that belong in the EmailClassification schema
    _SCHEMA_FIELDS = frozenset({
        "category", "priority", "urgency", "action_required",
        "deadline", "summary", "confidence",
    })
    # Map common LLM text values to valid category enum values
    _CATEGORY_ALIASES = {
        "promotional": "promotion", "promo": "promotion", "marketing": "promotion",
        "academic": "college", "school": "college", "university": "college",
        "education": "college", "edu": "college",
        "job": "work", "professional": "work", "office": "work", "business": "work",
        "banking": "finance", "financial": "finance", "payment": "finance",
        "order": "shopping", "purchase": "shopping", "ecommerce": "shopping",
        "social_media": "social", "social media": "social",
        "news": "newsletter", "update": "notification", "alert": "notification",
        "junk": "spam", "phishing": "spam",
    }

    def _heal_schema(self, parsed: dict, email: "EmailMessage") -> dict:
        """Aggressively heal malformed LLM JSON output into valid schema fields.

        Handles: reasoning text as keys, missing fields, bad enum values,
        nested classification dicts, and non-null deadline placeholders.
        """
        # Step 1: If the dict has non-schema keys that map to dicts with
        #   schema fields inside, flatten them up (LLM wrapped its answer).
        for key in list(parsed.keys()):
            if key not in self._SCHEMA_FIELDS and isinstance(parsed[key], dict):
                inner = parsed[key]
                if any(f in inner for f in self._SCHEMA_FIELDS):
                    # Merge inner fields into top-level, inner wins
                    for f, v in inner.items():
                        if f in self._SCHEMA_FIELDS:
                            parsed[f] = v
                    del parsed[key]

        # Step 2: Strip any remaining non-schema keys (reasoning text keys)
        for key in list(parsed.keys()):
            if key not in self._SCHEMA_FIELDS:
                del parsed[key]

        # Step 3: Infer missing 'category' from reasoning text in original keys
        #   or from the email subject
        if "category" not in parsed or not parsed["category"]:
            # Try to guess from email content
            text = (email.subject or "").lower() + " " + (email.body_text or "")[:500].lower()
            guessed = "other"
            kw_map = {
                "college": ("assignment", "professor", "syllabus", "grade", "exam",
                            "semester", "lecture", "course", "campus", "university"),
                "finance": ("invoice", "payment", "bank", "transaction", "receipt",
                            "balance", "credit", "debit", "billing", "statement"),
                "promotion": ("sale", "discount", "offer", "deal", "promo", "off",
                              "coupon", "limited time", "subscribe", "unsubscribe"),
                "notification": ("alert", "verify", "confirm", "otp", "code",
                                 "security", "sign-in", "login", "password"),
                "shopping": ("order", "shipped", "delivered", "tracking", "return"),
                "newsletter": ("newsletter", "digest", "weekly", "monthly"),
                "social": ("friend", "follow", "liked", "commented", "tagged"),
            }
            for cat, keywords in kw_map.items():
                if any(kw in text for kw in keywords):
                    guessed = cat
                    break
            parsed["category"] = guessed

        # Normalize category aliases
        cat_val = str(parsed.get("category", "other")).lower().strip()
        parsed["category"] = self._CATEGORY_ALIASES.get(cat_val, cat_val)

        # Step 4: Fill missing 'priority'
        if "priority" not in parsed or not parsed["priority"]:
            cat = parsed.get("category", "other")
            if cat in ("college", "work", "finance"):
                parsed["priority"] = "high"
            elif cat in ("promotion", "newsletter", "spam"):
                parsed["priority"] = "low"
            else:
                parsed["priority"] = "normal"

        # Step 5: Fill missing 'action_required'
        if "action_required" not in parsed:
            text = ((email.subject or "") + " " + (email.body_text or "")[:500]).lower()
            parsed["action_required"] = any(kw in text for kw in (
                "please submit", "action required", "respond by", "deadline",
                "due by", "rsvp", "please reply", "please confirm",
            ))

        # Step 6: Fill missing 'urgency' (infer from priority)
        if "urgency" not in parsed or not parsed["urgency"]:
            priority_val = str(parsed.get("priority", "normal")).lower()
            urgency_map = {"critical": "immediate", "high": "high", "normal": "normal", "low": "low"}
            parsed["urgency"] = urgency_map.get(priority_val, "normal")

        # Step 7: Fix deadline placeholders → null
        deadline_val = parsed.get("deadline")
        if isinstance(deadline_val, str) and deadline_val.strip().lower() in (
            "n/a", "na", "none", "null", "no", "no deadline", "-", "",
            "not applicable", "no deadline mentioned", "not specified",
        ):
            parsed["deadline"] = None

        # Step 8: Fill missing 'confidence' and 'summary'
        if "confidence" not in parsed or parsed["confidence"] is None:
            parsed["confidence"] = 0.85
        if "summary" not in parsed or not str(parsed.get("summary", "")).strip():
            parsed["summary"] = f"Email regarding: {email.subject}" if email.subject else "General email"

        return parsed

    def classify(self, email: EmailMessage) -> EmailClassification:
        user_prompt = self._build_prompt(email)
        last_error: Exception | None = None

        for attempt in range(1, self._config.max_retries + 1):
            try:
                raw_text = self._call_api(user_prompt)
                parsed = GoogleGeminiClassifier._parse_json(raw_text)
                if isinstance(parsed, dict):
                    parsed = self._heal_schema(parsed, email)
                classification = validate_classification(parsed)
                logger.info(
                    "Ollama classification succeeded on attempt %d for message_id=%s "
                    "— category=%s, priority=%s",
                    attempt,
                    email.id,
                    classification.category.value,
                    classification.priority.value,
                )
                return classification
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Ollama classification attempt %d failed for message_id=%s: %s",
                    attempt,
                    email.id,
                    exc,
                )

            if attempt < self._config.max_retries:
                backoff = 2 ** attempt
                logger.debug("Retrying Ollama in %ds...", backoff)
                time.sleep(backoff)

        raise ClassificationError(
            f"Ollama classification failed after {self._config.max_retries} attempts "
            f"for message_id={email.id}: {last_error}"
        )

    def _build_prompt(self, email: EmailMessage) -> str:
        timestamp_str = (
            email.timestamp.isoformat() if email.timestamp else "unknown"
        )
        attachments_summary = summarize_attachments(email.attachments)
        return build_user_prompt(
            sender=email.sender,
            subject=email.subject,
            body=email.clean_body,
            timestamp=timestamp_str,
            attachments_summary=attachments_summary,
            max_body_chars=self._config.max_body_chars,
        )

    def _call_api(self, user_prompt: str) -> str:
        import requests

        url = f"{self._base_url}/api/chat"
        payload = {
            "model": self._model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "format": "json",
            "stream": False,
            "options": {"temperature": self._config.temperature},
        }

        try:
            resp = requests.post(url, json=payload, timeout=self._config.timeout_seconds)
            if resp.status_code == 404:
                error_msg = resp.json().get("error", "Model not found") if resp.text else "Model not found"
                raise ClassificationError(
                    f"Ollama model '{self._model_name}' not found: {error_msg}. "
                    f"Check 'ollama list' to confirm model name."
                )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "")
            if not content:
                raise ClassificationError("Ollama returned an empty response")
            return content
        except requests.ConnectionError as ce:
            raise ClassificationError(
                f"Could not connect to Ollama at {self._base_url}. "
                "Ensure Ollama is running ('ollama serve' or start Ollama app)."
            ) from ce
        except Exception as e:
            raise ClassificationError(f"Ollama request error: {e}") from e


# ---------------------------------------------------------------------------
# 100% Offline Rule-Based Classifier (Zero Network, Zero Model Download)
# ---------------------------------------------------------------------------

class OfflineRuleClassifier(BaseClassifier):
    """100% offline semantic rule classifier.

    Requires no GPU, no local LLM installation, no cloud API key, and runs instantly.
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self._config = config
        logger.info("OfflineRuleClassifier initialized (100% offline rule-based)")

    @property
    def provider_name(self) -> str:
        return "Offline Rules Engine (Zero Cloud / 100% Local)"

    def classify(self, email: EmailMessage) -> EmailClassification:
        from ai.schema import Category, Priority, Urgency

        text = f"{email.subject} {email.sender} {email.clean_body}".lower()

        # Category determination
        category = Category.OTHER
        confidence = 0.85

        if any(k in text for k in ("university", "college", "professor", "exam", "assignment", "syllabus", "course", "campus", ".edu", "grade", "homework")):
            category = Category.COLLEGE
        elif any(k in text for k in ("unsubscribe", "newsletter", "weekly digest", "issue #", "substack", "bulletin")):
            category = Category.NEWSLETTER
        elif any(k in text for k in ("% off", "discount", "sale", "promo", "deal", "coupon", "clearance", "black friday", "special offer")):
            category = Category.PROMOTION
        elif any(k in text for k in ("order #", "shipped", "tracking", "amazon", "delivery", "delivered", "package", "receipt")):
            category = Category.SHOPPING
        elif any(k in text for k in ("bank", "statement", "payment due", "credit card", "wire transfer", "transaction", "balance", "payroll")):
            category = Category.FINANCE
        elif any(k in text for k in ("linkedin", "invitation to connect", "facebook", "twitter", "instagram", "tagged you in")):
            category = Category.SOCIAL
        elif any(k in text for k in ("meeting", "sync", "standup", "sprint", "jira", "github", "pull request", "client", "quarterly")):
            category = Category.WORK
        elif any(k in text for k in ("lottery", "congratulations you won", "million dollars", "claim your prize", "bitcoin investment")):
            category = Category.SPAM
        elif any(k in text for k in ("alert", "security notification", "verification code", "one-time passcode", "otp", "login attempt")):
            category = Category.NOTIFICATION
        else:
            category = Category.OTHER
            confidence = 0.65

        # Action required determination
        action_required = any(
            k in text for k in (
                "please submit", "action required", "respond by", "deadline", "due by",
                "rsvp", "please reply", "please confirm", "needs your approval", "fill out", "action item"
            )
        )

        # Priority and Urgency
        if any(k in text for k in ("urgent", "immediate action", "asap", "emergency", "critical")):
            priority = Priority.CRITICAL
            urgency = Urgency.IMMEDIATE
        elif action_required or category in (Category.COLLEGE, Category.WORK, Category.FINANCE):
            priority = Priority.HIGH
            urgency = Urgency.HIGH if any(w in text for w in ("today", "tomorrow", "tonight", "due")) else Urgency.NORMAL
        elif category in (Category.PROMOTION, Category.NEWSLETTER):
            priority = Priority.LOW
            urgency = Urgency.LOW
        else:
            priority = Priority.NORMAL
            urgency = Urgency.NORMAL

        summary = f"{category.value.capitalize()} email regarding: {email.subject[:80] if email.subject else 'general update'}."

        return EmailClassification(
            category=category,
            priority=priority,
            urgency=urgency,
            action_required=action_required,
            deadline=None,
            summary=summary,
            confidence=confidence,
        )


# ---------------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------------

def get_classifier(config: LLMConfig) -> BaseClassifier:
    """Instantiate and return the appropriate classifier based on configuration.

    Supports:
    - "google": Google Gemini API (free tier)
    - "ollama": Local Ollama instance (100% offline, local LLM)
    - "offline" / "local" / "rule": 100% offline rule-based semantic classifier
    """
    provider = (config.provider or "").strip().lower()
    if provider in ("ollama", "local_llm"):
        return OllamaClassifier(config)
    elif provider in ("offline", "local", "rule", "rules", "rule_based"):
        return OfflineRuleClassifier(config)
    elif provider == "google":
        return GoogleGeminiClassifier(config)
    else:
        logger.warning(
            "Unknown LLM_PROVIDER '%s', falling back to OfflineRuleClassifier",
            config.provider,
        )
        return OfflineRuleClassifier(config)

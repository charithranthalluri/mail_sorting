"""Unit tests for the prompt builder module."""

from __future__ import annotations

from ai.prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_user_prompt


def test_prompt_version() -> None:
    assert PROMPT_VERSION == "1.0.0"


def test_system_prompt_contains_rules() -> None:
    assert "REQUIRED JSON SCHEMA" in SYSTEM_PROMPT
    assert "college" in SYSTEM_PROMPT
    assert "critical" in SYSTEM_PROMPT
    assert "action_required" in SYSTEM_PROMPT


def test_build_user_prompt() -> None:
    prompt = build_user_prompt(
        sender="registrar@university.edu",
        subject="Fall Registration",
        body="Registration opens tomorrow.",
        timestamp="2025-08-01T09:00:00Z",
        attachments_summary="1 attachment: schedule.pdf (application/pdf, 50KB)",
    )
    assert "From: registrar@university.edu" in prompt
    assert "Subject: Fall Registration" in prompt
    assert "Attachments: 1 attachment: schedule.pdf" in prompt
    assert "Registration opens tomorrow." in prompt


def test_build_user_prompt_truncation() -> None:
    long_body = "x" * 10000
    prompt = build_user_prompt(
        sender="test@test.com",
        subject="Long Email",
        body=long_body,
        timestamp="2025-01-01",
        attachments_summary="",
        max_body_chars=500,
    )
    assert "[... body truncated ...]" in prompt
    assert len(prompt) < 1500

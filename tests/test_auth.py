"""Unit tests for the Gmail authentication module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from google.auth.exceptions import RefreshError

from config import GmailConfig
from gmail.auth import (
    AuthenticationError,
    _load_token,
    _refresh_token,
    _run_oauth_flow,
    _save_token,
    get_gmail_service,
)


class TestLoadToken:
    def test_missing_token_returns_none(self, tmp_path: Path) -> None:
        token_path = tmp_path / "nonexistent_token.json"
        assert _load_token(token_path) is None

    @patch("gmail.auth.Credentials.from_authorized_user_file")
    def test_valid_token_loaded(
        self, mock_from_file: MagicMock, tmp_path: Path
    ) -> None:
        token_path = tmp_path / "token.json"
        token_path.write_text("{}", encoding="utf-8")
        mock_creds = MagicMock()
        mock_from_file.return_value = mock_creds

        loaded = _load_token(token_path)
        assert loaded == mock_creds


class TestRefreshToken:
    def test_refresh_success(self) -> None:
        mock_creds = MagicMock()
        refreshed = _refresh_token(mock_creds)
        assert refreshed == mock_creds
        mock_creds.refresh.assert_called_once()

    def test_refresh_failure_raises_auth_error(self) -> None:
        mock_creds = MagicMock()
        mock_creds.refresh.side_effect = RefreshError("Token revoked")
        with pytest.raises(AuthenticationError, match="Failed to refresh"):
            _refresh_token(mock_creds)


class TestOAuthFlow:
    def test_missing_credentials_file_raises_auth_error(self, tmp_path: Path) -> None:
        missing_file = tmp_path / "missing_creds.json"
        with pytest.raises(AuthenticationError, match="credentials file not found"):
            _run_oauth_flow(missing_file, ["scope1"])


class TestSaveToken:
    def test_save_token_writes_file(self, tmp_path: Path) -> None:
        token_path = tmp_path / "subdir" / "token.json"
        mock_creds = MagicMock()
        mock_creds.to_json.return_value = '{"token": "xyz"}'
        _save_token(mock_creds, token_path)

        assert token_path.exists()
        assert token_path.read_text(encoding="utf-8") == '{"token": "xyz"}'


class TestGetGmailService:
    @patch("gmail.auth.build")
    @patch("gmail.auth._load_token")
    def test_service_with_valid_token(
        self, mock_load: MagicMock, mock_build: MagicMock, tmp_path: Path
    ) -> None:
        mock_creds = MagicMock()
        mock_creds.valid = True
        mock_load.return_value = mock_creds

        mock_resource = MagicMock()
        mock_build.return_value = mock_resource

        config = GmailConfig(token_file=str(tmp_path / "token.json"))
        service = get_gmail_service(config)

        assert service == mock_resource
        mock_build.assert_called_with("gmail", "v1", credentials=mock_creds, cache_discovery=False)

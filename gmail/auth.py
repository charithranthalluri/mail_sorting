"""OAuth2 authentication for the Gmail API.

Handles the full OAuth2 lifecycle:
- Loading saved tokens from disk
- Interactive browser-based authorization for first-time setup
- Automatic refresh of expired access tokens
- Persisting refreshed tokens back to disk

Never logs secrets, tokens, or credential contents.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from config import GmailConfig
from utils.logger import get_logger

if TYPE_CHECKING:
    from googleapiclient.discovery import Resource  # type: ignore[attr-defined]

logger = get_logger(__name__)


class AuthenticationError(Exception):
    """Raised when Gmail API authentication fails."""


def _load_token(token_path: Path) -> Credentials | None:
    """Load an existing OAuth2 token from disk.

    Args:
        token_path: Path to the saved token JSON file.

    Returns:
        Credentials if a valid token file exists, else None.
    """
    if not token_path.exists():
        logger.debug("No saved token found at %s", token_path)
        return None

    try:
        creds = Credentials.from_authorized_user_file(str(token_path))
        logger.info("Loaded saved token from %s", token_path)
        return creds
    except (ValueError, KeyError) as exc:
        logger.warning("Saved token at %s is invalid, will re-authenticate: %s", token_path, exc)
        return None


def _refresh_token(creds: Credentials) -> Credentials:
    """Attempt to refresh an expired token.

    Args:
        creds: Expired but refresh-capable credentials.

    Returns:
        Refreshed credentials.

    Raises:
        AuthenticationError: If the refresh fails.
    """
    try:
        creds.refresh(Request())
        logger.info("Access token refreshed successfully")
        return creds
    except RefreshError as exc:
        logger.error("Token refresh failed: %s", exc)
        raise AuthenticationError(
            "Failed to refresh OAuth2 token. Re-authorization may be required."
        ) from exc


def _run_oauth_flow(credentials_file: Path, scopes: list[str]) -> Credentials:
    """Run the interactive OAuth2 authorization flow.

    Opens a local browser window so the user can grant consent.

    Args:
        credentials_file: Path to the OAuth2 client secrets JSON file.
        scopes: Gmail API scopes to request.

    Returns:
        Freshly authorized credentials.

    Raises:
        AuthenticationError: If the credentials file is missing or the flow fails.
    """
    if not credentials_file.exists():
        raise AuthenticationError(
            f"OAuth2 credentials file not found: {credentials_file}. "
            "Download it from the Google Cloud Console."
        )

    logger.info("Starting interactive OAuth2 authorization flow")
    try:
        flow = InstalledAppFlow.from_client_secrets_file(
            str(credentials_file),
            scopes=scopes,
        )
        creds = flow.run_local_server(port=0)
        logger.info("OAuth2 authorization completed successfully")
        return creds
    except Exception as exc:
        logger.error("OAuth2 authorization flow failed: %s", exc)
        raise AuthenticationError(
            "Interactive OAuth2 authorization failed."
        ) from exc


def _save_token(creds: Credentials, token_path: Path) -> None:
    """Persist credentials to disk for future runs.

    Args:
        creds: Authorized credentials to save.
        token_path: Destination file path.
    """
    try:
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json(), encoding="utf-8")
        logger.info("Token saved to %s", token_path)
    except OSError as exc:
        logger.warning("Failed to save token to %s: %s", token_path, exc)


def get_gmail_service(config: GmailConfig) -> "Resource":
    """Build and return an authenticated Gmail API service.

    Authentication precedence:
    1. Load a saved token from ``config.token_file``.
    2. If the token is expired but refreshable, refresh it.
    3. Otherwise, run the interactive OAuth2 flow.

    The resulting token is saved to disk after any change.

    Args:
        config: Gmail-specific configuration (paths, scopes).

    Returns:
        An authenticated ``googleapiclient.discovery.Resource`` for the
        Gmail API (v1).

    Raises:
        AuthenticationError: If authentication cannot be completed.
    """
    token_path = Path(config.token_file)
    credentials_path = Path(config.credentials_file)

    creds = _load_token(token_path)

    if creds is not None and creds.valid:
        logger.debug("Existing token is valid")
    else:
        refreshed = False
        if creds is not None and creds.expired and creds.refresh_token:
            logger.info("Token expired, attempting refresh")
            try:
                creds = _refresh_token(creds)
                _save_token(creds, token_path)
                refreshed = True
            except AuthenticationError:
                logger.warning("Refresh token is expired or revoked. Starting re-authorization...")

        if not refreshed:
            creds = _run_oauth_flow(credentials_path, config.scopes)
            _save_token(creds, token_path)

    try:
        service = build("gmail", "v1", credentials=creds, cache_discovery=False)
        logger.info("Gmail API service created successfully")
        return service
    except Exception as exc:
        logger.error("Failed to build Gmail API service: %s", exc)
        raise AuthenticationError("Could not build Gmail API service.") from exc

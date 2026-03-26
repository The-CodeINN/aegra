"""Admin authentication service.

Provides a process-lifetime cached admin token that is automatically refreshed
by logging in with ADMIN_EMAIL_ADDRESS + ADMIN_PASSWORD whenever the current
token is missing or close to expiry.

Usage::

    from aegra_api.services.admin_auth import admin_token_manager

    token = await admin_token_manager.get_token()
"""

from __future__ import annotations

import asyncio
import base64
import json
import time

import httpx
import structlog

from aegra_api.settings import settings

logger = structlog.get_logger()

# How many seconds before expiry we treat the token as stale and proactively
# refresh it.  This prevents a token from expiring mid-request.
_EXPIRY_BUFFER_SECONDS = 120


def _decode_jwt_exp(token: str) -> float | None:
    """Extract the `exp` claim from a JWT without verifying the signature."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        payload_b64 = parts[1]
        # Base64url → standard base64 with padding
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        exp = payload.get("exp")
        return float(exp) if exp is not None else None
    except Exception:
        return None


def _token_is_valid(token: str) -> bool:
    """Return True if the token exists and is not about to expire."""
    if not token:
        return False
    exp = _decode_jwt_exp(token)
    if exp is None:
        # No expiry claim — treat as always valid (e.g. non-expiring dev tokens).
        return True
    return time.time() < (exp - _EXPIRY_BUFFER_SECONDS)


class AdminTokenManager:
    """Thread-safe, lazily-refreshed admin token cache.

    A single asyncio.Lock serialises concurrent callers so only one login
    request is in-flight at a time.
    """

    def __init__(self) -> None:
        self._token: str | None = None
        self._lock: asyncio.Lock = asyncio.Lock()

    async def get_token(self) -> str:
        """Return a valid admin JWT, logging in if the cached one has expired."""
        # Fast-path: cached token is still good.
        if self._token and _token_is_valid(self._token):
            return self._token

        async with self._lock:
            # Re-check inside the lock in case another coroutine just refreshed.
            if self._token and _token_is_valid(self._token):
                return self._token

            self._token = await self._login()
            return self._token

    async def _login(self) -> str:
        """Authenticate with the LMS and return a fresh admin token."""
        email = settings.app.ADMIN_EMAIL_ADDRESS
        password = settings.app.ADMIN_PASSWORD
        lms_url = settings.app.LMS_URL.rstrip("/")

        if not email or not password:
            raise RuntimeError(
                "ADMIN_EMAIL_ADDRESS and ADMIN_PASSWORD must be set to obtain an admin token. "
                "Set them in your .env file."
            )

        endpoint = f"{lms_url}/api/v1/auth/login"
        logger.info("admin_token_manager.refreshing", endpoint=endpoint, email=email)

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                endpoint,
                json={"email": email, "password": password},
            )
            response.raise_for_status()
            payload = response.json()

        token = payload.get("token")
        if not isinstance(token, str) or not token.strip():
            raise RuntimeError(
                f"LMS login succeeded but response contained no 'token' field. Response keys: {list(payload.keys())}"
            )

        logger.info("admin_token_manager.refreshed", email=email)
        return token

    def invalidate(self) -> None:
        """Force the next call to `get_token` to re-authenticate."""
        self._token = None


# Module-level singleton shared across the entire process.
admin_token_manager = AdminTokenManager()

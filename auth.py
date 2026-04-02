"""
Authentication configuration for LangGraph Agent Server.

This module provides environment-based authentication switching between:
- noop: No authentication (allow all requests)
- custom: Custom authentication integration

Set AUTH_TYPE environment variable to choose authentication mode.
"""

from typing import Any

import structlog
from langgraph_sdk import Auth

from aegra_api.settings import settings

logger = structlog.getLogger(__name__)

# Initialize LangGraph Auth instance
auth = Auth()

# Get authentication type from environment
AUTH_TYPE = settings.app.AUTH_TYPE

if AUTH_TYPE == "noop":
    logger.info("Using noop authentication (no auth required)")

    @auth.authenticate
    async def authenticate(headers: dict[str, str]) -> Auth.types.MinimalUserDict:
        """No-op authentication that allows all requests."""
        _ = headers  # Suppress unused warning
        return {
            "identity": "anonymous",
            "display_name": "Anonymous User",
            "is_authenticated": True,
        }

    @auth.on
    async def authorize(ctx: Auth.types.AuthContext, value: dict[str, Any]) -> dict[str, Any]:
        """No-op authorization that allows access to all resources."""
        _ = ctx, value  # Suppress unused warnings
        return {}  # Empty filter = no access restrictions

elif AUTH_TYPE == "custom":
    logger.info("Using custom authentication")

    import jwt
    from jwt.exceptions import ExpiredSignatureError, InvalidTokenError

    # Get JWT secret from settings
    JWT_SECRET = settings.app.LMS_JWT_SECRET
    if not JWT_SECRET:
        raise ValueError("LMS_JWT_SECRET environment variable is required for custom auth")

    @auth.authenticate
    async def authenticate(headers: dict[str, str]) -> Auth.types.MinimalUserDict:
        """
        Custom authentication handler using JWT token validation.

        Decodes JWT tokens sent from the frontend and validates them
        using the LMS_JWT_SECRET.
        """
        # Extract authorization header
        authorization = (
            headers.get("authorization")
            or headers.get("Authorization")
            or headers.get(b"authorization")
            or headers.get(b"Authorization")
        )

        # Handle bytes headers
        if isinstance(authorization, bytes):
            authorization = authorization.decode("utf-8")

        if not authorization:
            logger.warning("Missing Authorization header")
            raise Auth.exceptions.HTTPException(status_code=401, detail="Authorization header required")

        # Extract token from Bearer format
        if not authorization.startswith("Bearer "):
            raise Auth.exceptions.HTTPException(
                status_code=401,
                detail="Invalid authorization format. Expected 'Bearer <token>'",
            )

        token = authorization.split("Bearer ", 1)[1]

        try:
            # Decode and validate JWT token
            payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])

            # Extract user information from token payload
            user_id = payload.get("userId") or payload.get("id")
            email = payload.get("email")
            name = payload.get("name")
            role = payload.get("role")

            if not user_id:
                raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid token: missing user ID")

            logger.info(f"Successfully authenticated user: {user_id} ({email})")

            return {
                "identity": str(user_id),
                "display_name": name or email or f"User {user_id}",
                "email": email,
                "role": role,
                "permissions": [role] if role else [],
                "is_authenticated": True,
                "token": token,  # Include the JWT token for LMS API calls
            }

        except ExpiredSignatureError:
            # Decode without verification to extract user context for the log
            try:
                unverified = jwt.decode(token, options={"verify_signature": False}, algorithms=["HS256"])
                uid = unverified.get("userId") or unverified.get("id") or "unknown"
                exp = unverified.get("exp")
                iat = unverified.get("iat")
            except Exception:
                uid, exp, iat = "unknown", None, None
            logger.warning(
                "Token has expired",
                user_id=uid,
                token_issued_at=iat,
                token_expired_at=exp,
                hint="User must log in again to obtain a fresh token.",
            )
            raise Auth.exceptions.HTTPException(status_code=401, detail="Token has expired") from None
        except InvalidTokenError as e:
            # Decode without verification to expose which user is affected and
            # whether this looks like a secret-mismatch (wrong environment token)
            # vs a truly malformed token.
            try:
                unverified = jwt.decode(token, options={"verify_signature": False}, algorithms=["HS256"])
                uid = unverified.get("userId") or unverified.get("id") or "unknown"
                iat = unverified.get("iat")
                exp = unverified.get("exp")
            except Exception:
                uid, iat, exp = "unknown", None, None
            error_str = str(e)
            is_sig_failure = "signature" in error_str.lower()
            logger.warning(
                "Token validation failed",
                user_id=uid,
                error=error_str,
                token_issued_at=iat,
                token_expired_at=exp,
                likely_cause=(
                    "JWT secret mismatch — user may have a token from a different "
                    "environment or a pre-rotation session. Ask them to log out and back in."
                    if is_sig_failure
                    else "Malformed or tampered token."
                ),
            )
            raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid authentication token") from e
        except Exception as e:
            logger.error(f"Token validation error: {e}", exc_info=True)
            raise Auth.exceptions.HTTPException(status_code=401, detail="Authentication failed") from e

    @auth.on
    async def authorize(ctx: Auth.types.AuthContext, value: dict[str, Any]) -> dict[str, Any]:
        """
        Multi-tenant authorization with user-scoped access control.
        """
        try:
            # Get user identity from authentication context
            user_id = ctx.user.identity

            if not user_id:
                logger.error("Missing user identity in auth context")
                raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid user identity")

            # Create owner filter for resource access control
            owner_filter = {"owner": user_id}

            # Add owner information to metadata for create/update operations
            metadata = value.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
                value["metadata"] = metadata
            metadata.update(owner_filter)

            # Return filter for database operations
            return owner_filter

        except Auth.exceptions.HTTPException:
            raise
        except Exception as e:
            logger.error(f"Authorization error: {e}", exc_info=True)
            raise Auth.exceptions.HTTPException(status_code=500, detail="Authorization system error") from e

else:
    raise ValueError(f"Unknown AUTH_TYPE: {AUTH_TYPE}. Supported values: 'noop', 'custom'")

"""Caching service for student learning tracks and career advisors.

This module provides caching to avoid repeatedly hitting the LMS API for
learning track information. Uses Redis when available, falls back to in-memory cache.

Cache Strategy:
1. Redis (distributed) - TTL of 1 hour for learning track
2. In-memory (fallback) - TTL of 1 hour for learning track

The advisor is never cached independently — it is always derived from the
cached learning track so the two can never become out-of-sync after a track
change or standalone advisor purchase.
"""

import asyncio
import time
from typing import Any, cast

import httpx
import structlog

from aegra_api.core.redis_manager import redis_manager
from aegra_api.data.career_advisors import (
    get_advisor_by_track,
    get_default_advisor,
)
from aegra_api.settings import settings

logger = structlog.getLogger(__name__)

# Cache TTL in seconds (1 hour)
LEARNING_TRACK_CACHE_TTL = 3600
NEGATIVE_TRACK_CACHE_TTL = 60

# In-memory cache fallback (user_id -> (learning_track, expiry_timestamp))
_memory_cache: dict[str, tuple[str | None, float]] = {}
_cache_lock = asyncio.Lock()

# LMS API URL from settings
LMS_API_URL = settings.app.LMS_URL


def _get_cache_key(user_id: str) -> str:
    """Generate Redis cache key for user's learning track."""
    return f"dedatahub:learning_track:v2:{user_id}"


def _get_advisor_cache_key(user_id: str) -> str:
    """Generate Redis cache key for user's advisor."""
    return f"dedatahub:advisor:v2:{user_id}"


async def _get_from_redis(key: str) -> str | None:
    """Get value from Redis cache."""
    try:
        client = redis_manager.get_client()
        value = await client.get(key)
        return cast(str | None, value)
    except Exception as e:
        logger.warning("Redis get failed", error=str(e))
        return None


async def _set_in_redis(key: str, value: str, ttl: int = LEARNING_TRACK_CACHE_TTL) -> bool:
    """Set value in Redis cache with TTL."""
    try:
        client = redis_manager.get_client()
        await client.setex(key, ttl, value)
        return True
    except Exception as e:
        logger.warning("Redis set failed", error=str(e))
        return False


async def _get_from_memory(user_id: str) -> str | None:
    """Get learning track from in-memory cache."""
    async with _cache_lock:
        if user_id in _memory_cache:
            track, expiry = _memory_cache[user_id]
            if time.time() < expiry:
                return cast(str | None, track)
            else:
                # Expired, remove from cache
                del _memory_cache[user_id]
    return None


async def _set_in_memory(user_id: str, track: str | None) -> None:
    """Set learning track in in-memory cache."""
    async with _cache_lock:
        expiry = time.time() + LEARNING_TRACK_CACHE_TTL
        _memory_cache[user_id] = (track, expiry)


async def _fetch_track_from_subscription(token: str) -> tuple[str | None, bool]:
    """Fetch the student's active subscription track from the LMS API."""
    subscription_endpoint = f"{LMS_API_URL}/api/v1/subscription/me"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                subscription_endpoint,
                headers={"accept": "*/*", "Authorization": f"Bearer {token}"},
            )
            response.raise_for_status()
            data = response.json()

            learning_track = cast(str | None, data.get("track"))
            logger.info("Fetched subscription track from LMS", learning_track=learning_track)
            return learning_track, True

    except httpx.HTTPStatusError as e:
        logger.warning("HTTP error fetching subscription track", status_code=e.response.status_code)
        return None, False
    except httpx.TimeoutException:
        logger.warning("Timeout while fetching subscription track")
        return None, False
    except Exception as e:
        logger.warning("Unexpected error fetching subscription track", error=str(e))
        return None, False


async def check_ai_mentor_addon(token: str) -> dict:
    """Check whether the student's subscription includes an active AI Mentor add-on.

    Returns a dict with:
      - active (bool): True if aiMentorAddOn.active is True on a non-expired subscription
      - expires_at (str | None): ISO timestamp from aiMentorAddOn.expiresAt, if present
    """
    subscription_endpoint = f"{LMS_API_URL}/api/v1/subscription/me"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                subscription_endpoint,
                headers={"accept": "*/*", "Authorization": f"Bearer {token}"},
            )
            response.raise_for_status()
            data = response.json()
            addon = data.get("aiMentorAddOn") or {}
            active = bool(addon.get("active", False))
            expires_at = addon.get("expiresAt")
            return {"active": active, "expires_at": expires_at}
    except httpx.HTTPStatusError as e:
        logger.warning("HTTP error checking ai_mentor_addon", status_code=e.response.status_code)
        return {"active": False, "expires_at": None}
    except Exception as e:
        logger.warning("Unexpected error checking ai_mentor_addon", error=str(e))
        return {"active": False, "expires_at": None}


async def _fetch_learning_track_from_lms(token: str) -> tuple[str | None, bool]:
    """Fetch the student's learning track from the LMS API.

    Active subscription is the source of truth. Onboarding is a fallback for
    accounts without an active track assignment yet.
    """
    subscription_track, subscription_ok = await _fetch_track_from_subscription(token)
    if subscription_track:
        return subscription_track, True

    onboarding_endpoint = f"{LMS_API_URL}/api/v1/ai-mentor/onboarding/me"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                onboarding_endpoint,
                headers={"accept": "*/*", "Authorization": f"Bearer {token}"},
            )
            response.raise_for_status()
            data = response.json()

            # Extract learning track from onboarding data.
            onboarding_data = data.get("onboarding", {})
            sec1 = onboarding_data.get("s1", {}) if isinstance(onboarding_data, dict) else {}
            learning_track = cast(
                str | None,
                onboarding_data.get("learningTrack") if isinstance(onboarding_data, dict) else None,
            ) or cast(str | None, sec1.get("learningTrack") if isinstance(sec1, dict) else None)

            logger.info("Fetched learning track from LMS", learning_track=learning_track)
            return learning_track, True

    except httpx.HTTPStatusError as e:
        logger.error("HTTP error fetching learning track", status_code=e.response.status_code)
        return None, False
    except httpx.TimeoutException:
        logger.error("Timeout while fetching learning track")
        return None, False
    except Exception as e:
        logger.error("Unexpected error fetching learning track", error=str(e))
        return None, False

    # If subscription lookup itself failed, treat this as non-cacheable failure.
    return None, subscription_ok


async def get_cached_learning_track(user_id: str, token: str) -> str | None:
    """Get the student's learning track, using cache when available.

    Cache lookup order:
    1. Redis (if available)
    2. In-memory cache
    3. LMS API (then cache the result)

    Args:
        user_id: The user's unique identifier
        token: JWT bearer token for LMS API authentication

    Returns:
        The learning track string or None if not found
    """
    cache_key = _get_cache_key(user_id)

    # Try Redis first
    cached = await _get_from_redis(cache_key)
    if cached is not None:
        logger.debug("Learning track cache hit (Redis)", user_id=user_id)
        return cached if cached != "__none__" else None

    # Try in-memory cache
    cached = await _get_from_memory(user_id)
    if cached is not None:
        logger.debug("Learning track cache hit (memory)", user_id=user_id)
        return cached

    # Cache miss - fetch from LMS
    logger.info("Learning track cache miss, fetching from LMS", user_id=user_id)
    learning_track, cacheable = await _fetch_learning_track_from_lms(token)

    if learning_track:
        await _set_in_redis(cache_key, learning_track)
        await _set_in_memory(user_id, learning_track)
    elif cacheable:
        # Keep negative cache short to reduce staleness after user updates.
        await _set_in_redis(cache_key, "__none__", ttl=NEGATIVE_TRACK_CACHE_TTL)
        await _set_in_memory(user_id, None)
    else:
        # Skip caching transient failures (timeouts, auth errors, upstream outage).
        return None

    return learning_track


async def get_cached_advisor(user_id: str, token: str) -> tuple[dict[str, Any], str | None]:
    """Get the student's career advisor, always derived from the cached learning track.

    The advisor is NOT cached separately — it is always derived from the learning
    track (which has its own Redis + in-memory cache).  This prevents the advisor
    and learning track from ever becoming out-of-sync when the student changes track
    or purchases a standalone advisor subscription with a different track selection.

    Args:
        user_id: The user's unique identifier
        token: JWT bearer token for LMS API authentication

    Returns:
        Tuple of (advisor_dict, learning_track)
    """
    learning_track = await get_cached_learning_track(user_id, token)

    if learning_track:
        advisor = get_advisor_by_track(learning_track)
        if not advisor:
            advisor = get_default_advisor()
    else:
        advisor = get_default_advisor()

    return advisor, learning_track


async def invalidate_user_cache(user_id: str) -> None:
    """Invalidate all cached data for a user.

    Call this when:
    - User updates their learning track
    - User completes onboarding
    - Admin resets user data
    """
    track_key = _get_cache_key(user_id)
    # Also clear any legacy advisor cache keys that may exist from older deployments.
    legacy_advisor_key = _get_advisor_cache_key(user_id)

    # Clear Redis cache
    try:
        client = redis_manager.get_client()
        await client.delete(track_key, legacy_advisor_key)
        logger.info("Invalidated Redis cache", user_id=user_id)
    except Exception as e:
        logger.warning("Failed to invalidate Redis cache", error=str(e))

    # Clear in-memory cache
    async with _cache_lock:
        _memory_cache.pop(user_id, None)

    logger.info("Invalidated all caches", user_id=user_id)


def clear_memory_cache() -> None:
    """Clear the entire in-memory cache. Useful for testing."""
    global _memory_cache
    _memory_cache = {}

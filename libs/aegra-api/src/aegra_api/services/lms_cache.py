"""LMS API response caching with Redis + in-memory fallback.

Caches LMS endpoint responses to reduce redundant API calls.
Uses the existing redis_manager for distributed caching and falls
back to an in-memory TTL cache when Redis is unavailable.

TTL Strategy (endpoints that rarely change get longer TTLs):
- Onboarding sections (section-1..8, /onboarding)  → 24 hours
- User profile (/user/profile)                      → 1 hour
- Enrollment/blackboard (/enrollment/*)              → 15 minutes
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx
import structlog

from aegra_api.core.redis import redis_manager

logger = structlog.get_logger()

# TTL constants (seconds)
TTL_ONBOARDING = 86400  # 24 hours — onboarding data rarely changes
TTL_PROFILE = 3600  # 1 hour — name/email rarely change
TTL_ENROLLMENT = 900  # 15 minutes — progress changes more often

# In-memory fallback: key → (json_str, expiry_ts)
_mem_cache: dict[str, tuple[str, float]] = {}
_cache_lock = asyncio.Lock()


def _ttl_for_path(path: str) -> int:
    """Return the appropriate TTL based on the endpoint path."""
    if "/onboarding" in path or "/ai-mentor/" in path:
        return TTL_ONBOARDING
    if "/user/profile" in path:
        return TTL_PROFILE
    if "/enrollment" in path:
        return TTL_ENROLLMENT
    return TTL_PROFILE  # safe default


def _cache_key(user_id: str, path: str) -> str:
    """Build a Redis/memory cache key."""
    return f"lms:{user_id}:{path}"


async def _redis_get(key: str) -> str | None:
    if not redis_manager.is_available():
        return None
    try:
        client = redis_manager.get_client()
        val = await client.get(key)
        return val  # type: ignore[return-value]
    except Exception as exc:
        logger.debug("lms_cache_redis_get_error", key=key, error=str(exc))
        return None


async def _redis_set(key: str, value: str, ttl: int) -> None:
    if not redis_manager.is_available():
        return
    try:
        client = redis_manager.get_client()
        await client.setex(key, ttl, value)
    except Exception as exc:
        logger.debug("lms_cache_redis_set_error", key=key, error=str(exc))


async def _mem_get(key: str) -> str | None:
    async with _cache_lock:
        entry = _mem_cache.get(key)
        if entry is None:
            return None
        value, expiry = entry
        if time.time() >= expiry:
            del _mem_cache[key]
            return None
        return value


async def _mem_set(key: str, value: str, ttl: int) -> None:
    async with _cache_lock:
        _mem_cache[key] = (value, time.time() + ttl)


async def cached_lms_fetch(
    client: httpx.AsyncClient,
    url: str,
    token: str,
    user_id: str,
) -> dict[str, Any]:
    """Fetch JSON from an LMS endpoint with Redis + in-memory caching.

    Cache lookup order:
    1. Redis (distributed, survives restarts)
    2. In-memory (process-local fallback)
    3. Live HTTP call → result stored in both layers

    Returns {} on HTTP or network errors (same behaviour as the
    existing ``_fetch_json`` helper).
    """
    from urllib.parse import urlparse

    path = urlparse(url).path  # e.g. /api/v1/user/profile
    key = _cache_key(user_id, path)
    ttl = _ttl_for_path(path)

    # 1. Redis
    cached = await _redis_get(key)
    if cached is not None:
        logger.debug("lms_cache_hit", source="redis", path=path)
        try:
            return json.loads(cached)
        except json.JSONDecodeError:
            pass

    # 2. In-memory
    cached = await _mem_get(key)
    if cached is not None:
        logger.debug("lms_cache_hit", source="memory", path=path)
        try:
            return json.loads(cached)
        except json.JSONDecodeError:
            pass

    # 3. Live fetch
    try:
        resp = await client.get(
            url,
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            timeout=10.0,
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
    except Exception as exc:
        logger.debug("lms_fetch_failed", url=url, error=str(exc))
        return {}

    # Store in both cache layers
    serialized = json.dumps(data)
    await _redis_set(key, serialized, ttl)
    await _mem_set(key, serialized, ttl)
    logger.debug("lms_cache_stored", path=path, ttl=ttl)
    return data


async def invalidate_lms_cache(user_id: str, paths: list[str] | None = None) -> None:
    """Invalidate cached LMS data for a user.

    If *paths* is ``None``, all known prefixes are cleared.
    """
    if paths is None:
        paths = [
            "/api/v1/user/profile",
            "/api/v1/enrollment/student/blackboard",
            "/api/v1/onboarding",
            "/api/v1/ai-mentor/onboarding/section-1",
            "/api/v1/ai-mentor/onboarding/section-2",
            "/api/v1/ai-mentor/onboarding/section-4",
            "/api/v1/ai-mentor/onboarding/section-5",
            "/api/v1/ai-mentor/onboarding/section-6",
            "/api/v1/ai-mentor/onboarding/section-7",
            "/api/v1/ai-mentor/onboarding/section-8",
            "/api/v1/ai-mentor/onboarding/me",
        ]

    keys = [_cache_key(user_id, p) for p in paths]

    # Redis
    if redis_manager.is_available():
        try:
            client = redis_manager.get_client()
            await client.delete(*keys)
        except Exception as exc:
            logger.debug("lms_cache_invalidate_redis_error", error=str(exc))

    # Memory
    async with _cache_lock:
        for k in keys:
            _mem_cache.pop(k, None)

    logger.info("lms_cache_invalidated", user_id=user_id, paths=paths)

"""Retry utilities for outbound API and model calls.

Ported from Claude-code's ``src/services/api/withRetry.ts``.

Pattern
-------
- Exponential back-off with jitter (base 500 ms, max 32 s)
- HTTP 429 Rate-Limit: reads ``Retry-After`` header; backs off accordingly
- HTTP 529 / Overload: up to ``MAX_OVERLOAD_RETRIES`` consecutive 529s then re-raise
- HTTP 401/403: surfaces immediately — no retry (auth must be fixed externally)
- All other exceptions: retried up to ``max_retries`` with exponential back-off
- ``CannotRetryError``: raised when the retry budget is exhausted

Usage
-----
    from react_agent.retry import with_retry, CannotRetryError

    # Decorator form
    @with_retry()
    async def call_api() -> dict:
        ...

    # Inline form — wrap any coroutine
    result = await with_retry(call_api)()

    # On failure after all retries
    try:
        result = await my_retried_fn()
    except CannotRetryError as e:
        logger.error("All retries exhausted: %s", e.context)
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Configuration constants  (matching Claude-code defaults)
# ---------------------------------------------------------------------------

MAX_RETRIES: int = 10
BASE_DELAY_MS: float = 500.0
MAX_DELAY_MS: float = 32_000.0
MAX_OVERLOAD_RETRIES: int = 3  # consecutive 529s before giving up
_JITTER_FACTOR: float = 0.2  # ±20 % random jitter added to each delay


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class RetryContext:
    """Metadata attached to ``CannotRetryError`` for structured logging."""

    fn_name: str
    attempts: int
    last_status_code: int | None = None
    last_error: str = ""


class CannotRetryError(RuntimeError):
    """Raised when all retry attempts are exhausted."""

    def __init__(self, message: str, context: RetryContext) -> None:
        super().__init__(message)
        self.context = context


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _delay_for_attempt(attempt: int) -> float:
    """Return exponential back-off delay in seconds, capped at MAX_DELAY_MS."""
    raw_ms = BASE_DELAY_MS * (2**attempt)
    capped_ms = min(raw_ms, MAX_DELAY_MS)
    jitter = capped_ms * _JITTER_FACTOR * (2 * random.random() - 1)  # nosec B311
    return max((capped_ms + jitter) / 1000.0, 0.1)


def _retry_after_seconds(headers: httpx.Headers | dict[str, str] | None) -> float | None:
    """Extract ``Retry-After`` seconds from response headers if present."""
    if not headers:
        return None
    raw = headers.get("Retry-After") or headers.get("retry-after")
    if raw is None:
        return None
    try:
        return max(float(raw.strip()), 1.0)
    except ValueError:
        return None


def _http_status(exc: BaseException) -> int | None:
    """Extract HTTP status code from an exception if available."""
    # httpx
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code
    # anthropic-sdk and other libs expose ``status_code`` on the exception
    sc = getattr(exc, "status_code", None)
    if isinstance(sc, int):
        return sc
    # Some wrappers put it in ``response``
    resp = getattr(exc, "response", None)
    if resp is not None:
        return getattr(resp, "status_code", None)
    return None


def _response_headers(exc: BaseException) -> httpx.Headers | dict | None:
    """Extract response headers from an exception if available."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.headers
    resp = getattr(exc, "response", None)
    if resp is not None:
        return getattr(resp, "headers", None)
    return None


# ---------------------------------------------------------------------------
# Core retry logic
# ---------------------------------------------------------------------------


def with_retry(
    max_retries: int = MAX_RETRIES,
    *,
    fail_open: bool = False,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator factory that wraps an async function with retry / back-off logic.

    Args:
        max_retries:  Maximum number of retry attempts (default ``MAX_RETRIES = 10``).
        fail_open:    If ``True``, swallow ``CannotRetryError`` and return ``None``
                      instead of re-raising.  Useful for optional guardrail calls
                      where availability > correctness.

    Raises:
        CannotRetryError: When the retry budget is exhausted and ``fail_open=False``.

    Example::

        @with_retry(max_retries=5)
        async def call_lms() -> dict:
            ...
    """

    def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> T | None:
            fn_name = getattr(fn, "__name__", repr(fn))
            consecutive_overload = 0
            last_exc: BaseException | None = None
            last_status: int | None = None

            for attempt in range(max_retries + 1):
                try:
                    return await fn(*args, **kwargs)

                except BaseException as exc:  # noqa: BLE001
                    last_exc = exc
                    status = _http_status(exc)
                    last_status = status

                    # ---- Auth errors: never retry --------------------------
                    if status in (401, 403):
                        logger.warning(
                            "retry: auth error %s on %s — not retrying",
                            status,
                            fn_name,
                        )
                        raise

                    # ---- Rate-limit ----------------------------------------
                    if status == 429:
                        headers = _response_headers(exc)
                        delay = _retry_after_seconds(headers) or _delay_for_attempt(attempt)
                        if attempt < max_retries:
                            logger.warning(
                                "retry: 429 rate-limit on %s — waiting %.1fs (attempt %d/%d)",
                                fn_name,
                                delay,
                                attempt + 1,
                                max_retries,
                            )
                            await asyncio.sleep(delay)
                            continue
                        # Budget exhausted on 429
                        break

                    # ---- Overload (529) ------------------------------------
                    if status == 529:
                        consecutive_overload += 1
                        if consecutive_overload >= MAX_OVERLOAD_RETRIES:
                            logger.error(
                                "retry: %d consecutive 529s on %s — giving up",
                                consecutive_overload,
                                fn_name,
                            )
                            break
                        delay = _delay_for_attempt(attempt)
                        logger.warning(
                            "retry: 529 overload on %s — waiting %.1fs (overload %d/%d)",
                            fn_name,
                            delay,
                            consecutive_overload,
                            MAX_OVERLOAD_RETRIES,
                        )
                        await asyncio.sleep(delay)
                        continue

                    # For other statuses, reset the overload counter
                    consecutive_overload = 0

                    # ---- Retryable error ----------------------------------
                    if attempt < max_retries:
                        delay = _delay_for_attempt(attempt)
                        logger.warning(
                            "retry: %s on %s — waiting %.1fs (attempt %d/%d): %s",
                            type(exc).__name__,
                            fn_name,
                            delay,
                            attempt + 1,
                            max_retries,
                            str(exc)[:120],
                        )
                        await asyncio.sleep(delay)
                    else:
                        break

            # --- All retries exhausted ----------------------------------------
            ctx = RetryContext(
                fn_name=fn_name,
                attempts=max_retries,
                last_status_code=last_status,
                last_error=str(last_exc)[:256] if last_exc else "",
            )
            err = CannotRetryError(
                f"{fn_name} failed after {max_retries} retries: {last_exc}",
                ctx,
            )
            logger.error(
                "retry: exhausted %d retries for %s (last_status=%s): %s",
                max_retries,
                fn_name,
                last_status,
                str(last_exc)[:200],
            )
            if fail_open:
                return None  # type: ignore[return-value]
            raise err from last_exc

        return wrapper  # type: ignore[return-value]

    return decorator

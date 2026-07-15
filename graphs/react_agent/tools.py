"""This module provides tools for the agent.

Includes:
- Web search functionality
- Student profile information retrieval from LMS
- Long-term memory storage and retrieval

These tools are intended as examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

import asyncio
import contextlib
import json
import logging
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

import httpx
from langchain_community.tools import BraveSearch
from langgraph.config import get_config
from langgraph.runtime import get_runtime

from react_agent.context import Context
from react_agent.memory import DEFAULT_MEMORY_NAMESPACE
from react_agent.retry import with_retry
from react_agent.sanitization import validate_resource_id as _validate_id
from react_agent.session_memory import SESSION_MEMORY_NAMESPACE_SUFFIX

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LMS response caching (Redis + in-memory cache)
# ---------------------------------------------------------------------------
# TTLs (seconds)
_TTL_ONBOARDING = 120
_TTL_PROFILE = 120
_TTL_ENROLLMENT = 45

# In-memory cache: key → (json_str, expiry_ts)
_mem_cache: dict[str, tuple[str, float]] = {}

# Per-key async locks prevent the thundering-herd race where N concurrent
# requests all find the cache cold and each fire an independent LMS call.
# A defaultdict(asyncio.Lock) isn't safe to initialise at module level because
# asyncio.Lock() must be created on the running event loop; we create lazily.
_mem_cache_locks: dict[str, asyncio.Lock] = {}

# Lazy Redis client reference (set once on first use)
_redis_client: Any = None
_redis_checked = False

# MongoDB / sync-service call timeout (seconds).  A hanging MongoDB connection
# would otherwise block an asyncio thread-pool worker indefinitely, starving
# all other coroutines sharing that event loop.
_MONGO_TIMEOUT_SECONDS = 10.0


async def _mongo_call(fn: Any, /, *args: Any, **kwargs: Any) -> Any:
    """Run a synchronous MongoDB call in a thread with a hard timeout.

    Prevents a hanging MongoDB connection from blocking the asyncio thread
    pool indefinitely.  Raises ``asyncio.TimeoutError`` on timeout.
    """
    return await asyncio.wait_for(
        asyncio.to_thread(fn, *args, **kwargs),
        timeout=_MONGO_TIMEOUT_SECONDS,
    )


def _normalize_auth_token(token: str | None) -> str | None:
    """Normalize user token value before forwarding to LMS.

    Accepts raw JWT or a mistakenly prefixed value like "Bearer <jwt>".
    """
    if not token:
        return None

    cleaned = token.strip()
    if not cleaned:
        return None

    if cleaned.lower().startswith("bearer "):
        cleaned = cleaned[7:].strip()

    return cleaned or None


def _ttl_for_path(path: str) -> int:
    # Critical live state: do not cache, always fetch from LMS.
    if path.endswith("/structure") or path.endswith("/progress") or "/attempts" in path:
        return 0
    if path.endswith("/subscription/me"):
        return 0
    if "/onboarding" in path or "/ai-mentor/" in path:
        return _TTL_ONBOARDING
    if "/user/profile" in path:
        return _TTL_PROFILE
    if "/enrollment" in path:
        return _TTL_ENROLLMENT
    return _TTL_PROFILE


def _cache_key(user_id: str, path: str) -> str:
    return f"agent:{user_id}:{path}"


def _get_redis_client() -> Any:
    """Return the shared Redis client from aegra_api.core.redis, or None."""
    global _redis_client, _redis_checked
    if _redis_checked:
        return _redis_client
    _redis_checked = True
    try:
        from aegra_api.core.redis import redis_manager

        if redis_manager.is_available():
            _redis_client = redis_manager.get_client()
    except Exception:
        _redis_client = None
    return _redis_client


async def _cached_lms_get_with_evidence(
    client: httpx.AsyncClient,
    url: str,
    token: str,
    user_id: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """GET LMS endpoint with evidence metadata for observability and confidence."""
    parsed = urlparse(url)
    path = parsed.path
    full_path = f"{parsed.netloc}{parsed.path}"
    uid = user_id or "anon"
    key = _cache_key(uid, full_path)
    ttl = _ttl_for_path(full_path)
    fetched_at = datetime.now(tz=UTC).isoformat()

    # Critical paths enforce live fetch.
    if ttl <= 0:

        @with_retry(max_retries=3)
        async def _live_get() -> dict[str, Any]:
            resp = await client.get(
                url,
                headers={"accept": "*/*", "Authorization": f"Bearer {token}"},
            )
            resp.raise_for_status()
            return resp.json()

        data: dict[str, Any] = await _live_get()
        return data, {
            "endpoint": path,
            "fetched_at": fetched_at,
            "cache_source": "none",
            "cache_ttl_seconds": 0,
            "live_verified": True,
        }

    rc = _get_redis_client()
    if rc is not None:
        with contextlib.suppress(Exception):
            val = await rc.get(key)
            if val is not None:
                return json.loads(val), {
                    "endpoint": path,
                    "fetched_at": fetched_at,
                    "cache_source": "redis",
                    "cache_ttl_seconds": ttl,
                    "live_verified": False,
                }

    # Per-key lock: guarantees only ONE coroutine hits the live LMS endpoint
    # when multiple requests arrive simultaneously for the same cold cache entry
    # (thundering-herd prevention).
    if key not in _mem_cache_locks:
        _mem_cache_locks[key] = asyncio.Lock()
    async with _mem_cache_locks[key]:
        # Re-check cache after acquiring the lock — a sibling coroutine may have
        # already populated it while we were waiting.
        entry = _mem_cache.get(key)
        if entry is not None:
            value, expiry = entry
            if time.time() < expiry:
                return json.loads(value), {
                    "endpoint": path,
                    "fetched_at": fetched_at,
                    "cache_source": "memory",
                    "cache_ttl_seconds": ttl,
                    "live_verified": False,
                }
            del _mem_cache[key]

        @with_retry(max_retries=3)
        async def _live_get_cached() -> dict[str, Any]:
            resp = await client.get(
                url,
                headers={"accept": "*/*", "Authorization": f"Bearer {token}"},
            )
            resp.raise_for_status()
            return resp.json()

        data = await _live_get_cached()

        serialized = json.dumps(data)
        if rc is not None:
            with contextlib.suppress(Exception):
                await rc.setex(key, ttl, serialized)
        _mem_cache[key] = (serialized, time.time() + ttl)

    return data, {
        "endpoint": path,
        "fetched_at": fetched_at,
        "cache_source": "live",
        "cache_ttl_seconds": ttl,
        "live_verified": True,
    }


# Import course content services
try:
    import sys
    from pathlib import Path

    # Add src to path if needed
    project_root = Path(__file__).parent.parent.parent
    src_path = project_root / "libs" / "aegra-api" / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from aegra_api.tools.course_content.mongo_client import get_course_content_mongo_client
    from aegra_api.tools.course_content.search_service import get_course_content_search_service

    COURSE_CONTENT_AVAILABLE = True
    logger.info("Course content services loaded successfully")
except ImportError as e:
    COURSE_CONTENT_AVAILABLE = False
    logger.warning(f"Course content services not available: {e}. Course search will be disabled.")


async def brave_search(query: str) -> str:
    """Search the web for general information and current events using Brave Search.

    Args:
        query: The search query string
    """
    runtime = get_runtime(Context)
    api_key = runtime.context.brave_search_api_key

    try:
        logger.info(f"Searching web with Brave for: {query}")

        if api_key:
            tool = BraveSearch.from_api_key(api_key=api_key, search_kwargs={"count": 3})
        else:
            # Fallback to environment variable if not in context
            tool = BraveSearch.from_search_kwargs(search_kwargs={"count": 3})

        # Execute search in thread to avoid blocking
        search_results = await asyncio.to_thread(tool.run, query)

        return search_results

    except Exception as e:
        logger.error(f"Error in Brave search: {str(e)}", exc_info=True)
        return f"Search failed: {str(e)}"


# async def search(query: str) -> dict[str, Any]:
#     """Search the web for general information and current events.
#
#     This function performs a search using the Tavily search engine, which provides
#     comprehensive, accurate, and trusted results. It's particularly useful for
#     answering questions about current events, general knowledge, and research.
#
#     Args:
#         query: The search query string
#     """
#     runtime = get_runtime(Context)
#     max_results = runtime.context.max_search_results
#
#     try:
#         logger.info(f"Searching web for: {query}")
#
#         # Initialize Tavily search with max results from context
#         web_search = TavilySearch(max_results=max_results, topic="general")
#
#         # Execute search in thread to avoid blocking
#         search_results = await asyncio.to_thread(web_search.invoke, {"query": query})
#
#         # Handle different response formats
#         if isinstance(search_results, list):
#             results_list = search_results
#         elif isinstance(search_results, dict):
#             results_list = search_results.get("results", [])
#         else:
#             logger.warning(f"Unexpected response type: {type(search_results)}")
#             return {
#                 "query": query,
#                 "results": [],
#                 "error": f"Unexpected response type: {type(search_results)}",
#             }
#
#         # Process and format results
#         processed_results = {"query": query, "results": []}
#
#         for result in results_list:
#             if isinstance(result, dict):
#                 processed_results["results"].append(
#                     {
#                         "title": result.get("title", "No title"),
#                         "url": result.get("url", ""),
#                         "content_preview": result.get("content", ""),
#                     }
#                 )
#             else:
#                 logger.warning(f"Unexpected result type: {type(result)}")
#
#         logger.info(
#             f"Found {len(processed_results['results'])} search results for '{query}'"
#         )
#         return processed_results
#
#     except Exception as e:
#         logger.error(f"Error in web search: {str(e)}", exc_info=True)
#         return {"query": query, "results": [], "error": f"Search failed: {str(e)}"}


# async def extract_webpage_content(urls: list[str]) -> list[dict[str, Any]]:
#     """Extract full content from webpages for detailed analysis.
#
#     Use this after the search tool to get complete information from promising results.
#     Extracts the main content, title, and other relevant information from web pages.
#
#     Args:
#         urls: List of URLs to extract content from (max 3 recommended)
#     """
#     try:
#         logger.info(f"Extracting content from {len(urls)} URLs")
#
#         # Initialize Tavily extract
#         web_extract = TavilyExtract()
#
#         # Execute extraction in thread to avoid blocking
#         results = await asyncio.to_thread(web_extract.invoke, {"urls": urls})
#
#         # Extract results from response
#         extracted_results = (
#             results.get("results", []) if isinstance(results, dict) else []
#         )
#
#         # Process results to ensure they have content
#         processed_results = []
#         for result in extracted_results:
#             if isinstance(result, dict):
#                 # Tavily uses 'raw_content' not 'content'
#                 content = result.get("raw_content", "")
#                 processed_results.append(
#                     {
#                         "url": result.get("url", ""),
#                         "title": result.get("title", ""),
#                         "content": content,
#                         "content_length": len(content),
#                     }
#                 )
#             else:
#                 processed_results.append(result)
#
#         logger.info(
#             f"Successfully extracted content from {len(processed_results)} pages"
#         )
#         return processed_results
#
#     except Exception as e:
#         logger.error(f"Error extracting webpage content: {str(e)}", exc_info=True)
#         return [{"error": f"Extraction failed: {str(e)}"}]


async def get_student_profile() -> dict[str, Any]:
    """Get the current student's profile information from the LMS.

    Retrieves the authenticated student's profile including their name, role,
    onboarding status, and other relevant information. Returns a dict with:
    - name: Student's full name
    - role: User role (typically 'student')
    - onboardingComplete: Whether student completed onboarding
    - onboardingSkipped: Whether student skipped onboarding
    """
    runtime = get_runtime(Context)

    # Get the user token from context
    token = _normalize_auth_token(runtime.context.user_token)
    if not token:
        logger.error("No user token available in context")
        return {
            "error": "Authentication required",
            "message": "Unable to fetch student profile without authentication token",
        }

    logger.info(f"Attempting to fetch profile with token (length: {len(token)})")

    # Get LMS API URL from context
    lms_url = runtime.context.lms_api_url.rstrip("/")
    profile_endpoint = f"{lms_url}/api/v1/user/profile"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            data, evidence = await _cached_lms_get_with_evidence(
                client,
                profile_endpoint,
                token,
                runtime.context.user_id,
            )

            user_data = data.get("user", data)

            # Extract only the required fields
            profile = {
                "name": user_data.get("name") or user_data.get("firstName"),
                "role": user_data.get("role"),
                "onboardingComplete": data.get("onboardingComplete") or user_data.get("onboardingComplete"),
                "onboardingSkipped": data.get("onboardingSkipped") or user_data.get("onboardingSkipped"),
                "sourceEndpoint": "/api/v1/user/profile",
                "evidence": evidence,
                "confidence": "high" if evidence.get("live_verified") else "medium",
            }

            logger.info(f"Successfully fetched profile for student: {profile.get('name')}")
            return profile

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching student profile: {e.response.status_code} - {e.response.text[:200]}")
        return {
            "error": "API request failed",
            "status_code": e.response.status_code,
            "message": str(e),
            "details": e.response.text[:200],
        }
    except httpx.TimeoutException:
        logger.error("Timeout while fetching student profile")
        return {
            "error": "Request timeout",
            "message": "The LMS API took too long to respond",
        }
    except Exception as e:
        logger.error(f"Unexpected error fetching student profile: {e}", exc_info=True)
        return {"error": "Unexpected error", "message": str(e)}


def _detect_onboarding_contradictions(onboarding_data: dict[str, Any]) -> list[str]:
    """Detect contradictions between structured fields and freeform text in onboarding data.

    Returns a list of human-readable notes describing each contradiction found.
    The model should use structured/dropdown fields as the source of truth and
    ask the user to clarify when contradictions are present.
    """
    notes: list[str] = []
    s2 = onboarding_data.get("s2", {})
    if not isinstance(s2, dict):
        return notes

    role_summary = (s2.get("roleSummary") or "").strip()
    years_experience = (s2.get("yearsExperience") or "").strip()
    years_tech = (s2.get("yearsTech") or "").strip()

    if not role_summary:
        return notes

    # Extract any "N years" claim from the freeform roleSummary
    match = re.search(r"(\d+)\s*(?:\+\s*)?years?\b", role_summary, re.IGNORECASE)
    if match:
        claimed_years = int(match.group(1))
        # Check against yearsExperience dropdown
        is_beginner_experience = years_experience.lower() in (
            "less than 1 year",
            "0",
            "none",
            "zero",
        )
        is_beginner_tech = "zero" in years_tech.lower() or "beginner" in years_tech.lower()

        if claimed_years >= 2 and (is_beginner_experience or is_beginner_tech):
            notes.append(
                f"DATA CONFLICT in employment section (s2): The freeform 'roleSummary' says "
                f"'{role_summary}' (implying {claimed_years} years of experience), but the "
                f"structured fields say yearsExperience='{years_experience}' and "
                f"yearsTech='{years_tech}'. The structured dropdown fields are more reliable. "
                f"When referencing this user's experience level, use the structured fields "
                f"(yearsExperience, yearsTech) as the source of truth and ask the user to "
                f"clarify the discrepancy rather than assuming either is correct."
            )

    return notes


async def get_student_onboarding() -> dict[str, Any]:
    """Get the current student's onboarding information from the LMS.

    Retrieves detailed onboarding data including learning track, preferences,
    technical background, and time commitment information. Returns a dict with:
    - learningTrack: Selected learning track (e.g., 'data-science')
    - timeCommitment: Schedule and hours per week
    - learningPreferences: Learning style, problem-solving approach, etc.
    - technicalBackground: Tools, experience level, tasks performed
    - completed: Whether onboarding is completed
    - completedSteps: List of completed onboarding steps
    """
    runtime = get_runtime(Context)

    # Get the user token from context
    token = _normalize_auth_token(runtime.context.user_token)
    if not token:
        logger.error("No user token available in context")
        return {
            "error": "Authentication required",
            "message": "Unable to fetch student onboarding without authentication token",
        }

    # Get LMS API URL from context
    lms_url = runtime.context.lms_api_url.rstrip("/")
    # Use the AI mentor onboarding snapshot endpoint because it contains
    # the complete onboarding journey, including work experience details.
    onboarding_endpoint = f"{lms_url}/api/v1/ai-mentor/onboarding/me"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            data, evidence = await _cached_lms_get_with_evidence(
                client,
                onboarding_endpoint,
                token,
                runtime.context.user_id,
            )

            # Extract and normalize onboarding data.
            onboarding_data = data.get("onboarding", {}) if isinstance(data, dict) else {}
            s1 = onboarding_data.get("s1", {}) if isinstance(onboarding_data, dict) else {}
            s7 = onboarding_data.get("s7", {}) if isinstance(onboarding_data, dict) else {}

            weekly_time = s1.get("weeklyTime") if isinstance(s1, dict) else None
            time_commitment = onboarding_data.get("timeCommitment", {})
            if not time_commitment and weekly_time:
                time_commitment = {"hoursPerWeek": weekly_time}

            learning_preferences = onboarding_data.get("learningPreferences", {})
            if not learning_preferences and (s1 or s7):
                learning_preferences = {
                    "learningStyle": s1.get("learningStyle"),
                    "feedbackStyle": s7.get("feedbackStyle"),
                    "availability": s7.get("availability"),
                    "motivators": s7.get("motivators"),
                    "riskTolerance": s7.get("riskTolerance"),
                }

            technical_background = onboarding_data.get("technicalBackground", {})
            if not technical_background:
                technical_background = {
                    "employment": onboarding_data.get("s2", {}),
                    "education": onboarding_data.get("s3", {}),
                    "skills": onboarding_data.get("s5", {}),
                    "challenges": onboarding_data.get("s6", {}),
                }

            # Structure the response with relevant fields
            onboarding = {
                "learningTrack": onboarding_data.get("learningTrack") or s1.get("learningTrack"),
                "timeCommitment": time_commitment,
                "learningPreferences": learning_preferences,
                "technicalBackground": technical_background,
                "completed": onboarding_data.get("completed"),
                "completedSteps": onboarding_data.get("completedSteps", []),
                "sourceEndpoint": "/api/v1/ai-mentor/onboarding/me",
                "sections": {
                    "s1": onboarding_data.get("s1", {}),
                    "s2": onboarding_data.get("s2", {}),
                    "s3": onboarding_data.get("s3", {}),
                    "s4": onboarding_data.get("s4", {}),
                    "s5": onboarding_data.get("s5", {}),
                    "s6": onboarding_data.get("s6", {}),
                    "s_track": onboarding_data.get("s_track", {}),
                    "s7": onboarding_data.get("s7", {}),
                    "s8": onboarding_data.get("s8", {}),
                },
                "evidence": evidence,
                "confidence": "high" if evidence.get("live_verified") else "medium",
            }

            # Flag any contradictions between freeform text and structured fields
            contradictions = _detect_onboarding_contradictions(onboarding_data)
            if contradictions:
                onboarding["data_quality_notes"] = contradictions

            logger.info(f"Successfully fetched onboarding for learning track: {onboarding.get('learningTrack')}")
            return onboarding

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching student onboarding: {e.response.status_code}")
        return {
            "error": "API request failed",
            "status_code": e.response.status_code,
            "message": str(e),
        }
    except httpx.TimeoutException:
        logger.error("Timeout while fetching student onboarding")
        return {
            "error": "Request timeout",
            "message": "The LMS API took too long to respond",
        }
    except Exception as e:
        logger.error(f"Unexpected error fetching student onboarding: {e}", exc_info=True)
        return {"error": "Unexpected error", "message": str(e)}


async def get_student_ai_career_advisor_onboarding() -> dict[str, Any]:
    """Get the student's comprehensive AI career advisor onboarding information from the LMS.

    Retrieves detailed onboarding data collected through the AI career advisor setup flow,
    including:
    - Professional situation and experience (s1: situation, weeklyTime, learningStyle)
    - Employment details (s2: employmentStatus, roleTitle, industry, yearsExperience, etc.)
    - Educational background (s3: highestEducation, fieldOfStudy, discoveredAI)
    - Career goals and timeline (s4: primaryGoal, targetRole, timeline, goalWhy)
    - Skills assessment and profiles (s5: LinkedIn, GitHub, confidentSkills, needHelpAreas)
    - Job search status (s6: appsSubmitted, interviews, biggestChallenge)
    - Learning track specialization (s_track: analytics, dataScience, dataEngineering, aiEngineering)
    - Career guidance preferences (s7: feedbackStyle, availability, motivators, riskTolerance)
    - Transformational outcomes (s8: transformationalOutcome, otherNotes)
    """
    runtime = get_runtime(Context)

    # Get the user token from context
    token = _normalize_auth_token(runtime.context.user_token)
    if not token:
        logger.error("No user token available in context")
        return {
            "error": "Authentication required",
            "message": "Unable to fetch AI career advisor onboarding without authentication token",
        }

    # Get LMS API URL from context
    lms_url = runtime.context.lms_api_url.rstrip("/")
    career_advisor_endpoint = f"{lms_url}/api/v1/ai-mentor/onboarding/me"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            data, evidence = await _cached_lms_get_with_evidence(
                client,
                career_advisor_endpoint,
                token,
                runtime.context.user_id,
            )

            # Extract the onboarding data
            onboarding_data = data.get("onboarding", {})

            # Structure the response with all onboarding sections
            career_advisor_onboarding = {
                "s1": onboarding_data.get("s1", {}),
                "s2": onboarding_data.get("s2", {}),
                "s3": onboarding_data.get("s3", {}),
                "s4": onboarding_data.get("s4", {}),
                "s5": onboarding_data.get("s5", {}),
                "s6": onboarding_data.get("s6", {}),
                "s_track": onboarding_data.get("s_track", {}),
                "s7": onboarding_data.get("s7", {}),
                "s8": onboarding_data.get("s8", {}),
                "learningTrack": onboarding_data.get("learningTrack"),
                "completedSteps": onboarding_data.get("completedSteps", []),
                "completed": onboarding_data.get("completed"),
                "sourceEndpoint": "/api/v1/ai-mentor/onboarding/me",
                "evidence": evidence,
                "confidence": "high" if evidence.get("live_verified") else "medium",
            }

            # Flag any contradictions between freeform text and structured fields
            contradictions = _detect_onboarding_contradictions(onboarding_data)
            if contradictions:
                career_advisor_onboarding["data_quality_notes"] = contradictions

            logger.info(
                f"Successfully fetched AI career advisor onboarding, completed: {career_advisor_onboarding.get('completed')}"
            )
            return career_advisor_onboarding

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching AI career advisor onboarding: {e.response.status_code}")
        return {
            "error": "API request failed",
            "status_code": e.response.status_code,
            "message": str(e),
        }
    except httpx.TimeoutException:
        logger.error("Timeout while fetching AI career advisor onboarding")
        return {
            "error": "Request timeout",
            "message": "The LMS API took too long to respond",
        }
    except Exception as e:
        logger.error(
            f"Unexpected error fetching AI career advisor onboarding: {e}",
            exc_info=True,
        )
        return {"error": "Unexpected error", "message": str(e)}


async def get_student_enrollment_overview() -> dict[str, Any]:
    """Get dashboard-grade enrollment and progress overview for the current student."""
    runtime = get_runtime(Context)
    user_id = runtime.context.user_id
    if not user_id:
        return {
            "ok": False,
            "source": "mongo",
            "confidence": "low",
            "error": "Authentication required",
            "message": "Unable to fetch enrollment overview without authenticated user context",
        }

    try:
        mongo_client = get_course_content_mongo_client()
        data = await _mongo_call(mongo_client.get_enrollment_overview, user_id)
        return {
            "ok": True,
            "source": "mongo",
            "confidence": "high",
            "data": data,
        }
    except Exception as e:
        logger.error(f"Error fetching enrollment overview from Mongo: {e}", exc_info=True)
        return {
            "ok": False,
            "source": "mongo",
            "confidence": "low",
            "error": str(e),
        }


async def get_course_structure(course_id: str) -> dict[str, Any]:
    """Get ordered module/lesson structure with lock and completion state for a course."""
    runtime = get_runtime(Context)
    user_id = runtime.context.user_id
    if not course_id:
        return {
            "ok": False,
            "source": "mongo",
            "confidence": "low",
            "error": "course_id is required",
        }
    try:
        _validate_id(course_id, "course_id")
    except ValueError as exc:
        return {"ok": False, "source": "mongo", "confidence": "low", "error": str(exc)}
    if not user_id:
        return {
            "ok": False,
            "source": "mongo",
            "confidence": "low",
            "error": "Authentication required",
        }

    enrolled_course_ids = list(runtime.context.enrolled_course_ids or [])
    try:
        mongo_client = get_course_content_mongo_client()
        if not enrolled_course_ids:
            enrolled_course_ids = await _mongo_call(mongo_client.get_active_enrolled_course_ids, user_id)
        if course_id not in enrolled_course_ids:
            return {
                "ok": False,
                "source": "mongo",
                "confidence": "low",
                "error": "Requested course is outside the user's enrolled scope.",
                "enrolled_course_ids": enrolled_course_ids,
            }

        data = await _mongo_call(mongo_client.get_course_structure, user_id, course_id)
        return {
            "ok": data is not None,
            "source": "mongo",
            "confidence": "high" if data is not None else "low",
            "data": data,
        }
    except Exception as e:
        logger.error(f"Error fetching course structure from Mongo: {e}", exc_info=True)
        return {
            "ok": False,
            "source": "mongo",
            "confidence": "low",
            "error": str(e),
        }


async def get_course_progress(course_id: str) -> dict[str, Any]:
    """Get detailed per-level/module/lesson progress for a course."""
    runtime = get_runtime(Context)
    user_id = runtime.context.user_id
    if not course_id:
        return {
            "ok": False,
            "source": "mongo",
            "confidence": "low",
            "error": "course_id is required",
        }
    try:
        _validate_id(course_id, "course_id")
    except ValueError as exc:
        return {"ok": False, "source": "mongo", "confidence": "low", "error": str(exc)}
    if not user_id:
        return {
            "ok": False,
            "source": "mongo",
            "confidence": "low",
            "error": "Authentication required",
        }

    enrolled_course_ids = list(runtime.context.enrolled_course_ids or [])
    try:
        mongo_client = get_course_content_mongo_client()
        if not enrolled_course_ids:
            enrolled_course_ids = await _mongo_call(mongo_client.get_active_enrolled_course_ids, user_id)
        if course_id not in enrolled_course_ids:
            return {
                "ok": False,
                "source": "mongo",
                "confidence": "low",
                "error": "Requested course is outside the user's enrolled scope.",
                "enrolled_course_ids": enrolled_course_ids,
            }

        data = await _mongo_call(mongo_client.get_course_progress, user_id, course_id)
        return {
            "ok": data is not None,
            "source": "mongo",
            "confidence": "high" if data is not None else "low",
            "data": data,
        }
    except Exception as e:
        logger.error(f"Error fetching course progress from Mongo: {e}", exc_info=True)
        return {
            "ok": False,
            "source": "mongo",
            "confidence": "low",
            "error": str(e),
        }


async def get_course_materials(course_id: str, include_content: bool = True, limit: int = 10) -> dict[str, Any]:
    """Get enrolled course materials and extract readable text from PDFs when possible."""
    runtime = get_runtime(Context)
    user_id = runtime.context.user_id
    if not course_id:
        return {
            "ok": False,
            "source": "mongo",
            "confidence": "low",
            "error": "course_id is required",
        }
    if not user_id:
        return {
            "ok": False,
            "source": "mongo",
            "confidence": "low",
            "error": "Authentication required",
        }

    enrolled_course_ids = list(runtime.context.enrolled_course_ids or [])
    try:
        mongo_client = get_course_content_mongo_client()
        if not enrolled_course_ids:
            enrolled_course_ids = await _mongo_call(mongo_client.get_active_enrolled_course_ids, user_id)
        if course_id not in enrolled_course_ids:
            return {
                "ok": False,
                "source": "mongo",
                "confidence": "low",
                "error": "Requested course is outside the user's enrolled scope.",
                "enrolled_course_ids": enrolled_course_ids,
            }

        materials = await _mongo_call(
            mongo_client.get_course_materials,
            course_id,
            include_content=include_content,
            limit=max(1, min(limit, 20)),
        )
        data = {
            "courseId": course_id,
            "materials": [
                {
                    "materialId": item.material_id,
                    "lessonId": item.lesson_id,
                    "lessonTitle": item.lesson_title,
                    "levelTitle": item.level_title,
                    "moduleTitle": item.module_title,
                    "title": item.raw.get("title") if isinstance(item.raw, dict) else None,
                    "fileType": item.file_type,
                    "fileUrl": item.file_url,
                    "downloadUrl": item.download_url,
                    "contentPreview": (item.content_text or "")[:8000] if include_content else None,
                }
                for item in materials
            ],
        }
        return {
            "ok": True,
            "source": "mongo",
            "confidence": "high",
            "data": data,
        }
    except Exception as e:
        logger.error(f"Error fetching course materials from Mongo: {e}", exc_info=True)
        return {
            "ok": False,
            "source": "mongo",
            "confidence": "low",
            "error": str(e),
        }


async def get_student_attempts(student_id: str) -> dict[str, Any]:
    """Get assessment attempts history and result context for a student."""
    runtime = get_runtime(Context)
    user_id = runtime.context.user_id
    if not student_id:
        return {
            "ok": False,
            "source": "mongo",
            "confidence": "low",
            "error": "student_id is required",
        }
    try:
        _validate_id(student_id, "student_id")
    except ValueError as exc:
        return {"ok": False, "source": "mongo", "confidence": "low", "error": str(exc)}
    if not user_id or student_id != user_id:
        return {
            "ok": False,
            "source": "mongo",
            "confidence": "low",
            "error": "Requested student is outside the authenticated scope.",
        }

    try:
        mongo_client = get_course_content_mongo_client()
        data = await _mongo_call(mongo_client.get_student_attempts, user_id)
        return {
            "ok": True,
            "source": "mongo",
            "confidence": "high",
            "data": data,
        }
    except Exception as e:
        logger.error(f"Error fetching student attempts from Mongo: {e}", exc_info=True)
        return {
            "ok": False,
            "source": "mongo",
            "confidence": "low",
            "error": str(e),
        }


async def get_subscription_state() -> dict[str, Any]:
    """Get current subscription entitlement and active track state."""
    runtime = get_runtime(Context)
    user_id = runtime.context.user_id
    if not user_id:
        return {
            "ok": False,
            "source": "mongo",
            "confidence": "low",
            "error": "Authentication required",
        }

    try:
        mongo_client = get_course_content_mongo_client()
        data = await _mongo_call(mongo_client.get_subscription_state, user_id)
        return {
            "ok": data is not None,
            "source": "mongo",
            "confidence": "high" if data is not None else "low",
            "data": data,
        }
    except Exception as e:
        logger.error(f"Error fetching subscription state from Mongo: {e}", exc_info=True)
        return {
            "ok": False,
            "source": "mongo",
            "confidence": "low",
            "error": str(e),
        }


async def search_course_content(
    query: str,
    course_id: str | None = None,
    max_results: int = 5,
) -> dict[str, Any]:
    """Search enrolled course content directly from Mongo Atlas Search.

    This tool searches through indexed course materials, lessons, and transcript
    segments using full-text retrieval and BM25-style ranking fused with lexical
    exact-match signals. Use this when:
    - Students ask about specific course topics or concepts
    - Looking for explanations from course materials
    - Finding relevant lessons or modules
    - Retrieving course-specific information

    Args:
        query: The search query (e.g., "What is machine learning?", "SQL joins tutorial")
        course_id: Optional course ID to search within a specific course
        max_results: Maximum number of results to return (default: 5)

    Returns:
        A dict containing:
        - query: The original search query
        - results: List of relevant course content chunks with:
            - content: The relevant text content
            - title: Title of the lesson/material
            - course_id: ID of the course
            - content_type: Type (lesson, material, course_description)
            - metadata: Additional context (level, module, etc.)
        - error: Error message if search fails
    """
    runtime = get_runtime(Context)

    if not COURSE_CONTENT_AVAILABLE:
        logger.error("Course content services are not available")
        return {
            "query": query,
            "results": [],
            "error": "Course search is not available. Course content services are not initialized.",
        }

    try:
        user_id = runtime.context.user_id
        if not user_id:
            return {
                "query": query,
                "results": [],
                "error": "User context is missing. Cannot resolve enrollment scope.",
            }

        enrolled_course_ids = list(runtime.context.enrolled_course_ids or [])
        mongo_client = get_course_content_mongo_client()
        search_service = get_course_content_search_service()

        if not enrolled_course_ids:
            enrolled_course_ids = await _mongo_call(mongo_client.get_active_enrolled_course_ids, user_id)

        if course_id and course_id not in enrolled_course_ids:
            return {
                "query": query,
                "results": [],
                "error": "Requested course is outside the user's enrolled scope.",
                "enrolled_course_ids": enrolled_course_ids,
            }

        if not enrolled_course_ids:
            return {
                "query": query,
                "results": [],
                "message": "No active enrollments found for this user.",
            }

        results = await _mongo_call(
            search_service.search_enrolled_course_content,
            query=query,
            enrolled_course_ids=enrolled_course_ids,
            limit=max_results,
            requested_course_id=course_id,
        )

        if not results:
            logger.info(f"No course content found for query: {query}")
            return {
                "query": query,
                "results": [],
                "message": "No relevant enrolled course content found in Mongo search.",
                "enrolled_course_ids": enrolled_course_ids,
            }

        logger.info(f"Found {len(results)} relevant course content chunks")
        return {
            "query": query,
            "results": results,
            "total_results": len(results),
            "enrolled_course_ids": enrolled_course_ids,
            "source": results[0].get("metadata", {}).get("source", "mongo_atlas_search"),
        }

    except Exception as e:
        logger.error(f"Error searching course content: {str(e)}", exc_info=True)
        return {
            "query": query,
            "results": [],
            "error": f"Course search failed: {str(e)}",
        }


async def _fetch_github_profile(username: str) -> dict[str, Any]:
    """Fetch a public GitHub profile and top repos via the GitHub REST API.

    The GitHub API allows unauthenticated access to public data (60 req/hour).
    This avoids the browser-session requirement of the github.com profile page.
    """
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "DeDataHubBot/1.0",
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        # Fetch profile and repos in parallel
        profile_resp, repos_resp = await asyncio.gather(
            client.get(f"https://api.github.com/users/{username}", headers=headers),
            client.get(
                f"https://api.github.com/users/{username}/repos",
                headers=headers,
                params={"sort": "pushed", "per_page": 10, "type": "owner"},
            ),
            return_exceptions=True,
        )

    profile: dict[str, Any] = {}
    if not isinstance(profile_resp, Exception):
        profile_resp.raise_for_status()
        p = profile_resp.json()
        profile = {
            "name": p.get("name"),
            "bio": p.get("bio"),
            "company": p.get("company"),
            "location": p.get("location"),
            "blog": p.get("blog"),
            "public_repos": p.get("public_repos"),
            "followers": p.get("followers"),
        }

    repos: list[dict[str, Any]] = []
    if not isinstance(repos_resp, Exception):
        repos_resp.raise_for_status()
        for r in repos_resp.json():
            if r.get("fork"):
                continue  # skip forks — own work only
            repos.append(
                {
                    "name": r.get("name"),
                    "description": r.get("description"),
                    "language": r.get("language"),
                    "stars": r.get("stargazers_count"),
                    "forks": r.get("forks_count"),
                    "topics": r.get("topics", []),
                    "updated_at": (r.get("pushed_at") or "")[:10],
                }
            )

    return {
        "source": "github_api",
        "username": username,
        "profile": profile,
        "repos": repos[:8],
    }


def _normalize_url_for_match(url: str) -> str:
    """Scheme/www/trailing-slash-insensitive form for comparing stored reference URLs."""
    cleaned = url.strip().lower().rstrip("/")
    cleaned = cleaned.removeprefix("https://").removeprefix("http://")
    return cleaned.removeprefix("www.")


async def _stored_reference_note(url: str) -> str | None:
    """Item 3 're-verify on use': flag a dead URL that a saved ReferenceMemory points at.

    ReferenceMemory carries no time-based staleness window — a link is only
    known bad when actually used. Best-effort; never raises.
    """
    try:
        runtime = get_runtime(Context)
        store = runtime.store
        user_id = runtime.context.user_id
        if not store or not user_id:
            return None
        items = await store.asearch((user_id, DEFAULT_MEMORY_NAMESPACE), filter={"kind": "ReferenceMemory"}, limit=20)
    except Exception:
        logger.debug("Stored-reference lookup failed; skipping dead-link note.", exc_info=True)
        return None

    target = _normalize_url_for_match(url)
    for item in items:
        content = item.value.get("content", {})
        if not isinstance(content, dict):
            continue
        location = str(content.get("location") or "")
        if location and _normalize_url_for_match(location) == target:
            resource = content.get("resource") or "a saved reference"
            return (
                f"This URL is stored in memory as a reference ({resource}) and appears to be dead. "
                "Confirm the correct link with the student and update that reference via manage_memory "
                "so future sessions don't rely on a broken link."
            )
    return None


async def read_webpage(url: str) -> dict[str, Any]:
    """Fetch and summarize a webpage so advice can be based on page content, not just snippet text."""
    if not isinstance(url, str) or not url.strip():
        return {"error": "invalid_url", "message": "A valid URL is required."}

    cleaned_url = url.strip()
    if not cleaned_url.startswith(("http://", "https://")):
        cleaned_url = f"https://{cleaned_url}"

    parsed = urlparse(cleaned_url)
    hostname = parsed.hostname or ""

    # GitHub profile pages require a browser session to render, but the REST API
    # provides richer structured data for public profiles — use it instead.
    if hostname in ("github.com", "www.github.com"):
        path_parts = [p for p in parsed.path.strip("/").split("/") if p]
        if path_parts:
            username = path_parts[0]
            try:
                return await _fetch_github_profile(username)
            except Exception as e:
                logger.warning(f"GitHub API fetch failed for {username}: {e}")
                return {
                    "error": "github_api_error",
                    "message": "GitHub profile could not be fetched. Use the self-reported onboarding fields instead.",
                    "url": cleaned_url,
                }
        return {
            "error": "invalid_github_url",
            "message": "Could not extract a GitHub username from the URL.",
            "url": cleaned_url,
        }

    # LinkedIn blocks direct HTTP access (HTTP 999 + TLS fingerprinting).
    # Strategy (three tiers, stop at the first that returns real content):
    #   Tier 1 — Jina Reader: renders JS and bypasses many bot-protection layers;
    #             works for public profiles that don't require login.
    #   Tier 2 — Brave Search with exact-URL query: searching the full URL in quotes
    #             finds cached/indexed snippets far more reliably than site: searches.
    #   Tier 3 — Brave Search with slug/name query: slug-only as a last attempt.
    if hostname in ("linkedin.com", "www.linkedin.com") or hostname.endswith(".linkedin.com"):
        path_parts = [p for p in parsed.path.strip("/").split("/") if p]
        # Extract the identifying slug from the path
        # /in/<username>   /company/<slug>   /posts/<id>   → use second segment
        # anything else → use first segment
        if len(path_parts) >= 2 and path_parts[0] in ("in", "posts", "company"):
            slug = path_parts[1]
        elif path_parts:
            slug = path_parts[0]
        else:
            slug = ""

        # Tier 1: Jina Reader — best chance of returning real structured content
        jina_url = f"https://r.jina.ai/{cleaned_url}"
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=25.0) as client:
                jina_resp = await client.get(
                    jina_url,
                    headers={
                        "Accept": "text/plain, text/markdown",
                        "X-Return-Format": "markdown",
                        "X-Remove-Selector": "nav, footer, header, .cookie-banner, #cookie-notice",
                        "User-Agent": "DeDataHubBot/1.0 (+https://dedatahub.io)",
                    },
                )
                if jina_resp.status_code == 200:
                    content = jina_resp.text.strip()
                    # Real profiles are substantial; login redirect pages are short
                    # and contain "sign in" / "join" near the top.
                    is_login_wall = len(content) < 800 or any(
                        phrase in content[:400].lower()
                        for phrase in ("sign in", "join now", "join linkedin", "log in to")
                    )
                    if content and not is_login_wall:
                        logger.info(f"Jina Reader fetched LinkedIn profile ({len(content)} chars)")
                        return {
                            "source": "jina_reader",
                            "url": cleaned_url,
                            "content": content[:8000],
                            "content_length": len(content),
                            "truncated": len(content) > 8000,
                        }
        except Exception as e:
            logger.warning(f"Jina Reader failed for LinkedIn URL {cleaned_url}: {e}")

        # Tier 2: Brave Search — exact URL in quotes (finds cached/federated results)
        # "site:linkedin.com" is heavily de-indexed; the full URL query works much better.
        try:
            query = f'"{cleaned_url}"'
            search_result = await brave_search(query)
            if search_result and not search_result.startswith("Search failed"):
                logger.info(f"Brave Search (exact URL) returned LinkedIn content for {cleaned_url}")
                return {
                    "source": "brave_search_linkedin",
                    "url": cleaned_url,
                    "note": "LinkedIn blocks direct access; content retrieved via Brave Search index.",
                    "content": search_result,
                }
        except Exception as e:
            logger.warning(f"Brave Search (exact URL) failed for LinkedIn {cleaned_url}: {e}")

        # Tier 3: Brave Search — slug-based query as final attempt
        if slug:
            try:
                query = f"{slug} linkedin profile"
                search_result = await brave_search(query)
                if search_result and not search_result.startswith("Search failed"):
                    logger.info(f"Brave Search (slug) returned LinkedIn content for {slug}")
                    return {
                        "source": "brave_search_linkedin",
                        "url": cleaned_url,
                        "note": "LinkedIn blocks direct access; content retrieved via Brave Search index.",
                        "content": search_result,
                    }
            except Exception as e:
                logger.warning(f"Brave Search (slug) failed for LinkedIn {cleaned_url}: {e}")

        return {
            "error": "linkedin_unavailable",
            "message": (
                "LinkedIn blocks automated access and all retrieval methods failed for this profile. "
                "Ask the student to paste their LinkedIn 'About' section, work experience, "
                "and key skills directly into the chat so you can give personalised advice based on their actual background."
            ),
            "url": cleaned_url,
        }

    # -----------------------------------------------------------------------
    # General webpage reading — primary: Jina Reader (r.jina.ai).
    # Jina renders JavaScript, bypasses many bot-protection layers, and returns
    # clean markdown. Works for SPAs, news articles, job boards, Twitter/X, etc.
    # No API key required for basic use (rate-limited to ~20 req/min on free tier).
    # Fallback: plain httpx GET for static / server-rendered pages.
    # -----------------------------------------------------------------------
    jina_url = f"https://r.jina.ai/{cleaned_url}"
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as client:
            jina_resp = await client.get(
                jina_url,
                headers={
                    "Accept": "text/plain, text/markdown",
                    "User-Agent": "DeDataHubBot/1.0 (+https://dedatahub.io)",
                    "X-Return-Format": "markdown",
                    "X-Remove-Selector": "nav, footer, header, .cookie-banner, #cookie-notice",
                },
            )
            if jina_resp.status_code == 200:
                content = jina_resp.text.strip()
                if content and len(content) > 100:
                    logger.info(f"Jina Reader fetched {cleaned_url} ({len(content)} chars)")
                    return {
                        "source": "jina_reader",
                        "url": cleaned_url,
                        "content": content[:8000],
                        "content_length": len(content),
                        "truncated": len(content) > 8000,
                    }
            logger.warning(f"Jina returned {jina_resp.status_code} for {cleaned_url}, falling back to httpx")
    except Exception as e:
        logger.warning(f"Jina Reader failed for {cleaned_url}: {e} — falling back to direct fetch")

    # Fallback: plain httpx (best-effort for static / server-rendered pages)
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=12.0) as client:
            response = await client.get(
                cleaned_url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; DeDataHubBot/1.0; +https://dedatahub.io)",
                    "Accept": "text/html,application/xhtml+xml,text/plain",
                },
            )
            response.raise_for_status()

            content_type = (response.headers.get("content-type") or "").lower()
            raw = response.text[:180000]

            title_match = re.search(r"(?is)<title[^>]*>(.*?)</title>", raw)
            title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else ""

            meta_desc_match = re.search(
                r'(?is)<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']',
                raw,
            )
            meta_description = re.sub(r"\s+", " ", meta_desc_match.group(1)).strip() if meta_desc_match else ""

            # Convert HTML to plain text quickly for model use.
            text = re.sub(r"(?is)<script.*?>.*?</script>", " ", raw)
            text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
            text = re.sub(r"(?s)<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()

            return {
                "source": "direct_fetch",
                "url": str(response.url),
                "status_code": response.status_code,
                "content_type": content_type,
                "title": title,
                "meta_description": meta_description,
                "content_preview": text[:5000],
            }

    except httpx.HTTPStatusError as e:
        result: dict[str, Any] = {
            "error": "http_error",
            "status_code": e.response.status_code,
            "message": (
                f"Unable to access the page ({e.response.status_code}). "
                "The site may require authentication or block automated access."
            ),
            "url": cleaned_url,
        }
        memory_note = await _stored_reference_note(cleaned_url)
        if memory_note:
            result["memory_note"] = memory_note
        return result
    except httpx.TimeoutException:
        return {
            "error": "timeout",
            "message": "The page timed out before it could be read.",
            "url": cleaned_url,
        }
    except Exception as e:
        return {
            "error": "unexpected_error",
            "message": str(e),
            "url": cleaned_url,
        }


async def get_portfolio_projects() -> dict[str, Any]:
    """Get the student's submitted project history from the LMS.

    Uses the canonical student submissions endpoint exposed by the LMS.
    This gives the advisor real project evidence to reference when checking
    portfolio coherence across modules.
    """
    runtime = get_runtime(Context)

    token = _normalize_auth_token(runtime.context.user_token)
    if not token:
        logger.error("No user token available in context")
        return {
            "error": "Authentication required",
            "message": "Unable to fetch project submissions without authentication token",
        }

    lms_url = runtime.context.lms_api_url.rstrip("/")
    endpoint = f"{lms_url}/api/v1/courses/projects/my-submissions"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                endpoint,
                headers={"accept": "application/json", "Authorization": f"Bearer {token}"},
            )
            resp.raise_for_status()
            data: dict[str, Any] | list[dict[str, Any]] = resp.json()
            submissions: list[dict[str, Any]]
            if isinstance(data, dict):
                if isinstance(data.get("submissions"), list):
                    submissions = data["submissions"]
                elif isinstance(data.get("data"), list):
                    submissions = data["data"]
                elif isinstance(data.get("projects"), list):
                    submissions = data["projects"]
                else:
                    submissions = []
            else:
                submissions = data
            logger.info(f"Fetched {len(submissions)} project submissions")
            return {
                "ok": True,
                "submissions": submissions,
                "count": len(submissions),
                "sourceEndpoint": "/api/v1/courses/projects/my-submissions",
            }

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching project submissions: {e.response.status_code}")
        return {
            "error": "API request failed",
            "status_code": e.response.status_code,
            "message": str(e),
        }
    except httpx.TimeoutException:
        logger.error("Timeout while fetching project submissions")
        return {
            "error": "Request timeout",
            "message": "The LMS API took too long to respond",
        }
    except Exception as e:
        logger.error(f"Unexpected error fetching project submissions: {e}", exc_info=True)
        return {"error": "Unexpected error", "message": str(e)}


async def review_project_submission(
    submission_id: str,
    feedback: str,
    reviewed: bool = True,
) -> dict[str, Any]:
    """Review a student's submitted project via the LMS admin route.

    This tool must use the admin token because the backing LMS endpoint is
    restricted to admin users. Use it after preparing the structured review so
    the review is persisted against the student's submission record.
    """
    runtime = get_runtime(Context)

    try:
        from aegra_api.services.admin_auth import admin_token_manager

        token = await admin_token_manager.get_token()
    except Exception as exc:
        logger.error(f"Failed to obtain admin token: {exc}")
        return {
            "error": "Authentication required",
            "message": "Unable to review project submission without admin authentication token",
        }

    if not submission_id.strip():
        return {
            "error": "Invalid submission ID",
            "message": "submission_id is required",
        }

    if not feedback.strip():
        return {
            "error": "Invalid feedback",
            "message": "feedback is required",
        }

    lms_url = runtime.context.lms_api_url.rstrip("/")
    endpoint = f"{lms_url}/api/v1/courses/projects/review/{submission_id}"
    payload = {"feedback": feedback.strip(), "reviewed": reviewed}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.patch(
                endpoint,
                json=payload,
                headers={
                    "accept": "application/json",
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}",
                },
            )
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
            logger.info(f"Reviewed project submission: {submission_id}")
            return {
                "ok": True,
                "submissionId": submission_id,
                "review": data,
                "sourceEndpoint": "/api/v1/courses/projects/review/{submissionId}",
            }

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error reviewing project submission: {e.response.status_code}")
        return {
            "error": "API request failed",
            "status_code": e.response.status_code,
            "message": str(e),
        }
    except httpx.TimeoutException:
        logger.error("Timeout while reviewing project submission")
        return {
            "error": "Request timeout",
            "message": "The LMS API took too long to respond",
        }
    except Exception as e:
        logger.error(f"Unexpected error reviewing project submission: {e}", exc_info=True)
        return {"error": "Unexpected error", "message": str(e)}


async def search_past_conversations(query: str, limit: int = 5) -> dict[str, Any]:
    """Search through past conversation histories to recall relevant discussions.

    Use this when the student references something from a previous session that is
    not captured in long-term memories (search_memory). This searches the detailed
    session notes kept for every prior conversation thread.

    When to use:
    - Student says "last time we talked about..." or "you helped me with X before"
    - You need continuity from a prior session to resume a project or plan
    - search_memory() returns nothing but the student insists the topic was discussed
    - Checking whether a specific project, skill, or goal came up in earlier sessions

    Args:
        query: Natural language description of what you are looking for.
        limit: Number of past threads to retrieve (1-10, default 5).
    """
    runtime = get_runtime(Context)
    user_id = runtime.context.user_id
    store = runtime.store

    if not user_id or not store:
        return {"found": 0, "message": "Conversation history not available.", "results": []}

    limit = min(max(1, limit), 10)

    try:
        results = await store.asearch(
            (user_id, SESSION_MEMORY_NAMESPACE_SUFFIX),
            query=query,
            limit=limit,
        )

        if not results:
            return {
                "found": 0,
                "message": "No relevant past conversations found.",
                "results": [],
            }

        formatted: list[dict[str, Any]] = []
        for item in results:
            thread_id = item.namespace[-1] if len(item.namespace) >= 3 else "unknown"
            notes = item.value.get("notes", "")
            thread_name = item.value.get("thread_name")
            updated_at = item.value.get("updated_at")

            entry: dict[str, Any] = {"thread_id": thread_id, "notes": notes}
            if thread_name:
                entry["thread_name"] = thread_name
            if updated_at:
                entry["last_updated"] = updated_at
            formatted.append(entry)

        return {"found": len(formatted), "query": query, "results": formatted}

    except Exception as e:
        logger.error("Failed to search past conversations: %s", e, exc_info=True)
        return {"error": "Search failed", "message": str(e), "results": []}


async def get_thread_summary_by_title(title: str) -> dict[str, Any]:
    """Look up the summary/notes for a past conversation thread by its title.

    Use this when the student references a specific named conversation, e.g.
    "the chat called 'Personalised Career Roadmap Development'" or
    "read the thread titled 'My Data Engineering Plan'".

    This performs a case-insensitive substring match against stored thread names
    and returns the session notes (a structured summary of what was discussed).

    Args:
        title: The thread title or a partial title to search for.
    """
    runtime = get_runtime(Context)
    user_id = runtime.context.user_id
    store = runtime.store

    if not user_id or not store:
        return {"found": 0, "message": "Conversation history not available.", "results": []}

    title_lower = title.lower().strip()

    try:
        all_items = await store.asearch(
            (user_id, SESSION_MEMORY_NAMESPACE_SUFFIX),
            limit=200,
        )
    except Exception as e:
        logger.error("Failed to list session notes: %s", e, exc_info=True)
        return {"error": "Lookup failed", "message": str(e), "results": []}

    matches: list[dict[str, Any]] = []
    all_titles: list[str] = []

    for item in all_items:
        thread_name: str = item.value.get("thread_name", "")
        if thread_name:
            all_titles.append(thread_name)
        if thread_name and title_lower in thread_name.lower():
            thread_id = item.namespace[-1] if len(item.namespace) >= 3 else "unknown"
            entry: dict[str, Any] = {
                "thread_id": thread_id,
                "thread_name": thread_name,
                "notes": item.value.get("notes", ""),
            }
            updated_at = item.value.get("updated_at")
            if updated_at:
                entry["last_updated"] = updated_at
            matches.append(entry)

    if matches:
        return {"found": len(matches), "query": title, "results": matches}

    # No exact substring match — return known thread titles so the agent
    # can inform the student which threads are available.
    if all_titles:
        return {
            "found": 0,
            "message": f"No thread with a title containing '{title}' was found.",
            "available_thread_titles": all_titles[:20],
            "results": [],
        }

    return {
        "found": 0,
        "message": "No past conversation threads with saved summaries were found.",
        "results": [],
    }


# ---------------------------------------------------------------------------
# Opportunities service (jobs + events)
# ---------------------------------------------------------------------------
try:
    from aegra_api.core.orm import get_session_maker as _get_session_maker
    from aegra_api.services.opportunity_service import OpportunityService as _OpportunityService

    OPPORTUNITIES_AVAILABLE = True
    logger.info("Opportunities service loaded successfully")
except ImportError as _e:
    OPPORTUNITIES_AVAILABLE = False
    logger.warning(f"Opportunities service not available: {_e}. Opportunity tools will be disabled.")


async def get_opportunities(
    opportunity_type: str | None = None,
    status: str = "new",
    limit: int = 10,
) -> dict[str, Any]:
    """Get the student's personalised job and event opportunities discovered by the platform.

    Returns a ranked list of opportunities matched to the student's learning track.
    Call this whenever the student asks about job openings, hiring opportunities, events,
    or wants to know what roles are available for them.

    Args:
        opportunity_type: Filter by 'job' or 'event'. Omit to return both.
        status: Which opportunities to show — 'new' (default, includes notified),
                'saved' (bookmarked), 'applied', or 'dismissed'.
        limit: How many to return (1–50, default 10).
    """
    runtime = get_runtime(Context)
    user_id = runtime.context.user_id

    if not user_id:
        return {
            "error": "Authentication required",
            "message": "Cannot fetch opportunities without user context.",
        }

    if not OPPORTUNITIES_AVAILABLE:
        return {
            "error": "Opportunities service unavailable",
            "message": "The opportunities backend is not reachable from this environment.",
        }

    limit = min(max(1, limit), 50)
    opp_type = opportunity_type.strip().lower() if opportunity_type else None
    if opp_type not in {None, "job", "event"}:
        return {"error": "opportunity_type must be 'job', 'event', or omitted for all"}

    try:
        session_maker = _get_session_maker()
        async with session_maker() as session:
            opportunities, total, has_more = await _OpportunityService.list_opportunities(
                session=session,
                user_id=user_id,
                opportunity_type=opp_type,
                status=status,
                limit=limit,
            )

        return {
            "ok": True,
            "total": total,
            "returned": len(opportunities),
            "has_more": has_more,
            "status_filter": status,
            "type_filter": opp_type or "all",
            "opportunities": [
                {
                    "id": opp.id,
                    "type": opp.opportunity_type,
                    "title": opp.title,
                    "company": opp.company,
                    "location": opp.location,
                    "url": opp.url,
                    "salary_range": opp.salary_range,
                    "match_score": float(opp.match_score) if opp.match_score else None,
                    "matched_track": opp.matched_track,
                    "status": opp.status,
                    "event_date": opp.event_date.isoformat() if opp.event_date else None,
                    "discovered_at": opp.discovered_at.isoformat() if opp.discovered_at else None,
                    "description": (opp.description or "")[:400] if opp.description else None,
                }
                for opp in opportunities
            ],
        }

    except Exception as e:
        logger.error("Error fetching opportunities for user=%s: %s", user_id, e, exc_info=True)
        return {"error": "Failed to fetch opportunities", "message": str(e)}


async def get_opportunity_strategy(opportunity_id: str) -> dict[str, Any]:
    """Get the AI-generated application or networking strategy for a specific opportunity.

    For job opportunities this returns a tailored application strategy — how to position
    the student's skills, what to highlight in a cover letter, and how to approach the role.
    For events it returns a networking strategy.

    Call this after get_opportunities() when the student wants to act on a specific
    opportunity and needs personalised guidance on how to pursue it.

    Args:
        opportunity_id: The ID of the opportunity (from get_opportunities results).
    """
    runtime = get_runtime(Context)
    user_id = runtime.context.user_id

    if not user_id:
        return {
            "error": "Authentication required",
            "message": "Cannot fetch strategy without user context.",
        }

    if not OPPORTUNITIES_AVAILABLE:
        return {
            "error": "Opportunities service unavailable",
            "message": "The opportunities backend is not reachable from this environment.",
        }

    if not _validate_id(opportunity_id):
        return {"error": "Invalid opportunity ID", "opportunity_id": opportunity_id}

    try:
        session_maker = _get_session_maker()
        async with session_maker() as session:
            data = await _OpportunityService.get_opportunity_with_strategy(session, opportunity_id, user_id)

        if not data:
            return {
                "error": "Opportunity not found",
                "opportunity_id": opportunity_id,
                "message": "This opportunity may have expired or doesn't belong to this user.",
            }

        return {"ok": True, **data}

    except Exception as e:
        logger.error(
            "Error fetching strategy for opportunity=%s user=%s: %s",
            opportunity_id,
            user_id,
            e,
            exc_info=True,
        )
        return {"error": "Failed to fetch strategy", "message": str(e)}


# ---------------------------------------------------------------------------
# TaskMemory (accountability ledger) — agent optimisation spec Item 2
# ---------------------------------------------------------------------------
# Backed by the existing action_items Postgres table (aegra_api), not a new
# LangGraph-store memory schema — that table already had a service layer,
# REST API, and downstream readers (notification_engine's deadline reminders,
# struggle detection) but no write path. These tools are that write path.
try:
    from aegra_api.core.orm import get_session_maker as _get_task_session_maker
    from aegra_api.services.accountability_service import AccountabilityService as _AccountabilityService

    TASK_MEMORY_AVAILABLE = True
    logger.info("Task memory (accountability) service loaded successfully")
except ImportError as _e:
    TASK_MEMORY_AVAILABLE = False
    logger.warning(f"Task memory service not available: {_e}. Task tools will be disabled.")

_TASK_UPDATE_STATUSES = frozenset({"in_progress", "completed", "skipped", "abandoned", "renegotiated"})
_TASK_PRIORITIES = frozenset({"high", "normal", "low"})


def _serialize_action_item(item: Any) -> dict[str, Any]:
    return {
        "id": item.id,
        "description": item.description,
        "status": item.status,
        "due_date": item.due_date.isoformat() if item.due_date else None,
        "priority": item.priority,
        "category": item.category,
        "evidence": item.evidence,
        "miss_count": item.miss_count,
        "created_at": item.created_at.isoformat() if item.created_at else None,
    }


def _parse_due_date(due_date: str | None) -> Any:
    """Parse an ISO date/datetime string. Returns None for empty or unparseable input.

    Callers pass free-text timeframes ("by next session") through
    ``category``/``description`` instead — only real dates belong here.
    """
    if not due_date or not due_date.strip():
        return None
    try:
        parsed = datetime.fromisoformat(due_date.strip())
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


async def fetch_open_task_group_for_prompt(user_id: str) -> dict[str, list[dict[str, Any]]] | None:
    """Fetch and serialize a user's open/overdue tasks. Returns ``None`` on any failure.

    Shared by the ``get_open_tasks`` tool (hot path, agent-initiated) and
    ``call_model``'s conversation-start injection (graph.py) — both need the
    exact same query and serialization, so this is the single implementation.
    """
    if not TASK_MEMORY_AVAILABLE:
        return None
    try:
        session_maker = _get_task_session_maker()
        async with session_maker() as session:
            group = await _AccountabilityService.get_open_and_overdue(session, user_id)
        return {
            "not_yet_due": [_serialize_action_item(item) for item in group.not_yet_due],
            "overdue": [_serialize_action_item(item) for item in group.overdue],
        }
    except Exception:
        logger.warning("Failed to fetch open tasks for user=%s", user_id, exc_info=True)
        return None


def _normalize_task_description(description: str) -> str:
    """Lowercased, whitespace-collapsed form for cold-path dedup comparison."""
    return re.sub(r"\s+", " ", description.strip().lower())


async def persist_extracted_tasks(
    user_id: str,
    tasks: list[dict[str, Any]],
    *,
    thread_id: str | None = None,
    advisor_persona: str | None = None,
) -> int:
    """Cold-path write for tasks extracted from conversation prose (spec Item 2, §4.2).

    Dedups each candidate against the user's currently-open tasks by normalized
    description before creating it — because ``consolidate_memories`` runs after
    every turn, this prevents a task from being re-created on subsequent turns.
    Returns the number of tasks actually created.
    """
    if not TASK_MEMORY_AVAILABLE or not tasks:
        return 0
    try:
        session_maker = _get_task_session_maker()
        async with session_maker() as session:
            group = await _AccountabilityService.get_open_and_overdue(session, user_id)
            existing = {_normalize_task_description(item.description) for item in (*group.not_yet_due, *group.overdue)}
            created = 0
            for task in tasks:
                description = (task.get("description") or "").strip()
                if not description or _normalize_task_description(description) in existing:
                    continue
                priority = task.get("priority")
                if priority not in _TASK_PRIORITIES:
                    priority = "normal"
                await _AccountabilityService.create_action_item(
                    session,
                    user_id,
                    description,
                    thread_id=thread_id,
                    due_date=_parse_due_date(task.get("due_date")),
                    priority=priority,
                    advisor_persona=advisor_persona,
                    source="conversation",
                )
                existing.add(_normalize_task_description(description))
                created += 1
        return created
    except Exception:
        logger.warning("Cold-path task persistence failed for user=%s", user_id, exc_info=True)
        return 0


async def get_open_tasks() -> dict[str, Any]:
    """Get the student's open and overdue tasks — what you've assigned that isn't done yet.

    ALWAYS call this at the very start of a new conversation (before your
    first reply, when there are no prior AI messages yet) so you can open by
    checking in on open commitments instead of starting from a blank slate.
    Also call it before assigning a new task, to avoid re-assigning
    something already open.

    Reasoning by task state (do not just recite the list):
      - Overdue, miss_count == 0 or 1: ask what blocked it, don't just restate the task.
      - Overdue, miss_count >= 2: do NOT assign it a third time — address the
        underlying blocker directly instead.
      - Not yet due: a brief, light-touch check-in only. No pressure.
    """
    runtime = get_runtime(Context)
    user_id = runtime.context.user_id

    if not user_id:
        return {"error": "Authentication required", "message": "Cannot fetch tasks without user context."}
    if not TASK_MEMORY_AVAILABLE:
        return {
            "error": "Task memory service unavailable",
            "message": "The task backend is not reachable from this environment.",
        }

    group = await fetch_open_task_group_for_prompt(user_id)
    if group is None:
        return {"error": "Failed to fetch tasks", "message": "Task lookup failed — see server logs."}
    return {"ok": True, **group}


async def manage_task(
    action: str,
    *,
    task_id: str | None = None,
    description: str | None = None,
    due_date: str | None = None,
    priority: str = "normal",
    category: str | None = None,
    status: str | None = None,
    evidence: str | None = None,
) -> dict[str, Any]:
    """Create or update a task you've assigned the student — the accountability ledger.

    Call this immediately when you assign a task in conversation — do not
    wait until the end, and do not rely on it being inferred automatically.

    Args:
        action: "create" to log a new task, or "update_status" to change an
            existing one.
        task_id: Required for "update_status". The task's id (from
            get_open_tasks results).
        description: Required for "create". A specific, concrete commitment —
            e.g. "Rebuild GitHub README with 3 pinned projects", not "work on
            portfolio".
        due_date: Optional ISO date/datetime (e.g. "2026-07-10"). Omit for
            vague timeframes like "by next session" — fold those into
            `description` instead.
        priority: One of high/normal/low. Defaults to normal.
        category: Optional free-text grouping, e.g. "Portfolio", "Job Search".
        status: Required for "update_status". One of: in_progress, completed,
            skipped, abandoned, renegotiated.
        evidence: Proof of completion — a URL, submission id, or the
            student's own description of what they did. Pass this when
            marking a task completed. If none is available, ask the student
            for it before marking done — never assume completion.
    """
    runtime = get_runtime(Context)
    user_id = runtime.context.user_id
    thread_id: str | None = None
    config = get_config()
    if config:
        thread_id = config.get("configurable", {}).get("thread_id")

    if not user_id:
        return {"error": "Authentication required", "message": "Cannot manage tasks without user context."}
    if not TASK_MEMORY_AVAILABLE:
        return {
            "error": "Task memory service unavailable",
            "message": "The task backend is not reachable from this environment.",
        }

    if action not in {"create", "update_status"}:
        return {"error": "Invalid action", "message": "action must be 'create' or 'update_status'"}

    try:
        session_maker = _get_task_session_maker()
        async with session_maker() as session:
            if action == "create":
                if not description or not description.strip():
                    return {"error": "description is required to create a task"}
                if priority not in _TASK_PRIORITIES:
                    return {"error": f"priority must be one of {sorted(_TASK_PRIORITIES)}"}

                advisor = runtime.context.advisor or {}
                advisor_persona = advisor.get("name", "").split()[0] if advisor.get("name") else None

                item = await _AccountabilityService.create_action_item(
                    session,
                    user_id,
                    description.strip(),
                    thread_id=thread_id,
                    due_date=_parse_due_date(due_date),
                    priority=priority,
                    category=category,
                    advisor_persona=advisor_persona,
                    source="conversation",
                )
                return {"ok": True, "task": _serialize_action_item(item)}

            # action == "update_status"
            if not task_id:
                return {"error": "task_id is required for update_status"}
            if status not in _TASK_UPDATE_STATUSES:
                return {"error": f"status must be one of {sorted(_TASK_UPDATE_STATUSES)}"}

            result = await _AccountabilityService.update_action_item_status(
                session, task_id, user_id, status, evidence=evidence
            )
            evidence_missing = status == "completed" and not evidence
            return {"ok": True, **result, "evidence_missing": evidence_missing}

    except ValueError as e:
        return {"error": "Task not found", "message": str(e)}
    except Exception as e:
        logger.error("Error managing task for user=%s action=%s: %s", user_id, action, e, exc_info=True)
        return {"error": "Failed to manage task", "message": str(e)}


# Build tools list dynamically based on availability
TOOLS: list[Callable[..., Any]] = [
    # search,
    # extract_webpage_content,
    brave_search,
    read_webpage,
    get_student_profile,
    get_student_onboarding,
    get_student_ai_career_advisor_onboarding,
    get_student_enrollment_overview,
    get_course_structure,
    get_course_materials,
    get_course_progress,
    get_student_attempts,
    get_subscription_state,
    get_portfolio_projects,
    review_project_submission,
    search_past_conversations,
    get_thread_summary_by_title,
]

# Add course search tool if a backend is available
if COURSE_CONTENT_AVAILABLE:
    TOOLS.append(search_course_content)
    logger.info("Course search tool enabled")

# Add opportunities tools if the service is available
if OPPORTUNITIES_AVAILABLE:
    TOOLS.extend([get_opportunities, get_opportunity_strategy])
    logger.info("Opportunities tools enabled")

# Add task memory tools if the accountability service is available
if TASK_MEMORY_AVAILABLE:
    TOOLS.extend([get_open_tasks, manage_task])
    logger.info("Task memory tools enabled")

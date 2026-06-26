"""Opportunity discovery engine.

Wires together the board-specific job sources (opportunity_job_sources) and the
event sources (opportunity_event_sources) into a single engine that can be called
from the API layer.

The public surface is unchanged from before the refactor:
    from aegra_api.services.opportunity_discovery import opportunity_engine
    await opportunity_engine.discover_for_user(session, user_id, auth_token)
    await opportunity_engine.create_notifications_batch(session, discovered)
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import httpx
import pycountry
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.core.accountability_orm import (
    DiscoveredOpportunity,
    Notification,
    UserPreferences,
)
from aegra_api.services.advisor_cache import get_cached_learning_track
from aegra_api.services.opportunity_event_sources import (
    EventbriteSource,
    LumaSource,
    MeetupSource,
    RawEventOpportunity,
    TixAfricaSource,
)
from aegra_api.services.opportunity_job_sources import (
    DEFAULT_MAX_RESULTS_PER_SOURCE,
    JobDiscoveryContext,
    JobSourceRegistry,
    RawJobOpportunity,
)
from aegra_api.services.student_profile import StudentProfile, fetch_student_profile
from aegra_api.settings import settings
from aegra_api.tools.course_content.mongo_client import get_course_content_mongo_client

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Track keyword map (used for relevance scoring and query building)
# ---------------------------------------------------------------------------

TRACK_KEYWORDS: dict[str, list[str]] = {
    "data-analytics": [
        "data analytics",
        "data analyst",
        "business intelligence",
        "Power BI",
        "Tableau",
        "SQL analyst",
    ],
    "data-science": [
        "data science",
        "data scientist",
        "machine learning",
        "ML engineer",
        "predictive analytics",
    ],
    "data-engineering": [
        "data engineering",
        "data engineer",
        "ETL",
        "data pipeline",
        "Spark",
        "Airflow",
    ],
    "ai-engineering": [
        "AI engineer",
        "artificial intelligence",
        "machine learning",
        "ML engineer",
        "LLM",
        "deep learning",
        "ML Ops",
        "generative AI",
        "NLP",
    ],
    "business-intelligence": [
        "business intelligence",
        "BI developer",
        "Power BI",
        "Looker",
        "reporting",
    ],
    "dev": [
        "software developer",
        "software engineer",
        "web developer",
        "full stack developer",
        "frontend developer",
        "backend developer",
    ],
}

# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

# US state abbreviations used to detect "City, ST" formatted job locations
# (e.g. "New York, NY", "San Francisco, CA") from the remote role search.
_US_STATE_CODES: frozenset[str] = frozenset(
    {
        "al",
        "ak",
        "az",
        "ar",
        "ca",
        "co",
        "ct",
        "de",
        "fl",
        "ga",
        "hi",
        "id",
        "il",
        "in",
        "ia",
        "ks",
        "ky",
        "la",
        "me",
        "md",
        "ma",
        "mi",
        "mn",
        "ms",
        "mo",
        "mt",
        "ne",
        "nv",
        "nh",
        "nj",
        "nm",
        "ny",
        "nc",
        "nd",
        "oh",
        "ok",
        "or",
        "pa",
        "ri",
        "sc",
        "sd",
        "tn",
        "tx",
        "ut",
        "vt",
        "va",
        "wa",
        "wv",
        "wi",
        "wy",
        "dc",
    }
)


_US_STATE_NAMES: frozenset[str] = frozenset(
    {
        "alabama",
        "alaska",
        "arizona",
        "arkansas",
        "california",
        "colorado",
        "connecticut",
        "delaware",
        "florida",
        "georgia",
        "hawaii",
        "idaho",
        "illinois",
        "indiana",
        "iowa",
        "kansas",
        "kentucky",
        "louisiana",
        "maine",
        "maryland",
        "massachusetts",
        "michigan",
        "minnesota",
        "mississippi",
        "missouri",
        "montana",
        "nebraska",
        "nevada",
        "new hampshire",
        "new jersey",
        "new mexico",
        "new york",
        "north carolina",
        "north dakota",
        "ohio",
        "oklahoma",
        "oregon",
        "pennsylvania",
        "rhode island",
        "south carolina",
        "south dakota",
        "tennessee",
        "texas",
        "utah",
        "vermont",
        "virginia",
        "washington",
        "west virginia",
        "wisconsin",
        "wyoming",
    }
)


def _is_us_location(loc: str) -> bool:
    """Return True when the location string refers to a US city/state."""
    lower = loc.lower()
    if "united states" in lower or "usa" in lower:
        return True
    # Detect "City, ST" format (e.g. "New York, NY") or "City, StateName"
    parts = [p.strip() for p in lower.split(",")]
    return any(p in _US_STATE_CODES or p in _US_STATE_NAMES for p in parts)


# Country aliases: maps common short forms / subdivisions to the canonical pycountry name.
# Used by _job_location_allowed so that job locations like "London, England, UK" or
# "Lagos, NG" are accepted for users whose readable location is "United Kingdom" / "Nigeria".
_COUNTRY_ALIASES: dict[str, set[str]] = {
    "united kingdom": {"uk", "england", "scotland", "wales", "northern ireland", "great britain", "gb"},
    "united states": {"usa", "us", "united states of america"},
    "nigeria": {"ng", "ngr"},
    "ghana": {"gh", "gha"},
    "south africa": {"rsa", "za", "south africa"},
    "united arab emirates": {"uae", "dubai", "abu dhabi"},
    "kenya": {"ke", "nbi"},
    "singapore": {"sg", "sgp"},
    "australia": {"au", "aus"},
    "canada": {"ca", "can"},
    "germany": {"de", "deu"},
    "france": {"fr", "fra"},
    "netherlands": {"nl", "nld"},
    "ireland": {"ie", "irl"},
    "new zealand": {"nz", "nzl"},
}

# Maps bare city names (lowercase) to their canonical country name.
# Job boards often post locations as plain city names ("London", "Lagos")
# without any country suffix, causing _job_location_allowed to miss them.
_MAJOR_CITIES: dict[str, str] = {
    # United Kingdom
    "london": "united kingdom",
    "manchester": "united kingdom",
    "birmingham": "united kingdom",
    "leeds": "united kingdom",
    "glasgow": "united kingdom",
    "edinburgh": "united kingdom",
    "bristol": "united kingdom",
    "liverpool": "united kingdom",
    "sheffield": "united kingdom",
    "cambridge": "united kingdom",
    "oxford": "united kingdom",
    "cardiff": "united kingdom",
    "belfast": "united kingdom",
    "nottingham": "united kingdom",
    # Nigeria
    "lagos": "nigeria",
    "abuja": "nigeria",
    "port harcourt": "nigeria",
    "ibadan": "nigeria",
    "kano": "nigeria",
    # Ghana
    "accra": "ghana",
    "kumasi": "ghana",
    # Kenya
    "nairobi": "kenya",
    "mombasa": "kenya",
    # South Africa
    "johannesburg": "south africa",
    "cape town": "south africa",
    "durban": "south africa",
    "pretoria": "south africa",
    # Germany
    "berlin": "germany",
    "munich": "germany",
    "hamburg": "germany",
    "frankfurt": "germany",
    "cologne": "germany",
    # France
    "paris": "france",
    # Netherlands
    "amsterdam": "netherlands",
    # Ireland
    "dublin": "ireland",
    # Australia
    "sydney": "australia",
    "melbourne": "australia",
    "brisbane": "australia",
    "perth": "australia",
    # Canada
    "toronto": "canada",
    "vancouver": "canada",
    "montreal": "canada",
    "calgary": "canada",
    # UAE
    "dubai": "united arab emirates",
    "abu dhabi": "united arab emirates",
    # Singapore (city = country)
    "singapore": "singapore",
}

# ---------------------------------------------------------------------------
# Waterfall tier scoring — non-overlapping bands so match_score sort = tier order
# ---------------------------------------------------------------------------

# Tier 1 (exact role + state) scores [0.90, 1.00]
# Tier 2 (exact role + country) scores [0.75, 0.89]
# Tier 3 (adjacent + state) scores [0.60, 0.74]
# Tier 4 (adjacent + country) scores [0.45, 0.59]
_TIER_BASE_SCORE: dict[int, Decimal] = {
    1: Decimal("0.90"),
    2: Decimal("0.75"),
    3: Decimal("0.60"),
    4: Decimal("0.45"),
}
_TIER_SCORE_CAP: dict[int, Decimal] = {
    1: Decimal("1.00"),
    2: Decimal("0.89"),
    3: Decimal("0.74"),
    4: Decimal("0.59"),
}
_PER_TIER_CAP = 5  # max results stored per tier per discovery run
_MAX_EVENTS_PER_SOURCE = 8  # max events saved per source per discovery run

# Indeed expects specific country strings; map ISO alpha-2 → jobspy/Indeed string
_INDEED_ALPHA2_MAP: dict[str, str] = {
    "GB": "UK",
    "US": "USA",
    "CA": "Canada",
    "AU": "Australia",
    "DE": "Germany",
    "FR": "France",
    "IN": "India",
    "NG": "Nigeria",
    "ZA": "South Africa",
    "KE": "Kenya",
    "SG": "Singapore",
    "AE": "United Arab Emirates",
    "SE": "Sweden",
    "NO": "Norway",
    "DK": "Denmark",
    "FI": "Finland",
    "ES": "Spain",
    "IT": "Italy",
    "PT": "Portugal",
    "PL": "Poland",
    "CH": "Switzerland",
    "NL": "Netherlands",
    "IE": "Ireland",
    "BR": "Brazil",
    "MX": "Mexico",
    "NZ": "New Zealand",
    "JP": "Japan",
}


def _readable_location(raw: str) -> str:
    """Convert a raw location (2-letter ISO code or full name) to a readable country name."""
    stripped = raw.strip()
    if not stripped:
        return stripped
    # Only attempt lookup for short strings that look like ISO codes or aliases
    if len(stripped) <= 3:
        try:
            # Fast direct alpha-2 lookup first
            country = pycountry.countries.get(alpha_2=stripped.upper())
            if country:
                return country.name
            # Fuzzy fallback handles aliases like "UAE", "USA"
            results = pycountry.countries.search_fuzzy(stripped)
            if results:
                return results[0].name
        except (LookupError, AttributeError):
            pass
    return stripped


def _normalise_track(track: str | None) -> str | None:
    """Normalise a track to kebab-lowercase (e.g. 'AI Engineering' → 'ai-engineering').

    Returns None for empty/None input so callers can safely use ``if track`` guards.
    """
    if not track or not track.strip():
        return None
    return track.lower().strip().replace(" ", "-")


def _dedupe_tracks(tracks: list[str]) -> list[str]:
    """Deduplicate tracks while preserving order and normalizing variants."""
    deduped: list[str] = []
    seen: set[str] = set()
    for raw in tracks:
        if not isinstance(raw, str):
            continue
        normalized = _normalise_track(raw)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(raw.strip())
    return deduped


def _dedupe_locations(locations: list[str]) -> list[str]:
    """Normalize and deduplicate location strings while preserving order."""
    deduped: list[str] = []
    seen: set[str] = set()
    for raw in locations:
        if not isinstance(raw, str):
            continue
        normalized = raw.strip()
        if not normalized:
            continue
        key = normalized.upper()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _keywords_for_track(track: str) -> list[str]:
    key = _normalise_track(track)
    return TRACK_KEYWORDS.get(key, [track.lower()])


def _primary_keyword(track: str) -> str:
    keywords = _keywords_for_track(track)
    return keywords[0] if keywords else track.replace("-", " ")


def _score_text(text: str, track: str) -> Decimal:
    content = text.lower()
    score = 0.50
    for kw in _keywords_for_track(track):
        if kw.lower() in content:
            score += 0.08
    return Decimal(str(min(round(score, 2), 1.0)))


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def _query_matches_title(source_query: str, job_title: str) -> bool:
    """Return True when the search query that fetched this job substantially
    matches the job title, so that target-role searches ("Data Product Manager")
    survive the relevance gate even when the title contains none of our narrow
    track keywords.

    Rule: ≥ 60 % of the meaningful words (> 1 char) from the query appear in
    the job title.  Using > 1 (not > 3) ensures short but meaningful acronyms
    like "AI", "ML", "BI" are included — without them "AI engineer" reduces to
    only ["engineer"] and falsely matches "Tier 1 IT Engineer".
    """
    if not source_query or not job_title:
        return False
    title_lower = job_title.lower()
    query_words = [w for w in source_query.lower().split() if len(w) > 1]
    if not query_words:
        return False
    matches = sum(1 for w in query_words if w in title_lower)
    return matches / len(query_words) >= 0.6


def _role_matches_title(role_term: str, job_title: str) -> bool:
    """Stricter relevance check used for the role bucket.

    Requires the last two meaningful words of the role term to both appear in
    the job title.  This prevents generic word overlaps from admitting unrelated
    jobs — e.g. for "Data Product Manager", the words "product" + "manager"
    must both be present, so "Data Scientist — Data & Analytics Products" (no
    "manager") and "Senior Cost Manager — Data Center" (no "product") are
    excluded while "Product Manager", "Senior Product Manager", and
    "Data Product Manager" all pass.
    """
    if not role_term or not job_title:
        return False
    words = [w for w in role_term.lower().split() if len(w) > 1]
    if not words:
        return False
    # Use the last two words as the core anchor; fall back to all words if < 2
    anchor = words[-2:] if len(words) >= 2 else words
    title_lower = job_title.lower()
    return all(w in title_lower for w in anchor)


class OpportunityDiscoveryEngine:
    """Discovers relevant events and jobs for a user and persists them."""

    def __init__(self) -> None:
        self._event_sources = [EventbriteSource(), MeetupSource(), LumaSource(), TixAfricaSource()]
        self._job_registry = JobSourceRegistry(settings.discovery.company_job_boards)

    @property
    def job_source_registry(self) -> JobSourceRegistry:
        """Compatibility alias for tests and callers expecting a public registry."""
        return self._job_registry

    # ------------------------------------------------------------------
    # LMS / profile helpers
    # ------------------------------------------------------------------

    async def get_user_enrollments(self, user_id: str, auth_token: str) -> list[dict]:
        """Fetch user's enrolled courses/tracks from LMS API."""
        lms_base_url = settings.app.LMS_URL
        if not lms_base_url:
            logger.warning("LMS API URL not configured")
            return []

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{lms_base_url}/api/v1/enrollment/student/blackboard",
                    headers={"Authorization": f"Bearer {auth_token}"},
                    timeout=10.0,
                )
                response.raise_for_status()
                data = response.json()
                enrollments = data.get("enrollments", [])
                logger.info(
                    "Fetched enrollments from LMS",
                    user_id=user_id,
                    count=len(enrollments),
                )
                return enrollments
        except httpx.HTTPError as e:
            logger.error("Failed to fetch enrollments", error=str(e), user_id=user_id)
            return []

    async def get_user_location(self, session: AsyncSession, user_id: str) -> str | None:
        result = await session.execute(select(UserPreferences).where(UserPreferences.user_id == user_id))
        prefs = result.scalar_one_or_none()
        return prefs.location if prefs else None

    def get_profile_locations(self, profile: StudentProfile | None) -> list[str]:
        if not profile:
            return []
        return _dedupe_locations(
            [
                *(profile.work_countries or []),
                profile.resident_country,
            ]
        )

    def get_profile_states(self, profile: StudentProfile | None) -> list[str]:
        """Return deduped city/state names from onboarding (Tier 1 & 3 locations)."""
        if not profile:
            return []
        return _dedupe_locations([*(profile.resident_cities or []), *(profile.work_cities or [])])

    def _job_location_state_allowed(self, job_location: str, state_names: set[str]) -> bool:
        """Return True if job location contains one of the user's city/state names, or is remote."""
        if not state_names:
            return False
        loc_lower = (job_location or "").lower().strip()
        if not loc_lower or "remote" in loc_lower or "hybrid" in loc_lower:
            return True
        for state in state_names:
            state_lower = state.lower()
            if state_lower in loc_lower or loc_lower in state_lower:
                return True
        return False

    async def _classify_track_adjacent_batch(self, job_titles: list[str], track: str) -> set[int]:
        """Use LLM to identify which job titles are adjacent to the user's track.

        Returns a set of indices into job_titles. Falls back to keyword matching on error.
        """
        if not job_titles:
            return set()
        try:
            from langchain_aws import ChatBedrockConverse
            from langchain_core.messages import HumanMessage, SystemMessage

            llm = ChatBedrockConverse(
                model="eu.anthropic.claude-haiku-4-5-20251001-v1:0",
                region_name=settings.aws.AWS_REGION_NAME,
                temperature=0,
                max_tokens=200,
            )
            numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(job_titles))
            messages = [
                SystemMessage(
                    content=(
                        "You are a career advisor. Determine which job titles are relevant or adjacent "
                        "to someone studying the given learning track. "
                        "Reply ONLY with a JSON array of index numbers, e.g. [0, 2, 4]."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Learning track: {track}\n\nJob titles:\n{numbered}\n\n"
                        "Return a JSON array of indices for adjacent/relevant titles. "
                        "Include roles that share skills, tools, or career paths with this track."
                    )
                ),
            ]
            resp = await llm.ainvoke(messages)
            text = resp.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return {int(i) for i in parsed if isinstance(i, int) and 0 <= i < len(job_titles)}
        except Exception as exc:
            logger.warning("track_adjacency_classification_failed", error=str(exc), track=track)
        # Keyword fallback when LLM fails
        keywords = _keywords_for_track(track)
        return {i for i, t in enumerate(job_titles) if any(kw.lower() in t.lower() for kw in keywords)}

    async def _get_student_profile(
        self,
        session: AsyncSession,
        user_id: str,
        auth_token: str | None = None,
    ) -> StudentProfile:
        if auth_token and auth_token != "scheduled_job_token":  # nosec B105
            try:
                return await fetch_student_profile(user_id, auth_token)
            except Exception as e:
                logger.warning("student_profile_fetch_failed", error=str(e), user_id=user_id)
        return StudentProfile(user_id=user_id)

    async def _get_track_from_prefs(
        self,
        session: AsyncSession,
        user_id: str,
    ) -> str | None:
        result = await session.execute(select(UserPreferences).where(UserPreferences.user_id == user_id))
        prefs = result.scalar_one_or_none()
        if prefs and prefs.preferences:
            stored = prefs.preferences
            learning_track = stored.get("learning_track") or stored.get("track")
            if isinstance(learning_track, str) and learning_track.strip():
                logger.info("discovery_track_from_prefs", user_id=user_id, track=learning_track)
                return learning_track.strip()

            tracks = stored.get("tracks", []) or []
            if isinstance(tracks, list):
                for track in tracks:
                    if isinstance(track, str) and track.strip():
                        logger.info("discovery_track_from_prefs_list", user_id=user_id, track=track)
                        return track.strip()
        return None

    def build_event_queries(self, track: str, _location: str | None = None) -> list[str]:
        """Compatibility wrapper around the internal event-query builder."""
        return self._build_event_queries(track)

    def _build_job_search_terms(
        self,
        tracks: list[str],
        profile: StudentProfile | None = None,
        queries_per_category: int = 1,
    ) -> list[str]:
        """Build prioritized job search terms from onboarding/profile + track."""
        search_terms: list[str] = []
        seen_terms: set[str] = set()

        def _add(term: str | None) -> None:
            if not isinstance(term, str):
                return
            normalized = term.strip()
            if not normalized:
                return
            lowered = normalized.lower()
            if lowered in seen_terms:
                return
            seen_terms.add(lowered)
            search_terms.append(normalized)

        for track in tracks:
            _add(_primary_keyword(track))

        # Always include the user's explicitly stated target role and current
        # role title as search terms — these are their own stated goals and
        # should never be gated behind a keyword-alignment score.  The previous
        # _is_aligned() guard used score > 0.50, but _score_text() starts at
        # exactly 0.50, so a target_role like "Data Product Manager" on the
        # ai-engineering track always scored exactly 0.50 and was silently
        # dropped, meaning the user never saw jobs matching what they actually
        # want.  Result-level scoring (matched_target_role tag) still filters
        # out irrelevant hits at display time.
        if profile and isinstance(profile.target_role, str) and profile.target_role.strip():
            _add(profile.target_role)
            # For 3-word roles, also add the 2-word anchor so country-specific
            # boards (Indeed, SmartRecruiters) surface broader results.
            # E.g. "Data Product Manager" → also search "Product Manager" to
            # find UK PM listings that don't use the full DPM title.
            _role_words = [w for w in profile.target_role.split() if len(w) > 1]
            if len(_role_words) >= 3:
                _add(" ".join(_role_words[-2:]).title())
        if profile and isinstance(profile.role_title, str) and profile.role_title.strip():
            _add(profile.role_title)

        if queries_per_category > 1:
            for track in tracks:
                for kw in _keywords_for_track(track)[1:queries_per_category]:
                    _add(kw)

        return search_terms

    def _match_raw_job(
        self,
        raw_job: RawJobOpportunity,
        tracks: list[str],
        locations: list[str],
        profile: StudentProfile | None = None,
    ) -> dict[str, Any] | None:
        """Accept a raw job returned by the search API and compute a display score.

        We trust the job board's full-text search to return relevant results for
        the queries we built (track keywords + user's target_role).  The old
        keyword-scoring hard-filter (score <= 0.50 → drop) was silently dropping
        all results whose title/description didn't contain our small list of track
        keywords — for example a "Data Product Manager" job fetched because the
        user's target_role is "Data Product Manager" would score exactly 0.50
        (no ai-engineering keywords in the text) and get discarded.

        Now we only filter on location; scoring is kept purely for display/sorting.
        A base score of 0.60 is assigned to every location-matched result so the
        frontend always has a non-zero value to sort on, with bonus points for
        explicit keyword hits.
        """
        readable_locations = {_readable_location(loc).lower() for loc in locations} if locations else {"remote"}
        if not self._job_location_allowed(raw_job.location, readable_locations):
            return None

        content = f"{raw_job.title} {raw_job.description}"
        # Compute informational score (not used as a filter gate).
        best_score = Decimal("0.60")  # baseline for any location-matched result
        best_track = tracks[0] if tracks else ""
        for track in tracks:
            score = _score_text(content, track)
            if score > best_score:
                best_score = score
                best_track = track

        reason_tags = ["matched_track", "matched_location"]
        profile_text = content.lower()
        if profile:
            if profile.target_role and profile.target_role.lower() in profile_text:
                reason_tags.append("matched_target_role")
            if profile.industry and profile.industry.lower() in profile_text:
                reason_tags.append("matched_industry")
            if any(skill.lower() in profile_text for skill in profile.confident_skills):
                reason_tags.append("matched_profile_skills")

        return {
            "opportunity_type": "job",
            "title": raw_job.title,
            "description": raw_job.description,
            "url": raw_job.url,
            "location": raw_job.location or (locations[0] if locations else "remote"),
            "event_date": None,
            "company": raw_job.company,
            "salary_range": raw_job.salary_range,
            "match_score": best_score,
            "matched_track": best_track,
            "reason_tags": reason_tags,
            "_source": raw_job.source,
            "_query": raw_job.source_query,
        }

    def _match_raw_event(
        self,
        raw_event: RawEventOpportunity,
        track: str,
        location: str,
        profile: StudentProfile | None = None,
    ) -> dict[str, Any] | None:
        """Accept a raw event returned by the search API and compute a display score.

        Same philosophy as _match_raw_job: trust the event source's full-text
        search.  The old hard-filter (score <= 0.50 → drop) was discarding events
        that didn't contain our narrow list of track keywords even though the API
        found them relevant.  Score is now informational only; base = 0.60.
        """
        score = _score_text(f"{raw_event.title} {raw_event.description}", track)
        if score <= Decimal("0.60"):
            score = Decimal("0.60")  # baseline — not a filter gate

        readable_locations = {_readable_location(location).lower()}
        if (
            raw_event.location
            and not any(kw in raw_event.location.lower() for kw in ("online", "remote", "virtual"))
            and not self._job_location_allowed(raw_event.location, readable_locations)
        ):
            return None

        reason_tags = ["matched_track"]
        profile_text = f"{raw_event.title} {raw_event.description}".lower()
        if profile:
            if profile.target_role and profile.target_role.lower() in profile_text:
                reason_tags.append("matched_target_role")
            if profile.industry and profile.industry.lower() in profile_text:
                reason_tags.append("matched_industry")
            if any(skill.lower() in profile_text for skill in profile.confident_skills):
                reason_tags.append("matched_profile_skills")

        return {
            "opportunity_type": "event",
            "title": raw_event.title,
            "description": raw_event.description,
            "url": raw_event.url,
            "location": raw_event.location or _readable_location(location),
            "event_date": raw_event.event_date.isoformat() if raw_event.event_date else None,
            "company": None,
            "salary_range": None,
            "match_score": score,
            "matched_track": track,
            "reason_tags": reason_tags,
            "_source": raw_event.source,
            "_query": self._primary_event_query_for_track(track),
        }

    def _primary_event_query_for_track(self, track: str) -> str:
        return _primary_keyword(track)

    # ------------------------------------------------------------------
    # Strategy generation
    # ------------------------------------------------------------------

    async def generate_networking_strategy(self, opportunity: dict[str, Any], user_track: str) -> dict[str, Any] | None:
        """Generate a personalised networking strategy for an event."""
        try:
            from langchain_aws import ChatBedrockConverse
            from langchain_core.messages import HumanMessage, SystemMessage

            llm = ChatBedrockConverse(
                model="eu.anthropic.claude-haiku-4-5-20251001-v1:0",
                region_name=settings.aws.AWS_REGION_NAME,
                temperature=0.7,
                max_tokens=500,
            )
            messages = [
                SystemMessage(
                    content=(
                        "You are a career networking coach. Generate a concise, actionable "
                        "networking strategy for a student attending a professional event. "
                        "Reply ONLY with valid JSON (no markdown fences)."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Event: {opportunity.get('title') or ''}\n"
                        f"Description: {(opportunity.get('description') or '')[:300]}\n"
                        f"Student's track: {user_track}\n\n"
                        "Return JSON with keys: why_relevant (2 sentences), "
                        "preparation (list of 3 bullet items), "
                        "conversation_starters (list of 3 questions), "
                        "goals (string, e.g. 'Make 3 connections'), "
                        "follow_up (string, 1 sentence)"
                    )
                ),
            ]
            resp = await llm.ainvoke(messages)
            text = resp.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            return json.loads(text)
        except Exception as e:
            logger.warning("networking_strategy_generation_failed", error=str(e))
            return None

    async def generate_application_strategy(
        self, opportunity: dict[str, Any], user_track: str
    ) -> dict[str, Any] | None:
        """Generate AI application strategy for a job opportunity."""
        try:
            from langchain_aws import ChatBedrockConverse
            from langchain_core.messages import HumanMessage, SystemMessage

            llm = ChatBedrockConverse(
                model="eu.anthropic.claude-haiku-4-5-20251001-v1:0",
                region_name=settings.aws.AWS_REGION_NAME,
                temperature=0.7,
                max_tokens=500,
            )
            messages = [
                SystemMessage(
                    content=(
                        "You are a career advisor helping a student apply for jobs. "
                        "Generate a concise application strategy. "
                        "Reply ONLY with valid JSON (no markdown fences)."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Job: {opportunity.get('title') or ''}\n"
                        f"Company: {opportunity.get('company') or 'Unknown'}\n"
                        f"Description: {(opportunity.get('description') or '')[:300]}\n"
                        f"Student's track: {user_track}\n\n"
                        "Return JSON with keys: fit_assessment (2 sentences), "
                        "priority ('immediate'|'this_week'|'low'), "
                        "resume_points (list of 3 bullet strings), "
                        "cover_letter_angle (1 sentence), "
                        "gap_mitigation (1 sentence), "
                        "timeline (string, e.g. 'Apply within 48 hours')"
                    )
                ),
            ]
            resp = await llm.ainvoke(messages)
            text = resp.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            return json.loads(text)
        except Exception as e:
            logger.warning("application_strategy_generation_failed", error=str(e))
            return None

    async def _generate_strategies_batch(
        self,
        parsed_results: list[dict[str, Any]],
        max_strategies: int = 15,
    ) -> list[dict[str, Any]]:
        """Generate AI strategies for top-scoring results in parallel."""
        if not parsed_results:
            return parsed_results

        scored = sorted(parsed_results, key=lambda x: x.get("match_score", 0), reverse=True)
        to_generate = scored[:max_strategies]
        remaining = scored[max_strategies:]

        sem = asyncio.Semaphore(2)

        async def _gen(item: dict[str, Any]) -> dict[str, Any]:
            async with sem:
                try:
                    opp_type = item.get("opportunity_type")
                    track = item.get("matched_track", "")
                    if opp_type == "event":
                        strategy = await self.generate_networking_strategy(item, track)
                        if strategy:
                            item["networking_strategy"] = strategy
                    elif opp_type == "job":
                        strategy = await self.generate_application_strategy(item, track)
                        if strategy:
                            item["application_strategy"] = strategy
                except Exception as e:
                    logger.warning(
                        "batch_strategy_generation_failed",
                        error=str(e),
                        title=item.get("title", "")[:60],
                    )
            return item

        results = await asyncio.gather(*[_gen(p) for p in to_generate])
        succeeded = sum(1 for r in results if r.get("networking_strategy") or r.get("application_strategy"))
        logger.info(
            "batch_strategy_generation_complete",
            total=len(parsed_results),
            generated=len(to_generate),
            with_strategy=succeeded,
            skipped=len(remaining),
        )
        return list(results) + remaining

    # ------------------------------------------------------------------
    # Event discovery (Eventbrite)
    # ------------------------------------------------------------------

    def _build_event_queries(self, track: str) -> list[str]:
        kw = _primary_keyword(track)
        alt = _keywords_for_track(track)[:3]
        queries = [kw]
        queries.extend(alt[1:])
        return queries

    async def _discover_events(
        self,
        tracks: list[str],
        locations: list[str],
        states: list[str],
        profile: StudentProfile | None,
        seen_urls: set[str],
        queries_per_category: int = 2,
    ) -> list[dict[str, Any]]:
        """Discover events via the 4-tier personalisation waterfall.

        Tier 1 — target role in event content + state match
        Tier 2 — target role in event content + country match
        Tier 3 — track keyword match + state match
        Tier 4 — track keyword match + country match
        """
        sem = asyncio.Semaphore(2)
        readable_locations: set[str] = set()
        for loc in locations:
            readable_locations.add(_readable_location(loc).lower())
            try:
                results = pycountry.countries.search_fuzzy(loc)
                if results:
                    readable_locations.add(results[0].alpha_2.lower())
            except (LookupError, AttributeError):
                pass

        states_set = {s.lower() for s in states} if states else set()
        target_role_lower = (profile.target_role or "").lower().strip() if profile else ""

        global_sources = [s for s in self._event_sources if s.name == "luma"]
        # Tix Africa filters by country internally — no benefit calling it per city/state.
        country_sources = [s for s in self._event_sources if s.name == "tix_africa"]
        location_sources = [s for s in self._event_sources if s.name not in {"luma", "tix_africa"}]

        def _parse_raw(
            raw_events: list[RawEventOpportunity],
            track: str,
            fallback_location: str,
        ) -> list[dict[str, Any]]:
            parsed: list[dict[str, Any]] = []
            for ev in raw_events:
                if not ev.url or ev.url in seen_urls:
                    continue
                seen_urls.add(ev.url)
                content_lower = f"{ev.title} {ev.description}".lower()
                kw_score = _score_text(f"{ev.title} {ev.description}", track)
                if ev.source in {"luma", "tix_africa"} and kw_score <= Decimal("0.50"):
                    kw_score = Decimal("0.58")
                elif kw_score <= Decimal("0.50"):
                    continue  # not relevant to track

                ev_loc = (ev.location or "").lower()
                is_online = not ev_loc or any(kw in ev_loc for kw in ("online", "remote", "virtual"))

                # Country-level location gate
                country_ok = is_online or self._job_location_allowed(ev.location, readable_locations)
                if not country_ok:
                    logger.debug("event_location_filtered", title=ev.title[:60], event_location=ev.location)
                    continue

                state_ok = is_online or self._job_location_state_allowed(ev.location, states_set)
                role_in_content = bool(target_role_lower and target_role_lower in content_lower)

                # Assign tier
                if role_in_content and state_ok:
                    tier = 1
                elif role_in_content and country_ok:
                    tier = 2
                elif state_ok:
                    tier = 3
                else:
                    tier = 4

                score = min(_TIER_BASE_SCORE[tier] + (kw_score - Decimal("0.50")), _TIER_SCORE_CAP[tier])
                reason_tags = ["matched_track"]
                if role_in_content:
                    reason_tags.append("matched_target_role")
                if state_ok and not is_online:
                    reason_tags.append("matched_location")

                parsed.append(
                    {
                        "opportunity_type": "event",
                        "title": ev.title,
                        "description": ev.description,
                        "url": ev.url,
                        "location": ev.location or _readable_location(fallback_location),
                        "event_date": ev.event_date.isoformat() if ev.event_date else None,
                        "company": None,
                        "salary_range": None,
                        "match_score": score,
                        "matched_track": track,
                        "reason_tags": reason_tags,
                        "tier": tier,
                        "_source": ev.source,
                        "_query": ev.source_query,
                    }
                )
            return parsed

        async def _fetch_global(query: str, track: str) -> list[dict[str, Any]]:
            """Fetch from location-agnostic sources (Luma) — called once per query."""
            async with sem:
                batches = await asyncio.gather(
                    *[src.fetch(query, "remote") for src in global_sources],
                    return_exceptions=True,
                )
                raw: list[RawEventOpportunity] = []
                for b in batches:
                    if isinstance(b, list):
                        raw.extend(b)
                return _parse_raw(raw, track, "remote")

        async def _fetch_located(query: str, location: str, track: str) -> list[dict[str, Any]]:
            """Fetch from location-aware sources (Meetup, Eventbrite) — once per location."""
            async with sem:
                batches = await asyncio.gather(
                    *[src.fetch(query, location) for src in location_sources],
                    return_exceptions=True,
                )
                raw: list[RawEventOpportunity] = []
                for b in batches:
                    if isinstance(b, list):
                        raw.extend(b)
                return _parse_raw(raw, track, location)

        async def _fetch_country(query: str, location: str, track: str) -> list[dict[str, Any]]:
            """Fetch from country-scoped sources (Tix Africa) — once per country, not per city."""
            async with sem:
                batches = await asyncio.gather(
                    *[src.fetch(query, location) for src in country_sources],
                    return_exceptions=True,
                )
                raw: list[RawEventOpportunity] = []
                for b in batches:
                    if isinstance(b, list):
                        raw.extend(b)
                return _parse_raw(raw, track, location)

        tasks = []
        for track in tracks:
            queries = self._build_event_queries(track)
            capped = queries[:queries_per_category]
            # Global sources: one task per query (not per location)
            for q in capped:
                tasks.append(_fetch_global(q, track))
            # Country-scoped sources (Tix Africa): once per country query, not per city.
            for loc in locations:
                for q in capped:
                    tasks.append(_fetch_country(q, loc, track))
            # Location-aware sources: country-level AND city/state-level so that
            # city-radius searches (Meetup, Eventbrite) surface local events
            # even when the city is not the country's geographic centre.
            for loc in [*locations, *states]:
                for q in capped:
                    tasks.append(_fetch_located(q, loc, track))

        gathered = await asyncio.gather(*tasks, return_exceptions=True)
        events: list[dict[str, Any]] = []
        for batch in gathered:
            if isinstance(batch, list):
                events.extend(batch)
        return events

    # ------------------------------------------------------------------
    # Job discovery (JobSourceRegistry)
    # ------------------------------------------------------------------

    def _indeed_country(self, locations: list[str]) -> str:
        """Resolve the best Indeed country string from the user's locations via pycountry."""
        for loc in locations:
            stripped = loc.strip()
            if not stripped:
                continue
            try:
                results = pycountry.countries.search_fuzzy(stripped)
                if results:
                    code = results[0].alpha_2
                    if code in _INDEED_ALPHA2_MAP:
                        return _INDEED_ALPHA2_MAP[code]
            except LookupError:
                pass
        return "UK"

    def _job_location_allowed(self, job_location: str, readable_locations: set[str]) -> bool:
        """Return True if the job location matches one of the user's allowed locations or is remote.

        Handles:
        - Remote / hybrid jobs (always allowed)
        - Direct substring match ("United Kingdom" in "London, United Kingdom")
        - Comma-part matching: each segment of "London, England, UK" is checked
          individually so short aliases like "UK" or ISO codes like "GB" are caught
        - _COUNTRY_ALIASES: "UK", "England", "GB" etc. all resolve to "United Kingdom"
        - ISO alpha-2 word-boundary check for short codes to prevent false positives
          (e.g. "NG" in "England" must not match Nigeria)
        """
        loc_lower = job_location.lower().strip()
        if not loc_lower or "remote" in loc_lower or "hybrid" in loc_lower:
            return True

        # Split compound location (e.g. "London, England, UK") into parts for granular matching
        loc_parts = {p.strip() for p in loc_lower.split(",") if p.strip()}

        for allowed in readable_locations:
            # Build the full set of accepted tokens for this allowed location
            accepted = {allowed} | _COUNTRY_ALIASES.get(allowed, set())

            for token in accepted:
                if len(token) <= 2:
                    # Word-boundary regex to prevent "ng" matching inside "England"
                    pattern = r"(?<![a-z])" + re.escape(token) + r"(?![a-z])"
                    if re.search(pattern, loc_lower):
                        return True
                else:
                    # Full-string or part-level substring match
                    if token in loc_lower or loc_lower in token:
                        return True
                    for part in loc_parts:
                        if token in part:
                            return True
                        # Guard: don't let short ISO codes (2-char) match as
                        # substrings of longer tokens.  "ng" (Nigeria) must not
                        # match "united ki**ng**dom" via `part in token`.
                        if len(part) > 2 and part in token:
                            return True

        # City-level fallback: job boards often post bare city names ("London",
        # "Lagos") without a country suffix. Map known cities → country and check.
        for part in loc_parts:
            city_country = _MAJOR_CITIES.get(part)
            if city_country and city_country in readable_locations:
                return True

        return False

    async def _discover_jobs(
        self,
        tracks: list[str],
        locations: list[str],
        states: list[str],
        profile: StudentProfile | None,
        seen_urls: set[str],
        queries_per_category: int = 2,
    ) -> list[dict[str, Any]]:
        """Discover jobs via the 4-tier personalisation waterfall.

        Tier 1 — exact target role + state/city match   (highest priority)
        Tier 2 — exact target role + country match
        Tier 3 — AI-adjacent role + state/city match
        Tier 4 — AI-adjacent role + country match       (widest)

        Up to _PER_TIER_CAP results per tier; de-duplication enforced across all tiers.
        """
        search_locations = [_readable_location(loc) for loc in locations] if locations else ["remote"]
        country_set = {loc.lower() for loc in search_locations}
        states_set = {s.lower() for s in states} if states else set()

        role_term: str = ""
        role_anchor: str = ""
        if profile and isinstance(profile.target_role, str) and profile.target_role.strip():
            role_term = profile.target_role.strip().lower()
            _rw = [w for w in role_term.split() if len(w) > 1]
            if len(_rw) >= 3:
                role_anchor = " ".join(_rw[-2:])

        search_terms = self._build_job_search_terms(tracks, profile=profile, queries_per_category=queries_per_category)
        role_terms_count = (1 if role_term else 0) + (1 if role_anchor else 0)
        max_terms = max(1 + role_terms_count, queries_per_category) if role_term else queries_per_category

        # ── Phase 1: fetch from location-aware sources ───────────────────────────
        # Search at both country AND city/state level so boards that rank by
        # geography (e.g. Adzuna) surface city-specific listings in Tier 1.
        all_raw_jobs: list[RawJobOpportunity] = []
        loc_ctx_kwargs = {
            "search_terms": search_terms,
            "max_search_terms": max_terms,
            "max_results_per_source": DEFAULT_MAX_RESULTS_PER_SOURCE * max_terms,
        }
        location_sources = self._job_registry.build_location_sources()
        _job_search_locs = [*search_locations, *states] if states else search_locations
        for loc in _job_search_locs:
            loc_ctx = JobDiscoveryContext(
                primary_location=loc,
                indeed_country=self._indeed_country([loc]),
                **loc_ctx_kwargs,  # type: ignore[arg-type]
            )
            loc_results = await asyncio.gather(
                *(src.fetch(loc_ctx) for src in location_sources),
                return_exceptions=True,
            )
            for src, result in zip(location_sources, loc_results, strict=False):
                if isinstance(result, list):
                    all_raw_jobs.extend(result)
                elif isinstance(result, Exception):
                    logger.warning("location_source_failed", source=src.name, location=loc, error=repr(result))

        # ── Phase 2: company ATS boards (fetched once, location filtered downstream) ─
        board_ctx = JobDiscoveryContext(
            primary_location=search_locations[0] if search_locations else "remote",
            indeed_country=self._indeed_country(search_locations),
            **loc_ctx_kwargs,  # type: ignore[arg-type]
        )
        board_sources = self._job_registry.build_board_sources()
        board_results = await asyncio.gather(
            *(src.fetch(board_ctx) for src in board_sources),
            return_exceptions=True,
        )
        for src, result in zip(board_sources, board_results, strict=False):
            if isinstance(result, list):
                all_raw_jobs.extend(result)
            elif isinstance(result, Exception):
                logger.warning("board_source_failed", source=src.name, error=repr(result))

        # ── Phase 3: dedicated remote search for target role (fills Tier 1/2 gaps) ──
        remote_role_urls: set[str] = set()
        if role_term:
            role_search_terms = [t for t in search_terms if t.strip().lower() == role_term]
            if role_search_terms:
                remote_ctx = JobDiscoveryContext(
                    search_terms=role_search_terms,
                    primary_location="remote",
                    indeed_country=self._indeed_country(search_locations),
                    max_search_terms=1,
                )
                remote_results = await asyncio.gather(
                    *(src.fetch(remote_ctx) for src in location_sources),
                    return_exceptions=True,
                )
                for src, result in zip(location_sources, remote_results, strict=False):
                    if isinstance(result, list):
                        for job in result:
                            if job.url:
                                remote_role_urls.add(job.url)
                        all_raw_jobs.extend(result)
                    elif isinstance(result, Exception):
                        logger.warning("remote_role_source_failed", source=src.name, role=role_term, error=repr(result))

        primary_location = search_locations[0] if search_locations else "remote"

        # ── Tier classification ───────────────────────────────────────────────────
        # First pass: split each raw job into role-match vs track-candidate and
        # check state / country location for each.
        role_candidates: list[tuple[RawJobOpportunity, bool]] = []  # (raw, state_ok)
        track_candidates: list[tuple[RawJobOpportunity, bool]] = []  # (raw, state_ok)

        for raw in all_raw_jobs:
            if not raw.url or raw.url in seen_urls:
                continue

            _sq = (raw.source_query or "").strip().lower()
            is_role_job = bool(
                role_term
                and (_sq in {role_term, role_anchor} or (not _sq and _role_matches_title(role_term, raw.title)))
            )

            # Country-level location check (gates entry to any tier).
            is_remote_role = is_role_job and raw.url in remote_role_urls
            country_ok = self._job_location_allowed(raw.location, country_set)
            if not country_ok and is_remote_role:
                loc_lower = (raw.location or "").lower()
                country_ok = not loc_lower or any(
                    kw in loc_lower for kw in ("remote", "hybrid", "worldwide", "anywhere")
                )
            if not country_ok:
                continue

            # Relevance gate.
            content = f"{raw.title} {raw.description}"
            best_score = Decimal("0.50")
            for track in tracks:
                s = _score_text(content, track)
                if s > best_score:
                    best_score = s

            if is_role_job:
                relevant = _role_matches_title(role_term, raw.title)
            else:
                relevant = _query_matches_title(raw.source_query or "", raw.title)

            if best_score <= Decimal("0.50") and not relevant:
                continue

            state_ok = self._job_location_state_allowed(raw.location, states_set)

            if is_role_job:
                role_candidates.append((raw, state_ok))
            else:
                track_candidates.append((raw, state_ok))

        # AI adjacency check for track candidates (Tiers 3 & 4).
        track_titles = [raw.title for raw, _ in track_candidates]
        adjacent_indices = await self._classify_track_adjacent_batch(track_titles, tracks[0] if tracks else "")
        adjacent_candidates = [
            (raw, state_ok) for i, (raw, state_ok) in enumerate(track_candidates) if i in adjacent_indices
        ]

        # ── Build tier buckets ────────────────────────────────────────────────────
        tier_buckets: dict[int, list[dict[str, Any]]] = {1: [], 2: [], 3: [], 4: []}

        def _make_job(raw: RawJobOpportunity, tier: int, best_track: str, reason_tags: list[str]) -> dict[str, Any]:
            # Score within the tier's band: base + keyword bonus, capped at tier ceiling.
            content = f"{raw.title} {raw.description}"
            kw_bonus = Decimal("0.0")
            for t in tracks:
                for kw in _keywords_for_track(t):
                    if kw.lower() in content.lower():
                        kw_bonus += Decimal("0.02")
            score = min(_TIER_BASE_SCORE[tier] + kw_bonus, _TIER_SCORE_CAP[tier])
            return {
                "opportunity_type": "job",
                "title": raw.title,
                "description": raw.description,
                "url": raw.url,
                "location": raw.location or primary_location,
                "event_date": None,
                "company": raw.company,
                "salary_range": raw.salary_range,
                "match_score": score,
                "matched_track": best_track,
                "reason_tags": reason_tags,
                "tier": tier,
                "_source": raw.source,
                "_query": raw.source_query,
            }

        best_track_name = tracks[0] if tracks else ""

        for raw, state_ok in role_candidates:
            if raw.url in seen_urls:
                continue
            if state_ok and len(tier_buckets[1]) < _PER_TIER_CAP:
                seen_urls.add(raw.url)
                tier_buckets[1].append(_make_job(raw, 1, best_track_name, ["matched_target_role", "matched_location"]))
            elif len(tier_buckets[2]) < _PER_TIER_CAP:
                seen_urls.add(raw.url)
                tier_buckets[2].append(_make_job(raw, 2, best_track_name, ["matched_target_role"]))

        for raw, state_ok in adjacent_candidates:
            if raw.url in seen_urls:
                continue
            if state_ok and len(tier_buckets[3]) < _PER_TIER_CAP:
                seen_urls.add(raw.url)
                tier_buckets[3].append(_make_job(raw, 3, best_track_name, ["matched_track", "matched_location"]))
            elif len(tier_buckets[4]) < _PER_TIER_CAP:
                seen_urls.add(raw.url)
                tier_buckets[4].append(_make_job(raw, 4, best_track_name, ["matched_track"]))

        all_jobs = tier_buckets[1] + tier_buckets[2] + tier_buckets[3] + tier_buckets[4]

        logger.info(
            "job_discovery_waterfall_complete",
            tier1=len(tier_buckets[1]),
            tier2=len(tier_buckets[2]),
            tier3=len(tier_buckets[3]),
            tier4=len(tier_buckets[4]),
            total=len(all_jobs),
            tracks=tracks,
            locations=search_locations,
            states=list(states_set),
        )
        return all_jobs

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    async def discover_for_user(
        self,
        session: AsyncSession,
        user_id: str,
        auth_token: str = "",
        max_tracks: int = 0,
        queries_per_category: int = 2,
        opportunity_type: str = "all",
        **_kwargs: Any,
    ) -> list[DiscoveredOpportunity]:
        """Run full discovery pipeline for a single user."""
        profile = await self._get_student_profile(session, user_id, auth_token)

        effective_track: str | None = None
        if auth_token and auth_token != "scheduled_job_token":  # nosec B105
            try:
                effective_track = await get_cached_learning_track(user_id, auth_token)
            except Exception as exc:
                logger.warning("discovery_cached_track_lookup_failed", user_id=user_id, error=str(exc))

        if not effective_track and profile.learning_track:
            effective_track = profile.learning_track
        if not effective_track:
            effective_track = await self._get_track_from_prefs(session, user_id)
        if not effective_track:
            try:
                effective_track = await asyncio.to_thread(
                    get_course_content_mongo_client().get_learning_track,
                    user_id,
                )
                if effective_track:
                    logger.info("discovery_track_from_mongo", user_id=user_id, track=effective_track)
            except Exception as exc:
                logger.warning("discovery_mongo_track_lookup_failed", user_id=user_id, error=str(exc))

        # Enrich profile with Mongo onboarding data when running without a user token
        # (scheduler path has no JWT so fetch_student_profile returns an empty profile).
        if not auth_token or auth_token == "scheduled_job_token":  # nosec B105
            try:
                mongo_profile = await asyncio.to_thread(
                    get_course_content_mongo_client().get_user_onboarding_data,
                    user_id,
                )
                if mongo_profile:
                    if not profile.resident_country and mongo_profile.get("resident_country"):
                        profile.resident_country = mongo_profile["resident_country"]
                    if not profile.work_countries and mongo_profile.get("work_countries"):
                        profile.work_countries = list(mongo_profile["work_countries"])
                    if not profile.resident_cities and mongo_profile.get("resident_cities"):
                        profile.resident_cities = list(mongo_profile["resident_cities"])
                    if not profile.work_cities and mongo_profile.get("work_cities"):
                        profile.work_cities = list(mongo_profile["work_cities"])
                    if not profile.target_role and mongo_profile.get("target_role"):
                        profile.target_role = mongo_profile["target_role"]
                    if not effective_track and mongo_profile.get("active_track"):
                        effective_track = mongo_profile["active_track"]
                        logger.info(
                            "discovery_track_from_s_track",
                            user_id=user_id,
                            track=effective_track,
                        )
                    logger.info("discovery_profile_enriched_from_mongo", user_id=user_id)
            except Exception as exc:
                logger.warning("discovery_mongo_profile_lookup_failed", user_id=user_id, error=str(exc))

        tracks = _dedupe_tracks([effective_track] if effective_track else [])
        if not tracks:
            logger.warning("discovery_no_tracks", user_id=user_id, reason="no_active_track_resolved")
            return []

        if max_tracks > 0:
            tracks = tracks[:max_tracks]

        locations = self.get_profile_locations(profile)
        if not locations:
            fallback_location = await self.get_user_location(session, user_id)
            if fallback_location:
                locations = [fallback_location]
        if not locations:
            locations = ["remote"]

        states = self.get_profile_states(profile)

        logger.info(
            "discovery_starting",
            user_id=user_id,
            track_count=len(tracks),
            tracks=tracks,
            locations=[_readable_location(loc) for loc in locations],
            states=states,
        )

        # Collect existing URLs to avoid duplicates
        existing = await session.execute(
            select(DiscoveredOpportunity.url).where(
                DiscoveredOpportunity.user_id == user_id,
                DiscoveredOpportunity.status.in_(["new", "notified"]),
            )
        )
        seen_urls: set[str] = {r[0] for r in existing.all() if r[0]}

        event_results: list[dict[str, Any]] = []
        job_results: list[dict[str, Any]] = []
        if opportunity_type in {"all", "event"}:
            raw_events = await self._discover_events(
                tracks, locations, states, profile, seen_urls, queries_per_category
            )
            # Cap per source so no single platform dominates the dashboard.
            source_counts: dict[str, int] = {}
            for ev in sorted(raw_events, key=lambda x: x.get("match_score", Decimal("0")), reverse=True):
                src = str(ev.get("_source") or "unknown")
                if source_counts.get(src, 0) < _MAX_EVENTS_PER_SOURCE:
                    source_counts[src] = source_counts.get(src, 0) + 1
                    event_results.append(ev)
        if opportunity_type in {"all", "job"}:
            job_results = await self._discover_jobs(
                tracks,
                locations,
                states,
                profile,
                seen_urls,
                queries_per_category,
            )

        all_parsed = await self._generate_strategies_batch(event_results + job_results)

        discovered: list[DiscoveredOpportunity] = []
        for parsed in all_parsed:
            opp_type = parsed["opportunity_type"]
            expires_days = 30 if opp_type == "event" else 14

            meta: dict[str, Any] = {
                "source": parsed.get("_source", "unknown"),
                "query": parsed.get("_query", ""),
                "search_locations": [_readable_location(loc) for loc in locations],
                "reason_tags": parsed.get("reason_tags", ["matched_track"]),
                "tier": parsed.get("tier", 4),
            }
            if parsed.get("networking_strategy"):
                meta["networking_strategy"] = parsed["networking_strategy"]
            if parsed.get("application_strategy"):
                meta["application_strategy"] = parsed["application_strategy"]

            opp = DiscoveredOpportunity(
                user_id=user_id,
                opportunity_type=opp_type,
                title=parsed["title"],
                description=parsed["description"],
                url=parsed["url"],
                location=parsed["location"],
                company=parsed.get("company"),
                salary_range=parsed.get("salary_range"),
                # Normalise to kebab-lowercase so the format is consistent
                # regardless of whether it came from the LMS API, MongoDB or
                # user preferences (e.g. "Data Analytics" → "data-analytics").
                matched_track=_normalise_track(parsed["matched_track"]),
                match_score=parsed["match_score"],
                expires_at=datetime.now(UTC) + timedelta(days=expires_days),
                metadata_json=meta,
            )
            session.add(opp)
            discovered.append(opp)

        # Count by type BEFORE commit so we don't trigger lazy-loads on expired objects
        event_count = sum(1 for p in all_parsed if p.get("opportunity_type") == "event")
        job_count = sum(1 for p in all_parsed if p.get("opportunity_type") == "job")

        await session.commit()

        logger.info(
            "opportunities_discovered",
            user_id=user_id,
            total=len(discovered),
            events=event_count,
            jobs=job_count,
        )
        return discovered

    # ------------------------------------------------------------------
    # Notification helpers
    # ------------------------------------------------------------------

    async def create_opportunity_notification(
        self,
        session: AsyncSession,
        opportunity: DiscoveredOpportunity,
    ) -> Notification:
        type_label = opportunity.opportunity_type

        if type_label == "event":
            title = "New Event Matches Your Track"
            content = f"We found a {opportunity.matched_track} event for you: {opportunity.title}"
            action_buttons = [
                {"action": "view", "title": "View Event", "url": opportunity.url},
                {
                    "action": "strategy",
                    "title": "Get Networking Strategy",
                    "url": f"/dashboard/opportunities?id={opportunity.id}",
                },
                {"action": "dismiss", "title": "Not Interested"},
            ]
        else:
            company_part = f" at {opportunity.company}" if opportunity.company else ""
            title = "Job Opportunity Alert"
            content = f"New {opportunity.matched_track} role: {opportunity.title}{company_part}"
            action_buttons = [
                {"action": "view", "title": "View Job", "url": opportunity.url},
                {
                    "action": "strategy",
                    "title": "Application Strategy",
                    "url": f"/dashboard/job-board?id={opportunity.id}",
                },
                {"action": "dismiss", "title": "Not Interested"},
            ]

        notification = Notification(
            user_id=opportunity.user_id,
            title=title,
            content=content,
            channel="in_app",
            priority="normal",
            category="opportunity",
            action_buttons=action_buttons,
            metadata_json={
                "opportunity_id": opportunity.id,
                "opportunity_type": opportunity.opportunity_type,
                "url": opportunity.url,
            },
            expires_at=opportunity.expires_at,
        )
        session.add(notification)
        opportunity.status = "notified"
        return notification

    async def create_notifications_batch(
        self,
        session: AsyncSession,
        opportunities: list[DiscoveredOpportunity],
    ) -> list[Notification]:
        """Create notifications for all opportunities in a single commit."""
        notifications = []
        for opp in opportunities:
            n = await self.create_opportunity_notification(session, opp)
            notifications.append(n)
        await session.commit()
        return notifications


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

opportunity_engine = OpportunityDiscoveryEngine()

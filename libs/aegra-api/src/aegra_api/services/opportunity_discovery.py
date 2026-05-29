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
from aegra_api.services.opportunity_event_sources import EventbriteSource, LumaSource, MeetupSource, RawEventOpportunity
from aegra_api.services.opportunity_job_sources import (
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
        self._event_sources = [EventbriteSource(), MeetupSource(), LumaSource()]
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
        profile: StudentProfile | None,
        seen_urls: set[str],
        queries_per_category: int = 2,
    ) -> list[dict[str, Any]]:
        """Discover events via all configured event sources (Eventbrite, Meetup, …)."""
        sem = asyncio.Semaphore(2)
        # Build allowed-location set: full country name + ISO alpha-2 code (lower)
        # so venue strings like "Albert's Schloss, GB" or "Lagos, NG" can match.
        readable_locations: set[str] = set()
        for loc in locations:
            readable_locations.add(_readable_location(loc).lower())
            try:
                results = pycountry.countries.search_fuzzy(loc)
                if results:
                    readable_locations.add(results[0].alpha_2.lower())
            except (LookupError, AttributeError):
                pass

        async def _fetch_one(query: str, location: str, track: str) -> list[dict[str, Any]]:
            async with sem:
                # Fan out across all event sources and merge results
                source_batches = await asyncio.gather(
                    *[src.fetch(query, location) for src in self._event_sources],
                    return_exceptions=True,
                )
                raw_events: list[RawEventOpportunity] = []
                for batch in source_batches:
                    if isinstance(batch, list):
                        raw_events.extend(batch)
                parsed: list[dict[str, Any]] = []
                for ev in raw_events:
                    if not ev.url or ev.url in seen_urls:
                        continue
                    seen_urls.add(ev.url)
                    score = _score_text(f"{ev.title} {ev.description}", track)
                    # Lu.ma pre-filters by category slug (tech/ai) so the slug itself
                    # acts as a relevance signal — give it one keyword match's worth.
                    if ev.source == "luma" and score <= Decimal("0.50"):
                        score = Decimal("0.58")
                    # Require at least one keyword match (base score 0.50 means no match)
                    elif score <= Decimal("0.50"):
                        continue
                    # Location guard: accept online/remote events anywhere;
                    # physical events must match one of the user's allowed locations
                    ev_loc = (ev.location or "").lower()
                    if (
                        ev_loc
                        and not any(kw in ev_loc for kw in ("online", "remote", "virtual"))
                        and not self._job_location_allowed(ev.location, readable_locations)
                    ):
                        logger.debug(
                            "event_location_filtered",
                            title=ev.title[:60],
                            event_location=ev.location,
                            allowed=list(readable_locations),
                        )
                        continue
                    parsed.append(
                        {
                            "opportunity_type": "event",
                            "title": ev.title,
                            "description": ev.description,
                            "url": ev.url,
                            "location": ev.location or _readable_location(location),
                            "event_date": ev.event_date.isoformat() if ev.event_date else None,
                            "company": None,
                            "salary_range": None,
                            "match_score": score,
                            "matched_track": track,
                            "reason_tags": ["matched_track"],
                            "_source": ev.source,
                            "_query": query,
                        }
                    )
                return parsed

        tasks = []
        for track in tracks:
            queries = self._build_event_queries(track)
            for loc in locations:
                for q in queries[:queries_per_category]:
                    tasks.append(_fetch_one(q, loc, track))

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
                    if any(token in part or part in token for part in loc_parts):
                        return True
        return False

    async def _discover_jobs(
        self,
        tracks: list[str],
        locations: list[str],
        profile: StudentProfile | None,
        seen_urls: set[str],
        queries_per_category: int = 2,
    ) -> list[dict[str, Any]]:
        """Discover jobs via the board-based JobSourceRegistry.

        Results are split into two buckets of 5 each:
        - Track bucket  — jobs found via track keywords (e.g. "AI engineer")
        - Role bucket   — jobs found via the user's target role (e.g. "Data Product Manager")

        Both search terms are always fetched across every user location.
        When no target_role is set the full 10-job budget goes to the track bucket.
        """
        search_locations = [_readable_location(loc) for loc in locations] if locations else ["remote"]
        allowed_set = {loc.lower() for loc in search_locations}

        # Identify the target-role term so we can route results to the right bucket.
        role_term: str = ""
        if profile and isinstance(profile.target_role, str) and profile.target_role.strip():
            role_term = profile.target_role.strip().lower()

        search_terms = self._build_job_search_terms(
            tracks,
            profile=profile,
            queries_per_category=queries_per_category,
        )

        # Always run at least 2 search terms (track + role) when a target role is set,
        # regardless of the queries_per_category setting.
        max_terms = max(2, queries_per_category) if role_term else queries_per_category

        # Fetch jobs for every user location so all work_countries are covered.
        all_raw_jobs: list[RawJobOpportunity] = []
        for loc in search_locations:
            context = JobDiscoveryContext(
                search_terms=search_terms,
                primary_location=loc,
                indeed_country=self._indeed_country([loc]),
                max_search_terms=max_terms,
            )
            try:
                batch = await self._job_registry.fetch_all(context)
                all_raw_jobs.extend(batch)
            except Exception as exc:
                logger.warning("job_registry_fetch_failed", location=loc, error=repr(exc))

        # When a target role is set, also run a dedicated remote search for that
        # role term. These results bypass location filtering (handled below) because
        # roles like "Data Product Manager" are predominantly posted in markets like
        # the US where they are absent from country-specific searches for NG/GH.
        # For users in well-represented markets (US/UK) the country-specific results
        # will already fill the role bucket before these are needed.
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
                try:
                    remote_batch = await self._job_registry.fetch_all(remote_ctx)
                    for job in remote_batch:
                        if job.url:
                            remote_role_urls.add(job.url)
                    all_raw_jobs.extend(remote_batch)
                except Exception as exc:
                    logger.warning("job_registry_remote_role_fetch_failed", role=role_term, error=repr(exc))

        primary_location = search_locations[0] if search_locations else "remote"
        all_jobs: list[dict[str, Any]] = []

        # Bucket caps: 5 track jobs + 5 role jobs = 10 per run.
        # If no target role, allow up to 10 track jobs.
        per_bucket = 5
        track_count = 0
        role_count = 0
        track_cap = per_bucket if role_term else per_bucket * 2
        role_cap = per_bucket if role_term else 0

        for raw in all_raw_jobs:
            if not raw.url or raw.url in seen_urls:
                continue

            is_role_job = role_term and (raw.source_query or "").strip().lower() == role_term

            # Location filter: always applied to track jobs.
            # For role jobs from the dedicated remote search we also allow jobs
            # from major English-speaking PM hiring markets (US, Canada,
            # Australia, Ireland) so that users in NG/GH — where PM roles are
            # rare — can still see realistic target-role listings. Jobs from
            # unrelated markets (India, Southeast Asia, etc.) are still excluded.
            is_remote_role = is_role_job and raw.url in remote_role_urls
            location_ok = self._job_location_allowed(raw.location, allowed_set)
            if not location_ok and is_remote_role:
                loc_lower = (raw.location or "").lower()
                location_ok = (
                    not loc_lower
                    or any(kw in loc_lower for kw in ("remote", "hybrid", "worldwide", "anywhere"))
                    or _is_us_location(raw.location or "")
                    or any(kw in loc_lower for kw in ("canada", "australia", "ireland"))
                )
            if not location_ok:
                logger.debug(
                    "job_location_filtered",
                    title=raw.title[:60],
                    job_location=raw.location,
                    allowed=list(allowed_set),
                )
                continue

            # Relevance gate: keep if track keywords appear OR the search query
            # substantially matches the job title.
            # Role jobs use a stricter anchor check (last 2 words of the role
            # term must both appear) to prevent generic word overlaps from
            # admitting unrelated jobs like "Data Scientist" for a
            # "Data Product Manager" query.
            content = f"{raw.title} {raw.description}"
            best_score = Decimal("0.50")
            best_track = tracks[0] if tracks else ""
            for track in tracks:
                score = _score_text(content, track)
                if score > best_score:
                    best_score = score
                    best_track = track

            if is_role_job:
                query_relevant = _role_matches_title(role_term, raw.title)
            else:
                query_relevant = _query_matches_title(raw.source_query or "", raw.title)

            if best_score <= Decimal("0.50") and not query_relevant:
                logger.debug(
                    "job_relevance_filtered",
                    title=raw.title[:60],
                    query=raw.source_query,
                )
                continue

            if best_score <= Decimal("0.50"):
                best_score = Decimal("0.60")
            if is_role_job:
                if role_count >= role_cap:
                    continue
                role_count += 1
            else:
                if track_count >= track_cap:
                    continue
                track_count += 1

            seen_urls.add(raw.url)
            all_jobs.append(
                {
                    "opportunity_type": "job",
                    "title": raw.title,
                    "description": raw.description,
                    "url": raw.url,
                    "location": raw.location or primary_location,
                    "event_date": None,
                    "company": raw.company,
                    "salary_range": raw.salary_range,
                    "match_score": best_score,
                    "matched_track": best_track,
                    "reason_tags": ["matched_track"],
                    "_source": raw.source,
                    "_query": raw.source_query,
                }
            )

        logger.info(
            "job_discovery_complete",
            total=len(all_jobs),
            track_jobs=track_count,
            role_jobs=role_count,
            tracks=tracks,
            locations=search_locations,
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
                    if not profile.target_role and mongo_profile.get("target_role"):
                        profile.target_role = mongo_profile["target_role"]
                    # Use the s_track-derived track as a final fallback when all
                    # other resolution paths (LMS API, subscription, prefs) return
                    # nothing.  This covers users on the ai-mentor plan whose
                    # subscription.track is null but whose onboarding s_track section
                    # clearly indicates which path they chose.
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

        logger.info(
            "discovery_starting",
            user_id=user_id,
            track_count=len(tracks),
            tracks=tracks,
            locations=[_readable_location(loc) for loc in locations],
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
            event_results = await self._discover_events(tracks, locations, profile, seen_urls, queries_per_category)
        if opportunity_type in {"all", "job"}:
            job_results = await self._discover_jobs(
                tracks,
                locations,
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

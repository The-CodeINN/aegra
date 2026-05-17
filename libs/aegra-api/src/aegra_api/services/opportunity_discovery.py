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
        "LLM",
        "deep learning",
        "ML Ops",
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


def _normalise_track(track: str) -> str:
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

        def _is_aligned(term: str) -> bool:
            return any(_score_text(term, track) > Decimal("0.50") for track in tracks)

        if profile and isinstance(profile.target_role, str) and _is_aligned(profile.target_role):
            _add(profile.target_role)
        if profile and isinstance(profile.role_title, str) and _is_aligned(profile.role_title):
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
        """Score a raw job against the user's active track and profile."""
        readable_locations = {_readable_location(loc).lower() for loc in locations} if locations else {"remote"}
        if not self._job_location_allowed(raw_job.location, readable_locations):
            return None

        content = f"{raw_job.title} {raw_job.description}"
        best_score = Decimal("0")
        best_track = tracks[0] if tracks else ""
        for track in tracks:
            score = _score_text(content, track)
            if score > best_score:
                best_score = score
                best_track = track

        if best_score <= Decimal("0.50"):
            return None

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
        """Score a raw event against a single active track and profile."""
        score = _score_text(f"{raw_event.title} {raw_event.description}", track)
        if raw_event.source == "luma" and score <= Decimal("0.50"):
            score = Decimal("0.58")
        elif score <= Decimal("0.50"):
            return None

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
        """Return True if the job location matches one of the user's allowed locations or is remote."""
        loc_lower = job_location.lower().strip()
        if not loc_lower or "remote" in loc_lower or "hybrid" in loc_lower:
            return True
        return any(allowed in loc_lower or loc_lower in allowed for allowed in readable_locations)

    async def _discover_jobs(
        self,
        tracks: list[str],
        locations: list[str],
        profile: StudentProfile | None,
        seen_urls: set[str],
        queries_per_category: int = 2,
    ) -> list[dict[str, Any]]:
        """Discover jobs via the board-based JobSourceRegistry.

        Runs a separate fetch for EACH user location (resident_country and
        work_countries) so that no profile location is skipped, then
        post-filters results to strictly keep only jobs matching one of the
        user's locations (or remote/hybrid).
        """
        search_locations = [_readable_location(loc) for loc in locations] if locations else ["remote"]
        # Lowercase set used for post-filtering
        allowed_set = {loc.lower() for loc in search_locations}

        search_terms = self._build_job_search_terms(
            tracks,
            profile=profile,
            queries_per_category=queries_per_category,
        )

        # Fetch jobs for every user location independently so both
        # resident_country and work_countries are searched
        all_raw_jobs: list[RawJobOpportunity] = []
        for loc in search_locations:
            context = JobDiscoveryContext(
                search_terms=search_terms,
                primary_location=loc,
                indeed_country=self._indeed_country([loc]),
                max_search_terms=queries_per_category,
            )
            try:
                batch = await self._job_registry.fetch_all(context)
                all_raw_jobs.extend(batch)
            except Exception as exc:
                logger.warning("job_registry_fetch_failed", location=loc, error=repr(exc))

        primary_location = search_locations[0] if search_locations else "remote"
        all_jobs: list[dict[str, Any]] = []
        source_counts: dict[str, int] = {}
        for raw in all_raw_jobs:
            if not raw.url or raw.url in seen_urls:
                continue

            # Strict location guard — only keep jobs from the user's own locations or remote
            if not self._job_location_allowed(raw.location, allowed_set):
                logger.debug(
                    "job_location_filtered",
                    title=raw.title[:60],
                    job_location=raw.location,
                    allowed=list(allowed_set),
                )
                continue

            # Match against all tracks, pick the best-scoring one
            best_score = Decimal("0")
            best_track = tracks[0] if tracks else ""
            content = f"{raw.title} {raw.description}"
            for track in tracks:
                score = _score_text(content, track)
                if score > best_score:
                    best_score = score
                    best_track = track

            # Require at least one keyword match (base score 0.50 means no match)
            if best_score <= Decimal("0.50"):
                continue

            source_key = raw.source or "unknown"
            source_count = source_counts.get(source_key, 0)
            if source_count >= 3:
                continue

            seen_urls.add(raw.url)
            source_counts[source_key] = source_count + 1
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
                matched_track=parsed["matched_track"],
                match_score=parsed["match_score"],
                expires_at=datetime.now(UTC) + timedelta(days=expires_days),
                metadata_json=meta,
            )
            session.add(opp)
            discovered.append(opp)

        await session.commit()

        logger.info(
            "opportunities_discovered",
            user_id=user_id,
            total=len(discovered),
            events=len([o for o in discovered if o.opportunity_type == "event"]),
            jobs=len([o for o in discovered if o.opportunity_type == "job"]),
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

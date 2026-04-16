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
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.core.accountability_orm import (
    DiscoveredOpportunity,
    Notification,
    UserPreferences,
)
from aegra_api.services.opportunity_event_sources import EventbriteSource, RawEventOpportunity
from aegra_api.services.opportunity_job_sources import (
    JobDiscoveryContext,
    JobSourceRegistry,
    RawJobOpportunity,
)
from aegra_api.services.student_profile import StudentProfile, fetch_student_profile
from aegra_api.settings import settings

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

COUNTRY_NAMES: dict[str, str] = {
    "GB": "United Kingdom",
    "UK": "United Kingdom",
    "US": "United States",
    "USA": "United States",
    "CA": "Canada",
    "AU": "Australia",
    "DE": "Germany",
    "FR": "France",
    "NL": "Netherlands",
    "IE": "Ireland",
    "NG": "Nigeria",
    "KE": "Kenya",
    "ZA": "South Africa",
    "IN": "India",
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
    "AT": "Austria",
    "BE": "Belgium",
    "NZ": "New Zealand",
    "JP": "Japan",
    "BR": "Brazil",
    "MX": "Mexico",
}


def _readable_location(raw: str) -> str:
    """Convert a raw location (could be a 2-letter code) to a readable name."""
    stripped = raw.strip()
    upper = stripped.upper()
    if upper in COUNTRY_NAMES:
        return COUNTRY_NAMES[upper]
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
        self._event_source = EventbriteSource()
        self._job_registry = JobSourceRegistry(settings.discovery.company_job_boards)

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

    async def _get_tracks_from_prefs_or_fallback(
        self,
        session: AsyncSession,
        user_id: str,
    ) -> list[str]:
        result = await session.execute(select(UserPreferences).where(UserPreferences.user_id == user_id))
        prefs = result.scalar_one_or_none()
        if prefs and prefs.preferences:
            stored = prefs.preferences
            tracks = stored.get("tracks", []) or []
            if not tracks:
                lt = stored.get("learning_track") or stored.get("track")
                if lt:
                    tracks = [lt] if isinstance(lt, str) else list(lt)
            if tracks:
                logger.info("discovery_tracks_from_prefs", user_id=user_id, tracks=tracks)
                return tracks

        fallback = list(TRACK_KEYWORDS.keys())
        logger.warning(
            "discovery_using_fallback_tracks",
            user_id=user_id,
            reason="no LMS enrollments or stored preferences",
            tracks=fallback,
        )
        return fallback

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
        seen_urls: set[str],
        queries_per_category: int = 2,
    ) -> list[dict[str, Any]]:
        """Discover events via Eventbrite."""
        sem = asyncio.Semaphore(2)

        async def _fetch_one(query: str, location: str, track: str) -> list[dict[str, Any]]:
            async with sem:
                raw_events: list[RawEventOpportunity] = await self._event_source.fetch(query, location)
                parsed: list[dict[str, Any]] = []
                for ev in raw_events:
                    if not ev.url or ev.url in seen_urls:
                        continue
                    seen_urls.add(ev.url)
                    score = _score_text(f"{ev.title} {ev.description}", track)
                    if score < Decimal("0.50"):
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
        """Pick the best Indeed country code from the user's locations."""
        mapping = {
            "GB": "UK",
            "UK": "UK",
            "UNITED KINGDOM": "UK",
            "US": "USA",
            "USA": "USA",
            "UNITED STATES": "USA",
            "CA": "Canada",
            "CANADA": "Canada",
            "AU": "Australia",
            "AUSTRALIA": "Australia",
            "DE": "Germany",
            "GERMANY": "Germany",
            "NG": "Nigeria",
            "NIGERIA": "Nigeria",
        }
        for loc in locations:
            key = loc.strip().upper()
            if key in mapping:
                return mapping[key]
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

        # Build a combined search-term list from all tracks
        search_terms: list[str] = []
        seen_terms: set[str] = set()
        for track in tracks:
            for kw in _keywords_for_track(track)[:queries_per_category]:
                if kw.lower() not in seen_terms:
                    seen_terms.add(kw.lower())
                    search_terms.append(kw)

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

            if best_score < Decimal("0.50"):
                continue

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
        **_kwargs: Any,
    ) -> list[DiscoveredOpportunity]:
        """Run full discovery pipeline for a single user."""
        profile = await self._get_student_profile(session, user_id, auth_token)

        tracks = _dedupe_tracks(
            [
                *([profile.learning_track] if profile.learning_track else []),
                *(profile.enrolled_tracks or []),
            ]
        )
        if not tracks:
            tracks = await self._get_tracks_from_prefs_or_fallback(session, user_id)
        if not tracks:
            logger.warning("discovery_no_tracks", user_id=user_id)
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

        event_results, job_results = await asyncio.gather(
            self._discover_events(tracks, locations, seen_urls, queries_per_category),
            self._discover_jobs(tracks, locations, seen_urls, queries_per_category),
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

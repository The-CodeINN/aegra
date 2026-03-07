"""Opportunity Discovery Engine — v7.

AI-powered discovery of events and job opportunities
matched to each user's enrolled courses, career goals, and profile.

Changes in v7
--------------
* Both events AND jobs use Serper.dev (Google Search via google.serper.dev)
* Brave, SerpAPI, Claude all REMOVED
* Provider selection REMOVED — single backend: Serper.dev
* Auto scan: 2× per day (scheduler)
* Manual scan: max 4 per user per day (rate-limited at API layer)
* AI strategy generation restored DURING discovery (per requirement.md)
  - Events → generate_networking_strategy()
  - Jobs → generate_application_strategy()
* Redundant LMS calls removed — profile.enrolled_tracks reused for tracks
* Parallel search per track via asyncio.gather for speed
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from urllib.parse import urlparse

import httpx
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.core.accountability_orm import (
    DiscoveredOpportunity,
    Notification,
    UserPreferences,
)
from aegra_api.services.student_profile import StudentProfile, fetch_student_profile
from aegra_api.settings import settings

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Country code → readable name (common ones)
# ---------------------------------------------------------------------------
COUNTRY_NAMES: dict[str, str] = {
    "GB": "United Kingdom",
    "UK": "United Kingdom",
    "US": "United States",
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
    # Already a readable name like "Glasgow, Scotland"
    return stripped


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


# ---------------------------------------------------------------------------
# Track → keyword mapping
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
# Domain-based classification
# ---------------------------------------------------------------------------
EVENT_DOMAINS = frozenset(
    [
        "eventbrite.com",
        "eventbrite.co.uk",
        "eventbrite.co",
        "meetup.com",
        "10times.com",
        "confs.tech",
        "dev.events",
        "lu.ma",
        "luma.com",
        "conference-service.com",
        "techmeetups.com",
    ]
)

JOB_DOMAINS = frozenset(
    [
        "linkedin.com",
        "indeed.com",
        "indeed.co.uk",
        "glassdoor.com",
        "glassdoor.co.uk",
        "jobs.ashbyhq.com",
        "boards.greenhouse.io",
        "lever.co",
        "jobs.workable.com",
        "jobs.smartrecruiters.com",
        "wellfound.com",
        "otta.com",
        "reed.co.uk",
        "totaljobs.com",
        "cwjobs.co.uk",
        "monster.com",
    ]
)

# Content keywords as fallback when domain is not in either set
EVENT_SIGNALS = frozenset(
    [
        "meetup",
        "event",
        "workshop",
        "conference",
        "webinar",
        "summit",
        "hackathon",
        "bootcamp",
        "seminar",
        "networking",
        "talk",
        "panel",
        "fireside",
    ]
)

JOB_SIGNALS = frozenset(
    [
        "job",
        "career",
        "hiring",
        "position",
        "apply",
        "vacancy",
        "opening",
        "recruit",
        "employment",
    ]
)


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


def _domain_of(url: str) -> str:
    """Extract the registerable domain from a URL."""
    try:
        host = urlparse(url).hostname or ""
        # Strip www. prefix
        if host.startswith("www."):
            host = host[4:]
        return host.lower()
    except Exception:
        return ""


def _classify_result(url: str, title: str, description: str) -> str | None:
    """Classify a search result as 'event', 'job', or None (skip).

    Priority:
    1. Domain match (most reliable)
    2. Content keyword match (fallback)
    """
    domain = _domain_of(url)

    # Check domain first — most reliable signal
    for ed in EVENT_DOMAINS:
        if domain == ed or domain.endswith("." + ed):
            return "event"
    for jd in JOB_DOMAINS:
        if domain == jd or domain.endswith("." + jd):
            return "job"

    # Fallback to content analysis
    content = f"{title} {description}".lower()
    event_hits = sum(1 for kw in EVENT_SIGNALS if kw in content)
    job_hits = sum(1 for kw in JOB_SIGNALS if kw in content)

    if event_hits > job_hits and event_hits >= 2:
        return "event"
    if job_hits > event_hits and job_hits >= 2:
        return "job"
    if event_hits > 0:
        return "event"
    if job_hits > 0:
        return "job"

    return None


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------
class OpportunityDiscoveryEngine:
    """Discovers relevant opportunities (events, jobs) for users.

    Architecture: Serper.dev (google.serper.dev) for both events and jobs.
    """

    def __init__(
        self,
        serper_api_key: str | None = None,
    ) -> None:
        self.serper_api_key = serper_api_key or settings.discovery.SERPER_API_KEY
        self.lms_base_url = settings.app.LMS_URL

    # ------------------------------------------------------------------
    # LMS integration
    # ------------------------------------------------------------------
    async def get_user_enrollments(self, user_id: str, auth_token: str) -> list[dict]:
        """Fetch user's enrolled courses/tracks from LMS API."""
        if not self.lms_base_url:
            logger.warning("LMS API URL not configured")
            return []

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.lms_base_url}/api/v1/enrollment/student/blackboard",
                    headers={"Authorization": f"Bearer {auth_token}"},
                    timeout=10.0,
                )
                response.raise_for_status()
                data = response.json()
                enrollments = data.get("enrollments", [])

                track_names = []
                for e in enrollments:
                    course = e.get("course", {})
                    name = course.get("title") or course.get("track") or e.get("trackName")
                    track_names.append(name)

                logger.info(
                    "Fetched enrollments from LMS",
                    user_id=user_id,
                    count=len(enrollments),
                    tracks=track_names,
                )
                return enrollments
        except httpx.HTTPError as e:
            logger.error("Failed to fetch enrollments", error=str(e), user_id=user_id)
            return []

    async def get_user_location(self, session: AsyncSession, user_id: str) -> str | None:
        """Get user's location from user_preferences."""
        result = await session.execute(select(UserPreferences).where(UserPreferences.user_id == user_id))
        prefs = result.scalar_one_or_none()
        return prefs.location if prefs else None

    def get_profile_locations(self, profile: StudentProfile | None) -> list[str]:
        """Extract ordered location candidates from the student profile."""
        if not profile:
            return []

        return _dedupe_locations(
            [
                *(profile.work_countries or []),
                profile.resident_country,
            ]
        )

    # ------------------------------------------------------------------
    # Query builders
    # ------------------------------------------------------------------
    def _keywords_for_track(self, track: str) -> list[str]:
        key = _normalise_track(track)
        return TRACK_KEYWORDS.get(key, [track.lower()])

    def _primary_keyword(self, track: str) -> str:
        """Get the single best keyword for a track."""
        keywords = self._keywords_for_track(track)
        return keywords[0] if keywords else track.replace("-", " ")

    def build_event_queries(
        self,
        track: str,
        location: str,
        profile: StudentProfile | None = None,
    ) -> list[str]:
        """Build event discovery queries for Serper.dev.

        Uses boolean OR operators for broad coverage across event types.
        """
        kw = self._primary_keyword(track)
        loc = _readable_location(location)
        alt_keywords = self._keywords_for_track(track)[:3]
        kw_or = " OR ".join(f'"{k}"' for k in alt_keywords)
        queries: list[str] = []

        # Broad event query with boolean operators
        queries.append(f'({kw_or}) ("conference" OR "summit" OR "meetup" OR "workshop") ("{loc}")')

        # Platform-targeted
        queries.append(f'site:eventbrite.com "{kw}" "{loc}"')
        queries.append(f'site:meetup.com "{kw}" "{loc}"')

        # Career fairs if relevant
        if profile and profile.primary_goal:
            goal = profile.primary_goal.lower()
            if any(w in goal for w in ["job", "career", "hire", "transition"]):
                queries.append(f'"career fair" OR "hiring event" "{kw}" "{loc}"')

        return queries

    def build_job_queries(
        self,
        track: str,
        location: str,
        profile: StudentProfile | None = None,
    ) -> list[str]:
        """Build job discovery queries for Serper.dev.

        Uses site: operators for specific job boards + a general query.
        """
        loc = _readable_location(location)

        # Track stays primary; target role is a secondary boost.
        primary_title = self._primary_keyword(track)
        secondary_titles: list[str] = []

        if profile and profile.target_role:
            secondary_titles.append(profile.target_role)

        if profile and profile.target_job_titles:
            secondary_titles.extend(profile.target_job_titles)

        deduped_titles: list[str] = []
        seen_titles: set[str] = set()
        for title in [primary_title, *secondary_titles]:
            normalized = title.strip().lower()
            if not normalized or normalized in seen_titles:
                continue
            seen_titles.add(normalized)
            deduped_titles.append(title.strip())

        # Experience level hint
        level_hint = ""
        if profile and profile.experience_level in ("entry-level", "junior"):
            level_hint = "junior"

        queries: list[str] = []
        job_boards = [
            "boards.greenhouse.io",
            "jobs.lever.co",
            "jobs.ashbyhq.com",
        ]

        def format_title(title: str) -> str:
            return f"{level_hint} {title}".strip() if level_hint else title

        primary_search_title = format_title(deduped_titles[0]) if deduped_titles else primary_title
        queries.append(f'"{primary_search_title}" "hiring" OR "job" "{loc}"')

        for secondary_title in deduped_titles[1:]:
            secondary_search_title = format_title(secondary_title)
            queries.append(f'"{secondary_search_title}" "hiring" OR "job" "{loc}"')

        for title in deduped_titles:
            search_title = format_title(title)
            for board in job_boards:
                queries.append(f'site:{board} "{search_title}" "{loc}"')

        return queries

    # ------------------------------------------------------------------
    # Serper.dev Google Search (for both events and jobs)
    # ------------------------------------------------------------------
    async def serper_search(self, query: str, num: int = 10) -> list[dict[str, Any]]:
        """Execute a Google search via Serper.dev POST API."""
        if not self.serper_api_key:
            logger.warning("serper_search_skipped", reason="SERPER_API_KEY not set")
            return []

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://google.serper.dev/search",
                    headers={
                        "X-API-KEY": self.serper_api_key,
                        "Content-Type": "application/json",
                    },
                    json={"q": query, "num": num},
                    timeout=15.0,
                )
                resp.raise_for_status()
                data = resp.json()
                results = data.get("organic", [])
                logger.info(
                    "serper_search_completed",
                    query=query[:80],
                    result_count=len(results),
                )
                return results
        except httpx.HTTPError as e:
            logger.error("serper_search_failed", error=str(e), query=query[:80])
            return []

    # ------------------------------------------------------------------
    # Result parsing (URL-domain based classification)
    # ------------------------------------------------------------------
    def _extract_company(self, title: str, url: str) -> str | None:
        clean = title.replace(" | LinkedIn", "").replace(" | Indeed", "").strip()
        is_linkedin = "linkedin.com" in url

        if " at " in clean:
            parts = clean.split(" at ")
            if len(parts) > 1:
                return parts[-1].split(" - ")[0].strip()
        elif " - " in clean:
            parts = clean.split(" - ")
            if is_linkedin and len(parts) >= 3:
                return parts[1].strip()
            if len(parts) > 1:
                return parts[-1].strip()
        return None

    def _score(self, result: dict[str, Any], track: str) -> Decimal:
        """Deterministic relevance score 0.00–1.00."""
        score = 0.50
        desc = result.get("description", "") or result.get("snippet", "")
        content = f"{result.get('title', '')} {desc}".lower()
        keywords = self._keywords_for_track(track)
        for kw in keywords:
            if kw.lower() in content:
                score += 0.08
        return Decimal(str(min(round(score, 2), 1.0)))

    def parse_result(
        self,
        result: dict[str, Any],
        track: str,
        location: str,
    ) -> dict[str, Any] | None:
        """Parse and classify a Serper.dev organic result.

        Serper returns: title, link, snippet, position, etc.
        """
        title = result.get("title", "")
        desc = result.get("snippet", "") or result.get("description", "")
        url = result.get("link", "") or result.get("url", "")

        if not url or not title:
            return None

        opp_type = _classify_result(url, title, desc)
        if not opp_type:
            return None

        clean_title = title.replace(" | LinkedIn", "").replace(" | Indeed", "").replace(" | Glassdoor", "").strip()

        parsed: dict[str, Any] = {
            "opportunity_type": opp_type,
            "title": clean_title,
            "description": desc,
            "url": url,
            "location": _readable_location(location),
            "matched_track": track,
            "match_score": self._score(result, track),
        }

        if opp_type == "job":
            parsed["company"] = self._extract_company(title, url)

        return parsed

    # ------------------------------------------------------------------
    # AI enrichment helpers (called on-demand, NOT during discovery)
    # ------------------------------------------------------------------
    async def generate_networking_strategy(self, opportunity: dict[str, Any], user_track: str) -> dict[str, Any] | None:
        """Generate a personalised networking strategy for an event."""
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=500, request_timeout=12)
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
            from langchain_core.messages import HumanMessage, SystemMessage
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=500, request_timeout=12)
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

    # ------------------------------------------------------------------
    # Batch strategy generation (runs during discovery)
    # ------------------------------------------------------------------
    async def _generate_strategies_batch(
        self,
        parsed_results: list[dict[str, Any]],
        max_strategies: int = 15,
    ) -> list[dict[str, Any]]:
        """Generate AI strategies for top-scoring results in parallel.

        Per requirement.md:
        - Events → generate_networking_strategy()
        - Jobs  → generate_application_strategy()

        Only the top `max_strategies` results (by match_score) get strategies
        generated during discovery — the rest can be loaded on-demand via
        /opportunities/{id}/strategy.
        """
        if not parsed_results:
            return parsed_results

        # Sort by match_score descending, pick top N for strategy generation
        scored = sorted(parsed_results, key=lambda x: x.get("match_score", 0), reverse=True)
        to_generate = scored[:max_strategies]
        remaining = scored[max_strategies:]

        sem = asyncio.Semaphore(4)

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
    # Core discovery flow
    # ------------------------------------------------------------------
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
        """Resolve track names from DB preferences or use fallback defaults.

        Called only when profile.enrolled_tracks is empty (LMS had no data).
        """
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

    async def _get_tracks_for_user(
        self,
        session: AsyncSession,
        user_id: str,
        auth_token: str | None = None,
    ) -> list[str]:
        """Resolve track names for a user."""
        # 1 — Try LMS
        if auth_token and auth_token != "scheduled_job_token":  # nosec B105
            enrollments = await self.get_user_enrollments(user_id, auth_token)
            if enrollments:
                tracks = []
                for e in enrollments:
                    course = e.get("course", {})
                    name = course.get("title") or course.get("track") or e.get("trackName", "")
                    if name:
                        tracks.append(name)
                if tracks:
                    logger.info("discovery_tracks_from_lms", user_id=user_id, tracks=tracks)
                    return tracks

        # 2 — Try preferences JSONB
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

        # 3 — Fallback
        fallback = list(TRACK_KEYWORDS.keys())
        logger.warning(
            "discovery_using_fallback_tracks",
            user_id=user_id,
            reason="no LMS enrollments or stored preferences",
            tracks=fallback,
        )
        return fallback

    async def _discover_events(
        self,
        tracks: list[str],
        locations: list[str],
        profile: StudentProfile | None,
        seen_urls: set[str],
        queries_per_category: int = 2,
    ) -> list[dict[str, Any]]:
        """Discover events via Serper.dev."""
        sem = asyncio.Semaphore(4)

        async def _search_one(query: str, track: str) -> list[dict[str, Any]]:
            async with sem:
                results = await self.serper_search(query, num=10)
                parsed_list: list[dict[str, Any]] = []
                for r in results:
                    url = r.get("link", "") or r.get("url", "")
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    parsed = self.parse_result(r, track, location)
                    if parsed and parsed["opportunity_type"] == "event" and parsed["match_score"] >= Decimal("0.50"):
                        parsed["_query"] = query
                        parsed["_source"] = "serper"
                        parsed_list.append(parsed)
                return parsed_list

        tasks = []
        for track_name in tracks:
            for location in locations:
                event_queries = self.build_event_queries(track_name, location, profile)
                for q in event_queries[:queries_per_category]:
                    tasks.append(_search_one(q, track_name))

        gathered = await asyncio.gather(*tasks, return_exceptions=True)
        all_events: list[dict[str, Any]] = []
        for batch in gathered:
            if isinstance(batch, list):
                all_events.extend(batch)
        return all_events

    async def _discover_jobs(
        self,
        tracks: list[str],
        locations: list[str],
        profile: StudentProfile | None,
        seen_urls: set[str],
        queries_per_category: int = 2,
    ) -> list[dict[str, Any]]:
        """Discover jobs via Serper.dev (site: operator queries)."""
        sem = asyncio.Semaphore(4)

        async def _search_one(query: str, track: str) -> list[dict[str, Any]]:
            async with sem:
                results = await self.serper_search(query, num=10)
                parsed_list: list[dict[str, Any]] = []
                for r in results:
                    url = r.get("link", "") or r.get("url", "")
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    parsed = self.parse_result(r, track, location)
                    if parsed and parsed["opportunity_type"] == "job" and parsed["match_score"] >= Decimal("0.50"):
                        parsed["_query"] = query
                        parsed["_source"] = "serper"
                        parsed_list.append(parsed)
                return parsed_list

        tasks = []
        for track_name in tracks:
            for location in locations:
                job_queries = self.build_job_queries(track_name, location, profile)
                for q in job_queries[:queries_per_category]:
                    tasks.append(_search_one(q, track_name))

        gathered = await asyncio.gather(*tasks, return_exceptions=True)
        all_jobs: list[dict[str, Any]] = []
        for batch in gathered:
            if isinstance(batch, list):
                all_jobs.extend(batch)
        return all_jobs

    async def discover_for_user(
        self,
        session: AsyncSession,
        user_id: str,
        auth_token: str = "",
        max_tracks: int = 0,
        queries_per_category: int = 2,
        **_kwargs: Any,
    ) -> list[DiscoveredOpportunity]:
        """Run full discovery pipeline for a single user.

        Events + Jobs → Serper.dev (google.serper.dev)
        AI strategy generated for each opportunity (per requirement.md):
        - Events → generate_networking_strategy()
        - Jobs → generate_application_strategy()
        """
        profile = await self._get_student_profile(session, user_id, auth_token)

        # Derive tracks from the profile that was already fetched (avoid
        # a redundant LMS call). Fall back to DB prefs / defaults only
        # if profile has no enrollment data.
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
            locations=[_readable_location(location) for location in locations],
            target_role=profile.target_role,
            is_job_searching=profile.is_job_searching,
        )

        # Collect existing URLs to avoid duplicating
        existing = await session.execute(
            select(DiscoveredOpportunity.url).where(
                DiscoveredOpportunity.user_id == user_id,
                DiscoveredOpportunity.status.in_(["new", "notified"]),
            )
        )
        seen_urls: set[str] = {r[0] for r in existing.all() if r[0]}
        logger.info("discovery_existing_urls", count=len(seen_urls))

        # Run event and job discovery in parallel
        event_results, job_results = await asyncio.gather(
            self._discover_events(tracks, locations, profile, seen_urls, queries_per_category),
            self._discover_jobs(tracks, locations, profile, seen_urls, queries_per_category),
        )

        all_parsed = event_results + job_results

        # ── Generate AI strategies in parallel (per requirement.md) ──
        all_parsed = await self._generate_strategies_batch(all_parsed)

        # Create DB records
        discovered: list[DiscoveredOpportunity] = []
        for parsed in all_parsed:
            opp_type = parsed["opportunity_type"]
            expires_days = 30 if opp_type == "event" else 14

            meta: dict[str, Any] = {
                "source": parsed.get("_source", "unknown"),
                "query": parsed.get("_query", ""),
                "search_locations": [_readable_location(location) for location in locations],
            }
            # Attach strategy to metadata so frontend can display it
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
            title = "🎯 New Event Matches Your Track"
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
            title = "💼 Job Opportunity Alert"
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

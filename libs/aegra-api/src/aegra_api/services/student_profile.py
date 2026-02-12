"""Student Profile Service — fetches rich student data from the LMS API.

Aggregates data from multiple LMS endpoints to build a comprehensive
student profile used for personalized opportunity discovery,
notification generation, and accountability tracking.

LMS endpoints used:
- GET /api/v1/user/profile          → name, email, profileImage
- GET /api/v1/enrollment/student/blackboard → enrolled tracks, progress
- GET /api/v1/ai-mentor/onboarding/section-1 → learning track, situation, weekly time
- GET /api/v1/ai-mentor/onboarding/section-2 → employment, role, industry, location, experience
- GET /api/v1/ai-mentor/onboarding/section-4 → goals, target role, timeline
- GET /api/v1/ai-mentor/onboarding/section-5 → skills, profiles
- GET /api/v1/ai-mentor/onboarding/section-6 → challenges, job search stats
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import httpx
import structlog

from aegra_api.services.lms_cache import cached_lms_fetch
from aegra_api.settings import settings

logger = structlog.get_logger()


@dataclass
class StudentProfile:
    """Aggregated student profile from LMS data."""

    user_id: str = ""

    # Basic info (from /user/profile)
    first_name: str = ""
    email: str = ""

    # Enrollments (from /enrollment/student/blackboard)
    enrolled_tracks: list[str] = field(default_factory=list)
    overall_progress: dict[str, float] = field(default_factory=dict)  # track -> progress %

    # Section 1: Quick start
    learning_track: str = ""
    current_situation: str = ""
    weekly_time: str = ""

    # Section 2: Background
    employment_status: str = ""
    role_title: str = ""
    industry: str = ""
    years_experience: str = ""
    years_tech: str = ""
    resident_country: str = ""
    work_countries: list[str] = field(default_factory=list)

    # Section 4: Goals
    primary_goal: str = ""
    target_role: str = ""
    timeline: str = ""
    goal_why: str = ""

    # Section 5: Skills
    confident_skills: list[str] = field(default_factory=list)
    need_help_areas: list[str] = field(default_factory=list)
    linkedin_url: str = ""
    github_url: str = ""
    portfolio_url: str = ""

    # Section 6: Challenges
    biggest_challenge: str = ""
    apps_submitted: str = ""
    interviews: str = ""

    @property
    def location(self) -> str:
        """Best-effort location string from country data."""
        if self.resident_country:
            return self.resident_country
        return "remote"

    @property
    def is_job_searching(self) -> bool:
        """Whether the student is actively searching for jobs."""
        goal_lower = self.primary_goal.lower()
        return any(
            kw in goal_lower for kw in ["job", "career", "role", "position", "hired", "employment", "transition"]
        )

    @property
    def target_job_titles(self) -> list[str]:
        """Job titles to search for based on target role and tracks."""
        titles = []
        if self.target_role:
            titles.append(self.target_role)
        for track in self.enrolled_tracks:
            track_lower = track.lower()
            if "analytics" in track_lower:
                titles.extend(["data analyst", "analytics analyst", "junior data analyst"])
            elif "science" in track_lower:
                titles.extend(["data scientist", "junior data scientist", "ML engineer"])
            elif "engineering" in track_lower:
                titles.extend(["data engineer", "junior data engineer", "ETL developer"])
            elif "ai" in track_lower:
                titles.extend(["AI engineer", "ML engineer", "GenAI engineer"])
        return list(set(titles))

    @property
    def experience_level(self) -> str:
        """Infer experience level from years in tech."""
        if not self.years_tech:
            return "entry-level"
        yt = self.years_tech.lower()
        if any(kw in yt for kw in ["0", "none", "no experience", "less than"]):
            return "entry-level"
        if any(kw in yt for kw in ["1", "2"]):
            return "junior"
        return "mid-level"


async def fetch_student_profile(user_id: str, auth_token: str) -> StudentProfile:
    """Fetch and aggregate student profile from multiple LMS endpoints.

    Makes parallel requests to minimize latency.
    Responses are cached via Redis / in-memory (see lms_cache.py).
    """
    lms_url = settings.app.LMS_URL
    if not lms_url:
        logger.warning("LMS_URL not configured")
        return StudentProfile(user_id=user_id)

    profile = StudentProfile(user_id=user_id)

    async with httpx.AsyncClient() as client:
        # Fetch all sections in parallel (caching handled per-endpoint)
        results = await asyncio.gather(
            cached_lms_fetch(client, f"{lms_url}/api/v1/user/profile", auth_token, user_id),
            cached_lms_fetch(client, f"{lms_url}/api/v1/enrollment/student/blackboard", auth_token, user_id),
            cached_lms_fetch(client, f"{lms_url}/api/v1/ai-mentor/onboarding/section-1", auth_token, user_id),
            cached_lms_fetch(client, f"{lms_url}/api/v1/ai-mentor/onboarding/section-2", auth_token, user_id),
            cached_lms_fetch(client, f"{lms_url}/api/v1/ai-mentor/onboarding/section-4", auth_token, user_id),
            cached_lms_fetch(client, f"{lms_url}/api/v1/ai-mentor/onboarding/section-5", auth_token, user_id),
            cached_lms_fetch(client, f"{lms_url}/api/v1/ai-mentor/onboarding/section-6", auth_token, user_id),
            return_exceptions=True,
        )

        user_data, enrollment_data, s1, s2, s4, s5, s6 = [r if isinstance(r, dict) else {} for r in results]

        # Parse user profile
        user = user_data.get("user", user_data)
        name = user.get("name", "") or user.get("firstName", "") or ""
        profile.first_name = name.split()[0] if name else ""
        profile.email = user.get("email", "")

        # Parse enrollments
        enrollments = enrollment_data.get("enrollments", [])
        for e in enrollments:
            course = e.get("course", {})
            track_name = course.get("title") or course.get("track") or e.get("trackName", "")
            if track_name:
                profile.enrolled_tracks.append(track_name)
                progress = e.get("overallProgress", 0)
                profile.overall_progress[track_name] = progress

        # Section 1: Quick start
        sec1 = s1.get("s1", {})
        if sec1.get("completed"):
            profile.learning_track = sec1.get("learningTrack", "")
            profile.current_situation = sec1.get("situation", "")
            profile.weekly_time = sec1.get("weeklyTime", "")

        # Section 2: Background
        sec2 = s2.get("s2", {})
        if sec2.get("completed"):
            profile.employment_status = sec2.get("employmentStatus", "")
            profile.role_title = sec2.get("roleTitle", "")
            profile.industry = sec2.get("industry", "")
            profile.years_experience = sec2.get("yearsExperience", "")
            profile.years_tech = sec2.get("yearsTech", "")
            profile.resident_country = sec2.get("residentCountry", "")
            profile.work_countries = sec2.get("workCountry", []) or []

        # Section 4: Goals
        sec4 = s4.get("s4", {})
        if sec4.get("completed"):
            profile.primary_goal = sec4.get("primaryGoal", "")
            profile.target_role = sec4.get("targetRole", "")
            profile.timeline = sec4.get("timeline", "")
            profile.goal_why = sec4.get("goalWhy", "")

        # Section 5: Skills
        sec5 = s5.get("s5", {})
        if sec5.get("completed"):
            profile.confident_skills = sec5.get("confidentSkills", []) or []
            profile.need_help_areas = sec5.get("needHelpAreas", []) or []
            profiles = sec5.get("profiles", {}) or {}
            profile.linkedin_url = profiles.get("linkedin", "")
            profile.github_url = profiles.get("github", "")
            profile.portfolio_url = profiles.get("portfolio", "")

        # Section 6: Challenges
        sec6 = s6.get("s6", {})
        if sec6.get("completed"):
            profile.biggest_challenge = sec6.get("biggestChallenge", "")
            job_search = sec6.get("jobSearch", {}) or {}
            profile.apps_submitted = job_search.get("appsSubmitted", "")
            profile.interviews = job_search.get("interviews", "")

    logger.info(
        "student_profile_fetched",
        user_id=user_id,
        tracks=profile.enrolled_tracks,
        has_goals=bool(profile.primary_goal),
        has_skills=bool(profile.confident_skills),
        location=profile.location,
        is_job_searching=profile.is_job_searching,
    )

    return profile

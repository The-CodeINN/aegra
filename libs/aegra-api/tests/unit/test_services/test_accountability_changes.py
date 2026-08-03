"""Unit tests for accountability-system changes.

Covers:
- Track normalisation (opportunity API, service, discovery)
- _build_student_context enrollment data (scheduler)
- generate_personalized_motivational_content LLM prompt fields (notification engine)
- scan-rate-limit helpers (opportunity API)
- Frontend fetch-in-finally pattern is implicit — tested via the backend shape it relies on
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Track normalisation helpers
# ---------------------------------------------------------------------------


class TestTrackNormalisation:
    """The same normalisation logic appears in three places; test each."""

    def _norm_api(self, track: str | None) -> str | None:
        from aegra_api.api.opportunities import _normalise_track

        return _normalise_track(track)

    def _norm_service(self, track: str | None) -> str | None:
        from aegra_api.services.opportunity_service import OpportunityService

        return OpportunityService._normalise_track(track)

    def test_api_normalises_spaces_to_dashes(self) -> None:
        assert self._norm_api("AI Engineering") == "ai-engineering"

    def test_api_normalises_mixed_case(self) -> None:
        assert self._norm_api("Data Analytics") == "data-analytics"

    def test_api_already_normalised_is_unchanged(self) -> None:
        assert self._norm_api("data-science") == "data-science"

    def test_api_returns_none_for_none(self) -> None:
        assert self._norm_api(None) is None

    def test_api_returns_none_for_blank(self) -> None:
        assert self._norm_api("   ") is None

    def test_service_normalises_spaces(self) -> None:
        assert self._norm_service("Data Engineering") == "data-engineering"

    def test_service_matches_api_result(self) -> None:
        for track in ["AI Engineering", "data analytics", "Data-Science", "data-engineering"]:
            assert self._norm_api(track) == self._norm_service(track)

    def test_discovery_normalise_function(self) -> None:
        """opportunity_discovery._normalise_track must match the same contract."""
        from aegra_api.services.opportunity_discovery import _normalise_track

        assert _normalise_track("AI Engineering") == "ai-engineering"
        assert _normalise_track(None) is None
        assert _normalise_track("") is None


# ---------------------------------------------------------------------------
# Scan rate-limit day-key helpers (timezone-aware)
# ---------------------------------------------------------------------------


class TestScanDayKey:
    def _key(self, tz_name=None, tz_offset=None, now=None):
        from aegra_api.api.opportunities import _scan_day_key

        return _scan_day_key(timezone_name=tz_name, timezone_offset_minutes=tz_offset, now=now)

    def _count(self, scan_log, tz_name=None, tz_offset=None, now=None):
        from aegra_api.api.opportunities import _scan_count_for_day

        return _scan_count_for_day(scan_log, timezone_name=tz_name, timezone_offset_minutes=tz_offset, now=now)

    def _increment(self, scan_log, tz_name=None, tz_offset=None, now=None):
        from aegra_api.api.opportunities import _increment_scan_log

        return _increment_scan_log(scan_log, timezone_name=tz_name, timezone_offset_minutes=tz_offset, now=now)

    def test_utc_day_key_format(self) -> None:
        now = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
        assert self._key(now=now) == "2024-06-15"

    def test_positive_offset_advances_day(self) -> None:
        # 23:30 UTC with +60 min offset → next calendar day
        now = datetime(2024, 6, 15, 23, 30, 0, tzinfo=UTC)
        key_utc = self._key(now=now)
        key_plus1h = self._key(tz_offset=60, now=now)
        assert key_plus1h > key_utc

    def test_negative_offset_retreats_day(self) -> None:
        # 00:30 UTC with -60 min offset → previous calendar day
        now = datetime(2024, 6, 15, 0, 30, 0, tzinfo=UTC)
        key_utc = self._key(now=now)
        key_minus1h = self._key(tz_offset=-60, now=now)
        assert key_minus1h < key_utc

    def test_scan_count_zero_for_empty_log(self) -> None:
        assert self._count({}) == 0

    def test_scan_count_zero_for_none(self) -> None:
        assert self._count(None) == 0

    def test_scan_count_returns_todays_value(self) -> None:
        now = datetime(2024, 6, 15, 10, 0, 0, tzinfo=UTC)
        log = {"2024-06-15": 3, "2024-06-14": 10}
        assert self._count(log, now=now) == 3

    def test_increment_creates_key(self) -> None:
        now = datetime(2024, 6, 15, 10, 0, 0, tzinfo=UTC)
        updated = self._increment({}, now=now)
        assert updated.get("2024-06-15") == 1

    def test_increment_adds_to_existing(self) -> None:
        now = datetime(2024, 6, 15, 10, 0, 0, tzinfo=UTC)
        updated = self._increment({"2024-06-15": 2}, now=now)
        assert updated["2024-06-15"] == 3

    def test_increment_prunes_entries_older_than_7_days(self) -> None:
        now = datetime(2024, 6, 15, 10, 0, 0, tzinfo=UTC)
        old_log = {"2024-06-07": 5}  # exactly 8 days before -> should be pruned
        updated = self._increment(old_log, now=now)
        assert "2024-06-07" not in updated


# ---------------------------------------------------------------------------
# _build_student_context — enrollment data from MongoDB
# ---------------------------------------------------------------------------


class TestBuildStudentContextEnrollment:
    """Verify that enrollment data from get_enrollment_overview() is
    correctly mapped into the student context dict."""

    def _make_session(self, *, streak: int = 5, name: str = "Alice Smith") -> AsyncMock:
        """Return a minimal mock SQLAlchemy async session."""
        session = AsyncMock()

        # UserPreferences scalar
        prefs_mock = MagicMock()
        prefs_mock.preferences = {"user_name": name}
        prefs_result = MagicMock()
        prefs_result.scalar_one_or_none.return_value = prefs_mock

        # UserActivityTracking scalar
        activity_mock = MagicMock()
        activity_mock.current_streak = streak
        activity_result = MagicMock()
        activity_result.scalar_one_or_none.return_value = activity_mock

        # Count scalars (completed, overdue, pending) → 0
        count_result = MagicMock()
        count_result.scalar.return_value = 0

        # AccountabilityService.get_open_and_overdue's own query (spec Item 7:
        # _build_student_context now also fetches named tasks, not just counts).
        open_tasks_result = MagicMock()
        open_tasks_result.scalars.return_value.all.return_value = []

        session.execute = AsyncMock(
            side_effect=[
                prefs_result,  # UserPreferences
                activity_result,  # UserActivityTracking
                count_result,  # completed tasks
                count_result,  # overdue tasks
                count_result,  # pending tasks
                open_tasks_result,  # AccountabilityService.get_open_and_overdue
            ]
        )
        return session

    def _make_enrollment_data(self) -> dict[str, Any]:
        return {
            "enrollments": [
                {
                    "courseId": "abc123",
                    "course": {"title": "AI Engineering Bootcamp", "slug": "ai-eng", "track": "ai-engineering"},
                    "overallProgress": 43,
                    "totalCompletedLessons": 17,
                    "totalWatchedHours": 12.5,
                    "finalAssessmentPassed": False,
                    "aiMentorActive": True,
                    "enrolledAt": datetime(2024, 1, 1, tzinfo=UTC),
                    "updatedAt": datetime(2024, 6, 10, tzinfo=UTC),
                }
            ]
        }

    @pytest.mark.asyncio
    async def test_enrollment_fields_populated(self) -> None:
        from aegra_api.services.scheduler import SchedulerService

        svc = SchedulerService()
        session = self._make_session()

        enrollment_data = self._make_enrollment_data()

        with (
            patch("aegra_api.services.scheduler.get_course_content_mongo_client") as mock_client_factory,
        ):
            mongo_client = MagicMock()
            mongo_client.get_user_onboarding_data.return_value = {"target_role": "AI Engineer"}
            mongo_client.get_enrollment_overview.return_value = enrollment_data
            mock_client_factory.return_value = mongo_client

            ctx = await svc._build_student_context(session, "user-123", "David", "ai-engineering")

        assert ctx["enrolled_course"] == "AI Engineering Bootcamp"
        assert ctx["course_progress_pct"] == 43
        assert ctx["total_completed_lessons"] == 17
        assert ctx["total_watched_hours"] == 12.5
        assert ctx["last_active_in_course"] == "Jun 10"

    @pytest.mark.asyncio
    async def test_empty_enrollments_leaves_defaults(self) -> None:
        from aegra_api.services.scheduler import SchedulerService

        svc = SchedulerService()
        session = self._make_session()

        with (
            patch("aegra_api.services.scheduler.get_course_content_mongo_client") as mock_client_factory,
        ):
            mongo_client = MagicMock()
            mongo_client.get_user_onboarding_data.return_value = None
            mongo_client.get_enrollment_overview.return_value = {"enrollments": []}
            mock_client_factory.return_value = mongo_client

            ctx = await svc._build_student_context(session, "user-456", "Alexandra", "data-analytics")

        assert ctx["enrolled_course"] == ""
        assert ctx["course_progress_pct"] == 0
        assert ctx["total_completed_lessons"] == 0

    @pytest.mark.asyncio
    async def test_enrollment_exception_does_not_crash(self) -> None:
        """Even if MongoDB is down, the rest of the context is still returned."""
        from aegra_api.services.scheduler import SchedulerService

        svc = SchedulerService()
        session = self._make_session(streak=7)

        with (
            patch("aegra_api.services.scheduler.get_course_content_mongo_client") as mock_client_factory,
        ):
            mongo_client = MagicMock()
            mongo_client.get_user_onboarding_data.return_value = None
            mongo_client.get_enrollment_overview.side_effect = ConnectionError("mongo down")
            mock_client_factory.return_value = mongo_client

            ctx = await svc._build_student_context(session, "user-789", "David", "ai-engineering")

        # Streak still comes through from PostgreSQL
        assert ctx["current_streak"] == 7
        # Course fields fall back to defaults
        assert ctx["enrolled_course"] == ""

    @pytest.mark.asyncio
    async def test_multi_enrollment_totals_are_summed(self) -> None:
        """total_completed_lessons and total_watched_hours are summed across enrollments."""
        from aegra_api.services.scheduler import SchedulerService

        svc = SchedulerService()
        session = self._make_session()

        multi_enrollment = {
            "enrollments": [
                {
                    "course": {"title": "Course A"},
                    "overallProgress": 80,
                    "totalCompletedLessons": 10,
                    "totalWatchedHours": 5.0,
                    "updatedAt": datetime(2024, 5, 1, tzinfo=UTC),
                },
                {
                    "course": {"title": "Course B"},
                    "overallProgress": 20,
                    "totalCompletedLessons": 3,
                    "totalWatchedHours": 2.5,
                    "updatedAt": datetime(2024, 6, 10, tzinfo=UTC),
                },
            ]
        }

        with (
            patch("aegra_api.services.scheduler.get_course_content_mongo_client") as mock_client_factory,
        ):
            mongo_client = MagicMock()
            mongo_client.get_user_onboarding_data.return_value = None
            mongo_client.get_enrollment_overview.return_value = multi_enrollment
            mock_client_factory.return_value = mongo_client

            ctx = await svc._build_student_context(session, "user-multi", "Marcus", "data-science")

        assert ctx["total_completed_lessons"] == 13  # 10 + 3
        assert ctx["total_watched_hours"] == 7.5  # 5.0 + 2.5
        assert ctx["enrolled_course"] == "Course A"  # first enrollment = primary
        assert ctx["last_active_in_course"] == "Jun 10"  # most recent updatedAt


# ---------------------------------------------------------------------------
# generate_personalized_motivational_content — new fields reach LLM
# ---------------------------------------------------------------------------


class TestPersonalizedContentFields:
    """Verify the LLM prompt includes course progress data when available."""

    @pytest.mark.asyncio
    async def test_course_fields_included_in_prompt(self) -> None:
        from aegra_api.services.notification_engine import NotificationEngine

        engine = NotificationEngine()

        student_ctx = {
            "first_name": "James",
            "learning_track": "ai-engineering",
            "current_streak": 12,
            "tasks_completed_this_week": 3,
            "overdue_tasks": 1,
            "pending_tasks": 4,
            "primary_goal": "AI Engineer",
            "enrolled_course": "AI Engineering Bootcamp",
            "course_progress_pct": 43,
            "total_completed_lessons": 17,
            "total_watched_hours": 12.5,
            "last_active_in_course": "Jun 10",
        }

        captured_messages: list[Any] = []

        class FakeLLM:
            async def ainvoke(self, messages):
                captured_messages.extend(messages)
                resp = MagicMock()
                resp.content = "Great progress, keep going!"
                return resp

        # ChatBedrockConverse is imported lazily inside the try block via
        # `from langchain_aws import ChatBedrockConverse`, so patch the source module.
        fake_llm_instance = FakeLLM()
        fake_langchain_aws = MagicMock()
        fake_langchain_aws.ChatBedrockConverse.return_value = fake_llm_instance

        with (
            patch.dict("sys.modules", {"langchain_aws": fake_langchain_aws}),
            patch("aegra_api.services.notification_engine.settings") as mock_settings,
        ):
            mock_settings.aws.AWS_REGION_NAME = "eu-west-1"
            title, body = await engine.generate_personalized_motivational_content(
                persona_name="David",
                student_context=student_ctx,
            )

        # Collect all text fed to the LLM
        all_text = " ".join(str(m.content) for m in captured_messages)

        assert "AI Engineering Bootcamp" in all_text
        assert "43" in all_text  # progress pct
        assert "17" in all_text  # lessons
        assert "12.5" in all_text  # hours watched
        assert "Jun 10" in all_text  # last active date
        assert "AI Engineer" in all_text  # career goal

    @pytest.mark.asyncio
    async def test_fallback_includes_course_progress(self) -> None:
        """When the LLM call fails the fallback text must reference course progress."""
        from aegra_api.services.notification_engine import NotificationEngine

        engine = NotificationEngine()

        student_ctx = {
            "first_name": "Grace",
            "learning_track": "data-science",
            "current_streak": 0,
            "tasks_completed_this_week": 5,
            "overdue_tasks": 0,
            "pending_tasks": 2,
            "primary_goal": "Data Scientist",
            "enrolled_course": "Data Science Pro",
            "course_progress_pct": 60,
            "total_completed_lessons": 24,
            "total_watched_hours": 18.0,
            "last_active_in_course": "May 20",
        }

        # Force LLM import to fail so the fallback is exercised
        with patch.dict("sys.modules", {"langchain_aws": None}):
            title, body = await engine.generate_personalized_motivational_content(
                persona_name="Marcus",
                student_context=student_ctx,
            )

        assert "60%" in body
        assert "Data Science Pro" in body
        assert "24" in body
        assert "18.0" in body
        assert "Data Scientist" in body

    @pytest.mark.asyncio
    async def test_title_format(self) -> None:
        """Title is always '<DayName> update from <PersonaName>'."""
        from aegra_api.services.notification_engine import NotificationEngine

        engine = NotificationEngine()
        ctx = {
            "first_name": "Sam",
            "learning_track": "data-analytics",
            "current_streak": 3,
            "tasks_completed_this_week": 1,
            "overdue_tasks": 0,
            "pending_tasks": 0,
            "primary_goal": "",
            "enrolled_course": "",
            "course_progress_pct": 0,
            "total_completed_lessons": 0,
            "total_watched_hours": 0,
            "last_active_in_course": "",
        }
        with patch.dict("sys.modules", {"langchain_aws": None}):
            title, _ = await engine.generate_personalized_motivational_content(
                persona_name="Alexandra",
                student_context=ctx,
            )

        day = datetime.now(UTC).strftime("%A")
        assert title == f"{day} update from Alexandra"


# ---------------------------------------------------------------------------
# Advisor resolution — track format independence
# ---------------------------------------------------------------------------


class TestAdvisorResolution:
    """_resolve_advisor_for_user must pick the right advisor regardless of
    how the track is formatted in MongoDB."""

    @pytest.mark.asyncio
    async def test_spaced_track_maps_to_david(self) -> None:
        from aegra_api.services.scheduler import SchedulerService

        with patch("aegra_api.services.scheduler.get_course_content_mongo_client") as mock_factory:
            client = MagicMock()
            client.get_learning_track.return_value = "AI Engineering"
            mock_factory.return_value = client

            name, track = await SchedulerService._resolve_advisor_for_user("user-1")

        assert name == "David"
        assert track == "ai-engineering"

    @pytest.mark.asyncio
    async def test_kebab_track_maps_to_david(self) -> None:
        from aegra_api.services.scheduler import SchedulerService

        with patch("aegra_api.services.scheduler.get_course_content_mongo_client") as mock_factory:
            client = MagicMock()
            client.get_learning_track.return_value = "ai-engineering"
            mock_factory.return_value = client

            name, track = await SchedulerService._resolve_advisor_for_user("user-2")

        assert name == "David"
        assert track == "ai-engineering"

    @pytest.mark.asyncio
    async def test_unknown_track_falls_back_to_default(self) -> None:
        from aegra_api.services.scheduler import SchedulerService

        with patch("aegra_api.services.scheduler.get_course_content_mongo_client") as mock_factory:
            client = MagicMock()
            client.get_learning_track.return_value = "some-unknown-track"
            mock_factory.return_value = client

            name, track = await SchedulerService._resolve_advisor_for_user("user-3")

        # Default advisor is Alexandra
        assert name == "Alexandra"

    @pytest.mark.asyncio
    async def test_mongo_error_falls_back_to_default(self) -> None:
        from aegra_api.services.scheduler import SchedulerService

        with patch("aegra_api.services.scheduler.get_course_content_mongo_client") as mock_factory:
            client = MagicMock()
            client.get_learning_track.side_effect = ConnectionError("unreachable")
            mock_factory.return_value = client

            name, track = await SchedulerService._resolve_advisor_for_user("user-4")

        assert name == "Alexandra"
        assert track == ""

"""Regression test: report_progress must resolve and pass the student's
actual advisor persona to create_notification (same class of bug as
check_celebrations — see test_services/test_celebration_persona.py).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aegra_api.api.accountability import ProgressEventRequest, report_progress


class TestReportProgressPersona:
    @pytest.mark.asyncio
    async def test_course_completed_passes_resolved_advisor_as_persona(self) -> None:
        session = AsyncMock()
        user = MagicMock(identity="user-1", display_name="Kudus", email="kudus@example.com")
        body = ProgressEventRequest(event_type="course_completed", course_name="AI Bootcamp")

        with (
            patch("aegra_api.services.accountability_service.AccountabilityService.record_activity", AsyncMock()),
            patch(
                "aegra_api.services.scheduler.SchedulerService._resolve_advisor_first_name",
                AsyncMock(return_value="Priya"),
            ) as mock_resolve,
            patch("aegra_api.services.notification_engine.notification_engine.create_notification") as mock_create,
        ):
            mock_create.return_value = AsyncMock()
            await report_progress(body, session=session, user=user)

        mock_resolve.assert_awaited_once_with("user-1")
        _, kwargs = mock_create.call_args
        assert kwargs["persona"] == "Priya"

    @pytest.mark.asyncio
    async def test_lesson_completed_skips_advisor_lookup(self) -> None:
        """Event types that never send a notification shouldn't pay for the Mongo lookup."""
        session = AsyncMock()
        user = MagicMock(identity="user-1", display_name="Kudus", email="kudus@example.com")
        body = ProgressEventRequest(event_type="lesson_completed")

        with (
            patch("aegra_api.services.accountability_service.AccountabilityService.record_activity", AsyncMock()),
            patch(
                "aegra_api.services.scheduler.SchedulerService._resolve_advisor_first_name",
                AsyncMock(return_value="Priya"),
            ) as mock_resolve,
        ):
            await report_progress(body, session=session, user=user)

        mock_resolve.assert_not_awaited()

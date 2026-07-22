"""Regression test: check_celebrations must resolve and pass the student's
actual advisor persona to create_notification.

Before this fix, check_celebrations passed neither `persona` nor
`advisor_persona`, so create_notification's own fallback chain
(`advisor_persona or persona or DEFAULT_PERSONA`) silently signed every
celebration email as "Alexandra" regardless of the student's actual track —
the same persona-mismatch bug the spec (Item 7) calls out.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestCheckCelebrationsPersona:
    @pytest.mark.asyncio
    async def test_passes_resolved_advisor_as_persona(self) -> None:
        from aegra_api.services.scheduler import SchedulerService

        svc = SchedulerService()

        session = AsyncMock()
        session_cm = MagicMock()
        session_cm.__aenter__ = AsyncMock(return_value=session)
        session_cm.__aexit__ = AsyncMock(return_value=False)

        user_ids_result = MagicMock()
        user_ids_result.scalars.return_value.all.return_value = ["user-1"]

        dedup_result = MagicMock()
        dedup_result.scalars.return_value.first.return_value = None

        session.execute = AsyncMock(side_effect=[user_ids_result, dedup_result])

        celebration = {"type": "course_completed", "title": "Nice work!", "content": "You did it."}

        with (
            patch("aegra_api.services.scheduler.db_manager") as mock_db_manager,
            patch("aegra_api.services.scheduler.async_sessionmaker", return_value=lambda: session_cm),
            patch("aegra_api.services.scheduler.notification_engine") as mock_engine,
            patch.object(
                SchedulerService, "_resolve_advisor_first_name", AsyncMock(return_value="Marcus")
            ) as mock_resolve,
        ):
            mock_db_manager.engine = MagicMock()
            mock_engine.check_celebrations = AsyncMock(return_value=[celebration])
            mock_engine.create_notification = AsyncMock(return_value=MagicMock())

            await svc.check_celebrations()

        mock_resolve.assert_awaited_once_with("user-1")
        _, kwargs = mock_engine.create_notification.call_args
        assert kwargs["persona"] == "Marcus"

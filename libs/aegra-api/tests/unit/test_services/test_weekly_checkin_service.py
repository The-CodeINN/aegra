"""Unit tests for WeeklyCheckinService.

All external dependencies (DB session, CronService, LangGraph service,
SchedulerService advisor resolution) are mocked.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from aegra_api.services.weekly_checkin_service import (
    _CHECKIN_GRAPH_ID,
    _CHECKIN_SCHEDULE,
    _PREF_CRON_KEY,
    _PREF_THREAD_KEY,
    WeeklyCheckinService,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ADVISOR: dict[str, Any] = {
    "name": "Alexandra Chen",
    "title": "Data Analytics Advisor",
    "experience": "8 years",
    "personality": "methodical",
    "background": "BI & data strategy",
    "communication_style": "structured",
    "expertise_areas": ["sql", "tableau"],
}

_USER_ID = "user-abc-123"


def _make_session(*, pref: Any = None) -> AsyncMock:
    """Build a minimal AsyncSession mock."""
    session = AsyncMock()
    session.get = AsyncMock(return_value=pref)
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.execute = AsyncMock()
    return session


def _make_cron_orm(cron_id: str = "cron-xyz") -> Mock:
    cron = Mock()
    cron.cron_id = cron_id
    return cron


def _make_pref(preferences: dict[str, Any] | None = None) -> Mock:
    pref = Mock()
    pref.preferences = preferences or {}
    return pref


# ---------------------------------------------------------------------------
# provision_user
# ---------------------------------------------------------------------------


class TestProvisionUser:
    """Tests for WeeklyCheckinService.provision_user."""

    @pytest.mark.asyncio
    async def test_creates_thread_and_cron_for_new_user(self) -> None:
        """A user with no preferences gets a fresh thread and cron."""
        session = _make_session(pref=None)
        cron_orm = _make_cron_orm("cron-001")
        svc = WeeklyCheckinService()

        with (
            patch(
                "aegra_api.services.weekly_checkin_service.SchedulerService._resolve_advisor_for_user",
                new=AsyncMock(return_value=("Alexandra", "data-analytics")),
            ),
            patch(
                "aegra_api.services.weekly_checkin_service.get_advisor_by_track",
                return_value=_ADVISOR,
            ),
            patch(
                "aegra_api.services.weekly_checkin_service.CronService.create_cron",
                new=AsyncMock(return_value=cron_orm),
            ),
        ):
            result = await svc.provision_user(
                _USER_ID,
                session=session,
                langgraph_service=MagicMock(),
            )

        assert result == "cron-001"
        # Thread row added
        session.add.assert_called()
        session.flush.assert_awaited_once()
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_skips_already_provisioned_user(self) -> None:
        """A user whose preferences already contain the cron key is skipped."""
        pref = _make_pref({_PREF_CRON_KEY: "existing-cron"})
        session = _make_session(pref=pref)
        svc = WeeklyCheckinService()

        result = await svc.provision_user(
            _USER_ID,
            session=session,
            langgraph_service=MagicMock(),
        )

        assert result is None
        session.flush.assert_not_awaited()
        session.commit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_falls_back_to_default_advisor_when_track_unknown(self) -> None:
        """When get_advisor_by_track returns None the default advisor is used."""
        session = _make_session(pref=None)
        cron_orm = _make_cron_orm("cron-002")
        svc = WeeklyCheckinService()
        default_advisor = {"name": "Alexandra Chen", "title": "Career Advisor"}

        with (
            patch(
                "aegra_api.services.weekly_checkin_service.SchedulerService._resolve_advisor_for_user",
                new=AsyncMock(return_value=("Alexandra", "")),
            ),
            patch(
                "aegra_api.services.weekly_checkin_service.get_advisor_by_track",
                return_value=None,
            ),
            patch(
                "aegra_api.services.weekly_checkin_service.get_default_advisor",
                return_value=default_advisor,
            ),
            patch(
                "aegra_api.services.weekly_checkin_service.CronService.create_cron",
                new=AsyncMock(return_value=cron_orm),
            ),
        ):
            result = await svc.provision_user(
                _USER_ID,
                session=session,
                langgraph_service=MagicMock(),
            )

        assert result == "cron-002"

    @pytest.mark.asyncio
    async def test_rolls_back_when_cron_creation_fails(self) -> None:
        """Session is rolled back if CronService.create_cron raises."""
        session = _make_session(pref=None)
        svc = WeeklyCheckinService()

        with (
            patch(
                "aegra_api.services.weekly_checkin_service.SchedulerService._resolve_advisor_for_user",
                new=AsyncMock(return_value=("Alexandra", "data-analytics")),
            ),
            patch(
                "aegra_api.services.weekly_checkin_service.get_advisor_by_track",
                return_value=_ADVISOR,
            ),
            patch(
                "aegra_api.services.weekly_checkin_service.CronService.create_cron",
                new=AsyncMock(side_effect=RuntimeError("assistant not found")),
            ),
            pytest.raises(RuntimeError, match="assistant not found"),
        ):
            await svc.provision_user(
                _USER_ID,
                session=session,
                langgraph_service=MagicMock(),
            )

        session.rollback.assert_awaited_once()
        session.commit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cron_uses_correct_schedule_and_graph(self) -> None:
        """Cron is created with the Monday-09:00 schedule and the agent graph."""
        session = _make_session(pref=None)
        cron_orm = _make_cron_orm("cron-003")
        captured: list[Any] = []
        svc = WeeklyCheckinService()

        async def _capture_create(self: Any, request: Any, *, user_identity: str, thread_id: str | None = None) -> Any:
            captured.append((request, user_identity, thread_id))
            return cron_orm

        with (
            patch(
                "aegra_api.services.weekly_checkin_service.SchedulerService._resolve_advisor_for_user",
                new=AsyncMock(return_value=("Alexandra", "data-analytics")),
            ),
            patch(
                "aegra_api.services.weekly_checkin_service.get_advisor_by_track",
                return_value=_ADVISOR,
            ),
            patch(
                "aegra_api.services.weekly_checkin_service.CronService.create_cron",
                new=_capture_create,
            ),
        ):
            await svc.provision_user(
                _USER_ID,
                session=session,
                langgraph_service=MagicMock(),
            )

        assert len(captured) == 1
        req, uid, thread_id = captured[0]
        assert req.schedule == _CHECKIN_SCHEDULE
        assert req.assistant_id == _CHECKIN_GRAPH_ID
        assert uid == _USER_ID
        assert thread_id is not None

    @pytest.mark.asyncio
    async def test_stores_ids_in_preferences(self) -> None:
        """Cron ID and thread ID end up in UserPreferences.preferences."""
        session = _make_session(pref=None)
        cron_orm = _make_cron_orm("cron-stored")
        svc = WeeklyCheckinService()
        stored_pref: list[Any] = []

        def _capture_add(obj: Any) -> None:
            if hasattr(obj, "preferences"):
                stored_pref.append(obj)

        session.add.side_effect = _capture_add

        with (
            patch(
                "aegra_api.services.weekly_checkin_service.SchedulerService._resolve_advisor_for_user",
                new=AsyncMock(return_value=("Alexandra", "data-analytics")),
            ),
            patch(
                "aegra_api.services.weekly_checkin_service.get_advisor_by_track",
                return_value=_ADVISOR,
            ),
            patch(
                "aegra_api.services.weekly_checkin_service.CronService.create_cron",
                new=AsyncMock(return_value=cron_orm),
            ),
        ):
            await svc.provision_user(
                _USER_ID,
                session=session,
                langgraph_service=MagicMock(),
            )

        assert len(stored_pref) == 1
        prefs = stored_pref[0].preferences
        assert _PREF_CRON_KEY in prefs
        assert prefs[_PREF_CRON_KEY] == "cron-stored"
        assert _PREF_THREAD_KEY in prefs


# ---------------------------------------------------------------------------
# deprovision_user
# ---------------------------------------------------------------------------


class TestDeprovisionUser:
    """Tests for WeeklyCheckinService.deprovision_user."""

    @pytest.mark.asyncio
    async def test_disables_cron_when_present(self) -> None:
        """An existing cron is disabled via an UPDATE statement."""
        pref = _make_pref({_PREF_CRON_KEY: "cron-to-disable"})
        session = _make_session(pref=pref)
        svc = WeeklyCheckinService()

        await svc.deprovision_user(_USER_ID, session=session)

        session.execute.assert_awaited_once()
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_op_when_no_preferences(self) -> None:
        """Nothing happens when the user has no preferences row."""
        session = _make_session(pref=None)
        svc = WeeklyCheckinService()

        await svc.deprovision_user(_USER_ID, session=session)

        session.execute.assert_not_awaited()
        session.commit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_op_when_no_cron_key(self) -> None:
        """Nothing happens when preferences exist but have no cron key."""
        pref = _make_pref({"some_other_key": "value"})
        session = _make_session(pref=pref)
        svc = WeeklyCheckinService()

        await svc.deprovision_user(_USER_ID, session=session)

        session.execute.assert_not_awaited()
        session.commit.assert_not_awaited()


# ---------------------------------------------------------------------------
# provision_all_pending
# ---------------------------------------------------------------------------


class TestProvisionAllPending:
    """Tests for WeeklyCheckinService.provision_all_pending."""

    @pytest.mark.asyncio
    async def test_provisions_users_without_checkin(self) -> None:
        """Users returned by the discovery query are each provisioned."""
        session = _make_session()
        scalars_mock = Mock()
        scalars_mock.all.return_value = ["user-1", "user-2"]
        execute_result = Mock()
        execute_result.scalars.return_value = scalars_mock
        session.execute = AsyncMock(return_value=execute_result)

        svc = WeeklyCheckinService()

        with (
            patch.object(
                svc,
                "provision_user",
                new=AsyncMock(return_value="new-cron"),
            ) as mock_provision,
            patch(
                "aegra_api.services.weekly_checkin_service.get_session_maker",
                return_value=MagicMock(
                    return_value=AsyncMock(
                        __aenter__=AsyncMock(return_value=session), __aexit__=AsyncMock(return_value=False)
                    )
                ),
            ),
        ):
            await svc.provision_all_pending(
                session=session,
                langgraph_service=MagicMock(),
            )

        assert mock_provision.await_count == 2

    @pytest.mark.asyncio
    async def test_no_op_when_all_provisioned(self) -> None:
        """No provision calls when the query returns an empty list."""
        session = _make_session()
        scalars_mock = Mock()
        scalars_mock.all.return_value = []
        execute_result = Mock()
        execute_result.scalars.return_value = scalars_mock
        session.execute = AsyncMock(return_value=execute_result)

        svc = WeeklyCheckinService()

        with patch.object(svc, "provision_user", new=AsyncMock()) as mock_provision:
            await svc.provision_all_pending(
                session=session,
                langgraph_service=MagicMock(),
            )

        mock_provision.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_failure_for_one_user_does_not_abort_others(self) -> None:
        """A provision failure for one user is logged and others continue."""
        session = _make_session()
        scalars_mock = Mock()
        scalars_mock.all.return_value = ["user-ok", "user-bad"]
        execute_result = Mock()
        execute_result.scalars.return_value = scalars_mock
        session.execute = AsyncMock(return_value=execute_result)

        svc = WeeklyCheckinService()
        provisioned: list[str] = []

        async def _side_effect(uid: str, *, session: Any, langgraph_service: Any) -> str | None:
            if uid == "user-bad":
                raise RuntimeError("boom")
            provisioned.append(uid)
            return "cron-x"

        with (
            patch.object(svc, "provision_user", new=_side_effect),
            patch(
                "aegra_api.services.weekly_checkin_service.get_session_maker",
                return_value=MagicMock(
                    return_value=AsyncMock(
                        __aenter__=AsyncMock(return_value=session), __aexit__=AsyncMock(return_value=False)
                    )
                ),
            ),
        ):
            await svc.provision_all_pending(
                session=session,
                langgraph_service=MagicMock(),
            )

        assert "user-ok" in provisioned

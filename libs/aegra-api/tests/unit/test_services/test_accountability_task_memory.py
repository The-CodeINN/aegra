"""Unit tests for the TaskMemory additions to AccountabilityService (agent optimisation spec Item 2).

Covers the write path (create_action_item), the read path used by
conversation-start injection (get_open_and_overdue), the miss-count
escalation counter (mark_missed), and evidence capture on completion.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from aegra_api.services.accountability_service import AccountabilityService, OpenTaskGroup


def _make_session_returning(scalar_result: object) -> AsyncMock:
    session = AsyncMock()
    session.scalar = AsyncMock(return_value=scalar_result)
    return session


class TestCreateActionItem:
    @pytest.mark.asyncio
    async def test_creates_item_with_given_fields(self) -> None:
        session = AsyncMock()

        item = await AccountabilityService.create_action_item(
            session,
            "user-1",
            "Rebuild GitHub README with 3 pinned projects",
            thread_id="thread-1",
            due_date=datetime(2026, 7, 10, tzinfo=UTC),
            priority="high",
            category="Portfolio",
            advisor_persona="David",
        )

        session.add.assert_called_once()
        session.commit.assert_awaited_once()
        assert item.user_id == "user-1"
        assert item.description == "Rebuild GitHub README with 3 pinned projects"
        assert item.priority == "high"
        assert item.thread_id == "thread-1"
        assert item.source == "conversation"

    @pytest.mark.asyncio
    async def test_defaults_priority_to_normal(self) -> None:
        session = AsyncMock()
        item = await AccountabilityService.create_action_item(session, "user-1", "Apply to 2 roles")
        assert item.priority == "normal"


class TestGetOpenAndOverdue:
    def _make_item(self, *, status: str, due_date: datetime | None, miss_count: int = 0) -> MagicMock:
        item = MagicMock()
        item.status = status
        item.due_date = due_date
        item.miss_count = miss_count
        return item

    def _make_session_with_items(self, items: list) -> AsyncMock:
        session = AsyncMock()
        result = MagicMock()
        result.scalars.return_value.all.return_value = items
        session.execute = AsyncMock(return_value=result)
        return session

    @pytest.mark.asyncio
    async def test_splits_overdue_from_not_yet_due(self) -> None:
        now = datetime(2026, 7, 3, 12, 0, 0, tzinfo=UTC)
        overdue_item = self._make_item(status="pending", due_date=now - timedelta(days=5))
        future_item = self._make_item(status="pending", due_date=now + timedelta(days=5))
        session = self._make_session_with_items([overdue_item, future_item])

        group = await AccountabilityService.get_open_and_overdue(session, "user-1")

        assert group.overdue == [overdue_item]
        assert group.not_yet_due == [future_item]

    @pytest.mark.asyncio
    async def test_task_with_no_due_date_is_not_overdue(self) -> None:
        item = self._make_item(status="in_progress", due_date=None)
        session = self._make_session_with_items([item])

        group = await AccountabilityService.get_open_and_overdue(session, "user-1")

        assert group.overdue == []
        assert group.not_yet_due == [item]

    @pytest.mark.asyncio
    async def test_timezone_naive_due_date_treated_as_utc(self) -> None:
        """DB-round-tripped datetimes can come back naive; must not crash comparing to aware `now`."""
        naive_past_due = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=1)
        item = self._make_item(status="pending", due_date=naive_past_due)
        session = self._make_session_with_items([item])

        group = await AccountabilityService.get_open_and_overdue(session, "user-1")

        assert group.overdue == [item]

    @pytest.mark.asyncio
    async def test_empty_group_is_falsy(self) -> None:
        session = self._make_session_with_items([])
        group = await AccountabilityService.get_open_and_overdue(session, "user-1")
        assert not group
        assert isinstance(group, OpenTaskGroup)


class TestMarkMissed:
    @pytest.mark.asyncio
    async def test_increments_miss_count(self) -> None:
        item = MagicMock()
        item.miss_count = 1
        session = _make_session_returning(item)

        updated = await AccountabilityService.mark_missed(session, "item-1", "user-1")

        assert updated.miss_count == 2
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_raises_when_item_not_found(self) -> None:
        session = _make_session_returning(None)
        with pytest.raises(ValueError, match="Item not found"):
            await AccountabilityService.mark_missed(session, "missing", "user-1")


class TestUpdateActionItemStatusWithEvidence:
    @pytest.mark.asyncio
    async def test_sets_evidence_when_provided(self) -> None:
        item = MagicMock()
        item.status = "pending"
        session = _make_session_returning(item)

        # Marking "completed" also triggers _record_action_completion, which
        # runs its own session.execute(...).scalar_one_or_none() lookup. Use
        # an existing activity row (not None) so the ORM-default-less
        # in-memory construction path (longest_streak=None pre-flush) isn't
        # exercised — that's an unrelated pre-existing edge case, not
        # something this test is about.
        activity = MagicMock()
        activity.current_streak = 2
        activity.longest_streak = 5
        activity.last_streak_date = None
        activity_result = MagicMock()
        activity_result.scalar_one_or_none.return_value = activity
        session.execute = AsyncMock(return_value=activity_result)

        await AccountabilityService.update_action_item_status(
            session, "item-1", "user-1", "completed", evidence="github.com/user/repo"
        )

        assert item.evidence == "github.com/user/repo"
        assert item.status == "completed"

    @pytest.mark.asyncio
    async def test_no_op_status_change_still_persists_evidence(self) -> None:
        """Evidence can be added after the fact without changing status."""
        item = MagicMock()
        item.status = "completed"
        session = _make_session_returning(item)

        result = await AccountabilityService.update_action_item_status(
            session, "item-1", "user-1", "completed", evidence="submission-42"
        )

        assert item.evidence == "submission-42"
        assert result["message"] == "no_change"
        session.commit.assert_awaited_once()

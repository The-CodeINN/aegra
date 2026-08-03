"""Regression tests for SchedulerService._should_send_reminder.

check_deadlines runs every 15 minutes. Before this fix, the same tier (e.g.
"7d" — due this week) kept resending an identical email every ~4h for as
long as the task sat in that tier, because the gate only checked elapsed
time, never whether the tier itself had already been notified. Confirmed on
prod: one task received the identical "Due This Week" email 3 times in a
single day; another received "Coming Up in 3 Days" 4 times over 2 days
before ever reaching the next tier.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from aegra_api.services.scheduler import SchedulerService


def _item(
    *, reminder_sent_count: int, last_reminder_tier: str | None, last_reminder_sent: datetime | None
) -> MagicMock:
    item = MagicMock()
    item.reminder_sent_count = reminder_sent_count
    item.last_reminder_tier = last_reminder_tier
    item.last_reminder_sent = last_reminder_sent
    return item


class TestShouldSendReminder:
    @pytest.mark.asyncio
    async def test_first_reminder_always_sends(self) -> None:
        svc = SchedulerService()
        item = _item(reminder_sent_count=0, last_reminder_tier=None, last_reminder_sent=None)

        assert await svc._should_send_reminder(item, "7d", datetime.now(UTC)) is True

    @pytest.mark.asyncio
    async def test_same_non_overdue_tier_never_resends(self) -> None:
        """The exact bug from prod: '7d' resent every ~4h for as long as the task sat in it."""
        svc = SchedulerService()
        now = datetime.now(UTC)
        item = _item(reminder_sent_count=1, last_reminder_tier="7d", last_reminder_sent=now - timedelta(hours=4))

        assert await svc._should_send_reminder(item, "7d", now) is False

    @pytest.mark.asyncio
    async def test_same_non_overdue_tier_never_resends_even_after_a_week(self) -> None:
        """Not just a longer cooldown — a non-overdue tier must never repeat, full stop."""
        svc = SchedulerService()
        now = datetime.now(UTC)
        item = _item(reminder_sent_count=1, last_reminder_tier="3d", last_reminder_sent=now - timedelta(days=7))

        assert await svc._should_send_reminder(item, "3d", now) is False

    @pytest.mark.asyncio
    async def test_tier_escalation_sends_a_fresh_reminder(self) -> None:
        svc = SchedulerService()
        now = datetime.now(UTC)
        item = _item(reminder_sent_count=1, last_reminder_tier="3d", last_reminder_sent=now - timedelta(hours=1))

        assert await svc._should_send_reminder(item, "24h", now) is True

    @pytest.mark.asyncio
    async def test_overdue_tier_does_not_resend_within_24h(self) -> None:
        svc = SchedulerService()
        now = datetime.now(UTC)
        item = _item(reminder_sent_count=2, last_reminder_tier="overdue", last_reminder_sent=now - timedelta(hours=5))

        assert await svc._should_send_reminder(item, "overdue", now) is False

    @pytest.mark.asyncio
    async def test_overdue_tier_resends_after_24h(self) -> None:
        svc = SchedulerService()
        now = datetime.now(UTC)
        item = _item(reminder_sent_count=2, last_reminder_tier="overdue", last_reminder_sent=now - timedelta(hours=25))

        assert await svc._should_send_reminder(item, "overdue", now) is True

    @pytest.mark.asyncio
    async def test_overdue_tier_stops_after_five_sends(self) -> None:
        svc = SchedulerService()
        now = datetime.now(UTC)
        item = _item(reminder_sent_count=5, last_reminder_tier="overdue", last_reminder_sent=now - timedelta(hours=48))

        assert await svc._should_send_reminder(item, "overdue", now) is False

    @pytest.mark.asyncio
    async def test_overdue_severe_tier_follows_the_same_overdue_rules(self) -> None:
        svc = SchedulerService()
        now = datetime.now(UTC)
        item = _item(
            reminder_sent_count=1, last_reminder_tier="overdue_severe", last_reminder_sent=now - timedelta(hours=25)
        )

        assert await svc._should_send_reminder(item, "overdue_severe", now) is True

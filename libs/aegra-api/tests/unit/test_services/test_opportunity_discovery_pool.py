"""Regression test: discover_for_user must not hold a pooled DB connection
across slow external API/LLM work.

Confirmed on prod: with _DISCOVERY_CONCURRENCY=5 concurrent users each
holding a session open through several seconds of event/job-board search and
strategy generation, the SQLAlchemy pool (sized far smaller than that
concurrency) hit "QueuePool ... timeout" 174 times in 48h, silently dropping
opportunity discovery for most users hit by it.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aegra_api.services.opportunity_discovery import OpportunityDiscoveryEngine
from aegra_api.services.student_profile import StudentProfile


def _make_session() -> MagicMock:
    session = MagicMock()
    execute_result = MagicMock()
    execute_result.all.return_value = []
    session.execute = AsyncMock(return_value=execute_result)
    session.close = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    return session


@pytest.mark.asyncio
async def test_session_is_closed_before_slow_external_discovery_work(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = OpportunityDiscoveryEngine()
    session = _make_session()
    call_order: list[str] = []

    async def _fake_get_student_profile(*_args: object, **_kwargs: object) -> StudentProfile:
        return StudentProfile(user_id="user-1")

    async def _fake_get_track_from_prefs(*_args: object, **_kwargs: object) -> str:
        return "ai-engineering"

    async def _fake_discover_events(*_args: object, **_kwargs: object) -> list[dict]:
        call_order.append("discover_events")
        assert session.close.await_count == 1, "session must be closed before slow external work starts"
        return []

    async def _fake_discover_jobs(*_args: object, **_kwargs: object) -> list[dict]:
        call_order.append("discover_jobs")
        assert session.close.await_count == 1, "session must be closed before slow external work starts"
        return []

    async def _fake_generate_strategies_batch(*_args: object, **_kwargs: object) -> list[dict]:
        call_order.append("generate_strategies")
        return []

    monkeypatch.setattr(engine, "_get_student_profile", _fake_get_student_profile)
    monkeypatch.setattr(engine, "_get_track_from_prefs", _fake_get_track_from_prefs)
    monkeypatch.setattr(engine, "_discover_events", _fake_discover_events)
    monkeypatch.setattr(engine, "_discover_jobs", _fake_discover_jobs)
    monkeypatch.setattr(engine, "_generate_strategies_batch", _fake_generate_strategies_batch)

    discovered = await engine.discover_for_user(
        session=session,
        user_id="user-1",
        auth_token="",
        max_tracks=1,
        queries_per_category=1,
    )

    assert discovered == []
    assert call_order == ["discover_events", "discover_jobs", "generate_strategies"]
    session.close.assert_awaited_once()
    # The write phase (add/commit) must still happen after — proves the
    # session remains usable post-close, not merely closed-and-abandoned.
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_tracks_resolved_returns_empty_without_slow_work(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard: the early-return-on-no-tracks path shouldn't call the (now-closed) session again unexpectedly."""
    engine = OpportunityDiscoveryEngine()
    session = _make_session()

    async def _fake_get_student_profile(*_args: object, **_kwargs: object) -> StudentProfile:
        return StudentProfile(user_id="user-1")

    async def _fake_get_track_from_prefs(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(engine, "_get_student_profile", _fake_get_student_profile)
    monkeypatch.setattr(engine, "_get_track_from_prefs", _fake_get_track_from_prefs)

    discovered = await engine.discover_for_user(session=session, user_id="user-1", auth_token="", max_tracks=1)

    assert discovered == []
    session.close.assert_not_awaited()

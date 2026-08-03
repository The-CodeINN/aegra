"""Unit tests for the shared advisor-context assembly module.

This module is what makes spec Item 7 (§9.1) a real structural unification
rather than a same-data-different-code coincidence: call_model (live chat)
and aegra_api's outreach path both call these exact functions.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GRAPHS_ROOT = PROJECT_ROOT / "graphs"
if str(GRAPHS_ROOT) not in sys.path:
    sys.path.insert(0, str(GRAPHS_ROOT))

from react_agent.context_assembly import (  # noqa: E402
    fetch_advisor_behavior_profile,
    fetch_semantic_memory_block,
)


def _store_item(kind: str, content: dict, updated_at: str = "2026-07-01T00:00:00") -> SimpleNamespace:
    return SimpleNamespace(value={"kind": kind, "content": content}, updated_at=updated_at, created_at=updated_at)


class TestFetchAdvisorBehaviorProfile:
    @pytest.mark.asyncio
    async def test_returns_none_without_store_or_user_id(self) -> None:
        assert await fetch_advisor_behavior_profile(None, "user-1") is None
        assert await fetch_advisor_behavior_profile(MagicMock(), None) is None

    @pytest.mark.asyncio
    async def test_returns_most_recently_updated_profile_content(self) -> None:
        old = _store_item("AdvisorBehaviorProfile", {"trait": "gentle"}, updated_at="2026-05-01T00:00:00")
        new = _store_item("AdvisorBehaviorProfile", {"trait": "direct challenge"}, updated_at="2026-07-01T00:00:00")
        store = MagicMock()
        store.asearch = AsyncMock(return_value=[old, new])

        result = await fetch_advisor_behavior_profile(store, "user-1")

        assert result == {"trait": "direct challenge"}

    @pytest.mark.asyncio
    async def test_returns_none_on_store_error(self) -> None:
        store = MagicMock()
        store.asearch = AsyncMock(side_effect=ConnectionError("pg down"))

        assert await fetch_advisor_behavior_profile(store, "user-1") is None


class TestFetchSemanticMemoryBlock:
    @pytest.mark.asyncio
    async def test_returns_empty_without_store_or_user_id(self) -> None:
        assert await fetch_semantic_memory_block(None, "user-1") == ("", "")
        assert await fetch_semantic_memory_block(MagicMock(), None) == ("", "")

    @pytest.mark.asyncio
    async def test_excludes_advisor_behavior_profile_kind(self) -> None:
        """Rendered separately via fetch_advisor_behavior_profile — never duplicated here."""
        goal = _store_item("CareerGoal", {"role": "Data Scientist"})
        behavior = _store_item("AdvisorBehaviorProfile", {"trait": "direct challenge"})
        store = MagicMock()
        store.asearch = AsyncMock(return_value=[goal, behavior])

        _, clean_block = await fetch_semantic_memory_block(store, "user-1")

        assert "Data Scientist" in clean_block
        assert "direct challenge" not in clean_block

    @pytest.mark.asyncio
    async def test_never_includes_a_name_field(self) -> None:
        """Regression guard: identity must never come from stored memory content."""
        goal = _store_item("StudentContext", {"fact": "3 years Python experience"})
        store = MagicMock()
        store.asearch = AsyncMock(return_value=[goal])

        raw_text, clean_block = await fetch_semantic_memory_block(store, "user-1")

        assert "3 years Python" in raw_text
        assert "3 years Python" in clean_block

    @pytest.mark.asyncio
    async def test_clean_block_strips_freshness_system_reminders(self) -> None:
        stale = _store_item("CareerGoal", {"role": "Data Engineer"}, updated_at="2020-01-01T00:00:00")
        store = MagicMock()
        store.asearch = AsyncMock(return_value=[stale])

        raw_text, clean_block = await fetch_semantic_memory_block(store, "user-1")

        assert "<system-reminder>" in raw_text
        assert "<system-reminder>" not in clean_block

    @pytest.mark.asyncio
    async def test_returns_empty_on_store_error(self) -> None:
        store = MagicMock()
        store.asearch = AsyncMock(side_effect=ConnectionError("pg down"))

        assert await fetch_semantic_memory_block(store, "user-1") == ("", "")

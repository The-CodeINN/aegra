"""Unit tests for cold-path task persistence + dedup (agent optimisation spec Item 2 §4.2).

consolidate_memories runs after every turn, so persist_extracted_tasks must
dedup extracted candidates against already-open tasks — otherwise a task
would be re-created on each subsequent turn. This is the highest-risk part
of the cold path.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GRAPHS_ROOT = PROJECT_ROOT / "graphs"
if str(GRAPHS_ROOT) not in sys.path:
    sys.path.insert(0, str(GRAPHS_ROOT))

from react_agent import tools as tools_module  # noqa: E402
from react_agent.tools import _normalize_task_description, persist_extracted_tasks  # noqa: E402


def test_normalize_collapses_whitespace_and_case() -> None:
    assert _normalize_task_description("  Rebuild   GitHub  README ") == "rebuild github readme"
    assert _normalize_task_description("Rebuild GitHub README") == _normalize_task_description("rebuild github readme")


def _existing_task(description: str) -> MagicMock:
    item = MagicMock()
    item.description = description
    return item


class TestPersistExtractedTasks:
    @pytest.mark.asyncio
    async def test_skips_task_matching_an_existing_open_task(self, monkeypatch: pytest.MonkeyPatch) -> None:
        session = AsyncMock()
        session_ctx = MagicMock()
        session_ctx.__aenter__ = AsyncMock(return_value=session)
        session_ctx.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr(tools_module, "_get_task_session_maker", lambda: lambda: session_ctx)
        monkeypatch.setattr(tools_module, "TASK_MEMORY_AVAILABLE", True)

        group = MagicMock()
        group.not_yet_due = [_existing_task("Rebuild GitHub README")]
        group.overdue = []
        create_calls: list = []

        async def _get_open_and_overdue(_session, _user_id):
            return group

        async def _create(_session, user_id, description, **kwargs):
            create_calls.append(description)
            return MagicMock()

        svc = MagicMock()
        svc.get_open_and_overdue = _get_open_and_overdue
        svc.create_action_item = _create
        monkeypatch.setattr(tools_module, "_AccountabilityService", svc)

        # Same task (different casing/whitespace) must be deduped away; the new one created.
        created = await persist_extracted_tasks(
            "user-1",
            [
                {"description": "rebuild   github readme"},
                {"description": "Apply to 2 analyst roles", "priority": "high"},
            ],
        )

        assert created == 1
        assert create_calls == ["Apply to 2 analyst roles"]

    @pytest.mark.asyncio
    async def test_dedups_duplicates_within_the_same_batch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        session = AsyncMock()
        session_ctx = MagicMock()
        session_ctx.__aenter__ = AsyncMock(return_value=session)
        session_ctx.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr(tools_module, "_get_task_session_maker", lambda: lambda: session_ctx)
        monkeypatch.setattr(tools_module, "TASK_MEMORY_AVAILABLE", True)

        group = MagicMock()
        group.not_yet_due = []
        group.overdue = []
        create_calls: list = []

        async def _get_open_and_overdue(_session, _user_id):
            return group

        async def _create(_session, user_id, description, **kwargs):
            create_calls.append(description)
            return MagicMock()

        svc = MagicMock()
        svc.get_open_and_overdue = _get_open_and_overdue
        svc.create_action_item = _create
        monkeypatch.setattr(tools_module, "_AccountabilityService", svc)

        created = await persist_extracted_tasks(
            "user-1",
            [
                {"description": "Practice SQL joins"},
                {"description": "practice sql joins"},  # dup of the above
            ],
        )

        assert created == 1
        assert create_calls == ["Practice SQL joins"]

    @pytest.mark.asyncio
    async def test_returns_zero_for_empty_task_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tools_module, "TASK_MEMORY_AVAILABLE", True)
        assert await persist_extracted_tasks("user-1", []) == 0

    @pytest.mark.asyncio
    async def test_returns_zero_when_task_memory_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tools_module, "TASK_MEMORY_AVAILABLE", False)
        assert await persist_extracted_tasks("user-1", [{"description": "x"}]) == 0

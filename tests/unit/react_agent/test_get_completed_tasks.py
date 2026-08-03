"""Unit tests for the get_completed_tasks tool (agent optimisation spec Item 2).

"Done tasks not forgotten": completed tasks must be answerable on request
("what have I done so far?") without ever being mixed into the open-task
surfaces (get_open_tasks / conversation-start injection / check-ins).
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

from react_agent import tools as tools_module  # noqa: E402
from react_agent.tools import get_completed_tasks  # noqa: E402


def _patch_runtime(monkeypatch: pytest.MonkeyPatch, user_id: str | None) -> None:
    runtime = SimpleNamespace(context=SimpleNamespace(user_id=user_id))
    monkeypatch.setattr(tools_module, "get_runtime", lambda _cls: runtime)


class TestGetCompletedTasks:
    @pytest.mark.asyncio
    async def test_returns_serialized_completed_tasks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_runtime(monkeypatch, "user-1")
        monkeypatch.setattr(tools_module, "TASK_MEMORY_AVAILABLE", True)

        session = AsyncMock()
        session_ctx = MagicMock()
        session_ctx.__aenter__ = AsyncMock(return_value=session)
        session_ctx.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr(tools_module, "_get_task_session_maker", lambda: lambda: session_ctx)

        item = MagicMock()
        item.id = "task-1"
        item.description = "Rebuild GitHub README"
        item.status = "completed"
        item.due_date = None
        item.priority = "normal"
        item.miss_count = 0
        item.evidence = "github.com/user/repo"
        item.advisor_note = None

        svc = MagicMock()
        svc.list_completed_items = AsyncMock(return_value=[item])
        monkeypatch.setattr(tools_module, "_AccountabilityService", svc)

        result = await get_completed_tasks()

        assert result["ok"] is True
        assert len(result["completed"]) == 1
        assert result["completed"][0]["description"] == "Rebuild GitHub README"
        svc.list_completed_items.assert_awaited_once_with(session, "user-1")

    @pytest.mark.asyncio
    async def test_requires_authentication(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_runtime(monkeypatch, None)
        monkeypatch.setattr(tools_module, "TASK_MEMORY_AVAILABLE", True)

        result = await get_completed_tasks()

        assert "error" in result

    @pytest.mark.asyncio
    async def test_returns_error_when_task_memory_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_runtime(monkeypatch, "user-1")
        monkeypatch.setattr(tools_module, "TASK_MEMORY_AVAILABLE", False)

        result = await get_completed_tasks()

        assert "error" in result

    @pytest.mark.asyncio
    async def test_returns_error_on_lookup_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_runtime(monkeypatch, "user-1")
        monkeypatch.setattr(tools_module, "TASK_MEMORY_AVAILABLE", True)

        session = AsyncMock()
        session_ctx = MagicMock()
        session_ctx.__aenter__ = AsyncMock(return_value=session)
        session_ctx.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr(tools_module, "_get_task_session_maker", lambda: lambda: session_ctx)

        svc = MagicMock()
        svc.list_completed_items = AsyncMock(side_effect=ConnectionError("db down"))
        monkeypatch.setattr(tools_module, "_AccountabilityService", svc)

        result = await get_completed_tasks()

        assert "error" in result

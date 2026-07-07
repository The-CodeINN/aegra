"""Unit tests for the cold-path task extraction (agent optimisation spec Item 2 §4.2).

The cold path is the background safety net that catches tasks the advisor
stated in prose but forgot to log via the manage_task hot-path tool. These
tests pin the JSON parser's robustness and the extraction orchestration's
dedup/mutual-exclusion behaviour.
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

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from react_agent import graph as graph_module  # noqa: E402
from react_agent.graph import _extract_and_log_tasks, _parse_extracted_tasks  # noqa: E402


class TestParseExtractedTasks:
    def test_parses_plain_json_array(self) -> None:
        parsed = _parse_extracted_tasks(
            '[{"description": "Rebuild README", "due_date": "2026-07-10", "priority": "high"}]'
        )
        assert parsed == [{"description": "Rebuild README", "due_date": "2026-07-10", "priority": "high"}]

    def test_strips_code_fences(self) -> None:
        parsed = _parse_extracted_tasks('```json\n[{"description": "Apply to 2 roles"}]\n```')
        assert parsed == [{"description": "Apply to 2 roles"}]

    def test_extracts_array_from_surrounding_prose(self) -> None:
        parsed = _parse_extracted_tasks('Sure: [{"description": "Practice joins"}] hope that helps')
        assert parsed == [{"description": "Practice joins"}]

    def test_returns_empty_for_no_array(self) -> None:
        assert _parse_extracted_tasks("There are no tasks in this exchange.") == []

    def test_returns_empty_for_malformed_json(self) -> None:
        assert _parse_extracted_tasks('[{"description": ') == []

    def test_drops_entries_without_description(self) -> None:
        parsed = _parse_extracted_tasks('[{"due_date": "2026-07-10"}, {"description": "Real task"}]')
        assert parsed == [{"description": "Real task"}]


class TestExtractAndLogTasks:
    @pytest.mark.asyncio
    async def test_persists_tasks_the_model_extracts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        model = MagicMock()
        model.ainvoke = AsyncMock(return_value=AIMessage(content='[{"description": "Rebuild GitHub README"}]'))

        persisted: list = []

        async def _fake_persist(user_id, tasks, *, thread_id=None, advisor_persona=None):
            persisted.append((user_id, tasks, thread_id, advisor_persona))
            return len(tasks)

        monkeypatch.setattr(graph_module, "persist_extracted_tasks", _fake_persist)

        messages = [
            HumanMessage(content="What should I do this week?"),
            AIMessage(content="Rebuild your GitHub README with 3 pinned projects before our next session."),
        ]
        await _extract_and_log_tasks(messages, model, "user-1", "thread-1", "David")

        assert len(persisted) == 1
        user_id, tasks, thread_id, advisor_persona = persisted[0]
        assert user_id == "user-1"
        assert tasks == [{"description": "Rebuild GitHub README"}]
        assert advisor_persona == "David"

    @pytest.mark.asyncio
    async def test_no_persist_when_model_returns_no_tasks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        model = MagicMock()
        model.ainvoke = AsyncMock(return_value=AIMessage(content="[]"))

        called = False

        async def _fake_persist(*args, **kwargs):
            nonlocal called
            called = True
            return 0

        monkeypatch.setattr(graph_module, "persist_extracted_tasks", _fake_persist)

        await _extract_and_log_tasks(
            [AIMessage(content="Great work today, nothing to assign.")], model, "user-1", None, None
        )
        assert called is False

    @pytest.mark.asyncio
    async def test_no_model_call_when_no_text_messages(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A window of only empty/tool messages shouldn't hit the extraction model."""
        model = MagicMock()
        model.ainvoke = AsyncMock()
        monkeypatch.setattr(graph_module, "persist_extracted_tasks", AsyncMock(return_value=0))

        await _extract_and_log_tasks([AIMessage(content="")], model, "user-1", None, None)
        model.ainvoke.assert_not_awaited()

"""Unit tests for the read-only advisor-memory reads used by proactive outreach.

Spec Item 7 (§9.1/§9.3): outreach reads episodic + procedural memory from the
same LangGraph store the live agent uses, without a live agent turn, and
degrades to empty context (never an error) when the store is unavailable.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from aegra_api.services import advisor_memory
from aegra_api.services.advisor_memory import fetch_advisor_memory_context


def _store_item(kind: str, content: dict | str, updated_at: str = "2026-07-01T00:00:00") -> SimpleNamespace:
    return SimpleNamespace(value={"kind": kind, "content": content}, updated_at=updated_at, created_at=updated_at)


def _patch_store(monkeypatch: pytest.MonkeyPatch, store: MagicMock | None) -> None:
    manager = MagicMock()
    if store is None:
        manager.get_store.side_effect = RuntimeError("Database not initialized")
    else:
        manager.get_store.return_value = store
    monkeypatch.setattr(advisor_memory, "db_manager", manager)


@pytest.mark.asyncio
async def test_returns_empty_context_when_store_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_store(monkeypatch, None)
    result = await fetch_advisor_memory_context("user-1", query="sql joins")
    assert result == {"relevant_episode": "", "behavior_profile": ""}


@pytest.mark.asyncio
async def test_returns_empty_context_for_missing_user_id(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_store(monkeypatch, MagicMock())
    assert await fetch_advisor_memory_context("") == {"relevant_episode": "", "behavior_profile": ""}


@pytest.mark.asyncio
async def test_renders_relevant_episode_from_semantic_search(monkeypatch: pytest.MonkeyPatch) -> None:
    episode = _store_item(
        "EpisodicMemory",
        {"event": "Failed module 3 SQL assessment", "outcome": "Agreed to redo joins practice"},
    )
    store = MagicMock()

    async def _asearch(namespace, *, filter=None, query=None, limit=10):
        if filter == {"kind": "EpisodicMemory"}:
            assert query == "sql joins"
            return [episode]
        return []

    store.asearch = AsyncMock(side_effect=_asearch)
    _patch_store(monkeypatch, store)

    result = await fetch_advisor_memory_context("user-1", query="sql joins")
    assert "Failed module 3 SQL assessment" in result["relevant_episode"]
    assert "redo joins practice" in result["relevant_episode"]


@pytest.mark.asyncio
async def test_picks_most_recent_behavior_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    old = _store_item("AdvisorBehaviorProfile", {"style": "gentle nudges"}, updated_at="2026-05-01T00:00:00")
    new = _store_item("AdvisorBehaviorProfile", {"style": "direct challenge"}, updated_at="2026-07-01T00:00:00")
    store = MagicMock()

    async def _asearch(namespace, *, filter=None, query=None, limit=10):
        if filter == {"kind": "AdvisorBehaviorProfile"}:
            return [old, new]
        return []

    store.asearch = AsyncMock(side_effect=_asearch)
    _patch_store(monkeypatch, store)

    result = await fetch_advisor_memory_context("user-1")
    assert "direct challenge" in result["behavior_profile"]
    assert "gentle nudges" not in result["behavior_profile"]


@pytest.mark.asyncio
async def test_falls_back_to_recency_search_when_semantic_query_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deployments without a vector index must still get the latest episode."""
    episode = _store_item("EpisodicMemory", {"event": "Landed first interview"})
    store = MagicMock()
    calls: list[dict] = []

    async def _asearch(namespace, *, filter=None, query=None, limit=10):
        calls.append({"filter": filter, "query": query})
        if query is not None:
            raise ValueError("Store does not have embeddings configured")
        if filter == {"kind": "EpisodicMemory"}:
            return [episode]
        return []

    store.asearch = AsyncMock(side_effect=_asearch)
    _patch_store(monkeypatch, store)

    result = await fetch_advisor_memory_context("user-1", query="interviews")
    assert "Landed first interview" in result["relevant_episode"]
    assert any(c["query"] is None and c["filter"] == {"kind": "EpisodicMemory"} for c in calls)


@pytest.mark.asyncio
async def test_store_errors_never_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    store = MagicMock()
    store.asearch = AsyncMock(side_effect=ConnectionError("pg down"))
    _patch_store(monkeypatch, store)

    result = await fetch_advisor_memory_context("user-1", query="anything")
    assert result == {"relevant_episode": "", "behavior_profile": ""}

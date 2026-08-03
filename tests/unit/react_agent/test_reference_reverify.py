"""Unit tests for ReferenceMemory 're-verify on use' (agent optimisation spec Item 3).

Saved reference links carry no time-based staleness window; instead, when
read_webpage actually hits a dead URL that a ReferenceMemory points at, the
tool result flags it so the agent corrects the stored reference.
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
from react_agent.tools import _normalize_url_for_match, _stored_reference_note  # noqa: E402


def _reference_item(resource: str, location: str) -> SimpleNamespace:
    return SimpleNamespace(value={"kind": "ReferenceMemory", "content": {"resource": resource, "location": location}})


def _patch_runtime(monkeypatch: pytest.MonkeyPatch, items: list | None, user_id: str = "user-1") -> None:
    store = MagicMock()
    store.asearch = AsyncMock(return_value=items or [])
    runtime = SimpleNamespace(store=store, context=SimpleNamespace(user_id=user_id))
    monkeypatch.setattr(tools_module, "get_runtime", lambda _cls: runtime)


class TestNormalizeUrlForMatch:
    def test_scheme_www_and_trailing_slash_are_ignored(self) -> None:
        assert _normalize_url_for_match("https://www.Example.com/portfolio/") == _normalize_url_for_match(
            "http://example.com/portfolio"
        )

    def test_different_paths_do_not_match(self) -> None:
        assert _normalize_url_for_match("https://example.com/a") != _normalize_url_for_match("https://example.com/b")


class TestStoredReferenceNote:
    @pytest.mark.asyncio
    async def test_flags_dead_url_matching_a_saved_reference(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_runtime(monkeypatch, [_reference_item("portfolio site", "https://www.kudus.dev/portfolio/")])
        note = await _stored_reference_note("https://kudus.dev/portfolio")
        assert note is not None
        assert "portfolio site" in note
        assert "manage_memory" in note

    @pytest.mark.asyncio
    async def test_returns_none_when_url_not_stored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_runtime(monkeypatch, [_reference_item("portfolio site", "https://kudus.dev/portfolio")])
        assert await _stored_reference_note("https://other-site.com/page") is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_references_exist(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_runtime(monkeypatch, [])
        assert await _stored_reference_note("https://kudus.dev/portfolio") is None

    @pytest.mark.asyncio
    async def test_returns_none_when_runtime_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise(_cls: object) -> None:
            raise RuntimeError("no runtime in scope")

        monkeypatch.setattr(tools_module, "get_runtime", _raise)
        assert await _stored_reference_note("https://kudus.dev/portfolio") is None

    @pytest.mark.asyncio
    async def test_ignores_non_dict_content_payloads(self, monkeypatch: pytest.MonkeyPatch) -> None:
        malformed = SimpleNamespace(value={"kind": "ReferenceMemory", "content": "just a string"})
        _patch_runtime(monkeypatch, [malformed])
        assert await _stored_reference_note("https://kudus.dev/portfolio") is None

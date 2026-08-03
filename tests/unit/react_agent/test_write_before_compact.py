"""Unit tests for write-before-compact (agent optimisation spec Item 5, §7.2).

The `summarize` node must wait for the previous turn's background
consolidate_memories writes to land before compacting history away, so
durable facts and open tasks always survive somewhere before being
summarized. Covers the tracking/await mechanism in isolation from the full
graph, which is expensive to construct in a unit test.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GRAPHS_ROOT = PROJECT_ROOT / "graphs"
if str(GRAPHS_ROOT) not in sys.path:
    sys.path.insert(0, str(GRAPHS_ROOT))

from react_agent import graph as graph_module  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_pending_writes() -> None:
    """Each test gets a clean slate — the dict is module-level state."""
    graph_module._pending_writes_by_thread.clear()


class TestSpawnBackgroundTaskTracking:
    @pytest.mark.asyncio
    async def test_task_with_thread_id_is_tracked(self) -> None:
        async def _noop() -> None:
            await asyncio.sleep(0)

        graph_module._spawn_background_task(_noop(), "test", thread_id="thread-1")

        assert "thread-1" in graph_module._pending_writes_by_thread
        assert len(graph_module._pending_writes_by_thread["thread-1"]) == 1

    @pytest.mark.asyncio
    async def test_task_without_thread_id_is_not_tracked(self) -> None:
        async def _noop() -> None:
            await asyncio.sleep(0)

        graph_module._spawn_background_task(_noop(), "test", thread_id=None)

        assert graph_module._pending_writes_by_thread == {}

    @pytest.mark.asyncio
    async def test_completed_task_is_removed_from_tracking(self) -> None:
        done = asyncio.Event()

        async def _mark_done() -> None:
            done.set()

        graph_module._spawn_background_task(_mark_done(), "test", thread_id="thread-1")
        await asyncio.wait_for(done.wait(), timeout=1)
        # Give the done-callback a tick to run after the awaited coroutine finishes.
        await asyncio.sleep(0)

        assert graph_module._pending_writes_by_thread.get("thread-1", set()) == set()


class TestAwaitPendingWrites:
    @pytest.mark.asyncio
    async def test_returns_immediately_when_nothing_pending(self) -> None:
        # Must not raise or hang when the thread has no tracked writes.
        await graph_module._await_pending_writes("thread-with-nothing-pending")
        await graph_module._await_pending_writes(None)

    @pytest.mark.asyncio
    async def test_waits_for_pending_write_to_complete(self) -> None:
        written = False

        async def _write() -> None:
            nonlocal written
            await asyncio.sleep(0.05)
            written = True

        graph_module._spawn_background_task(_write(), "test", thread_id="thread-1")
        await graph_module._await_pending_writes("thread-1")

        assert written is True

    @pytest.mark.asyncio
    async def test_times_out_gracefully_without_raising(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(graph_module, "_WRITE_BEFORE_COMPACT_TIMEOUT_SECONDS", 0.05)

        async def _hangs() -> None:
            await asyncio.sleep(10)

        graph_module._spawn_background_task(_hangs(), "test", thread_id="thread-1")

        # Should not raise TimeoutError — it's a safety valve, not a hard failure.
        await graph_module._await_pending_writes("thread-1")

        # Clean up the still-running background task so it doesn't leak into other tests.
        for task in list(graph_module._background_tasks):
            task.cancel()

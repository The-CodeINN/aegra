"""Replay harness: invoke the compiled agent graph against a golden-conversation fixture.

Runs the *real* compiled graph (``react_agent.graph.builder``) against a seeded
in-memory store, so it exercises actual prompt assembly, memory recall, and
compaction logic rather than a mock. Because the model may decide to call
live tools (LMS API, MongoDB, Brave search), running these end-to-end
requires a fully configured environment — matching this repo's existing
e2e convention (see ``scripts/test_graph_scope.py``). Behavioural
assertions live in ``test_golden_conversations.py``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
GRAPHS_ROOT = PROJECT_ROOT / "graphs"
if str(GRAPHS_ROOT) not in sys.path:
    sys.path.insert(0, str(GRAPHS_ROOT))

EVALS_ROOT = Path(__file__).resolve().parent
if str(EVALS_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALS_ROOT))

from fixtures import ConversationFixture  # noqa: E402
from langgraph.store.base import BaseStore  # noqa: E402
from langgraph.store.memory import InMemoryStore  # noqa: E402
from react_agent.graph import builder  # noqa: E402
from react_agent.memory import DEFAULT_MEMORY_NAMESPACE  # noqa: E402


@dataclass(kw_only=True)
class ReplayResult:
    """Outcome of replaying one fixture through the compiled graph."""

    fixture: ConversationFixture
    final_text: str
    raw_state: dict[str, Any]


def _extract_final_text(messages: list[Any]) -> str:
    """Pull the text of the last AI message with non-empty content."""
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            text = "".join(block.get("text", "") for block in content if isinstance(block, dict))
            if text.strip():
                return text
    return ""


async def replay_fixture(fixture: ConversationFixture, *, store: BaseStore | None = None) -> ReplayResult:
    """Run *fixture* through the real compiled graph and return the final response.

    Pass the same ``store`` across two calls (e.g. two "sessions" for the
    same user) to test cross-conversation persistence — each call still
    seeds ``fixture.seed_memories`` into it, so callers seed once via the
    first fixture and leave later fixtures' ``seed_memories`` empty.
    """
    store = store if store is not None else InMemoryStore()
    if fixture.seed_memories:
        namespace = (fixture.context.get("user_id", "eval-user"), DEFAULT_MEMORY_NAMESPACE)
        for i, memory in enumerate(fixture.seed_memories):
            await store.aput(namespace, f"seed-{i}", {"kind": memory.kind, "content": memory.content})

    test_graph = builder.compile(store=store, name="ReAct Agent (eval replay)")

    result = await test_graph.ainvoke(
        {"messages": fixture.messages},
        context=fixture.context,
        config={"configurable": {"thread_id": f"eval-{fixture.name}"}},
    )

    final_text = _extract_final_text(result.get("messages", []))
    return ReplayResult(fixture=fixture, final_text=final_text, raw_state=result)

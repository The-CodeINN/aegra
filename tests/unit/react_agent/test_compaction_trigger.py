"""Unit tests for the compaction trigger threshold (agent optimisation spec Item 5).

The old 6,000-token trigger was aggressive enough to compress a normal
single-sitting conversation mid-flow — exactly when a advising chat needs
its recent thread intact. These tests pin the raised threshold and prove
a realistic-length conversation no longer crosses it.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GRAPHS_ROOT = PROJECT_ROOT / "graphs"
if str(GRAPHS_ROOT) not in sys.path:
    sys.path.insert(0, str(GRAPHS_ROOT))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.messages.utils import count_tokens_approximately  # noqa: E402
from react_agent.graph import _SUMMARY_TRIGGER_TOKENS  # noqa: E402

_OLD_TRIGGER_TOKENS = 6000


def _synthetic_conversation_with_tool_results(*, num_turns: int, tool_result_chars: int) -> list:
    """Build a message list sized like a realistic single sitting.

    Real conversations cross the compaction trigger via tool-call results
    (course content, onboarding data, search results), not turn text volume
    — this mirrors that rather than padding human/assistant text directly.
    """
    messages: list = []
    for i in range(num_turns):
        messages.append(HumanMessage(content=f"Turn {i}: can you check my course progress and advise me?"))
        messages.append(
            AIMessage(
                content=f"Turn {i}: let me look that up for you.",
                tool_calls=[{"name": "get_course_progress", "args": {}, "id": f"call-{i}"}],
            )
        )
        messages.append(ToolMessage(content="x" * tool_result_chars, tool_call_id=f"call-{i}"))
        messages.append(AIMessage(content=f"Turn {i}: here's what I found and my advice for you."))
    return messages


def test_trigger_raised_from_old_aggressive_default() -> None:
    assert _SUMMARY_TRIGGER_TOKENS > _OLD_TRIGGER_TOKENS
    assert _SUMMARY_TRIGGER_TOKENS >= 15000


def test_realistic_single_sitting_does_not_cross_new_trigger() -> None:
    """A normal single sitting (a handful of tool-backed turns) must never compress."""
    messages = _synthetic_conversation_with_tool_results(num_turns=5, tool_result_chars=6000)
    tokens = count_tokens_approximately(messages)

    assert tokens < _SUMMARY_TRIGGER_TOKENS


def test_same_conversation_would_have_crossed_old_trigger() -> None:
    """Confirms the raise is a real behavioural change, not just a bigger number."""
    messages = _synthetic_conversation_with_tool_results(num_turns=5, tool_result_chars=6000)
    tokens = count_tokens_approximately(messages)

    assert tokens > _OLD_TRIGGER_TOKENS


def test_very_long_sitting_still_eventually_triggers() -> None:
    """The trigger is raised, not removed — a genuinely long sitting still compresses."""
    messages = _synthetic_conversation_with_tool_results(num_turns=20, tool_result_chars=4000)
    tokens = count_tokens_approximately(messages)

    assert tokens > _SUMMARY_TRIGGER_TOKENS

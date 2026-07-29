"""Unit tests for _correct_hallucinated_response (graph.py).

Confirmed on prod via checkpoint analysis: the hallucination screen was
"log only" — it correctly flagged a response telling a student who'd
already passed Module 1 to "go crush Lesson 1", but the wrong text still
reached the user unmodified. This gives the model one bounded, fail-open
chance to self-correct using its FULL conversation context, rather than
either silently letting flagged output through or hard-blocking on a
classifier whose narrow evidence window is known to produce false
positives (e.g. names established via a tool call that got compacted out
of recent messages).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

GRAPHS_ROOT = Path(__file__).resolve().parents[3] / "graphs"
if str(GRAPHS_ROOT) not in sys.path:
    sys.path.insert(0, str(GRAPHS_ROOT))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from react_agent import graph as graph_module  # noqa: E402
from react_agent.cost_tracker import SessionCost  # noqa: E402
from react_agent.graph import _correct_hallucinated_response  # noqa: E402


def _runtime() -> MagicMock:
    rt = MagicMock()
    rt.context.model = "test-model"
    return rt


@pytest.mark.asyncio
async def test_applies_correction_when_model_reissues_fixed_text(monkeypatch: pytest.MonkeyPatch) -> None:
    original = AIMessage(content="Now go crush Lesson 1!", id="msg-1")
    corrected_msg = AIMessage(content="Great work on Module 2 so far!")

    async def _fake_invoke(messages, runtime, system_message):
        return corrected_msg, [], {}, []

    monkeypatch.setattr(graph_module, "_invoke_integrated_agent", _fake_invoke)

    events: list[dict] = []
    response, cost = await _correct_hallucinated_response(
        original,
        "Student already completed Module 1; Lesson 1 is stale.",
        [HumanMessage(content="hi")],
        _runtime(),
        "system prompt",
        SessionCost(),
        events,
    )

    assert response.content == "Great work on Module 2 so far!"
    assert response.id == "msg-1"  # preserves original message id for stream continuity
    assert any(e["event_type"] == "hallucination_correction_applied" for e in events)


@pytest.mark.asyncio
async def test_keeps_original_when_correction_reaffirms_same_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """The false-positive case: model has full context and reissues the same claim unchanged."""
    original = AIMessage(content="Hey Alonge, welcome back!", id="msg-1")
    reaffirmed = AIMessage(content="Hey Alonge, welcome back!")

    async def _fake_invoke(messages, runtime, system_message):
        return reaffirmed, [], {}, []

    monkeypatch.setattr(graph_module, "_invoke_integrated_agent", _fake_invoke)

    events: list[dict] = []
    response, _ = await _correct_hallucinated_response(
        original, "Name not found in narrow evidence window.", [], _runtime(), "system prompt", SessionCost(), events
    )

    assert response.content == "Hey Alonge, welcome back!"


@pytest.mark.asyncio
async def test_keeps_original_when_correction_pass_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    original = AIMessage(content="original text", id="msg-1")
    empty_msg = AIMessage(content="")

    async def _fake_invoke(messages, runtime, system_message):
        return empty_msg, [], {}, []

    monkeypatch.setattr(graph_module, "_invoke_integrated_agent", _fake_invoke)

    events: list[dict] = []
    response, _ = await _correct_hallucinated_response(
        original, "some detail", [], _runtime(), "system prompt", SessionCost(), events
    )

    assert response.content == "original text"
    assert any(e["event_type"] == "hallucination_correction_skipped" for e in events)


@pytest.mark.asyncio
async def test_keeps_original_when_correction_pass_tries_to_call_a_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    original = AIMessage(content="original text", id="msg-1")
    tool_call_msg = AIMessage(
        content="let me check", tool_calls=[{"name": "get_student_profile", "args": {}, "id": "1"}]
    )

    async def _fake_invoke(messages, runtime, system_message):
        return tool_call_msg, [], {}, []

    monkeypatch.setattr(graph_module, "_invoke_integrated_agent", _fake_invoke)

    events: list[dict] = []
    response, _ = await _correct_hallucinated_response(
        original, "some detail", [], _runtime(), "system prompt", SessionCost(), events
    )

    assert response.content == "original text"
    assert any(e["event_type"] == "hallucination_correction_skipped" for e in events)


@pytest.mark.asyncio
async def test_fails_open_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    original = AIMessage(content="original text", id="msg-1")

    async def _raise(messages, runtime, system_message):
        raise RuntimeError("model overloaded")

    monkeypatch.setattr(graph_module, "_invoke_integrated_agent", _raise)

    events: list[dict] = []
    response, cost = await _correct_hallucinated_response(
        original, "some detail", [], _runtime(), "system prompt", SessionCost(), events
    )

    assert response.content == "original text"
    assert any(e["event_type"] == "hallucination_correction_failed" for e in events)


@pytest.mark.asyncio
async def test_correction_prompt_includes_hallucination_details_and_original_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = AIMessage(content="Now go crush Lesson 1!", id="msg-1")
    captured_messages: list = []

    async def _fake_invoke(messages, runtime, system_message):
        captured_messages.extend(messages)
        return AIMessage(content="fixed"), [], {}, []

    monkeypatch.setattr(graph_module, "_invoke_integrated_agent", _fake_invoke)

    prior = [HumanMessage(content="thanks!")]
    await _correct_hallucinated_response(
        original,
        "Lesson 1 is stale; student is on Module 2.",
        prior,
        _runtime(),
        "system prompt",
        SessionCost(),
        [],
    )

    assert captured_messages[0] is prior[0]
    assert captured_messages[1] is original
    assert "Lesson 1 is stale; student is on Module 2." in captured_messages[2].content


@pytest.mark.asyncio
async def test_tracks_llm_usage_when_correction_reports_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    original = AIMessage(content="original", id="msg-1")
    usage = {"input_tokens": 100, "output_tokens": 20}
    tracked: list[tuple] = []

    async def _fake_invoke(messages, runtime, system_message):
        return AIMessage(content="fixed"), [], usage, []

    def _fake_track(session_cost, model, usage_metadata):
        tracked.append((model, usage_metadata))
        return session_cost

    monkeypatch.setattr(graph_module, "_invoke_integrated_agent", _fake_invoke)
    monkeypatch.setattr(graph_module, "track_llm_usage", _fake_track)

    await _correct_hallucinated_response(original, "detail", [], _runtime(), "system prompt", SessionCost(), [])

    assert tracked == [("test-model", usage)]

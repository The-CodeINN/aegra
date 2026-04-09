"""Define the state structures for the agent."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from langmem.short_term import RunningSummary

from react_agent.cost_tracker import SessionCost, merge_session_cost


def merge_tool_counts(existing: dict[str, int], new: dict[str, int]) -> dict[str, int]:
    """Merge tool call counts by taking the maximum value for each tool.

    This ensures that when state updates happen, we preserve the highest count
    seen for each tool, preventing the counter from resetting.
    """
    if not existing:
        return new
    if not new:
        return existing

    merged = existing.copy()
    for tool_name, count in new.items():
        merged[tool_name] = max(merged.get(tool_name, 0), count)
    return merged


@dataclass
class InputState:
    """Defines the input state for the agent, representing a narrower interface to the outside world.

    This class is used to define the initial state and structure of incoming data.
    """

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(default_factory=list)
    """
    Messages tracking the primary execution state of the agent.

    Typically accumulates a pattern of:
    1. HumanMessage - user input
    2. AIMessage with .tool_calls - agent picking tool(s) to use to collect information
    3. ToolMessage(s) - the responses (or errors) from the executed tools
    4. AIMessage without .tool_calls - agent responding in unstructured format to the user
    5. HumanMessage - user responds with the next conversational turn

    Steps 2-5 may repeat as needed.

    The `add_messages` annotation ensures that new messages are merged with existing ones,
    updating by ID to maintain an "append-only" state unless a message with the same ID is provided.
    """


@dataclass
class State(InputState):
    """Represents the complete state of the agent, extending InputState with additional attributes.

    This class can be used to store any information needed throughout the agent's lifecycle.
    """

    is_last_step: IsLastStep = field(default=False)
    """
    Indicates whether the current step is the last one before the graph raises an error.

    This is a 'managed' variable, controlled by the state machine rather than user code.
    It is set to 'True' when the step count reaches recursion_limit - 1.
    """

    tool_call_counts: Annotated[dict[str, int], merge_tool_counts] = field(default_factory=dict)
    """
    Tracks the number of times each tool has been called in this run.

    Used to enforce limits on expensive or data-heavy tools like get_student_ai_career_advisor_onboarding.
    The merge_tool_counts reducer ensures counts are preserved across state updates.
    """

    thread_name: str = field(default="")
    """
    AI-generated short title for this conversation thread.
    Set once after the first complete exchange and never changed again.
    Exposed to the frontend via the thread state values.
    """

    context: dict[str, RunningSummary] = field(default_factory=dict)
    """
    LangMem short-term memory context.  Written by the ``summarize`` node and
    read on subsequent turns so the SummarizationNode can incrementally update
    the running summary rather than re-summarising already-condensed messages.

    Shape: ``{"running_summary": RunningSummary}``
    """

    summarized_messages: list[AnyMessage] = field(default_factory=list)
    """
    Token-bounded message window produced by the ``summarize`` node (LangMem
    SummarizationNode).  This is what ``call_model`` actually sends to the LLM
    each turn — it may contain a summary message prepended to recent messages.

    Not annotated with ``add_messages`` because SummarizationNode always
    overwrites the entire window rather than appending to it.
    """

    guardrail_blocked: bool = field(default=False)
    """
    Set to ``True`` by the ``screen_input`` node when a prompt-injection or
    jailbreak attempt is detected.  Causes the graph to short-circuit to
    ``__end__`` without invoking the main model.
    """

    summarization_failure_count: int = field(default=0)
    """
    Consecutive summarization failures tracked by the ``summarize`` node.
    When this reaches 3 (``_SUMMARIZATION_CIRCUIT_BREAKER_LIMIT``), the node
    skips compression and passes the full message list straight to ``call_model``,
    preventing the 3,000+ failure-per-session spiral seen in production.
    Reset to 0 on any successful summarization.
    """

    has_attempted_reactive_compact: bool = field(default=False)
    """
    Set to ``True`` after the first context-overflow recovery attempt inside
    ``call_model``.  Prevents an infinite retry loop when the model context
    is too large even after summarization.
    """

    has_attempted_context_collapse: bool = field(default=False)
    """
    Set to ``True`` after the first context collapse (emergency compaction).
    Prevents infinite collapse retries.
    """

    has_attempted_microcompact: bool = field(default=False)
    """
    Set to ``True`` after the first microcompact pass to avoid repeating
    the same compression within the same error-recovery cycle.
    """

    session_cost: Annotated[SessionCost, merge_session_cost] = field(default_factory=SessionCost)
    """
    Accumulated cost and token usage for this session.
    Updated after every LLM invocation via the cost tracker.
    Uses a merge reducer that always takes the latest (cumulative) value.
    """

    execution_events: list[dict[str, Any]] = field(default_factory=list)
    """
    Structured runtime events emitted by the execution layer.

    These events make fallback decisions, retry exhaustion, and compaction
    behavior visible to observability and tests without changing the user-
    facing message stream.
    """

    # --- Session memory (ported from Claude-code's SESSIONMEMORY.md) --------

    session_notes: str = field(default="")
    """
    Structured session notes maintained across turns.  Updated by the
    ``update_session_memory`` node after each final AI response.  Injected
    into the system prompt so the agent retains context after compaction.
    """

    session_memory_token_count: int = field(default=0)
    """
    Token count at the time of the last session memory extraction.
    Used by ``should_extract_session_memory`` to throttle extraction.
    """

    # --- Mutual exclusion for memory consolidation --------------------------

    agent_wrote_memory: bool = field(default=False)
    """
    Set to ``True`` when the agent called ``manage_memory`` during the hot
    path.  When set, ``consolidate_memories`` skips background extraction
    to avoid duplication (ported from Claude-code's mutual exclusion pattern).
    Reset to ``False`` at the start of each turn.
    """

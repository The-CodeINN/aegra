from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GRAPHS_ROOT = PROJECT_ROOT / "graphs"
if str(GRAPHS_ROOT) not in sys.path:
    sys.path.insert(0, str(GRAPHS_ROOT))  # noqa: E402

from react_agent.execution import (  # noqa: E402
    build_runtime_tools,
    build_tool_limit_notice,
    build_tool_policy_registry,
    serialize_event,
)
from react_agent.execution.events import ExecutionEvent  # noqa: E402


async def get_student_profile() -> dict[str, str]:
    """Return a stub student profile."""
    return {"ok": "yes"}


async def search_course_content() -> dict[str, str]:
    """Return a stub course-content search result."""
    return {"ok": "yes"}


async def custom_tool() -> dict[str, str]:
    """Return a stub custom tool result."""
    return {"ok": "yes"}


def test_build_tool_policy_registry_applies_known_overrides() -> None:
    policies = build_tool_policy_registry([get_student_profile, search_course_content, custom_tool])

    assert policies["get_student_profile"].programmatic_safe is True
    assert policies["get_student_profile"].defer_loading is False
    assert policies["get_student_profile"].evidence_required is True

    assert policies["search_course_content"].call_limit == 10
    assert policies["search_course_content"].max_retries == 2
    assert policies["search_course_content"].output_schema_name == "CourseContentSearchResult"

    assert policies["custom_tool"].programmatic_safe is False
    assert policies["custom_tool"].defer_loading is True
    assert policies["custom_tool"].call_limit is None


def test_build_tool_limit_notice_lists_tools_over_limit() -> None:
    policies = build_tool_policy_registry([get_student_profile, search_course_content])

    notice = build_tool_limit_notice(
        {"search_course_content": 10, "get_student_profile": 1},
        policies,
    )

    assert notice is not None
    assert "search_course_content" in notice
    assert "10/10" in notice
    assert "get_student_profile" not in notice


def test_build_runtime_tools_for_bedrock_keeps_plain_tools() -> None:
    runtime_tools, policies = build_runtime_tools(
        tools=[get_student_profile, custom_tool],
        model_name="bedrock/eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
        anthropic_tool_search_enabled=True,
        anthropic_tool_search_variant="bm25",
        anthropic_programmatic_tool_calling_enabled=True,
    )

    assert runtime_tools == [get_student_profile, custom_tool]
    assert policies["get_student_profile"].programmatic_safe is True


def test_build_runtime_tools_for_anthropic_adds_provider_helpers() -> None:
    runtime_tools, policies = build_runtime_tools(
        tools=[get_student_profile, custom_tool],
        model_name="anthropic/claude-sonnet-4-5-20250929",
        anthropic_tool_search_enabled=True,
        anthropic_tool_search_variant="bm25",
        anthropic_programmatic_tool_calling_enabled=True,
    )

    tool_names = []
    for tool in runtime_tools:
        name = getattr(tool, "name", None)
        if name is None and isinstance(tool, dict):
            name = tool.get("name")
        tool_names.append(name)

    assert len(runtime_tools) == 4
    assert "get_student_profile" in tool_names
    assert "custom_tool" in tool_names
    assert "tool_search_tool_bm25" in tool_names
    assert "code_execution" in tool_names
    assert policies["custom_tool"].defer_loading is True


def test_serialize_event_preserves_structured_metadata() -> None:
    payload = serialize_event(
        ExecutionEvent(
            event_type="fallback_mode",
            level="warning",
            message="Returned a safe fallback response.",
            metadata={"component": "model", "reason": "retry_exhausted"},
        )
    )

    assert payload["event_type"] == "fallback_mode"
    assert payload["level"] == "warning"
    assert payload["metadata"]["reason"] == "retry_exhausted"
    assert payload["timestamp"]

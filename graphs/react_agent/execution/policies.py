"""Declarative tool policy definitions for the react agent runtime."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any, Literal

ToolRiskLevel = Literal["low", "medium", "high"]
ToolConcurrencyClass = Literal["parallel_safe", "serialized"]


@dataclass(frozen=True, slots=True)
class ToolPolicy:
    """Execution policy attached to an agent tool."""

    name: str
    timeout_seconds: float = 10.0
    max_retries: int = 0
    risk_level: ToolRiskLevel = "low"
    concurrency_class: ToolConcurrencyClass = "parallel_safe"
    call_limit: int | None = None
    output_schema_name: str = "untyped"
    evidence_required: bool = False
    defer_loading: bool = False
    programmatic_safe: bool = False

    @property
    def allowed_callers(self) -> tuple[str, ...]:
        if self.programmatic_safe:
            return ("direct", "code_execution_20260120")
        return ()


_NON_DEFERRED_TOOLS = {
    "get_student_profile",
    "get_student_onboarding",
    "brave_search",
}

_PROGRAMMATIC_SAFE_TOOLS = {
    "brave_search",
    "read_webpage",
    "get_student_profile",
    "get_student_onboarding",
    "get_student_ai_career_advisor_onboarding",
    "get_student_enrollment_overview",
    "get_course_structure",
    "get_course_materials",
    "get_course_progress",
    "get_student_attempts",
    "get_subscription_state",
    "get_portfolio_projects",
    "review_project_submission",
    "search_course_content",
    "manage_memory",
    "search_memory",
}

_POLICY_OVERRIDES: dict[str, ToolPolicy] = {
    "brave_search": ToolPolicy(
        name="brave_search",
        timeout_seconds=15.0,
        max_retries=2,
        output_schema_name="BraveSearchResult",
    ),
    "read_webpage": ToolPolicy(
        name="read_webpage",
        timeout_seconds=15.0,
        max_retries=1,
        output_schema_name="WebpageReadResult",
    ),
    "get_student_profile": ToolPolicy(
        name="get_student_profile",
        max_retries=3,
        output_schema_name="StudentProfileResult",
        evidence_required=True,
    ),
    "get_student_onboarding": ToolPolicy(
        name="get_student_onboarding",
        max_retries=3,
        output_schema_name="StudentOnboardingResult",
        evidence_required=True,
    ),
    "get_student_ai_career_advisor_onboarding": ToolPolicy(
        name="get_student_ai_career_advisor_onboarding",
        max_retries=3,
        call_limit=3,
        output_schema_name="CareerAdvisorOnboardingResult",
        evidence_required=True,
    ),
    "get_student_enrollment_overview": ToolPolicy(
        name="get_student_enrollment_overview",
        max_retries=3,
        output_schema_name="EnrollmentOverviewResult",
        evidence_required=True,
    ),
    "get_course_structure": ToolPolicy(
        name="get_course_structure",
        max_retries=3,
        call_limit=6,
        output_schema_name="CourseStructureResult",
        evidence_required=True,
    ),
    "get_course_materials": ToolPolicy(
        name="get_course_materials",
        max_retries=3,
        call_limit=6,
        output_schema_name="CourseMaterialsResult",
        evidence_required=True,
    ),
    "get_course_progress": ToolPolicy(
        name="get_course_progress",
        max_retries=3,
        call_limit=6,
        output_schema_name="CourseProgressResult",
        evidence_required=True,
    ),
    "get_student_attempts": ToolPolicy(
        name="get_student_attempts",
        max_retries=3,
        call_limit=6,
        output_schema_name="StudentAttemptsResult",
        evidence_required=True,
    ),
    "get_subscription_state": ToolPolicy(
        name="get_subscription_state",
        max_retries=3,
        output_schema_name="SubscriptionStateResult",
        evidence_required=True,
    ),
    "get_portfolio_projects": ToolPolicy(
        name="get_portfolio_projects",
        max_retries=3,
        output_schema_name="PortfolioProjectsResult",
        evidence_required=True,
    ),
    "review_project_submission": ToolPolicy(
        name="review_project_submission",
        timeout_seconds=20.0,
        max_retries=1,
        risk_level="high",
        concurrency_class="serialized",
        call_limit=2,
        output_schema_name="ProjectReviewResult",
        evidence_required=True,
    ),
    "search_course_content": ToolPolicy(
        name="search_course_content",
        timeout_seconds=20.0,
        max_retries=2,
        call_limit=10,
        output_schema_name="CourseContentSearchResult",
        evidence_required=True,
    ),
    "manage_memory": ToolPolicy(
        name="manage_memory",
        timeout_seconds=10.0,
        max_retries=1,
        risk_level="medium",
        concurrency_class="serialized",
        output_schema_name="MemoryMutationResult",
    ),
    "search_memory": ToolPolicy(
        name="search_memory",
        timeout_seconds=10.0,
        max_retries=1,
        output_schema_name="MemorySearchResult",
    ),
}


def resolve_tool_name(tool_fn: Any) -> str:
    """Return the stable name for a tool object or callable."""

    return getattr(tool_fn, "name", getattr(tool_fn, "__name__", "tool"))


def build_tool_policy_registry(tools: Iterable[Any]) -> dict[str, ToolPolicy]:
    """Return effective policy objects for the provided tools."""

    registry: dict[str, ToolPolicy] = {}
    for tool_fn in tools:
        name = resolve_tool_name(tool_fn)
        base = _POLICY_OVERRIDES.get(name, ToolPolicy(name=name))
        registry[name] = replace(
            base,
            defer_loading=name not in _NON_DEFERRED_TOOLS,
            programmatic_safe=name in _PROGRAMMATIC_SAFE_TOOLS,
        )
    return registry


def build_tool_limit_notice(tool_call_counts: dict[str, int], policies: dict[str, ToolPolicy]) -> str | None:
    """Return the runtime warning appended to the system prompt when tools are over limit."""

    over_limit = [
        f"{policy.name} (called {tool_call_counts.get(policy.name, 0)}/{policy.call_limit} times)"
        for policy in policies.values()
        if policy.call_limit is not None and tool_call_counts.get(policy.name, 0) >= policy.call_limit
    ]
    if not over_limit:
        return None
    return (
        "\n\n<tool_limit_notice>"
        "The following tools have reached their call limit for this conversation "
        "and MUST NOT be called again: "
        + ", ".join(sorted(over_limit))
        + ". Use the information already retrieved to complete your response."
        "</tool_limit_notice>"
    )

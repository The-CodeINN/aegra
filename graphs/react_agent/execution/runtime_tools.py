"""Helpers for building provider-specific runtime tools from policy definitions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from anthropic.types.beta import (
    BetaCodeExecutionTool20260120Param,
    BetaToolSearchToolBm25_20251119Param,
    BetaToolSearchToolRegex20251119Param,
)
from langchain.tools import tool as lc_tool

from react_agent.execution.policies import ToolPolicy, build_tool_policy_registry, resolve_tool_name


def _is_anthropic_model(model_name: str) -> bool:
    return model_name.split("/", maxsplit=1)[0].lower() == "anthropic"


def _is_bedrock_model(model_name: str) -> bool:
    return model_name.split("/", maxsplit=1)[0].lower() == "bedrock"


def build_runtime_tools(
    *,
    tools: Sequence[Any],
    model_name: str,
    anthropic_tool_search_enabled: bool,
    anthropic_tool_search_variant: str,
    anthropic_programmatic_tool_calling_enabled: bool,
) -> tuple[list[Any], dict[str, ToolPolicy]]:
    """Return provider-compatible tools plus the effective policy registry."""

    all_tools = list(tools)
    policies = build_tool_policy_registry(all_tools)

    if _is_bedrock_model(model_name):
        return all_tools, policies
    if not _is_anthropic_model(model_name):
        return all_tools, policies

    wrapped_tools: list[Any] = []
    for tool_fn in all_tools:
        tool_name = resolve_tool_name(tool_fn)
        policy = policies[tool_name]
        extras: dict[str, Any] = {}

        if anthropic_tool_search_enabled and policy.defer_loading:
            extras["defer_loading"] = True

        if anthropic_programmatic_tool_calling_enabled and policy.programmatic_safe:
            extras["allowed_callers"] = list(policy.allowed_callers)

        if extras:
            wrapped_tools.append(lc_tool(name_or_callable=tool_fn, extras=extras))
        else:
            wrapped_tools.append(tool_fn)

    if anthropic_tool_search_enabled:
        variant = (anthropic_tool_search_variant or "bm25").lower().strip()
        if variant == "regex":
            wrapped_tools.append(
                BetaToolSearchToolRegex20251119Param(
                    name="tool_search_tool_regex",
                    type="tool_search_tool_regex_20251119",
                )
            )
        else:
            wrapped_tools.append(
                BetaToolSearchToolBm25_20251119Param(
                    name="tool_search_tool_bm25",
                    type="tool_search_tool_bm25_20251119",
                )
            )

    if anthropic_programmatic_tool_calling_enabled:
        wrapped_tools.append(
            BetaCodeExecutionTool20260120Param(
                name="code_execution",
                type="code_execution_20260120",
            )
        )

    return wrapped_tools, policies

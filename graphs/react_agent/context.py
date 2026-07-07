"""Define the configurable parameters for the agent."""

from __future__ import annotations

import os
from dataclasses import MISSING, dataclass, field, fields
from typing import Annotated, Any

from react_agent import prompts


def _default_lms_api_url() -> str:
    """Resolve LMS URL from env, settings, or a safe local default."""
    lms_url = os.getenv("LMS_URL")
    if lms_url:
        return lms_url

    try:
        from aegra_api.settings import settings

        return settings.app.LMS_URL
    except Exception as exc:
        raise RuntimeError(
            "LMS_URL is not configured. Set LMS_URL in environment or configure AppSettings.LMS_URL."
        ) from exc


@dataclass(kw_only=True)
class Context:
    """The context for the agent."""

    system_prompt: str = field(
        default="",  # Will be dynamically generated based on advisor
        metadata={
            "description": "The system prompt to use for the agent's interactions. "
            "This prompt sets the context and behavior for the agent."
        },
    )

    advisor: dict[str, Any] | None = field(
        default=None,
        metadata={
            "description": "The career advisor assigned to the student based on their learning track. "
            "Contains name, title, experience, personality, expertise_areas, communication_style, background."
        },
    )

    learning_track: str | None = field(
        default=None,
        metadata={"description": "The student's current learning track (e.g., 'data-analytics', 'data-science')."},
    )

    roadmap_generated: bool = field(
        default=False,
        metadata={"description": "Whether this user has already successfully generated their career roadmap before."},
    )

    roadmap_variant: str | None = field(
        default=None,
        metadata={
            "description": "A/B variant for returning students' roadmap structure: 'adaptive' (loose, "
            "task/episode-led) or 'full' (fixed 7-part). Only takes effect when roadmap_generated=True — "
            "first-time students always get the full structure. See graphs/react_agent/prompts.py."
        },
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        # default="openai/gpt-5-mini-2025-08-07",  # noqa: ERA001
        # default="anthropic/claude-sonnet-4-5-20250929",  # noqa: ERA001
        default="bedrock/eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
        metadata={
            "description": "The name of the language model to use for the agent's main interactions. "
            "Should be in the form: provider/model-name."
        },
    )

    enable_thinking: bool = field(
        default=False,
        metadata={
            "description": "Enable extended thinking (reasoning) for Claude models. "
            "When enabled, the AI will show its reasoning process."
        },
    )

    thinking_budget: int = field(
        default=10000,
        metadata={
            "description": "Token budget for extended thinking (min 1024, max 128000). "
            "Higher values allow more thorough reasoning but cost more."
        },
    )

    max_search_results: int = field(
        default=10,
        metadata={"description": "The maximum number of search results to return for each search query."},
    )

    user_token: str | None = field(
        default=None,
        repr=False,  # Never include in repr — prevents JWT leaking into logs
        metadata={"description": "JWT access token for authenticating with external LMS API."},
    )

    user_id: str | None = field(
        default=None,
        metadata={"description": "User ID extracted from JWT token for memory namespacing."},
    )

    enrolled_course_ids: list[str] = field(
        default_factory=list,
        metadata={
            "description": "Authoritative enrolled course IDs for this user. Search tools must restrict retrieval to this scope."
        },
    )

    lms_api_url: str = field(
        default_factory=_default_lms_api_url,
        metadata={"description": "Base URL for the LMS API to fetch student information."},
    )

    brave_search_api_key: str | None = field(
        default=None,
        repr=False,  # Never include in repr — prevents API key leaking into logs
        metadata={"description": "The API key for Brave Search."},
    )

    anthropic_prompt_caching_enabled: bool = field(
        default=True,
        metadata={
            "description": "Enable prompt caching for Claude models on both providers — direct "
            "Anthropic API and Bedrock Converse. See graphs/react_agent/prompt_caching.py."
        },
    )

    anthropic_prompt_caching_ttl: str = field(
        default="1h",
        metadata={
            "description": "Prompt cache TTL for the static system-prompt block. Supported "
            "values: '5m' or '1h'. The static block is stable for the whole session, so '1h' "
            "avoids cache expiry mid-conversation on longer sittings."
        },
    )

    anthropic_tool_search_enabled: bool = field(
        default=True,
        metadata={"description": "Enable Anthropic server-side tool search for deferred tools."},
    )

    anthropic_tool_search_variant: str = field(
        default="bm25",
        metadata={"description": "Anthropic tool search variant: 'bm25' or 'regex'."},
    )

    anthropic_programmatic_tool_calling_enabled: bool = field(
        default=False,
        metadata={
            "description": "Enable Anthropic code-execution based programmatic tool calling for supported read-only tools."
        },
    )

    guardrails_enabled: bool = field(
        default=True,
        metadata={"description": "Enable input injection screening and output leak detection guardrails."},
    )

    guardrail_model: str = field(
        default="bedrock/eu.anthropic.claude-haiku-4-5-20251001-v1:0",
        metadata={
            "description": "Lightweight model used for input injection classification. "
            "Should be a fast, low-cost model. Format: provider/model-name."
        },
    )

    def __post_init__(self) -> None:
        """Apply env overrides for default values and build the dynamic prompt."""
        for f in fields(self):
            if not f.init:
                continue

            env_value = os.environ.get(f.name.upper())
            if env_value is None:
                continue

            current_value = getattr(self, f.name)
            has_default = f.default is not MISSING
            is_default_value = has_default and current_value == f.default
            is_missing_like = current_value is None or current_value == "" or current_value == []

            if not (is_default_value or is_missing_like):
                continue

            base_value = f.default if has_default else current_value
            setattr(self, f.name, _coerce_env_value(env_value, base_value))

        if not self.system_prompt:
            self.system_prompt = prompts.get_dynamic_system_prompt(
                advisor=self.advisor,
                learning_track=self.learning_track,
                roadmap_generated=self.roadmap_generated,
                roadmap_variant=self.roadmap_variant,
            )


def _coerce_env_value(raw: str, default: Any) -> Any:
    if isinstance(default, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        try:
            return int(raw)
        except ValueError:
            return default
    return raw

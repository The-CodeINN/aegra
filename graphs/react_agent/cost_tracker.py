"""Per-session cost and token tracking.

Ported from Claude-code's ``src/cost-tracker.ts``.

Accumulates input/output/cache token usage and computes USD cost after
each LLM invocation.  The ``SessionCost`` dataclass is stored in graph
state so it persists across turns and can be inspected for observability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model pricing (USD per 1 token — NOT per 1K)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Per-token pricing for a single model."""

    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0


# Pricing data sourced from provider pricing pages.
# Keys are the raw model IDs (without provider prefix).
# Bedrock-only: EU cross-region inference profiles.
_MODEL_PRICING: dict[str, ModelPricing] = {
    # Claude Sonnet 4.5 (primary model)
    "eu.anthropic.claude-sonnet-4-5-20250929-v1:0": ModelPricing(
        input=3e-6,
        output=15e-6,
        cache_read=0.3e-6,
        cache_write=3.75e-6,
    ),
    # Claude Haiku 4.5 (fallback / guardrail / title generation)
    "eu.anthropic.claude-haiku-4-5-20251001-v1:0": ModelPricing(
        input=0.8e-6,
        output=4e-6,
        cache_read=0.08e-6,
        cache_write=1e-6,
    ),
    # Kimi K2.5 (Moonshot AI — intermediate fallback, no caching support)
    "moonshotai.kimi-k2.5": ModelPricing(
        input=0.6e-6,
        output=3e-6,
    ),
}

# Fallback pricing for unknown models (assumes mid-range)
_FALLBACK_PRICING = ModelPricing(input=3e-6, output=15e-6)


# ---------------------------------------------------------------------------
# Per-model usage accumulator
# ---------------------------------------------------------------------------


@dataclass
class ModelUsage:
    """Token usage for a single model within a session."""

    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    invocations: int = 0


# ---------------------------------------------------------------------------
# Session-level cost state
# ---------------------------------------------------------------------------


@dataclass
class SessionCost:
    """Accumulated cost and token usage for an entire agent session.

    Stored in graph state via a merge reducer so it persists across turns.
    """

    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_write_tokens: int = 0
    total_api_duration_ms: float = 0.0
    total_tool_duration_ms: float = 0.0
    model_usage: dict[str, ModelUsage] = field(default_factory=dict)
    turns: int = 0

    # Cost ceiling — emit a warning event when total exceeds this.
    # Default $2 per session is generous; can be tuned via context.
    cost_warning_threshold_usd: float = 2.0


def merge_session_cost(existing: SessionCost, new: SessionCost) -> SessionCost:
    """Merge reducer for SessionCost — take the newer (higher) values."""
    if not existing:
        return new
    if not new:
        return existing
    # The new value is always the accumulated total, not a delta.
    return new


def _resolve_model_key(model_name: str) -> str:
    """Strip the provider prefix (e.g., 'anthropic/' or 'bedrock/') from a model name."""
    if "/" in model_name:
        return model_name.split("/", maxsplit=1)[1]
    return model_name


def _get_pricing(model_name: str) -> ModelPricing:
    """Look up pricing for a model, falling back to default if unknown."""
    key = _resolve_model_key(model_name)
    return _MODEL_PRICING.get(key, _FALLBACK_PRICING)


def extract_usage_from_message(msg: Any) -> dict[str, Any]:
    """Normalise an AIMessage's token usage into the flat shape ``track_llm_usage`` expects.

    ``ChatAnthropic`` (direct API) preserves the raw Anthropic usage block —
    already flat with ``cache_read_input_tokens``/``cache_creation_input_tokens`` —
    on ``response_metadata["usage"]``. ``ChatBedrockConverse`` instead *pops*
    that key and reports cache tokens nested under
    ``usage_metadata["input_token_details"]["cache_read"/"cache_creation"]``
    (LangChain's standardized shape), leaving ``response_metadata["usage"]``
    empty. Reading only ``response_metadata`` (as this code used to) silently
    dropped all token/cost/cache accounting for the Bedrock path — the
    default production model. Check the standardized field first; it's
    present for both providers, whereas the raw fallback only exists on
    Anthropic-direct responses.
    """
    usage_metadata = getattr(msg, "usage_metadata", None)
    if usage_metadata:
        details = usage_metadata.get("input_token_details") or {}
        cache_read = details.get("cache_read", 0) or 0
        cache_write = details.get("cache_creation", 0) or 0
        # LangChain's standardized ``input_tokens`` counts the FULL prompt,
        # cache tokens included (Bedrock Converse's convention). The pricing
        # formula below expects the Anthropic-raw convention instead —
        # ``input_tokens`` net of cache tokens, priced separately at the
        # cache rate — so subtract them back out here to avoid double-billing.
        input_tokens = max(0, (usage_metadata.get("input_tokens", 0) or 0) - cache_read - cache_write)
        return {
            "input_tokens": input_tokens,
            "output_tokens": usage_metadata.get("output_tokens", 0) or 0,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_write,
        }

    return getattr(msg, "response_metadata", {}).get("usage", {})


def track_llm_usage(
    session_cost: SessionCost,
    model_name: str,
    usage: dict[str, Any],
    api_duration_ms: float = 0.0,
) -> SessionCost:
    """Record token usage from a single LLM invocation and return updated cost.

    Args:
        session_cost: Current accumulated session cost.
        model_name: Full model name (e.g., ``"anthropic/claude-sonnet-4-20250514"``).
        usage: Usage dict from the LLM response metadata, expected to contain
               ``input_tokens``, ``output_tokens``, and optionally
               ``cache_read_input_tokens``, ``cache_creation_input_tokens``.
        api_duration_ms: Time taken for the API call in milliseconds.

    Returns:
        Updated ``SessionCost`` with the new usage added.
    """
    pricing = _get_pricing(model_name)
    model_key = _resolve_model_key(model_name)

    input_tokens = usage.get("input_tokens", 0) or 0
    output_tokens = usage.get("output_tokens", 0) or 0
    cache_read = usage.get("cache_read_input_tokens", 0) or 0
    cache_write = usage.get("cache_creation_input_tokens", 0) or 0

    turn_cost = (
        input_tokens * pricing.input
        + output_tokens * pricing.output
        + cache_read * pricing.cache_read
        + cache_write * pricing.cache_write
    )

    # Update per-model usage
    model_usage = dict(session_cost.model_usage)
    mu = model_usage.get(model_key, ModelUsage(model=model_key))
    model_usage[model_key] = ModelUsage(
        model=model_key,
        input_tokens=mu.input_tokens + input_tokens,
        output_tokens=mu.output_tokens + output_tokens,
        cache_read_tokens=mu.cache_read_tokens + cache_read,
        cache_write_tokens=mu.cache_write_tokens + cache_write,
        cost_usd=mu.cost_usd + turn_cost,
        invocations=mu.invocations + 1,
    )

    updated = SessionCost(
        total_cost_usd=session_cost.total_cost_usd + turn_cost,
        total_input_tokens=session_cost.total_input_tokens + input_tokens,
        total_output_tokens=session_cost.total_output_tokens + output_tokens,
        total_cache_read_tokens=session_cost.total_cache_read_tokens + cache_read,
        total_cache_write_tokens=session_cost.total_cache_write_tokens + cache_write,
        total_api_duration_ms=session_cost.total_api_duration_ms + api_duration_ms,
        total_tool_duration_ms=session_cost.total_tool_duration_ms,
        model_usage=model_usage,
        turns=session_cost.turns + 1,
        cost_warning_threshold_usd=session_cost.cost_warning_threshold_usd,
    )

    if turn_cost > 0:
        logger.debug(
            "Cost tracked: model=%s turn=$%.6f total=$%.4f (in=%d out=%d cache_r=%d cache_w=%d)",
            model_key,
            turn_cost,
            updated.total_cost_usd,
            input_tokens,
            output_tokens,
            cache_read,
            cache_write,
        )

    return updated


def track_tool_duration(session_cost: SessionCost, duration_ms: float) -> SessionCost:
    """Add tool execution time to the session cost."""
    return SessionCost(
        total_cost_usd=session_cost.total_cost_usd,
        total_input_tokens=session_cost.total_input_tokens,
        total_output_tokens=session_cost.total_output_tokens,
        total_cache_read_tokens=session_cost.total_cache_read_tokens,
        total_cache_write_tokens=session_cost.total_cache_write_tokens,
        total_api_duration_ms=session_cost.total_api_duration_ms,
        total_tool_duration_ms=session_cost.total_tool_duration_ms + duration_ms,
        model_usage=session_cost.model_usage,
        turns=session_cost.turns,
        cost_warning_threshold_usd=session_cost.cost_warning_threshold_usd,
    )


def is_over_cost_threshold(session_cost: SessionCost) -> bool:
    """Return ``True`` if the session has exceeded its cost warning threshold."""
    return session_cost.total_cost_usd >= session_cost.cost_warning_threshold_usd

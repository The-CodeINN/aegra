"""Provider-aware system-prompt caching (spec Item 1).

Splits the system prompt into a stable static block (persona, directives,
roadmap structure, anti-fabrication rules, tool descriptions, memory
instructions) and a volatile dynamic block (system time, tool-limit
notices, session context, proactive memory recall), and places an explicit
cache breakpoint between them so the static block is cache-eligible on
every turn instead of being invalidated by the volatile suffix that used
to be interleaved into the same string.

Anthropic direct API and Bedrock Converse use different wire formats for
cache breakpoints, so each gets its own block construction:

- Anthropic: ``cache_control`` set directly on the static content block.
- Bedrock Converse: a literal ``{"cachePoint": {...}}`` block inserted
  between the static and dynamic text blocks. ``ChatBedrockConverse``
  passes system content blocks through verbatim (see ``_messages_to_bedrock``
  in ``langchain_aws``), so a pre-built cachePoint placed exactly where we
  want it is honoured as-is — no reliance on the library's own
  end-of-prompt auto-append (see ``_apply_cache_points``), which would
  otherwise cache the whole prompt as one (still-volatile) block.

Deliberately does not use ``AnthropicPromptCachingMiddleware``: that
middleware tags the *last* content block of the system message, which
would land the breakpoint on the dynamic (volatile) block once the prompt
is split into two blocks — exactly backwards. Its other job, tool-schema
caching, is replicated here per provider instead.
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool

ProviderKind = Literal["anthropic", "bedrock", "other"]

# Bedrock prompt caching (cachePoint blocks) is only supported by specific
# underlying models. This codebase's "bedrock/" prefix covers both Claude
# and non-Claude models (e.g. the Kimi K2.5 overload fallback in
# graph.py's _FALLBACK_MODELS) — sending a cachePoint to an unsupported
# model risks a hard API error, which is the last thing an overload
# fallback path should introduce. Only treat it as cache-eligible when the
# underlying model is actually Claude.
_BEDROCK_CACHE_ELIGIBLE_MARKER = "claude"


def resolve_provider_kind(model_name: str) -> ProviderKind:
    """Map a ``provider/model`` string to a caching provider kind."""
    if model_name.startswith("bedrock/"):
        underlying = model_name.split("/", maxsplit=1)[1].lower()
        return "bedrock" if _BEDROCK_CACHE_ELIGIBLE_MARKER in underlying else "other"
    if model_name.startswith("anthropic/"):
        return "anthropic"
    return "other"


def build_cached_system_prompt(
    static_block: str,
    dynamic_block: str,
    *,
    provider: ProviderKind,
    cache_ttl: str = "1h",
) -> str | SystemMessage:
    """Return the value to pass as ``create_agent(system_prompt=...)``.

    For Anthropic/Bedrock (Claude), returns a two-block ``SystemMessage``
    with the cache breakpoint immediately after ``static_block``. For any
    other provider, falls back to a plain concatenated string — no caching
    mechanism is wired here for those.
    """
    if provider == "anthropic":
        return SystemMessage(
            content=[
                {
                    "type": "text",
                    "text": static_block,
                    "cache_control": {"type": "ephemeral", "ttl": cache_ttl},
                },
                {"type": "text", "text": dynamic_block},
            ]
        )

    if provider == "bedrock":
        # TTL must match the cache points langchain-aws appends via the
        # cache_control bind kwarg (tools + last message): Bedrock rejects a
        # longer-TTL block after a shorter one, and a bare cachePoint defaults
        # to 5m. Mirror _apply_cache_points: omit ttl only for the 5m default.
        cache_point: dict[str, Any] = {"type": "default"}
        if cache_ttl and cache_ttl != "5m":
            cache_point["ttl"] = cache_ttl
        return SystemMessage(
            content=[
                {"text": static_block},
                {"cachePoint": cache_point},
                {"text": dynamic_block},
            ]
        )

    return f"{static_block}\n{dynamic_block}" if dynamic_block else static_block


def flatten_system_message(system_message: str | SystemMessage) -> str | SystemMessage:
    """Return a cache-marker-free plain-string form of a system message.

    Needed when handing a system message built for a caching-capable model
    (a multi-block ``SystemMessage`` with a ``cachePoint`` / ``cache_control``
    breakpoint) to a provider that does NOT support caching — e.g. the Kimi
    K2.5 overload fallback. Sending a Bedrock ``cachePoint`` block to a model
    that doesn't support it can turn a recoverable 529 into a hard API error.
    Plain-string inputs (already cache-free) are returned unchanged.
    """
    if isinstance(system_message, str):
        return system_message
    content = system_message.content
    if not isinstance(content, list):
        return system_message

    texts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if "cachePoint" in block:
            continue  # drop the Bedrock breakpoint marker entirely
        text = block.get("text")
        if text:
            texts.append(text)
    return "\n".join(texts)


def tag_last_tool_for_caching(tools: list[Any], *, provider: ProviderKind, cache_ttl: str = "1h") -> list[Any]:
    """Tag the last tool definition so the whole tool-schema block is cached.

    Anthropic-only: mirrors ``AnthropicPromptCachingMiddleware._tag_tools`` —
    a single breakpoint on the last tool caches the entire contiguous
    tool-schema block. Bedrock tool caching is instead handled by
    ``ChatBedrockConverse``'s own ``cache_control`` bind kwarg (bound in
    ``graph.py``'s ``_build_runtime_model``), which appends a cachePoint to
    the tool list itself — nothing to do here for that path.
    """
    if provider != "anthropic" or not tools:
        return tools

    last = tools[-1]
    if not isinstance(last, BaseTool):
        return tools

    cache_control = {"type": "ephemeral", "ttl": cache_ttl}
    new_extras = {**(last.extras or {}), "cache_control": cache_control}
    return [*tools[:-1], last.model_copy(update={"extras": new_extras})]

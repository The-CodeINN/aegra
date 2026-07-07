from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GRAPHS_ROOT = PROJECT_ROOT / "graphs"
if str(GRAPHS_ROOT) not in sys.path:
    sys.path.insert(0, str(GRAPHS_ROOT))

from react_agent import prompt_caching, prompts  # noqa: E402


def test_static_prompt_is_byte_identical_across_turns_with_different_system_time() -> None:
    """Item 1's core regression: the cached block must never change turn to turn.

    Building the static block twice — with the volatile dynamic block fed a
    different system_time each time — must produce byte-identical static
    output, since Anthropic/Bedrock prompt caching only hits when the cached
    prefix is byte-identical.
    """
    static_prompt = prompts.get_dynamic_system_prompt(
        advisor=None,
        learning_track="data-analytics",
        roadmap_generated=True,
    )

    static_block_turn_1 = prompts.build_runtime_system_prompt(static_prompt)
    static_block_turn_2 = prompts.build_runtime_system_prompt(static_prompt)

    assert static_block_turn_1 == static_block_turn_2

    dynamic_turn_1 = prompts.build_dynamic_prompt_block(system_time="2026-07-03T09:00:00+00:00")
    dynamic_turn_2 = prompts.build_dynamic_prompt_block(system_time="2026-07-03T09:05:00+00:00")
    assert dynamic_turn_1 != dynamic_turn_2, "sanity check: dynamic block should change with system_time"


def test_static_prompt_contains_no_system_time_placeholder() -> None:
    """{system_time} must not leak into the cacheable static block at all."""
    static_prompt = prompts.get_dynamic_system_prompt(roadmap_generated=True)
    static_block = prompts.build_runtime_system_prompt(static_prompt)

    assert "{system_time}" not in static_block
    assert "System Time" not in static_block


def test_build_dynamic_prompt_block_includes_all_volatile_sections() -> None:
    block = prompts.build_dynamic_prompt_block(
        system_time="2026-07-03T09:00:00+00:00",
        tool_limit_notice="<tool_limit_notice>foo</tool_limit_notice>",
        session_block="<session_context>bar</session_context>",
        proactive_memory_block="<proactive_memory_recall>baz</proactive_memory_recall>",
    )
    assert "2026-07-03T09:00:00+00:00" in block
    assert "<tool_limit_notice>foo</tool_limit_notice>" in block
    assert "<session_context>bar</session_context>" in block
    assert "<proactive_memory_recall>baz</proactive_memory_recall>" in block


def test_resolve_provider_kind() -> None:
    assert prompt_caching.resolve_provider_kind("bedrock/eu.anthropic.claude-sonnet-4-5-20250929-v1:0") == "bedrock"
    assert prompt_caching.resolve_provider_kind("anthropic/claude-sonnet-4-5-20250929") == "anthropic"
    assert prompt_caching.resolve_provider_kind("openai/gpt-5-mini") == "other"


def test_resolve_provider_kind_excludes_non_claude_bedrock_models() -> None:
    """The Kimi K2.5 overload fallback must never get a Bedrock cachePoint."""
    assert prompt_caching.resolve_provider_kind("bedrock/moonshotai.kimi-k2.5") == "other"


def test_build_cached_system_prompt_places_breakpoint_on_static_block_anthropic() -> None:
    message = prompt_caching.build_cached_system_prompt(
        "STATIC CONTENT", "DYNAMIC CONTENT", provider="anthropic", cache_ttl="1h"
    )
    blocks = message.content
    assert blocks[0]["text"] == "STATIC CONTENT"
    assert blocks[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert blocks[1]["text"] == "DYNAMIC CONTENT"
    assert "cache_control" not in blocks[1]


def test_build_cached_system_prompt_places_cache_point_between_blocks_bedrock() -> None:
    message = prompt_caching.build_cached_system_prompt(
        "STATIC CONTENT", "DYNAMIC CONTENT", provider="bedrock", cache_ttl="1h"
    )
    blocks = message.content
    assert blocks[0] == {"text": "STATIC CONTENT"}
    assert blocks[1] == {"cachePoint": {"type": "default"}}
    assert blocks[2] == {"text": "DYNAMIC CONTENT"}


def test_build_cached_system_prompt_falls_back_to_plain_string_for_other_providers() -> None:
    result = prompt_caching.build_cached_system_prompt("STATIC", "DYNAMIC", provider="other")
    assert result == "STATIC\nDYNAMIC"


def test_tag_last_tool_for_caching_only_applies_to_anthropic() -> None:
    from langchain_core.tools import tool

    @tool
    def dummy_tool(x: str) -> str:
        """A dummy tool."""
        return x

    tools = [dummy_tool]
    tagged_anthropic = prompt_caching.tag_last_tool_for_caching(tools, provider="anthropic")
    assert tagged_anthropic[-1].extras.get("cache_control") == {"type": "ephemeral", "ttl": "1h"}

    tagged_bedrock = prompt_caching.tag_last_tool_for_caching(tools, provider="bedrock")
    assert tagged_bedrock is tools
    assert not (tagged_bedrock[-1].extras or {}).get("cache_control")


def test_flatten_system_message_strips_bedrock_cache_point() -> None:
    """Overload fallback to a non-caching model must not carry a cachePoint block."""
    msg = prompt_caching.build_cached_system_prompt("STATIC", "DYNAMIC", provider="bedrock")
    assert prompt_caching.flatten_system_message(msg) == "STATIC\nDYNAMIC"


def test_flatten_system_message_strips_anthropic_cache_control() -> None:
    msg = prompt_caching.build_cached_system_prompt("STATIC", "DYNAMIC", provider="anthropic")
    assert prompt_caching.flatten_system_message(msg) == "STATIC\nDYNAMIC"


def test_flatten_system_message_passes_through_plain_string() -> None:
    assert prompt_caching.flatten_system_message("already plain") == "already plain"

"""Anthropic API payload sanitizer middleware for tool search compatibility.

This middleware removes incompatible fields from tool search result blocks
before they're sent to the Anthropic API.
"""

from typing import Any


def sanitize_anthropic_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Recursively remove 'index' fields from tool_search_tool_result blocks in Anthropic payloads.

    The Anthropic beta API rejects tool_search_tool_result blocks that include
    an 'index' field. This function sanitizes the payload before sending.

    Args:
        payload: The request payload dict that will be sent to Anthropic

    Returns:
        Sanitized payload with incompatible fields removed
    """
    if not isinstance(payload, dict):
        return payload

    payload_copy = dict(payload)

    # Sanitize messages list if present
    if "messages" in payload_copy and isinstance(payload_copy["messages"], list):
        sanitized_messages = []
        for msg in payload_copy["messages"]:
            if isinstance(msg, dict):
                msg_copy = dict(msg)
                # Sanitize content blocks
                if "content" in msg_copy and isinstance(msg_copy["content"], list):
                    msg_copy["content"] = _sanitize_content_blocks(msg_copy["content"])
                sanitized_messages.append(msg_copy)
            else:
                sanitized_messages.append(msg)
        payload_copy["messages"] = sanitized_messages

    return payload_copy


def _sanitize_content_blocks(blocks: list[Any]) -> list[Any]:
    """Sanitize content blocks, removing index fields from tool_search_tool_result."""
    sanitized = []
    for block in blocks:
        if isinstance(block, dict):
            block_copy = dict(block)

            # Remove index field if it's directly on the block
            block_copy.pop("index", None)

            # Sanitize tool_search_tool_result field
            if "tool_search_tool_result" in block_copy:
                result = block_copy["tool_search_tool_result"]
                if isinstance(result, dict):
                    result_copy = dict(result)
                    result_copy.pop("index", None)
                    block_copy["tool_search_tool_result"] = result_copy

            sanitized.append(block_copy)
        else:
            sanitized.append(block)

    return sanitized

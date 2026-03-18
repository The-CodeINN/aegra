"""Utilities for message processing and sanitization for Anthropic compatibility."""

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage


def _sanitize_content_block(block: Any) -> Any:
    """Sanitize a single content block, removing Anthropic-incompatible fields.

    Removes 'index' fields from tool_search_tool_result blocks.
    """
    if not isinstance(block, dict):
        return block

    block_copy = dict(block)

    # Check for tool_search_tool_result field and sanitize it
    if "tool_search_tool_result" in block_copy and isinstance(block_copy["tool_search_tool_result"], dict):
        sanitized_result = dict(block_copy["tool_search_tool_result"])
        # Remove the problematic 'index' field
        sanitized_result.pop("index", None)
        block_copy["tool_search_tool_result"] = sanitized_result

    # Also check for direct 'index' field in the block itself
    block_copy.pop("index", None)

    return block_copy


def sanitize_messages_for_anthropic(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Remove Anthropic-incompatible fields from messages before model invocation.

    Specifically handles tool_search_tool_result blocks that may contain 'index'
    fields which are not permitted by the Anthropic API.

    Args:
        messages: List of LangChain message objects

    Returns:
        List of messages with problematic fields removed
    """
    sanitized = []

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.content:
            # Handle message content which may be a list of content blocks
            if isinstance(msg.content, list):
                sanitized_content = [_sanitize_content_block(block) for block in msg.content]

                # Create a new AIMessage with the sanitized content
                sanitized_msg = AIMessage(
                    content=sanitized_content,
                    id=msg.id,
                    response_metadata=getattr(msg, "response_metadata", None),
                )
                sanitized.append(sanitized_msg)
            else:
                # Content is not a list, pass through unchanged
                sanitized.append(msg)
        else:
            # Not an AIMessage, pass through unchanged
            sanitized.append(msg)

    return sanitized

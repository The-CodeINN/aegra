"""Custom ChatAnthropic wrapper with Anthropic payload sanitization."""

from typing import Any

from langchain_anthropic import ChatAnthropic

from react_agent.message_utils import _sanitize_content_block


def _sanitize_anthropic_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Recursively remove 'index' fields from tool_search_tool_result blocks."""
    if not isinstance(payload, dict):
        return payload

    payload_copy = dict(payload)
    if "messages" in payload_copy and isinstance(payload_copy["messages"], list):
        sanitized_messages = []
        for msg in payload_copy["messages"]:
            if isinstance(msg, dict):
                msg_copy = dict(msg)
                if "content" in msg_copy and isinstance(msg_copy["content"], list):
                    msg_copy["content"] = [_sanitize_content_block(b) for b in msg_copy["content"]]
                sanitized_messages.append(msg_copy)
            else:
                sanitized_messages.append(msg)
        payload_copy["messages"] = sanitized_messages

    return payload_copy


class SanitizedChatAnthropic(ChatAnthropic):
    """ChatAnthropic subclass that sanitizes payloads for Anthropic API compatibility.

    Removes incompatible fields (like 'index' from tool_search_tool_result blocks)
    before sending requests to the Anthropic API.
    """

    async def _acreate(self, payload: dict) -> Any:
        """Override _acreate to sanitize the payload before sending to Anthropic."""
        sanitized_payload = _sanitize_anthropic_payload(payload)
        return await super()._acreate(sanitized_payload)

    def _create(self, payload: dict) -> Any:
        """Override _create to sanitize the payload before sending to Anthropic (sync version)."""
        sanitized_payload = _sanitize_anthropic_payload(payload)
        return super()._create(sanitized_payload)

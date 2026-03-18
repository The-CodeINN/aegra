"""Custom ChatAnthropic wrapper with Anthropic payload sanitization."""

from typing import Any

from langchain_anthropic import ChatAnthropic

from react_agent.anthropic_payload_sanitizer import sanitize_anthropic_payload


class SanitizedChatAnthropic(ChatAnthropic):
    """ChatAnthropic subclass that sanitizes payloads for Anthropic API compatibility.

    Removes incompatible fields (like 'index' from tool_search_tool_result blocks)
    before sending requests to the Anthropic API.
    """

    async def _acreate(self, payload: dict) -> Any:
        """Override _acreate to sanitize the payload before sending to Anthropic."""
        # Sanitize the payload before calling parent
        sanitized_payload = sanitize_anthropic_payload(payload)
        return await super()._acreate(sanitized_payload)

    def _create(self, payload: dict) -> Any:
        """Override _create to sanitize the payload before sending to Anthropic (sync version)."""
        # Sanitize the payload before calling parent
        sanitized_payload = sanitize_anthropic_payload(payload)
        return super()._create(sanitized_payload)

"""Utilities for message processing and sanitization for Anthropic compatibility."""

import base64
import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

# ---------------------------------------------------------------------------
# MIME type sets
# ---------------------------------------------------------------------------

# Image MIME types supported by both Anthropic and Bedrock
_IMAGE_MIME_TYPES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})

# Bedrock Converse native document formats (MIME -> Bedrock format string)
# Source: langchain_aws MIME_TO_FORMAT map
_BEDROCK_DOC_MIME_TO_FORMAT: dict[str, str] = {
    "application/pdf": "pdf",
    "text/csv": "csv",
    "application/msword": "doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.ms-excel": "xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "text/html": "html",
    "text/plain": "txt",
    "text/markdown": "md",
}

# Anthropic native document MIME types (only PDF is supported as a document block)
_ANTHROPIC_DOC_MIME_TYPES = frozenset({"application/pdf"})

# Text MIME types the frontend tags as type:"text-plain" — decoded to plain text
# for Anthropic (which can't take them as document blocks).
# NOTE: text/csv, text/html, text/markdown, text/plain are intentionally NOT here
# because Bedrock handles them natively and Anthropic gets them decoded below.
_DECODE_AS_TEXT_MIME_TYPES = frozenset(
    {
        "text/x-python",
        "application/javascript",
        "text/javascript",
        "application/x-javascript",
        "text/x-java",
        "text/css",
        "text/x-c++src",
        "text/x-go",
        "text/x-php",
        "text/x-ruby",
        "application/typescript",
        "text/x-typescript",
        "application/json",
        "application/xml",
        "text/xml",
    }
)

# ---------------------------------------------------------------------------
# Bedrock filename sanitization
# ---------------------------------------------------------------------------

# Bedrock ConverseStream restriction: alphanumeric, space, hyphen, parentheses,
# square brackets only; no consecutive spaces.
_BEDROCK_FILENAME_INVALID = re.compile(r"[^a-zA-Z0-9 ()\[\]-]")
_BEDROCK_CONSECUTIVE_SPACES = re.compile(r" {2,}")


def _sanitize_bedrock_filename(name: str) -> str:
    """Sanitize a filename so it meets Bedrock ConverseStream restrictions."""
    sanitized = _BEDROCK_FILENAME_INVALID.sub(" ", name)
    sanitized = _BEDROCK_CONSECUTIVE_SPACES.sub(" ", sanitized).strip()
    return sanitized or "document"


# ---------------------------------------------------------------------------
# Block normalizer
# ---------------------------------------------------------------------------


def _normalize_frontend_content_block(block: Any, provider: str = "anthropic") -> Any:
    """Convert a frontend-format content block to the provider-specific API format.

    The frontend (fileToContentBlock) emits three block shapes:

    - ``type:"image"``      — images (JPEG/PNG/GIF/WEBP)
    - ``type:"file"``       — binary documents (PDF/DOCX/XLSX/DOC/XLS)
    - ``type:"text-plain"`` — text/code files (CSV, HTML, Markdown, Python, JS, …)

    All carry ``source_type:"base64"``, ``mime_type``, ``data`` (base64 string),
    and ``metadata.filename``.

    Provider targets
    ----------------
    ``"anthropic"``
        - Images  → ``{type:"image", source:{type:"base64", media_type, data}}``
        - PDF     → ``{type:"document", source:{type:"base64", media_type:"application/pdf", data}}``
        - Other binary (DOCX/XLSX …) → stub text block (not supported natively)
        - text/code → base64-decoded ``{type:"text", text:"[File: name]\\n<content>"}``

    ``"bedrock"``
        - Images  → ``{type:"image", source:{type:"base64", mediaType, data}}``
          (``mediaType`` camelCase — required by ``langchain_aws._lc_content_to_bedrock``)
        - All native document MIMEs (PDF/CSV/HTML/MD/DOCX/XLSX/DOC/XLS/TXT)
          → ``{type:"document", document:{format, name, source:{bytes}}}``
        - Remaining text/code → base64-decoded ``{type:"text", text:"[File: name]\\n<content>"}``
    """
    if not isinstance(block, dict):
        return block

    # Only touch blocks that arrived in the custom frontend format
    if block.get("source_type") != "base64":
        return block

    block_type: str = block.get("type", "")
    mime_type: str = block.get("mime_type", "")
    data: str = block.get("data", "")
    filename: str = (block.get("metadata") or {}).get("filename") or (block.get("metadata") or {}).get("name") or "file"

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------
    if block_type == "image" and mime_type in _IMAGE_MIME_TYPES:
        if provider == "bedrock":
            # langchain_aws reads block["source"]["mediaType"] (camelCase)
            return {
                "type": "image",
                "source": {"type": "base64", "mediaType": mime_type, "data": data},
            }
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": mime_type, "data": data},
        }

    # ------------------------------------------------------------------
    # Binary file blocks (PDF, DOCX, XLSX, …)
    # ------------------------------------------------------------------
    if block_type == "file":
        if provider == "bedrock" and mime_type in _BEDROCK_DOC_MIME_TO_FORMAT:
            bedrock_format = _BEDROCK_DOC_MIME_TO_FORMAT[mime_type]
            try:
                doc_bytes = base64.b64decode(data)
            except Exception:
                doc_bytes = data.encode() if isinstance(data, str) else data
            return {
                "type": "document",
                "document": {
                    "format": bedrock_format,
                    "name": _sanitize_bedrock_filename(filename),
                    "source": {"bytes": doc_bytes},
                },
            }

        if mime_type in _ANTHROPIC_DOC_MIME_TYPES:
            return {
                "type": "document",
                "source": {"type": "base64", "media_type": mime_type, "data": data},
            }

        # Unsupported binary format for this provider — surface a stub.
        return {
            "type": "text",
            "text": (
                f"[Attached file: {filename} ({mime_type}) — "
                "binary format not supported by the AI model; "
                "please share the text content directly.]"
            ),
        }

    # ------------------------------------------------------------------
    # Text / code blocks (frontend type:"text-plain")
    # ------------------------------------------------------------------
    if block_type == "text-plain":
        # Bedrock supports several of these natively as document blocks.
        if provider == "bedrock" and mime_type in _BEDROCK_DOC_MIME_TO_FORMAT:
            bedrock_format = _BEDROCK_DOC_MIME_TO_FORMAT[mime_type]
            try:
                doc_bytes = base64.b64decode(data)
            except Exception:
                doc_bytes = data.encode() if isinstance(data, str) else data
            return {
                "type": "document",
                "document": {
                    "format": bedrock_format,
                    "name": _sanitize_bedrock_filename(filename),
                    "source": {"bytes": doc_bytes},
                },
            }

        # Everything else (and Anthropic for all text/code types): decode to plain text.
        try:
            decoded = base64.b64decode(data).decode("utf-8", errors="replace")
        except Exception:
            decoded = data
        return {"type": "text", "text": f"[File: {filename}]\n{decoded}"}

    # Unknown custom block — return unchanged.
    return block


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


def sanitize_messages_for_anthropic(messages: list[BaseMessage], provider: str = "anthropic") -> list[BaseMessage]:
    """Remove provider-incompatible fields and normalise content blocks in all messages.

    Performs two passes:

    1. **HumanMessages**: converts frontend-format content blocks (custom
       ``source_type``/``mime_type``/``data`` layout) to the correct format
       for the target provider (``"anthropic"`` or ``"bedrock"``).
    2. **AIMessages**: strips ``index`` fields from ``tool_search_tool_result``
       blocks that are rejected by the Anthropic beta API.

    Args:
        messages: List of LangChain message objects.
        provider: ``"anthropic"`` (default) or ``"bedrock"``.

    Returns:
        List of messages with blocks in the correct format for the provider.
    """
    sanitized = []

    for msg in messages:
        if isinstance(msg, HumanMessage) and msg.content:
            if isinstance(msg.content, list):
                normalized_content = [
                    _normalize_frontend_content_block(block, provider=provider) for block in msg.content
                ]
                sanitized_msg = HumanMessage(
                    content=normalized_content,
                    id=msg.id,
                )
                sanitized.append(sanitized_msg)
            else:
                sanitized.append(msg)
        elif isinstance(msg, AIMessage) and msg.content:
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
            # Not an AIMessage or HumanMessage with list content, pass through unchanged
            sanitized.append(msg)

    return sanitized

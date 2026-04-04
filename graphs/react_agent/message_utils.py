"""Utilities for message processing and sanitization for Anthropic compatibility."""

import base64
import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

logger = logging.getLogger(__name__)

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
# Document extraction validation
# ---------------------------------------------------------------------------

# Minimum byte thresholds for uploaded content to be considered valid.
_MIN_CONTENT_BYTES = 50  # Below this, content is likely corrupt or empty
_LARGE_IMAGE_BYTES = 50_000  # Above this, an image is likely a phone camera shot


def validate_document_upload(data: str, mime_type: str, filename: str) -> tuple[bool, str | None]:
    """Validate that an uploaded document contains extractable content.

    Returns:
        (is_valid, warning_message) — ``True`` + ``None`` if content is usable.
        ``False`` + human-readable message if the content is corrupt or empty.
    """
    if not data:
        return False, (
            f"[Document upload failed for '{filename}' — no content received. "
            "Please try uploading again, or paste the text content directly.]"
        )

    try:
        raw_bytes = base64.b64decode(data)
    except Exception:
        return False, (
            f"[Document '{filename}' could not be decoded. "
            "The file may be corrupted. Please upload as PDF or paste text directly.]"
        )

    if len(raw_bytes) < _MIN_CONTENT_BYTES:
        return False, (
            f"[Document '{filename}' appears empty or too small ({len(raw_bytes)} bytes). "
            "Please check the file and try again.]"
        )

    return True, None


def _build_extraction_warning(mime_type: str, filename: str, raw_size: int) -> str | None:
    """Return a warning note if document extraction quality may be degraded.

    For example, phone camera shots of documents (large images) often produce
    poor text extraction results.
    """
    if mime_type.startswith("image/") and raw_size > _LARGE_IMAGE_BYTES:
        return (
            f"[Note: '{filename}' appears to be a photo of a document. "
            "For best results, consider uploading the original PDF or Word file, "
            "or pasting the text content directly into the chat.]"
        )
    return None


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

    # ---- Validate uploaded content before processing ----
    is_valid, error_msg = validate_document_upload(data, mime_type, filename)
    if not is_valid:
        logger.warning("Document upload validation failed: file=%s mime=%s", filename, mime_type)
        return {"type": "text", "text": error_msg}

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------
    if block_type == "image" and mime_type in _IMAGE_MIME_TYPES:
        # Check if this is likely a phone photo of a document
        try:
            raw_size = len(base64.b64decode(data))
        except Exception:
            raw_size = 0
        extraction_warning = _build_extraction_warning(mime_type, filename, raw_size)

        if provider == "bedrock":
            # langchain_aws reads block["source"]["mediaType"] (camelCase)
            result_block = {
                "type": "image",
                "source": {"type": "base64", "mediaType": mime_type, "data": data},
            }
        else:
            result_block = {
                "type": "image",
                "source": {"type": "base64", "media_type": mime_type, "data": data},
            }

        # If this is a large image (likely a document photo), return both
        # the image block AND a text warning so the model sees the note.
        if extraction_warning:
            logger.info("Document photo detected: file=%s size=%d", filename, raw_size)
            return [result_block, {"type": "text", "text": extraction_warning}]
        return result_block

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
                normalized_content = []
                for block in msg.content:
                    result = _normalize_frontend_content_block(block, provider=provider)
                    # _normalize_frontend_content_block may return a list of blocks
                    # (e.g., image + extraction warning) so flatten them
                    if isinstance(result, list):
                        normalized_content.extend(result)
                    else:
                        normalized_content.append(result)
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


# ---------------------------------------------------------------------------
# Tool result size management  (ported from Claude-code context-budget logic)
# ---------------------------------------------------------------------------

_TOOL_RESULT_PLACEHOLDER = "[Tool result truncated — too large]"
_OLD_TOOL_RESULT_PLACEHOLDER = (
    "<system-reminder>"
    "This tool result was cleared to free context space as the conversation grew. "
    "Do NOT re-call this tool — the information was already retrieved and used in "
    "an earlier response. Reconstruct from your conversation history or summary."
    "</system-reminder>"
)


def apply_tool_result_budget(
    messages: list[BaseMessage],
    max_chars: int = 50_000,
) -> list[BaseMessage]:
    """Truncate oversized ToolMessage content so it fits within *max_chars*.

    Very large tool results (e.g. full document dumps) can blow the context
    window on their own.  This caps each individual result to *max_chars*
    characters, replacing the tail with a placeholder notice.  Inspired by
    Claude-code's ``TOOL_RESULT_BUDGET`` pattern.

    Non-ToolMessage messages are returned unchanged.
    """
    out: list[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, ToolMessage) and isinstance(msg.content, str) and len(msg.content) > max_chars:
            truncated = msg.content[:max_chars] + f"\n\n{_TOOL_RESULT_PLACEHOLDER}"
            out.append(ToolMessage(content=truncated, tool_call_id=msg.tool_call_id, id=msg.id))
        else:
            out.append(msg)
    return out


def snip_old_tool_results(
    messages: list[BaseMessage],
    keep_last: int = 20,
) -> list[BaseMessage]:
    """Replace the content of old ToolMessages with a placeholder.

    Keeps the *keep_last* most-recent ToolMessages intact and replaces all
    earlier ones with ``_OLD_TOOL_RESULT_PLACEHOLDER``.  This frees context
    space consumed by accumulated tool results from many turns ago while
    preserving the message structure (tool_call_id pairing) so the model
    does not see orphaned tool calls.

    Inspired by Claude-code's ``snipOldToolResults`` utility.
    """
    tool_indices: list[int] = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]
    snip_up_to = len(tool_indices) - keep_last
    if snip_up_to <= 0:
        return messages

    snip_set = set(tool_indices[:snip_up_to])
    out: list[BaseMessage] = []
    for i, msg in enumerate(messages):
        if i in snip_set and isinstance(msg, ToolMessage):
            out.append(
                ToolMessage(
                    content=_OLD_TOOL_RESULT_PLACEHOLDER,
                    tool_call_id=msg.tool_call_id,
                    id=msg.id,
                )
            )
        else:
            out.append(msg)
    return out

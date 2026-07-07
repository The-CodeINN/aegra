"""Multi-tier context compaction system.

Ported from Claude-code's 4-tier compaction architecture:

Tier 0 — **Snip**: Free.  Replace old tool results with placeholders.
         Already implemented in ``message_utils.snip_old_tool_results``.

Tier 1 — **Microcompact**: Cheap.  Compress individual oversized message
         content blocks in-place using a lightweight model.

Tier 2 — **Autocompact**: Medium.  Full summarization pass on older message
         segments.  Already implemented via LangMem ``SummarizationNode``.

Tier 3 — **Context Collapse**: Expensive, last resort.  Emergency re-
         summarize of everything except the last N messages, stripping
         all media attachments.

Tier selection (spec Item 5): Tiers 1 and 2-3 are triggered independently and
are not redundant, despite superficially all being "compaction". Tier 2
(``_SUMMARY_TRIGGER_TOKENS`` in graph.py, raised from 6k to 16k) is the
proactive first line of defense, run before every ``call_model`` turn. Tier 1
is a second proactive check against the *real* context window ratio
(``select_compaction_tier`` below), catching whatever slips past Tier 2 —
in practice this rarely fires, since Tier 2's own token budget keeps
``prepared_messages`` well under Tier 1's threshold. Tier 3 (plus the
reactive retry of Tier 2) is a purely reactive last resort, only invoked when
a model call actually throws a context-overflow error. See ``state.py``'s
``has_attempted_reactive_compact`` docstring for why these need separate
guard flags rather than one shared flag.

Write-before-compact: ``consolidate_memories`` (graph.py) persists durable
facts and open tasks from a turn's ``state.messages`` snapshot as a
background task, independent of whatever the ``summarize`` node later does
to the message *window*. Recent raw messages are always kept verbatim by
``SummarizationNode`` (only the older prefix is ever compressed), so a fact
discussed in a still-recent turn is never lossy-summarized before
``consolidate_memories`` has had a chance to extract it. The one residual
gap is timing, not data loss: ``consolidate_memories`` is fire-and-forget,
so under unusually fast back-to-back turns (not normal human-paced
conversation) the background extraction for turn N might still be in
flight when turn N+1's compaction runs — benign today because compaction
only touches the aged-out prefix, but worth knowing before changing either
mechanism's timing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)


class CompactionTier(StrEnum):
    """Compaction tiers in order of cost."""

    SNIP = "snip"
    MICROCOMPACT = "microcompact"
    AUTOCOMPACT = "autocompact"
    COLLAPSE = "collapse"


@dataclass
class CompactionState:
    """Tracks compaction activity within a session for observability."""

    last_compact_turn: int = 0
    compact_count: int = 0
    total_tokens_saved: int = 0
    tier_used: CompactionTier | None = None


# ---------------------------------------------------------------------------
# Tier 1: Microcompact — compress oversized individual blocks
# ---------------------------------------------------------------------------

# Blocks larger than this will be compressed
_MICROCOMPACT_CHAR_THRESHOLD = 3000

# Target size after compression
_MICROCOMPACT_TARGET_CHARS = 1000

_MICROCOMPACT_PROMPT = """Compress the following content to approximately {target} characters.
Preserve all key facts, data points, and conclusions. Remove verbose explanations,
repeated information, and filler text. Return ONLY the compressed content — no preamble.

Content to compress:
{content}"""


async def microcompact_messages(
    messages: list[AnyMessage],
    model: Any,
    char_threshold: int = _MICROCOMPACT_CHAR_THRESHOLD,
    target_chars: int = _MICROCOMPACT_TARGET_CHARS,
) -> tuple[list[AnyMessage], int]:
    """Compress individual message content blocks that exceed the threshold.

    Targets ToolMessage results and long AIMessage content.
    HumanMessages are NEVER compressed (they contain the user's actual input).

    Args:
        messages: Full message list.
        model: A chat model for compression.
        char_threshold: Only compress blocks larger than this.
        target_chars: Target size after compression.

    Returns:
        (compressed_messages, chars_saved) — new message list + total chars freed.
    """
    out: list[AnyMessage] = []
    total_saved = 0

    for msg in messages:
        # Never compress user input
        if isinstance(msg, HumanMessage):
            out.append(msg)
            continue

        content = msg.content if isinstance(msg.content, str) else ""

        if not content or len(content) <= char_threshold:
            out.append(msg)
            continue

        # Compress the content
        try:
            response = await model.ainvoke(
                [
                    {
                        "role": "system",
                        "content": "You are a text compressor. Return only compressed content.",
                    },
                    {
                        "role": "user",
                        "content": _MICROCOMPACT_PROMPT.format(target=target_chars, content=content[:8000]),
                    },
                ],
                config=RunnableConfig(callbacks=[]),
            )
            compressed = response.content if hasattr(response, "content") else str(response)
            if isinstance(compressed, list):
                compressed = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in compressed)

            saved = len(content) - len(compressed)
            if saved > 0:
                total_saved += saved
                if isinstance(msg, ToolMessage):
                    out.append(
                        ToolMessage(
                            content=compressed,
                            tool_call_id=msg.tool_call_id,
                            id=msg.id,
                        )
                    )
                elif isinstance(msg, AIMessage):
                    out.append(
                        AIMessage(
                            content=compressed,
                            id=msg.id,
                            tool_calls=getattr(msg, "tool_calls", []),
                            response_metadata=getattr(msg, "response_metadata", {}),
                        )
                    )
                else:
                    out.append(msg)
            else:
                # Compression didn't help — keep original
                out.append(msg)
        except Exception:
            logger.warning("Microcompact failed for message %s — keeping original", msg.id)
            out.append(msg)

    if total_saved > 0:
        logger.info("Microcompact saved %d characters across messages", total_saved)

    return out, total_saved


# ---------------------------------------------------------------------------
# Tier 3: Context Collapse — emergency full re-summarize
# ---------------------------------------------------------------------------

_COLLAPSE_SUMMARY_PROMPT = """Summarise the following conversation history into a concise recap.
Focus on:
1. Key facts about the user (name, background, goals, current situation)
2. What tools were called and their key results
3. What advice or plans were discussed
4. Any commitments or next steps agreed upon

Keep it under {max_chars} characters. Return ONLY the summary — no preamble.

Conversation:
{conversation}"""

_COLLAPSE_MAX_SUMMARY_CHARS = 2000
_COLLAPSE_KEEP_LAST_N = 4


async def context_collapse(
    messages: list[AnyMessage],
    model: Any,
    keep_last_n: int = _COLLAPSE_KEEP_LAST_N,
    max_summary_chars: int = _COLLAPSE_MAX_SUMMARY_CHARS,
) -> list[AnyMessage]:
    """Emergency compaction: summarize everything except the last N messages.

    Strips all media attachments from older messages. This is the last resort
    before a prompt-too-long error.

    Args:
        messages: Full message list.
        model: A chat model for summarization.
        keep_last_n: Number of recent messages to keep intact.
        max_summary_chars: Maximum length of the generated summary.

    Returns:
        Collapsed message list: [summary_message] + last N messages.
    """
    if len(messages) <= keep_last_n:
        return messages

    old_messages = messages[:-keep_last_n]
    recent_messages = messages[-keep_last_n:]

    # Build a text representation of old messages for summarization
    conversation_parts: list[str] = []
    for msg in old_messages:
        role = "User" if isinstance(msg, HumanMessage) else ("Assistant" if isinstance(msg, AIMessage) else "Tool")
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        # Strip media content from the text representation
        if isinstance(msg.content, list):
            text_parts = []
            for block in msg.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            content = " ".join(text_parts)
        # Truncate individual messages to prevent the summary prompt from being too large
        conversation_parts.append(f"{role}: {content[:500]}")

    conversation_text = "\n".join(conversation_parts)

    try:
        response = await model.ainvoke(
            [
                {
                    "role": "system",
                    "content": "You are a conversation summarizer. Return only the summary.",
                },
                {
                    "role": "user",
                    "content": _COLLAPSE_SUMMARY_PROMPT.format(
                        max_chars=max_summary_chars,
                        conversation=conversation_text[:10000],
                    ),
                },
            ],
            config=RunnableConfig(callbacks=[]),
        )

        summary = response.content if hasattr(response, "content") else str(response)
        if isinstance(summary, list):
            summary = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in summary)

        summary_msg = HumanMessage(
            content=(f"[Context collapsed — earlier conversation summarized]\n\n{summary[:max_summary_chars]}")
        )

        logger.info(
            "Context collapse: %d old messages → summary (%d chars), keeping last %d messages",
            len(old_messages),
            len(summary),
            len(recent_messages),
        )

        return [summary_msg] + list(recent_messages)

    except Exception:
        logger.exception("Context collapse summarization failed — keeping recent messages only")
        # If summarization fails, just keep the recent messages with a note
        note = HumanMessage(
            content=(
                "[Earlier conversation context was cleared due to length. "
                "Key context may be missing — ask the user to re-state important details.]"
            )
        )
        return [note] + list(recent_messages)


# ---------------------------------------------------------------------------
# Tier selection: choose the cheapest sufficient tier
# ---------------------------------------------------------------------------


def estimate_token_count(messages: list[AnyMessage]) -> int:
    """Estimate the token count of a message list using LangChain's fast estimator."""
    return count_tokens_approximately(messages)


def select_compaction_tier(
    messages: list[AnyMessage],
    context_window: int = 200_000,
    *,
    has_attempted_microcompact: bool = False,
    has_attempted_collapse: bool = False,
) -> CompactionTier | None:
    """Determine which compaction tier is needed based on current token usage.

    Returns ``None`` if no compaction is needed.
    """
    tokens = estimate_token_count(messages)
    usage_ratio = tokens / context_window if context_window > 0 else 0

    # No compaction needed below 60% usage
    if usage_ratio < 0.6:
        return None

    # 60-75%: microcompact (if not already tried)
    if usage_ratio < 0.75:
        return CompactionTier.MICROCOMPACT if not has_attempted_microcompact else None

    # 75-90%: autocompact (handled by SummarizationNode — caller should check)
    if usage_ratio < 0.90:
        return CompactionTier.AUTOCOMPACT

    # 90%+: context collapse (emergency)
    if not has_attempted_collapse:
        return CompactionTier.COLLAPSE

    return None

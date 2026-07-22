"""Shared advisor memory-context assembly.

Extracted from ``call_model`` (graph.py) so the exact same fetching,
selection, and rendering logic can be reused by read-only outreach
generation (aegra_api's scheduler/notification_engine — spec Item 7, §9.1)
instead of a parallel reimplementation that could silently drift from what
the live agent actually shows the model.

Both callers pass a LangGraph ``BaseStore``-compatible object — the live
agent's ``runtime.store`` and aegra_api's ``db_manager.get_store()`` are the
same underlying ``AsyncPostgresStore`` instance, so this module has no
dependency on which process obtained it.
"""

import logging
import re
from typing import Any

from react_agent.memory import DEFAULT_MEMORY_NAMESPACE, memory_freshness_note

logger = logging.getLogger(__name__)


async def fetch_advisor_behavior_profile(store: Any, user_id: str | None) -> dict[str, Any] | None:
    """Return the student's most-recently-updated AdvisorBehaviorProfile's raw content, or None.

    "Most recently updated wins" is the same selection rule used everywhere
    this is read — changing it here changes it for both live chat and
    outreach at once, which is the point. Returns the raw ``content`` dict
    (``trait``/``evidence``/``how_to_apply``) rather than a rendered block:
    live chat renders it as an XML system-prompt block
    (``prompts.build_advisor_behavior_block``) while outreach renders it as a
    plain-text line for an email data summary — different targets, same
    fetch + selection.
    """
    if not store or not user_id:
        return None
    namespace = (user_id, DEFAULT_MEMORY_NAMESPACE)
    try:
        behavior_items = await store.asearch(namespace, filter={"kind": "AdvisorBehaviorProfile"}, limit=5)
        if not behavior_items:
            return None
        latest_behavior = max(
            behavior_items,
            key=lambda item: getattr(item, "updated_at", None) or getattr(item, "created_at", None) or "",
        )
        return latest_behavior.value.get("content", {})
    except Exception:
        logger.debug("Advisor behavior profile load failed; continuing without.", exc_info=True)
        return None


async def fetch_semantic_memory_block(store: Any, user_id: str | None, *, limit: int = 20) -> tuple[str, str]:
    """Return (raw_text, clean_block) for general semantic recall.

    Covers CareerGoal / StudentContext / FeedbackMemory / ReferenceMemory —
    durable facts about the student, excluding AdvisorBehaviorProfile (that
    has its own emphasized block via ``fetch_advisor_behavior_block``) and
    excluding identity/name content (never stored here — see the Identity
    rule in prompts.py and MEMORY_EXTRACTION_INSTRUCTIONS in memory.py).

    ``raw_text`` keeps per-item freshness ``<system-reminder>`` tags intact,
    for grounding checks (e.g. hallucination screening) that need to know a
    memory was flagged stale. ``clean_block`` strips them — the tags are for
    the caller's own reasoning, not something to hand the model directly, or
    it starts making false claims like "it's been a few days" from a tag it
    was never meant to narrate.
    """
    if not store or not user_id:
        return "", ""
    namespace = (user_id, DEFAULT_MEMORY_NAMESPACE)
    try:
        existing_memories = await store.asearch(namespace, limit=limit)
    except Exception:
        logger.debug("Semantic memory load failed; continuing without.", exc_info=True)
        return "", ""

    if not existing_memories:
        return "", ""

    memory_lines: list[str] = []
    for item in existing_memories:
        kind = item.value.get("kind", "unknown")
        if kind == "AdvisorBehaviorProfile":
            continue
        content = item.value.get("content", {})
        updated_at = getattr(item, "updated_at", None) or getattr(item, "created_at", None)
        freshness = memory_freshness_note(updated_at, kind=kind)
        summary = str(content) if isinstance(content, dict) else str(content)
        line = f"  [{kind}] {summary[:200]}"
        if freshness:
            line += f"\n  {freshness}"
        memory_lines.append(line)

    if not memory_lines:
        return "", ""

    raw_text = "\n".join(memory_lines)
    clean_block = re.sub(r"\n?\s*<system-reminder>.*?</system-reminder>", "", raw_text, flags=re.DOTALL)
    return raw_text, clean_block

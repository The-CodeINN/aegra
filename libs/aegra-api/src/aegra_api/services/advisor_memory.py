"""Read-only access to the advisor's LangGraph memory store for outreach.

Proactive outreach (agent optimisation spec Item 7, sections 9.1/9.3) must read
the SAME memory stack the live advisor uses — episodic memory for shared
history, the procedural behavior profile for tone, and durable semantic facts
(career goal, background) — without running a live agent turn. This module is
that read-only assembly path; it never writes.

Structural unification: the AdvisorBehaviorProfile fetch/selection and the
general semantic-memory fetch/rendering are NOT reimplemented here — they're
imported from ``react_agent.context_assembly``, the exact same functions
``call_model`` calls for live chat, so outreach can never silently drift from
what the live agent would actually say. Only the episodic-memory fetch has no
live-agent equivalent to share (the live agent recalls episodes on demand via
its own ``search_memory`` tool call, which outreach — having no live agent
turn — cannot do; it fetches one proactively instead).
"""

from __future__ import annotations

import logging
from typing import Any

from aegra_api.core.database import db_manager

logger = logging.getLogger(__name__)

# Mirrors DEFAULT_MEMORY_NAMESPACE in graphs/react_agent/memory.py — the
# namespace the live agent's LangMem tools and consolidation write to.
MEMORY_NAMESPACE_LABEL = "memories"

_EPISODE_CHAR_LIMIT = 400
_PROFILE_CHAR_LIMIT = 300


def _render_memory_content(content: Any, limit: int) -> str:
    """Flatten a stored memory's content payload into one prompt-safe line."""
    if isinstance(content, dict):
        parts = [f"{key}: {value}" for key, value in content.items() if value]
        return "; ".join(parts)[:limit]
    return str(content)[:limit]


async def _search_kind(store: Any, namespace: tuple[str, str], kind: str, *, query: str | None, limit: int) -> list:
    """Search one memory kind, degrading to recency order if semantic search fails."""
    try:
        return await store.asearch(namespace, filter={"kind": kind}, query=query, limit=limit)
    except Exception:
        if query is None:
            logger.debug("Store search failed for %s; skipping.", kind, exc_info=True)
            return []
    try:
        return await store.asearch(namespace, filter={"kind": kind}, limit=limit)
    except Exception:
        logger.debug("Store search failed for %s; skipping.", kind, exc_info=True)
        return []


async def fetch_advisor_memory_context(user_id: str, *, query: str | None = None) -> dict[str, str]:
    """Return advisor memory context for outreach personalisation.

    Keys: ``relevant_episode``, ``behavior_profile``, ``semantic_context``.

    ``query`` (e.g. overdue task descriptions + goal) drives semantic selection
    of the single most relevant episode — never the whole event log (the spec's
    isolation principle). Best-effort: empty strings when the store is
    unavailable or the student has no memories of a given type yet.
    """
    result = {"relevant_episode": "", "behavior_profile": "", "semantic_context": ""}
    if not user_id:
        return result

    try:
        store = db_manager.get_store()
    except RuntimeError:
        logger.debug("LangGraph store unavailable; outreach proceeds without advisor memory.")
        return result

    # Lazy import: graphs/ is only added to sys.path by aegra's dependency
    # loader (aegra.json `dependencies`), which may not have run yet at this
    # module's own import time — deferring avoids an import-order race.
    from react_agent.context_assembly import fetch_advisor_behavior_profile, fetch_semantic_memory_block

    namespace = (user_id, MEMORY_NAMESPACE_LABEL)

    episodes = await _search_kind(store, namespace, "EpisodicMemory", query=query, limit=1)
    if episodes:
        result["relevant_episode"] = _render_memory_content(episodes[0].value.get("content", {}), _EPISODE_CHAR_LIMIT)

    behavior_content = await fetch_advisor_behavior_profile(store, user_id)
    if behavior_content:
        result["behavior_profile"] = _render_memory_content(behavior_content, _PROFILE_CHAR_LIMIT)

    # Durable facts (CareerGoal/StudentContext/FeedbackMemory/ReferenceMemory)
    # the live agent has learned — previously unavailable to outreach, which
    # only had the static onboarding-form goal from Mongo, so a goal the
    # agent updated mid-conversation could never reach an email.
    _, semantic_block = await fetch_semantic_memory_block(store, user_id, limit=20)
    result["semantic_context"] = semantic_block

    return result

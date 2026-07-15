"""Read-only access to the advisor's LangGraph memory store for outreach.

Proactive outreach (agent optimisation spec Item 7, sections 9.1/9.3) must read
the SAME memory stack the live advisor uses — episodic memory for shared
history and the procedural behavior profile for tone — without running a live
agent turn. This module is that read-only assembly path; it never writes.
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
    """Return ``{"relevant_episode", "behavior_profile"}`` for outreach personalisation.

    ``query`` (e.g. overdue task descriptions + goal) drives semantic selection
    of the single most relevant episode — never the whole event log (the spec's
    isolation principle). Best-effort: empty strings when the store is
    unavailable or the student has no episodic/procedural memories yet.
    """
    result = {"relevant_episode": "", "behavior_profile": ""}
    if not user_id:
        return result

    try:
        store = db_manager.get_store()
    except RuntimeError:
        logger.debug("LangGraph store unavailable; outreach proceeds without advisor memory.")
        return result

    namespace = (user_id, MEMORY_NAMESPACE_LABEL)

    episodes = await _search_kind(store, namespace, "EpisodicMemory", query=query, limit=1)
    if episodes:
        result["relevant_episode"] = _render_memory_content(episodes[0].value.get("content", {}), _EPISODE_CHAR_LIMIT)

    # Most recent profile wins — same selection rule as the live agent's
    # proactive load in graphs/react_agent/graph.py.
    profiles = await _search_kind(store, namespace, "AdvisorBehaviorProfile", query=None, limit=5)
    if profiles:
        latest = max(
            profiles,
            key=lambda item: getattr(item, "updated_at", None) or getattr(item, "created_at", None) or "",
        )
        result["behavior_profile"] = _render_memory_content(latest.value.get("content", {}), _PROFILE_CHAR_LIMIT)

    return result

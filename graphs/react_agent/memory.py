"""Long-term memory schemas and constants for the agent.

Typed Pydantic schemas are shared between the hot-path tools
(create_manage_memory_tool / create_search_memory_tool) and the background
extraction node (create_memory_store_manager) so all three write to the same
structured store.

Memory type taxonomy (ported from Claude-code):
  - CareerGoal     — target roles the student is working toward
  - StudentContext — persistent facts about the student
  - FeedbackMemory — corrections and guidance the student has given the agent
  - ReferenceMemory — pointers to external resources the student has shared
"""

import logging
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_NAMESPACE = "memories"

# ---------------------------------------------------------------------------
# Memory freshness — ported from Claude-code's memoryAge.ts
# ---------------------------------------------------------------------------
# Models are poor at date arithmetic. A raw ISO timestamp does not trigger
# staleness reasoning the way "47 days ago" does.  We attach a human-readable
# age note to recalled memories so the agent knows to verify before asserting.
#
# Per-schema windows (spec Item 3) — a blanket 1-day threshold wrongly ages
# career goals and background facts that are stable for months. Different
# schemas age at different rates:
#   - CareerGoal / StudentContext: stable over months; re-confirm occasionally.
#   - FeedbackMemory: the student's stated preferences can shift as they improve.
#   - ReferenceMemory: not time-based at all — a URL either still works or it
#     doesn't; that's checked when read_webpage() actually uses it, not here.
#     ``None`` here means "never flag by age".
#   - EpisodicMemory: a past event stays a valid piece of history forever —
#     surfaced by semantic relevance to the current topic (search_memory),
#     not by recency. ``None`` — never flagged by age.
#   - AdvisorBehaviorProfile: a learned behavioral trait is as stable as
#     CareerGoal/StudentContext; same window.
# Unknown/unrecognised kinds fall back to the original conservative 1-day
# window rather than silently going stale-free.
_FRESHNESS_WINDOWS_DAYS: dict[str, int | None] = {
    "CareerGoal": 75,
    "StudentContext": 75,
    "FeedbackMemory": 30,
    "ReferenceMemory": None,
    "EpisodicMemory": None,
    "AdvisorBehaviorProfile": 75,
}
_DEFAULT_FRESHNESS_THRESHOLD_DAYS = 1

# Maximum number of memories per user before pruning kicks in.
MAX_MEMORIES_PER_USER = 200


def memory_age_days(updated_at: datetime | str | None) -> int:
    """Return floor-rounded days since the memory was last updated."""
    if updated_at is None:
        return 0
    if isinstance(updated_at, str):
        try:
            updated_at = datetime.fromisoformat(updated_at)
        except (ValueError, TypeError):
            return 0
    now = datetime.now(tz=UTC)
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=UTC)
    delta = (now - updated_at).days
    return max(0, delta)


def memory_freshness_note(updated_at: datetime | str | None, kind: str | None = None) -> str:
    """Return a staleness caveat for memories older than their schema's freshness window.

    ``kind`` selects the window from ``_FRESHNESS_WINDOWS_DAYS`` (e.g.
    "CareerGoal"); unrecognised or omitted kinds use the original 1-day
    default. A ``None`` window (currently only ``ReferenceMemory``) means
    this schema is never flagged by age — returns "" unconditionally.
    """
    threshold = _FRESHNESS_WINDOWS_DAYS.get(kind, _DEFAULT_FRESHNESS_THRESHOLD_DAYS)
    if threshold is None:
        return ""

    days = memory_age_days(updated_at)
    if days <= threshold:
        return ""
    return (
        f"<system-reminder>This memory is {days} days old. "
        "Memories are point-in-time observations — the student's situation "
        "may have changed. Verify with the student before asserting as current fact."
        "</system-reminder>"
    )


# ---------------------------------------------------------------------------
# Memory schemas
# ---------------------------------------------------------------------------


class CareerGoal(BaseModel):
    """A career goal or target role the student is actively working toward."""

    role: str = Field(description="The target job title or career role.")
    industry: str | None = Field(
        default=None,
        description="Industry or domain (e.g. 'fintech', 'healthcare AI').",
    )
    timeline: str | None = Field(
        default=None,
        description="Student's stated timeline or urgency (e.g. '6 months', 'as soon as possible').",
    )
    priority: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="Relative priority of this goal compared to other goals.",
    )


class StudentContext(BaseModel):
    """A persistent fact about the student's background, preferences, or constraints."""

    fact: str = Field(description="The specific fact to remember, written concisely.")
    category: Literal[
        "experience",
        "education",
        "skill",
        "preference",
        "constraint",
        "location",
        "motivation",
    ] = Field(description="The type of information this fact represents.")
    confidence: Literal["stated", "inferred"] = Field(
        default="stated",
        description="Whether the student stated this directly or the agent inferred it.",
    )


class FeedbackMemory(BaseModel):
    """Guidance the student has given about how the agent should approach work.

    Both corrections ("don't do X") and confirmations ("yes, keep doing that").
    Ported from Claude-code's feedback memory type with Why/HowToApply structure.
    """

    rule: str = Field(
        description="The core guidance or rule. E.g. 'Never fabricate CV content'.",
    )
    why: str | None = Field(
        default=None,
        description="The reason the student gave — often a past incident or strong preference.",
    )
    how_to_apply: str | None = Field(
        default=None,
        description="When and where this guidance kicks in. Helps judge edge cases.",
    )
    source: Literal["correction", "confirmation"] = Field(
        default="correction",
        description="Whether the student corrected a mistake or confirmed a good approach.",
    )


class ReferenceMemory(BaseModel):
    """A pointer to where information can be found in external systems.

    Stores locations of external resources the student has shared so the agent
    remembers where to look for up-to-date information.
    """

    resource: str = Field(
        description="What the resource is. E.g. 'LinkedIn profile', 'portfolio site'.",
    )
    location: str = Field(
        description="URL, path, or identifier for the resource.",
    )
    purpose: str | None = Field(
        default=None,
        description="Why this resource is relevant or when to consult it.",
    )


class EpisodicMemory(BaseModel):
    """A specific past event and how it felt — shared history, not a static fact.

    Distinct from ``StudentContext`` (durable background facts): this
    captures discrete moments — breakthroughs, setbacks, decisions — with
    situational and emotional context, so the advisor can reference "when
    you struggled with joins last month" rather than only ever recalling
    static profile facts. Extracted cold-path only (spec Item 4) — the agent
    doesn't log these itself mid-conversation; a background pass identifies
    which parts of the conversation were actually notable events.
    """

    event: str = Field(description="What happened. E.g. 'Failed module 3 SQL assessment, 2nd attempt'.")
    emotional_context: str | None = Field(
        default=None,
        description="How the student felt or reacted. E.g. 'Felt discouraged, considered pausing'.",
    )
    outcome: str | None = Field(
        default=None,
        description="What was decided or what helped. E.g. 'Agreed to redo joins practice before retrying'.",
    )


class AdvisorBehaviorProfile(BaseModel):
    """A learned pattern in how to mentor THIS specific student — procedural memory.

    Not who the student is (that's ``StudentContext``) and not an explicit
    correction they gave (that's ``FeedbackMemory``) — this is inferred from
    how the student has actually responded across conversations, e.g.
    "responds well to direct challenge" vs "shuts down under pressure".
    Extracted cold-path; injected proactively each turn (unlike episodic
    memory, which is recalled on demand via search_memory) so tone adapts
    from the very first message of a conversation.
    """

    trait: str = Field(
        description="The behavioral pattern observed. E.g. 'Responds well to direct challenge, not gentle framing'.",
    )
    evidence: str | None = Field(
        default=None,
        description="What in past conversations suggested this — keeps it grounded, not guessed.",
    )
    how_to_apply: str | None = Field(
        default=None,
        description="How this should shape tone or approach going forward.",
    )


# Shared schema list consumed by hot-path tools and background extractor.
MEMORY_SCHEMAS: list[type[BaseModel]] = [
    CareerGoal,
    StudentContext,
    FeedbackMemory,
    ReferenceMemory,
    EpisodicMemory,
    AdvisorBehaviorProfile,
]

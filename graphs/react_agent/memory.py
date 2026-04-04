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

_FRESHNESS_THRESHOLD_DAYS = 1  # Only warn for memories older than this

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


def memory_freshness_note(updated_at: datetime | str | None) -> str:
    """Return a staleness caveat for memories older than the threshold.

    Returns an empty string for fresh (≤1 day old) memories.
    """
    days = memory_age_days(updated_at)
    if days <= _FRESHNESS_THRESHOLD_DAYS:
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


# Shared schema list consumed by hot-path tools and background extractor.
MEMORY_SCHEMAS: list[type[BaseModel]] = [
    CareerGoal,
    StudentContext,
    FeedbackMemory,
    ReferenceMemory,
]

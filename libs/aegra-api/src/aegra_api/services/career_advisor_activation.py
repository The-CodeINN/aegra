"""Helpers for first-time Career Advisor activation behavior."""

from __future__ import annotations

import random
import re
from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.core.accountability_orm import UserPreferences
from aegra_api.core.orm import _get_session_maker

logger = structlog.get_logger()

ROADMAP_GENERATED_PREFERENCE_KEY = "career_roadmap_generated"

# Agent optimisation spec Item 6: A/B variant for returning students' roadmap
# structure. Assigned once per user, on first use, and persisted — a user
# must never flip variants mid-experiment.
ROADMAP_VARIANT_PREFERENCE_KEY = "roadmap_variant"
ROADMAP_VARIANTS = ("full", "adaptive")

FIRST_TIME_ROADMAP_REDIRECT_MESSAGE = (
    "Before we dive in, the best place to start is generating your personalised career roadmap "
    "- it uses your profile to show you exactly where to focus and what to learn next. Click "
    "'Generate my career roadmap' above, or just type it, and I'll build it for you right now."
)

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.strip().lower())


def is_career_roadmap_trigger(text: str | None) -> bool:
    """Return True when the message should trigger roadmap generation."""
    if not text or not text.strip():
        return False

    normalized = _normalize_text(text)
    return "career roadmap" in normalized


async def get_roadmap_generated_flag(session: AsyncSession, user_id: str) -> bool:
    """Load the persisted roadmap-generated flag for the user."""
    prefs = await session.scalar(select(UserPreferences).where(UserPreferences.user_id == user_id))
    if not prefs or not isinstance(prefs.preferences, dict):
        return False
    return bool(prefs.preferences.get(ROADMAP_GENERATED_PREFERENCE_KEY, False))


async def get_roadmap_generated_flag_for_user(user_id: str) -> bool:
    """Load the persisted roadmap-generated flag using a short-lived session."""
    maker = _get_session_maker()
    async with maker() as session:
        return await get_roadmap_generated_flag(session, user_id)


async def set_roadmap_generated_flag(session: AsyncSession, user_id: str, value: bool = True) -> None:
    """Persist the roadmap-generated flag for the user."""
    prefs = await session.scalar(select(UserPreferences).where(UserPreferences.user_id == user_id))
    if not prefs:
        prefs = UserPreferences(user_id=user_id)
        session.add(prefs)

    pref_json = dict(prefs.preferences or {})
    pref_json[ROADMAP_GENERATED_PREFERENCE_KEY] = value
    prefs.preferences = pref_json
    prefs.updated_at = datetime.now(UTC)
    await session.commit()


async def mark_roadmap_generated_for_user(user_id: str) -> None:
    """Best-effort helper used after a successful roadmap-generation run."""
    maker = _get_session_maker()
    async with maker() as session:
        await set_roadmap_generated_flag(session, user_id, True)
    logger.info("career_roadmap_generated_flag_set", user_id=user_id)


def pick_roadmap_variant(existing_preferences: dict[str, Any] | None) -> str:
    """Return the sticky variant from ``existing_preferences``, or randomly pick one.

    Pure — no DB access — so callers that already have a fetched
    ``UserPreferences`` row (e.g. ``WeeklyCheckinService.provision_user``,
    which writes it back as part of a larger combined commit) don't need a
    second round-trip through ``get_or_assign_roadmap_variant`` below.
    """
    existing = (existing_preferences or {}).get(ROADMAP_VARIANT_PREFERENCE_KEY)
    if existing in ROADMAP_VARIANTS:
        return existing
    return random.choice(ROADMAP_VARIANTS)  # nosec B311 — A/B experiment bucketing, not security-sensitive


async def get_or_assign_roadmap_variant(session: AsyncSession, user_id: str) -> str:
    """Return this user's roadmap A/B variant, assigning + persisting one (sticky) on first use.

    Only meaningful for returning students — the caller should only invoke
    this once ``roadmap_generated`` is already True for the user, since a
    variant assigned before their first roadmap has nothing to govern yet.
    """
    prefs = await session.scalar(select(UserPreferences).where(UserPreferences.user_id == user_id))
    variant = pick_roadmap_variant(prefs.preferences if prefs else None)
    if prefs and prefs.preferences and prefs.preferences.get(ROADMAP_VARIANT_PREFERENCE_KEY) == variant:
        return variant  # already persisted, nothing to write

    if not prefs:
        prefs = UserPreferences(user_id=user_id)
        session.add(prefs)

    pref_json = dict(prefs.preferences or {})
    pref_json[ROADMAP_VARIANT_PREFERENCE_KEY] = variant
    prefs.preferences = pref_json
    prefs.updated_at = datetime.now(UTC)
    await session.commit()
    logger.info("roadmap_variant_assigned", user_id=user_id, variant=variant)
    return variant


async def get_or_assign_roadmap_variant_for_user(user_id: str) -> str:
    """Load/assign the roadmap A/B variant using a short-lived session."""
    maker = _get_session_maker()
    async with maker() as session:
        return await get_or_assign_roadmap_variant(session, user_id)


def extract_latest_human_text(input_data: dict[str, Any] | None) -> str:
    """Extract the most recent human/user message text from run input."""
    if not isinstance(input_data, dict):
        return ""

    messages = input_data.get("messages")
    if not isinstance(messages, list):
        return ""

    for message in reversed(messages):
        if not isinstance(message, dict):
            continue

        role = message.get("role") or message.get("type")
        if role not in {"human", "user"}:
            continue

        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str) and text.strip():
                        text_parts.append(text.strip())
            return " ".join(text_parts).strip()

    return ""

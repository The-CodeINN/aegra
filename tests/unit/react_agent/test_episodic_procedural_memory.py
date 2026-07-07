"""Unit tests for episodic + procedural memory (agent optimisation spec Item 4)."""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GRAPHS_ROOT = PROJECT_ROOT / "graphs"
if str(GRAPHS_ROOT) not in sys.path:
    sys.path.insert(0, str(GRAPHS_ROOT))

from react_agent import prompts  # noqa: E402
from react_agent.memory import (  # noqa: E402
    MEMORY_SCHEMAS,
    AdvisorBehaviorProfile,
    EpisodicMemory,
    memory_freshness_note,
)


def _days_ago(days: int) -> datetime:
    return datetime.now(tz=UTC) - timedelta(days=days)


def test_episodic_and_behavior_profile_are_registered_schemas() -> None:
    assert EpisodicMemory in MEMORY_SCHEMAS
    assert AdvisorBehaviorProfile in MEMORY_SCHEMAS


def test_episodic_memory_requires_only_event() -> None:
    memory = EpisodicMemory(event="Failed module 3 SQL assessment, 2nd attempt")
    assert memory.emotional_context is None
    assert memory.outcome is None


def test_advisor_behavior_profile_requires_only_trait() -> None:
    profile = AdvisorBehaviorProfile(trait="Responds well to direct challenge")
    assert profile.evidence is None


def test_episodic_memory_never_flagged_by_age() -> None:
    """Old events stay valid as history — surfaced by relevance, not recency."""
    assert memory_freshness_note(_days_ago(400), kind="EpisodicMemory") == ""


def test_advisor_behavior_profile_uses_75_day_window() -> None:
    assert memory_freshness_note(_days_ago(10), kind="AdvisorBehaviorProfile") == ""
    assert "days old" in memory_freshness_note(_days_ago(90), kind="AdvisorBehaviorProfile")


def test_build_advisor_behavior_block_empty_without_trait() -> None:
    assert prompts.build_advisor_behavior_block({}) == ""


def test_build_advisor_behavior_block_renders_trait_only() -> None:
    block = prompts.build_advisor_behavior_block({"trait": "Prefers concrete steps over theory"})
    assert "Prefers concrete steps over theory" in block
    assert "Evidence:" not in block


def test_build_advisor_behavior_block_renders_full_profile() -> None:
    block = prompts.build_advisor_behavior_block(
        {
            "trait": "Shuts down under pressure",
            "evidence": "Went quiet when challenged directly on missed deadline",
            "how_to_apply": "Use gentle framing, break into smaller steps",
        }
    )
    assert "Shuts down under pressure" in block
    assert "Went quiet when challenged directly" in block
    assert "gentle framing" in block


def test_decision_autonomy_directive_is_in_static_block() -> None:
    static_prompt = prompts.get_dynamic_system_prompt(roadmap_generated=True)
    static_block = prompts.build_runtime_system_prompt(static_prompt)
    assert "<decision_autonomy>" in static_block
    assert "<advisor_behavior_profile>" in static_block  # referenced, not just task ledger

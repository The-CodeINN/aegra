"""Unit tests for adaptive roadmap A/B selection (agent optimisation spec Item 6)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GRAPHS_ROOT = PROJECT_ROOT / "graphs"
if str(GRAPHS_ROOT) not in sys.path:
    sys.path.insert(0, str(GRAPHS_ROOT))

from react_agent import prompts  # noqa: E402
from react_agent.context import Context  # noqa: E402

_SEVEN_PART_MARKER = "NON-NEGOTIABLE"
_ADAPTIVE_MARKER = "has been here before"


def test_returning_student_full_variant_gets_seven_part_structure() -> None:
    prompt = prompts.get_dynamic_system_prompt(roadmap_generated=True, roadmap_variant="full")
    assert _SEVEN_PART_MARKER in prompt
    assert _ADAPTIVE_MARKER not in prompt


def test_returning_student_adaptive_variant_gets_loose_structure() -> None:
    prompt = prompts.get_dynamic_system_prompt(roadmap_generated=True, roadmap_variant="adaptive")
    assert _ADAPTIVE_MARKER in prompt
    assert _SEVEN_PART_MARKER not in prompt


def test_first_time_student_always_gets_full_structure_regardless_of_variant() -> None:
    """The one hard guarantee: a first-time student never sees the adaptive structure."""
    prompt = prompts.get_dynamic_system_prompt(roadmap_generated=False, roadmap_variant="adaptive")
    assert _SEVEN_PART_MARKER in prompt
    assert _ADAPTIVE_MARKER not in prompt


def test_missing_variant_defaults_to_full_structure() -> None:
    prompt = prompts.get_dynamic_system_prompt(roadmap_generated=True, roadmap_variant=None)
    assert _SEVEN_PART_MARKER in prompt


def test_unrecognised_variant_value_falls_back_to_full_structure() -> None:
    prompt = prompts.get_dynamic_system_prompt(roadmap_generated=True, roadmap_variant="not-a-real-variant")
    assert _SEVEN_PART_MARKER in prompt


def test_adaptive_structure_references_task_ledger_and_episodes() -> None:
    prompt = prompts.get_dynamic_system_prompt(roadmap_generated=True, roadmap_variant="adaptive")
    assert "open_tasks" in prompt
    assert "episodic memory" in prompt.lower() or "search_memory" in prompt


def test_context_defaults_roadmap_variant_to_none() -> None:
    ctx = Context(user_id="u1", roadmap_generated=True, lms_api_url="http://localhost:9")
    assert ctx.roadmap_variant is None
    assert _SEVEN_PART_MARKER in ctx.system_prompt


def test_context_threads_roadmap_variant_into_system_prompt() -> None:
    ctx = Context(
        user_id="u1",
        roadmap_generated=True,
        roadmap_variant="adaptive",
        lms_api_url="http://localhost:9",
    )
    assert _ADAPTIVE_MARKER in ctx.system_prompt

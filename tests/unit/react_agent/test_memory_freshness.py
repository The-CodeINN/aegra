"""Unit tests for per-category memory freshness (agent optimisation spec Item 3).

Before this, a single blanket 1-day threshold applied to every memory
schema — wrongly ageing career goals and background facts that are stable
for months. These tests pin the per-schema windows.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GRAPHS_ROOT = PROJECT_ROOT / "graphs"
if str(GRAPHS_ROOT) not in sys.path:
    sys.path.insert(0, str(GRAPHS_ROOT))

from react_agent.memory import memory_freshness_note  # noqa: E402


def _days_ago(days: int) -> datetime:
    return datetime.now(tz=UTC) - timedelta(days=days)


def test_three_day_old_career_goal_carries_no_stale_warning() -> None:
    """The exact regression the spec calls out: Monday->Thursday must not go stale."""
    note = memory_freshness_note(_days_ago(3), kind="CareerGoal")
    assert note == ""


def test_career_goal_stale_after_75_days() -> None:
    note = memory_freshness_note(_days_ago(80), kind="CareerGoal")
    assert "80 days old" in note


def test_student_context_uses_same_75_day_window() -> None:
    assert memory_freshness_note(_days_ago(10), kind="StudentContext") == ""
    assert "days old" in memory_freshness_note(_days_ago(90), kind="StudentContext")


def test_feedback_memory_stale_after_30_days() -> None:
    assert memory_freshness_note(_days_ago(20), kind="FeedbackMemory") == ""
    assert "days old" in memory_freshness_note(_days_ago(35), kind="FeedbackMemory")


def test_reference_memory_never_flagged_by_age() -> None:
    """URLs are re-verified on use (read_webpage), not by elapsed time."""
    assert memory_freshness_note(_days_ago(1000), kind="ReferenceMemory") == ""


def test_unrecognised_kind_falls_back_to_conservative_one_day_window() -> None:
    assert memory_freshness_note(_days_ago(2), kind="SomeUnknownSchema") != ""


def test_omitted_kind_falls_back_to_conservative_one_day_window() -> None:
    assert memory_freshness_note(_days_ago(2)) != ""
    assert memory_freshness_note(_days_ago(1)) == ""

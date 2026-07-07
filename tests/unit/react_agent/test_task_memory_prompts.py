from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GRAPHS_ROOT = PROJECT_ROOT / "graphs"
if str(GRAPHS_ROOT) not in sys.path:
    sys.path.insert(0, str(GRAPHS_ROOT))

from react_agent import prompts  # noqa: E402


def test_build_open_tasks_block_returns_empty_string_when_nothing_open() -> None:
    assert prompts.build_open_tasks_block(None) == ""
    assert prompts.build_open_tasks_block({"not_yet_due": [], "overdue": []}) == ""


def test_build_open_tasks_block_flags_miss_count_two_for_escalation() -> None:
    block = prompts.build_open_tasks_block(
        {
            "not_yet_due": [],
            "overdue": [{"description": "Rebuild README", "due_date": "2026-06-20", "miss_count": 2}],
        }
    )
    assert "Rebuild README" in block
    assert "do NOT assign this again" in block


def test_build_open_tasks_block_does_not_escalate_first_miss() -> None:
    block = prompts.build_open_tasks_block(
        {
            "not_yet_due": [],
            "overdue": [{"description": "Apply to 2 roles", "due_date": "2026-06-25", "miss_count": 1}],
        }
    )
    assert "Apply to 2 roles" in block
    assert "do NOT assign this again" not in block
    assert "ask what blocked it" in block


def test_build_open_tasks_block_lists_not_yet_due_without_escalation_language() -> None:
    block = prompts.build_open_tasks_block(
        {
            "not_yet_due": [{"description": "Practice SQL joins", "due_date": "2026-07-10"}],
            "overdue": [],
        }
    )
    assert "Practice SQL joins" in block
    assert "OVERDUE" not in block


def test_task_decision_logic_is_included_in_static_block() -> None:
    static_prompt = prompts.get_dynamic_system_prompt(roadmap_generated=True)
    static_block = prompts.build_runtime_system_prompt(static_prompt)
    assert "<task_decision_logic>" in static_block
    assert "miss_count >= 2" in static_block


def test_build_dynamic_prompt_block_includes_open_tasks_block() -> None:
    block = prompts.build_dynamic_prompt_block(
        system_time="2026-07-03T09:00:00+00:00",
        open_tasks_block="<open_tasks>foo</open_tasks>",
    )
    assert "<open_tasks>foo</open_tasks>" in block

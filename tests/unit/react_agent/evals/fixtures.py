"""Golden-conversation fixture loading for the agent eval harness (spec Item 0).

Fixtures live as JSON files under ``tests/fixtures/golden_conversations/``.
Each fixture is a ``(messages, context)`` pair plus optional seed memories
and named assertions to run against the agent's final response — see
``replay.py`` for how a fixture is executed and ``test_golden_conversations.py``
for how assertions are checked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FIXTURES_DIR = Path(__file__).resolve().parents[3] / "fixtures" / "golden_conversations"


@dataclass(kw_only=True)
class SeedMemory:
    """A long-term memory item to pre-load into the store before replay."""

    kind: str
    content: dict[str, Any]


@dataclass(kw_only=True)
class ConversationFixture:
    """One golden conversation: input messages/context plus expected properties."""

    name: str
    description: str
    context: dict[str, Any]
    messages: list[dict[str, str]]
    seed_memories: list[SeedMemory] = field(default_factory=list)
    assertions: list[str] = field(default_factory=list)
    source_file: Path | None = None


def _load_one(path: Path) -> ConversationFixture:
    raw = json.loads(path.read_text(encoding="utf-8"))
    seed_memories = [SeedMemory(kind=m["kind"], content=m["content"]) for m in raw.get("seed_memories", [])]
    return ConversationFixture(
        name=raw["name"],
        description=raw.get("description", ""),
        context=raw.get("context", {}),
        messages=raw["messages"],
        seed_memories=seed_memories,
        assertions=raw.get("assertions", []),
        source_file=path,
    )


def load_all_fixtures(directory: Path = FIXTURES_DIR) -> list[ConversationFixture]:
    """Load every ``*.json`` golden-conversation fixture, sorted by filename.

    Returns an empty list (rather than raising) when the directory doesn't
    exist yet, so pytest collection doesn't fail before any fixtures are added.
    """
    if not directory.is_dir():
        return []
    return [_load_one(p) for p in sorted(directory.glob("*.json"))]

"""Unit tests for the roadmap A/B variant assignment (agent optimisation spec Item 6)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aegra_api.services.career_advisor_activation import (
    ROADMAP_VARIANT_PREFERENCE_KEY,
    get_or_assign_roadmap_variant,
    pick_roadmap_variant,
)


class TestPickRoadmapVariant:
    """Pure function — no DB access, used by WeeklyCheckinService.provision_user
    to avoid a second round-trip when it already has a fetched pref row."""

    def test_returns_existing_variant(self) -> None:
        assert pick_roadmap_variant({ROADMAP_VARIANT_PREFERENCE_KEY: "adaptive"}) == "adaptive"
        assert pick_roadmap_variant({ROADMAP_VARIANT_PREFERENCE_KEY: "full"}) == "full"

    def test_picks_a_variant_when_none_exists(self) -> None:
        assert pick_roadmap_variant(None) in {"full", "adaptive"}
        assert pick_roadmap_variant({}) in {"full", "adaptive"}

    def test_ignores_unrecognised_stored_value(self) -> None:
        assert pick_roadmap_variant({ROADMAP_VARIANT_PREFERENCE_KEY: "bogus"}) in {"full", "adaptive"}


def _make_session(*, existing_prefs: dict | None) -> AsyncMock:
    session = AsyncMock()
    prefs = None
    if existing_prefs is not None:
        prefs = MagicMock()
        prefs.preferences = existing_prefs
    session.scalar = AsyncMock(return_value=prefs)
    return session


class TestGetOrAssignRoadmapVariant:
    @pytest.mark.asyncio
    async def test_assigns_a_variant_when_none_exists(self) -> None:
        session = _make_session(existing_prefs=None)

        variant = await get_or_assign_roadmap_variant(session, "user-1")

        assert variant in {"full", "adaptive"}
        session.add.assert_called_once()
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_existing_variant_without_reassigning(self) -> None:
        session = _make_session(existing_prefs={ROADMAP_VARIANT_PREFERENCE_KEY: "adaptive"})

        variant = await get_or_assign_roadmap_variant(session, "user-1")

        assert variant == "adaptive"
        session.add.assert_not_called()
        session.commit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_existing_full_variant_is_also_respected(self) -> None:
        session = _make_session(existing_prefs={ROADMAP_VARIANT_PREFERENCE_KEY: "full"})

        variant = await get_or_assign_roadmap_variant(session, "user-1")

        assert variant == "full"
        session.commit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_assignment_is_stable_across_repeated_calls(self) -> None:
        """Simulates two sequential calls sharing persisted state — must not flip."""
        prefs = {}
        session = _make_session(existing_prefs=None)

        async def commit_side_effect() -> None:
            # Simulate the write actually landing in "prefs" for the next call.
            call_kwargs = session.add.call_args
            written = call_kwargs.args[0] if call_kwargs and call_kwargs.args else None
            if written is not None:
                prefs.update(written.preferences)

        session.commit.side_effect = commit_side_effect

        first_variant = await get_or_assign_roadmap_variant(session, "user-1")

        # Second call sees the persisted preference.
        session2 = _make_session(existing_prefs=prefs)
        second_variant = await get_or_assign_roadmap_variant(session2, "user-1")

        assert first_variant == second_variant

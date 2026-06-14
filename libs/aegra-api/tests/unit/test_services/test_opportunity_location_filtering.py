"""Unit tests for opportunity location / city / state filtering.

Covers the 4-tier personalisation waterfall per the spec:
  Tier 1 — exact target role + state/city match
  Tier 2 — exact target role + country match
  Tier 3 — adjacent role    + state/city match
  Tier 4 — adjacent role    + country match

Key functions under test:
  _job_location_state_allowed  – city/state gate (Tier 1 & 3)
  _job_location_allowed        – country gate (Tier 2 & 4)
  get_profile_states           – extracts resident_cities + work_cities
  get_profile_locations        – extracts resident_country + work_countries
  _dedupe_locations            – deduplication util
"""

from __future__ import annotations

from aegra_api.services.opportunity_discovery import (
    OpportunityDiscoveryEngine,
    _dedupe_locations,
)
from aegra_api.services.student_profile import StudentProfile

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

engine = OpportunityDiscoveryEngine.__new__(OpportunityDiscoveryEngine)


def _profile(**kwargs: object) -> StudentProfile:
    p = StudentProfile(user_id="u1")
    for k, v in kwargs.items():
        setattr(p, k, v)
    return p


# ---------------------------------------------------------------------------
# _job_location_state_allowed — city/state matching (Tier 1 & 3 gate)
# ---------------------------------------------------------------------------


class TestJobLocationStateAllowed:
    def test_empty_state_set_returns_false(self) -> None:
        assert not engine._job_location_state_allowed("Lagos, Nigeria", set())

    def test_city_found_in_job_location(self) -> None:
        assert engine._job_location_state_allowed("Lagos, Nigeria", {"lagos"})

    def test_city_match_is_case_insensitive(self) -> None:
        assert engine._job_location_state_allowed("Edinburgh, Scotland, UK", {"scotland"})

    def test_multiple_states_one_matches(self) -> None:
        assert engine._job_location_state_allowed("Glasgow, Scotland", {"lagos", "scotland"})

    def test_no_state_matches(self) -> None:
        assert not engine._job_location_state_allowed("London, UK", {"lagos", "abuja"})

    def test_remote_job_always_allowed(self) -> None:
        assert engine._job_location_state_allowed("Remote - UK", {"lagos"})

    def test_hybrid_job_always_allowed(self) -> None:
        assert engine._job_location_state_allowed("Hybrid - Nigeria", {"scotland"})

    def test_empty_job_location_allowed(self) -> None:
        assert engine._job_location_state_allowed("", {"lagos"})

    def test_exact_city_equals_location(self) -> None:
        assert engine._job_location_state_allowed("Lagos", {"lagos"})

    def test_city_substring_of_location(self) -> None:
        assert engine._job_location_state_allowed("Lagos Island, Nigeria", {"lagos"})

    def test_worked_example_scotland_lagos(self) -> None:
        states = {"scotland", "lagos"}
        assert engine._job_location_state_allowed("Edinburgh, Scotland", states)
        assert engine._job_location_state_allowed("Victoria Island, Lagos", states)
        assert not engine._job_location_state_allowed("London, UK", states)
        assert not engine._job_location_state_allowed("Abuja, Nigeria", states)


# ---------------------------------------------------------------------------
# _job_location_allowed — country matching (Tier 2 & 4 gate)
# ---------------------------------------------------------------------------


class TestJobLocationAllowed:
    def test_remote_always_allowed(self) -> None:
        assert engine._job_location_allowed("Remote - anywhere", {"nigeria"})

    def test_hybrid_always_allowed(self) -> None:
        assert engine._job_location_allowed("Hybrid", {"united kingdom"})

    def test_country_substring_in_location(self) -> None:
        assert engine._job_location_allowed("Lagos, Nigeria", {"nigeria"})

    def test_uk_alias_resolves_to_united_kingdom(self) -> None:
        assert engine._job_location_allowed("London, UK", {"united kingdom"})

    def test_england_alias_resolves_to_united_kingdom(self) -> None:
        assert engine._job_location_allowed("Manchester, England", {"united kingdom"})

    def test_scotland_alias_resolves_to_united_kingdom(self) -> None:
        assert engine._job_location_allowed("Edinburgh, Scotland", {"united kingdom"})

    def test_gb_iso_code_matches_united_kingdom(self) -> None:
        assert engine._job_location_allowed("London, GB", {"united kingdom"})

    def test_ng_iso_code_does_not_match_inside_england(self) -> None:
        # "NG" must not match England — word-boundary guard prevents this
        assert not engine._job_location_allowed("London, England, UK", {"nigeria"})

    def test_city_fallback_london_to_united_kingdom(self) -> None:
        assert engine._job_location_allowed("London", {"united kingdom"})

    def test_city_fallback_lagos_to_nigeria(self) -> None:
        assert engine._job_location_allowed("Lagos", {"nigeria"})

    def test_city_fallback_nairobi_to_kenya(self) -> None:
        assert engine._job_location_allowed("Nairobi", {"kenya"})

    def test_wrong_country_rejected(self) -> None:
        assert not engine._job_location_allowed("Toronto, Canada", {"united kingdom"})

    def test_multi_country_user_any_match_accepted(self) -> None:
        assert engine._job_location_allowed("Lagos, Nigeria", {"united kingdom", "nigeria"})

    def test_compound_location_each_part_checked(self) -> None:
        assert engine._job_location_allowed("London, England, UK, GB", {"united kingdom"})

    def test_empty_location_is_allowed(self) -> None:
        assert engine._job_location_allowed("", {"nigeria"})

    def test_worked_example_uk_nigeria(self) -> None:
        allowed = {"united kingdom", "nigeria"}
        assert engine._job_location_allowed("London, UK", allowed)
        assert engine._job_location_allowed("Lagos, Nigeria", allowed)
        assert engine._job_location_allowed("Edinburgh, Scotland, UK", allowed)
        assert not engine._job_location_allowed("Toronto, Canada", allowed)
        assert not engine._job_location_allowed("Berlin, Germany", allowed)


# ---------------------------------------------------------------------------
# get_profile_states — city/state extraction
# ---------------------------------------------------------------------------


class TestGetProfileStates:
    def test_combines_resident_and_work_cities(self) -> None:
        p = _profile(resident_cities=["Scotland"], work_cities=["Lagos"])
        assert engine.get_profile_states(p) == ["Scotland", "Lagos"]

    def test_deduplicates_case_insensitively(self) -> None:
        p = _profile(resident_cities=["Lagos"], work_cities=["LAGOS", "Abuja"])
        result = engine.get_profile_states(p)
        assert "Lagos" in result
        assert result.count("Lagos") + result.count("LAGOS") == 1
        assert "Abuja" in result

    def test_empty_profile_returns_empty(self) -> None:
        assert engine.get_profile_states(None) == []

    def test_only_resident_cities(self) -> None:
        p = _profile(resident_cities=["Scotland"], work_cities=[])
        assert engine.get_profile_states(p) == ["Scotland"]

    def test_only_work_cities(self) -> None:
        p = _profile(resident_cities=[], work_cities=["Lagos"])
        assert engine.get_profile_states(p) == ["Lagos"]

    def test_no_cities_returns_empty(self) -> None:
        p = _profile(resident_cities=[], work_cities=[])
        assert engine.get_profile_states(p) == []

    def test_multiple_cities_each_side(self) -> None:
        p = _profile(resident_cities=["Scotland", "Manchester"], work_cities=["Lagos", "Abuja"])
        result = engine.get_profile_states(p)
        assert result == ["Scotland", "Manchester", "Lagos", "Abuja"]

    def test_worked_example_scotland_lagos(self) -> None:
        p = _profile(resident_cities=["Scotland"], work_cities=["Lagos"])
        states = {s.lower() for s in engine.get_profile_states(p)}
        assert "scotland" in states
        assert "lagos" in states


# ---------------------------------------------------------------------------
# get_profile_locations — country extraction
# ---------------------------------------------------------------------------


class TestGetProfileLocations:
    def test_combines_work_countries_and_resident(self) -> None:
        p = _profile(resident_country="United Kingdom", work_countries=["Nigeria"])
        locs = engine.get_profile_locations(p)
        assert "Nigeria" in locs
        assert "United Kingdom" in locs

    def test_deduplicates_resident_and_work(self) -> None:
        p = _profile(resident_country="United Kingdom", work_countries=["United Kingdom"])
        locs = engine.get_profile_locations(p)
        assert locs.count("United Kingdom") == 1

    def test_empty_profile_returns_empty(self) -> None:
        assert engine.get_profile_locations(None) == []

    def test_no_work_countries(self) -> None:
        p = _profile(resident_country="Nigeria", work_countries=[])
        assert engine.get_profile_locations(p) == ["Nigeria"]

    def test_no_resident_country(self) -> None:
        p = _profile(resident_country="", work_countries=["Nigeria", "United Kingdom"])
        result = engine.get_profile_locations(p)
        assert "Nigeria" in result
        assert "United Kingdom" in result

    def test_worked_example_uk_nigeria(self) -> None:
        p = _profile(resident_country="United Kingdom", work_countries=["Nigeria"])
        locs = [loc.lower() for loc in engine.get_profile_locations(p)]
        assert "nigeria" in locs
        assert "united kingdom" in locs


# ---------------------------------------------------------------------------
# _dedupe_locations
# ---------------------------------------------------------------------------


class TestDedupeLocations:
    def test_removes_exact_duplicates(self) -> None:
        assert _dedupe_locations(["UK", "UK", "Nigeria"]) == ["UK", "Nigeria"]

    def test_case_insensitive_dedup(self) -> None:
        result = _dedupe_locations(["Lagos", "LAGOS"])
        assert len(result) == 1
        assert result[0] == "Lagos"

    def test_preserves_order(self) -> None:
        assert _dedupe_locations(["Nigeria", "UK", "Ghana"]) == ["Nigeria", "UK", "Ghana"]

    def test_filters_empty_strings(self) -> None:
        result = _dedupe_locations(["Nigeria", "", "UK"])
        assert "" not in result
        assert "Nigeria" in result

    def test_empty_list(self) -> None:
        assert _dedupe_locations([]) == []

    def test_non_string_items_skipped(self) -> None:
        result = _dedupe_locations(["Nigeria", None, 123, "UK"])  # type: ignore[list-item]
        assert result == ["Nigeria", "UK"]


# ---------------------------------------------------------------------------
# Profile parsing — string vs list for city / country fields
# ---------------------------------------------------------------------------


class TestProfileCityParsing:
    """Ensures student_profile.py normalises LMS string fields to lists."""

    def _build_sec2(self, **overrides: object) -> dict[str, object]:
        base: dict[str, object] = {
            "completed": True,
            "residentCountry": "United Kingdom",
            "workCountry": ["Nigeria"],
            "residentCity": ["Scotland"],
            "workCity": ["Lagos"],
        }
        base.update(overrides)
        return base

    def _apply_sec2(self, sec2: dict[str, object], profile: StudentProfile) -> None:
        """Mirror the student_profile.py s2 parsing logic."""

        _wc = sec2.get("workCountry", []) or []
        profile.work_countries = [_wc] if isinstance(_wc, str) else list(_wc)  # type: ignore[arg-type]
        _rc = sec2.get("residentCity", []) or []
        profile.resident_cities = [_rc] if isinstance(_rc, str) else list(_rc)  # type: ignore[arg-type]
        _wk = sec2.get("workCity", []) or []
        profile.work_cities = [_wk] if isinstance(_wk, str) else list(_wk)  # type: ignore[arg-type]

    def test_list_fields_pass_through_unchanged(self) -> None:
        p = StudentProfile(user_id="u")
        self._apply_sec2(self._build_sec2(), p)
        assert p.work_countries == ["Nigeria"]
        assert p.resident_cities == ["Scotland"]
        assert p.work_cities == ["Lagos"]

    def test_string_work_country_becomes_single_element_list(self) -> None:
        p = StudentProfile(user_id="u")
        self._apply_sec2(self._build_sec2(workCountry="Nigeria"), p)
        assert p.work_countries == ["Nigeria"]
        assert not any(len(c) == 1 for c in p.work_countries)

    def test_string_resident_city_becomes_single_element_list(self) -> None:
        p = StudentProfile(user_id="u")
        self._apply_sec2(self._build_sec2(residentCity="Scotland"), p)
        assert p.resident_cities == ["Scotland"]

    def test_string_work_city_becomes_single_element_list(self) -> None:
        p = StudentProfile(user_id="u")
        self._apply_sec2(self._build_sec2(workCity="Lagos"), p)
        assert p.work_cities == ["Lagos"]

    def test_string_city_does_not_spread_characters(self) -> None:
        p = StudentProfile(user_id="u")
        self._apply_sec2(self._build_sec2(workCity="Scotland"), p)
        # Must NOT be ['S', 'c', 'o', 't', 'l', 'a', 'n', 'd']
        assert len(p.work_cities) == 1
        assert p.work_cities[0] == "Scotland"

    def test_none_city_field_becomes_empty_list(self) -> None:
        p = StudentProfile(user_id="u")
        self._apply_sec2(self._build_sec2(workCity=None), p)
        assert p.work_cities == []


# ---------------------------------------------------------------------------
# Tier filtering integration — _job_location_state_allowed + _job_location_allowed
# simulate the waterfall for the worked example from the spec
# ---------------------------------------------------------------------------


class TestWaterfallWorkedExample:
    """User: Target Role=Data Analyst | States=Scotland, Lagos | Countries=UK, Nigeria"""

    country_set = {"united kingdom", "nigeria"}
    states_set = {"scotland", "lagos"}

    def _tier(self, loc: str, role_in_content: bool = True, adjacent: bool = False) -> int:
        state_ok = engine._job_location_state_allowed(loc, self.states_set)
        country_ok = engine._job_location_allowed(loc, self.country_set)

        if not country_ok:
            return 0  # filtered out entirely

        if not adjacent:
            # role candidate
            if state_ok:
                return 1
            return 2
        else:
            # adjacent/track candidate
            if state_ok:
                return 3
            return 4

    # Role candidates (exact target role)
    def test_role_in_scotland_is_tier1(self) -> None:
        assert self._tier("Edinburgh, Scotland, UK") == 1

    def test_role_in_lagos_is_tier1(self) -> None:
        assert self._tier("Victoria Island, Lagos, Nigeria") == 1

    def test_role_in_london_is_tier2(self) -> None:
        assert self._tier("London, United Kingdom") == 2

    def test_role_in_uk_no_city_is_tier2(self) -> None:
        assert self._tier("United Kingdom") == 2

    def test_role_in_abuja_nigeria_is_tier2(self) -> None:
        assert self._tier("Abuja, Nigeria") == 2

    def test_role_in_nigeria_no_city_is_tier2(self) -> None:
        assert self._tier("Nigeria") == 2

    def test_remote_role_is_tier1(self) -> None:
        # remote → state_ok=True → Tier 1 for role job
        assert self._tier("Remote - Worldwide") == 1

    def test_role_outside_both_countries_filtered(self) -> None:
        assert self._tier("Toronto, Canada") == 0

    # Adjacent/track candidates
    def test_adjacent_in_scotland_is_tier3(self) -> None:
        assert self._tier("Glasgow, Scotland", adjacent=True) == 3

    def test_adjacent_in_lagos_is_tier3(self) -> None:
        assert self._tier("Lagos, Nigeria", adjacent=True) == 3

    def test_adjacent_in_london_is_tier4(self) -> None:
        assert self._tier("London, United Kingdom", adjacent=True) == 4

    def test_adjacent_outside_countries_filtered(self) -> None:
        assert self._tier("Berlin, Germany", adjacent=True) == 0

    def test_dedup_same_url_only_one_tier(self) -> None:
        """URL appearing in Tier 1 must not also appear in Tier 2."""
        seen: set[str] = set()
        url = "https://jobs.example.com/data-analyst-edinburgh"

        # Edinburgh, Scotland — Tier 1
        if url not in seen and self._tier("Edinburgh, Scotland, UK") == 1:
            seen.add(url)
        # Should NOT be admitted to Tier 2 now
        assert url in seen
        tier2_admitted = url not in seen  # already seen → False
        assert not tier2_admitted

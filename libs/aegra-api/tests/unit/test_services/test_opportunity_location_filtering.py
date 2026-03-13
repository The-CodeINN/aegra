"""Unit tests for opportunity location compatibility filtering."""

from aegra_api.services.opportunity_discovery import _is_location_compatible


def test_remote_role_is_accepted_for_any_location() -> None:
    assert _is_location_compatible("United Kingdom", "Data Engineer", "Remote, anywhere in Europe")


def test_matching_country_is_accepted() -> None:
    assert _is_location_compatible("GB", "Junior Data Analyst", "London, United Kingdom")


def test_contradictory_country_is_rejected() -> None:
    assert not _is_location_compatible("United Kingdom", "Data Engineer", "Toronto, Canada")

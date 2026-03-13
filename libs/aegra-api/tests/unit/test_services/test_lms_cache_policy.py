"""Unit tests for LMS cache TTL policy."""

from aegra_api.services.lms_cache import _ttl_for_path


def test_ttl_profile_is_short() -> None:
    assert _ttl_for_path("/api/v1/user/profile") == 120


def test_ttl_enrollment_overview_is_short() -> None:
    assert _ttl_for_path("/api/v1/enrollment/student/blackboard") == 45


def test_ttl_critical_structure_is_live_only() -> None:
    assert _ttl_for_path("/api/v1/enrollment/course-1/structure") == 0


def test_ttl_critical_progress_is_live_only() -> None:
    assert _ttl_for_path("/api/v1/enrollment/course-1/progress") == 0


def test_ttl_subscription_is_live_only() -> None:
    assert _ttl_for_path("/api/v1/subscription/me") == 0

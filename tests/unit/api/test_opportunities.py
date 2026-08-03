from datetime import UTC, datetime

from aegra_api.api.opportunities import _increment_scan_log, _scan_count_for_day, _scan_day_key


def test_scan_day_key_uses_client_timezone() -> None:
    now = datetime(2026, 4, 15, 23, 58, tzinfo=UTC)

    assert _scan_day_key(now=now) == "2026-04-15"
    assert _scan_day_key(now=now, timezone_name="Africa/Lagos") == "2026-04-16"


def test_scan_count_uses_client_local_day_boundary() -> None:
    now = datetime(2026, 4, 15, 23, 58, tzinfo=UTC)
    scan_log = {"2026-04-15": 4}

    assert _scan_count_for_day(scan_log, now=now) == 4
    assert _scan_count_for_day(scan_log, now=now, timezone_name="Africa/Lagos") == 0


def test_increment_scan_log_records_under_client_local_day() -> None:
    now = datetime(2026, 4, 15, 23, 58, tzinfo=UTC)

    updated = _increment_scan_log(
        {"2026-04-15": 4},
        now=now,
        timezone_name="Africa/Lagos",
    )

    assert updated == {"2026-04-15": 4, "2026-04-16": 1}

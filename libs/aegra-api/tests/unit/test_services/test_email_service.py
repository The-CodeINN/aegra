from __future__ import annotations

import pytest

from aegra_api.services import email_service


@pytest.mark.asyncio
async def test_resolve_student_contact_uses_student_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(email_service.settings.app, "LMS_URL", "https://dedatahub-api.vercel.app")
    monkeypatch.setattr(email_service.settings.app, "ADMIN_TOKEN", "admin-token")

    calls: list[str] = []

    async def fake_cached_lms_fetch(client, url: str, token: str, user_id: str) -> dict:
        calls.append(url)
        assert token == "admin-token"
        assert user_id == "student-123"
        if url.endswith("/api/v1/user/students/student-123"):
            return {
                "student": {
                    "name": "Richard Emijere",
                    "email": "emijere.richard@gmail.com",
                }
            }
        return {}

    monkeypatch.setattr(email_service, "cached_lms_fetch", fake_cached_lms_fetch)

    contact = await email_service.resolve_student_contact("student-123")

    assert contact == {
        "first_name": "Richard",
        "email": "emijere.richard@gmail.com",
    }
    assert calls == ["https://dedatahub-api.vercel.app/api/v1/user/students/student-123"]


@pytest.mark.asyncio
async def test_resolve_student_contact_returns_empty_when_student_lookup_has_no_email(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(email_service.settings.app, "LMS_URL", "https://dedatahub-api.vercel.app")
    monkeypatch.setattr(email_service.settings.app, "ADMIN_TOKEN", "admin-token")

    calls: list[str] = []

    async def fake_cached_lms_fetch(client, url: str, token: str, user_id: str) -> dict:
        calls.append(url)
        assert url.endswith("/api/v1/user/students/student-456")
        return {"student": {"name": "Dev"}}

    monkeypatch.setattr(email_service, "cached_lms_fetch", fake_cached_lms_fetch)

    contact = await email_service.resolve_student_contact("student-456")

    assert contact == {}
    assert calls == ["https://dedatahub-api.vercel.app/api/v1/user/students/student-456"]

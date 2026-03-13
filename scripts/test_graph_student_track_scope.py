"""Strict student-token graph scope test.

This script uses ONLY real student tokens (same auth mode as frontend).
No JWT minting and no admin fallback path.

Inputs:
- --student-token <jwt> (repeatable)
- --student-credential email:password (repeatable, used only to obtain real student token)
- STUDENT_TEST_TOKENS='["jwt1", "jwt2"]'
- STUDENT_TEST_CREDENTIALS='[{"email":"a@b.com","password":"secret"}]'
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any

import httpx
from dotenv import load_dotenv


def _load_env() -> None:
    load_dotenv()


@dataclass
class StudentSession:
    token: str
    user_id: str
    email: str | None


def _pick(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = data.get(key)
        if value is not None:
            return value
    return None


def _extract_run_text(output_payload: Any) -> str:
    if isinstance(output_payload, str):
        return output_payload
    if not isinstance(output_payload, dict):
        return json.dumps(output_payload, default=str)

    messages = output_payload.get("messages")
    if isinstance(messages, list):
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        parts.append(part["text"])
                if parts:
                    return "\n".join(parts)

    for key in ("output", "response", "text", "answer"):
        value = output_payload.get(key)
        if isinstance(value, str):
            return value

    return json.dumps(output_payload, default=str)


def _parse_credential_pair(raw: str) -> tuple[str, str]:
    if ":" not in raw:
        raise ValueError("Credential must be email:password")
    email, password = raw.split(":", 1)
    email = email.strip()
    password = password.strip()
    if not email or not password:
        raise ValueError("Credential must include non-empty email and password")
    return email, password


def _load_env_credentials() -> list[tuple[str, str]]:
    raw = os.getenv("STUDENT_TEST_CREDENTIALS", "").strip()
    if not raw:
        return []
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise ValueError("STUDENT_TEST_CREDENTIALS must be a JSON array")
    out: list[tuple[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        email = item.get("email")
        password = item.get("password")
        if isinstance(email, str) and isinstance(password, str) and email.strip() and password.strip():
            out.append((email.strip(), password.strip()))
    return out


def _load_env_tokens() -> list[str]:
    raw = os.getenv("STUDENT_TEST_TOKENS", "").strip()
    if not raw:
        return []
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise ValueError("STUDENT_TEST_TOKENS must be a JSON array")
    return [t.strip() for t in parsed if isinstance(t, str) and t.strip()]


async def _student_login(lms_url: str, email: str, password: str) -> str:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{lms_url.rstrip('/')}/api/v1/auth/login",
            json={"email": email, "password": password},
        )
        response.raise_for_status()
        payload = response.json()
    token = payload.get("token")
    if not isinstance(token, str) or not token.strip():
        raise RuntimeError(f"Login succeeded but token missing for {email}")
    return token


async def _lms_profile(lms_url: str, token: str) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {token}", "accept": "*/*"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{lms_url.rstrip('/')}/api/v1/user/profile", headers=headers)
        response.raise_for_status()
        payload = response.json()
    return payload if isinstance(payload, dict) else {}


async def _lms_onboarding(lms_url: str, token: str) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {token}", "accept": "*/*"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{lms_url.rstrip('/')}/api/v1/onboarding", headers=headers)
        response.raise_for_status()
        payload = response.json()
    return payload if isinstance(payload, dict) else {}


async def _lms_subscription(lms_url: str, token: str) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {token}", "accept": "*/*"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{lms_url.rstrip('/')}/api/v1/subscription/me", headers=headers)
        response.raise_for_status()
        payload = response.json()
    return payload if isinstance(payload, dict) else {}


async def _lms_enrollments(lms_url: str, token: str) -> list[dict[str, Any]]:
    headers = {"Authorization": f"Bearer {token}", "accept": "*/*"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{lms_url.rstrip('/')}/api/v1/enrollment/active", headers=headers)
        response.raise_for_status()
        payload = response.json()
    if isinstance(payload, dict) and isinstance(payload.get("enrollments"), list):
        return [item for item in payload["enrollments"] if isinstance(item, dict)]
    return []


async def _graph_wait_run(graph_url: str, token: str, assistant_id: str, prompt: str) -> str:
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=180.0) as client:
        thread_resp = await client.post(f"{graph_url.rstrip('/')}/threads", json={}, headers=headers)
        thread_resp.raise_for_status()
        thread_id = thread_resp.json().get("thread_id")
        if not isinstance(thread_id, str):
            raise RuntimeError("Graph /threads did not return thread_id")

        run_resp = await client.post(
            f"{graph_url.rstrip('/')}/threads/{thread_id}/runs/wait",
            json={
                "assistant_id": assistant_id,
                "input": {"messages": [{"role": "user", "content": prompt}]},
            },
            headers=headers,
        )
        run_resp.raise_for_status()
        return _extract_run_text(run_resp.json())


def _looks_blocked(text: str) -> bool:
    lowered = text.lower()
    markers = [
        "not enrolled",
        "outside your track",
        "do not have access",
        "don't have access",
        "cannot access",
        "can't access",
    ]
    return any(marker in lowered for marker in markers)


async def main() -> int:
    _load_env()

    parser = argparse.ArgumentParser(description="Strict student-token track scope test")
    parser.add_argument("--lms-url", default=os.getenv("LMS_URL", "https://dedatahub-api.vercel.app"))
    parser.add_argument("--graph-url", default=os.getenv("GRAPH_URL", "http://localhost:8000"))
    parser.add_argument("--assistant-id", default="agent")
    parser.add_argument("--student-credential", action="append", default=[])
    parser.add_argument("--student-token", action="append", default=[])
    args = parser.parse_args()

    credentials: list[tuple[str, str]] = []
    for raw in args.student_credential:
        credentials.append(_parse_credential_pair(raw))
    credentials.extend(_load_env_credentials())

    direct_tokens = [t.strip() for t in args.student_token if t and t.strip()]
    direct_tokens.extend(_load_env_tokens())

    if not credentials and not direct_tokens:
        raise RuntimeError(
            "No student token source provided. Use --student-token or --student-credential, "
            "or set STUDENT_TEST_TOKENS/STUDENT_TEST_CREDENTIALS."
        )

    sessions: list[StudentSession] = []

    for email, password in credentials:
        token = await _student_login(args.lms_url, email, password)
        profile = await _lms_profile(args.lms_url, token)
        user = profile.get("user") if isinstance(profile, dict) and isinstance(profile.get("user"), dict) else profile
        if not isinstance(user, dict):
            raise RuntimeError(f"Student login returned token but /user/profile failed for {email}")
        user_id = str(_pick(user, "_id", "id", "userId") or "").strip()
        if not user_id:
            raise RuntimeError(f"Student profile missing user id for {email}")
        sessions.append(StudentSession(token=token, user_id=user_id, email=email))

    for token in direct_tokens:
        profile = await _lms_profile(args.lms_url, token)
        user = profile.get("user") if isinstance(profile, dict) and isinstance(profile.get("user"), dict) else profile
        if not isinstance(user, dict):
            raise RuntimeError("Provided student token does not resolve /api/v1/user/profile")
        user_id = str(_pick(user, "_id", "id", "userId") or "").strip()
        if not user_id:
            raise RuntimeError("Provided student token profile missing user id")
        email = _pick(user, "email")
        sessions.append(StudentSession(token=token, user_id=user_id, email=email if isinstance(email, str) else None))

    print(f"Using LMS URL: {args.lms_url}")
    print(f"Using Graph URL: {args.graph_url}")

    report: list[dict[str, Any]] = []
    for sess in sessions:
        onboarding = await _lms_onboarding(args.lms_url, sess.token)
        subscription = await _lms_subscription(args.lms_url, sess.token)
        enrollments = await _lms_enrollments(args.lms_url, sess.token)

        onboard = onboarding.get("onboarding") if isinstance(onboarding, dict) else {}
        if not isinstance(onboard, dict):
            onboard = {}

        learning_track = _pick(subscription, "track")
        if not isinstance(learning_track, str) or not learning_track.strip():
            learning_track = _pick(onboard, "learningTrack")
        learning_track = learning_track.strip() if isinstance(learning_track, str) else None

        enrolled_titles: list[str] = []
        for enrollment in enrollments:
            title = _pick(enrollment, "courseTitle", "title", "courseName")
            if isinstance(title, str) and title.strip():
                enrolled_titles.append(title.strip())

        in_scope_target = enrolled_titles[0] if enrolled_titles else (learning_track or "my enrolled track")
        known_tracks = ["data analytics", "data science", "data engineering", "ai engineering"]
        out_scope_track = next(
            (t for t in known_tracks if not learning_track or t != learning_track.lower()), "cybersecurity"
        )

        in_prompt = f"I am enrolled in {in_scope_target}. Give me guidance for my enrolled path only."
        out_prompt = (
            f"I want detailed guidance for {out_scope_track}. "
            "If that is outside my enrolled track, refuse and redirect me to my own track."
        )

        in_answer = await _graph_wait_run(args.graph_url, sess.token, args.assistant_id, in_prompt)
        out_answer = await _graph_wait_run(args.graph_url, sess.token, args.assistant_id, out_prompt)

        report.append(
            {
                "user_id": sess.user_id,
                "email": sess.email,
                "learning_track": learning_track,
                "enrolled_titles": enrolled_titles,
                "in_scope_preview": in_answer[:350],
                "out_scope_preview": out_answer[:350],
                "out_scope_blocked": _looks_blocked(out_answer),
            }
        )

    print(json.dumps({"results": report}, indent=2, ensure_ascii=True))

    leaks = [row for row in report if not row["out_scope_blocked"]]
    if leaks:
        print("\nPOTENTIAL_SCOPE_LEAKS_DETECTED: true")
        return 2

    print("\nPOTENTIAL_SCOPE_LEAKS_DETECTED: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

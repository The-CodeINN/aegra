"""Test that the graph correctly retrieves and uses course content.

Usage (server must be running on localhost:8000):
    uv run python scripts/test_graph_course_content.py

The script:
1. Logs in a real student via the LMS.
2. Sends a prompt asking for specific course material or concept explanation.
3. Prints the agent's response to verify it pulled data from the local course DB.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import httpx
import jwt
from dotenv import load_dotenv

load_dotenv()

# -- Config -------------------------------------------------------------------
LMS_URL = os.getenv("LMS_URL", "https://dedatahub-api.vercel.app")
GRAPH_URL = os.getenv("GRAPH_URL", "http://localhost:8000")
ASSISTANT_ID = "agent"
JWT_SECRET = os.getenv("LMS_JWT_SECRET", "")


def mint_graph_token(user_id: str, email: str, name: str, role: str) -> str:
    if not JWT_SECRET:
        raise RuntimeError("LMS_JWT_SECRET not set in .env - cannot mint graph token")
    payload = {
        "userId": user_id,
        "id": user_id,
        "email": email,
        "name": name,
        "role": role,
        "iat": int(time.time()),
        "exp": int(time.time()) + 86400,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


@dataclass
class StudentData:
    email: str
    token: str
    lms_token: str
    user_id: str
    name: str


def _extract_run_text(payload: Any) -> str:
    """Pull the final assistant text from a /runs/wait response."""
    if isinstance(payload, str):
        return payload
    if not isinstance(payload, dict):
        return json.dumps(payload, default=str)

    messages = payload.get("messages")
    if isinstance(messages, list):
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [p["text"] for p in content if isinstance(p, dict) and isinstance(p.get("text"), str)]
                if parts:
                    return "\n".join(parts)

    for key in ("output", "response", "text", "answer"):
        val = payload.get(key)
        if isinstance(val, str):
            return val

    return json.dumps(payload, default=str)


async def lms_login(email: str, password: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(f"{LMS_URL}/api/v1/auth/login", json={"email": email, "password": password})
        r.raise_for_status()
        return r.json()


async def graph_run(auth_token: str, lms_token: str, prompt: str) -> str:
    """Create a thread and run a prompt through the graph, returning the final text."""
    headers = {"Authorization": f"Bearer {auth_token}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=180.0) as client:
        # Create thread
        thread_r = await client.post(f"{GRAPH_URL}/threads", json={}, headers=headers)
        thread_r.raise_for_status()
        thread_id = thread_r.json().get("thread_id")
        if not thread_id:
            raise RuntimeError("Graph /threads did not return thread_id")

        # Run and wait
        run_r = await client.post(
            f"{GRAPH_URL}/threads/{thread_id}/runs/wait",
            json={
                "assistant_id": ASSISTANT_ID,
                "input": {"messages": [{"role": "user", "content": prompt}]},
                "config": {"configurable": {"user_token": lms_token}},
            },
            headers=headers,
        )
        run_r.raise_for_status()
        return _extract_run_text(run_r.json())


async def main() -> int:
    # 1. Login
    email = os.getenv("TEST_STUDENT_EMAIL", "").strip()
    password = os.getenv("TEST_STUDENT_PASSWORD", "").strip()
    if not email or not password:
        raise RuntimeError("Set TEST_STUDENT_EMAIL and TEST_STUDENT_PASSWORD in environment")

    print("\n============================================================")
    print(f"  Logging in: {email}")
    print("============================================================")

    login_data = await lms_login(email, password)
    token = login_data["token"]
    user = login_data.get("user", {})
    user_id = str(user.get("id") or user.get("_id") or user.get("userId"))
    name = user.get("name", "Unknown")
    role = user.get("role", "student")

    graph_token = mint_graph_token(user_id=user_id, email=email, name=name, role=role)

    print(f"  [OK] Logged in as {name} (ID: {user_id})")

    # 2. Query Graph about Course Content
    # We ask a specific technical question that should trigger a course content search
    prompt = (
        "I'm reviewing the recent modules in my Data Science course about SQL. "
        "Can you explain the main concepts regarding SQL JOINs based on the course materials? "
        "Outline what I should know from the material about Inner vs Left joins."
    )

    print(f"\n  -> PROMPT: {prompt}")
    t0 = time.time()
    answer = await graph_run(graph_token, token, prompt)
    elapsed = time.time() - t0

    print(f"\n  [OK] Response ({elapsed:.1f}s):")
    print("-" * 60)
    print(answer)
    print("-" * 60)

    return 0


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        with contextlib.suppress(AttributeError):
            sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(asyncio.run(main()))

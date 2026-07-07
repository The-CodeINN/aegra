"""Export real conversation threads as golden-conversation eval fixtures (spec Item 0).

Pulls thread histories from a running Aegra server via the LangGraph SDK and
writes them into the fixture JSON format consumed by
``tests/unit/react_agent/evals/replay.py``. Output is written to a scratch
directory by default — review and redact PII (names, emails, LinkedIn/GitHub
URLs) before moving any fixture into
``tests/fixtures/golden_conversations/``.

Usage (server must be running, e.g. via `aegra dev` or against staging):
    uv run python scripts/export_golden_conversations.py \
        --graph-url http://localhost:8000 \
        --limit 30 \
        --out-dir /tmp/golden_export

Requires an admin/service JWT with permission to list threads across users —
set LMS_ADMIN_TOKEN or pass --token. This script only reads state; it never
mutates threads.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langgraph_sdk import get_client

logger = logging.getLogger(__name__)

_DEFAULT_GRAPH_URL = "http://localhost:8000"
_DEFAULT_LIMIT = 30

# Only export threads with at least this many human turns — very short
# threads make poor regression fixtures (nothing to replay continuity on).
_MIN_HUMAN_TURNS = 2

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_URL_RE = re.compile(r"https?://\S+")


def _redact(text: str) -> str:
    """Best-effort PII scrub — emails and URLs. Still requires human review."""
    text = _EMAIL_RE.sub("[redacted-email]", text)
    return _URL_RE.sub("[redacted-url]", text)


def _extract_role_content(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for msg in messages:
        msg_type = msg.get("type") or msg.get("role")
        role = "user" if msg_type in {"human", "user"} else "assistant" if msg_type in {"ai", "assistant"} else None
        if role is None:
            continue
        content = msg.get("content")
        text = (
            content
            if isinstance(content, str)
            else " ".join(b.get("text", "") for b in content if isinstance(b, dict))
            if isinstance(content, list)
            else ""
        )
        if not text.strip():
            continue
        out.append({"role": role, "content": _redact(text)})
    return out


async def export_threads(
    *,
    graph_url: str,
    limit: int,
    out_dir: Path,
    token: str | None,
) -> None:
    headers = {"Authorization": f"Bearer {token}"} if token else None
    client = get_client(url=graph_url, headers=headers)

    threads = await client.threads.search(limit=limit)
    out_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    for thread in threads:
        thread_id = thread["thread_id"]
        state = await client.threads.get_state(thread_id)
        values = state.get("values", {})
        messages = _extract_role_content(values.get("messages", []))

        human_turns = sum(1 for m in messages if m["role"] == "user")
        if human_turns < _MIN_HUMAN_TURNS:
            logger.info("Skipping thread %s — only %d human turn(s)", thread_id, human_turns)
            continue

        fixture = {
            "name": f"exported_{thread_id[:8]}",
            "description": (
                "Exported from a real conversation via export_golden_conversations.py. "
                "REVIEW AND REDACT before committing — verify no PII remains."
            ),
            "context": {
                "user_id": f"eval-exported-{thread_id[:8]}",
                "lms_api_url": "http://localhost:9",
            },
            "seed_memories": [],
            "messages": messages,
            "assertions": ["no_fabricated_linkedin_github", "no_system_prompt_leak"],
        }

        out_path = out_dir / f"exported_{thread_id[:8]}.json"
        out_path.write_text(json.dumps(fixture, indent=2), encoding="utf-8")
        exported += 1
        logger.info("Wrote %s (%d messages)", out_path, len(messages))

    logger.info("Exported %d fixture(s) to %s", exported, out_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-url", default=os.getenv("GRAPH_URL", _DEFAULT_GRAPH_URL))
    parser.add_argument("--limit", type=int, default=_DEFAULT_LIMIT)
    parser.add_argument("--out-dir", type=Path, default=Path(tempfile.gettempdir()) / "golden_export")
    parser.add_argument("--token", default=os.getenv("LMS_ADMIN_TOKEN"))
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_dotenv()
    args = _parse_args()
    asyncio.run(
        export_threads(
            graph_url=args.graph_url,
            limit=args.limit,
            out_dir=args.out_dir,
            token=args.token,
        )
    )


if __name__ == "__main__":
    main()

"""Verify local-first course content setup.

Checks:
- DATABASE_URL connectivity
- LMS authentication
- Presence of local course-content tables
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

project_root = Path(__file__).resolve().parent.parent
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file)

sys.path.insert(0, str(project_root / "libs" / "aegra-api" / "src"))

from aegra_api.tools.course_content.lms_client import LMSClient  # noqa: E402


def check_database() -> bool:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        user = os.getenv("POSTGRES_USER")
        password = os.getenv("POSTGRES_PASSWORD")
        host = os.getenv("POSTGRES_HOST")
        port = os.getenv("POSTGRES_PORT")
        name = os.getenv("POSTGRES_DB")
        if all([user, password, host, port, name]):
            database_url = f"postgresql+psycopg://{user}:{password}@{host}:{port}/{name}"

    if not database_url:
        print("DATABASE_URL not configured")
        return False

    if "asyncpg" in database_url:
        database_url = database_url.replace("postgresql+asyncpg://", "postgresql+psycopg://")

    engine = create_engine(database_url)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            tables = [
                "local_courses",
                "local_lessons",
                "local_materials",
                "transcript_segments",
                "course_content_sync_state",
                "course_content_events",
            ]
            for table_name in tables:
                exists = conn.execute(
                    text("SELECT to_regclass(:table_name)"),
                    {"table_name": table_name},
                ).scalar()
                print(f"{table_name}: {'ok' if exists else 'missing'}")
        return True
    except Exception as exc:
        print(f"database check failed: {exc}")
        return False
    finally:
        engine.dispose()


async def check_lms() -> bool:
    try:
        client = LMSClient()
        courses = await client.get_all_courses()
        print(f"lms auth ok: {len(courses)} courses visible")
        return True
    except Exception as exc:
        print(f"lms check failed: {exc}")
        return False


async def main() -> int:
    db_ok = check_database()
    lms_ok = await check_lms()
    return 0 if db_ok and lms_ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

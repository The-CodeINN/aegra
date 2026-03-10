"""
Scheduler smoke test: seeds a test opportunity + enables job mail for the
"Dev" user, then calls generate_daily_digest() directly to verify the full
 scheduler → LMS lookup → SMTP delivery pipeline.
"""

import asyncio
import os
import sys

# Windows: asyncpg requires SelectorEventLoop
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Point Python at the local package source
sys.path.insert(0, r"C:\Users\thecodeinn\Documents\dedata\ai-service\libs\aegra-api\src")
os.chdir(r"C:\Users\thecodeinn\Documents\dedata\ai-service")

from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

from aegra_api.core.database import db_manager  # noqa: E402
from aegra_api.services.scheduler import scheduler_service  # noqa: E402
from aegra_api.settings import settings  # noqa: E402

USER_ID = "68c30006cc08c47f660b1941"
USER_EMAIL = "emijere.richard@gmail.com"
USER_NAME = "Richard"
TEST_TITLE = "Scheduler smoke test opportunity"
TEST_CONTENT = "This opportunity digest was created to verify the DeDataHub scheduler email flow."

UPSERT_SQL = """
INSERT INTO user_preferences (user_id, notifications_enabled, preferences, updated_at)
VALUES (
    :user_id,
    true,
    jsonb_build_object(
        'job_opportunity_mail_enabled', true,
        'job_opportunity_mail_frequency', 'daily',
        'user_email', cast(:user_email as text),
        'user_name', cast(:user_name as text)
    ),
    now()
)
ON CONFLICT (user_id) DO UPDATE
    SET notifications_enabled = true,
        preferences = (COALESCE(user_preferences.preferences, '{}'::jsonb)
            || jsonb_build_object(
                'job_opportunity_mail_enabled', true,
                'job_opportunity_mail_frequency', 'daily',
                'user_email', cast(:user_email as text),
                'user_name', cast(:user_name as text)
            )) - 'last_job_opportunity_digest_sent_at',
        updated_at = now()
"""

INSERT_NOTIF_SQL = """
INSERT INTO notifications
    (user_id, title, content, channel, priority, status, category,
     metadata, action_buttons, scheduled_at, created_at)
VALUES
    (:user_id, :title, :content,
     'in_app', 'normal', 'pending', 'opportunity',
     '{}'::jsonb, '[]'::jsonb, now(), now())
"""

CHECK_SQL = """
SELECT title, delivered_at, sent_at
FROM notifications
WHERE user_id = :user_id AND title = :title
ORDER BY created_at DESC
LIMIT 1
"""

PREF_SQL = """
SELECT preferences::text
FROM user_preferences
WHERE user_id = :user_id
"""


async def main():
    engine = create_async_engine(settings.db.database_url, pool_pre_ping=True)
    db_manager.engine = engine
    try:
        print("=== Seeding test data ===")
        async with engine.begin() as conn:
            await conn.execute(text(UPSERT_SQL), {"user_id": USER_ID, "user_email": USER_EMAIL, "user_name": USER_NAME})
            print(f"  ✓ user_preferences updated for {USER_ID}")

            await conn.execute(
                text(INSERT_NOTIF_SQL),
                {"user_id": USER_ID, "title": TEST_TITLE, "content": TEST_CONTENT},
            )
            print(f"  ✓ test notification inserted: '{TEST_TITLE}'")

        print("\n=== Running generate_daily_digest() ===")
        await scheduler_service.generate_daily_digest()
        print("  ✓ generate_daily_digest() completed")

        print("\n=== Verifying DB state ===")
        async with engine.begin() as conn:
            rows = (
                await conn.execute(
                    text(CHECK_SQL),
                    {"user_id": USER_ID, "title": TEST_TITLE},
                )
            ).fetchall()
            prefs = (await conn.execute(text(PREF_SQL), {"user_id": USER_ID})).fetchall()

        if rows:
            row = rows[0]
            print(f"  Notification row:  {dict(row._mapping)}")
            if row.delivered_at or row.sent_at:
                print("  ✓ PASS — notification marked as delivered/sent")
            else:
                print("  ✗ FAIL — notification NOT marked as delivered/sent")
        else:
            print("  ✗ FAIL — notification row not found after digest run")

        if prefs:
            print(f"  Preferences JSONB: {prefs[0][0]}")

        print("\n=== SCHEDULER_TEST_COMPLETE ===")

    except Exception as exc:
        import traceback

        print(f"\n!!! ERROR: {exc}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        await engine.dispose()
        db_manager.engine = None


asyncio.run(main())

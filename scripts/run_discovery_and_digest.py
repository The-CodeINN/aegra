"""
Run real opportunity discovery for the user, then send the digest email.
No mocks, no test data.
"""

import asyncio
import os
import sys

asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
sys.path.insert(0, r"C:\Users\thecodeinn\Documents\dedata\ai-service\libs\aegra-api\src")
os.chdir(r"C:\Users\thecodeinn\Documents\dedata\ai-service")

from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402

from aegra_api.core.database import db_manager  # noqa: E402
from aegra_api.services.notification_engine import notification_engine  # noqa: E402
from aegra_api.services.opportunity_discovery import opportunity_engine  # noqa: E402
from aegra_api.services.scheduler import scheduler_service  # noqa: E402
from aegra_api.settings import settings  # noqa: E402

# Propagate OPENAI_API_KEY from .env into os.environ so langchain_openai picks it up
if settings.discovery.OPENAI_API_KEY and not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = settings.discovery.OPENAI_API_KEY

USER_ID = "68c30006cc08c47f660b1941"
USER_EMAIL = "emijere.richard@gmail.com"
USER_NAME = "Richard"
# The user's actual enrolled tracks — stored in preferences so discovery uses them
USER_TRACKS = ["data-analytics", "data-science"]


async def main() -> None:
    engine = create_async_engine(settings.db.database_url, pool_pre_ping=True)
    db_manager.engine = engine
    session_maker = async_sessionmaker(engine, expire_on_commit=False)

    # 1. Ensure user preferences have email + job mail enabled, cooldown cleared
    print("=== Updating preferences ===")
    import json as _json

    tracks_json = _json.dumps(USER_TRACKS)
    async with engine.begin() as conn:
        await conn.execute(
            text("""
            INSERT INTO user_preferences (user_id, notifications_enabled, preferences, updated_at)
            VALUES (
                :uid, true,
                jsonb_build_object(
                    'job_opportunity_mail_enabled', true,
                    'job_opportunity_mail_frequency', 'daily',
                    'user_email', cast(:email as text),
                    'user_name',  cast(:name  as text),
                    'tracks',     cast(:tracks as text)::jsonb,
                    'learning_track', cast(:lt as text)
                ),
                now()
            )
            ON CONFLICT (user_id) DO UPDATE
                SET notifications_enabled = true,
                    preferences = (
                        COALESCE(user_preferences.preferences, '{}'::jsonb)
                        || jsonb_build_object(
                            'job_opportunity_mail_enabled', true,
                            'job_opportunity_mail_frequency', 'daily',
                            'user_email', cast(:email as text),
                            'user_name',  cast(:name  as text),
                            'tracks',     cast(:tracks as text)::jsonb,
                            'learning_track', cast(:lt as text)
                        )
                    ) - 'last_job_opportunity_digest_sent_at',
                    updated_at = now()
        """),
            {"uid": USER_ID, "email": USER_EMAIL, "name": USER_NAME, "tracks": tracks_json, "lt": USER_TRACKS[0]},
        )
    print(f"  OK preferences set — tracks={USER_TRACKS}, cooldown cleared")

    # 2. Run real opportunity discovery
    print("\n=== Running opportunity discovery ===")
    async with session_maker() as session:
        discovered = await opportunity_engine.discover_for_user(
            session=session,
            user_id=USER_ID,
            auth_token="",  # nosec B106
            max_tracks=3,
            queries_per_category=2,
        )
        print(f"  OK {len(discovered)} new opportunities found")

        for opp in discovered:
            if opp.opportunity_type == "event":
                title = "[event] New Event Matches Your Track"
                content = f"We found a {opp.matched_track} event: {opp.title}"
            else:
                company_part = f" at {opp.company}" if opp.company else ""
                title = "[job] Job Opportunity Alert"
                content = f"New {opp.matched_track} role: {opp.title}{company_part}"

            await notification_engine.create_notification(
                session=session,
                user_id=USER_ID,
                title=title,
                content=content,
                priority="normal",
                category="opportunity",
                action_buttons=[
                    {"action": "view", "title": "View", "url": opp.url},
                    {"action": "dismiss", "title": "Dismiss"},
                ],
                metadata={"opportunity_id": opp.id, "opportunity_type": opp.opportunity_type, "url": opp.url},
                check_frequency=False,
            )
            opp.status = "notified"
            print(f"    • [{opp.opportunity_type}] {opp.title}")
            if opp.company:
                print(f"      @ {opp.company}")
            print(f"      {opp.url}")

        await session.commit()

    # 3. Send the digest
    print("\n=== Sending digest email ===")
    await scheduler_service.generate_daily_digest()
    print(f"  OK Digest sent to {USER_EMAIL}")

    await engine.dispose()


asyncio.run(main())

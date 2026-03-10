"""Check real opportunity notifications and discovered opportunities in the DB."""

import asyncio
import os
import sys

asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
sys.path.insert(0, r"C:\Users\thecodeinn\Documents\dedata\ai-service\libs\aegra-api\src")
os.chdir(r"C:\Users\thecodeinn\Documents\dedata\ai-service")

from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

from aegra_api.core.database import db_manager  # noqa: E402
from aegra_api.settings import settings  # noqa: E402

USER_ID = "68c30006cc08c47f660b1941"


async def main():
    engine = create_async_engine(settings.db.database_url, pool_pre_ping=True)
    db_manager.engine = engine
    async with engine.connect() as conn:
        # Real opportunity notifications
        rows = await conn.execute(
            text("""
            SELECT id, title, content, category, priority, created_at, delivered_at, metadata
            FROM notifications
            WHERE user_id = :uid
              AND category = 'opportunity'
            ORDER BY created_at DESC
            LIMIT 20
        """),
            {"uid": USER_ID},
        )
        results = rows.mappings().all()
        print(f"=== Opportunity Notifications ({len(results)} rows) ===")
        for r in results:
            status = "sent" if r["delivered_at"] else "pending"
            print(f"  [{status}] {r['title']}")
            print(f"    {r['content'][:120]}")
            meta = r["metadata"]
            if meta and isinstance(meta, dict):
                url = meta.get("url") or meta.get("action_url")
                if url:
                    print(f"    URL: {url}")
            print()

        # Discovered opportunities table
        rows2 = await conn.execute(
            text("""
            SELECT title, company, url, opportunity_type, matched_track, status, created_at
            FROM discovered_opportunities
            WHERE user_id = :uid
            ORDER BY created_at DESC
            LIMIT 10
        """),
            {"uid": USER_ID},
        )
        opps = rows2.mappings().all()
        print(f"=== Discovered Opportunities ({len(opps)} rows) ===")
        for o in opps:
            print(f"  [{o['status']}] {o['title']} @ {o['company']} ({o['opportunity_type']})")
            print(f"    URL: {o['url']}")
            print()

    await engine.dispose()


asyncio.run(main())

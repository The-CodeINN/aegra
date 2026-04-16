"""Test script to debug opportunity discovery (events via Eventbrite, jobs via JobSpy)."""

import asyncio
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs", "aegra-api", "src")
)

from aegra_api.services.opportunity_discovery import opportunity_engine
from aegra_api.services.opportunity_event_sources import EventbriteSource

TRACK = "data-analytics"
LOCATION = "United Kingdom"


async def test_eventbrite_events():
    """Test event discovery via direct Eventbrite scraping."""
    print(f"\n=== Event Queries for '{TRACK}' in '{LOCATION}' ===")

    event_queries = opportunity_engine.build_event_queries(TRACK, LOCATION)
    print(f"Generated {len(event_queries)} queries: {event_queries[:3]}")
    source = EventbriteSource()

    for query in event_queries[:2]:
        print(f"\n--- Query: {query[:80]} ---")
        results = await source.fetch(query, LOCATION)
        print(f"Results: {len(results)}")
        for result in results[:3]:
            parsed = opportunity_engine._match_raw_event(result, TRACK, LOCATION)
            status = f"✅ {parsed['opportunity_type']}" if parsed else "❌ filtered"
            print(f"  {status} | {result.title[:55]}")
            print(f"           {result.url[:70]}")


def test_jobspy_jobs():
    """Test job discovery via JobSpy (sync)."""
    from aegra_api.services.opportunity_discovery import (
        JOBSPY_RESULTS_PER_SITE,
        OpportunityDiscoveryEngine,
    )

    print(f"\n=== JobSpy Jobs for '{TRACK}' in '{LOCATION}' ===")
    rows = OpportunityDiscoveryEngine._scrape_jobspy_sync(
        search_term="data analyst",
        location=LOCATION,
        country_indeed="UK",
        results_per_site=JOBSPY_RESULTS_PER_SITE,
        google_search_term=f"data analyst jobs in {LOCATION}",
    )
    print(f"Raw rows returned: {len(rows)}")
    for row in rows[:5]:
        site = row.get("site", "?")
        title = str(row.get("title") or "")
        company = str(row.get("company") or "")
        url = str(row.get("job_url") or "")
        print(f"  [{site}] {title[:45]} @ {company[:30]}")
        print(f"           {url[:70]}")

    # Run through the parser
    print("\n--- Parsed results (relevance-filtered) ---")
    engine = opportunity_engine
    for row in rows:
        parsed = engine._parse_jobspy_row(row, TRACK, [LOCATION])
        if parsed:
            score = parsed["match_score"]
            print(f"  ✅ {parsed['title'][:45]} @ {parsed.get('company', '')[:30]}  score={score}")


async def main():
    await test_eventbrite_events()
    test_jobspy_jobs()


if __name__ == "__main__":
    asyncio.run(main())

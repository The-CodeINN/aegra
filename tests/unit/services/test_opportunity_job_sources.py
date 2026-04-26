import json
from datetime import UTC, datetime, timedelta

from aegra_api.services.opportunity_discovery import OpportunityDiscoveryEngine
from aegra_api.services.opportunity_event_sources import EventbriteSource, RawEventOpportunity
from aegra_api.services.opportunity_job_sources import (
    AshbySource,
    BoardSourceConfig,
    GreenhouseSource,
    JobDiscoveryContext,
    JobSourceRegistry,
    LeverSource,
    RawJobOpportunity,
    WorkableSource,
)
from aegra_api.services.student_profile import StudentProfile
from aegra_api.settings import DiscoverySettings


def test_discovery_settings_parse_company_job_boards() -> None:
    settings = DiscoverySettings(
        DISCOVERY_COMPANY_JOB_BOARDS_FILE="",
        DISCOVERY_COMPANY_JOB_BOARDS_JSON=(
            '[{"provider":"greenhouse","company":"openai"},'
            '{"provider":"rippling","board_token":"rippling","label":"Rippling"}]'
        ),
    )

    assert settings.company_job_boards == [
        {"provider": "greenhouse", "company": "openai"},
        {"provider": "rippling", "board_token": "rippling", "label": "Rippling"},
    ]


def test_discovery_settings_load_company_job_boards_from_file(tmp_path) -> None:
    boards_file = tmp_path / "boards.json"
    boards_file.write_text(
        json.dumps(
            [
                {
                    "provider": "greenhouse",
                    "company": "paystack",
                    "label": "Paystack",
                    "country": "Nigeria",
                }
            ]
        ),
        encoding="utf-8",
    )

    settings = DiscoverySettings(
        DISCOVERY_COMPANY_JOB_BOARDS_FILE=str(boards_file),
        DISCOVERY_COMPANY_JOB_BOARDS_JSON="[]",
    )

    assert settings.company_job_boards == [
        {
            "provider": "greenhouse",
            "company": "paystack",
            "label": "Paystack",
            "country": "Nigeria",
        }
    ]


def test_discovery_settings_merge_file_and_inline_company_job_boards(tmp_path) -> None:
    boards_file = tmp_path / "boards.json"
    boards_file.write_text(
        json.dumps(
            [
                {
                    "provider": "greenhouse",
                    "company": "paystack",
                    "label": "Paystack",
                }
            ]
        ),
        encoding="utf-8",
    )

    settings = DiscoverySettings(
        DISCOVERY_COMPANY_JOB_BOARDS_FILE=str(boards_file),
        DISCOVERY_COMPANY_JOB_BOARDS_JSON='[{"provider":"ashby","company":"notion","label":"Notion"}]',
    )

    assert settings.company_job_boards == [
        {"provider": "greenhouse", "company": "paystack", "label": "Paystack"},
        {"provider": "ashby", "company": "notion", "label": "Notion"},
    ]


def test_discovery_settings_preserves_board_location_metadata(tmp_path) -> None:
    boards_file = tmp_path / "boards.json"
    boards_file.write_text(
        json.dumps(
            [
                {
                    "provider": "greenhouse",
                    "company": "paystack",
                    "label": "Paystack",
                    "country": "Nigeria",
                    "regions": ["Africa", "EMEA"],
                }
            ]
        ),
        encoding="utf-8",
    )

    settings = DiscoverySettings(
        DISCOVERY_COMPANY_JOB_BOARDS_FILE=str(boards_file),
        DISCOVERY_COMPANY_JOB_BOARDS_JSON="[]",
    )

    assert settings.company_job_boards == [
        {
            "provider": "greenhouse",
            "company": "paystack",
            "label": "Paystack",
            "country": "Nigeria",
            "regions": ["Africa", "EMEA"],
        }
    ]


def test_job_source_registry_filters_boards_by_location() -> None:
    registry = JobSourceRegistry(
        [
            {"provider": "greenhouse", "company": "paystack", "country": "Nigeria"},
            {"provider": "greenhouse", "company": "monzo", "country": "United Kingdom"},
        ]
    )

    sources = registry.build_sources(
        JobDiscoveryContext(
            search_terms=["data analytics"],
            primary_location="Nigeria",
            indeed_country="Nigeria",
        )
    )

    source_names = [source.name for source in sources]
    greenhouse_companies = [source.config.company for source in sources if isinstance(source, GreenhouseSource)]

    assert source_names[:3] == ["jobspy:linkedin", "jobspy:indeed", "workable"]
    assert greenhouse_companies == ["paystack"]


def test_build_event_queries_avoid_search_engine_syntax() -> None:
    engine = OpportunityDiscoveryEngine()

    queries = engine.build_event_queries("data-analytics", "Nigeria")

    assert queries
    assert all("site:eventbrite.com" not in query for query in queries)


def test_build_job_search_terms_skip_unrelated_target_role() -> None:
    engine = OpportunityDiscoveryEngine()
    profile = StudentProfile(target_role="AI Engineer/LLM Developer")

    search_terms = engine._build_job_search_terms(["data-analytics"], profile)

    assert search_terms == ["data analytics"]


def test_build_job_search_terms_include_aligned_profile_fields() -> None:
    engine = OpportunityDiscoveryEngine()
    profile = StudentProfile(
        target_role="Data Analyst",
        role_title="Business Intelligence Analyst",
    )

    search_terms = engine._build_job_search_terms(["data-analytics"], profile)

    assert search_terms == [
        "data analytics",
        "Data Analyst",
        "Business Intelligence Analyst",
    ]


async def test_greenhouse_source_fetch_parses_jobs(monkeypatch) -> None:
    source = GreenhouseSource(BoardSourceConfig(provider="greenhouse", company="openai", label="OpenAI"))

    async def fake_get_json(url: str, **kwargs):
        assert "boards-api.greenhouse.io" in url
        return {
            "jobs": [
                {
                    "id": 123,
                    "title": "Data Engineer",
                    "absolute_url": "https://boards.greenhouse.io/openai/jobs/123",
                    "location": {"name": "San Francisco, CA"},
                    "content": "<p>Build data pipelines with Python and SQL.</p>",
                }
            ]
        }

    monkeypatch.setattr(source, "_get_json", fake_get_json)
    jobs = await source.fetch(
        JobDiscoveryContext(search_terms=["data engineer"], primary_location="USA", indeed_country="USA")
    )

    assert len(jobs) == 1
    assert jobs[0].title == "Data Engineer"
    assert jobs[0].company == "OpenAI"
    assert jobs[0].location == "San Francisco, CA"
    assert "Python and SQL" in jobs[0].description
    assert jobs[0].source == "greenhouse"


async def test_ashby_source_fetch_parses_jobs(monkeypatch) -> None:
    source = AshbySource(BoardSourceConfig(provider="ashby", company="notion", label="Notion"))

    async def fake_get_json(url: str, **kwargs):
        assert "api.ashbyhq.com" in url
        return {
            "organization": {"name": "Notion"},
            "jobs": [
                {
                    "id": "job_1",
                    "title": "Analytics Engineer",
                    "jobUrl": "https://jobs.ashbyhq.com/notion/job_1",
                    "descriptionPlain": "Build analytics datasets and partner with data teams.",
                    "location": "Remote",
                    "secondaryLocations": [{"location": "London"}],
                    "compensation": {
                        "minAmount": 120000,
                        "maxAmount": 150000,
                        "interval": "yr",
                    },
                }
            ],
        }

    monkeypatch.setattr(source, "_get_json", fake_get_json)
    jobs = await source.fetch(
        JobDiscoveryContext(search_terms=["analytics engineer"], primary_location="Remote", indeed_country="USA")
    )

    assert len(jobs) == 1
    assert jobs[0].title == "Analytics Engineer"
    assert jobs[0].company == "Notion"
    assert jobs[0].location == "Remote, London"
    assert jobs[0].salary_range == "$120,000-$150,000/yr"


async def test_workable_source_fetch_parses_jobs(monkeypatch) -> None:
    source = WorkableSource()

    async def fake_get_json(**params):
        assert params["location"] == "Lagos, Nigeria"
        assert params["day_range"] == 7
        assert params["query"] == "data engineer"
        return {
            "title": "Workable Jobs",
            "totalSize": 1,
            "jobs": [
                {
                    "id": "job_1",
                    "title": "Data Engineer",
                    "description": "<p>Build reliable data pipelines in Lagos.</p>",
                    "employmentType": "Full time",
                    "state": "published",
                    "url": "https://jobs.workable.com/view/abc123",
                    "company": {"title": "Apex Network"},
                    "location": {
                        "city": "Lagos",
                        "subregion": "Lagos",
                        "countryName": "Nigeria",
                    },
                    "workplace": "onsite",
                }
            ],
        }

    monkeypatch.setattr(source, "_get_json", fake_get_json)
    jobs = await source.fetch(
        JobDiscoveryContext(
            search_terms=["data engineer"],
            primary_location="Lagos, Nigeria",
            indeed_country="Nigeria",
        )
    )

    assert len(jobs) == 1
    assert jobs[0].title == "Data Engineer"
    assert jobs[0].company == "Apex Network"
    assert jobs[0].location == "Lagos, Nigeria"
    assert "reliable data pipelines" in jobs[0].description
    assert jobs[0].source == "workable"
    assert jobs[0].source_query == "data engineer"


async def test_workable_source_skips_old_jobs(monkeypatch) -> None:
    source = WorkableSource()
    stale_date = (datetime.now(UTC) - timedelta(days=40)).isoformat()

    async def fake_get_json(**params):
        return {
            "jobs": [
                {
                    "id": "job_old",
                    "title": "Old Data Engineer",
                    "description": "<p>Old role</p>",
                    "url": "https://jobs.workable.com/view/old-job",
                    "company": {"title": "Old Co"},
                    "location": {
                        "city": "Lagos",
                        "countryName": "Nigeria",
                    },
                    "updated": stale_date,
                }
            ]
        }

    monkeypatch.setattr(source, "_get_json", fake_get_json)
    jobs = await source.fetch(
        JobDiscoveryContext(
            search_terms=["data engineer"],
            primary_location="Lagos, Nigeria",
            indeed_country="Nigeria",
        )
    )

    assert jobs == []


async def test_lever_source_skips_old_jobs(monkeypatch) -> None:
    source = LeverSource(BoardSourceConfig(provider="lever", company="plaid", label="Plaid"))
    stale_timestamp_ms = int((datetime.now(UTC) - timedelta(days=35)).timestamp() * 1000)

    async def fake_get_json(url: str, **kwargs):
        assert "api.lever.co" in url
        return [
            {
                "text": "Old Backend Engineer",
                "hostedUrl": "https://jobs.lever.co/plaid/old-backend-engineer",
                "descriptionPlain": "Legacy backend role.",
                "categories": {"location": "New York, United States"},
                "createdAt": stale_timestamp_ms,
            }
        ]

    monkeypatch.setattr(source, "_get_json", fake_get_json)
    jobs = await source.fetch(
        JobDiscoveryContext(
            search_terms=["backend engineer"],
            primary_location="New York, United States",
            indeed_country="USA",
        )
    )

    assert jobs == []


async def test_eventbrite_source_parses_json_ld(monkeypatch) -> None:
    source = EventbriteSource()

    async def fake_fetch_html(url: str) -> str:
        assert "eventbrite.com" in url
        return """
        <html><body>
        <script type="application/ld+json">
        {
            "@context": "https://schema.org",
            "@type": "ItemList",
            "itemListElement": [
                {
                    "@type": "ListItem",
                    "position": 1,
                    "item": {
                        "@type": "Event",
                        "name": "Data Analytics Networking Night",
                        "description": "Meet analytics professionals and discuss SQL and BI careers.",
                        "url": "https://www.eventbrite.com/e/data-analytics-networking-night-tickets-1",
                        "startDate": "2026-05-01T18:00:00+00:00",
                        "location": {"name": "London, United Kingdom"}
                    }
                }
            ]
        }
        </script>
        </body></html>
        """

    monkeypatch.setattr(source, "_fetch_html", fake_fetch_html)
    events = await source.fetch("data analytics", "United Kingdom")

    assert len(events) == 1
    assert events[0].title == "Data Analytics Networking Night"
    assert events[0].location == "London, United Kingdom"
    assert events[0].source == "eventbrite"
    assert events[0].event_date is not None


def test_match_raw_job_uses_best_track() -> None:
    engine = OpportunityDiscoveryEngine()
    raw_job = RawJobOpportunity(
        title="Data Engineer",
        url="https://jobs.example.com/data-engineer",
        description="Build ETL pipelines with Airflow, Spark, and SQL.",
        location="Remote",
        company="Example",
        source="greenhouse",
    )

    matched = engine._match_raw_job(
        raw_job,
        tracks=["data-engineering", "data-science"],
        locations=["remote"],
        profile=None,
    )

    assert matched is not None
    assert matched["matched_track"] == "data-engineering"
    assert matched["company"] == "Example"
    assert matched["_source"] == "greenhouse"
    assert "matched_location" in matched["reason_tags"]


def test_match_raw_job_uses_profile_fields_for_relevance() -> None:
    engine = OpportunityDiscoveryEngine()
    raw_job = RawJobOpportunity(
        title="Senior Data Analyst",
        url="https://jobs.example.com/data-analyst",
        description="Build fintech dashboards with SQL and business intelligence tooling.",
        location="Remote",
        company="Example",
        source="workable",
    )
    profile = StudentProfile(
        target_role="Data Analyst",
        industry="Fintech",
        confident_skills=["SQL"],
    )

    matched = engine._match_raw_job(
        raw_job,
        tracks=["data-analytics"],
        locations=["remote"],
        profile=profile,
    )

    assert matched is not None
    assert "matched_target_role" in matched["reason_tags"]
    assert "matched_industry" in matched["reason_tags"]
    assert "matched_profile_skills" in matched["reason_tags"]


def test_match_raw_event_uses_profile_fields_for_relevance() -> None:
    engine = OpportunityDiscoveryEngine()
    profile_event = RawEventOpportunity(
        title="Fintech Data Analyst Meetup",
        url="https://www.eventbrite.com/e/fintech-data-analytics-meetup",
        description="A networking event for data analysts using SQL, dashboards, and business intelligence in fintech.",
        location="Nigeria",
    )
    profile = StudentProfile(
        target_role="Data Analyst",
        industry="Fintech",
        confident_skills=["SQL"],
    )

    matched = engine._match_raw_event(
        profile_event,
        track="data-analytics",
        location="Nigeria",
        profile=profile,
    )

    assert matched is not None
    assert "matched_target_role" in matched["reason_tags"]
    assert "matched_industry" in matched["reason_tags"]
    assert "matched_profile_skills" in matched["reason_tags"]


def test_build_job_search_terms_prioritizes_target_role() -> None:
    engine = OpportunityDiscoveryEngine()
    profile = StudentProfile(target_role="AI Engineer")

    terms = engine._build_job_search_terms(
        tracks=["ai-engineering"],
        profile=profile,
        queries_per_category=2,
    )

    assert terms[0].lower() == "ai engineer"
    assert "artificial intelligence" in terms


async def test_discover_jobs_caps_results_per_source(monkeypatch) -> None:
    engine = OpportunityDiscoveryEngine()

    raw_jobs = [
        RawJobOpportunity(
            title=f"Data Analytics Role {index}",
            url=f"https://jobs.example.com/{index}",
            description="Data analytics with SQL and BI in Nigeria",
            location="Nigeria",
            company="Example",
            source="jobspy:linkedin" if index < 4 else "workable",
        )
        for index in range(6)
    ]

    async def fake_fetch_all(context: JobDiscoveryContext):
        return raw_jobs

    monkeypatch.setattr(engine.job_source_registry, "fetch_all", fake_fetch_all)

    jobs = await engine._discover_jobs(
        tracks=["data-analytics"],
        locations=["Nigeria"],
        profile=None,
        seen_urls=set(),
        queries_per_category=2,
    )

    by_source: dict[str, int] = {}
    for job in jobs:
        by_source[job["_source"]] = by_source.get(job["_source"], 0) + 1

    assert by_source == {"jobspy:linkedin": 3, "workable": 2}

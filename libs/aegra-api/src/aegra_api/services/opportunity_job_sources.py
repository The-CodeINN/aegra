"""Board-specific job scraping providers for opportunity discovery.

This follows the source-oriented strategy used by job-board-scraper:
each ATS/source has its own fetcher, while the discovery engine handles
matching and persistence against a unified opportunity model.
"""

from __future__ import annotations

import asyncio
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import structlog

try:
    from jobspy import scrape_jobs as _jobspy_scrape_jobs

    from aegra_api.utils.jobspy_patches import _apply as _apply_jobspy_patches

    _apply_jobspy_patches()
    _JOBSPY_AVAILABLE = True
except ImportError:
    _JOBSPY_AVAILABLE = False
    _jobspy_scrape_jobs = None  # type: ignore[assignment]

logger = structlog.get_logger()


ASHBY_JOB_BOARD_URL = "https://api.ashbyhq.com/posting-api/job-board/{company}"
GREENHOUSE_JOBS_URL = "https://boards-api.greenhouse.io/v1/boards/{company}/jobs"
LEVER_JOBS_URL = "https://api.lever.co/v0/postings/{company}"
RIPPLING_JOBS_URL = "https://api.rippling.com/platform/api/ats/v1/board/{board_token}/jobs"
WORKABLE_JOBS_URL = "https://jobs.workable.com/api/v1/jobs"
DEFAULT_MAX_JOB_AGE_DAYS = 7
DEFAULT_MAX_RESULTS_PER_SOURCE = 3


def _clean_text(value: str | None) -> str:
    if not value:
        return ""
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", value)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _safe_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _join_parts(parts: list[str]) -> str:
    return ", ".join(part for part in parts if part)


def _join_unique_parts(parts: list[str]) -> str:
    seen: set[str] = set()
    normalized: list[str] = []
    for part in parts:
        text = _safe_text(part)
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(text)
    return ", ".join(normalized)


def _format_salary(min_amount: Any, max_amount: Any, interval: str | None = None) -> str | None:
    try:
        min_value = int(min_amount) if min_amount is not None else None
    except (TypeError, ValueError):
        min_value = None

    try:
        max_value = int(max_amount) if max_amount is not None else None
    except (TypeError, ValueError):
        max_value = None

    if min_value is None and max_value is None:
        return None

    suffix = interval or "yr"
    if min_value is not None and max_value is not None:
        return f"${min_value:,}-${max_value:,}/{suffix}"
    if min_value is not None:
        return f"${min_value:,}+/{suffix}"
    return f"Up to ${max_value:,}/{suffix}"


def _parse_posted_at(value: Any) -> datetime | None:
    if value in (None, ""):
        return None

    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)

    if isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp > 1_000_000_000_000:
            timestamp /= 1000.0
        try:
            return datetime.fromtimestamp(timestamp, tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None

    text = _safe_text(value)
    if not text:
        return None

    normalized = text.replace("Z", "+00:00")
    for candidate in (normalized, normalized.replace(" ", "T", 1)):
        try:
            parsed = datetime.fromisoformat(candidate)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except ValueError:
            continue

    return None


def _first_posted_at(*values: Any) -> datetime | None:
    for value in values:
        parsed = _parse_posted_at(value)
        if parsed is not None:
            return parsed
    return None


def _is_recent_job(posted_at: datetime | None, max_age_days: int) -> bool:
    if posted_at is None:
        return True
    return posted_at >= datetime.now(UTC) - timedelta(days=max_age_days)


@dataclass(slots=True)
class RawJobOpportunity:
    title: str
    url: str
    description: str = ""
    location: str = ""
    company: str | None = None
    salary_range: str | None = None
    source: str = ""
    source_query: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class JobDiscoveryContext:
    search_terms: list[str]
    primary_location: str
    indeed_country: str
    max_search_terms: int = 2
    results_per_source: int = DEFAULT_MAX_RESULTS_PER_SOURCE
    max_results_per_source: int = DEFAULT_MAX_RESULTS_PER_SOURCE
    max_job_age_days: int = DEFAULT_MAX_JOB_AGE_DAYS


@dataclass(slots=True)
class BoardSourceConfig:
    provider: str
    company: str | None = None
    board_token: str | None = None
    url: str | None = None
    label: str | None = None
    country: str | None = None
    regions: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> BoardSourceConfig | None:
        provider = _safe_text(raw.get("provider")).lower()
        if not provider:
            return None

        return cls(
            provider=provider,
            company=_safe_text(raw.get("company")) or None,
            board_token=_safe_text(raw.get("board_token")) or None,
            url=_safe_text(raw.get("url")) or None,
            label=_safe_text(raw.get("label")) or None,
            country=_safe_text(raw.get("country")) or None,
            regions=[_safe_text(region) for region in raw.get("regions", []) if _safe_text(region)],
        )


class JobSource(ABC):
    name: str

    @abstractmethod
    async def fetch(self, context: JobDiscoveryContext) -> list[RawJobOpportunity]:
        raise NotImplementedError


def _normalize_jobspy_rows(rows: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if rows is None:
        return normalized

    for _, row in rows.iterrows():
        normalized.append({key: (None if (value != value) else value) for key, value in row.items()})

    return normalized


def _scrape_jobspy_sync(
    *,
    site_name: str,
    search_term: str,
    location: str,
    country_indeed: str,
    results_wanted: int,
    google_search_term: str,
) -> list[dict[str, Any]]:
    if not _JOBSPY_AVAILABLE or _jobspy_scrape_jobs is None:
        return []

    try:
        rows = _jobspy_scrape_jobs(
            site_name=[site_name],
            search_term=search_term,
            google_search_term=google_search_term,
            location=location,
            results_wanted=results_wanted,
            hours_old=72,
            country_indeed=country_indeed,
            verbose=0,
        )
        return _normalize_jobspy_rows(rows)
    except Exception as exc:
        logger.warning("jobspy_source_failed", site=site_name, search_term=search_term, error=str(exc))
        return []


class JobSpySource(JobSource):
    def __init__(self, site_name: str) -> None:
        self.site_name = site_name
        self.name = f"jobspy:{site_name}"

    async def fetch(self, context: JobDiscoveryContext) -> list[RawJobOpportunity]:
        if not _JOBSPY_AVAILABLE:
            return []

        jobs: list[RawJobOpportunity] = []
        for search_term in context.search_terms[: context.max_search_terms]:
            google_search_term = (
                f"{search_term} jobs in {context.primary_location}"
                if context.primary_location.lower() not in ("", "remote")
                else f"{search_term} remote jobs"
            )
            rows = await asyncio.to_thread(
                _scrape_jobspy_sync,
                site_name=self.site_name,
                search_term=search_term,
                location=context.primary_location,
                country_indeed=context.indeed_country,
                results_wanted=context.results_per_source,
                google_search_term=google_search_term,
            )
            for row in rows:
                title = _safe_text(row.get("title"))
                url = _safe_text(row.get("job_url"))
                if not title or not url:
                    continue

                jobs.append(
                    RawJobOpportunity(
                        title=title,
                        url=url,
                        description=_safe_text(row.get("description"))[:1200],
                        location=_safe_text(row.get("location")),
                        company=_safe_text(row.get("company")) or None,
                        salary_range=_format_salary(
                            row.get("min_amount"), row.get("max_amount"), _safe_text(row.get("interval")) or None
                        ),
                        source=self.name,
                        source_query=search_term,
                        metadata={"site": self.site_name},
                    )
                )

                if len(jobs) >= context.max_results_per_source:
                    return jobs[: context.max_results_per_source]

        return jobs[: context.max_results_per_source]


class WorkableSource(JobSource):
    name = "workable"

    async def _get_json(self, **params: Any) -> dict[str, Any]:
        async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as client:
            response = await client.get(WORKABLE_JOBS_URL, params=params)
            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, dict) else {}

    def _extract_location(self, job: dict[str, Any]) -> str:
        workplace = _safe_text(job.get("workplace")).lower()
        if workplace == "remote":
            return "Remote"

        location = job.get("location") or {}
        if isinstance(location, dict):
            normalized = _join_unique_parts(
                [
                    _safe_text(location.get("city")),
                    _safe_text(location.get("subregion")),
                    _safe_text(location.get("countryName")),
                ]
            )
            if normalized:
                return normalized

        locations = job.get("locations") or []
        if isinstance(locations, list):
            flattened: list[str] = []
            for item in locations:
                if isinstance(item, dict):
                    flattened.append(
                        _join_unique_parts(
                            [
                                _safe_text(item.get("city")),
                                _safe_text(item.get("subregion")),
                                _safe_text(item.get("countryName")),
                            ]
                        )
                    )
                else:
                    flattened.append(_safe_text(item))
            normalized = _join_parts([value for value in flattened if value])
            if normalized:
                return normalized

        return ""

    async def fetch(self, context: JobDiscoveryContext) -> list[RawJobOpportunity]:
        jobs: list[RawJobOpportunity] = []
        seen_urls: set[str] = set()

        for search_term in context.search_terms[: context.max_search_terms]:
            try:
                payload = await self._get_json(
                    location=context.primary_location,
                    day_range=7,
                    query=search_term,
                )
            except Exception as exc:
                logger.warning(
                    "workable_fetch_failed",
                    search_term=search_term,
                    location=context.primary_location,
                    error=repr(exc),
                    error_type=type(exc).__name__,
                )
                continue

            for job in payload.get("jobs", []):
                posted_at = _first_posted_at(job.get("updated"), job.get("created"))
                if not _is_recent_job(posted_at, context.max_job_age_days):
                    continue

                title = _safe_text(job.get("title"))
                url = _safe_text(job.get("url"))
                if not title or not url or url in seen_urls:
                    continue

                seen_urls.add(url)
                company = job.get("company") if isinstance(job.get("company"), dict) else {}
                company_name = _safe_text(company.get("title") or company.get("name")) or None

                jobs.append(
                    RawJobOpportunity(
                        title=title,
                        url=url,
                        description=_clean_text(job.get("description"))[:1200],
                        location=self._extract_location(job),
                        company=company_name,
                        source=self.name,
                        source_query=search_term,
                        metadata={
                            "provider": self.name,
                            "employment_type": _safe_text(job.get("employmentType")) or None,
                            "posted_at": posted_at.isoformat() if posted_at else None,
                            "state": _safe_text(job.get("state")) or None,
                            "workplace": _safe_text(job.get("workplace")) or None,
                        },
                    )
                )

                if len(jobs) >= context.max_results_per_source:
                    return jobs[: context.max_results_per_source]

        return jobs[: context.max_results_per_source]


class ConfiguredBoardSource(JobSource, ABC):
    def __init__(self, config: BoardSourceConfig) -> None:
        self.config = config

    async def _get_json(self, url: str, **kwargs: Any) -> Any:
        async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as client:
            response = await client.get(url, **kwargs)
            response.raise_for_status()
            return response.json()

    async def _post_json(self, url: str, **kwargs: Any) -> Any:
        async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as client:
            response = await client.post(url, **kwargs)
            response.raise_for_status()
            return response.json()


class GreenhouseSource(ConfiguredBoardSource):
    name = "greenhouse"

    def _resolve_company(self) -> str | None:
        if self.config.company:
            return self.config.company
        if not self.config.url:
            return None

        parsed = urlparse(self.config.url)
        query = parse_qs(parsed.query)
        if query.get("for"):
            return query["for"][0]

        parts = [part for part in parsed.path.split("/") if part]
        if parts:
            return parts[-1]
        return None

    async def fetch(self, context: JobDiscoveryContext) -> list[RawJobOpportunity]:
        company = self._resolve_company()
        if not company:
            return []

        try:
            payload = await self._get_json(GREENHOUSE_JOBS_URL.format(company=company), params={"content": "true"})
        except Exception as exc:
            logger.warning("greenhouse_fetch_failed", company=company, error=repr(exc), error_type=type(exc).__name__)
            return []

        jobs: list[RawJobOpportunity] = []
        company_name = self.config.label or company
        for job in payload.get("jobs", []):
            posted_at = _first_posted_at(
                job.get("updated_at"), job.get("updatedAt"), job.get("created_at"), job.get("createdAt")
            )
            if not _is_recent_job(posted_at, context.max_job_age_days):
                continue

            title = _safe_text(job.get("title"))
            url = _safe_text(job.get("absolute_url"))
            if not title or not url:
                continue

            location = _safe_text((job.get("location") or {}).get("name"))
            if not location:
                location = _join_parts([_safe_text(office.get("name")) for office in job.get("offices", [])])

            description = _clean_text(job.get("content"))[:1200]
            jobs.append(
                RawJobOpportunity(
                    title=title,
                    url=url,
                    description=description,
                    location=location,
                    company=company_name,
                    source=self.name,
                    metadata={
                        "provider": self.name,
                        "board": company,
                        "posted_at": posted_at.isoformat() if posted_at else None,
                    },
                )
            )

            if len(jobs) >= context.max_results_per_source:
                return jobs[: context.max_results_per_source]

        return jobs[: context.max_results_per_source]


class LeverSource(ConfiguredBoardSource):
    name = "lever"

    def _resolve_company(self) -> str | None:
        if self.config.company:
            return self.config.company
        if not self.config.url:
            return None

        parsed = urlparse(self.config.url)
        parts = [part for part in parsed.path.split("/") if part]
        return parts[-1] if parts else None

    async def fetch(self, context: JobDiscoveryContext) -> list[RawJobOpportunity]:
        company = self._resolve_company()
        if not company:
            return []

        try:
            payload = await self._get_json(LEVER_JOBS_URL.format(company=company), params={"mode": "json"})
        except Exception as exc:
            logger.warning("lever_fetch_failed", company=company, error=repr(exc), error_type=type(exc).__name__)
            return []

        jobs: list[RawJobOpportunity] = []
        company_name = self.config.label or company
        for job in payload if isinstance(payload, list) else []:
            posted_at = _first_posted_at(
                job.get("createdAt"), job.get("updatedAt"), job.get("created_at"), job.get("updated_at")
            )
            if not _is_recent_job(posted_at, context.max_job_age_days):
                continue

            title = _safe_text(job.get("text"))
            url = _safe_text(job.get("hostedUrl") or job.get("applyUrl"))
            if not title or not url:
                continue

            categories = job.get("categories") or {}
            location = _safe_text(categories.get("location"))
            department = _safe_text(categories.get("team") or categories.get("department"))
            description = _clean_text(job.get("descriptionPlain") or job.get("description"))[:1200]
            if department:
                description = _join_parts([description, f"Department: {department}"])

            jobs.append(
                RawJobOpportunity(
                    title=title,
                    url=url,
                    description=description,
                    location=location,
                    company=company_name,
                    source=self.name,
                    metadata={
                        "provider": self.name,
                        "board": company,
                        "posted_at": posted_at.isoformat() if posted_at else None,
                    },
                )
            )

            if len(jobs) >= context.max_results_per_source:
                return jobs[: context.max_results_per_source]

        return jobs[: context.max_results_per_source]


class AshbySource(ConfiguredBoardSource):
    name = "ashby"

    def _resolve_company(self) -> str | None:
        if self.config.company:
            return self.config.company
        if not self.config.url:
            return None

        parsed = urlparse(self.config.url)
        parts = [part for part in parsed.path.split("/") if part]
        return parts[-1] if parts else None

    async def fetch(self, context: JobDiscoveryContext) -> list[RawJobOpportunity]:
        company = self._resolve_company()
        if not company:
            return []

        try:
            payload = await self._get_json(
                ASHBY_JOB_BOARD_URL.format(company=company), params={"includeCompensation": "true"}
            )
        except Exception as exc:
            logger.warning("ashby_fetch_failed", company=company, error=repr(exc), error_type=type(exc).__name__)
            return []

        jobs_payload = payload.get("jobs") or payload.get("jobPostings") or []
        company_name = self.config.label or _safe_text((payload.get("organization") or {}).get("name")) or company

        jobs: list[RawJobOpportunity] = []
        for job in jobs_payload:
            posted_at = _first_posted_at(
                job.get("publishedAt"),
                job.get("postedAt"),
                job.get("updatedAt"),
                job.get("createdAt"),
                job.get("published_at"),
                job.get("created_at"),
            )
            if not _is_recent_job(posted_at, context.max_job_age_days):
                continue

            title = _safe_text(job.get("title") or job.get("name"))
            url = _safe_text(job.get("jobUrl") or job.get("url"))
            if not title or not url:
                continue

            secondary_locations = []
            for location in job.get("secondaryLocations") or []:
                secondary_locations.append(_safe_text(location.get("location") or location.get("locationName")))

            location = _join_parts(
                [
                    _safe_text(job.get("location") or job.get("locationName")),
                    *secondary_locations,
                ]
            )
            compensation = job.get("compensation") or {}
            salary_range = (
                _format_salary(
                    compensation.get("minAmount") or job.get("salaryMin"),
                    compensation.get("maxAmount") or job.get("salaryMax"),
                    _safe_text(compensation.get("interval")) or None,
                )
                or _safe_text(job.get("compensationTierSummary"))
                or None
            )

            jobs.append(
                RawJobOpportunity(
                    title=title,
                    url=url,
                    description=_clean_text(job.get("descriptionPlain") or job.get("descriptionHtml"))[:1200],
                    location=location,
                    company=company_name,
                    salary_range=salary_range,
                    source=self.name,
                    metadata={
                        "provider": self.name,
                        "board": company,
                        "posted_at": posted_at.isoformat() if posted_at else None,
                    },
                )
            )

            if len(jobs) >= context.max_results_per_source:
                return jobs[: context.max_results_per_source]

        return jobs[: context.max_results_per_source]


class RipplingSource(ConfiguredBoardSource):
    name = "rippling"

    def _resolve_board_token(self) -> str | None:
        if self.config.board_token:
            return self.config.board_token
        if self.config.company:
            return self.config.company
        if not self.config.url:
            return None

        parsed = urlparse(self.config.url)
        parts = [part for part in parsed.path.split("/") if part]
        return parts[-1] if parts else None

    async def fetch(self, context: JobDiscoveryContext) -> list[RawJobOpportunity]:
        board_token = self._resolve_board_token()
        if not board_token:
            return []

        try:
            payload = await self._get_json(RIPPLING_JOBS_URL.format(board_token=board_token))
        except Exception as exc:
            logger.warning(
                "rippling_fetch_failed", board_token=board_token, error=repr(exc), error_type=type(exc).__name__
            )
            return []

        company_name = self.config.label or board_token
        jobs: list[RawJobOpportunity] = []
        raw_jobs = payload.get("jobs") if isinstance(payload, dict) else payload
        for job in raw_jobs if isinstance(raw_jobs, list) else []:
            posted_at = _first_posted_at(
                job.get("updatedAt"), job.get("createdAt"), job.get("updated_at"), job.get("created_at")
            )
            if not _is_recent_job(posted_at, context.max_job_age_days):
                continue

            title = _safe_text(job.get("name") or job.get("title"))
            url = _safe_text(job.get("url"))
            if not title or not url:
                continue

            department = _safe_text((job.get("department") or {}).get("label"))
            location = _safe_text((job.get("workLocation") or {}).get("label"))
            description = department if department else ""

            jobs.append(
                RawJobOpportunity(
                    title=title,
                    url=url,
                    description=description,
                    location=location,
                    company=company_name,
                    source=self.name,
                    metadata={
                        "provider": self.name,
                        "board_token": board_token,
                        "posted_at": posted_at.isoformat() if posted_at else None,
                    },
                )
            )

            if len(jobs) >= context.max_results_per_source:
                return jobs[: context.max_results_per_source]

        return jobs[: context.max_results_per_source]


class JobSourceRegistry:
    def __init__(self, board_configs: list[dict[str, str]] | None = None) -> None:
        self._board_configs = []
        for raw in board_configs or []:
            config = BoardSourceConfig.from_mapping(raw)
            if config:
                self._board_configs.append(config)

    @staticmethod
    def _location_tokens(location: str) -> set[str]:
        normalized = _safe_text(location).lower()
        if not normalized or normalized == "remote":
            return set()

        tokens = {normalized}
        if "," in normalized:
            tokens.add(normalized.rsplit(",", 1)[-1].strip())
        return {token for token in tokens if token}

    def _config_matches_location(self, config: BoardSourceConfig, location: str) -> bool:
        location_tokens = self._location_tokens(location)
        if not location_tokens:
            return True

        metadata_tokens: set[str] = set()
        if config.country:
            metadata_tokens.add(config.country.strip().lower())
        metadata_tokens.update(region.strip().lower() for region in config.regions if region.strip())

        if not metadata_tokens:
            return True

        return bool(location_tokens & metadata_tokens)

    def build_sources(self, context: JobDiscoveryContext | None = None) -> list[JobSource]:
        sources: list[JobSource] = [
            JobSpySource("linkedin"),
            JobSpySource("indeed"),
            WorkableSource(),
        ]
        provider_map: dict[str, type[ConfiguredBoardSource]] = {
            "greenhouse": GreenhouseSource,
            "lever": LeverSource,
            "ashby": AshbySource,
            "rippling": RipplingSource,
        }

        for config in self._board_configs:
            if context is not None and not self._config_matches_location(config, context.primary_location):
                continue

            provider_cls = provider_map.get(config.provider)
            if not provider_cls:
                logger.warning("unknown_job_board_provider", provider=config.provider)
                continue
            sources.append(provider_cls(config))

        return sources

    async def fetch_all(self, context: JobDiscoveryContext) -> list[RawJobOpportunity]:
        sources = self.build_sources(context)
        results = await asyncio.gather(*(source.fetch(context) for source in sources), return_exceptions=True)

        jobs: list[RawJobOpportunity] = []
        for source, result in zip(sources, results, strict=False):
            if isinstance(result, Exception):
                logger.warning("job_source_error", source=source.name, error=str(result))
                continue
            jobs.extend(result)

        return jobs

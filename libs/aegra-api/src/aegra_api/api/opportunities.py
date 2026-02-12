"""API endpoints for Opportunity Discovery.

Endpoints:
- GET   /opportunities               list discovered opportunities
- GET   /opportunities/stats         aggregated stats
- GET   /opportunities/{id}          single opportunity
- GET   /opportunities/{id}/strategy AI strategy for opportunity
- POST  /opportunities/{id}/save     bookmark / save
- POST  /opportunities/{id}/dismiss  not interested
- POST  /opportunities/{id}/applied  mark applied
- POST  /opportunities/discover      manual scan (max 4/day per user)
"""

from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.core.accountability_orm import DiscoveredOpportunity, UserPreferences
from aegra_api.core.auth_deps import get_current_user
from aegra_api.core.orm import get_session
from aegra_api.models import User
from aegra_api.services.opportunity_discovery import opportunity_engine
from aegra_api.services.opportunity_service import OpportunityService
from aegra_api.settings import settings

router = APIRouter(prefix="/opportunities")


# ── Pydantic Models ──────────────────────────────────────────────────


class OpportunityResponse(BaseModel):
    id: str
    opportunity_type: str  # event | job
    title: str
    description: str | None = None
    url: str | None = None
    location: str | None = None
    event_date: str | None = None
    company: str | None = None
    salary_range: str | None = None
    match_score: float | None = None
    matched_track: str | None = None
    status: str
    discovered_at: str
    expires_at: str | None = None
    metadata: dict
    # AI strategy fields (populated from metadata) — may be dict or str
    networking_strategy: dict | str | None = None
    application_strategy: dict | str | None = None

    class Config:
        from_attributes = True

    @classmethod
    def from_orm_model(cls, opp: DiscoveredOpportunity) -> "OpportunityResponse":
        meta = opp.metadata_json or {}
        return cls(
            id=opp.id,
            opportunity_type=opp.opportunity_type,
            title=opp.title,
            description=opp.description,
            url=opp.url,
            location=opp.location,
            event_date=opp.event_date.isoformat() if opp.event_date else None,
            company=opp.company,
            salary_range=opp.salary_range,
            match_score=float(opp.match_score) if opp.match_score else None,
            matched_track=opp.matched_track,
            status=opp.status,
            discovered_at=opp.discovered_at.isoformat(),
            expires_at=opp.expires_at.isoformat() if opp.expires_at else None,
            metadata=meta,
            networking_strategy=meta.get("networking_strategy"),
            application_strategy=meta.get("application_strategy"),
        )


class OpportunityListResponse(BaseModel):
    opportunities: list[OpportunityResponse]
    total: int
    has_more: bool


class DiscoverRequest(BaseModel):
    auth_token: str | None = None


# ── Rate limiting helpers ────────────────────────────────────────────


async def _get_scan_count_today(session: AsyncSession, user_id: str) -> int:
    """Return how many manual scans the user has done today."""
    result = await session.execute(select(UserPreferences).where(UserPreferences.user_id == user_id))
    prefs = result.scalar_one_or_none()
    if not prefs or not prefs.preferences:
        return 0
    scan_log = prefs.preferences.get("scan_log", {})
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    return scan_log.get(today, 0)


async def _record_scan(session: AsyncSession, user_id: str) -> None:
    """Increment the user's manual scan count for today."""
    result = await session.execute(select(UserPreferences).where(UserPreferences.user_id == user_id))
    prefs = result.scalar_one_or_none()
    if not prefs:
        prefs = UserPreferences(user_id=user_id, preferences={})
        session.add(prefs)

    preferences = dict(prefs.preferences or {})
    scan_log = dict(preferences.get("scan_log", {}))
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    scan_log[today] = scan_log.get(today, 0) + 1

    # Keep only the last 7 days
    cutoff = (datetime.now(UTC) - timedelta(days=7)).strftime("%Y-%m-%d")
    scan_log = {k: v for k, v in scan_log.items() if k >= cutoff}

    preferences["scan_log"] = scan_log
    prefs.preferences = preferences
    await session.commit()


# ── List / Filter ────────────────────────────────────────────────────


@router.get("", response_model=OpportunityListResponse)
async def list_opportunities(
    opportunity_type: str | None = Query(None, description="Filter by type: event, job"),
    status: str = Query("new", description="Filter by status: new, notified, saved, dismissed, applied"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """List discovered opportunities."""
    opportunities, total, has_more = await OpportunityService.list_opportunities(
        session=session,
        user_id=user.identity,
        opportunity_type=opportunity_type,
        status=status,
        limit=limit,
        offset=offset,
    )
    return {
        "opportunities": [OpportunityResponse.from_orm_model(o) for o in opportunities],
        "total": total,
        "has_more": has_more,
    }


# ── Stats ────────────────────────────────────────────────────────────


@router.get("/stats")
async def opportunity_stats(
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Return aggregated counts by type and status."""
    return await OpportunityService.get_stats(session, user.identity)


# ── Single opportunity ───────────────────────────────────────────────


@router.get("/{opportunity_id}", response_model=OpportunityResponse)
async def get_opportunity(
    opportunity_id: str,
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
) -> OpportunityResponse:
    opportunity = await OpportunityService.get_opportunity(session, opportunity_id, user.identity)
    if not opportunity:
        raise HTTPException(status_code=404, detail="Opportunity not found")
    return OpportunityResponse.from_orm_model(opportunity)


# ── Strategy ─────────────────────────────────────────────────────────


@router.get("/{opportunity_id}/strategy")
async def get_strategy(
    opportunity_id: str,
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Return the AI-generated strategy for an opportunity."""
    data = await OpportunityService.get_opportunity_with_strategy(session, opportunity_id, user.identity)
    if not data:
        raise HTTPException(status_code=404, detail="Opportunity not found")
    return data


# ── Save / Dismiss / Applied ─────────────────────────────────────────


@router.post("/{opportunity_id}/save")
async def save_opportunity(
    opportunity_id: str,
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
) -> dict[str, str]:
    """Bookmark an opportunity."""
    try:
        return await OpportunityService.save_opportunity(session, opportunity_id, user.identity)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.post("/{opportunity_id}/dismiss")
async def dismiss_opportunity(
    opportunity_id: str,
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
) -> dict[str, str]:
    try:
        return await OpportunityService.dismiss_opportunity(session, opportunity_id, user.identity)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.post("/{opportunity_id}/applied")
async def mark_applied(
    opportunity_id: str,
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
) -> dict[str, str]:
    try:
        return await OpportunityService.mark_opportunity_applied(session, opportunity_id, user.identity)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


# ── Manual discovery ─────────────────────────────────────────────────


@router.post("/discover")
async def trigger_discovery(
    request: DiscoverRequest | None = None,
    authorization: str | None = Header(None),
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Manually trigger opportunity discovery scan (max 4/day per user)."""
    # Rate limiting
    max_scans = settings.discovery.DISCOVERY_MAX_MANUAL_SCANS_PER_DAY
    count = await _get_scan_count_today(session, user.identity)
    if count >= max_scans:
        raise HTTPException(
            status_code=429,
            detail=f"Daily scan limit reached ({max_scans} per day). Try again tomorrow.",
        )

    token = request.auth_token if request and request.auth_token else None

    if not token and authorization:
        scheme, _, param = authorization.partition(" ")
        token = param if scheme.lower() == "bearer" else authorization

    if not token:
        raise HTTPException(status_code=401, detail="Authentication token required for discovery")

    discovered = await opportunity_engine.discover_for_user(
        session=session,
        user_id=user.identity,
        auth_token=token,
        max_tracks=2,
        queries_per_category=2,
    )

    # Batch-create all notifications in a single commit
    await opportunity_engine.create_notifications_batch(session, discovered)

    # Record the scan for rate limiting
    await _record_scan(session, user.identity)

    return {
        "status": "success",
        "discovered_count": len(discovered),
        "events": len([o for o in discovered if o.opportunity_type == "event"]),
        "jobs": len([o for o in discovered if o.opportunity_type == "job"]),
        "scans_remaining_today": max_scans - count - 1,
    }

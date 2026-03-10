"""API Router for Accountability features.

Endpoints:
- /action-items          GET   list active items
- /action-items/{id}     POST  update status
- /preferences           GET   get user notification preferences
- /preferences           PUT   update preferences
- /activity              POST  record user activity
- /activity/stats        GET   get user activity stats

Note: Notification listing and mutation endpoints (mark read, dismiss, etc.)
have been moved to the WebSocket endpoint at /ws/notifications.
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.core.auth_deps import get_current_user
from aegra_api.core.orm import get_session
from aegra_api.models import User
from aegra_api.services.accountability_service import AccountabilityService
from aegra_api.services.advisor_cache import check_ai_mentor_addon

router = APIRouter(tags=["Accountability"])


# ── Pydantic models ──────────────────────────────────────────────────


class ActionItemResponse(BaseModel):
    id: str
    description: str
    status: str
    due_date: datetime | None = None
    priority: str
    category: str | None = None
    advisor_persona: str | None = None
    source: str | None = None
    dependencies: list | None = None
    reminder_sent_count: int = 0
    created_at: datetime

    class Config:
        from_attributes = True


class PreferencesRequest(BaseModel):
    notifications_enabled: bool | None = None
    email_enabled: bool | None = None
    job_opportunity_mail_enabled: bool | None = None
    job_opportunity_mail_frequency: str | None = None
    location: str | None = None
    push_subscription: dict | None = None
    max_daily: int | None = None
    digest_mode: bool | None = None
    quiet_hours_start: int | None = None
    quiet_hours_end: int | None = None
    disabled_categories: list[str] | None = None


class ActivityRequest(BaseModel):
    activity_type: str  # login, conversation, course


class ProgressEventRequest(BaseModel):
    """Report a learning progress event from the LMS frontend."""

    event_type: str  # course_completed, lesson_completed, quiz_passed, quiz_failed, milestone
    course_name: str | None = None
    track_name: str | None = None
    progress_percentage: float | None = None
    score: float | None = None
    details: dict | None = None


# ── Action Items ─────────────────────────────────────────────────────


@router.get("/action-items", response_model=list[ActionItemResponse])
async def list_action_items(
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
) -> Any:
    """List active action items for the current user."""
    return await AccountabilityService.list_action_items(session, user.identity)


@router.post("/action-items/{item_id}")
async def update_action_item(
    item_id: str,
    status: str,
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    try:
        return await AccountabilityService.update_action_item_status(session, item_id, user.identity, status)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


# ── Preferences ──────────────────────────────────────────────────────


@router.get("/preferences")
async def get_preferences(
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    prefs = await AccountabilityService.get_preferences(session, user.identity)
    if not prefs:
        return {
            "user_id": user.identity,
            "notifications_enabled": True,
            "location": None,
            "preferences": {
                "job_opportunity_mail_enabled": False,
                "job_opportunity_mail_frequency": "weekly",
            },
        }
    pref_data = prefs.preferences or {}
    pref_data.setdefault("job_opportunity_mail_enabled", False)
    pref_data.setdefault("job_opportunity_mail_frequency", "weekly")
    return {
        "user_id": prefs.user_id,
        "notifications_enabled": prefs.notifications_enabled,
        "location": prefs.location,
        "preferences": pref_data,
    }


@router.put("/preferences")
async def update_preferences(
    body: PreferencesRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    data = body.model_dump(exclude_none=True)

    # Gate: enabling job opportunity mail requires an active AI Mentor add-on
    if data.get("job_opportunity_mail_enabled") is True:
        auth_header = request.headers.get("Authorization", "")
        token = auth_header.removeprefix("Bearer ").strip()
        addon = await check_ai_mentor_addon(token)
        if not addon["active"]:
            raise HTTPException(
                status_code=403,
                detail="Jobs & Opportunity email requires an active AI Mentor add-on subscription.",
            )
        # Cache the add-on state so the scheduler can verify without a token
        data["ai_mentor_addon_active"] = True
        if addon.get("expires_at"):
            data["ai_mentor_addon_expires_at"] = addon["expires_at"]

    # Always persist the caller's email/name so the scheduler can use them
    # without calling the LMS.
    if user.email:
        data["user_email"] = user.email
    if user.display_name:
        data["user_name"] = user.display_name
    prefs = await AccountabilityService.upsert_preferences(session, user.identity, data)
    pref_data = prefs.preferences or {}
    pref_data.setdefault("job_opportunity_mail_enabled", False)
    pref_data.setdefault("job_opportunity_mail_frequency", "weekly")
    return {
        "user_id": prefs.user_id,
        "notifications_enabled": prefs.notifications_enabled,
        "location": prefs.location,
        "preferences": pref_data,
    }


# ── Activity tracking ────────────────────────────────────────────────


@router.post("/activity")
async def record_activity(
    body: ActivityRequest,
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
) -> dict[str, str]:
    await AccountabilityService.record_activity(session, user.identity, body.activity_type)
    return {"status": "recorded"}


@router.get("/activity/stats")
async def get_activity_stats(
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    activity = await AccountabilityService.get_activity(session, user.identity)
    if not activity:
        return {
            "current_streak": 0,
            "longest_streak": 0,
            "engagement_score": 0,
        }
    return {
        "current_streak": activity.current_streak,
        "longest_streak": activity.longest_streak,
        "engagement_score": float(activity.engagement_score),
        "last_login": activity.last_login.isoformat() if activity.last_login else None,
        "last_conversation": (activity.last_conversation.isoformat() if activity.last_conversation else None),
        "last_course_activity": (activity.last_course_activity.isoformat() if activity.last_course_activity else None),
        "last_action_completed": (
            activity.last_action_completed.isoformat() if activity.last_action_completed else None
        ),
    }


# ── Progress tracking (LMS integration) [§3.2] ─────────────────────


@router.post("/progress")
async def report_progress(
    body: ProgressEventRequest,
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Report a learning progress event from the frontend.

    Used by the frontend to notify the AI backend when a student
    completes a course, passes a quiz, reaches a milestone, etc.
    This triggers celebration/struggle notifications as appropriate.
    """
    from aegra_api.services.notification_engine import notification_engine

    # Record as course activity
    await AccountabilityService.record_activity(session, user.identity, "course")

    result: dict[str, Any] = {"status": "recorded", "event_type": body.event_type}

    # Generate celebration notifications based on event type
    if body.event_type == "course_completed":
        await notification_engine.create_notification(
            session=session,
            user_id=user.identity,
            title="🎓 Course Completed!",
            content=(
                f"Congratulations on completing {body.course_name or 'your course'}! "
                "This is a major achievement. Your dedication is paying off!"
            ),
            category="celebration",
            priority="high",
            action_buttons=[
                {"action": "share", "title": "Share Achievement"},
                {"action": "next", "title": "What's Next?", "url": "/dashboard/ai-career-advisor"},
            ],
            metadata={"event_type": body.event_type, "course_name": body.course_name},
            check_frequency=False,
            student_context={"first_name": user.display_name, "email": getattr(user, "email", None)},
        )
        result["notification"] = "celebration_sent"

    elif body.event_type == "milestone":
        pct = body.progress_percentage or 0
        await notification_engine.create_notification(
            session=session,
            user_id=user.identity,
            title=f"📈 {int(pct)}% Progress Milestone!",
            content=(f"You've reached {int(pct)}% in {body.track_name or 'your track'}. Keep this momentum going!"),
            category="celebration",
            priority="normal",
            metadata={"event_type": body.event_type, "progress": pct},
            check_frequency=True,
            student_context={"first_name": user.display_name, "email": getattr(user, "email", None)},
        )
        result["notification"] = "milestone_sent"

    elif body.event_type == "quiz_failed" and body.score is not None:
        await notification_engine.create_notification(
            session=session,
            user_id=user.identity,
            title="💪 Don't Give Up!",
            content=(
                f"That quiz was tough — you scored {int(body.score)}%. "
                "Every expert was once a beginner. Want to review the material together?"
            ),
            category="motivation",
            priority="normal",
            action_buttons=[
                {"action": "review", "title": "Review Material", "url": "/dashboard/my-tracks"},
                {"action": "chat", "title": "Get Help", "url": "/dashboard/ai-career-advisor"},
            ],
            metadata={"event_type": body.event_type, "score": body.score},
            check_frequency=True,
            student_context={"first_name": user.display_name, "email": getattr(user, "email", None)},
        )
        result["notification"] = "encouragement_sent"

    return result

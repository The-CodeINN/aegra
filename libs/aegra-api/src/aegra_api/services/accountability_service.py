"""Service for managing accountability items, notifications and user preferences.

Enhanced with:
- Richer notification queries (category, all statuses)
- Mark-all-read
- User preference management
- Activity tracking updates
- Notification dismissal
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from aegra_api.core.accountability_orm import (
    ActionItem,
    Notification,
    UserActivityTracking,
    UserPreferences,
)

logger = structlog.getLogger(__name__)

# Status values an open/in-progress task can be in. "pending" is this
# table's existing name for what the agent optimisation spec calls "open" —
# kept as-is rather than renamed, to avoid a data migration; "abandoned" and
# "renegotiated" are new terminal-ish states the agent can set directly.
OPEN_STATUSES = ("pending", "in_progress")


@dataclass(kw_only=True)
class OpenTaskGroup:
    """A user's open tasks, split by due-date status."""

    not_yet_due: list[ActionItem]
    overdue: list[ActionItem]

    def __bool__(self) -> bool:
        return bool(self.not_yet_due or self.overdue)


class AccountabilityService:
    """Service for managing accountability items and notifications."""

    # ------------------------------------------------------------------
    # Action Items
    # ------------------------------------------------------------------
    @staticmethod
    async def list_action_items(
        session: AsyncSession, user_id: str, statuses: list[str] | None = None
    ) -> Sequence[ActionItem]:
        if statuses is None:
            statuses = ["pending", "in_progress"]

        query = (
            select(ActionItem)
            .where(ActionItem.user_id == user_id, ActionItem.status.in_(statuses))
            .order_by(ActionItem.due_date.asc().nulls_last(), ActionItem.created_at.desc())
        )
        result = await session.execute(query)
        return result.scalars().all()

    @staticmethod
    async def create_action_item(
        session: AsyncSession,
        user_id: str,
        description: str,
        *,
        thread_id: str | None = None,
        due_date: datetime | None = None,
        priority: str = "normal",
        category: str | None = None,
        advisor_persona: str | None = None,
        source: str = "conversation",
        advisor_note: str | None = None,
    ) -> ActionItem:
        """Log a task the advisor assigned — the write path TaskMemory was missing.

        Called from both the agent's hot-path tool (assigned during the live
        conversation) and the cold-path background scan (commitments the
        agent phrased in prose but didn't log explicitly).
        """
        item = ActionItem(
            user_id=user_id,
            thread_id=thread_id,
            description=description,
            due_date=due_date,
            priority=priority,
            category=category,
            advisor_persona=advisor_persona,
            source=source,
            advisor_note=advisor_note,
        )
        session.add(item)
        await session.commit()
        await session.refresh(item)
        return item

    @staticmethod
    async def list_completed_items(session: AsyncSession, user_id: str, limit: int = 20) -> Sequence[ActionItem]:
        """Return the user's most recently completed tasks — for "what have I done" recall.

        Kept separate from list_action_items (which defaults to open-only and
        feeds check-ins/dashboard) so completed tasks are answerable on request
        without ever leaking into the open-task surfaces.
        """
        query = (
            select(ActionItem)
            .where(ActionItem.user_id == user_id, ActionItem.status == "completed")
            .order_by(ActionItem.updated_at.desc())
            .limit(limit)
        )
        result = await session.execute(query)
        return result.scalars().all()

    @staticmethod
    async def get_open_and_overdue(session: AsyncSession, user_id: str) -> OpenTaskGroup:
        """Return this user's open tasks, split into not-yet-due vs overdue.

        This is what conversation-start injection (call_model) and outreach
        generation (notification_engine) both read — the single source of
        truth for "what did the advisor assign that isn't done yet".
        """
        now = datetime.now(UTC)
        query = (
            select(ActionItem)
            .where(ActionItem.user_id == user_id, ActionItem.status.in_(OPEN_STATUSES))
            .order_by(ActionItem.due_date.asc().nulls_last())
        )
        result = await session.execute(query)
        items = result.scalars().all()

        overdue: list[ActionItem] = []
        not_yet_due: list[ActionItem] = []
        for item in items:
            due = item.due_date
            if due is not None and due.tzinfo is None:
                due = due.replace(tzinfo=UTC)
            if due is not None and due < now:
                overdue.append(item)
            else:
                not_yet_due.append(item)

        return OpenTaskGroup(not_yet_due=not_yet_due, overdue=overdue)

    @staticmethod
    async def mark_missed(session: AsyncSession, item_id: str, user_id: str) -> ActionItem:
        """Increment miss_count for a task confirmed overdue-and-not-done.

        Called when the agent (or the weekly check-in) chases an overdue
        task and the student confirms it still isn't done — drives the
        escalation logic (don't reassign a third time; address the blocker).
        """
        stmt = select(ActionItem).where(ActionItem.id == item_id, ActionItem.user_id == user_id)
        item = await session.scalar(stmt)
        if not item:
            raise ValueError("Item not found")

        item.miss_count += 1
        item.last_reminder_sent = datetime.now(UTC)
        item.updated_at = datetime.now(UTC)
        await session.commit()
        await session.refresh(item)
        return item

    @staticmethod
    async def update_action_item_status(
        session: AsyncSession,
        item_id: str,
        user_id: str,
        status: str,
        *,
        evidence: str | None = None,
    ) -> dict:
        stmt = select(ActionItem).where(ActionItem.id == item_id, ActionItem.user_id == user_id)
        item = await session.scalar(stmt)

        if not item:
            raise ValueError("Item not found")

        if evidence:
            item.evidence = evidence

        if item.status == status:
            await session.commit()
            return {"status": "updated", "message": "no_change"}

        item.status = status
        item.updated_at = datetime.now(UTC)

        # Update activity tracking on completion
        if status == "completed":
            await AccountabilityService._record_action_completion(session, user_id)

        await session.commit()
        return {"status": "updated"}

    @staticmethod
    async def _record_action_completion(session: AsyncSession, user_id: str) -> None:
        """Update activity tracking when an action item is completed."""
        result = await session.execute(select(UserActivityTracking).where(UserActivityTracking.user_id == user_id))
        activity = result.scalar_one_or_none()
        if not activity:
            activity = UserActivityTracking(user_id=user_id)
            session.add(activity)

        now = datetime.now(UTC)
        activity.last_action_completed = now
        activity.updated_at = now

        # Update streak
        today = now.date()
        if activity.last_streak_date:
            delta = (today - activity.last_streak_date).days
            if delta == 1:
                activity.current_streak += 1
            elif delta > 1:
                activity.current_streak = 1
        else:
            activity.current_streak = 1

        activity.last_streak_date = today
        if activity.current_streak > activity.longest_streak:
            activity.longest_streak = activity.current_streak

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------
    @staticmethod
    async def list_notifications(
        session: AsyncSession,
        user_id: str,
        limit: int = 50,
        status: str | None = "pending",
        category: str | None = None,
    ) -> Sequence[Notification]:
        """List notifications with optional status and category filters."""
        filters = [Notification.user_id == user_id]

        if status:
            filters.append(Notification.status == status)

        if category and category != "all":
            filters.append(Notification.category == category)

        query = select(Notification).where(*filters).order_by(Notification.created_at.desc()).limit(limit)
        result = await session.execute(query)
        return result.scalars().all()

    @staticmethod
    async def list_all_notifications(session: AsyncSession, user_id: str, limit: int = 50) -> Sequence[Notification]:
        """Return both pending and read notifications (for notification center)."""
        query = (
            select(Notification)
            .where(
                Notification.user_id == user_id,
                Notification.status.in_(["pending", "read"]),
            )
            .order_by(Notification.created_at.desc())
            .limit(limit)
        )
        result = await session.execute(query)
        return result.scalars().all()

    @staticmethod
    async def mark_notification_read(session: AsyncSession, notification_id: str, user_id: str) -> dict:
        stmt = select(Notification).where(Notification.id == notification_id, Notification.user_id == user_id)
        notification = await session.scalar(stmt)
        if not notification:
            raise ValueError("Notification not found")

        notification.status = "read"
        notification.read_at = datetime.now(UTC)
        await session.commit()
        return {"status": "updated"}

    @staticmethod
    async def mark_all_read(session: AsyncSession, user_id: str) -> dict:
        """Mark all pending notifications as read."""
        now = datetime.now(UTC)
        stmt = (
            update(Notification)
            .where(
                Notification.user_id == user_id,
                Notification.status == "pending",
            )
            .values(status="read", read_at=now)
        )
        result = await session.execute(stmt)
        await session.commit()
        return {"updated": result.rowcount}

    @staticmethod
    async def dismiss_notification(session: AsyncSession, notification_id: str, user_id: str) -> dict:
        """Permanently dismiss a notification."""
        from datetime import UTC
        from datetime import datetime as dt

        stmt = (
            update(Notification)
            .where(
                Notification.id == notification_id,
                Notification.user_id == user_id,
            )
            .values(status="dismissed", dismissed_at=dt.now(UTC))
        )
        result = await session.execute(stmt)
        if result.rowcount == 0:
            raise ValueError("Notification not found")
        await session.commit()
        return {"status": "dismissed"}

    # ------------------------------------------------------------------
    # User Preferences
    # ------------------------------------------------------------------
    @staticmethod
    async def get_preferences(session: AsyncSession, user_id: str) -> UserPreferences | None:
        result = await session.execute(select(UserPreferences).where(UserPreferences.user_id == user_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def upsert_preferences(session: AsyncSession, user_id: str, data: dict[str, Any]) -> UserPreferences:
        """Create or update user notification preferences."""
        result = await session.execute(select(UserPreferences).where(UserPreferences.user_id == user_id))
        prefs = result.scalar_one_or_none()

        if not prefs:
            prefs = UserPreferences(user_id=user_id)
            session.add(prefs)

        # Top-level fields
        if "notifications_enabled" in data:
            prefs.notifications_enabled = data["notifications_enabled"]
        if "location" in data:
            prefs.location = data["location"]
        if "push_subscription" in data:
            prefs.push_subscription = data["push_subscription"]

        # Merge into preferences JSONB.
        # Use dict() to create a new object — reassigning the same dict that SQLAlchemy
        # loaded as the committed state would not be detected as a change for JSONB columns
        # that don't use MutableDict, causing the UPDATE to be silently skipped.
        pref_json = dict(prefs.preferences) if prefs.preferences else {}
        for key in (
            "max_daily",
            "digest_mode",
            "quiet_hours_start",
            "quiet_hours_end",
            "disabled_categories",
            "email_enabled",
            "job_opportunity_mail_enabled",
            "job_opportunity_mail_frequency",
            "last_job_opportunity_digest_sent_at",
            "user_email",
            "user_name",
            "course_updates_enabled",
            "platform_updates_enabled",
            "ai_mentor_addon_active",
            "ai_mentor_addon_expires_at",
        ):
            if key in data:
                pref_json[key] = data[key]
        pref_json.setdefault("job_opportunity_mail_enabled", False)
        pref_json.setdefault("job_opportunity_mail_frequency", "weekly")
        prefs.preferences = pref_json
        flag_modified(prefs, "preferences")
        prefs.updated_at = datetime.now(UTC)

        await session.commit()
        return prefs

    # ------------------------------------------------------------------
    # Activity tracking
    # ------------------------------------------------------------------
    @staticmethod
    async def record_activity(session: AsyncSession, user_id: str, activity_type: str) -> None:
        """Record a user activity (login, conversation, course, etc.)."""
        result = await session.execute(select(UserActivityTracking).where(UserActivityTracking.user_id == user_id))
        activity = result.scalar_one_or_none()
        if not activity:
            activity = UserActivityTracking(user_id=user_id)
            session.add(activity)

        now = datetime.now(UTC)
        activity.updated_at = now

        if activity_type == "login":
            activity.last_login = now
        elif activity_type == "conversation":
            activity.last_conversation = now
        elif activity_type == "course":
            activity.last_course_activity = now

        await session.commit()

    @staticmethod
    async def get_activity(session: AsyncSession, user_id: str) -> UserActivityTracking | None:
        result = await session.execute(select(UserActivityTracking).where(UserActivityTracking.user_id == user_id))
        return result.scalar_one_or_none()

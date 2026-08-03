"""Enhanced Scheduler service for the Accountability Partner system.

Aligned with Requirements Specification v1.0.

Features:
- Tiered deadline reminders (7d, 3d, 24h, 2h, overdue, severe overdue) [§2.2.1]
- Inactivity detection (3d, 6d, 10d, 15d) with risk scoring [§2.2.2]
- Progress celebration checks [§2.4.1]
- Struggle detection & intervention [§2.4.2]
- Motivational nudges (Mon/Wed/Fri/Sun) [§2.4.3]
- Daily digest generation [§2.6.1]
- Notification cleanup (90-day retention) [§2.6]
- Opportunity expiration + twice-daily discovery (5 AM + 5 PM UTC)
- Daily opportunity refresh: clears stale unactioned data at 5 AM UTC, then
  immediately re-discovers fresh, location-personalised opportunities
- Concurrent per-user discovery with semaphore (max 5 in parallel)
- Frequency-aware notification creation via NotificationEngine
"""

import asyncio
import re
from datetime import UTC, datetime, timedelta

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler  # type: ignore[import-untyped]
from apscheduler.triggers.cron import CronTrigger  # type: ignore[import-untyped]
from apscheduler.triggers.date import DateTrigger  # type: ignore[import-untyped]
from apscheduler.triggers.interval import IntervalTrigger  # type: ignore[import-untyped]
from sqlalchemy import and_, delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm.attributes import flag_modified

from aegra_api.core.accountability_orm import (
    ActionItem,
    DiscoveredOpportunity,
    Notification,
    UserActivityTracking,
    UserPreferences,
)
from aegra_api.core.database import db_manager
from aegra_api.data.career_advisors import get_advisor_by_track, get_default_advisor
from aegra_api.services.accountability_service import AccountabilityService
from aegra_api.services.advisor_memory import fetch_advisor_memory_context
from aegra_api.services.email_service import build_digest_email, resolve_student_contact, send_email
from aegra_api.services.notification_engine import notification_engine
from aegra_api.services.opportunity_discovery import opportunity_engine
from aegra_api.settings import settings
from aegra_api.tools.course_content.mongo_client import get_course_content_mongo_client

# weekly_checkin_service is imported lazily inside the job method to avoid
# a circular import: weekly_checkin → cron_service → langgraph_service.

logger = structlog.getLogger(__name__)


def _parse_digest_timestamp(raw_value: str | None) -> datetime | None:
    if not raw_value:
        return None
    try:
        return datetime.fromisoformat(raw_value)
    except ValueError:
        return None


class SchedulerService:
    def __init__(self) -> None:
        self.scheduler = AsyncIOScheduler()

    @staticmethod
    async def _resolve_advisor_for_user(user_id: str) -> tuple[str, str]:
        """Return (advisor_first_name, normalised_track) for a user.

        Reads learningTrack from the onboardings collection — the canonical source
        set during sign-up. Falls back to ("Alexandra", "") when not found.
        """
        mongo_client = get_course_content_mongo_client()
        try:
            raw_track = await asyncio.to_thread(mongo_client.get_learning_track, user_id)
            if raw_track:
                normalised = raw_track.lower().strip().replace(" ", "-")
                advisor = get_advisor_by_track(normalised)
                if advisor:
                    return advisor["name"].split()[0], normalised
        except Exception as exc:
            logger.warning("advisor_resolve_failed", user_id=user_id, error=str(exc))

        return get_default_advisor()["name"].split()[0], ""

    @staticmethod
    async def _resolve_advisor_first_name(user_id: str) -> str:
        """Convenience wrapper — returns only the advisor first name."""
        name, _ = await SchedulerService._resolve_advisor_for_user(user_id)
        return name

    def start(self) -> None:
        if not self.scheduler.running:
            # Deadline reminders — every 15 min
            self.scheduler.add_job(
                self.check_deadlines,
                IntervalTrigger(minutes=15),
                id="check_deadlines",
                replace_existing=True,
            )
            # Inactivity check — every 6 h
            self.scheduler.add_job(
                self.check_inactivity,
                IntervalTrigger(hours=6),
                id="check_inactivity",
                replace_existing=True,
            )
            # Celebration check — every 4 h
            self.scheduler.add_job(
                self.check_celebrations,
                IntervalTrigger(hours=4),
                id="check_celebrations",
                replace_existing=True,
            )
            # Cleanup old notifications — every 12 h
            self.scheduler.add_job(
                self.check_cleanup,
                IntervalTrigger(hours=12),
                id="check_cleanup",
                replace_existing=True,
            )
            # Expire opportunities — hourly
            self.scheduler.add_job(
                self.expire_opportunities,
                IntervalTrigger(hours=1),
                id="expire_opportunities",
                replace_existing=True,
            )
            # Discovery — 5 PM UTC (paired with 5 AM daily_refresh, giving true twice-daily)
            self.scheduler.add_job(
                self.run_discovery_job,
                CronTrigger(hour=17),
                id="run_discovery_job",
                replace_existing=True,
            )
            # Daily opportunity refresh — clear stale data + re-discover at 5 AM UTC
            self.scheduler.add_job(
                self.daily_refresh_opportunities,
                CronTrigger(hour=5),
                id="daily_refresh_opportunities",
                replace_existing=True,
            )
            # Motivational nudges — Mon/Wed/Fri/Sun at 9:00 AM UTC [§2.4.3]
            self.scheduler.add_job(
                self.send_motivational_nudges,
                CronTrigger(day_of_week="mon,wed,fri,sun", hour=9),
                id="motivational_nudges",
                replace_existing=True,
            )
            # Daily digest — every day at 8:00 PM UTC [§2.6.1]
            self.scheduler.add_job(
                self.generate_daily_digest,
                CronTrigger(hour=20),
                id="daily_digest",
                replace_existing=True,
            )
            # Struggle detection — every 8 hours [§2.4.2]
            self.scheduler.add_job(
                self.check_struggles,
                IntervalTrigger(hours=8),
                id="check_struggles",
                replace_existing=True,
            )
            # One-shot backfill: enable Jobs email for all paid users on startup
            self.scheduler.add_job(
                self.backfill_job_opportunity_opt_in,
                DateTrigger(run_date=datetime.now(UTC)),
                id="backfill_job_opportunity_opt_in",
                replace_existing=True,
            )

            # Provision weekly advisor check-in crons for any user who
            # doesn't have one yet. Runs once on startup then daily so
            # newly signed-up users are picked up automatically.
            self.scheduler.add_job(
                self.provision_weekly_checkins,
                IntervalTrigger(hours=24),
                id="provision_weekly_checkins",
                replace_existing=True,
                next_run_time=datetime.now(UTC),
            )

            self.scheduler.start()
            logger.info("Scheduler started with all accountability jobs")

    def shutdown(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown()

    # ------------------------------------------------------------------
    # One-shot startup backfill
    # ------------------------------------------------------------------
    async def backfill_job_opportunity_opt_in(self) -> None:
        """Enable Jobs & Opportunity email for existing paid users who haven't opted in.

        Runs once on startup. For every UserPreferences row:
        - If ``job_opportunity_mail_enabled`` is not True: checks the LMS Mongo
          subscriptions and sets ``ai_mentor_addon_active`` + ``job_opportunity_mail_enabled``
          for users with an active AI Mentor add-on.
        - For all AI Mentor addon users missing ``user_email``: resolves and caches
          the email address from the LMS so notification sends don't need a live
          LMS lookup every time.
        """
        from aegra_api.services.email_service import resolve_student_contact

        try:
            if not db_manager.engine:
                return
            session_maker = async_sessionmaker(db_manager.engine, expire_on_commit=False)
            async with session_maker() as session:
                result = await session.execute(select(UserPreferences))
                all_prefs = result.scalars().all()

                updated = 0
                email_cached = 0
                for prefs in all_prefs:
                    pref_json = prefs.preferences or {}
                    new_pref_json = dict(pref_json)
                    changed = False

                    # ── Opt-in pass: enable job mail for AI Mentor users ──────
                    if not pref_json.get("job_opportunity_mail_enabled", False):
                        try:
                            sub = await asyncio.to_thread(
                                get_course_content_mongo_client().get_subscription_state,
                                prefs.user_id,
                            )
                            if sub:
                                addon = sub.get("aiMentorAddOn") or {}
                                if addon.get("active", False):
                                    new_pref_json["ai_mentor_addon_active"] = True
                                    new_pref_json["job_opportunity_mail_enabled"] = True
                                    changed = True
                                    updated += 1
                        except Exception as exc:
                            logger.warning("backfill_opt_in_user_failed", user_id=prefs.user_id, error=str(exc))

                    # ── Email cache pass: store user_email / fix placeholder names ─
                    _stored_name = new_pref_json.get("user_name", "")
                    _name_is_placeholder = _stored_name.startswith("User ") and len(_stored_name) > 5
                    _needs_email = new_pref_json.get("ai_mentor_addon_active") and not new_pref_json.get("user_email")
                    _needs_name = new_pref_json.get("ai_mentor_addon_active") and (
                        not _stored_name or _name_is_placeholder
                    )
                    if _needs_email or _needs_name:
                        try:
                            contact = await resolve_student_contact(prefs.user_id)
                            if contact.get("email"):
                                new_pref_json["user_email"] = contact["email"]
                                changed = True
                                email_cached += 1
                            if contact.get("first_name") and _needs_name:
                                new_pref_json["user_name"] = contact["first_name"]
                                changed = True
                        except Exception as exc:
                            logger.warning("backfill_email_cache_failed", user_id=prefs.user_id, error=str(exc))

                    if changed:
                        prefs.preferences = new_pref_json
                        flag_modified(prefs, "preferences")

                if updated > 0 or email_cached > 0:
                    await session.commit()
                logger.info(
                    "backfill_job_opportunity_opt_in_complete",
                    users_updated=updated,
                    emails_cached=email_cached,
                )
        except Exception as exc:
            logger.error("backfill_job_opportunity_opt_in_failed", error=str(exc), exc_info=True)

    # ------------------------------------------------------------------
    # Weekly advisor check-in provisioning
    # ------------------------------------------------------------------
    async def provision_weekly_checkins(self) -> None:
        """Ensure every active user has a thread-bound weekly check-in cron.

        Runs on startup and every 24 h so newly signed-up users are picked
        up automatically without requiring a server restart.
        """
        from aegra_api.services.langgraph_service import get_langgraph_service
        from aegra_api.services.weekly_checkin_service import weekly_checkin_service

        try:
            if not db_manager.engine:
                return
            session_maker = async_sessionmaker(db_manager.engine, expire_on_commit=False)
            async with session_maker() as session:
                await weekly_checkin_service.provision_all_pending(
                    session=session,
                    langgraph_service=get_langgraph_service(),
                )
        except Exception as exc:
            logger.error("provision_weekly_checkins_failed", error=str(exc), exc_info=True)

    # ------------------------------------------------------------------
    # Deadline reminders
    # ------------------------------------------------------------------
    async def check_deadlines(self) -> None:
        """Generate tiered deadline notifications through NotificationEngine."""
        try:
            if not db_manager.engine:
                return
            session_maker = async_sessionmaker(db_manager.engine, expire_on_commit=False)
            async with session_maker() as session:
                now = datetime.now(UTC)

                result = await session.execute(
                    select(ActionItem).where(
                        and_(
                            ActionItem.status.in_(["pending", "in_progress"]),
                            ActionItem.due_date.isnot(None),
                        )
                    )
                )
                items = result.scalars().all()

                for item in items:
                    if not item.due_date:
                        continue

                    tier, priority, title, content_tpl = notification_engine.compute_deadline_tier(item.due_date, now)
                    if not tier:
                        continue

                    # Check if we should send based on last reminder timing
                    should_send = await self._should_send_reminder(item, tier, now)
                    if not should_send:
                        continue

                    # Compute template values
                    hours_diff = (item.due_date - now).total_seconds() / 3600
                    days = max(1, abs(int(hours_diff / 24)))
                    content = content_tpl.format(description=item.description, days=days)

                    # Older/externally-created tasks may not have a persona stamped
                    # on them — fall back to a live resolution rather than letting
                    # create_notification's own DEFAULT_PERSONA fallback silently
                    # pick the wrong advisor for this student's track.
                    task_persona = item.advisor_persona or await self._resolve_advisor_first_name(item.user_id)

                    notif = await notification_engine.create_notification(
                        session=session,
                        user_id=item.user_id,
                        title=title,
                        content=content,
                        category="deadline",
                        priority=priority,
                        persona=task_persona,
                        action_buttons=[
                            {"action": "complete", "title": "Mark Complete"},
                            {"action": "snooze", "title": "Snooze 1hr"},
                            {
                                "action": "chat",
                                "title": "Talk to Advisor",
                                "url": "/dashboard/ai-career-advisor",
                            },
                        ],
                        metadata={"action_item_id": item.id, "reminder_tier": tier},
                        check_frequency=True,
                    )

                    if notif:
                        item.reminder_sent_count += 1
                        item.last_reminder_sent = now
                        item.last_reminder_tier = tier
                        logger.info(
                            "deadline_reminder_sent",
                            item_id=item.id,
                            tier=tier,
                            count=item.reminder_sent_count,
                        )

                await session.commit()
        except Exception as e:
            logger.error("check_deadlines error", error=str(e), exc_info=True)

    async def _should_send_reminder(self, item: ActionItem, tier: str, now: datetime) -> bool:
        """Send once per tier crossing; only overdue tiers may repeat, and only with restraint.

        check_deadlines runs every 15 minutes, so gating on elapsed time alone
        (the previous behaviour) resent the SAME tier's identical email every
        ~4h for as long as the task sat in that tier — up to a dozen "Due
        This Week" emails before the task ever reached "3d". A tier is a
        one-time milestone crossing, not a recurring state to keep notifying.
        """
        if item.reminder_sent_count == 0:
            return True
        if item.last_reminder_tier != tier:
            return True  # escalated (or moved) to a different tier — worth a fresh nudge

        # Same tier as last time: 7d/3d/24h/2h never repeat within themselves.
        # Only overdue tiers keep chasing, and even then at most once a day,
        # capped at 5 total sends.
        if not tier.startswith("overdue"):
            return False
        if item.reminder_sent_count >= 5:
            return False
        if item.last_reminder_sent:
            hours_since = (now - item.last_reminder_sent).total_seconds() / 3600
            if hours_since < 24:
                return False
        return True

    # ------------------------------------------------------------------
    # Inactivity detection
    # ------------------------------------------------------------------
    async def check_inactivity(self) -> None:
        try:
            if not db_manager.engine:
                return
            session_maker = async_sessionmaker(db_manager.engine, expire_on_commit=False)
            async with session_maker() as session:
                now = datetime.now(UTC)
                three_days_ago = now - timedelta(days=3)

                result = await session.execute(
                    select(UserActivityTracking).where(
                        or_(
                            UserActivityTracking.last_login < three_days_ago,
                            UserActivityTracking.last_conversation < three_days_ago,
                        )
                    )
                )
                inactive_users = result.scalars().all()

                for activity in inactive_users:
                    last_activity = max(
                        filter(
                            None,
                            [
                                activity.last_login,
                                activity.last_conversation,
                                activity.last_course_activity,
                            ],
                        ),
                        default=None,
                    )
                    if not last_activity:
                        continue

                    days_inactive = (now - last_activity).days
                    tier, priority, title, content_tpl = notification_engine.compute_inactivity_tier(days_inactive)
                    if not tier:
                        continue

                    # Deduplicate — only one inactivity notification per 3 days
                    exists = await session.execute(
                        select(Notification).where(
                            and_(
                                Notification.user_id == activity.user_id,
                                Notification.category == "inactivity",
                                Notification.created_at > (now - timedelta(days=3)),
                            )
                        )
                    )
                    if exists.scalars().first():
                        continue

                    content = content_tpl.format(days=days_inactive)
                    advisor_name, track = await self._resolve_advisor_for_user(activity.user_id)
                    # Real student context (spec Item 7) so generate_persona_message
                    # can name an actual open task instead of leaving the elapsed-time
                    # template as generic as pure inactivity metrics alone would be.
                    student_ctx = await self._build_student_context(session, activity.user_id, advisor_name, track)

                    sent = await notification_engine.create_notification(
                        session=session,
                        user_id=activity.user_id,
                        title=title,
                        content=content,
                        category="inactivity",
                        priority=priority,
                        persona=advisor_name,
                        student_context=student_ctx,
                        action_buttons=[
                            {
                                "action": "resume",
                                "title": "Resume Learning",
                                "url": "/dashboard/my-tracks",
                            },
                            {
                                "action": "chat",
                                "title": "Talk to Advisor",
                                "url": "/dashboard/ai-career-advisor",
                            },
                        ],
                        check_frequency=True,
                    )
                    if sent:
                        logger.info(
                            "inactivity_notification",
                            user_id=activity.user_id,
                            days=days_inactive,
                            tier=tier,
                        )

                await session.commit()
        except Exception as e:
            logger.error("check_inactivity error", error=str(e), exc_info=True)

    # ------------------------------------------------------------------
    # Progress celebrations
    # ------------------------------------------------------------------
    async def check_celebrations(self) -> None:
        try:
            if not db_manager.engine:
                return
            session_maker = async_sessionmaker(db_manager.engine, expire_on_commit=False)
            async with session_maker() as session:
                result = await session.execute(select(UserActivityTracking.user_id))
                user_ids = result.scalars().all()

                for user_id in user_ids:
                    celebrations = await notification_engine.check_celebrations(session, user_id)
                    for cel in celebrations:
                        # Deduplicate by type
                        exists = await session.execute(
                            select(Notification).where(
                                and_(
                                    Notification.user_id == user_id,
                                    Notification.category == "celebration",
                                    Notification.metadata_json["celebration_type"].astext == cel["type"],
                                    Notification.created_at > (datetime.now(UTC) - timedelta(days=1)),
                                )
                            )
                        )
                        if exists.scalars().first():
                            continue

                        # Resolve the student's actual advisor so the email
                        # sign-off matches their track instead of silently
                        # falling through to create_notification's own
                        # DEFAULT_PERSONA ("Alexandra") for every student.
                        advisor_name = await self._resolve_advisor_first_name(user_id)

                        await notification_engine.create_notification(
                            session=session,
                            user_id=user_id,
                            title=cel["title"],
                            content=cel["content"],
                            category="celebration",
                            priority=cel.get("priority", "normal"),
                            metadata={"celebration_type": cel["type"]},
                            persona=advisor_name,
                            check_frequency=True,
                        )

                await session.commit()
        except Exception as e:
            logger.error("check_celebrations error", error=str(e), exc_info=True)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    async def check_cleanup(self) -> None:
        """Tiered notification cleanup to prevent DB bloat.

        Retention tiers:
        - Expired (expires_at in the past): 1 hour grace period after expiry
        - Everything else (all statuses): 3 days
        """
        try:
            if not db_manager.engine:
                return
            session_maker = async_sessionmaker(db_manager.engine, expire_on_commit=False)
            async with session_maker() as session:
                now = datetime.now(UTC)
                total_deleted = 0

                # 1. Expired notifications — delete 1 hour after their expiry time
                expired_cutoff = now - timedelta(hours=1)
                stmt = delete(Notification).where(
                    and_(
                        Notification.expires_at.isnot(None),
                        Notification.expires_at < expired_cutoff,
                    )
                )
                result = await session.execute(stmt)
                total_deleted += result.rowcount

                # 2. Everything else — keep 3 days regardless of status
                general_cutoff = now - timedelta(days=3)
                stmt = delete(Notification).where(Notification.created_at < general_cutoff)
                result = await session.execute(stmt)
                total_deleted += result.rowcount

                if total_deleted > 0:
                    logger.info("notification_cleanup", deleted=total_deleted)
                await session.commit()
        except Exception as e:
            logger.error("check_cleanup error", error=str(e), exc_info=True)

    # ------------------------------------------------------------------
    # Opportunity expiration
    # ------------------------------------------------------------------
    async def expire_opportunities(self) -> None:
        try:
            if not db_manager.engine:
                return
            session_maker = async_sessionmaker(db_manager.engine, expire_on_commit=False)
            async with session_maker() as session:
                now = datetime.now(UTC)
                stmt = (
                    update(DiscoveredOpportunity)
                    .where(
                        and_(
                            DiscoveredOpportunity.expires_at < now,
                            DiscoveredOpportunity.status.notin_(["expired", "dismissed", "applied"]),
                        )
                    )
                    .values(status="expired")
                )
                result = await session.execute(stmt)
                if result.rowcount > 0:
                    logger.info("opportunities_expired", count=result.rowcount)
                await session.commit()
        except Exception as e:
            logger.error("expire_opportunities error", error=str(e), exc_info=True)

    # ------------------------------------------------------------------
    # Daily opportunity refresh (clear stale + re-discover)
    # ------------------------------------------------------------------
    async def daily_refresh_opportunities(self) -> None:
        """Delete yesterday's unsaved/unapplied opportunities and run fresh discovery.

        Runs once per day at 5 AM UTC so the board always shows live, personalised data.
        Preserved statuses: 'saved', 'applied', 'dismissed' — these are user actions.
        Cleared statuses: 'new', 'notified', 'expired'.
        """
        logger.info("daily_refresh_opportunities_started")
        try:
            if not db_manager.engine:
                logger.warning("daily_refresh_aborted", reason="no db engine")
                return
            session_maker = async_sessionmaker(db_manager.engine, expire_on_commit=False)
            async with session_maker() as session:
                stmt = delete(DiscoveredOpportunity).where(
                    DiscoveredOpportunity.status.in_(["new", "notified", "expired"])
                )
                result = await session.execute(stmt)
                deleted_count = result.rowcount
                await session.commit()
                logger.info("daily_refresh_stale_cleared", deleted=deleted_count)

            # Run fresh discovery for all active users immediately after clearing
            await self.run_discovery_job()
            logger.info("daily_refresh_opportunities_complete")
        except Exception as e:
            logger.error("daily_refresh_opportunities error", error=str(e), exc_info=True)

    # ------------------------------------------------------------------
    # Discovery job
    # ------------------------------------------------------------------

    # Maximum number of users to discover for concurrently.
    # Keeps external job-board API pressure bounded.
    _DISCOVERY_CONCURRENCY = 5

    async def _discover_and_notify_user(self, session_maker: async_sessionmaker, user_id: str) -> None:  # type: ignore[type-arg]
        """Run discovery for a single user and create in-app/web-push notifications."""
        queries_per_category = settings.discovery.DISCOVERY_QUERIES_PER_CATEGORY
        try:
            logger.info("discovery_job_user_start", user_id=user_id)
            # Each user gets its own session to avoid cross-user state leakage
            async with session_maker() as session:
                discovered = await opportunity_engine.discover_for_user(
                    session=session,
                    user_id=user_id,
                    auth_token="",  # nosec B106
                    max_tracks=1,
                    queries_per_category=queries_per_category,
                )
                logger.info(
                    "discovery_job_user_done",
                    user_id=user_id,
                    opportunities_found=len(discovered),
                )
                for opp in discovered:
                    type_label = opp.opportunity_type
                    if type_label == "event":
                        title = "🎯 New Event Matches Your Track"
                        content = f"We found a {opp.matched_track} event: {opp.title}"
                    else:
                        company_part = f" at {opp.company}" if opp.company else ""
                        title = "💼 Job Opportunity Alert"
                        content = f"New {opp.matched_track} role: {opp.title}{company_part}"

                    await notification_engine.create_notification(
                        session=session,
                        user_id=user_id,
                        title=title,
                        content=content,
                        priority="normal",
                        category="opportunity",
                        action_buttons=[
                            {"action": "view", "title": "View", "url": opp.url},
                            {"action": "dismiss", "title": "Dismiss"},
                        ],
                        metadata={
                            "opportunity_id": opp.id,
                            "opportunity_type": opp.opportunity_type,
                            "url": opp.url,
                        },
                        check_frequency=False,
                    )
                    opp.status = "notified"
                await session.commit()
                logger.info("discovery_job_notifications_sent", user_id=user_id, count=len(discovered))
        except Exception as e:
            logger.error("discovery_failed_for_user", user_id=user_id, error=str(e), exc_info=True)

    async def run_discovery_job(self) -> None:
        """Periodic opportunity discovery for all active users, run concurrently.

        We query the union of UserActivityTracking and UserPreferences so that
        newly-onboarded users (who have preferences but have not yet triggered any
        activity-tracking events) are always included in the daily discovery run.
        Without this, the daily_refresh deletes their stale opportunities but never
        re-discovers new ones, leaving their job/event boards empty.
        """
        logger.info("discovery_job_started")
        try:
            if not db_manager.engine:
                logger.warning("discovery_job_aborted", reason="no db engine")
                return
            session_maker = async_sessionmaker(db_manager.engine, expire_on_commit=False)
            async with session_maker() as session:
                activity_result = await session.execute(select(UserActivityTracking.user_id))
                prefs_result = await session.execute(select(UserPreferences.user_id))
                # Union both sources — de-duplicate with a set
                user_ids = list({*activity_result.scalars().all(), *prefs_result.scalars().all()})

            logger.info("discovery_job_users_found", count=len(user_ids))
            if not user_ids:
                logger.warning("discovery_job_no_users", reason="no users in activity_tracking or user_preferences")
                return

            semaphore = asyncio.Semaphore(self._DISCOVERY_CONCURRENCY)

            async def _bounded(uid: str) -> None:
                async with semaphore:
                    await self._discover_and_notify_user(session_maker, uid)

            await asyncio.gather(*(_bounded(uid) for uid in user_ids))
            logger.info("discovery_job_complete", user_count=len(user_ids))
        except Exception as e:
            logger.error("run_discovery_job error", error=str(e), exc_info=True)

    # ------------------------------------------------------------------
    # Motivational nudges [§2.4.3]
    # ------------------------------------------------------------------

    async def _build_student_context(
        self,
        session: AsyncSession,
        user_id: str,
        advisor_name: str,
        track: str,
    ) -> dict:
        """Fetch per-user metrics from the DB and return a context dict.

        Used to generate personalised email content that references each
        student's actual streak, task counts, and career goal rather than
        sending the same generic message to everyone.
        """
        now = datetime.now(UTC)
        week_ago = now - timedelta(days=7)
        context: dict = {
            "learning_track": track,
            "advisor_name": advisor_name,
            "current_streak": 0,
            "tasks_completed_this_week": 0,
            "overdue_tasks": 0,
            "pending_tasks": 0,
            # Named tasks, not just counts (spec Item 7) — [{description, due_date, miss_count}, ...]
            "overdue_task_details": [],
            "pending_task_details": [],
            # Advisor memory (spec Item 7) — shared history + learned tone +
            # durable facts, all via context_assembly (shared with live chat)
            "relevant_episode": "",
            "behavior_profile": "",
            "semantic_context": "",
            "first_name": "",
            "primary_goal": "",
            # Course progress fields — populated from MongoDB enrollment data
            "enrolled_course": "",
            "course_progress_pct": 0,
            "total_completed_lessons": 0,
            "total_watched_hours": 0.0,
            "last_active_in_course": "",
        }

        try:
            # ── Student name (from cached preferences) ───────────────
            prefs_result = await session.execute(select(UserPreferences).where(UserPreferences.user_id == user_id))
            prefs = prefs_result.scalar_one_or_none()
            if prefs and prefs.preferences:
                pref_json = prefs.preferences
                raw_name = pref_json.get("user_name", "")
                # Detect the "User <hex_id>" placeholder stored when the JWT
                # profile has no real name — treat as missing so we fall through
                # to the resolve_student_contact() lookup below.
                if raw_name and re.match(r"^User\s+[0-9a-f]{10,}$", raw_name, re.IGNORECASE):
                    raw_name = ""
                context["first_name"] = raw_name.split()[0] if raw_name else ""
                stored_email = pref_json.get("user_email")
                if stored_email:
                    context["email"] = stored_email

            # ── Real name fallback via admin API ──────────────────────
            if not context["first_name"]:
                try:
                    from aegra_api.services.email_service import resolve_student_contact

                    contact = await resolve_student_contact(user_id)
                    if contact.get("first_name"):
                        context["first_name"] = contact["first_name"]
                    if not context.get("email") and contact.get("email"):
                        context["email"] = contact["email"]
                except Exception:  # nosec B110
                    pass

            # ── Streak & activity ─────────────────────────────────────
            activity_result = await session.execute(
                select(UserActivityTracking).where(UserActivityTracking.user_id == user_id)
            )
            activity = activity_result.scalar_one_or_none()
            if activity:
                context["current_streak"] = activity.current_streak or 0

            # ── Task counts ───────────────────────────────────────────
            completed_result = await session.execute(
                select(func.count(ActionItem.id)).where(
                    and_(
                        ActionItem.user_id == user_id,
                        ActionItem.status == "completed",
                        ActionItem.updated_at >= week_ago,
                    )
                )
            )
            context["tasks_completed_this_week"] = completed_result.scalar() or 0

            overdue_result = await session.execute(
                select(func.count(ActionItem.id)).where(
                    and_(
                        ActionItem.user_id == user_id,
                        ActionItem.status.in_(["pending", "in_progress"]),
                        ActionItem.due_date < now,
                    )
                )
            )
            context["overdue_tasks"] = overdue_result.scalar() or 0

            pending_result = await session.execute(
                select(func.count(ActionItem.id)).where(
                    and_(
                        ActionItem.user_id == user_id,
                        ActionItem.status.in_(["pending", "in_progress"]),
                        or_(ActionItem.due_date >= now, ActionItem.due_date.is_(None)),
                    )
                )
            )
            context["pending_tasks"] = pending_result.scalar() or 0

            # ── Named tasks (spec Item 7) ──────────────────────────────
            # Same query AccountabilityService.get_open_and_overdue uses for
            # conversation-start injection — outreach and live chat read from
            # the same source of truth, not two independently-computed views.
            task_group = await AccountabilityService.get_open_and_overdue(session, user_id)
            context["overdue_task_details"] = [
                {
                    "description": item.description,
                    "due_date": item.due_date.isoformat() if item.due_date else None,
                    "miss_count": item.miss_count,
                }
                for item in task_group.overdue
            ]
            context["pending_task_details"] = [
                {
                    "description": item.description,
                    "due_date": item.due_date.isoformat() if item.due_date else None,
                }
                for item in task_group.not_yet_due
            ]

            # ── Onboarding goal (best-effort from Mongo) ──────────────
            try:
                onboarding = await asyncio.to_thread(
                    get_course_content_mongo_client().get_user_onboarding_data,
                    user_id,
                )
                if onboarding:
                    context["primary_goal"] = onboarding.get("target_role", "") or ""
            except Exception:  # nosec B110 — best-effort Mongo read; failure is non-fatal
                pass

            # ── Course enrollment & progress (best-effort from Mongo) ──
            try:
                enrollment_data = await asyncio.to_thread(
                    get_course_content_mongo_client().get_enrollment_overview,
                    user_id,
                )
                enrollments = (enrollment_data or {}).get("enrollments", [])
                if enrollments:
                    # Primary course = first active enrollment returned
                    primary = enrollments[0]
                    course_info = primary.get("course") or {}
                    context["enrolled_course"] = course_info.get("title") or ""
                    context["course_progress_pct"] = int(primary.get("overallProgress") or 0)

                    # Aggregate across all enrollments
                    context["total_completed_lessons"] = sum(
                        int(e.get("totalCompletedLessons") or 0) for e in enrollments
                    )
                    context["total_watched_hours"] = round(
                        sum(float(e.get("totalWatchedHours") or 0) for e in enrollments), 1
                    )

                    # Most recently active enrollment date
                    most_recent = max(
                        (e for e in enrollments if e.get("updatedAt")),
                        key=lambda e: e["updatedAt"],
                        default=None,
                    )
                    if most_recent and most_recent.get("updatedAt"):
                        updated = most_recent["updatedAt"]
                        if hasattr(updated, "strftime"):
                            context["last_active_in_course"] = updated.strftime("%b %d")
                        else:
                            # ISO string — trim to date portion
                            context["last_active_in_course"] = str(updated)[:10]
            except Exception:  # nosec B110 — best-effort Mongo read; failure is non-fatal
                pass

            # ── Advisor memory (spec Item 7) — episode + tone profile ──
            # Semantic query built from what outreach is about to discuss, so
            # the recalled episode is relevant, not just the most recent one.
            memory_query_parts = [t["description"] for t in context["overdue_task_details"][:2]]
            if context["primary_goal"]:
                memory_query_parts.append(context["primary_goal"])
            memory_ctx = await fetch_advisor_memory_context(user_id, query="; ".join(memory_query_parts) or None)
            context.update(memory_ctx)

        except Exception as exc:
            logger.warning("student_context_build_failed", user_id=user_id, error=str(exc))

        return context

    async def send_motivational_nudges(self) -> None:
        """Send personalised motivational nudges on Mon/Wed/Fri/Sun per spec §2.4.3.

        Unlike the previous implementation that sent identical copy to every user,
        this version:
        1. Resolves the *correct* advisor for each user's learning track.
        2. Fetches that user's real metrics (streak, tasks done, overdue count).
        3. Uses an LLM to generate a unique message that references those numbers.

        The email body is generated separately from the short in-app notification so
        that each student receives content genuinely written for them.
        """
        try:
            if not db_manager.engine:
                return
            session_maker = async_sessionmaker(db_manager.engine, expire_on_commit=False)
            async with session_maker() as session:
                result = await session.execute(select(UserActivityTracking.user_id))
                user_ids = result.scalars().all()

                sent = 0
                for user_id in user_ids:
                    try:
                        advisor_name, track = await self._resolve_advisor_for_user(user_id)
                        student_ctx = await self._build_student_context(session, user_id, advisor_name, track)

                        # Generate personalised title + email body via LLM
                        email_title, email_body = await notification_engine.generate_personalized_motivational_content(
                            persona_name=advisor_name,
                            student_context=student_ctx,
                        )

                        # Short in-app notification — a single sentence drawn from the context
                        streak = student_ctx["current_streak"]
                        overdue = student_ctx["overdue_tasks"]
                        overdue_details = student_ctx["overdue_task_details"]
                        if streak > 0:
                            short_content = f"You're on a {streak}-day streak — keep it going! 🔥"
                        elif overdue_details:
                            # Name the actual task, not a bare count (spec Item 7).
                            first_task = overdue_details[0].get("description", "")
                            short_content = f"'{first_task}' is still open. Let's tackle it together."
                        elif overdue:
                            short_content = f"You have {overdue} overdue task{'s' if overdue != 1 else ''}. Let's clear the backlog together."
                        else:
                            short_content = "Check in with your advisor to keep your career journey on track."

                        await notification_engine.create_notification(
                            session=session,
                            user_id=user_id,
                            title=email_title,
                            content=short_content,
                            category="motivation",
                            priority="low",
                            persona=None,  # body already personalised — skip LLM rewrite
                            # But the email sign-off still needs the correct advisor —
                            # explicit, since persona=None would otherwise fall through
                            # to create_notification's DEFAULT_PERSONA (spec Item 7).
                            advisor_persona=advisor_name,
                            action_buttons=[
                                {
                                    "action": "chat",
                                    "title": f"Chat with {advisor_name}",
                                    "url": "/dashboard/ai-career-advisor",
                                },
                            ],
                            check_frequency=True,
                            student_context=student_ctx,
                            email_body_override=email_body,  # rich personalised body for email
                        )
                        sent += 1
                    except Exception as e:
                        logger.warning("motivational_nudge_failed", user_id=user_id, error=str(e))

                await session.commit()
                logger.info("motivational_nudges_sent", user_count=len(user_ids), emails_attempted=sent)
        except Exception as e:
            logger.error("send_motivational_nudges error", error=str(e), exc_info=True)

    # ------------------------------------------------------------------
    # Daily digest [§2.6.1]
    # ------------------------------------------------------------------
    async def generate_daily_digest(self) -> None:
        """Send job and opportunity digests according to student email cadence."""
        try:
            if not db_manager.engine:
                return
            session_maker = async_sessionmaker(db_manager.engine, expire_on_commit=False)
            async with session_maker() as session:
                result = await session.execute(select(UserPreferences))
                preference_rows = result.scalars().all()
                now = datetime.now(UTC)

                for prefs in preference_rows:
                    try:
                        pref_json = (prefs.preferences or {}).copy()
                        if not pref_json.get("job_opportunity_mail_enabled", False):
                            continue

                        # Require active AI Mentor add-on.
                        # The ai_mentor_addon_active flag is set to True at preference-save
                        # time (PUT /preferences enforces subscription check). The expiry
                        # date is NOT re-checked here because it is a stale cached value —
                        # renewed subscriptions don't update the cached date, causing all
                        # users to be incorrectly blocked once the original expiry passes.
                        if not pref_json.get("ai_mentor_addon_active", False):
                            continue

                        frequency = pref_json.get("job_opportunity_mail_frequency", "weekly")
                        if frequency not in {"daily", "weekly"}:
                            frequency = "weekly"

                        last_sent_at = _parse_digest_timestamp(pref_json.get("last_job_opportunity_digest_sent_at"))
                        if frequency == "daily" and last_sent_at and (now - last_sent_at) < timedelta(hours=20):
                            continue
                        if frequency == "weekly" and last_sent_at and (now - last_sent_at) < timedelta(days=7):
                            continue

                        window_start = last_sent_at or (
                            now - (timedelta(days=7) if frequency == "weekly" else timedelta(days=1))
                        )

                        notif_result = await session.execute(
                            select(Notification)
                            .where(
                                and_(
                                    Notification.user_id == prefs.user_id,
                                    Notification.category == "opportunity",
                                    Notification.created_at >= window_start,
                                    Notification.created_at <= now,
                                    Notification.delivered_at.is_(None),
                                )
                            )
                            .order_by(Notification.created_at.desc())
                        )
                        notifications = notif_result.scalars().all()

                        if not notifications:
                            continue

                        digest_items = []
                        for n in notifications:
                            meta = n.metadata_json or {}
                            url = meta.get("url") or meta.get("action_url") or ""
                            digest_items.append(
                                {
                                    "title": n.title,
                                    "content": n.content,
                                    "category": n.category or "general",
                                    "priority": n.priority,
                                    "url": url,
                                }
                            )

                        student_email = pref_json.get("user_email")
                        student_name = pref_json.get("user_name", "")
                        # Resolve from LMS when email is missing OR stored name is a
                        # placeholder ("User <id>") set by an older backfill pass.
                        _name_is_placeholder = student_name.startswith("User ") and len(student_name) > 5
                        if not student_email or not student_name or _name_is_placeholder:
                            student_contact = await resolve_student_contact(prefs.user_id)
                            if student_contact.get("email"):
                                student_email = student_contact["email"]
                            if student_contact.get("first_name"):
                                student_name = student_contact["first_name"]
                        if not student_email:
                            logger.warning("daily_digest_skipped_missing_email", user_id=prefs.user_id)
                            continue

                        advisor_first_name, _ = await SchedulerService._resolve_advisor_for_user(prefs.user_id)

                        subject, html_body, text_body = build_digest_email(
                            student_name=student_name,
                            items=digest_items,
                            cadence=frequency,
                            advisor_persona=advisor_first_name,
                        )
                        sent = await send_email(
                            to_email=student_email,
                            subject=subject,
                            html_body=html_body,
                            text_body=text_body,
                        )
                        if not sent:
                            logger.warning("daily_digest_email_failed", user_id=prefs.user_id)
                            continue

                        for notification in notifications:
                            notification.delivered_at = now
                            notification.sent_at = now

                        pref_json["last_job_opportunity_digest_sent_at"] = now.isoformat()
                        prefs.preferences = pref_json
                        prefs.updated_at = now

                        logger.info(
                            "opportunity_digest_sent",
                            user_id=prefs.user_id,
                            frequency=frequency,
                            notification_count=len(digest_items),
                        )
                    except Exception as e:
                        logger.warning("daily_digest_user_failed", user_id=prefs.user_id, error=str(e))

                await session.commit()
        except Exception as e:
            logger.error("generate_daily_digest error", error=str(e), exc_info=True)

    # ------------------------------------------------------------------
    # Struggle detection [§2.4.2]
    # ------------------------------------------------------------------
    async def check_struggles(self) -> None:
        """Detect struggling students and send intervention notifications."""
        try:
            if not db_manager.engine:
                return
            session_maker = async_sessionmaker(db_manager.engine, expire_on_commit=False)
            async with session_maker() as session:
                result = await session.execute(select(UserActivityTracking.user_id))
                user_ids = result.scalars().all()

                for user_id in user_ids:
                    try:
                        struggle = await notification_engine.detect_struggle(session, user_id)
                        if not struggle:
                            continue

                        # Deduplicate — only one struggle notification per 48 hours
                        exists = await session.execute(
                            select(Notification).where(
                                and_(
                                    Notification.user_id == user_id,
                                    Notification.category == "motivation",
                                    Notification.created_at > (datetime.now(UTC) - timedelta(hours=48)),
                                )
                            )
                        )
                        if exists.scalars().first():
                            continue

                        advisor_name = await self._resolve_advisor_first_name(user_id)
                        await notification_engine.create_notification(
                            session=session,
                            user_id=user_id,
                            title=struggle["title"],
                            content=struggle["content"],
                            category=struggle["category"],
                            priority=struggle["priority"],
                            persona=advisor_name,
                            action_buttons=[
                                {
                                    "action": "chat",
                                    "title": "Talk to Advisor",
                                    "url": "/dashboard/ai-career-advisor",
                                },
                                {
                                    "action": "reschedule",
                                    "title": "Adjust My Plan",
                                    "url": "/dashboard/action-items",
                                },
                            ],
                            check_frequency=True,
                        )
                        logger.info("struggle_notification_sent", user_id=user_id)
                    except Exception as e:
                        logger.warning("struggle_check_failed", user_id=user_id, error=str(e))

                await session.commit()
        except Exception as e:
            logger.error("check_struggles error", error=str(e), exc_info=True)


# Global instance
scheduler_service = SchedulerService()

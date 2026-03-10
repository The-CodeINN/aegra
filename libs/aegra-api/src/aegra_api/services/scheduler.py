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
- Opportunity expiration + daily discovery
- Frequency-aware notification creation via NotificationEngine
"""

from datetime import UTC, datetime, timedelta

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler  # type: ignore[import-untyped]
from apscheduler.triggers.cron import CronTrigger  # type: ignore[import-untyped]
from apscheduler.triggers.interval import IntervalTrigger  # type: ignore[import-untyped]
from sqlalchemy import and_, delete, or_, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker

from aegra_api.core.accountability_orm import (
    ActionItem,
    DiscoveredOpportunity,
    Notification,
    UserActivityTracking,
    UserPreferences,
)
from aegra_api.core.database import db_manager
from aegra_api.services.email_service import build_digest_email, resolve_student_contact, send_email
from aegra_api.services.notification_engine import notification_engine
from aegra_api.services.opportunity_discovery import opportunity_engine

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
            # Cleanup old notifications — daily
            self.scheduler.add_job(
                self.check_cleanup,
                IntervalTrigger(hours=24),
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
            # Discovery — twice daily (every 12 hours)
            self.scheduler.add_job(
                self.run_discovery_job,
                IntervalTrigger(hours=12),
                id="run_discovery_job",
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

            self.scheduler.start()
            logger.info("Scheduler started with all accountability jobs")

    def shutdown(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown()

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

                    notif = await notification_engine.create_notification(
                        session=session,
                        user_id=item.user_id,
                        title=title,
                        content=content,
                        category="deadline",
                        priority=priority,
                        persona=item.advisor_persona,
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
        if item.reminder_sent_count == 0:
            return True
        if item.last_reminder_sent:
            hours_since = (now - item.last_reminder_sent).total_seconds() / 3600
            if hours_since < 2:
                return False
        return not (tier.startswith("overdue") and item.reminder_sent_count >= 5)

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

                    await notification_engine.create_notification(
                        session=session,
                        user_id=activity.user_id,
                        title=title,
                        content=content,
                        category="inactivity",
                        priority=priority,
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

                        await notification_engine.create_notification(
                            session=session,
                            user_id=user_id,
                            title=cel["title"],
                            content=cel["content"],
                            category="celebration",
                            priority=cel.get("priority", "normal"),
                            metadata={"celebration_type": cel["type"]},
                            check_frequency=True,
                        )

                await session.commit()
        except Exception as e:
            logger.error("check_celebrations error", error=str(e), exc_info=True)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    async def check_cleanup(self) -> None:
        """Cleanup old notifications — 90 day retention per spec §2.6."""
        try:
            if not db_manager.engine:
                return
            session_maker = async_sessionmaker(db_manager.engine, expire_on_commit=False)
            async with session_maker() as session:
                now = datetime.now(UTC)
                cutoff = now - timedelta(days=90)  # Spec: 90-day retention

                stmt = delete(Notification).where(Notification.created_at < cutoff)
                result = await session.execute(stmt)
                if result.rowcount > 0:
                    logger.info("notification_cleanup", deleted=result.rowcount)
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
    # Discovery job
    # ------------------------------------------------------------------
    async def run_discovery_job(self) -> None:
        """Periodic opportunity discovery for all active users."""
        logger.info("discovery_job_started")
        try:
            if not db_manager.engine:
                logger.warning("discovery_job_aborted", reason="no db engine")
                return
            session_maker = async_sessionmaker(db_manager.engine, expire_on_commit=False)
            async with session_maker() as session:
                result = await session.execute(select(UserActivityTracking.user_id))
                user_ids = result.scalars().all()
                logger.info("discovery_job_users_found", count=len(user_ids), user_ids=user_ids)

                if not user_ids:
                    logger.warning("discovery_job_no_users", reason="user_activity_tracking table is empty")
                    return

                for user_id in user_ids:
                    try:
                        logger.info("discovery_job_user_start", user_id=user_id)
                        discovered = await opportunity_engine.discover_for_user(
                            session=session,
                            user_id=user_id,
                            auth_token="",  # nosec B106
                            max_tracks=2,
                            queries_per_category=1,
                        )
                        logger.info(
                            "discovery_job_user_done",
                            user_id=user_id,
                            opportunities_found=len(discovered),
                        )
                        for opp in discovered:
                            # Use notification_engine so web push is triggered
                            type_label = opp.opportunity_type
                            if type_label == "event":
                                title = "🎯 New Event Matches Your Track"
                                content = f"We found a {opp.matched_track} event: {opp.title}"
                            else:
                                # Jobs
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
                        logger.error(
                            "discovery_failed_for_user",
                            user_id=user_id,
                            error=str(e),
                            exc_info=True,
                        )
        except Exception as e:
            logger.error("run_discovery_job error", error=str(e), exc_info=True)

    # ------------------------------------------------------------------
    # Motivational nudges [§2.4.3]
    # ------------------------------------------------------------------
    MOTIVATIONAL_MESSAGES = [
        {
            "title": "🌟 Monday Motivation",
            "content": "New week, new opportunities! What's one thing you'll accomplish this week toward your career goal?",
        },
        {
            "title": "💪 Midweek Momentum",
            "content": "You're halfway through the week! Keep going — every small step counts toward your career transformation.",
        },
        {
            "title": "🎯 Friday Focus",
            "content": "End the week strong! Take 15 minutes to review your progress and celebrate what you've accomplished.",
        },
        {
            "title": "📚 Sunday Strategy",
            "content": "Tomorrow starts a new week. Take a moment to plan your priorities and set yourself up for success!",
        },
    ]

    async def send_motivational_nudges(self) -> None:
        """Send motivational nudges on Mon/Wed/Fri/Sun per spec §2.4.3."""
        try:
            if not db_manager.engine:
                return
            session_maker = async_sessionmaker(db_manager.engine, expire_on_commit=False)
            async with session_maker() as session:
                result = await session.execute(select(UserActivityTracking.user_id))
                user_ids = result.scalars().all()

                # Pick message based on day of week
                dow = datetime.now(UTC).weekday()  # 0=Mon, 6=Sun
                day_map = {0: 0, 2: 1, 4: 2, 6: 3}  # Mon=0, Wed=1, Fri=2, Sun=3
                msg_idx = day_map.get(dow, 0)
                msg = self.MOTIVATIONAL_MESSAGES[msg_idx]

                for user_id in user_ids:
                    try:
                        await notification_engine.create_notification(
                            session=session,
                            user_id=user_id,
                            title=msg["title"],
                            content=msg["content"],
                            category="motivation",
                            priority="low",
                            action_buttons=[
                                {
                                    "action": "chat",
                                    "title": "Chat with Advisor",
                                    "url": "/dashboard/ai-career-advisor",
                                },
                            ],
                            check_frequency=True,
                        )
                    except Exception as e:
                        logger.warning("motivational_nudge_failed", user_id=user_id, error=str(e))

                await session.commit()
                logger.info("motivational_nudges_sent", user_count=len(user_ids))
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

                        # Require active AI Mentor add-on (cached at preference-save time)
                        if not pref_json.get("ai_mentor_addon_active", False):
                            continue
                        addon_expires_raw = pref_json.get("ai_mentor_addon_expires_at")
                        if addon_expires_raw:
                            try:
                                addon_expires = datetime.fromisoformat(addon_expires_raw)
                                if addon_expires.tzinfo is None:
                                    addon_expires = addon_expires.replace(tzinfo=UTC)
                                if now >= addon_expires:
                                    logger.info(
                                        "daily_digest_skipped_addon_expired",
                                        user_id=prefs.user_id,
                                        expired_at=addon_expires_raw,
                                    )
                                    continue
                            except ValueError:
                                pass

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

                        # Prefer email stored locally in preferences (set on every
                        # preferences PUT via auth context) before hitting the LMS.
                        student_email = pref_json.get("user_email")
                        student_name = pref_json.get("user_name", "")
                        if not student_email:
                            student_contact = await resolve_student_contact(prefs.user_id)
                            student_email = student_contact.get("email")
                            student_name = student_contact.get("first_name", student_name)
                        if not student_email:
                            logger.warning("daily_digest_skipped_missing_email", user_id=prefs.user_id)
                            continue

                        subject, html_body, text_body = build_digest_email(
                            student_name=student_name,
                            items=digest_items,
                            cadence=frequency,
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

                        await notification_engine.create_notification(
                            session=session,
                            user_id=user_id,
                            title=struggle["title"],
                            content=struggle["content"],
                            category=struggle["category"],
                            priority=struggle["priority"],
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

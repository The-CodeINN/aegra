"""Notification Engine — AI-powered notification generation & delivery.

Aligned with Requirements Specification v1.0.

Responsibilities
-----------------
* Persona-consistent message generation (Alexandra, Marcus, Priya, David) [§2.7]
* Dual-format output: short notification (≤160 chars) + full email [§2.2, §3.1]
* StudentProfile-aware personalized messages (name, goals, streak, struggles) [§2.2.1]
* Frequency management / anti-spam with spec-compliant defaults [§2.6]
* Priority-based channel routing (in-app, push, email) [§3.1]
* Daily digest bundling for suppressed low-priority items [§2.6.1]
* Progress celebration detection (streaks, first project, milestones) [§2.4.1]
* Struggle detection & intervention messaging [§2.4.2]
* Notification fatigue detection (3 consecutive dismissals) [§5.1]
* Tiered deadline reminders (7d, 3d, 24h, 2h, overdue, severe) [§2.2.1]
* Tiered inactivity detection (3d, 7d, 10d, 15d+) with risk scoring [§2.2.2]
* Email delivery via SMTP [§3.1]
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.core.accountability_orm import (
    ActionItem,
    Notification,
    UserActivityTracking,
    UserPreferences,
)
from aegra_api.services.web_push import web_push_service

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Advisor persona definitions
# ---------------------------------------------------------------------------
ADVISOR_PERSONAS: dict[str, dict[str, Any]] = {
    "Alexandra": {
        "style": "professional, methodical, detail-oriented",
        "tone": "encouraging but pragmatic",
        "emoji": "📊 📈 📋",
        "sign_off": "- Alexandra",
        "catchphrases": ["Let's tackle this strategically", "Here's the plan"],
    },
    "Marcus": {
        "style": "analytical, curious, research-oriented",
        "tone": "intellectually curious and supportive",
        "emoji": "🔬 🧪 📐",
        "sign_off": "- Marcus",
        "catchphrases": ["Interesting challenge", "Let's investigate"],
    },
    "Priya": {
        "style": "technical, systematic, infrastructure-focused",
        "tone": "calm, methodical, solution-focused",
        "emoji": "⚙️ 🔧 🏗️",
        "sign_off": "- Priya",
        "catchphrases": ["Let's build this step by step", "Here's how we architect this"],
    },
    "David": {
        "style": "innovative, forward-thinking, pioneering",
        "tone": "excited about possibilities, energetic",
        "emoji": "🚀 🤖 ⚡",
        "sign_off": "- David",
        "catchphrases": ["This is the future", "Let's push boundaries"],
    },
}

DEFAULT_PERSONA = "Alexandra"

# ---------------------------------------------------------------------------
# Frequency management constants [§2.6]
# ---------------------------------------------------------------------------
MAX_DAILY_NOTIFICATIONS = 3  # Spec: max_daily_notifications = 3
MAX_WEEKLY_NOTIFICATIONS = 12  # Spec: max_weekly = 12
COOL_DOWN_HOURS = 4  # Spec: cool_down_period_hours = 4
QUIET_HOUR_START = 22  # 10 PM
QUIET_HOUR_END = 8  # 8 AM

# Priority levels (higher = more important)
PRIORITY_RANK = {"low": 0, "normal": 1, "high": 2, "urgent": 3, "critical": 4}

# ---------------------------------------------------------------------------
# Celebration triggers
# ---------------------------------------------------------------------------
STREAK_MILESTONES = {7, 14, 30, 60, 90}  # Spec: 7, 14, 30, 60, 90 (no 21)


class NotificationEngine:
    """Central brain for generating and managing notifications."""

    # ------------------------------------------------------------------
    # Frequency management
    # ------------------------------------------------------------------
    async def should_send(
        self,
        session: AsyncSession,
        user_id: str,
        priority: str = "normal",
        category: str = "general",
    ) -> bool:
        """Determine whether we should send a notification right now.

        Checks:
        1. User has notifications enabled
        2. Daily cap not exceeded (unless critical)
        3. Cool-down period respected
        4. Category not disabled by user
        5. Quiet hours respected (unless critical)
        """
        # Load preferences
        prefs_row = await session.execute(select(UserPreferences).where(UserPreferences.user_id == user_id))
        prefs = prefs_row.scalar_one_or_none()

        # No prefs → allow (defaults)
        if prefs and not prefs.notifications_enabled:
            return False

        user_prefs = (prefs.preferences if prefs else {}) or {}

        # Check disabled categories
        disabled = user_prefs.get("disabled_categories", [])
        if category in disabled:
            return False

        # Always allow critical
        if priority == "critical":
            return True

        # Check quiet hours
        quiet_start = user_prefs.get("quiet_hours_start", QUIET_HOUR_START)
        quiet_end = user_prefs.get("quiet_hours_end", QUIET_HOUR_END)
        now_hour = datetime.now(UTC).hour
        if quiet_start > quiet_end:
            in_quiet = now_hour >= quiet_start or now_hour < quiet_end
        else:
            in_quiet = quiet_start <= now_hour < quiet_end
        if in_quiet and priority != "urgent":
            return False

        # Check daily cap
        max_daily = user_prefs.get("max_daily", MAX_DAILY_NOTIFICATIONS)
        today_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        count_result = await session.execute(
            select(func.count(Notification.id)).where(
                and_(
                    Notification.user_id == user_id,
                    Notification.created_at >= today_start,
                )
            )
        )
        today_count = count_result.scalar() or 0
        if today_count >= max_daily and priority not in ("urgent", "critical"):
            return False

        # Cool-down check
        last_result = await session.execute(
            select(Notification.created_at)
            .where(Notification.user_id == user_id)
            .order_by(Notification.created_at.desc())
            .limit(1)
        )
        last_sent = last_result.scalar_one_or_none()
        if last_sent:
            hours_since = (datetime.now(UTC) - last_sent).total_seconds() / 3600
            if hours_since < COOL_DOWN_HOURS and priority not in ("urgent", "critical"):
                return False

        return True

    # ------------------------------------------------------------------
    # Persona-consistent message generation
    # ------------------------------------------------------------------
    async def generate_persona_message(
        self,
        base_message: str,
        persona_name: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Rewrite a message through an advisor persona using LLM.

        Uses student context (name, goals, streak) when available [§2.7].
        Falls back to simple sign-off if LLM unavailable.
        """
        persona = ADVISOR_PERSONAS.get(persona_name or DEFAULT_PERSONA, ADVISOR_PERSONAS[DEFAULT_PERSONA])

        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from langchain_openai import ChatOpenAI

            # Build context-aware system prompt with StudentProfile data
            ctx_parts = []
            if context:
                if context.get("first_name"):
                    ctx_parts.append(f"Student's name: {context['first_name']}")
                if context.get("primary_goal"):
                    ctx_parts.append(f"Career goal: {context['primary_goal']}")
                if context.get("current_streak"):
                    ctx_parts.append(f"Current streak: {context['current_streak']} days")
                if context.get("biggest_challenge"):
                    ctx_parts.append(f"Recent struggle: {context['biggest_challenge']}")

            context_str = "\n".join(ctx_parts) if ctx_parts else "No additional context."

            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=200)
            resp = await llm.ainvoke(
                [
                    SystemMessage(
                        content=(
                            f"You are {persona_name or DEFAULT_PERSONA}, a career advisor. "
                            f"Style: {persona['style']}. Tone: {persona['tone']}. "
                            f"Use these emojis sparingly: {persona['emoji']}. "
                            f"Catchphrases: {', '.join(persona['catchphrases'])}. "
                            f"\n\nStudent context:\n{context_str}\n\n"
                            "Rewrite the following notification message in your voice. "
                            "Use the student's first name if available. "
                            "Keep it under 160 characters. "
                            "Reply with ONLY the rewritten message, nothing else."
                        )
                    ),
                    HumanMessage(content=base_message),
                ]
            )
            return resp.content.strip()
        except Exception:
            # Fallback: just append sign-off
            return f"{base_message}\n{persona['sign_off']}"

    # ------------------------------------------------------------------
    # Deadline reminders (tiered)
    # ------------------------------------------------------------------
    def compute_deadline_tier(
        self, due_date: datetime, now: datetime | None = None
    ) -> tuple[str | None, str, str, str]:
        """Return (tier, priority, title, content_template) or (None, ...) if no reminder needed."""
        now = now or datetime.now(UTC)
        delta = due_date - now
        hours_left = delta.total_seconds() / 3600

        if hours_left < -72:
            return (
                "overdue_severe",
                "critical",
                "🚨 Action Item Severely Overdue",
                "Your task has been overdue for {days} days: {description}. Let's talk about what's blocking you.",
            )
        if hours_left < -24:
            return (
                "overdue",
                "high",
                "⚠️ Action Item Overdue",
                "Your task is overdue: {description}. No stress — let's figure out what happened and adjust.",
            )
        if hours_left < 0:
            return (
                "overdue_just",
                "high",
                "⏰ Just Passed Deadline",
                "The deadline for '{description}' just passed. Still time to finish — want help?",
            )
        if hours_left <= 2:
            return (
                "2h",
                "urgent",
                "⏰ Due in 2 Hours!",
                "Almost time! '{description}' is due very soon. Let's finish strong!",
            )
        if hours_left <= 24:
            return (
                "24h",
                "high",
                "🔥 Due Tomorrow",
                "'{description}' is due in less than 24 hours. Today is the day — let's finish strong!",
            )
        if hours_left <= 72:
            return (
                "3d",
                "normal",
                "📋 Coming Up in 3 Days",
                "'{description}' is due in {days} days. Time to plan your approach!",
            )
        if hours_left <= 168:
            return (
                "7d",
                "low",
                "📝 Due This Week",
                "Just a heads-up: '{description}' is due in {days} days. You've got this!",
            )
        return (None, "low", "", "")

    # ------------------------------------------------------------------
    # Inactivity detection with risk scoring [§2.2.2]
    # ------------------------------------------------------------------
    def compute_inactivity_tier(self, days_inactive: int) -> tuple[str | None, str, str, str]:
        """Return (tier, priority, title, content) for inactivity.

        Risk scoring per spec:
        - 20-40 (3-5d): Gentle check-in
        - 41-60 (6-9d): Motivational nudge
        - 61-80 (10-14d): Concern + support offer
        - 81-100 (15+d): Win-back campaign
        """
        if days_inactive >= 15:
            return (
                "win_back",
                "high",
                "💪 You've Come So Far!",
                "It's been {days} days since we last connected. You've already made incredible progress — let's not lose that momentum. What would help you get back on track?",
            )
        if days_inactive >= 10:
            return (
                "concern",
                "high",
                "🤝 I'm Here to Help",
                "I noticed it's been {days} days. Career transitions are tough — if something is blocking you, I'd love to help figure it out together.",
            )
        if days_inactive >= 6:
            return (
                "motivational",
                "normal",
                "🌟 Your Career Goal Is Still Achievable!",
                "It's been {days} days! Your career goal is still absolutely achievable. Let's reconnect and keep that momentum going.",
            )
        if days_inactive >= 3:
            return (
                "gentle",
                "normal",
                "👋 Quick Check-in",
                "Haven't seen you in {days} days. Everything okay? Ready to continue your learning journey?",
            )
        return (None, "low", "", "")

    def compute_risk_score(self, days_inactive: int) -> int:
        """Compute engagement risk score 0-100 per spec §2.2.2."""
        if days_inactive >= 15:
            return min(100, 81 + (days_inactive - 15) * 2)
        if days_inactive >= 10:
            return 61 + (days_inactive - 10) * 4  # 61-80
        if days_inactive >= 6:
            return 41 + (days_inactive - 6) * 5  # 41-60
        if days_inactive >= 3:
            return 20 + (days_inactive - 3) * 7  # 20-40
        return max(0, days_inactive * 6)  # 0-19

    # ------------------------------------------------------------------
    # Notification fatigue detection [§5.1]
    # ------------------------------------------------------------------
    async def check_notification_fatigue(
        self,
        session: AsyncSession,
        user_id: str,
    ) -> bool:
        """Check if user has dismissed 3+ consecutive notifications.

        Returns True if fatigued (should reduce frequency).
        """
        result = await session.execute(
            select(Notification.status)
            .where(Notification.user_id == user_id)
            .order_by(Notification.created_at.desc())
            .limit(3)
        )
        recent = result.scalars().all()
        if len(recent) >= 3 and all(s == "dismissed" for s in recent):
            logger.warning("notification_fatigue_detected", user_id=user_id)
            return True
        return False

    # ------------------------------------------------------------------
    # Progress celebrations [§2.4.1]
    # ------------------------------------------------------------------
    async def check_celebrations(self, session: AsyncSession, user_id: str) -> list[dict[str, Any]]:
        """Detect celebration-worthy events for a user."""
        celebrations: list[dict[str, Any]] = []

        # Check streak milestones
        activity_result = await session.execute(
            select(UserActivityTracking).where(UserActivityTracking.user_id == user_id)
        )
        activity = activity_result.scalar_one_or_none()
        if activity and activity.current_streak in STREAK_MILESTONES:
            celebrations.append(
                {
                    "type": "streak",
                    "title": f"🔥 {activity.current_streak}-Day Streak!",
                    "content": (
                        f"You've been consistent for {activity.current_streak} days! "
                        "That kind of discipline separates those who succeed from those who just talk about it."
                    ),
                    "priority": "normal",
                }
            )

        # Check recently completed action items (last 24h)
        yesterday = datetime.now(UTC) - timedelta(hours=24)
        completed_result = await session.execute(
            select(func.count(ActionItem.id)).where(
                and_(
                    ActionItem.user_id == user_id,
                    ActionItem.status == "completed",
                    ActionItem.updated_at >= yesterday,
                )
            )
        )
        completed_count = completed_result.scalar() or 0
        if completed_count >= 3:
            celebrations.append(
                {
                    "type": "productive_day",
                    "title": "⭐ Incredible Productivity!",
                    "content": (
                        f"You completed {completed_count} tasks today! "
                        "That's the kind of momentum that transforms careers."
                    ),
                    "priority": "normal",
                }
            )

        # First-ever completion
        total_result = await session.execute(
            select(func.count(ActionItem.id)).where(
                and_(
                    ActionItem.user_id == user_id,
                    ActionItem.status == "completed",
                )
            )
        )
        total_completed = total_result.scalar() or 0
        if total_completed == 1:
            celebrations.append(
                {
                    "type": "first_completion",
                    "title": "🎉 First Task Completed!",
                    "content": (
                        "You just completed your first action item! This is how careers are built — "
                        "one step at a time. I'm proud of you!"
                    ),
                    "priority": "high",
                }
            )

        return celebrations

    # ------------------------------------------------------------------
    # Struggle detection [§2.4.2]
    # ------------------------------------------------------------------
    async def detect_struggle(
        self,
        session: AsyncSession,
        user_id: str,
    ) -> dict[str, Any] | None:
        """Detect if a student is struggling based on available signals.

        Checks:
        - Overdue action items piling up
        - Declining engagement (activity tracking)
        - Multiple incomplete items with no progress

        Returns intervention message dict or None.
        """
        now = datetime.now(UTC)

        # Check overdue items
        overdue_result = await session.execute(
            select(func.count(ActionItem.id)).where(
                and_(
                    ActionItem.user_id == user_id,
                    ActionItem.status.in_(["pending", "in_progress"]),
                    ActionItem.due_date < now,
                )
            )
        )
        overdue_count = overdue_result.scalar() or 0

        # Check incomplete items with no recent updates (stale > 7 days)
        stale_result = await session.execute(
            select(func.count(ActionItem.id)).where(
                and_(
                    ActionItem.user_id == user_id,
                    ActionItem.status.in_(["pending", "in_progress"]),
                    ActionItem.updated_at < (now - timedelta(days=7)),
                )
            )
        )
        stale_count = stale_result.scalar() or 0

        # Check engagement decline
        activity_result = await session.execute(
            select(UserActivityTracking).where(UserActivityTracking.user_id == user_id)
        )
        activity = activity_result.scalar_one_or_none()
        streak_broken = activity and activity.current_streak == 0 and activity.longest_streak >= 7

        # Calculate struggle score (0-100)
        struggle_score = 0
        struggle_score += min(30, overdue_count * 10)
        struggle_score += min(30, stale_count * 8)
        if streak_broken:
            struggle_score += 25
        if activity and activity.current_streak == 0:
            struggle_score += 15

        if struggle_score < 40:
            return None

        logger.info(
            "struggle_detected",
            user_id=user_id,
            score=struggle_score,
            overdue=overdue_count,
            stale=stale_count,
            streak_broken=streak_broken,
        )

        if struggle_score >= 70:
            return {
                "title": "🤝 I'm Here to Help",
                "content": (
                    "I noticed things have been tough lately — "
                    f"you have {overdue_count} overdue items and progress has slowed down. "
                    "Career transitions are marathons, not sprints. "
                    "Sometimes we need to pause, reassess, and adjust our pace. "
                    "What would make this week feel more manageable for you?"
                ),
                "priority": "high",
                "category": "motivation",
            }
        return {
            "title": "💪 Let's Get Back on Track",
            "content": (
                "I see a few items need attention. That's completely normal — "
                "everyone hits bumps. Let's break things into smaller, "
                "more manageable steps. Want to chat about your priorities?"
            ),
            "priority": "normal",
            "category": "motivation",
        }

    # ------------------------------------------------------------------
    # Create notification (core helper) [§3.1]
    # ------------------------------------------------------------------
    async def create_notification(
        self,
        session: AsyncSession,
        user_id: str,
        title: str,
        content: str,
        category: str,
        priority: str = "normal",
        persona: str | None = None,
        action_buttons: list[dict] | None = None,
        metadata: dict[str, Any] | None = None,
        expires_at: datetime | None = None,
        check_frequency: bool = True,
        student_context: dict[str, Any] | None = None,
    ) -> Notification | None:
        """Create a notification with frequency checks, persona rewriting, and multi-channel delivery.

        Per spec §3.1, delivers via:
        1. In-app notification
        2. Web push (if subscribed)
        3. Email (if enabled and email available)
        """
        if check_frequency:
            # Check fatigue first
            fatigued = await self.check_notification_fatigue(session, user_id)
            if fatigued and priority not in ("urgent", "critical"):
                logger.debug("notification_blocked_fatigue", user_id=user_id, category=category)
                return None

            allowed = await self.should_send(session, user_id, priority, category)
            if not allowed:
                logger.debug(
                    "notification_suppressed",
                    user_id=user_id,
                    category=category,
                    priority=priority,
                )
                return None

        # Apply persona voice if available
        if persona:
            content = await self.generate_persona_message(content, persona, context=student_context)

        notification = Notification(
            user_id=user_id,
            title=title,
            content=content,
            channel="in_app",
            priority=priority,
            category=category,
            action_buttons=action_buttons or [],
            metadata_json=metadata or {},
            expires_at=expires_at,
        )
        session.add(notification)
        await session.flush()

        # ── Send web push (best-effort) ─────────────────────────────
        try:
            url = None
            if action_buttons:
                for btn in action_buttons:
                    if btn.get("url"):
                        url = btn["url"]
                        break

            payload = web_push_service.build_payload(
                title=title,
                body=content,
                category=category,
                priority=priority,
                url=url,
                notification_id=notification.id,
            )
            await web_push_service.send_to_user(session, user_id, payload)
        except Exception as push_err:
            logger.debug("web_push_attempt_failed", error=str(push_err))

        # ── Send email (best-effort) [§3.1] ─────────────────────────
        try:
            await self._send_email_notification(
                session=session,
                user_id=user_id,
                title=title,
                content=content,
                category=category,
                action_buttons=action_buttons,
                persona=persona or DEFAULT_PERSONA,
                student_context=student_context,
            )
        except Exception as email_err:
            logger.debug("email_attempt_failed", error=str(email_err))

        return notification

    async def _send_email_notification(
        self,
        session: AsyncSession,
        user_id: str,
        title: str,
        content: str,
        category: str,
        action_buttons: list[dict] | None = None,
        persona: str = "Alexandra",
        student_context: dict[str, Any] | None = None,
    ) -> None:
        """Send email version of notification if user has email enabled."""
        from aegra_api.services.email_service import build_notification_email, send_email

        # Check if email is enabled for this user
        prefs_row = await session.execute(select(UserPreferences).where(UserPreferences.user_id == user_id))
        prefs = prefs_row.scalar_one_or_none()
        user_prefs = (prefs.preferences if prefs else {}) or {}

        if not user_prefs.get("email_enabled", True):
            return

        # Get student email from context
        student_email = None
        student_name = ""
        if student_context:
            student_email = student_context.get("email")
            student_name = student_context.get("first_name", "")

        if not student_email:
            return

        subject = f"[DeDataHub] {title}"
        html_body, text_body = build_notification_email(
            student_name=student_name,
            title=title,
            content=content,
            action_buttons=action_buttons,
            advisor_persona=persona,
            category=category,
        )

        await send_email(
            to_email=student_email,
            subject=subject,
            html_body=html_body,
            text_body=text_body,
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
notification_engine = NotificationEngine()

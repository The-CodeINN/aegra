"""Weekly advisor check-in service.

Creates and manages a thread-bound cron per user so their AI advisor
automatically initiates a weekly progress review. The persistent thread
means every Monday firing continues the same conversation — the advisor
has full history, goals, and course context in scope.

Architecture:
  provision_user()         — called once per user; creates thread + cron
  deprovision_user()       — disables the cron when a user churns/opts out
  provision_all_pending()  — batch provision for new users; safe to re-run

Cron IDs and thread IDs are stored in UserPreferences.preferences JSONB
under the keys defined by _PREF_THREAD_KEY and _PREF_CRON_KEY.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import structlog
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from aegra_api.core.accountability_orm import UserActivityTracking, UserPreferences
from aegra_api.core.orm import Cron as CronORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.core.orm import get_session_maker
from aegra_api.data.career_advisors import get_advisor_by_track, get_default_advisor
from aegra_api.models.crons import CronCreate
from aegra_api.services.career_advisor_activation import ROADMAP_VARIANT_PREFERENCE_KEY, pick_roadmap_variant
from aegra_api.services.cron_service import CronService
from aegra_api.services.langgraph_service import LangGraphService
from aegra_api.services.scheduler import SchedulerService

logger = structlog.getLogger(__name__)

_CHECKIN_SCHEDULE = "0 9 * * 1"  # Monday 09:00 UTC
_CHECKIN_GRAPH_ID = "agent"
_PREF_THREAD_KEY = "weekly_checkin_thread_id"
_PREF_CRON_KEY = "weekly_checkin_cron_id"
_PROVISION_CONCURRENCY = 5

_CHECKIN_INPUT: dict[str, Any] = {
    "messages": [
        {
            "role": "user",
            "content": ("Weekly advisor check-in — let's review my progress and set goals for this week."),
        }
    ]
}


class WeeklyCheckinService:
    """Manages per-user thread-bound weekly advisor check-in crons."""

    async def provision_user(
        self,
        user_id: str,
        *,
        session: AsyncSession,
        langgraph_service: LangGraphService,
    ) -> str | None:
        """Create a persistent thread + weekly cron for *user_id*.

        Returns the new ``cron_id`` string on success, ``None`` when the user
        is already provisioned. Safe to call multiple times — idempotent.
        """
        pref = await session.get(UserPreferences, user_id)
        if pref and (pref.preferences or {}).get(_PREF_CRON_KEY):
            return None

        _, track = await SchedulerService._resolve_advisor_for_user(user_id)
        advisor_info: dict[str, Any] = get_advisor_by_track(track) or get_default_advisor()
        # Weekly check-ins are by definition a returning-student surface —
        # assign (or reuse) the same sticky A/B variant as live chat (spec Item 6).
        # Uses the already-fetched `pref` (no second round-trip); persisted
        # below alongside the thread/cron IDs in the single combined commit.
        roadmap_variant = pick_roadmap_variant(pref.preferences if pref else None)

        # Flush the thread into the current transaction so it is visible to
        # create_cron's FK validation without committing yet. create_cron
        # will commit everything (thread + cron) together.
        thread_id = str(uuid4())
        session.add(
            ThreadORM(
                thread_id=thread_id,
                user_id=user_id,
                status="idle",
                metadata_json={},
            )
        )
        await session.flush()

        cron_svc = CronService(session, langgraph_service)
        try:
            cron_orm = await cron_svc.create_cron(
                CronCreate(
                    assistant_id=_CHECKIN_GRAPH_ID,
                    schedule=_CHECKIN_SCHEDULE,
                    input=_CHECKIN_INPUT,
                    context={
                        "learning_track": track or None,
                        "advisor": advisor_info,
                        "roadmap_generated": True,
                        "roadmap_variant": roadmap_variant,
                    },
                    enabled=True,
                ),
                user_identity=user_id,
                thread_id=thread_id,
            )
        except Exception:
            await session.rollback()
            raise

        # create_cron commits, so the thread and cron are now persisted.
        # Write the IDs back into preferences in a second commit.
        if pref is None:
            pref = UserPreferences(user_id=user_id, preferences={})
            session.add(pref)

        pref.preferences = {
            **(pref.preferences or {}),
            _PREF_THREAD_KEY: thread_id,
            _PREF_CRON_KEY: str(cron_orm.cron_id),
            ROADMAP_VARIANT_PREFERENCE_KEY: roadmap_variant,
        }
        flag_modified(pref, "preferences")
        await session.commit()

        logger.info(
            "weekly_checkin_provisioned",
            user_id=user_id,
            thread_id=thread_id,
            cron_id=str(cron_orm.cron_id),
        )
        return str(cron_orm.cron_id)

    async def deprovision_user(self, user_id: str, *, session: AsyncSession) -> None:
        """Disable the weekly check-in cron for *user_id*.

        Does nothing when no cron is recorded for the user.
        """
        pref = await session.get(UserPreferences, user_id)
        cron_id = (pref.preferences or {}).get(_PREF_CRON_KEY) if pref else None
        if not cron_id:
            return

        await session.execute(
            update(CronORM)
            .where(CronORM.cron_id == cron_id, CronORM.user_id == user_id)
            .values(enabled=False, updated_at=datetime.now(UTC))
        )
        await session.commit()
        logger.info("weekly_checkin_deprovisioned", user_id=user_id, cron_id=cron_id)

    async def provision_all_pending(
        self,
        *,
        session: AsyncSession,
        langgraph_service: LangGraphService,
    ) -> None:
        """Provision weekly check-ins for all users who don't have one yet.

        Uses *session* only for the initial discovery query. Each individual
        provision runs in its own session so a single failure does not roll
        back others.
        """
        already_provisioned_sq = select(UserPreferences.user_id).where(
            UserPreferences.preferences.has_key(_PREF_CRON_KEY)
        )
        result = await session.execute(
            select(UserActivityTracking.user_id).where(UserActivityTracking.user_id.not_in(already_provisioned_sq))
        )
        user_ids = list(result.scalars().all())

        if not user_ids:
            logger.info("weekly_checkin_provision_all_pending: no new users")
            return

        logger.info("weekly_checkin_provision_all_start", count=len(user_ids))
        sem = asyncio.Semaphore(_PROVISION_CONCURRENCY)

        async def _one(uid: str) -> None:
            async with sem:
                try:
                    maker = get_session_maker()
                    async with maker() as s:
                        await self.provision_user(uid, session=s, langgraph_service=langgraph_service)
                except Exception as exc:
                    logger.warning("weekly_checkin_provision_failed", user_id=uid, error=str(exc))

        await asyncio.gather(*[_one(uid) for uid in user_ids])
        logger.info("weekly_checkin_provision_all_done", provisioned=len(user_ids))


weekly_checkin_service = WeeklyCheckinService()

"""Unit tests for proactive outreach reading from task/persona memory (agent optimisation spec Item 7).

Covers the spec's own "done when" golden test: a student with two overdue
tasks gets an email naming both tasks and their due dates, names the goal,
uses one consistent advisor in body and sign-off, and at miss_count >= 2
pivots to blocker framing instead of re-listing the task. A student with
zero open tasks gets an honest check-in with no invented task.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aegra_api.services.notification_engine import DEFAULT_PERSONA, NotificationEngine


def _student_context(**overrides: object) -> dict:
    base = {
        "first_name": "Kudus",
        "learning_track": "ai-engineering",
        "current_streak": 0,
        "tasks_completed_this_week": 0,
        "overdue_tasks": 0,
        "pending_tasks": 0,
        "overdue_task_details": [],
        "pending_task_details": [],
        "primary_goal": "",
        "enrolled_course": "",
        "course_progress_pct": 0,
        "total_completed_lessons": 0,
        "total_watched_hours": 0.0,
        "last_active_in_course": "",
    }
    base.update(overrides)
    return base


class _FakeLLM:
    def __init__(self) -> None:
        self.captured_messages: list = []

    async def ainvoke(self, messages: list) -> MagicMock:
        self.captured_messages.extend(messages)
        resp = MagicMock()
        resp.content = "A generated body that would normally come from the LLM."
        return resp


def _patched_bedrock(fake_llm: _FakeLLM) -> tuple:
    fake_module = MagicMock()
    fake_module.ChatBedrockConverse.return_value = fake_llm
    return (
        patch.dict("sys.modules", {"langchain_aws": fake_module}),
        patch("aegra_api.services.notification_engine.settings"),
    )


class TestNamedTasksInPrompt:
    @pytest.mark.asyncio
    async def test_two_overdue_tasks_named_with_due_dates_in_llm_prompt(self) -> None:
        engine = NotificationEngine()
        student_ctx = _student_context(
            primary_goal="Data Product Manager",
            overdue_task_details=[
                {
                    "description": "Rebuild GitHub README with 3 pinned projects",
                    "due_date": "2026-06-27",
                    "miss_count": 0,
                },
                {"description": "Apply to 2 analyst roles", "due_date": "2026-06-27", "miss_count": 0},
            ],
        )
        fake_llm = _FakeLLM()
        p1, p2 = _patched_bedrock(fake_llm)
        with p1, p2 as mock_settings:
            mock_settings.aws.AWS_REGION_NAME = "eu-west-1"
            await engine.generate_personalized_motivational_content(persona_name="David", student_context=student_ctx)

        all_text = " ".join(str(m.content) for m in fake_llm.captured_messages)
        assert "Rebuild GitHub README with 3 pinned projects" in all_text
        assert "Apply to 2 analyst roles" in all_text
        assert "2026-06-27" in all_text
        assert "Data Product Manager" in all_text
        # The prompt must forbid the count-only phrasing that caused the original bug.
        assert "do not just say a count" in all_text.lower() or "name them specifically" in all_text.lower()

    @pytest.mark.asyncio
    async def test_miss_count_two_triggers_blocker_pivot_instruction(self) -> None:
        engine = NotificationEngine()
        student_ctx = _student_context(
            overdue_task_details=[
                {"description": "Rebuild GitHub README", "due_date": "2026-06-20", "miss_count": 2},
            ],
        )
        fake_llm = _FakeLLM()
        p1, p2 = _patched_bedrock(fake_llm)
        with p1, p2 as mock_settings:
            mock_settings.aws.AWS_REGION_NAME = "eu-west-1"
            await engine.generate_personalized_motivational_content(persona_name="David", student_context=student_ctx)

        all_text = " ".join(str(m.content) for m in fake_llm.captured_messages)
        assert "do not" in all_text.lower() and "re-list" in all_text.lower()
        assert "blocking" in all_text.lower() or "blocker" in all_text.lower() or "actually" in all_text.lower()

    @pytest.mark.asyncio
    async def test_fallback_path_names_tasks_not_just_counts(self) -> None:
        """When the LLM call fails, the fallback text must still name real tasks."""
        engine = NotificationEngine()
        student_ctx = _student_context(
            overdue_task_details=[
                {"description": "Rebuild GitHub README", "due_date": "2026-06-20", "miss_count": 0},
            ],
        )
        with patch.dict("sys.modules", {"langchain_aws": None}):
            _title, body = await engine.generate_personalized_motivational_content(
                persona_name="David", student_context=student_ctx
            )

        assert "Rebuild GitHub README" in body

    @pytest.mark.asyncio
    async def test_fallback_path_pivots_to_blocker_at_miss_count_two(self) -> None:
        engine = NotificationEngine()
        student_ctx = _student_context(
            overdue_task_details=[
                {"description": "Rebuild GitHub README", "due_date": "2026-06-20", "miss_count": 2},
            ],
        )
        with patch.dict("sys.modules", {"langchain_aws": None}):
            _title, body = await engine.generate_personalized_motivational_content(
                persona_name="David", student_context=student_ctx
            )

        assert "Rebuild GitHub README" in body
        assert "in the way" in body.lower() or "what's actually" in body.lower()


class TestEmptyStateHonesty:
    @pytest.mark.asyncio
    async def test_zero_open_tasks_prompt_instructs_no_invention(self) -> None:
        engine = NotificationEngine()
        student_ctx = _student_context()  # no tasks, no counts
        fake_llm = _FakeLLM()
        p1, p2 = _patched_bedrock(fake_llm)
        with p1, p2 as mock_settings:
            mock_settings.aws.AWS_REGION_NAME = "eu-west-1"
            await engine.generate_personalized_motivational_content(
                persona_name="Alexandra", student_context=student_ctx
            )

        all_text = " ".join(str(m.content) for m in fake_llm.captured_messages).lower()
        assert "do not invent" in all_text or "not invent or imply" in all_text

    @pytest.mark.asyncio
    async def test_zero_open_tasks_fallback_is_honest_not_invented(self) -> None:
        engine = NotificationEngine()
        student_ctx = _student_context()
        with patch.dict("sys.modules", {"langchain_aws": None}):
            _title, body = await engine.generate_personalized_motivational_content(
                persona_name="Alexandra", student_context=student_ctx
            )

        assert "don't have any open tasks" in body.lower()


class TestPersonaConsistency:
    @pytest.mark.asyncio
    async def test_email_sign_off_uses_the_passed_persona_not_a_fresh_mongo_lookup(self) -> None:
        """The core bug: _send_email_notification used to re-resolve the advisor
        independently via Mongo, which could disagree with the already-resolved
        persona that voiced the body. It must now just use what's passed in."""
        engine = NotificationEngine()
        session = AsyncMock()
        prefs = MagicMock()
        prefs.preferences = {"ai_mentor_addon_active": True, "email_enabled": True}
        prefs_result = MagicMock()
        prefs_result.scalar_one_or_none.return_value = prefs
        session.execute = AsyncMock(return_value=prefs_result)

        captured_persona: list[str] = []

        def _fake_build_email(**kwargs: object) -> tuple[str, str]:
            captured_persona.append(kwargs["advisor_persona"])
            return "<html></html>", "text"

        with (
            patch("aegra_api.services.email_service.build_notification_email", side_effect=_fake_build_email),
            patch("aegra_api.services.email_service.send_email", new=AsyncMock()),
        ):
            await engine._send_email_notification(
                session=session,
                user_id="user-1",
                title="Weekly update",
                content="body",
                category="motivation",
                persona="David",
                student_context={"email": "kudus@example.com", "first_name": "Kudus"},
            )

        # No Mongo lookup should occur at all — persona is taken as given.
        assert captured_persona == ["David"]

    @pytest.mark.asyncio
    async def test_missing_persona_falls_back_to_default_not_a_mongo_lookup(self) -> None:
        engine = NotificationEngine()
        session = AsyncMock()
        prefs = MagicMock()
        prefs.preferences = {"ai_mentor_addon_active": True, "email_enabled": True}
        prefs_result = MagicMock()
        prefs_result.scalar_one_or_none.return_value = prefs
        session.execute = AsyncMock(return_value=prefs_result)

        captured_persona: list[str] = []

        def _fake_build_email(**kwargs: object) -> tuple[str, str]:
            captured_persona.append(kwargs["advisor_persona"])
            return "<html></html>", "text"

        with (
            patch("aegra_api.services.email_service.build_notification_email", side_effect=_fake_build_email),
            patch("aegra_api.services.email_service.send_email", new=AsyncMock()),
        ):
            await engine._send_email_notification(
                session=session,
                user_id="user-1",
                title="Weekly update",
                content="body",
                category="motivation",
                persona="",
                student_context={"email": "kudus@example.com", "first_name": "Kudus"},
            )

        assert captured_persona == [DEFAULT_PERSONA]

    @pytest.mark.asyncio
    async def test_create_notification_prefers_advisor_persona_over_persona_for_email(self) -> None:
        """send_motivational_nudges passes persona=None (skip in-app rewrite) but
        advisor_persona=<resolved> (email sign-off must still be correct)."""
        engine = NotificationEngine()
        session = AsyncMock()
        session.add = MagicMock()
        session.flush = AsyncMock()

        with (
            patch.object(engine, "check_notification_fatigue", new=AsyncMock(return_value=False)),
            patch.object(engine, "should_send", new=AsyncMock(return_value=True)),
            patch.object(engine, "_send_email_notification", new=AsyncMock()) as mock_send_email,
        ):
            await engine.create_notification(
                session=session,
                user_id="user-1",
                title="Sunday update from David",
                content="short in-app text",
                category="motivation",
                persona=None,
                advisor_persona="David",
                student_context={"first_name": "Kudus"},
                email_body_override="Full personalised body.",
            )

        assert mock_send_email.await_args.kwargs["persona"] == "David"

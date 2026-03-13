"""Sync LMS course content into local retrieval tables."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import create_engine, delete
from sqlalchemy.orm import sessionmaker

from aegra_api.core.orm import Base  # type: ignore[import-untyped]
from aegra_api.settings import settings  # type: ignore[import-untyped]
from aegra_api.tools.course_content.lms_client import LMSClient
from aegra_api.tools.course_content.models import (
    CourseContentSyncState,
    LocalCourse,
    LocalLesson,
    LocalMaterial,
    TranscriptSegment,
)


class CourseContentSyncService:
    def __init__(self, database_url: str | None = None, lms_client: LMSClient | None = None):
        self.database_url = database_url or settings.db.database_url
        if not self.database_url:
            raise ValueError("DATABASE_URL is required")
        if "asyncpg" in self.database_url:
            self.database_url = self.database_url.replace("postgresql+asyncpg://", "postgresql+psycopg://")
        self.engine = create_engine(self.database_url)
        self.Session = sessionmaker(bind=self.engine)
        self.lms_client = lms_client or LMSClient()

    def initialize_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    @staticmethod
    def _segment_text(text: str, target_size: int = 1600) -> list[str]:
        cleaned = (text or "").strip()
        if not cleaned:
            return []
        paragraphs = [part.strip() for part in cleaned.split("\n\n") if part.strip()]
        if not paragraphs:
            paragraphs = [cleaned]

        segments: list[str] = []
        current = ""
        for part in paragraphs:
            candidate = f"{current}\n\n{part}".strip() if current else part
            if len(candidate) <= target_size:
                current = candidate
                continue
            if current:
                segments.append(current)
            if len(part) <= target_size:
                current = part
                continue
            start = 0
            while start < len(part):
                segments.append(part[start : start + target_size])
                start += target_size
            current = ""
        if current:
            segments.append(current)
        return segments

    async def sync_course(self, course_id: str) -> dict[str, Any]:
        self.initialize_schema()
        session = self.Session()
        state = session.query(CourseContentSyncState).filter_by(course_id=course_id).first()
        if not state:
            state = CourseContentSyncState(course_id=course_id, status="processing")
            session.add(state)
            session.commit()

        try:
            course = await self.lms_client.get_course(course_id)
            if not course:
                state.status = "failed"
                state.last_error = "Course not found"
                session.commit()
                return {"course_id": course_id, "status": "failed", "error": "Course not found"}

            lessons = await self.lms_client.get_course_lessons(course_id)
            materials = await self.lms_client.get_course_materials(course_id)

            session.execute(delete(TranscriptSegment).where(TranscriptSegment.course_id == course_id))
            session.execute(delete(LocalMaterial).where(LocalMaterial.course_id == course_id))
            session.execute(delete(LocalLesson).where(LocalLesson.course_id == course_id))
            session.execute(delete(LocalCourse).where(LocalCourse.course_id == course_id))
            session.commit()

            session.add(
                LocalCourse(
                    course_id=course.course_id,
                    slug=course.slug,
                    title=course.title,
                    description=course.description,
                    overview=course.overview,
                    track=course.track,
                    levels=course.levels,
                    source_updated_at=course.raw.get("updatedAt") if isinstance(course.raw, dict) else None,
                    synced_at=datetime.utcnow(),
                )
            )

            segment_count = 0
            for lesson in lessons:
                transcript_text = None
                transcript = lesson.get("transcript")
                if isinstance(transcript, str) and transcript.strip():
                    transcript_text = transcript
                    if transcript.startswith(("http://", "https://")):
                        fetched = await self.lms_client._fetch_text_file(transcript)
                        if fetched.strip():
                            transcript_text = fetched
                elif lesson.get("content"):
                    transcript_text = str(lesson.get("content"))

                session.add(
                    LocalLesson(
                        lesson_id=lesson.get("_id"),
                        course_id=course_id,
                        level_title=lesson.get("levelTitle") or lesson.get("level_title"),
                        module_index=lesson.get("moduleIndex") or lesson.get("module_index"),
                        lesson_index=lesson.get("lessonIndex") or lesson.get("lesson_index"),
                        title=lesson.get("title") or "Untitled Lesson",
                        summary=(transcript_text or "")[:800] or None,
                        transcript_text=transcript_text,
                        transcript_source_url=transcript
                        if isinstance(transcript, str) and transcript.startswith(("http://", "https://"))
                        else None,
                        quiz=lesson.get("quiz") or [],
                        lesson_metadata=lesson,
                        synced_at=datetime.utcnow(),
                    )
                )

                if lesson.get("_id") and transcript_text:
                    for idx, segment in enumerate(self._segment_text(transcript_text)):
                        session.add(
                            TranscriptSegment(
                                course_id=course_id,
                                lesson_id=lesson.get("_id"),
                                level_title=lesson.get("levelTitle") or lesson.get("level_title"),
                                module_index=lesson.get("moduleIndex") or lesson.get("module_index"),
                                lesson_index=lesson.get("lessonIndex") or lesson.get("lesson_index"),
                                lesson_title=lesson.get("title"),
                                segment_index=idx,
                                content=segment,
                                synced_at=datetime.utcnow(),
                            )
                        )
                        segment_count += 1

            for material in materials:
                session.add(
                    LocalMaterial(
                        material_id=material.material_id,
                        course_id=course_id,
                        lesson_id=material.lesson_id,
                        lesson_title=material.lesson_title,
                        level_title=material.level_title,
                        module_title=material.module_title,
                        file_type=material.file_type,
                        file_url=material.file_url,
                        download_url=material.download_url,
                        content_text=material.content_text,
                        material_metadata=material.raw,
                        synced_at=datetime.utcnow(),
                    )
                )

            state.status = "completed"
            state.lessons_count = len(lessons)
            state.materials_count = len(materials)
            state.segments_count = segment_count
            state.last_error = None
            state.source_updated_at = course.raw.get("updatedAt") if isinstance(course.raw, dict) else None
            state.synced_at = datetime.utcnow()
            session.commit()
            return {
                "course_id": course_id,
                "status": "completed",
                "lessons": len(lessons),
                "materials": len(materials),
                "segments": segment_count,
            }
        except Exception as e:
            session.rollback()
            state.status = "failed"
            state.last_error = str(e)
            session.commit()
            return {"course_id": course_id, "status": "failed", "error": str(e)}
        finally:
            session.close()

    async def sync_all_courses(self) -> list[dict[str, Any]]:
        courses = await self.lms_client.get_all_courses()
        results: list[dict[str, Any]] = []
        for course in courses:
            results.append(await self.sync_course(course.course_id))
        return results

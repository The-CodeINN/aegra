"""Database models for local-first course content retrieval."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from aegra_api.core.orm import Base


class LocalCourse(Base):
    __tablename__ = "local_courses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    course_id: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    slug: Mapped[str | None] = mapped_column(String(255), nullable=True)
    title: Mapped[str] = mapped_column(String(500))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    overview: Mapped[str | None] = mapped_column(Text, nullable=True)
    track: Mapped[str | None] = mapped_column(String(255), nullable=True)
    levels: Mapped[dict] = mapped_column(JSONB, default=dict)
    source_updated_at: Mapped[str | None] = mapped_column(String(255), nullable=True)
    synced_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class LocalLesson(Base):
    __tablename__ = "local_lessons"
    __table_args__ = (
        UniqueConstraint("course_id", "level_title", "module_index", "lesson_index", name="uq_local_lessons_path"),
        Index("idx_local_lessons_course_level_module", "course_id", "level_title", "module_index"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lesson_id: Mapped[str | None] = mapped_column(String(255), index=True, nullable=True)
    course_id: Mapped[str] = mapped_column(String(255), index=True)
    level_title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    module_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    lesson_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    title: Mapped[str] = mapped_column(String(500))
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    transcript_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    transcript_source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    quiz: Mapped[dict | list] = mapped_column(JSONB, default=list)
    lesson_metadata: Mapped[dict] = mapped_column(JSONB, default=dict)
    synced_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class LocalMaterial(Base):
    __tablename__ = "local_materials"
    __table_args__ = (Index("idx_local_materials_course_lesson", "course_id", "lesson_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    material_id: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    course_id: Mapped[str] = mapped_column(String(255), index=True)
    lesson_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    lesson_title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    level_title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    module_title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    file_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    file_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    download_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    material_metadata: Mapped[dict] = mapped_column(JSONB, default=dict)
    synced_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class TranscriptSegment(Base):
    __tablename__ = "transcript_segments"
    __table_args__ = (
        UniqueConstraint("course_id", "lesson_id", "segment_index", name="uq_transcript_segments_path"),
        Index("idx_transcript_segments_course_lesson", "course_id", "lesson_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    course_id: Mapped[str] = mapped_column(String(255), index=True)
    lesson_id: Mapped[str] = mapped_column(String(255), index=True)
    level_title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    module_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    lesson_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    lesson_title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    segment_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    start_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    synced_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class CourseContentSyncState(Base):
    __tablename__ = "course_content_sync_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    course_id: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(50), default="pending")
    lessons_count: Mapped[int] = mapped_column(Integer, default=0)
    materials_count: Mapped[int] = mapped_column(Integer, default=0)
    segments_count: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_updated_at: Mapped[str | None] = mapped_column(String(255), nullable=True)
    synced_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )


class CourseContentEvent(Base):
    __tablename__ = "course_content_events"
    __table_args__ = (Index("idx_course_content_events_status_created", "status", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String(100))
    course_id: Mapped[str | None] = mapped_column(String(255), index=True, nullable=True)
    lesson_id: Mapped[str | None] = mapped_column(String(255), index=True, nullable=True)
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)
    status: Mapped[str] = mapped_column(String(50), default="pending")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    is_repair: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

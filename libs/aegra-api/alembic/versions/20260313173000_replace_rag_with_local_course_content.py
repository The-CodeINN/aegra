"""replace rag with local course content

Revision ID: b7d9f1c2a345
Revises: a1b2c3d4e5f6
Create Date: 2026-03-13 17:30:00.000000
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "b7d9f1c2a345"
down_revision = "a1b2c3d4e5f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index("idx_indexing_status_course_id", table_name="indexing_status", if_exists=True)
    op.drop_table("indexing_status", if_exists=True)
    op.drop_index("idx_course_chunks_embedding_hnsw", table_name="course_chunks", if_exists=True)
    op.drop_index("idx_course_chunks_chunk_id", table_name="course_chunks", if_exists=True)
    op.drop_index("idx_course_chunks_level", table_name="course_chunks", if_exists=True)
    op.drop_index("idx_course_chunks_content_type", table_name="course_chunks", if_exists=True)
    op.drop_index("idx_course_chunks_course_id", table_name="course_chunks", if_exists=True)
    op.drop_table("course_chunks", if_exists=True)

    op.create_table(
        "local_courses",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("course_id", sa.String(length=255), nullable=False),
        sa.Column("slug", sa.String(length=255), nullable=True),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("overview", sa.Text(), nullable=True),
        sa.Column("track", sa.String(length=255), nullable=True),
        sa.Column("levels", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("source_updated_at", sa.String(length=255), nullable=True),
        sa.Column("synced_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("course_id"),
    )
    op.create_index("ix_local_courses_course_id", "local_courses", ["course_id"], unique=True)

    op.create_table(
        "local_lessons",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("lesson_id", sa.String(length=255), nullable=True),
        sa.Column("course_id", sa.String(length=255), nullable=False),
        sa.Column("level_title", sa.String(length=255), nullable=True),
        sa.Column("module_index", sa.Integer(), nullable=True),
        sa.Column("lesson_index", sa.Integer(), nullable=True),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("transcript_text", sa.Text(), nullable=True),
        sa.Column("transcript_source_url", sa.Text(), nullable=True),
        sa.Column("quiz", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("lesson_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("synced_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("course_id", "level_title", "module_index", "lesson_index", name="uq_local_lessons_path"),
    )
    op.create_index("ix_local_lessons_lesson_id", "local_lessons", ["lesson_id"], unique=False)
    op.create_index(
        "idx_local_lessons_course_level_module",
        "local_lessons",
        ["course_id", "level_title", "module_index"],
        unique=False,
    )
    op.execute(
        "CREATE INDEX idx_local_lessons_fts ON local_lessons USING gin (to_tsvector('english', coalesce(title, '') || ' ' || coalesce(summary, '') || ' ' || coalesce(transcript_text, '')))"
    )

    op.create_table(
        "local_materials",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("material_id", sa.String(length=255), nullable=False),
        sa.Column("course_id", sa.String(length=255), nullable=False),
        sa.Column("lesson_id", sa.String(length=255), nullable=True),
        sa.Column("lesson_title", sa.String(length=500), nullable=True),
        sa.Column("level_title", sa.String(length=255), nullable=True),
        sa.Column("module_title", sa.String(length=500), nullable=True),
        sa.Column("file_type", sa.String(length=100), nullable=True),
        sa.Column("file_url", sa.Text(), nullable=True),
        sa.Column("download_url", sa.Text(), nullable=True),
        sa.Column("content_text", sa.Text(), nullable=True),
        sa.Column("material_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("synced_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("material_id"),
    )
    op.create_index("idx_local_materials_course_lesson", "local_materials", ["course_id", "lesson_id"], unique=False)
    op.create_index("ix_local_materials_material_id", "local_materials", ["material_id"], unique=True)
    op.execute(
        "CREATE INDEX idx_local_materials_fts ON local_materials USING gin (to_tsvector('english', coalesce(lesson_title, '') || ' ' || coalesce(module_title, '') || ' ' || coalesce(content_text, '')))"
    )

    op.create_table(
        "transcript_segments",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("course_id", sa.String(length=255), nullable=False),
        sa.Column("lesson_id", sa.String(length=255), nullable=False),
        sa.Column("level_title", sa.String(length=255), nullable=True),
        sa.Column("module_index", sa.Integer(), nullable=True),
        sa.Column("lesson_index", sa.Integer(), nullable=True),
        sa.Column("lesson_title", sa.String(length=500), nullable=True),
        sa.Column("segment_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("start_seconds", sa.Integer(), nullable=True),
        sa.Column("end_seconds", sa.Integer(), nullable=True),
        sa.Column("synced_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("course_id", "lesson_id", "segment_index", name="uq_transcript_segments_path"),
    )
    op.create_index(
        "idx_transcript_segments_course_lesson", "transcript_segments", ["course_id", "lesson_id"], unique=False
    )
    op.execute(
        "CREATE INDEX idx_transcript_segments_fts ON transcript_segments USING gin (to_tsvector('english', content))"
    )

    op.create_table(
        "course_content_sync_state",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("course_id", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("lessons_count", sa.Integer(), nullable=False),
        sa.Column("materials_count", sa.Integer(), nullable=False),
        sa.Column("segments_count", sa.Integer(), nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("source_updated_at", sa.String(length=255), nullable=True),
        sa.Column("synced_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("course_id"),
    )
    op.create_index("ix_course_content_sync_state_course_id", "course_content_sync_state", ["course_id"], unique=True)

    op.create_table(
        "course_content_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("event_type", sa.String(length=100), nullable=False),
        sa.Column("course_id", sa.String(length=255), nullable=True),
        sa.Column("lesson_id", sa.String(length=255), nullable=True),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("processed_at", sa.DateTime(), nullable=True),
        sa.Column("is_repair", sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_course_content_events_status_created", "course_content_events", ["status", "created_at"], unique=False
    )


def downgrade() -> None:
    op.drop_index("idx_course_content_events_status_created", table_name="course_content_events")
    op.drop_table("course_content_events")
    op.drop_index("ix_course_content_sync_state_course_id", table_name="course_content_sync_state")
    op.drop_table("course_content_sync_state")
    op.drop_index("idx_transcript_segments_fts", table_name="transcript_segments")
    op.drop_index("idx_transcript_segments_course_lesson", table_name="transcript_segments")
    op.drop_table("transcript_segments")
    op.drop_index("idx_local_materials_fts", table_name="local_materials")
    op.drop_index("ix_local_materials_material_id", table_name="local_materials")
    op.drop_index("idx_local_materials_course_lesson", table_name="local_materials")
    op.drop_table("local_materials")
    op.drop_index("idx_local_lessons_fts", table_name="local_lessons")
    op.drop_index("idx_local_lessons_course_level_module", table_name="local_lessons")
    op.drop_index("ix_local_lessons_lesson_id", table_name="local_lessons")
    op.drop_table("local_lessons")
    op.drop_index("ix_local_courses_course_id", table_name="local_courses")
    op.drop_table("local_courses")

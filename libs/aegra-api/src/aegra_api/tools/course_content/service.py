"""Local-first search service for course content."""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from aegra_api.settings import settings  # type: ignore[import-untyped]


class CourseContentService:
    def __init__(self, database_url: str | None = None):
        self.database_url = database_url or settings.db.database_url
        if not self.database_url:
            raise ValueError("DATABASE_URL is required")
        if "asyncpg" in self.database_url:
            self.database_url = self.database_url.replace("postgresql+asyncpg://", "postgresql+psycopg://")
        self.engine = create_engine(self.database_url)
        self.Session = sessionmaker(bind=self.engine)

    def resolve_lesson_reference(self, query: str, course_id: str | None = None) -> dict[str, Any] | None:
        module_match = re.search(r"module\s+(\d+)", query, re.IGNORECASE)
        lesson_match = re.search(r"lesson\s+(\d+)(?:\.(\d+))?", query, re.IGNORECASE)
        if not module_match and not lesson_match:
            return None

        module_index = int(module_match.group(1)) - 1 if module_match else None
        lesson_index = None
        if lesson_match:
            lesson_index = int(lesson_match.group(2) or lesson_match.group(1)) - 1

        params: dict[str, Any] = {
            "course_id": course_id,
            "module_index": module_index,
            "lesson_index": lesson_index,
        }

        session = self.Session()
        try:
            sql = """
            SELECT course_id, lesson_id, title, level_title, module_index, lesson_index, summary, transcript_text
            FROM local_lessons
            WHERE (:course_id IS NULL OR course_id = CAST(:course_id AS VARCHAR))
              AND (:module_index IS NULL OR module_index = CAST(:module_index AS INTEGER))
              AND (:lesson_index IS NULL OR lesson_index = CAST(:lesson_index AS INTEGER))
            ORDER BY course_id, level_title, module_index, lesson_index
            LIMIT 1
            """
            row = session.execute(text(sql), params).mappings().first()
            return dict(row) if row else None
        finally:
            session.close()

    def search(self, query: str, course_id: str | None = None, k: int = 5) -> list[dict[str, Any]]:
        resolved = self.resolve_lesson_reference(query, course_id=course_id)
        if resolved:
            results: list[dict[str, Any]] = []
            lesson_content = (resolved.get("transcript_text") or resolved.get("summary") or "").strip()
            if lesson_content:
                results.append(
                    {
                        "content": lesson_content[:4000],
                        "course_id": resolved.get("course_id"),
                        "title": resolved.get("title"),
                        "content_type": "lesson",
                        "metadata": resolved,
                    }
                )
            return results

        session = self.Session()
        try:
            sql = """
            WITH parsed_query AS (
                SELECT websearch_to_tsquery('english', :query) AS tsq
            ), lesson_hits AS (
                SELECT
                    course_id,
                    title,
                    'lesson' AS content_type,
                    COALESCE(summary, transcript_text, '') AS content,
                    ts_rank_cd(
                        setweight(to_tsvector('english', COALESCE(title, '')), 'A') ||
                        setweight(to_tsvector('english', COALESCE(summary, '')), 'B') ||
                        setweight(to_tsvector('english', COALESCE(transcript_text, '')), 'C'),
                        (SELECT tsq FROM parsed_query),
                        32
                    ) AS bm25_score,
                    CASE
                        WHEN lower(COALESCE(title, '') || ' ' || COALESCE(summary, '') || ' ' || COALESCE(transcript_text, '')) LIKE '%' || lower(:query) || '%' THEN 1.0
                        ELSE 0.0
                    END AS lexical_score,
                    jsonb_build_object(
                        'lesson_id', lesson_id,
                        'level_title', level_title,
                        'module_index', module_index,
                        'lesson_index', lesson_index
                    ) AS metadata
                FROM local_lessons
                WHERE (:course_id IS NULL OR course_id = CAST(:course_id AS VARCHAR))
                  AND to_tsvector('english', COALESCE(title, '') || ' ' || COALESCE(summary, '') || ' ' || COALESCE(transcript_text, '')) @@ (SELECT tsq FROM parsed_query)
            ), material_hits AS (
                SELECT
                    course_id,
                    COALESCE(lesson_title, module_title, file_type, 'Material') AS title,
                    'material' AS content_type,
                    COALESCE(content_text, '') AS content,
                    ts_rank_cd(
                        setweight(to_tsvector('english', COALESCE(lesson_title, '')), 'A') ||
                        setweight(to_tsvector('english', COALESCE(module_title, '')), 'B') ||
                        setweight(to_tsvector('english', COALESCE(content_text, '')), 'C'),
                        (SELECT tsq FROM parsed_query),
                        32
                    ) AS bm25_score,
                    CASE
                        WHEN lower(COALESCE(lesson_title, '') || ' ' || COALESCE(module_title, '') || ' ' || COALESCE(content_text, '')) LIKE '%' || lower(:query) || '%' THEN 1.0
                        ELSE 0.0
                    END AS lexical_score,
                    jsonb_build_object(
                        'material_id', material_id,
                        'lesson_id', lesson_id,
                        'level_title', level_title,
                        'module_title', module_title,
                        'file_url', file_url,
                        'download_url', download_url
                    ) AS metadata
                FROM local_materials
                WHERE (:course_id IS NULL OR course_id = CAST(:course_id AS VARCHAR))
                  AND content_text IS NOT NULL
                  AND to_tsvector('english', COALESCE(lesson_title, '') || ' ' || COALESCE(module_title, '') || ' ' || COALESCE(content_text, '')) @@ (SELECT tsq FROM parsed_query)
            ), segment_hits AS (
                SELECT
                    course_id,
                    COALESCE(lesson_title, 'Transcript') AS title,
                    'transcript_segment' AS content_type,
                    content,
                    ts_rank_cd(to_tsvector('english', content), (SELECT tsq FROM parsed_query), 32) AS bm25_score,
                    CASE
                        WHEN lower(content) LIKE '%' || lower(:query) || '%' THEN 1.0
                        ELSE 0.0
                    END AS lexical_score,
                    jsonb_build_object(
                        'lesson_id', lesson_id,
                        'level_title', level_title,
                        'module_index', module_index,
                        'lesson_index', lesson_index,
                        'segment_index', segment_index
                    ) AS metadata
                FROM transcript_segments
                WHERE (:course_id IS NULL OR course_id = CAST(:course_id AS VARCHAR))
                  AND to_tsvector('english', content) @@ (SELECT tsq FROM parsed_query)
            ), combined_hits AS (
                SELECT * FROM lesson_hits
                UNION ALL
                SELECT * FROM material_hits
                UNION ALL
                SELECT * FROM segment_hits
            ), ranked_hits AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (ORDER BY bm25_score DESC, lexical_score DESC) AS bm25_rank,
                    ROW_NUMBER() OVER (ORDER BY lexical_score DESC, bm25_score DESC) AS lexical_rank
                FROM combined_hits
            )
            SELECT
                course_id,
                title,
                content_type,
                content,
                metadata,
                bm25_score,
                lexical_score,
                (1.0 / (60 + bm25_rank)) + (1.0 / (60 + lexical_rank)) AS hybrid_score
            FROM ranked_hits
            ORDER BY hybrid_score DESC, bm25_score DESC
            LIMIT :limit
            """
            params: dict[str, Any] = {"query": query, "limit": k, "course_id": course_id}
            rows = session.execute(text(sql), params).mappings().all()
            return [
                {
                    "content": (row.get("content") or "")[:4000],
                    "course_id": row.get("course_id"),
                    "title": row.get("title"),
                    "content_type": row.get("content_type"),
                    "metadata": {
                        **(row.get("metadata") or {}),
                        "search_scores": {
                            "hybrid": float(row.get("hybrid_score") or 0.0),
                            "bm25": float(row.get("bm25_score") or 0.0),
                            "lexical": float(row.get("lexical_score") or 0.0),
                        },
                    },
                }
                for row in rows
            ]
        finally:
            session.close()

Mongo-backed course content access

This package provides direct Mongo reads for course and enrollment context.

Principles

- No LMS backend API dependency for course retrieval/search path.
- Scope all course answers by active student enrollments.
- Keep Mongo data-access and ranking logic separated.

Integration

- Use `get_course_content_mongo_client()` for enrollment/course/progress/material reads.
- Use `get_course_content_search_service()` for enrollment-scoped hybrid search.

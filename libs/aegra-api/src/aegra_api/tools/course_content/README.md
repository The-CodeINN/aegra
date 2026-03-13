Local-first course content access

This package replaces the old vector-first RAG setup.

Principles
- Mirror LMS course structure into local PostgreSQL tables.
- Resolve exact lesson references deterministically.
- Use PostgreSQL full-text search for transcript and material lookup.
- Keep LMS API usage in the sync pipeline, not in the user-answer path.

CLI
- Sync one course:
  python -m aegra_api.tools.course_content.cli sync-course --course-id <course_id>
- Sync all courses:
  python -m aegra_api.tools.course_content.cli sync-all
- Search local content:
  python -m aegra_api.tools.course_content.cli search --query "module 2 lesson 3.1"

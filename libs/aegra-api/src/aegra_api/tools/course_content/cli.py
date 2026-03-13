"""CLI for local-first course content sync and search."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from aegra_api.tools.course_content.service import CourseContentService
from aegra_api.tools.course_content.sync import CourseContentSyncService


async def main() -> None:
    parser = argparse.ArgumentParser(description="Local-first course content tools")
    subparsers = parser.add_subparsers(dest="command")

    sync_course = subparsers.add_parser("sync-course")
    sync_course.add_argument("--course-id", required=True)

    subparsers.add_parser("sync-all")

    search = subparsers.add_parser("search")
    search.add_argument("--query", required=True)
    search.add_argument("--course-id")
    search.add_argument("--limit", type=int, default=5)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "sync-course":
        service = CourseContentSyncService()
        result = await service.sync_course(args.course_id)
        print(json.dumps(result, indent=2))
        return

    if args.command == "sync-all":
        service = CourseContentSyncService()
        result = await service.sync_all_courses()
        print(json.dumps(result, indent=2))
        return

    if args.command == "search":
        service = CourseContentService()
        result = service.search(args.query, course_id=args.course_id, k=args.limit)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

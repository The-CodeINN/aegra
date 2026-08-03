"""Enrollment-scoped hybrid search for course content on Mongo.

Search is separated from the Mongo client so data access and ranking logic
can evolve independently.
"""

from __future__ import annotations

from typing import Any

from bson import ObjectId
from bson.errors import InvalidId
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

from aegra_api.tools.course_content.mongo_client import CourseContentMongoClient


class CourseContentHybridSearchService:
    def __init__(self, mongo_client: CourseContentMongoClient):
        self.mongo_client = mongo_client
        self.courses: Collection = mongo_client.db["courses"]

    def _build_allowed_object_ids(
        self, enrolled_course_ids: list[str], requested_course_id: str | None
    ) -> list[ObjectId]:
        allowed_ids = set(enrolled_course_ids)
        if requested_course_id:
            if requested_course_id not in allowed_ids:
                return []
            allowed_ids = {requested_course_id}

        object_ids: list[ObjectId] = []
        for cid in allowed_ids:
            try:
                object_ids.append(ObjectId(cid))
            except (InvalidId, TypeError):
                continue
        return object_ids

    def _atlas_search(self, query: str, object_ids: list[ObjectId], limit: int) -> list[dict[str, Any]]:
        pipeline = [
            {
                "$search": {
                    "index": "default",
                    "compound": {
                        "must": [{"text": {"query": query, "path": {"wildcard": "*"}}}],
                        "filter": [{"in": {"path": "_id", "value": object_ids}}],
                    },
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "title": 1,
                    "slug": 1,
                    "description": 1,
                    "overview": 1,
                    "track": 1,
                    "score": {"$meta": "searchScore"},
                }
            },
            {"$limit": int(max(1, min(limit, 20)))},
        ]
        try:
            return list(self.courses.aggregate(pipeline))
        except PyMongoError:
            return []

    def _lexical_backup(self, query: str, object_ids: list[ObjectId], limit: int) -> list[dict[str, Any]]:
        pattern = {"$regex": query, "$options": "i"}
        return list(
            self.courses.find(
                {
                    "_id": {"$in": object_ids},
                    "$or": [
                        {"title": pattern},
                        {"description": pattern},
                        {"overview": pattern},
                        {"track": pattern},
                    ],
                },
                {
                    "_id": 1,
                    "title": 1,
                    "slug": 1,
                    "description": 1,
                    "overview": 1,
                    "track": 1,
                },
                limit=int(max(1, min(limit, 20))),
            )
        )

    def search_enrolled_course_content(
        self,
        *,
        query: str,
        enrolled_course_ids: list[str],
        limit: int = 5,
        requested_course_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if not query.strip() or not enrolled_course_ids:
            return []

        object_ids = self._build_allowed_object_ids(enrolled_course_ids, requested_course_id)
        if not object_ids:
            return []

        docs = self._atlas_search(query, object_ids, limit)
        source = "mongo_atlas_search"
        if not docs:
            docs = self._lexical_backup(query, object_ids, limit)
            source = "mongo_direct_search"

        results: list[dict[str, Any]] = []
        for doc in docs:
            course_id = self.mongo_client._to_str_id(doc.get("_id"))
            if not course_id:
                continue
            content = "\n\n".join(
                part for part in [doc.get("description") or "", doc.get("overview") or ""] if part
            ).strip()
            if not content:
                content = doc.get("title") or ""

            results.append(
                {
                    "content": content[:4000],
                    "course_id": course_id,
                    "title": doc.get("title") or "Untitled Course",
                    "content_type": "course",
                    "metadata": {
                        "slug": doc.get("slug"),
                        "track": doc.get("track"),
                        "score": doc.get("score"),
                        "source": source,
                    },
                }
            )
        return results


_course_content_search_service: CourseContentHybridSearchService | None = None


def get_course_content_search_service() -> CourseContentHybridSearchService:
    global _course_content_search_service
    if _course_content_search_service is None:
        from aegra_api.tools.course_content.mongo_client import get_course_content_mongo_client

        _course_content_search_service = CourseContentHybridSearchService(get_course_content_mongo_client())
    return _course_content_search_service

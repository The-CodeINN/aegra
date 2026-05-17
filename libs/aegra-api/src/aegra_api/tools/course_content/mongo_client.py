"""Mongo-backed client for course content and enrollment state.

This client is read-only and intended to replace LMS backend API reads for:
- course retrieval and structure
- enrollment-scoped course context
- progress, attempts, and subscription state reads
"""

from __future__ import annotations

from typing import Any

from bson import ObjectId
from pydantic import BaseModel
from pymongo import MongoClient

from aegra_api.settings import settings
from aegra_api.tools.course_content.material_reader import fetch_and_extract_text


class CourseData(BaseModel):
    course_id: str
    slug: str | None = None
    title: str
    description: str | None = None
    overview: str | None = None
    track: str | None = None
    levels: list[dict[str, Any]] = []
    raw: dict[str, Any] = {}


class MaterialData(BaseModel):
    material_id: str
    course_id: str
    lesson_id: str | None = None
    lesson_title: str | None = None
    level_title: str | None = None
    module_title: str | None = None
    file_type: str | None = None
    file_url: str | None = None
    download_url: str | None = None
    content_text: str | None = None
    raw: dict[str, Any] = {}


class CourseContentMongoClient:
    def __init__(self, mongo_uri: str | None = None, db_name: str | None = None):
        self.mongo_uri = mongo_uri or settings.app.MONGODB_URI
        if not self.mongo_uri:
            raise ValueError("MONGODB_URI is required for direct Mongo reads")

        self._client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=10000)
        explicit_db_name = db_name or settings.app.MONGODB_DB_NAME
        if explicit_db_name:
            self.db = self._client[explicit_db_name]
        else:
            default_db = self._client.get_default_database()
            if default_db is None:
                raise ValueError("No default Mongo database in MONGODB_URI and MONGODB_DB_NAME not set")
            self.db = default_db

    @staticmethod
    def _pick_text(data: dict[str, Any], *keys: str) -> str | None:
        for key in keys:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    @staticmethod
    def _normalize_label(value: Any) -> str | None:
        if isinstance(value, str):
            normalized = value.strip().casefold()
            return normalized if normalized else None
        return None

    @staticmethod
    def _to_object_id_or_str(value: str) -> ObjectId | str:
        try:
            return ObjectId(value)
        except Exception:
            return value

    @staticmethod
    def _to_str_id(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, ObjectId):
            return str(value)
        if isinstance(value, str) and value.strip():
            return value
        return None

    @staticmethod
    def _safe_list(value: Any) -> list[Any]:
        return value if isinstance(value, list) else []

    @staticmethod
    def _safe_dict(value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _material_mime_type(file_type: str | None, url: str | None) -> str:
        normalized = (file_type or "").strip().lower()
        if normalized == "pdf" or (url and url.lower().endswith(".pdf")):
            return "application/pdf"
        if normalized in {"txt", "text"}:
            return "text/plain"
        if normalized == "json":
            return "application/json"
        return "application/octet-stream"

    def get_active_enrolled_course_ids(self, user_id: str) -> list[str]:
        if not user_id:
            return []

        student_key = self._to_object_id_or_str(user_id)
        rows = self.db["enrollments"].find({"student": student_key, "status": True}, {"course": 1})

        out: list[str] = []
        seen: set[str] = set()
        for row in rows:
            course_id = self._to_str_id(row.get("course"))
            if course_id and course_id not in seen:
                seen.add(course_id)
                out.append(course_id)
        return out

    def get_course(self, course_id: str) -> CourseData | None:
        doc = self.db["courses"].find_one({"_id": self._to_object_id_or_str(course_id)})
        if not isinstance(doc, dict):
            return None
        return CourseData(
            course_id=self._to_str_id(doc.get("_id")) or course_id,
            slug=doc.get("slug"),
            title=doc.get("title") or "",
            description=doc.get("description"),
            overview=doc.get("overview"),
            track=doc.get("track"),
            levels=self._safe_list(doc.get("levels")),
            raw=doc,
        )

    def get_all_courses(self) -> list[CourseData]:
        rows = self.db["courses"].find(
            {},
            {"title": 1, "slug": 1, "description": 1, "overview": 1, "track": 1, "levels": 1},
        )
        out: list[CourseData] = []
        for doc in rows:
            course_id = self._to_str_id(doc.get("_id"))
            if not course_id:
                continue
            out.append(
                CourseData(
                    course_id=course_id,
                    slug=doc.get("slug"),
                    title=doc.get("title") or "",
                    description=doc.get("description"),
                    overview=doc.get("overview"),
                    track=doc.get("track"),
                    levels=self._safe_list(doc.get("levels")),
                    raw=doc,
                )
            )
        return out

    def get_course_levels(self, course_id: str) -> list[dict[str, Any]]:
        course = self.get_course(course_id)
        return [level for level in (course.levels if course else []) if isinstance(level, dict)]

    def get_course_modules(self, course_id: str) -> list[dict[str, Any]]:
        modules: list[dict[str, Any]] = []
        for level_index, level in enumerate(self.get_course_levels(course_id)):
            level_title = level.get("levelTitle") or level.get("title")
            for module_index, module in enumerate(self._safe_list(level.get("modules"))):
                if not isinstance(module, dict):
                    continue
                item = dict(module)
                item.setdefault("levelTitle", level_title)
                item.setdefault("levelIndex", level_index)
                item.setdefault("moduleIndex", module_index)
                modules.append(item)
        return modules

    def get_course_lessons(self, course_id: str) -> list[dict[str, Any]]:
        lessons: list[dict[str, Any]] = []
        for level_index, level in enumerate(self.get_course_levels(course_id)):
            level_title = level.get("levelTitle") or level.get("title")
            for module_index, module in enumerate(self._safe_list(level.get("modules"))):
                if not isinstance(module, dict):
                    continue
                module_title = module.get("moduleTitle") or module.get("title")
                for lesson_index, lesson in enumerate(self._safe_list(module.get("lessons"))):
                    if not isinstance(lesson, dict):
                        continue
                    item = dict(lesson)
                    item.setdefault("levelTitle", level_title)
                    item.setdefault("levelIndex", level_index)
                    item.setdefault("moduleTitle", module_title)
                    item.setdefault("moduleIndex", module_index)
                    item.setdefault("lessonIndex", lesson_index)
                    lessons.append(item)
        return lessons

    def get_course_materials(
        self,
        course_id: str,
        *,
        include_content: bool = False,
        limit: int | None = None,
    ) -> list[MaterialData]:
        materials: list[MaterialData] = []
        for lesson in self.get_course_lessons(course_id):
            resources = self._safe_list(lesson.get("materials") or lesson.get("resources"))
            for idx, item in enumerate(resources):
                if not isinstance(item, dict):
                    continue
                lesson_id = self._to_str_id(lesson.get("_id"))
                file_type = item.get("fileType") or item.get("type")
                file_url = item.get("fileUrl") or item.get("url")
                download_url = item.get("downloadUrl")
                content_text = item.get("content") if isinstance(item.get("content"), str) else None
                if include_content and not content_text:
                    for material_url in [download_url, file_url]:
                        if not material_url:
                            continue
                        mime_type = self._material_mime_type(file_type, material_url)
                        if mime_type not in {"application/pdf", "text/plain", "application/json"}:
                            continue
                        content_text = fetch_and_extract_text(
                            material_url,
                            mime_type,
                            str(item.get("title") or f"material-{idx}"),
                        )
                        if content_text:
                            break
                materials.append(
                    MaterialData(
                        material_id=str(item.get("_id") or item.get("id") or f"{lesson_id or 'lesson'}:{idx}"),
                        course_id=course_id,
                        lesson_id=lesson_id,
                        lesson_title=lesson.get("title"),
                        level_title=lesson.get("levelTitle"),
                        module_title=lesson.get("moduleTitle"),
                        file_type=file_type,
                        file_url=file_url,
                        download_url=download_url,
                        content_text=content_text,
                        raw=item,
                    )
                )
                if limit is not None and len(materials) >= limit:
                    return materials
        return materials

    def get_enrollment_overview(self, user_id: str) -> dict[str, Any]:
        student_key = self._to_object_id_or_str(user_id)
        rows = list(
            self.db["enrollments"].find(
                {"student": student_key, "status": True},
                {
                    "course": 1,
                    "progress": 1,
                    "overallProgress": 1,
                    "totalCompletedLessons": 1,
                    "totalWatchedHours": 1,
                    "finalAssessmentPassed": 1,
                    "enrolledAt": 1,
                    "updatedAt": 1,
                    "aiMentorActive": 1,
                },
            )
        )
        course_ids = [row.get("course") for row in rows if row.get("course") is not None]
        course_docs = {
            self._to_str_id(doc.get("_id")): doc
            for doc in self.db["courses"].find({"_id": {"$in": course_ids}}, {"title": 1, "slug": 1, "track": 1})
        }

        enrollments: list[dict[str, Any]] = []
        for row in rows:
            course_id = self._to_str_id(row.get("course"))
            course_meta = course_docs.get(course_id) or {}
            enrollments.append(
                {
                    "courseId": course_id,
                    "course": {
                        "title": course_meta.get("title"),
                        "slug": course_meta.get("slug"),
                        "track": course_meta.get("track"),
                    },
                    "overallProgress": row.get("overallProgress", 0),
                    "progress": self._safe_list(row.get("progress")),
                    "totalCompletedLessons": row.get("totalCompletedLessons", 0),
                    "totalWatchedHours": row.get("totalWatchedHours", 0),
                    "finalAssessmentPassed": row.get("finalAssessmentPassed", False),
                    "aiMentorActive": row.get("aiMentorActive", False),
                    "enrolledAt": row.get("enrolledAt"),
                    "updatedAt": row.get("updatedAt"),
                }
            )
        return {"enrollments": enrollments}

    def get_course_progress(self, user_id: str, course_id: str) -> dict[str, Any] | None:
        student_key = self._to_object_id_or_str(user_id)
        course_key = self._to_object_id_or_str(course_id)
        doc = self.db["enrollments"].find_one(
            {"student": student_key, "course": course_key, "status": True},
            {
                "progress": 1,
                "overallProgress": 1,
                "totalCompletedLessons": 1,
                "totalWatchedHours": 1,
                "finalAssessmentPassed": 1,
                "updatedAt": 1,
            },
        )
        if not isinstance(doc, dict):
            return None
        return {
            "courseId": course_id,
            "overallProgress": doc.get("overallProgress", 0),
            "progress": self._safe_list(doc.get("progress")),
            "totalCompletedLessons": doc.get("totalCompletedLessons", 0),
            "totalWatchedHours": doc.get("totalWatchedHours", 0),
            "finalAssessmentPassed": doc.get("finalAssessmentPassed", False),
            "updatedAt": doc.get("updatedAt"),
        }

    def get_course_structure(self, user_id: str, course_id: str) -> dict[str, Any] | None:
        course = self.get_course(course_id)
        if course is None:
            return None

        enrollment_progress = self.get_course_progress(user_id, course_id)
        progress_levels = self._safe_list((enrollment_progress or {}).get("progress"))

        progress_by_level_title: dict[str, dict[str, Any]] = {}
        progress_by_level_index: dict[int, dict[str, Any]] = {}
        for idx, level in enumerate(progress_levels):
            if not isinstance(level, dict):
                continue
            level_title = self._pick_text(level, "levelTitle", "title", "name")
            normalized_level_title = self._normalize_label(level_title)
            if normalized_level_title:
                progress_by_level_title[normalized_level_title] = level
            level_idx = level.get("levelIndex")
            if isinstance(level_idx, int):
                progress_by_level_index[level_idx] = level
            elif idx not in progress_by_level_index:
                progress_by_level_index[idx] = level

        levels_out: list[dict[str, Any]] = []
        for level_index, level in enumerate(course.levels):
            if not isinstance(level, dict):
                continue
            level_title = self._pick_text(level, "levelTitle", "title", "name") or f"Level {level_index + 1}"
            level_progress = (
                progress_by_level_title.get(self._normalize_label(level_title) or "")
                or progress_by_level_index.get(level_index)
                or {}
            )

            progress_modules_by_title: dict[str, dict[str, Any]] = {}
            progress_modules_by_index: dict[int, dict[str, Any]] = {}
            for idx, item in enumerate(self._safe_list(level_progress.get("modules"))):
                if not isinstance(item, dict):
                    continue
                module_progress_title = self._pick_text(item, "moduleTitle", "title", "name")
                normalized_module_progress_title = self._normalize_label(module_progress_title)
                if normalized_module_progress_title:
                    progress_modules_by_title[normalized_module_progress_title] = item
                module_idx = item.get("moduleIndex")
                if isinstance(module_idx, int):
                    progress_modules_by_index[module_idx] = item
                elif idx not in progress_modules_by_index:
                    progress_modules_by_index[idx] = item

            modules_out: list[dict[str, Any]] = []
            for module_index, module in enumerate(self._safe_list(level.get("modules"))):
                if not isinstance(module, dict):
                    continue
                module_title = self._pick_text(module, "moduleTitle", "title", "name") or f"Module {module_index + 1}"
                module_progress = (
                    progress_modules_by_title.get(self._normalize_label(module_title) or "")
                    or progress_modules_by_index.get(module_index)
                    or {}
                )
                completed_lessons = self._safe_list(module_progress.get("completedLessons"))
                lesson_progress_by_id = {
                    self._to_str_id(item.get("lessonId")): item
                    for item in completed_lessons
                    if isinstance(item, dict) and self._to_str_id(item.get("lessonId"))
                }

                lessons_out: list[dict[str, Any]] = []
                for lesson_index, lesson in enumerate(self._safe_list(module.get("lessons"))):
                    if not isinstance(lesson, dict):
                        continue
                    lesson_id = self._to_str_id(lesson.get("_id"))
                    lesson_progress = lesson_progress_by_id.get(lesson_id, {})
                    lessons_out.append(
                        {
                            "lessonId": lesson_id,
                            "title": lesson.get("title"),
                            "description": lesson.get("description") or lesson.get("summary"),
                            "lessonIndex": lesson.get("lessonIndex", lesson_index),
                            "locked": lesson_progress.get("locked", False),
                            "completed": lesson_progress.get("completed", False),
                            "watchedPercentage": lesson_progress.get("watchedPercentage", 0),
                            "duration": lesson_progress.get("duration", 0),
                        }
                    )

                modules_out.append(
                    {
                        "moduleTitle": module_title,
                        "moduleIndex": module.get("moduleIndex", module_index),
                        "progressPercentage": module_progress.get("progressPercentage", 0),
                        "assessmentPassed": module_progress.get("assessmentPassed", False),
                        "lessons": lessons_out,
                    }
                )

            levels_out.append(
                {
                    "levelTitle": level_title,
                    "levelIndex": level.get("levelIndex", level_index),
                    "progressPercentage": level_progress.get("progressPercentage", 0),
                    "assessmentScore": level_progress.get("assessmentScore"),
                    "assessmentPassed": level_progress.get("assessmentPassed", False),
                    "completed": level_progress.get("completed", False),
                    "testedOut": level_progress.get("testedOut", False),
                    "modules": modules_out,
                }
            )

        return {
            "courseId": course_id,
            "title": course.title,
            "slug": course.slug,
            "track": course.track,
            "levels": levels_out,
            "overallProgress": (enrollment_progress or {}).get("overallProgress", 0),
            "updatedAt": (enrollment_progress or {}).get("updatedAt"),
        }

    def get_student_attempts(self, user_id: str) -> dict[str, Any]:
        student_key = self._to_object_id_or_str(user_id)
        rows = list(
            self.db["assessmentattempts"]
            .find(
                {"student": student_key},
                {
                    "course": 1,
                    "levelTitle": 1,
                    "moduleTitle": 1,
                    "score": 1,
                    "passed": 1,
                    "attemptType": 1,
                    "duration": 1,
                    "submittedAt": 1,
                },
            )
            .sort("submittedAt", -1)
        )
        return {
            "attempts": [
                {
                    "courseId": self._to_str_id(row.get("course")),
                    "levelTitle": row.get("levelTitle"),
                    "moduleTitle": row.get("moduleTitle"),
                    "score": row.get("score"),
                    "passed": row.get("passed"),
                    "attemptType": row.get("attemptType"),
                    "duration": row.get("duration"),
                    "submittedAt": row.get("submittedAt"),
                }
                for row in rows
            ]
        }

    def get_subscription_state(self, user_id: str) -> dict[str, Any] | None:
        user_key = self._to_object_id_or_str(user_id)
        doc = self.db["subscriptions"].find_one(
            {"user": user_key},
            {
                "plan": 1,
                "planType": 1,
                "track": 1,
                "duration": 1,
                "amount": 1,
                "paymentStatus": 1,
                "startDate": 1,
                "endDate": 1,
                "active": 1,
                "activeState": 1,
                "aiMentorAddOn": 1,
                "updatedAt": 1,
            },
            sort=[("updatedAt", -1)],
        )
        if not isinstance(doc, dict):
            return None
        return {
            "plan": self._to_str_id(doc.get("plan")),
            "planType": doc.get("planType"),
            "track": doc.get("track"),
            "duration": doc.get("duration"),
            "amount": doc.get("amount"),
            "paymentStatus": doc.get("paymentStatus"),
            "startDate": doc.get("startDate"),
            "endDate": doc.get("endDate"),
            "active": doc.get("active"),
            "activeState": doc.get("activeState"),
            "aiMentorAddOn": self._safe_dict(doc.get("aiMentorAddOn")),
            "updatedAt": doc.get("updatedAt"),
        }

    def get_user_onboarding_data(self, user_id: str) -> dict[str, Any] | None:
        """Return onboarding location and goal data directly from the LMS database.

        Reads the ``aimentoronboardings`` collection which backs the
        ``/api/v1/ai-mentor/onboarding/me`` LMS endpoint. Used by background
        scheduler jobs that have no user JWT token.

        Returns a dict with ``resident_country``, ``work_countries``, and
        ``target_role``, or ``None`` if no completed onboarding data is found.
        """
        user_key = self._to_object_id_or_str(user_id)
        doc = self.db["aimentoronboardings"].find_one(
            {"user": user_key},
            {"onboarding.s2": 1, "onboarding.s4": 1},
        )
        if not isinstance(doc, dict):
            return None
        onboarding = self._safe_dict(doc.get("onboarding"))
        s2 = self._safe_dict(onboarding.get("s2"))
        s4 = self._safe_dict(onboarding.get("s4"))
        resident_country: str | None = None
        work_countries: list[str] = []
        target_role: str | None = None
        if s2.get("completed"):
            resident_country = self._pick_text(s2, "residentCountry")
            work_country = s2.get("workCountry")
            if isinstance(work_country, list):
                work_countries = [c for c in work_country if isinstance(c, str) and c.strip()]
        if s4.get("completed"):
            target_role = self._pick_text(s4, "targetRole")
        if not resident_country and not work_countries and not target_role:
            return None
        return {
            "resident_country": resident_country,
            "work_countries": work_countries,
            "target_role": target_role,
        }

    def get_learning_track(self, user_id: str) -> str | None:
        """Return the user's active learning track directly from the LMS database.

        Used by background jobs (scheduler) that have no user JWT token.
        Queries the subscriptions collection, which is the same source of
        truth used by the LMS subscription API endpoint.

        Args:
            user_id: The user's unique identifier.

        Returns:
            The track slug (e.g. ``"ai-engineering"``) or ``None`` if not found.
        """
        sub = self.get_subscription_state(user_id)
        if sub and isinstance(sub.get("track"), str) and sub["track"].strip():
            return sub["track"].strip()
        return None


_mongo_course_client: CourseContentMongoClient | None = None


def get_course_content_mongo_client() -> CourseContentMongoClient:
    global _mongo_course_client
    if _mongo_course_client is None:
        _mongo_course_client = CourseContentMongoClient()
    return _mongo_course_client

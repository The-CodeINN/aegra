"""LMS client for local-first course content sync."""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel

from aegra_api.settings import settings  # type: ignore[import-untyped]


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


class LMSClient:
    def __init__(self, base_url: str | None = None, admin_token: str | None = None):
        self.base_url = (base_url or settings.app.LMS_URL).rstrip("/")
        self.admin_token = admin_token or settings.app.ADMIN_TOKEN
        if not self.admin_token:
            raise ValueError("ADMIN_TOKEN is required for LMS API access")
        self.headers = {
            "Authorization": f"Bearer {self.admin_token}",
            "Content-Type": "application/json",
            "accept": "*/*",
        }

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    async def _fetch_text_file(self, url: str) -> str:
        if not url or not url.startswith(("http://", "https://")):
            return ""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=20.0)
                response.raise_for_status()
                content_type = (response.headers.get("content-type") or "").lower()
                if (
                    "text/" in content_type
                    or "json" in content_type
                    or url.lower().endswith((".txt", ".md", ".csv", ".json", ".srt", ".vtt"))
                ):
                    return response.text
        except Exception:
            return ""
        return ""

    async def get_course(self, course_id: str) -> CourseData | None:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    self._url(f"/api/v1/courses/{course_id}"), headers=self.headers, timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                course = data.get("course", {}) if isinstance(data, dict) else {}
                if not isinstance(course, dict):
                    return None
                return CourseData(
                    course_id=str(course.get("_id") or course_id),
                    slug=course.get("slug"),
                    title=course.get("title", ""),
                    description=course.get("description"),
                    overview=course.get("overview"),
                    track=course.get("track"),
                    levels=course.get("levels", []),
                    raw=course,
                )
            except httpx.HTTPError:
                return None

    async def get_all_courses(self) -> list[CourseData]:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(self._url("/api/v1/courses"), headers=self.headers, timeout=30.0)
                response.raise_for_status()
                data = response.json()
                items = data.get("courses", []) if isinstance(data, dict) else []
                courses: list[CourseData] = []
                for item in items:
                    if not isinstance(item, dict) or not item.get("_id"):
                        continue
                    courses.append(
                        CourseData(
                            course_id=str(item.get("_id")),
                            slug=item.get("slug"),
                            title=item.get("title", ""),
                            description=item.get("description"),
                            overview=item.get("overview"),
                            track=item.get("track"),
                            levels=item.get("levels", []),
                            raw=item,
                        )
                    )
                return courses
            except httpx.HTTPError:
                return []

    async def get_course_lessons(self, course_id: str) -> list[dict[str, Any]]:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    self._url(f"/api/v1/courses/{course_id}/lessons"), headers=self.headers, timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                lessons = data.get("lessons", []) if isinstance(data, dict) else []
                return [item for item in lessons if isinstance(item, dict)]
            except httpx.HTTPError:
                return []

    async def get_course_materials(self, course_id: str) -> list[MaterialData]:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    self._url(f"/api/v1/courses/{course_id}/materials"), headers=self.headers, timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                items = data.get("materials", []) if isinstance(data, dict) else []
                materials: list[MaterialData] = []
                for idx, item in enumerate(items):
                    if not isinstance(item, dict):
                        continue
                    file_url = item.get("fileUrl") or item.get("url")
                    download_url = item.get("downloadUrl")
                    content_text = await self._fetch_text_file(download_url or file_url or "")
                    materials.append(
                        MaterialData(
                            material_id=str(item.get("_id") or item.get("id") or idx),
                            course_id=course_id,
                            lesson_id=item.get("lessonId"),
                            lesson_title=item.get("lessonTitle"),
                            level_title=item.get("levelTitle"),
                            module_title=item.get("moduleTitle"),
                            file_type=item.get("fileType") or item.get("type"),
                            file_url=file_url,
                            download_url=download_url,
                            content_text=content_text or None,
                            raw=item,
                        )
                    )
                return materials
            except httpx.HTTPError:
                return []

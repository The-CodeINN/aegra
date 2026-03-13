"""Remote material reading helpers for course content.

Supports extracting text from remote lesson materials, especially PDFs,
so the agent can read enrolled course materials directly from their URLs.
"""

from __future__ import annotations

import io
from typing import Final

import httpx

MAX_DOWNLOAD_SIZE: Final[int] = 8 * 1024 * 1024
MAX_CONTENT_LENGTH: Final[int] = 50000
MAX_PDF_PAGES: Final[int] = 10


def extract_text_from_bytes(data: bytes, mime_type: str, filename: str = "unknown") -> str:
    if mime_type == "application/pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        text_parts: list[str] = []
        pages_to_extract = min(MAX_PDF_PAGES, len(reader.pages))
        for page_num, page in enumerate(reader.pages[:pages_to_extract]):
            text = page.extract_text()
            if text:
                text_parts.append(f"--- Page {page_num + 1} ---\n{text}")

        full_text = "\n\n".join(text_parts) if text_parts else f"[No text found in PDF: {filename}]"
        truncated = full_text[:MAX_CONTENT_LENGTH]
        suffix = ""
        if len(full_text) > MAX_CONTENT_LENGTH:
            suffix = "\n\n[...content truncated for length...]"
        elif len(reader.pages) > pages_to_extract:
            suffix = f"\n\n[...{len(reader.pages) - pages_to_extract} more pages not shown...]"
        return truncated + suffix

    if mime_type.startswith("text/") or mime_type in {
        "application/json",
        "application/xml",
        "application/javascript",
        "application/typescript",
    }:
        text = data.decode("utf-8", errors="replace")
        return text[:MAX_CONTENT_LENGTH] + (
            "\n\n[...content truncated for length...]" if len(text) > MAX_CONTENT_LENGTH else ""
        )

    return f"[Unsupported material type for extraction: {filename} ({mime_type})]"


def fetch_and_extract_text(url: str, mime_type: str, filename: str = "unknown") -> str | None:
    if not url or not url.startswith(("http://", "https://")):
        return None
    try:
        with httpx.Client(timeout=20.0, follow_redirects=True) as client, client.stream("GET", url) as response:
            response.raise_for_status()
            content = bytearray()
            for chunk in response.iter_bytes():
                content.extend(chunk)
                if len(content) > MAX_DOWNLOAD_SIZE:
                    return f"[Material too large to extract safely: {filename}]"

        return extract_text_from_bytes(bytes(content), mime_type, filename)
    except Exception:
        return None

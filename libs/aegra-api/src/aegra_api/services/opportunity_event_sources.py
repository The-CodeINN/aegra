"""Event-specific scraping providers for opportunity discovery."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

import httpx
import structlog

logger = structlog.get_logger()


EVENTBRITE_LOCATION_SLUGS: dict[str, str] = {
    "remote": "online",
    "online": "online",
    "virtual": "online",
    "gb": "united-kingdom--gb",
    "uk": "united-kingdom--gb",
    "united kingdom": "united-kingdom--gb",
    "us": "united-states--us",
    "usa": "united-states--us",
    "united states": "united-states--us",
    "ca": "canada--ca",
    "canada": "canada--ca",
    "au": "australia--au",
    "australia": "australia--au",
    "de": "germany--de",
    "germany": "germany--de",
    "fr": "france--fr",
    "france": "france--fr",
    "nl": "netherlands--nl",
    "netherlands": "netherlands--nl",
    "ie": "ireland--ie",
    "ireland": "ireland--ie",
    "ng": "nigeria--ng",
    "nigeria": "nigeria--ng",
    "ke": "kenya--ke",
    "kenya": "kenya--ke",
    "za": "south-africa--za",
    "south africa": "south-africa--za",
    "in": "india--in",
    "india": "india--in",
    "sg": "singapore--sg",
    "singapore": "singapore--sg",
    "ae": "united-arab-emirates--ae",
    "united arab emirates": "united-arab-emirates--ae",
    "se": "sweden--se",
    "sweden": "sweden--se",
    "no": "norway--no",
    "norway": "norway--no",
    "dk": "denmark--dk",
    "denmark": "denmark--dk",
    "fi": "finland--fi",
    "finland": "finland--fi",
    "es": "spain--es",
    "spain": "spain--es",
    "it": "italy--it",
    "italy": "italy--it",
    "pt": "portugal--pt",
    "portugal": "portugal--pt",
    "pl": "poland--pl",
    "poland": "poland--pl",
    "ch": "switzerland--ch",
    "switzerland": "switzerland--ch",
    "at": "austria--at",
    "austria": "austria--at",
    "be": "belgium--be",
    "belgium": "belgium--be",
    "nz": "new-zealand--nz",
    "new zealand": "new-zealand--nz",
    "jp": "japan--jp",
    "japan": "japan--jp",
    "br": "brazil--br",
    "brazil": "brazil--br",
    "mx": "mexico--mx",
    "mexico": "mexico--mx",
}


def _clean_text(value: str | None) -> str:
    if not value:
        return ""
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", value)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _parse_event_date(raw_value: str | None) -> datetime | None:
    if not raw_value:
        return None

    normalized = raw_value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _extract_location_name(location: Any) -> str:
    if isinstance(location, str):
        return location.strip()
    if not isinstance(location, dict):
        return ""

    name = str(location.get("name") or "").strip()
    if name:
        return name

    address = location.get("address") or location.get("postalAddress") or {}
    parts = [
        str(address.get("streetAddress") or "").strip(),
        str(address.get("addressLocality") or "").strip(),
        str(address.get("addressRegion") or "").strip(),
        str(address.get("addressCountry") or "").strip(),
    ]
    return ", ".join(part for part in parts if part)


@dataclass(slots=True)
class RawEventOpportunity:
    title: str
    url: str
    description: str = ""
    location: str = ""
    event_date: datetime | None = None
    source: str = "eventbrite"
    source_query: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class EventbriteSource:
    name = "eventbrite"

    def _location_slug(self, location: str) -> str:
        normalized = location.strip().lower()
        if not normalized:
            return "online"
        if normalized in EVENTBRITE_LOCATION_SLUGS:
            return EVENTBRITE_LOCATION_SLUGS[normalized]
        return _slugify(normalized) or "online"

    def build_search_url(self, query: str, location: str) -> str:
        location_slug = self._location_slug(location)
        return f"https://www.eventbrite.com/d/{location_slug}/all-events/?q={quote(query)}"

    async def _fetch_html(self, url: str) -> str:
        async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as client:
            response = await client.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Referer": "https://www.eventbrite.com/",
                },
            )
            response.raise_for_status()
            return response.text

    def _extract_json_ld_blocks(self, html: str) -> list[dict[str, Any]]:
        matches = re.findall(
            r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>',
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        blocks: list[dict[str, Any]] = []
        for match in matches:
            payload = match.strip()
            if not payload:
                continue
            try:
                decoded = json.loads(payload)
            except json.JSONDecodeError:
                continue

            if isinstance(decoded, list):
                for item in decoded:
                    if isinstance(item, dict):
                        blocks.append(item)
            elif isinstance(decoded, dict):
                blocks.append(decoded)

        return blocks

    def _extract_events_from_blocks(
        self,
        blocks: list[dict[str, Any]],
        query: str,
        search_url: str,
    ) -> list[RawEventOpportunity]:
        events: list[RawEventOpportunity] = []
        seen_urls: set[str] = set()

        def _walk(node: Any) -> None:
            if isinstance(node, list):
                for item in node:
                    _walk(item)
                return

            if not isinstance(node, dict):
                return

            node_type = str(node.get("@type") or "")
            if node_type == "ItemList":
                _walk(node.get("itemListElement") or [])
                return

            if node_type == "ListItem":
                _walk(node.get("item"))
                return

            if node_type and node_type != "Event":
                return

            title = str(node.get("name") or node.get("title") or "").strip()
            url = str(node.get("url") or "").strip()
            if not title or not url or url in seen_urls:
                return

            seen_urls.add(url)
            events.append(
                RawEventOpportunity(
                    title=title,
                    url=url,
                    description=_clean_text(node.get("description"))[:1200],
                    location=_extract_location_name(node.get("location")),
                    event_date=_parse_event_date(node.get("startDate")),
                    source=self.name,
                    source_query=query,
                    metadata={
                        "image": node.get("image"),
                        "search_url": search_url,
                    },
                )
            )

        for block in blocks:
            _walk(block)

        return events

    async def fetch(self, query: str, location: str) -> list[RawEventOpportunity]:
        search_url = self.build_search_url(query, location)
        try:
            html = await self._fetch_html(search_url)
        except Exception as exc:
            logger.warning(
                "eventbrite_fetch_failed",
                query=query,
                location=location,
                error=repr(exc),
                error_type=type(exc).__name__,
            )
            return []

        blocks = self._extract_json_ld_blocks(html)
        events = self._extract_events_from_blocks(blocks, query, search_url)
        logger.info("eventbrite_search_completed", query=query, location=location, result_count=len(events))
        return events

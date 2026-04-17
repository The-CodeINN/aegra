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

    address = location.get("address") or location.get("postalAddress") or {}
    country = str(address.get("addressCountry") or "").strip()

    name = str(location.get("name") or "").strip()
    if name:
        # Always append the country code so location filters can match by ISO code (e.g. "GB", "NG")
        if country and country.upper() not in name.upper():
            return f"{name}, {country}"
        return name

    parts = [
        str(address.get("streetAddress") or "").strip(),
        str(address.get("addressLocality") or "").strip(),
        str(address.get("addressRegion") or "").strip(),
        country,
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


# ---------------------------------------------------------------------------
# Meetup source — scrapes /find/ search page JSON-LD
# ---------------------------------------------------------------------------

# Meetup uses plain location strings (city or country name) in the URL
MEETUP_ONLINE_LOCATIONS: frozenset[str] = frozenset({"remote", "online", "virtual", "anywhere"})


def _strip_markdown(text: str) -> str:
    """Remove common markdown syntax so descriptions are plain text."""
    # Bold / italic
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    # Links: [text](url)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Headings
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Code blocks
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]+`", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class MeetupSource:
    """Scrapes Meetup's public /find/ search page for events matching a keyword.

    Uses the JSON-LD structured data embedded in the page — the same approach
    used by EventbriteSource — so no API key or authentication is required.

    The search is always augmented with an ``ONLINE`` pass so that virtual
    events relevant to the user's track are surfaced regardless of location.
    Physical-location results are also fetched when a non-remote location is
    provided; the caller (opportunity_discovery) handles post-filtering.
    """

    name = "meetup"

    _BASE_HEADERS: dict[str, str] = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.meetup.com/",
    }

    def _is_online_location(self, location: str) -> bool:
        return location.strip().lower() in MEETUP_ONLINE_LOCATIONS

    def build_search_url(self, query: str, location: str, online_only: bool = False) -> str:
        params = f"keywords={quote(query)}&source=EVENTS"
        if online_only or self._is_online_location(location):
            params += "&online_events_only=true"
        else:
            params += f"&location={quote(location)}"
        return f"https://www.meetup.com/find/?{params}"

    async def _fetch_html(self, url: str) -> str:
        async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as client:
            response = await client.get(url, headers=self._BASE_HEADERS)
            response.raise_for_status()
            return response.text

    def _extract_events_from_html(self, html: str, query: str, search_url: str) -> list[RawEventOpportunity]:
        """Parse JSON-LD blocks from the Meetup /find/ page.

        Meetup embeds events as a flat JSON array (each item has @type=Event)
        in the second <script type="application/ld+json"> block.
        """
        ld_matches = re.findall(
            r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>',
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )

        events: list[RawEventOpportunity] = []
        seen_urls: set[str] = set()

        for match in ld_matches:
            payload = match.strip()
            if not payload:
                continue
            try:
                decoded = json.loads(payload)
            except json.JSONDecodeError:
                continue

            # Normalise to a flat list of candidate objects
            candidates: list[Any] = []
            if isinstance(decoded, list):
                candidates = decoded
            elif isinstance(decoded, dict):
                t = decoded.get("@type")
                if t == "ItemList":
                    for elem in decoded.get("itemListElement") or []:
                        if isinstance(elem, dict):
                            candidates.append(elem.get("item") or elem)
                elif t == "Event":
                    candidates = [decoded]

            for item in candidates:
                if not isinstance(item, dict) or item.get("@type") != "Event":
                    continue

                title = str(item.get("name") or "").strip()
                url = str(item.get("url") or "").strip()
                if not title or not url or url in seen_urls:
                    continue
                seen_urls.add(url)

                # Location: VirtualLocation → "online"; Place → use name
                raw_loc = item.get("location") or {}
                loc_type = str(raw_loc.get("@type") or "") if isinstance(raw_loc, dict) else ""
                if "virtual" in loc_type.lower() or "online" in loc_type.lower():
                    location_str = "online"
                else:
                    location_str = _extract_location_name(raw_loc)

                description = _strip_markdown(_clean_text(item.get("description")))[:1200]

                events.append(
                    RawEventOpportunity(
                        title=title,
                        url=url,
                        description=description,
                        location=location_str,
                        event_date=_parse_event_date(item.get("startDate")),
                        source=self.name,
                        source_query=query,
                        metadata={"search_url": search_url},
                    )
                )

        return events

    async def fetch(self, query: str, location: str) -> list[RawEventOpportunity]:
        """Fetch events for a keyword query and location.

        Always runs two searches: one for online events and (if a non-remote
        location is given) one scoped to that physical location.  Duplicates
        are removed by URL before returning.
        """
        results: list[RawEventOpportunity] = []
        seen_urls: set[str] = set()
        is_online = self._is_online_location(location)

        # Online pass — always run
        online_url = self.build_search_url(query, location, online_only=True)
        try:
            html = await self._fetch_html(online_url)
            for ev in self._extract_events_from_html(html, query, online_url):
                if ev.url not in seen_urls:
                    seen_urls.add(ev.url)
                    results.append(ev)
        except Exception as exc:
            logger.warning(
                "meetup_fetch_failed",
                query=query,
                location="online",
                error=repr(exc),
                error_type=type(exc).__name__,
            )

        # Physical-location pass — only when a real location is specified
        if not is_online:
            phys_url = self.build_search_url(query, location, online_only=False)
            try:
                html = await self._fetch_html(phys_url)
                for ev in self._extract_events_from_html(html, query, phys_url):
                    if ev.url not in seen_urls:
                        seen_urls.add(ev.url)
                        results.append(ev)
            except Exception as exc:
                logger.warning(
                    "meetup_fetch_failed",
                    query=query,
                    location=location,
                    error=repr(exc),
                    error_type=type(exc).__name__,
                )

        logger.info(
            "meetup_search_completed",
            query=query,
            location=location,
            result_count=len(results),
        )
        return results


# ---------------------------------------------------------------------------
# Lu.ma source — uses the public discover API (slug-based, no auth required)
# ---------------------------------------------------------------------------

# Only these two slugs return 200 from the Lu.ma discover API.
# All data-related tracks map to "tech"; AI engineering maps to "ai".
_LUMA_AI_KEYWORDS: frozenset[str] = frozenset(
    {"ai", "llm", "machine learning", "deep learning", "neural", "generative", "mlops", "rag"}
)

_LUMA_BASE_URL = "https://api2.luma.com/discover/get-paginated-events"
_LUMA_EVENT_BASE = "https://lu.ma/"


def _luma_slug_for_query(query: str) -> str:
    """Map a track keyword query to the best Lu.ma category slug."""
    q = query.lower()
    if any(kw in q for kw in _LUMA_AI_KEYWORDS):
        return "ai"
    return "tech"


class LumaSource:
    """Fetches events from the Lu.ma discover API.

    Lu.ma does not support keyword search — only category slugs ('tech', 'ai').
    We map each incoming track query to the closest slug and rely on the
    caller's `_score_text` filter (title-based) to keep only relevant events.
    Two pages of results are fetched per call to increase coverage.
    """

    name = "luma"

    _HEADERS: dict[str, str] = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0 Safari/537.36"
        ),
        "Accept": "application/json",
        "Referer": "https://lu.ma/",
    }

    def _is_online(self, event: dict) -> bool:
        return str(event.get("location_type") or "").lower() == "online"

    def _location_str(self, event: dict) -> str:
        if self._is_online(event):
            return "online"
        geo = event.get("geo_address_info") or {}
        return str(geo.get("city_state") or geo.get("city") or "").strip()

    async def _fetch_page(
        self,
        client: httpx.AsyncClient,
        slug: str,
        cursor: str | None,
        limit: int,
    ) -> dict:
        params: dict[str, str] = {"slug": slug, "pagination_limit": str(limit)}
        if cursor:
            params["after_cursor"] = cursor
        resp = await client.get(_LUMA_BASE_URL, params=params, headers=self._HEADERS, timeout=15.0)
        resp.raise_for_status()
        return resp.json()

    def _parse_entries(self, entries: list, query: str, slug: str) -> list[RawEventOpportunity]:
        results: list[RawEventOpportunity] = []
        for entry in entries:
            event = entry.get("event") or {}
            name = str(event.get("name") or "").strip()
            url_slug = str(event.get("url") or "").strip()
            if not name or not url_slug:
                continue
            url = _LUMA_EVENT_BASE + url_slug
            results.append(
                RawEventOpportunity(
                    title=name,
                    url=url,
                    description="",  # Lu.ma API does not expose event descriptions
                    location=self._location_str(event),
                    event_date=_parse_event_date(event.get("start_at")),
                    source=self.name,
                    source_query=query,
                    metadata={"luma_api_id": event.get("api_id"), "luma_slug": slug},
                )
            )
        return results

    async def fetch(self, query: str, location: str) -> list[RawEventOpportunity]:
        slug = _luma_slug_for_query(query)
        results: list[RawEventOpportunity] = []
        seen_urls: set[str] = set()

        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                # Fetch up to 2 pages (20 events) per call
                cursor: str | None = None
                for _ in range(2):
                    data = await self._fetch_page(client, slug, cursor, limit=10)
                    entries = data.get("entries") or []
                    for ev in self._parse_entries(entries, query, slug):
                        if ev.url not in seen_urls:
                            seen_urls.add(ev.url)
                            results.append(ev)
                    if not data.get("has_more"):
                        break
                    cursor = data.get("next_cursor")
                    if not cursor:
                        break
        except Exception as exc:
            logger.warning(
                "luma_fetch_failed",
                query=query,
                slug=slug,
                error=repr(exc),
                error_type=type(exc).__name__,
            )

        logger.info("luma_search_completed", query=query, slug=slug, result_count=len(results))
        return results

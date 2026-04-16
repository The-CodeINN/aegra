"""Runtime monkey-patches for upstream python-jobspy bugs.

These fixes are applied once at import time and survive across all
environments (dev, CI, Docker) without requiring source edits in .venv.

Fixes applied
-------------
glassdoor/_get_location  — URL-encode the location term before interpolating
    into the query string. Raw strings like "United Kingdom" or "New York, NY"
    produce a malformed URL → Glassdoor returns HTTP 400 → zero results.

glassdoor/_fetch_jobs_page  — Glassdoor's /graph endpoint sometimes returns
    partial errors (e.g. 503 on SEO metadata) alongside valid job listing
    data. The upstream code treats ANY "errors" key as a total failure and
    discards the response. We only bail when jobListings data is actually
    missing.

Not fixable without a real browser
-----------------------------------
google  — requires cookie-backed sessions, JS execution, and a matching TLS
    fingerprint. Plain requests cannot satisfy Google's bot detection.
zip_recruiter  — returns 403 without valid account credentials.
"""

from __future__ import annotations


def _apply() -> None:
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from urllib.parse import quote

        import jobspy.glassdoor as _gd_mod
        import requests
        from jobspy.exception import GlassdoorException
        from jobspy.glassdoor.util import get_cursor_for_page
        from jobspy.model import JobPost, ScraperInput

        _Glassdoor = _gd_mod.Glassdoor
        log = _gd_mod.log  # type: ignore[attr-defined]

        # --- Fix 1: URL-encode location in _get_location ---
        _orig_get_location = _Glassdoor._get_location

        def _patched_get_location(self, location: str, is_remote: bool):
            if location and not is_remote:
                location = quote(location)
            return _orig_get_location(self, location, is_remote)

        _Glassdoor._get_location = _patched_get_location

        # --- Fix 2: Treat partial GraphQL errors as non-fatal ---
        def _patched_fetch_jobs_page(
            self,
            scraper_input: ScraperInput,
            location_id: int,
            location_type: str,
            page_num: int,
            cursor: str | None,
        ):
            jobs: list[JobPost] = []
            self.scraper_input = scraper_input
            try:
                payload = self._add_payload(location_id, location_type, page_num, cursor)
                response = self.session.post(
                    f"{self.base_url}/graph",
                    timeout_seconds=15,
                    data=payload,
                )
                if response.status_code != 200:
                    raise GlassdoorException(f"bad response status code: {response.status_code}")
                res_json = response.json()[0]
                # Only treat errors as fatal when job listing data is missing.
                if "errors" in res_json and (
                    "data" not in res_json or not res_json["data"] or "jobListings" not in res_json["data"]
                ):
                    raise ValueError("Error encountered in API response")
            except (
                requests.exceptions.ReadTimeout,
                GlassdoorException,
                ValueError,
                Exception,
            ) as e:
                log.error(f"Glassdoor: {str(e)}")
                return jobs, None

            jobs_data = res_json["data"]["jobListings"]["jobListings"]

            with ThreadPoolExecutor(max_workers=self.jobs_per_page) as executor:
                future_to_job_data = {executor.submit(self._process_job, job): job for job in jobs_data}
                for future in as_completed(future_to_job_data):
                    try:
                        job_post = future.result()
                        if job_post:
                            jobs.append(job_post)
                    except Exception as exc:
                        raise GlassdoorException(f"Glassdoor generated an exception: {exc}")

            return jobs, get_cursor_for_page(res_json["data"]["jobListings"]["paginationCursors"], page_num + 1)

        _Glassdoor._fetch_jobs_page = _patched_fetch_jobs_page  # type: ignore[method-assign]

    except Exception:
        # jobspy not installed or API changed — fail silently so the
        # rest of the application still starts.
        return


_apply()

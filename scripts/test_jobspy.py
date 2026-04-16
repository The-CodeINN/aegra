"""Quick smoke test for python-jobspy — reliable sites only."""

import importlib
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# Apply upstream bug fixes before importing jobspy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "libs", "aegra-api", "src"))
importlib.import_module("aegra_api.utils.jobspy_patches")
scrape_jobs = importlib.import_module("jobspy").scrape_jobs

# Sites confirmed to work without heavy blocking.
# google: requires cookie-backed sessions + JS rendering — returns 0 results with plain requests.
# zip_recruiter: returns 403 (no valid credentials).
SITES_TO_TEST = ["linkedin", "indeed", "glassdoor"]

print(f"Testing sites: {SITES_TO_TEST}")
print("-" * 60)

jobs = scrape_jobs(
    site_name=SITES_TO_TEST,
    search_term="data analyst",
    google_search_term="data analyst jobs in United Kingdom",
    location="United Kingdom",
    results_wanted=3,
    hours_old=72,
    country_indeed="UK",
    verbose=1,
)

print(f"\nFound {len(jobs)} total jobs across all sites")
print("Columns:", list(jobs.columns))
print()

if not jobs.empty:
    # Show breakdown per site
    print("Results per site:")
    for site, count in jobs["site"].value_counts().items():
        print(f"  {site}: {count}")
    print()

    # Show sample rows
    print("Sample jobs:")
    for _, row in jobs.head(10).iterrows():
        print(f"  [{row['site']}] {row['title']} @ {row['company']} — {row['location']}")
        print(f"    URL: {str(row['job_url'])[:80]}")
        min_amt = row.get("min_amount")
        max_amt = row.get("max_amount")
        interval = row.get("interval")
        if min_amt or max_amt:
            print(f"    Salary: {min_amt} - {max_amt} ({interval})")
        print()

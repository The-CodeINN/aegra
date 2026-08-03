import json
from datetime import UTC, datetime

from sqlalchemy import create_engine, text

from aegra_api.settings import settings

engine = create_engine(settings.db.database_url_sqlalchemy_sync)
user_id = "68c30006cc08c47f660b1941"

utc_now = datetime.now(UTC)
local_now = datetime.now()

with engine.connect() as conn:
    result = conn.execute(text("SELECT preferences FROM user_preferences WHERE user_id = :uid"), {"uid": user_id})
    row = result.fetchone()
    print("UTC:", utc_now.isoformat())
    print("Local:", local_now.isoformat())
    if row and row[0]:
        prefs = row[0] if isinstance(row[0], dict) else json.loads(row[0])
        scan_log = prefs.get("scan_log")
        print("scan_log:", json.dumps(scan_log, indent=2, default=str))
    else:
        print("No preferences found for user", user_id)

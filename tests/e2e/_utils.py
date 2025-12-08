import json
import os

try:
    from langgraph_sdk import get_client
except Exception as e:
    raise RuntimeError(
        "langgraph-sdk is required for E2E tests. Install via extras 'e2e' or add to your environment."
    ) from e


def elog(title: str, payload):
    """Emit pretty JSON logs for E2E visibility."""
    try:
        formatted = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    except Exception:
        formatted = str(payload)
    # Use ensure_ascii=True to avoid Windows Unicode encoding issues
    try:
        print(f"\n=== {title} ===\n{formatted}\n")
    except UnicodeEncodeError:
        # Fallback to ASCII-safe version on Windows
        try:
            formatted_safe = json.dumps(payload, ensure_ascii=True, indent=2, default=str)
            print(f"\n=== {title} ===\n{formatted_safe}\n")
        except UnicodeEncodeError:
            # Last resort: encode title and use safe output
            safe_title = title.encode('ascii', 'replace').decode('ascii')
            print(f"\n=== {safe_title} ===\n{formatted_safe}\n")


def get_e2e_client():
    """Construct a LangGraph SDK client from env and log the target URL."""
    server_url = os.getenv("SERVER_URL", "http://localhost:8000")
    print(f"[E2E] Using SERVER_URL={server_url}")
    return get_client(url=server_url)

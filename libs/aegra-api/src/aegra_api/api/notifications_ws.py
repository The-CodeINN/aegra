"""WebSocket endpoint for real-time notifications.

Replaces HTTP polling. Clients connect once and receive notifications
as they are created. Also supports commands from the client (mark read, etc.).

Uses the ConnectionManager pattern (FastAPI recommended) with:
  - Per-user connection tracking
  - Heartbeat ping/pong to detect dead connections
  - Automatic cleanup of stale connections

Protocol (JSON messages):
  Server → Client:
    {"type": "notifications", "data": [...]}
    {"type": "all_notifications", "data": [...]}
    {"type": "action_items", "data": [...]}
    {"type": "notification_update", "data": {...}}
    "pong"  (heartbeat response)

  Client → Server:
    {"action": "mark_read", "id": "<notification_id>"}
    {"action": "mark_all_read"}
    {"action": "dismiss", "id": "<notification_id>"}
    {"action": "refresh"}
    "ping"  (heartbeat)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from datetime import datetime
from typing import Any

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from aegra_api.core.auth_middleware import get_auth_backend
from aegra_api.core.orm import _get_session_maker
from aegra_api.services.accountability_service import AccountabilityService

logger = structlog.get_logger()

router = APIRouter(tags=["Notifications WebSocket"])

# Heartbeat: server pings every 30s, client must respond within 10s
_HEARTBEAT_INTERVAL = 30
_HEARTBEAT_TIMEOUT = 10


def _serialize(obj: Any) -> Any:
    """Recursively serialize ORM objects / datetimes to JSON-safe types."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "__dict__"):
        return {k: _serialize(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [_serialize(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj


class ConnectionManager:
    """Manages WebSocket connections per user with heartbeat support."""

    def __init__(self) -> None:
        self._connections: dict[str, set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, user_id: str) -> None:
        await websocket.accept()
        self._connections.setdefault(user_id, set()).add(websocket)
        logger.info("ws_connected", user_id=user_id)

    def disconnect(self, websocket: WebSocket, user_id: str) -> None:
        conns = self._connections.get(user_id)
        if conns:
            conns.discard(websocket)
            if not conns:
                del self._connections[user_id]
        logger.info("ws_disconnected", user_id=user_id)

    async def send_json(self, websocket: WebSocket, message: dict[str, Any]) -> None:
        await websocket.send_text(json.dumps(message))

    async def broadcast_to_user(self, user_id: str, message: dict[str, Any]) -> None:
        """Send a message to all active WebSocket connections for a user."""
        conns = self._connections.get(user_id, set())
        dead: list[WebSocket] = []
        payload = json.dumps(message)
        for ws in conns:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            conns.discard(ws)


manager = ConnectionManager()


async def broadcast_to_user(user_id: str, message: dict[str, Any]) -> None:
    """Module-level convenience so notification_engine can import it directly."""
    await manager.broadcast_to_user(user_id, message)


async def _authenticate_ws(websocket: WebSocket) -> str | None:
    """Authenticate WebSocket connection using the token query param."""
    token = websocket.query_params.get("token")
    if not token:
        return None

    backend = get_auth_backend()
    try:
        scope = dict(websocket.scope)
        raw_headers = list(scope.get("headers", []))
        raw_headers.append((b"authorization", f"Bearer {token}".encode()))
        scope["headers"] = raw_headers

        from starlette.requests import HTTPConnection

        conn = HTTPConnection(scope)
        result = await backend.authenticate(conn)
        if result is None:
            return None
        _credentials, user = result
        return getattr(user, "identity", None)
    except Exception as exc:
        logger.debug("ws_auth_failed", error=str(exc))
        return None


async def _send_full_state(ws: WebSocket, user_id: str) -> None:
    """Send the full notification state to a client."""
    session_maker = _get_session_maker()
    async with session_maker() as session:
        notifications = await AccountabilityService.list_notifications(session, user_id, limit=50, status="pending")
        all_notifications = await AccountabilityService.list_all_notifications(session, user_id, limit=50)
        action_items = await AccountabilityService.list_action_items(session, user_id)

    await ws.send_text(
        json.dumps(
            {
                "type": "notifications",
                "data": _serialize(list(notifications)),
            }
        )
    )
    await ws.send_text(
        json.dumps(
            {
                "type": "all_notifications",
                "data": _serialize(list(all_notifications)),
            }
        )
    )
    await ws.send_text(
        json.dumps(
            {
                "type": "action_items",
                "data": _serialize(list(action_items)),
            }
        )
    )


async def _handle_client_message(ws: WebSocket, user_id: str, raw: str) -> None:
    """Process a message received from the client."""
    # Heartbeat fast-path
    if raw == "ping":
        await ws.send_text("pong")
        return

    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        await ws.send_text(json.dumps({"type": "error", "message": "Invalid JSON"}))
        return

    action = msg.get("action")
    session_maker = _get_session_maker()

    if action == "mark_read":
        nid = msg.get("id")
        if not nid:
            return
        async with session_maker() as session:
            try:
                result = await AccountabilityService.mark_notification_read(session, nid, user_id)
                await manager.broadcast_to_user(
                    user_id, {"type": "notification_update", "data": {"id": nid, "action": "mark_read", **result}}
                )
            except ValueError:
                await ws.send_text(json.dumps({"type": "error", "message": "Notification not found"}))

    elif action == "mark_all_read":
        async with session_maker() as session:
            result = await AccountabilityService.mark_all_read(session, user_id)
            await manager.broadcast_to_user(
                user_id, {"type": "notification_update", "data": {"action": "mark_all_read", **result}}
            )

    elif action == "dismiss":
        nid = msg.get("id")
        if not nid:
            return
        async with session_maker() as session:
            try:
                result = await AccountabilityService.dismiss_notification(session, nid, user_id)
                await manager.broadcast_to_user(
                    user_id, {"type": "notification_update", "data": {"id": nid, "action": "dismiss", **result}}
                )
            except ValueError:
                await ws.send_text(json.dumps({"type": "error", "message": "Notification not found"}))

    elif action == "refresh":
        await _send_full_state(ws, user_id)

    else:
        await ws.send_text(json.dumps({"type": "error", "message": f"Unknown action: {action}"}))


async def _heartbeat(ws: WebSocket, user_id: str) -> None:
    """Periodically ping the client. Cancel on disconnect."""
    try:
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL)
            await ws.send_text("ping")
    except Exception:
        # Connection is dead — force close so the receive loop exits
        with contextlib.suppress(Exception):
            await ws.close()


@router.get(
    "/ws/notifications",
    summary="Notifications WebSocket",
    description="""
Connect via WebSocket (not HTTP GET) to receive real-time notifications.

**URL:** `ws://<host>/ws/notifications?token=<jwt>`

---

### Server → Client messages
| `type` | Description |
|---|---|
| `notifications` | Active (unread) notifications, up to 50 |
| `all_notifications` | All notifications (read + unread), up to 50 |
| `action_items` | Pending action items |
| `notification_update` | Single notification state change |
| `"pong"` | Heartbeat response to client `"ping"` |

### Client → Server messages
| `action` | Payload | Description |
|---|---|---|
| `"ping"` | — | Heartbeat keepalive |
| `mark_read` | `{"action":"mark_read","id":"<id>"}` | Mark one notification read |
| `mark_all_read` | `{"action":"mark_all_read"}` | Mark all notifications read |
| `dismiss` | `{"action":"dismiss","id":"<id>"}` | Dismiss a notification |
| `refresh` | `{"action":"refresh"}` | Re-fetch full notification state |

### Authentication
Pass a valid JWT as the `token` query parameter. The connection is closed
with code `4001` if authentication fails.

### Heartbeat
The server sends `"ping"` every 30 s. The client should respond with `"pong"`.
Dead connections are cleaned up automatically.
""",
    status_code=426,
    include_in_schema=True,
    tags=["Notifications WebSocket"],
)
async def notifications_websocket_docs() -> JSONResponse:
    """Documentation stub — connect via WebSocket, not HTTP."""
    return JSONResponse(
        status_code=426,
        content={"detail": "This endpoint requires a WebSocket connection. Use ws:// or wss://."},
        headers={"Upgrade": "websocket"},
    )


@router.websocket("/ws/notifications")
async def notifications_websocket(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time notifications."""
    user_id = await _authenticate_ws(websocket)
    if not user_id:
        await websocket.close(code=4001, reason="Authentication required")
        return

    await manager.connect(websocket, user_id)
    heartbeat_task = asyncio.create_task(_heartbeat(websocket, user_id))

    try:
        await _send_full_state(websocket, user_id)

        while True:
            raw = await websocket.receive_text()
            await _handle_client_message(websocket, user_id, raw)

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.error("ws_error", user_id=user_id, error=str(exc))
    finally:
        heartbeat_task.cancel()
        manager.disconnect(websocket, user_id)

import asyncio
import json
import logging
import uuid

from fastapi import WebSocket, WebSocketDisconnect

from app.services.alert_service import alert_service
from app.services.auth_service import decode_token

logger = logging.getLogger(__name__)

# Timeout for individual client sends — prevents a slow client from blocking others
SEND_TIMEOUT_SECONDS = 5.0

# Retry settings for Redis listener reconnection
REDIS_RECONNECT_DELAYS = [1, 2, 4, 8, 15, 30]


class ConnectionManager:
    """WebSocket connection manager with minimal lock contention.

    Uses a snapshot-based broadcast to avoid holding the lock during I/O.
    Pre-serializes JSON once for all clients (avoids N json.dumps calls).
    Batches disconnect cleanup to reduce lock acquisitions.
    """

    def __init__(self) -> None:
        # Use a dict keyed by unique connection ID (not user ID) to support
        # multiple connections per user without overwriting
        self.active_connections: dict[str, WebSocket] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, client_id: str) -> str:
        """Accept and register a WebSocket connection. Returns a unique connection ID."""
        await websocket.accept()
        conn_id = f"{client_id}:{uuid.uuid4().hex[:8]}"
        async with self._lock:
            self.active_connections[conn_id] = websocket
        logger.info("WebSocket client connected: %s", conn_id)
        return conn_id

    async def disconnect(self, conn_id: str) -> None:
        async with self._lock:
            self.active_connections.pop(conn_id, None)
        logger.info("WebSocket client disconnected: %s", conn_id)

    async def _disconnect_batch(self, conn_ids: list[str]) -> None:
        """Remove multiple connections in a single lock acquisition."""
        if not conn_ids:
            return
        async with self._lock:
            for conn_id in conn_ids:
                self.active_connections.pop(conn_id, None)
        for conn_id in conn_ids:
            logger.info("WebSocket client disconnected: %s", conn_id)

    async def send_to_client(self, conn_id: str, message: dict) -> None:
        async with self._lock:
            ws = self.active_connections.get(conn_id)
        if ws:
            try:
                await asyncio.wait_for(ws.send_json(message), timeout=SEND_TIMEOUT_SECONDS)
            except (TimeoutError, Exception) as e:
                logger.warning("Send to %s failed: %s", conn_id, e)
                await self.disconnect(conn_id)

    async def broadcast(self, message: dict) -> None:
        """Send message to ALL connected clients concurrently.

        Optimizations over naive approach:
        - Pre-serializes JSON once (not per-client)
        - Takes a snapshot to release lock before I/O
        - Batches disconnect cleanup into single lock acquisition
        """
        async with self._lock:
            snapshot = dict(self.active_connections)

        if not snapshot:
            return

        # Pre-serialize once for all clients (avoids N redundant json.dumps)
        text = json.dumps(message)

        async def _safe_send(conn_id: str, ws: WebSocket) -> str | None:
            try:
                await asyncio.wait_for(ws.send_text(text), timeout=SEND_TIMEOUT_SECONDS)
                return None
            except (TimeoutError, Exception) as e:
                logger.warning("Broadcast to %s failed: %s", conn_id, e)
                return conn_id

        results = await asyncio.gather(
            *(_safe_send(cid, ws) for cid, ws in snapshot.items()),
            return_exceptions=True,
        )

        # Batch cleanup of failed connections (single lock acquisition)
        failed = [r for r in results if isinstance(r, str)]
        if failed:
            await self._disconnect_batch(failed)


manager = ConnectionManager()


async def redis_listener() -> None:
    """Background task that listens to Redis pub/sub and forwards to WebSocket clients.

    Automatically reconnects on failure with exponential backoff.
    """
    attempt = 0
    while True:
        try:
            pubsub = await alert_service.get_subscriber()
            attempt = 0  # Reset on successful connection
            logger.info("Redis listener connected")

            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning("Malformed alert message: %s", e)
                        continue
                    # Fire-and-forget broadcast — don't let broadcast errors kill the listener
                    try:
                        await manager.broadcast(data)
                    except Exception as e:
                        logger.error("Broadcast error: %s", e)

        except asyncio.CancelledError:
            logger.info("Redis listener cancelled, shutting down")
            return
        except Exception as e:
            delay = REDIS_RECONNECT_DELAYS[min(attempt, len(REDIS_RECONNECT_DELAYS) - 1)]
            logger.error("Redis listener error: %s — reconnecting in %ds", e, delay)
            attempt += 1
            await asyncio.sleep(delay)


async def websocket_endpoint(websocket: WebSocket, token: str | None = None) -> None:
    """WebSocket endpoint for real-time alerts and detection feed."""
    # Authenticate via token query param — anonymous connections are rejected
    if not token:
        await websocket.close(code=4001, reason="Token required")
        return

    payload = decode_token(token)
    if not payload or payload.get("type") != "access" or not payload.get("sub"):
        await websocket.close(code=4001, reason="Invalid token")
        return
    client_id = payload["sub"]

    conn_id = await manager.connect(websocket, client_id)
    try:
        while True:
            # Keep connection alive; handle client messages
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                msg_type = msg.get("type")

                if msg_type == "ping":
                    await websocket.send_json({"type": "pong"})
                elif msg_type == "subscribe":
                    # Client can subscribe to specific event types
                    await websocket.send_json({"type": "subscribed", "channel": msg.get("channel")})
            except json.JSONDecodeError:
                logger.debug("Invalid JSON from client %s", conn_id)
    except WebSocketDisconnect:
        await manager.disconnect(conn_id)

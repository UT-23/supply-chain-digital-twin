"""
WebSocket fan-out for live twin state and agent decisions.

One hub, many viewers. Broadcasts are best-effort: a client that has gone away or
stopped reading is dropped rather than allowed to block the publisher.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any

from fastapi import WebSocket

log = logging.getLogger("ws")

# Per-client send timeout; a slow consumer is disconnected rather than stalling
# the simulation or agent loop behind it.
SEND_TIMEOUT = 5.0


class Hub:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    @property
    def client_count(self) -> int:
        return len(self._clients)

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._clients.add(websocket)
        log.info("ws client connected (%d total)", len(self._clients))

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(websocket)
        log.info("ws client disconnected (%d remaining)", len(self._clients))

    async def send_to(self, websocket: WebSocket, payload: dict[str, Any]) -> None:
        await websocket.send_text(json.dumps(payload, default=str))

    async def broadcast(self, payload: dict[str, Any]) -> None:
        async with self._lock:
            targets = list(self._clients)
        if not targets:
            return

        text = json.dumps(payload, default=str)
        dead: list[WebSocket] = []

        for ws in targets:
            try:
                await asyncio.wait_for(ws.send_text(text), timeout=SEND_TIMEOUT)
            except Exception:  # noqa: BLE001 - any send failure means drop the client
                dead.append(ws)

        if dead:
            async with self._lock:
                for ws in dead:
                    self._clients.discard(ws)
                    with contextlib.suppress(Exception):
                        await ws.close()
            log.info("dropped %d dead ws client(s)", len(dead))


hub = Hub()

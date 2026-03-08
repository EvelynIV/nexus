"""Validated websocket writer for Realtime server events."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Mapping, Optional

from fastapi import WebSocket
from openai.types.realtime import RealtimeError, RealtimeErrorEvent, RealtimeServerEvent
from pydantic import TypeAdapter

from .ids import event_id

_SERVER_EVENT_ADAPTER: TypeAdapter[RealtimeServerEvent] = TypeAdapter(RealtimeServerEvent)


class RealtimeServerWriter:
    """Send only schema-valid Realtime server events."""

    def __init__(self, websocket: WebSocket):
        self._websocket = websocket
        self._lock = asyncio.Lock()

    async def send_event(self, event: Any) -> None:
        payload = self._to_payload(event)
        validated = _SERVER_EVENT_ADAPTER.validate_python(payload)
        async with self._lock:
            await self._websocket.send_text(
                json.dumps(validated.model_dump(exclude_none=True), ensure_ascii=False)
            )

    async def send_error(
        self,
        *,
        message: str,
        error_type: str = "invalid_request_error",
        code: Optional[str] = None,
        event_ref: Optional[str] = None,
        param: Optional[str] = None,
    ) -> None:
        error_event = RealtimeErrorEvent(
            type="error",
            event_id=event_id(),
            error=RealtimeError(
                type=error_type,
                message=message,
                code=code,
                event_id=event_ref,
                param=param,
            ),
        )
        await self.send_event(error_event)

    def _to_payload(self, event: Any) -> Mapping[str, Any]:
        if isinstance(event, Mapping):
            return event
        if hasattr(event, "model_dump"):
            return event.model_dump(exclude_none=True)
        raise TypeError(f"Unsupported event payload type: {type(event)!r}")

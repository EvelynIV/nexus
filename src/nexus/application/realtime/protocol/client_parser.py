"""Parse and validate incoming websocket client events."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from openai.types.realtime import RealtimeClientEvent
from pydantic import TypeAdapter, ValidationError


_CLIENT_EVENT_ADAPTER: TypeAdapter[RealtimeClientEvent] = TypeAdapter(RealtimeClientEvent)


@dataclass
class ClientEventParseError(Exception):
    message: str
    code: str = "invalid_request"
    error_type: str = "invalid_request_error"
    event_id: Optional[str] = None


class RealtimeClientParser:
    """Strict parser for OpenAI Realtime client events."""

    def parse_text(self, raw_text: str) -> RealtimeClientEvent:
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ClientEventParseError(message=f"Invalid JSON payload: {exc.msg}") from exc

        event_id = payload.get("event_id") if isinstance(payload, dict) else None
        try:
            return _CLIENT_EVENT_ADAPTER.validate_python(payload)
        except ValidationError as exc:
            raise ClientEventParseError(
                message=f"Invalid client event: {exc.errors()[0]['msg']}",
                event_id=event_id,
            ) from exc

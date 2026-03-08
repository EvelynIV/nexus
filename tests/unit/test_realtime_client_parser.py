from __future__ import annotations

import pytest
from openai.types.realtime import ResponseCreateEvent

from nexus.application.realtime.protocol.client_parser import (
    ClientEventParseError,
    RealtimeClientParser,
)


def test_parse_valid_response_create_event() -> None:
    parser = RealtimeClientParser()
    event = parser.parse_text('{"type":"response.create"}')

    assert isinstance(event, ResponseCreateEvent)
    assert event.type == "response.create"


def test_parse_invalid_json_raises_error() -> None:
    parser = RealtimeClientParser()

    with pytest.raises(ClientEventParseError) as exc_info:
        parser.parse_text('{"type":"response.create"')

    assert "Invalid JSON payload" in exc_info.value.message


def test_parse_unknown_event_type_raises_error() -> None:
    parser = RealtimeClientParser()

    with pytest.raises(ClientEventParseError) as exc_info:
        parser.parse_text('{"type":"unknown.event"}')

    assert "Invalid client event" in exc_info.value.message

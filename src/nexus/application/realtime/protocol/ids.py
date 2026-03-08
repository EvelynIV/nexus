"""ID generators for realtime protocol resources."""

from __future__ import annotations

import uuid


def event_id() -> str:
    return f"event_{uuid.uuid4()}"


def response_id() -> str:
    return f"resp_{uuid.uuid4()}"


def item_id() -> str:
    return f"item_{uuid.uuid4()}"


def conversation_id() -> str:
    return f"conv_{uuid.uuid4()}"

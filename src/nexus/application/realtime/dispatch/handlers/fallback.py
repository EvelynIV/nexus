from __future__ import annotations

from openai.types.realtime import RealtimeClientEvent

from ..context import RealtimeDispatchContext


async def handle_unknown_event(event: RealtimeClientEvent, ctx: RealtimeDispatchContext) -> None:
    await ctx.session.writer.send_error(
        message=f"Unsupported client event type: {event.type}",
        code="unsupported_event_type",
        error_type="invalid_request_error",
    )

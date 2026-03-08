from __future__ import annotations

from openai.types.realtime import RealtimeClientEvent, ResponseCreateEvent

from ..context import RealtimeDispatchContext


async def handle_response_create(event: RealtimeClientEvent, ctx: RealtimeDispatchContext) -> None:
    assert isinstance(event, ResponseCreateEvent)
    await ctx.service.handle_response_create(ctx.session, event)

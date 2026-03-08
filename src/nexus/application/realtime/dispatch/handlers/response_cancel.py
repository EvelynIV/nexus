from __future__ import annotations

from openai.types.realtime import RealtimeClientEvent, ResponseCancelEvent

from ..context import RealtimeDispatchContext


async def handle_response_cancel(event: RealtimeClientEvent, ctx: RealtimeDispatchContext) -> None:
    assert isinstance(event, ResponseCancelEvent)
    await ctx.service.handle_response_cancel(ctx.session, event)

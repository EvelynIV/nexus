from __future__ import annotations

from openai.types.realtime import RealtimeClientEvent, SessionUpdateEvent

from ..context import RealtimeDispatchContext


async def handle_session_update(event: RealtimeClientEvent, ctx: RealtimeDispatchContext) -> None:
    assert isinstance(event, SessionUpdateEvent)
    await ctx.service.apply_session_update(ctx.session, event.session, model=event.session.model)

from __future__ import annotations

from openai.types.realtime import InputAudioBufferCommitEvent, RealtimeClientEvent

from ..context import RealtimeDispatchContext


async def handle_input_audio_commit(event: RealtimeClientEvent, ctx: RealtimeDispatchContext) -> None:
    assert isinstance(event, InputAudioBufferCommitEvent)
    await ctx.service.handle_input_audio_commit(ctx.session, event)

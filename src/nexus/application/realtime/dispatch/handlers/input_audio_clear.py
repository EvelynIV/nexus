from __future__ import annotations

import asyncio

from openai.types.realtime import InputAudioBufferClearEvent, RealtimeClientEvent

from ..context import RealtimeDispatchContext


async def handle_input_audio_clear(event: RealtimeClientEvent, ctx: RealtimeDispatchContext) -> None:
    assert isinstance(event, InputAudioBufferClearEvent)
    while True:
        try:
            ctx.session.audio_queue.get_nowait()
        except asyncio.QueueEmpty:
            break

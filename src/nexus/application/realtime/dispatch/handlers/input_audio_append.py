from __future__ import annotations

import base64

import numpy as np
from openai.types.realtime import InputAudioBufferAppendEvent, RealtimeClientEvent

from ..context import RealtimeDispatchContext


async def handle_input_audio_append(event: RealtimeClientEvent, ctx: RealtimeDispatchContext) -> None:
    assert isinstance(event, InputAudioBufferAppendEvent)
    audio_base64 = event.audio
    if not audio_base64:
        return

    audio_bytes = base64.b64decode(audio_base64)
    audio_chunk = np.frombuffer(audio_bytes, dtype=np.int16)
    await ctx.session.audio_queue.put(audio_chunk)

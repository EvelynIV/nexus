from __future__ import annotations

import numpy as np
import pytest

from nexus.application.realtime.emitters.response_contexts import AudioResponseContext


class CollectingSession:
    def __init__(self):
        self.events = []

    async def send_event(self, event):
        self.events.append(event)

    def get_cancel_reason(self) -> str:
        return "turn_detected"


class FakeTTSInferencer:
    """Fake TTS that yields valid PCM16 audio data large enough for the
    48 kHz→24 kHz resampler to produce output."""

    async def speech_stream(
        self,
        *,
        input: str,
        model: str = "tts-1",
        voice: str = "alloy",
        response_format: str = "pcm",
        speed: float = 1.0,
        **kwargs,
    ):
        del model, voice, response_format, speed, kwargs
        # Generate ~0.1 s of 48 kHz PCM16 silence-ish data (4800 samples = 9600 bytes).
        # This is large enough for even sinc_best to produce resampled output.
        n_samples = 4800
        pcm = np.zeros(n_samples, dtype=np.int16).tobytes()
        yield pcm


@pytest.mark.asyncio
async def test_audio_context_emits_audio_and_transcript_events_without_text_events() -> None:
    session = CollectingSession()
    ctx = AudioResponseContext(
        session=session,
        tts_inferencer=FakeTTSInferencer(),
        modalities=["audio"],
        format_type="audio/pcm",
        voice="alloy",
        speed=1.0,
    )

    await ctx.__aenter__()
    await ctx.add_model_text_delta("你好，音频模式。")
    await ctx.synthesize_audio()
    await ctx.finish()

    event_types = [event.type for event in session.events]
    assert "response.output_audio.delta" in event_types
    assert "response.output_audio.done" in event_types
    assert "response.output_audio_transcript.delta" in event_types
    assert "response.output_audio_transcript.done" in event_types
    assert "response.output_text.delta" not in event_types
    assert "response.output_text.done" not in event_types
    assert all(not isinstance(event, dict) for event in session.events)


@pytest.mark.asyncio
async def test_audio_context_with_text_modalities_keeps_audio_transcript_events() -> None:
    session = CollectingSession()
    ctx = AudioResponseContext(
        session=session,
        tts_inferencer=FakeTTSInferencer(),
        modalities=["audio", "text"],
        format_type="audio/pcm",
        voice="alloy",
        speed=1.0,
    )

    await ctx.__aenter__()
    await ctx.add_model_text_delta("第一段。")
    await ctx.add_model_text_delta("第二段。")
    await ctx.synthesize_audio()
    await ctx.finish()

    event_types = [event.type for event in session.events]
    assert "response.output_audio.delta" in event_types
    assert "response.output_audio.done" in event_types
    assert "response.output_audio_transcript.delta" in event_types
    assert "response.output_audio_transcript.done" in event_types
    assert "response.output_text.delta" not in event_types
    assert "response.output_text.done" not in event_types

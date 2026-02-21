from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import List

from nexus.infrastructure.tts import Inferencer as TTSInferencer
from nexus.infrastructure.tts.text_normalizer import normalize_for_tts, split_text_by_punctuation

MIN_TTS_SEGMENT_CHARS = 30
DEFAULT_TTS_SEGMENT_CONCURRENCY = 3
AUDIO_CHUNK_SIZE = 4096


def split_text_to_tts_segments(
    text: str,
    *,
    min_segment_chars: int = MIN_TTS_SEGMENT_CHARS,
) -> List[str]:
    """Split text into TTS-ready segments while enforcing a minimum length."""
    normalized = normalize_for_tts(text)
    if not normalized:
        return []

    sentences = split_text_by_punctuation(normalized) or [normalized]

    segments: list[str] = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        current = f"{current}{sentence}" if current else sentence
        if len(current) >= min_segment_chars:
            segments.append(current)
            current = ""

    if current:
        if segments:
            segments[-1] += current
        else:
            # Single short utterance should still be synthesized.
            segments.append(current)

    return [segment for segment in segments if segment]


def realtime_audio_format_to_tts_response_format(format_type: str) -> str:
    """Map Realtime audio format type to TTS API response_format."""
    if format_type == "audio/pcm":
        return "pcm"
    raise ValueError(f"Unsupported realtime audio output format: {format_type}")


async def _collect_segment_chunks(
    inferencer: TTSInferencer,
    segment_text: str,
    voice: str,
    response_format: str,
    speed: float,
) -> list[bytes]:
    """Collect all audio chunks for a single segment asynchronously."""
    chunks: list[bytes] = []
    async for chunk in inferencer.speech_stream(
        input=segment_text,
        voice=voice,
        response_format=response_format,
        speed=speed,
    ):
        if chunk:
            chunks.append(chunk)
    return chunks


async def stream_tts_audio_for_text(
    *,
    inferencer: TTSInferencer,
    text: str,
    voice: str,
    speed: float,
    format_type: str,
    send_chunk: Callable[[bytes], Awaitable[None]],
    min_segment_chars: int = MIN_TTS_SEGMENT_CHARS,
    concurrency: int = DEFAULT_TTS_SEGMENT_CONCURRENCY,
) -> None:
    """Synthesize text in parallel segments and stream chunks in order."""
    segments = split_text_to_tts_segments(text, min_segment_chars=min_segment_chars)
    if not segments:
        return

    response_format = realtime_audio_format_to_tts_response_format(format_type)
    semaphore = asyncio.Semaphore(max(concurrency, 1))

    async def produce_segment(segment_text: str) -> list[bytes]:
        async with semaphore:
            return await _collect_segment_chunks(
                inferencer, segment_text, voice, response_format, speed
            )

    # Launch all segments concurrently (bounded by semaphore).
    tasks = [asyncio.create_task(produce_segment(seg)) for seg in segments]

    try:
        # Stream chunks in segment order.
        for task in tasks:
            chunks = await task
            for chunk in chunks:
                await send_chunk(chunk)
    finally:
        # Cancel any remaining tasks on error.
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

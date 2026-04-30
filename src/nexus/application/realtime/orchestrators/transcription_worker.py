from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

import numpy as np

from nexus.application.realtime.orchestrators.response_orchestrator import (
    TranscriptionStreamTracker,
    send_transcribe_interim,
    send_transcribe_response,
)
from nexus.application.realtime.text_processing import PreparedRealtimeUserTurn, prepare_realtime_user_turn
from nexus.domain.realtime import RealtimeSessionState
from nexus.infrastructure.audio import StreamingResampler
from nexus.infrastructure.asr import AsyncInferencer

logger = logging.getLogger(__name__)


def _should_skip_asr_result(asr_result) -> bool:
    end_time = asr_result.get_end_time()
    if asr_result.is_final:
        return False
    if end_time > 0:
        return False
    logger.info(
        "Skipping non-final ASR result with non-positive end timestamp: end_time=%s transcript=%r",
        end_time,
        asr_result.transcript,
    )
    return True


async def run_transcription_worker(
    *,
    inferencer: AsyncInferencer,
    session: RealtimeSessionState,
    interim_results: bool,
    auto_response_enabled: bool,
    response_worker: Callable[[RealtimeSessionState, PreparedRealtimeUserTurn], Awaitable[None]],
) -> None:
    """Stream ASR results and trigger downstream Responses orchestration.

    When *interim_results* is ``True`` the worker sends incremental
    ``conversation.item.input_audio_transcription.delta`` events for
    every non-final ASR result, giving the client a real-time view of
    the ongoing transcription. Only final ASR results trigger the
    downstream Responses worker when automatic responses are enabled.
    """
    tracker = TranscriptionStreamTracker()

    async def _asr_audio_iter():
        resampler: StreamingResampler | None = None
        if session.audio_input_sample_rate != session.asr_sample_rate:
            resampler = StreamingResampler(
                input_rate=session.audio_input_sample_rate,
                output_rate=session.asr_sample_rate,
            )
            logger.info(
                "ASR streaming resampler enabled %dHz -> %dHz",
                session.audio_input_sample_rate,
                session.asr_sample_rate,
            )

        async for chunk in session.audio_iter():
            if resampler is None:
                yield chunk
                continue

            resampled = await resampler.aprocess(chunk.tobytes())
            if resampled:
                yield np.frombuffer(resampled, dtype=np.int16).copy()

        if resampler is not None:
            tail = await resampler.aflush()
            if tail:
                yield np.frombuffer(tail, dtype=np.int16).copy()

    async for asr_result in inferencer.transcribe(
        _asr_audio_iter(),
        sample_rate=session.asr_sample_rate,
        interim_results=interim_results,
    ):
        if _should_skip_asr_result(asr_result):
            continue

        if not asr_result.is_final:
            # Interim result: send streaming delta, do not trigger a model response.
            try:
                await send_transcribe_interim(
                    session,
                    asr_result,
                    tracker,
                )
            except Exception as exc:  # pragma: no cover
                logger.error("Error sending interim transcribe delta: %s", exc)
            continue

        # Final result – complete the event sequence
        prepared_turn = prepare_realtime_user_turn(asr_result)

        try:
            await send_transcribe_response(
                session,
                asr_result,
                tracker,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Error sending transcribe response: %s", exc)

        if not auto_response_enabled:
            continue

        current_task = session.get_current_response_task()
        if current_task is not None and not current_task.done():
            logger.info("New transcription received, cancelling current response task")
            session.request_cancel()
            current_task.cancel()
            try:
                await current_task
            except asyncio.CancelledError:
                logger.info("Response task cancelled")
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Error awaiting cancelled response task: %s", exc)

        session.reset_cancel()
        response_task = asyncio.create_task(response_worker(session, prepared_turn))
        session.set_current_response_task(response_task)

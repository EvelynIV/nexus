from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from nexus.application.realtime.orchestrators.response_orchestrator import (
    TranscriptionStreamTracker,
    send_transcribe_interim,
    send_transcribe_response,
)
from nexus.domain.realtime import RealtimeSessionState
from nexus.infrastructure.asr import AsyncInferencer

logger = logging.getLogger(__name__)


async def run_transcription_worker(
    *,
    inferencer: AsyncInferencer,
    session: RealtimeSessionState,
    interim_results: bool,
    is_chat_model: bool,
    chat_worker: Callable[[RealtimeSessionState, str], Awaitable[None]],
) -> None:
    """Stream ASR results and trigger downstream chat orchestration.

    When *interim_results* is ``True`` the worker sends incremental
    ``conversation.item.input_audio_transcription.delta`` events for
    every non-final ASR result, giving the client a real-time view of
    the ongoing transcription.  Only ``is_final=True`` results trigger
    the downstream chat worker.
    """
    tracker = TranscriptionStreamTracker()

    async for asr_result in inferencer.transcribe(
        session.audio_iter(),
        sample_rate=session.sample_rate,
        interim_results=interim_results,
    ):
        if not asr_result.is_final:
            # Interim result – send streaming delta, do NOT trigger chat
            try:
                await send_transcribe_interim(session, asr_result, tracker)
            except Exception as exc:  # pragma: no cover
                logger.error("Error sending interim transcribe delta: %s", exc)
            continue

        # Final result – complete the event sequence
        try:
            await send_transcribe_response(session, asr_result, tracker)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Error sending transcribe response: %s", exc)

        if not is_chat_model:
            continue

        current_task = session.get_current_chat_task()
        if current_task is not None and not current_task.done():
            logger.info("New transcription received, cancelling current chat task")
            session.request_cancel()
            current_task.cancel()
            try:
                await current_task
            except asyncio.CancelledError:
                logger.info("Chat task cancelled")
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Error awaiting cancelled chat task: %s", exc)

        session.reset_cancel()
        chat_task = asyncio.create_task(chat_worker(session, asr_result.transcript))
        session.set_current_chat_task(chat_task)

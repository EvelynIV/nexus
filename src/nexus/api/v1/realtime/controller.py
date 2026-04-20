from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from nexus.application.realtime.dispatch import RealtimeDispatchContext, build_default_registry
from nexus.application.realtime.protocol import (
    BroadcastRealtimeSink,
    ClientEventParseError,
    RealtimeClientParser,
    RealtimeReplySink,
)
from nexus.application.realtime.service import RealtimeApplicationService
from nexus.domain.realtime import RealtimeSessionState


logger = logging.getLogger(__name__)


@dataclass
class QueuedClientMessage:
    raw_text: str
    reply_sink: RealtimeReplySink


class RealtimeSessionController:
    def __init__(
        self,
        *,
        session: RealtimeSessionState,
        service: RealtimeApplicationService,
        model: str,
        broadcaster: BroadcastRealtimeSink,
    ) -> None:
        self.session = session
        self.service = service
        self.model = model
        self.broadcaster = broadcaster
        self._parser = RealtimeClientParser()
        self._registry = build_default_registry()
        self._queue: asyncio.Queue[QueuedClientMessage | None] = asyncio.Queue()
        self._dispatch_task: asyncio.Task | None = None
        self._transcription_task: asyncio.Task | None = None
        self._close_lock = asyncio.Lock()
        self._closed = False

    async def start(self) -> None:
        if self._dispatch_task is None:
            self._dispatch_task = asyncio.create_task(self._run_dispatch_loop())
        if self._transcription_task is None:
            self._transcription_task = await self.service.start_transcription_worker(
                self.session,
                is_chat_model="transcribe" not in self.model.lower(),
            )

    async def attach_sink(self, sink: RealtimeReplySink, *, send_session_created: bool = False) -> None:
        await self.broadcaster.add_sink(sink)
        if send_session_created:
            await self.service.emit_session_created(self.session, self.model, sink=sink)

    async def detach_sink(self, sink: RealtimeReplySink) -> None:
        await self.broadcaster.remove_sink(sink)

    async def enqueue_text(self, raw_text: str, reply_sink: RealtimeReplySink) -> None:
        await self._queue.put(QueuedClientMessage(raw_text=raw_text, reply_sink=reply_sink))

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            await self._queue.put(None)
            if self._dispatch_task is not None:
                try:
                    await self._dispatch_task
                except asyncio.CancelledError:
                    pass
            if self._transcription_task is not None:
                self._transcription_task.cancel()
                try:
                    await self._transcription_task
                except asyncio.CancelledError:
                    pass
            await self.session.close_audio_output()
            await self.service.close_session(self.session)

    async def _run_dispatch_loop(self) -> None:
        while True:
            queued = await self._queue.get()
            if queued is None:
                return

            try:
                event = self._parser.parse_text(queued.raw_text)
            except ClientEventParseError as exc:
                await queued.reply_sink.send_error(
                    message=exc.message,
                    error_type=exc.error_type,
                    code=exc.code,
                    event_ref=exc.event_id,
                )
                continue

            try:
                await self._registry.dispatch(
                    event,
                    RealtimeDispatchContext(
                        session=self.session,
                        service=self.service,
                        model=self.model,
                        reply_sink=queued.reply_sink,
                    ),
                )
            except Exception as exc:  # pragma: no cover - defensive boundary
                logger.exception("Unhandled realtime dispatch error: %s", exc)
                await queued.reply_sink.send_error(
                    message=str(exc),
                    error_type="server_error",
                    code="internal_server_error",
                    event_ref=getattr(event, "event_id", None),
                )

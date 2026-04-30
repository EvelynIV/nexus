from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from typing import Any, Optional, Protocol, runtime_checkable


logger = logging.getLogger(__name__)


@runtime_checkable
class RealtimeEventSink(Protocol):
    async def send_event(self, event: Any) -> None: ...


@runtime_checkable
class RealtimeReplySink(RealtimeEventSink, Protocol):
    async def send_error(
        self,
        *,
        message: str,
        error_type: str = "invalid_request_error",
        code: Optional[str] = None,
        event_ref: Optional[str] = None,
        param: Optional[str] = None,
    ) -> None: ...


class BroadcastRealtimeSink:
    """Broadcast realtime server events to every attached transport sink."""

    def __init__(self, sinks: Iterable[RealtimeReplySink] | None = None) -> None:
        self._sinks: list[RealtimeReplySink] = list(sinks or [])
        self._lock = asyncio.Lock()

    async def add_sink(self, sink: RealtimeReplySink) -> None:
        async with self._lock:
            if sink not in self._sinks:
                self._sinks.append(sink)

    async def remove_sink(self, sink: RealtimeReplySink) -> None:
        async with self._lock:
            self._sinks = [item for item in self._sinks if item is not sink]

    async def send_event(self, event: Any) -> None:
        sinks = await self._snapshot()
        for sink in sinks:
            try:
                await sink.send_event(event)
            except Exception as exc:  # pragma: no cover - defensive boundary
                logger.warning("Failed broadcasting realtime event via %r: %s", sink, exc)

    async def send_error(
        self,
        *,
        message: str,
        error_type: str = "invalid_request_error",
        code: Optional[str] = None,
        event_ref: Optional[str] = None,
        param: Optional[str] = None,
    ) -> None:
        sinks = await self._snapshot()
        for sink in sinks:
            try:
                await sink.send_error(
                    message=message,
                    error_type=error_type,
                    code=code,
                    event_ref=event_ref,
                    param=param,
                )
            except Exception as exc:  # pragma: no cover - defensive boundary
                logger.warning("Failed broadcasting realtime error via %r: %s", sink, exc)

    async def size(self) -> int:
        async with self._lock:
            return len(self._sinks)

    async def _snapshot(self) -> list[RealtimeReplySink]:
        async with self._lock:
            return list(self._sinks)


class NullRealtimeReplySink:
    async def send_event(self, event: Any) -> None:
        del event

    async def send_error(
        self,
        *,
        message: str,
        error_type: str = "invalid_request_error",
        code: Optional[str] = None,
        event_ref: Optional[str] = None,
        param: Optional[str] = None,
    ) -> None:
        del message, error_type, code, event_ref, param

"""Event registry for realtime client event handlers."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Dict

from openai.types.realtime import RealtimeClientEvent

from .context import RealtimeDispatchContext

logger = logging.getLogger(__name__)

EventHandler = Callable[[RealtimeClientEvent, RealtimeDispatchContext], Awaitable[None]]


class EventHandlerRegistry:
    def __init__(self):
        self._handlers: Dict[str, EventHandler] = {}
        self._fallback: EventHandler | None = None

    def register(self, event_type: str, handler: EventHandler) -> None:
        self._handlers[event_type] = handler

    def set_fallback(self, handler: EventHandler) -> None:
        self._fallback = handler

    async def dispatch(self, event: RealtimeClientEvent, ctx: RealtimeDispatchContext) -> None:
        handler = self._handlers.get(event.type)
        if handler is None:
            logger.debug("No handler registered for event type: %s", event.type)
            if self._fallback is not None:
                await self._fallback(event, ctx)
            return
        await handler(event, ctx)

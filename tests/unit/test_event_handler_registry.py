from __future__ import annotations

import pytest

from nexus.application.realtime.dispatch.context import RealtimeDispatchContext
from nexus.application.realtime.dispatch.registry import EventHandlerRegistry
from nexus.application.realtime.protocol.client_parser import RealtimeClientParser


@pytest.mark.asyncio
async def test_registry_dispatches_registered_handler_and_fallback() -> None:
    parser = RealtimeClientParser()
    registry = EventHandlerRegistry()

    called = {"main": False, "fallback": False}

    async def main_handler(event, ctx):
        del event, ctx
        called["main"] = True

    async def fallback_handler(event, ctx):
        del event, ctx
        called["fallback"] = True

    registry.register("response.create", main_handler)
    registry.set_fallback(fallback_handler)

    ctx = RealtimeDispatchContext(session=None, service=None, model="model")

    await registry.dispatch(parser.parse_text('{"type":"response.create"}'), ctx)
    await registry.dispatch(parser.parse_text('{"type":"response.cancel"}'), ctx)

    assert called["main"] is True
    assert called["fallback"] is True

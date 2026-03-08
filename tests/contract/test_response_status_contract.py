from __future__ import annotations

import pytest

from nexus.application.realtime.emitters.response_contexts import (
    McpCallResponseContext,
    TextResponseContext,
)


class CollectingSession:
    def __init__(self):
        self.events = []

    async def send_event(self, event):
        self.events.append(event)


@pytest.mark.asyncio
async def test_cancelled_text_response_emits_response_done_with_cancelled_status():
    session = CollectingSession()
    ctx = TextResponseContext(session)

    await ctx.__aenter__()
    await ctx.send_text_delta("partial")
    await ctx.finish(cancelled=True)

    done_events = [event for event in session.events if getattr(event, "type", None) == "response.done"]
    assert done_events
    assert done_events[-1].response.status == "cancelled"
    assert done_events[-1].response.status_details.reason == "turn_detected"


@pytest.mark.asyncio
async def test_mcp_call_error_emits_failed_event():
    session = CollectingSession()
    ctx = McpCallResponseContext(session=session, name="tool", server_label="srv")

    await ctx.__aenter__()
    await ctx.send_arguments_delta('{"x":1}')
    await ctx.finish_arguments()
    ctx.set_error("boom")
    await ctx.__aexit__(Exception, Exception("boom"), None)

    event_types = [getattr(event, "type", None) for event in session.events]
    assert "response.mcp_call.failed" in event_types

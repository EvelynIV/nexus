from __future__ import annotations

import pytest

from nexus.application.realtime.emitters.response_contexts import (
    FunctionCallResponseContext,
    McpListToolsContext,
    McpCallResponseContext,
    TextResponseContext,
)


class CollectingSession:
    def __init__(self):
        self.events = []
        self._conversation_tail_item_id = None
        self._conversation_previous_item_ids = {}

    async def send_event(self, event):
        self.events.append(event)

    def register_server_conversation_item(self, item_id: str):
        if item_id in self._conversation_previous_item_ids:
            return self._conversation_previous_item_ids[item_id]
        previous_item_id = self._conversation_tail_item_id
        self._conversation_previous_item_ids[item_id] = previous_item_id
        self._conversation_tail_item_id = item_id
        return previous_item_id


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


@pytest.mark.asyncio
async def test_text_response_uses_previous_item_id_for_added_and_done():
    session = CollectingSession()
    session.register_server_conversation_item("user_1")
    ctx = TextResponseContext(session)

    await ctx.__aenter__()
    await ctx.send_text_delta("partial")
    await ctx.finish()

    added_event = next(
        event
        for event in session.events
        if getattr(event, "type", None) == "conversation.item.added"
    )
    done_event = next(
        event
        for event in session.events
        if getattr(event, "type", None) == "conversation.item.done"
    )
    assert added_event.previous_item_id == "user_1"
    assert done_event.previous_item_id == "user_1"


@pytest.mark.asyncio
async def test_function_call_uses_previous_item_id_for_added_and_done():
    session = CollectingSession()
    session.register_server_conversation_item("assistant_1")
    ctx = FunctionCallResponseContext(session=session, name="tool", call_id="call_1")

    await ctx.__aenter__()
    await ctx.send_arguments_delta('{"x":1}')
    await ctx.__aexit__(None, None, None)

    added_event = next(
        event
        for event in session.events
        if getattr(event, "type", None) == "conversation.item.added"
    )
    done_event = next(
        event
        for event in session.events
        if getattr(event, "type", None) == "conversation.item.done"
    )
    assert added_event.previous_item_id == "assistant_1"
    assert done_event.previous_item_id == "assistant_1"


@pytest.mark.asyncio
async def test_mcp_list_tools_uses_previous_item_id_for_added_and_done():
    session = CollectingSession()
    session.register_server_conversation_item("assistant_1")
    ctx = McpListToolsContext(session=session, server_label="srv")

    await ctx.__aenter__()
    ctx.set_tools([{"name": "tool"}])
    await ctx.__aexit__(None, None, None)

    added_event = next(
        event
        for event in session.events
        if isinstance(event, dict) and event.get("type") == "conversation.item.added"
    )
    done_event = next(
        event
        for event in session.events
        if isinstance(event, dict) and event.get("type") == "conversation.item.done"
    )
    assert added_event["previous_item_id"] == "assistant_1"
    assert done_event["previous_item_id"] == "assistant_1"


@pytest.mark.asyncio
async def test_mcp_call_uses_previous_item_id_for_added_and_done():
    session = CollectingSession()
    session.register_server_conversation_item("assistant_1")
    ctx = McpCallResponseContext(session=session, name="tool", server_label="srv")

    await ctx.__aenter__()
    await ctx.send_arguments_delta('{"x":1}')
    await ctx.finish_arguments()
    ctx.set_output("ok")
    await ctx.__aexit__(None, None, None)

    added_event = next(
        event
        for event in session.events
        if isinstance(event, dict) and event.get("type") == "conversation.item.added"
    )
    done_event = next(
        event
        for event in session.events
        if isinstance(event, dict) and event.get("type") == "conversation.item.done"
    )
    assert added_event["previous_item_id"] == "assistant_1"
    assert done_event["previous_item_id"] == "assistant_1"

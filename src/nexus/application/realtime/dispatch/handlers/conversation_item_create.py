from __future__ import annotations

from openai.types.realtime import ConversationItemCreateEvent, ConversationItemCreatedEvent, RealtimeClientEvent

from nexus.application.realtime.protocol.ids import event_id, item_id

from ..context import RealtimeDispatchContext


async def handle_conversation_item_create(
    event: RealtimeClientEvent,
    ctx: RealtimeDispatchContext,
) -> None:
    assert isinstance(event, ConversationItemCreateEvent)

    item_payload = event.item.model_dump(exclude_none=True)
    item_payload.setdefault("id", item_id())
    item_payload.setdefault("status", "completed")
    item_payload.setdefault("object", "realtime.item")

    if item_payload.get("type") == "function_call_output":
        ctx.session.add_tool_result(
            tool_call_id=item_payload["call_id"],
            content=item_payload.get("output", ""),
        )

    await ctx.session.send_event(
        ConversationItemCreatedEvent(
            type="conversation.item.created",
            event_id=event_id(),
            item=item_payload,
            previous_item_id=event.previous_item_id,
        )
    )

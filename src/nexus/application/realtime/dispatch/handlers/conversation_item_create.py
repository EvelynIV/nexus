from __future__ import annotations

from typing import Any

from openai.types.realtime import ConversationItemCreateEvent, RealtimeClientEvent

from nexus.application.realtime.protocol.ids import event_id, item_id

from ..context import RealtimeDispatchContext


def _extract_user_text(item_payload: dict[str, Any]) -> str:
    parts: list[str] = []
    for part in item_payload.get("content") or []:
        if not isinstance(part, dict):
            continue
        text = part.get("text") or part.get("transcript")
        if text:
            parts.append(str(text))
    return "".join(parts)


async def handle_conversation_item_create(
    event: RealtimeClientEvent,
    ctx: RealtimeDispatchContext,
) -> None:
    assert isinstance(event, ConversationItemCreateEvent)

    item_payload = event.item.model_dump(exclude_none=True)
    item_payload.setdefault("id", item_id())
    item_payload.setdefault("status", "completed")
    item_payload.setdefault("object", "realtime.item")

    should_continue = False
    if item_payload.get("type") == "function_call_output":
        call_id = item_payload.get("call_id")
        if not call_id:
            await ctx.reply_sink.send_error(
                message="function_call_output item requires call_id",
                error_type="invalid_request_error",
                code="missing_call_id",
                event_ref=getattr(event, "event_id", None),
            )
            return
        ctx.session.add_tool_result(
            tool_call_id=call_id,
            content=item_payload.get("output", ""),
        )
        should_continue = True
    elif item_payload.get("type") == "message" and item_payload.get("role") == "user":
        text = _extract_user_text(item_payload)
        if text:
            ctx.session.responses_session.add_user_message(text)
    elif item_payload.get("type") == "mcp_approval_response":
        ctx.session.responses_session.add_input_item(item_payload)
        should_continue = True

    registered_previous_item_id = ctx.session.register_server_conversation_item(item_payload["id"])
    previous_item_id = event.previous_item_id or registered_previous_item_id
    await ctx.session.send_event(
        {
            "type": "conversation.item.added",
            "event_id": event_id(),
            "item": item_payload,
            "previous_item_id": previous_item_id,
        }
    )
    await ctx.session.send_event(
        {
            "type": "conversation.item.done",
            "event_id": event_id(),
            "item": item_payload,
            "previous_item_id": previous_item_id,
        }
    )
    if should_continue:
        await ctx.service.handle_response_create(ctx.session, event, reply_sink=ctx.reply_sink)

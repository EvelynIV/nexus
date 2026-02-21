from __future__ import annotations

from .registry import EventHandlerRegistry
from .handlers import (
    handle_conversation_item_create,
    handle_input_audio_append,
    handle_input_audio_clear,
    handle_input_audio_commit,
    handle_response_cancel,
    handle_response_create,
    handle_session_update,
    handle_unknown_event,
)


def build_default_registry() -> EventHandlerRegistry:
    registry = EventHandlerRegistry()
    registry.register("session.update", handle_session_update)
    registry.register("input_audio_buffer.append", handle_input_audio_append)
    registry.register("input_audio_buffer.commit", handle_input_audio_commit)
    registry.register("input_audio_buffer.clear", handle_input_audio_clear)
    registry.register("response.create", handle_response_create)
    registry.register("response.cancel", handle_response_cancel)
    registry.register("conversation.item.create", handle_conversation_item_create)
    registry.set_fallback(handle_unknown_event)
    return registry

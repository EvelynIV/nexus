from __future__ import annotations

import json

from .models import AssistantAccumulator, ConversationUpdate, ProcessedRealtimeEvent


class RealtimeEventProcessor:
    def __init__(self) -> None:
        self._user_buffers: dict[str, str] = {}
        self._assistant_buffers: dict[str, AssistantAccumulator] = {}

    def reset(self) -> None:
        self._user_buffers.clear()
        self._assistant_buffers.clear()

    def process(self, event: dict) -> ProcessedRealtimeEvent:
        event_type = str(event.get("type") or "")

        if event_type in {"session.created", "session.updated"}:
            return ProcessedRealtimeEvent(status_message=event_type)

        if event_type == "error":
            payload = event.get("error") or {}
            message = payload.get("message") or event.get("message") or json.dumps(event, ensure_ascii=False)
            return ProcessedRealtimeEvent(error_message=str(message))

        if event_type == "conversation.item.input_audio_transcription.delta":
            key = self._event_key(event)
            self._user_buffers[key] = self._user_buffers.get(key, "") + str(event.get("delta") or "")
            return ProcessedRealtimeEvent(
                chat_updates=[ConversationUpdate(role="user", text=self._user_buffers[key], final=False)]
            )

        if event_type in {
            "conversation.item.input_audio_transcription.completed",
            "conversation.item.input_audio_transcription.done",
        }:
            key = self._event_key(event)
            text = str(event.get("transcript") or self._user_buffers.pop(key, "")).strip()
            updates = [ConversationUpdate(role="user", text=text, final=True)] if text else []
            return ProcessedRealtimeEvent(chat_updates=updates)

        if event_type == "response.output_text.delta":
            state = self._assistant_state(event)
            state.text += str(event.get("delta") or "")
            return ProcessedRealtimeEvent(
                chat_updates=[ConversationUpdate(role="assistant", text=state.text, final=False)]
            )

        if event_type == "response.output_text.done":
            state = self._assistant_state(event)
            text = str(event.get("text") or state.text).strip()
            if text:
                state.emitted = True
                return ProcessedRealtimeEvent(
                    chat_updates=[ConversationUpdate(role="assistant", text=text, final=True)]
                )
            return ProcessedRealtimeEvent()

        if event_type == "response.output_audio_transcript.delta":
            state = self._assistant_state(event)
            state.transcript += str(event.get("delta") or "")
            if not state.text:
                return ProcessedRealtimeEvent(
                    chat_updates=[ConversationUpdate(role="assistant", text=state.transcript, final=False)]
                )
            return ProcessedRealtimeEvent()

        if event_type == "response.output_audio_transcript.done":
            state = self._assistant_state(event)
            transcript = str(event.get("transcript") or state.transcript).strip()
            if transcript and not state.emitted and not state.text.strip():
                state.emitted = True
                return ProcessedRealtimeEvent(
                    chat_updates=[ConversationUpdate(role="assistant", text=transcript, final=True)]
                )
            return ProcessedRealtimeEvent()

        if event_type == "response.done":
            updates: list[ConversationUpdate] = []
            for key, state in list(self._assistant_buffers.items()):
                fallback = state.text.strip() or state.transcript.strip()
                if fallback and not state.emitted:
                    state.emitted = True
                    updates.append(ConversationUpdate(role="assistant", text=fallback, final=True))
                if state.emitted:
                    self._assistant_buffers.pop(key, None)
            return ProcessedRealtimeEvent(chat_updates=updates)

        return ProcessedRealtimeEvent()

    def _assistant_state(self, event: dict) -> AssistantAccumulator:
        key = self._event_key(event)
        state = self._assistant_buffers.get(key)
        if state is None:
            state = AssistantAccumulator()
            self._assistant_buffers[key] = state
        return state

    @staticmethod
    def _event_key(event: dict) -> str:
        return str(
            event.get("item_id")
            or event.get("output_item_id")
            or event.get("response_id")
            or event.get("content_index")
            or event.get("event_id")
            or "default"
        )


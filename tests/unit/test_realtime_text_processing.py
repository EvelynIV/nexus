from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from nexus.application.realtime.orchestrators.response_orchestrator import process_response_stream
from nexus.application.realtime.text_processing import (
    SanitizedModelOutputAccumulator,
    prepare_realtime_user_turn,
)
from nexus.domain.realtime.session_state import RealtimeSessionState
from nexus.infrastructure.asr import TranscriptionResult
from nexus.sessions.responses_session import ResponsesSession


class _FakeResponsesInferencer:
    def __init__(self) -> None:
        self.last_input: list[dict[str, Any]] | None = None

    async def create(self, **kwargs):
        self.last_input = kwargs["input"]

        async def _stream():
            if False:
                yield None

        return _stream()


@dataclass
class _CollectingSession:
    events: list[Any] = field(default_factory=list)
    _conversation_tail_item_id: str | None = None
    _conversation_previous_item_ids: dict[str, str | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.writer = SimpleNamespace(send_error=_noop_async)

    async def send_event(self, event: Any) -> None:
        self.events.append(event)

    def is_cancel_requested(self) -> bool:
        return False

    def get_cancel_reason(self) -> str:
        return "turn_detected"

    def register_server_conversation_item(self, item_id: str) -> str | None:
        if item_id in self._conversation_previous_item_ids:
            return self._conversation_previous_item_ids[item_id]
        previous_item_id = self._conversation_tail_item_id
        self._conversation_previous_item_ids[item_id] = previous_item_id
        self._conversation_tail_item_id = item_id
        return previous_item_id


async def _noop_async(**kwargs) -> None:
    del kwargs


def _event_type(event: Any) -> str | None:
    if isinstance(event, dict):
        return event.get("type")
    return getattr(event, "type", None)


def _event_delta(event: Any) -> str | None:
    if isinstance(event, dict):
        return event.get("delta")
    return getattr(event, "delta", None)


def _event_call_id(event: Any) -> str | None:
    if isinstance(event, dict):
        return event.get("call_id")
    return getattr(event, "call_id", None)


def test_prepare_realtime_user_turn_uses_speaker_name_for_model_context() -> None:
    turn = prepare_realtime_user_turn(
        TranscriptionResult(
            transcript="来给我讲个有趣的故事吧",
            is_final=True,
            speaker_id="speaker-123",
            speaker_name="migo",
            speaker_confidence=0.56,
            metadata={"emotion": "happy"},
            speaker_changed=True,
            turn_completed=True,
        )
    )

    assert turn.raw_transcript == "来给我讲个有趣的故事吧"
    assert turn.display_transcript == "来给我讲个有趣的故事吧"
    assert turn.speaker_id == "speaker-123"
    assert turn.speaker_name == "migo"
    assert turn.speaker_confidence == pytest.approx(0.56)
    assert turn.metadata == {"emotion": "happy"}
    assert turn.speaker_changed is True
    assert turn.turn_completed is True
    assert "当前说话人是migo" in turn.model_text
    assert "用户说：来给我讲个有趣的故事吧" in turn.model_text


def test_prepare_realtime_user_turn_falls_back_to_speaker_id() -> None:
    turn = prepare_realtime_user_turn(
        TranscriptionResult(
            transcript="你好",
            is_final=True,
            speaker_id="speaker-456",
        )
    )

    assert turn.display_transcript == "你好"
    assert turn.speaker_name is None
    assert "当前说话人是speaker-456" in turn.model_text


def test_prepare_realtime_user_turn_without_speaker_keeps_transcript() -> None:
    turn = prepare_realtime_user_turn(
        TranscriptionResult(
            transcript="直接按原文处理",
            is_final=True,
        )
    )

    assert turn.display_transcript == "直接按原文处理"
    assert turn.model_text == "直接按原文处理"


def test_sanitized_output_accumulator_strips_markdown_symbols_and_emoji() -> None:
    accumulator = SanitizedModelOutputAccumulator()

    delta_1 = accumulator.push("### ")
    delta_2 = accumulator.push("**你好")
    delta_3 = accumulator.push("🙂世界**\n- ok")

    assert delta_1 == ("", "")
    assert delta_2 == ("你好", "你好")
    assert delta_3[0] == "世界 ok"
    assert accumulator.display_text == "你好世界 ok"
    assert accumulator.tts_text == "你好世界。ok"


@pytest.mark.asyncio
async def test_process_response_stream_emits_clean_text_deltas() -> None:
    session = _CollectingSession()

    async def _response_stream():
        yield {"type": "response.created", "response": {"id": "resp_test"}}
        yield {
            "type": "response.output_item.added",
            "item": {"id": "msg_test", "type": "message", "role": "assistant", "status": "in_progress"},
            "output_index": 1,
        }
        yield {"type": "response.output_text.delta", "item_id": "msg_test", "delta": "## 你好"}
        yield {"type": "response.output_text.delta", "item_id": "msg_test", "delta": "🙂世界\n"}
        yield {"type": "response.output_text.delta", "item_id": "msg_test", "delta": "**朋友**"}
        yield {
            "type": "response.output_item.done",
            "item": {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "你好世界朋友"}],
            },
        }
        yield {"type": "response.completed", "response": {"id": "resp_test", "status": "completed"}}

    result = await process_response_stream(
        session=session,
        response_stream=_response_stream(),
        modalities=["text"],
    )

    text_deltas = [
        _event_delta(event)
        for event in session.events
        if _event_type(event) == "response.output_text.delta"
    ]

    assert text_deltas == ["你好", "世界", "朋友"]
    assert result.content == "你好世界朋友"


@pytest.mark.asyncio
async def test_process_response_stream_hides_reasoning_text() -> None:
    session = _CollectingSession()

    async def _response_stream():
        yield {"type": "response.created", "response": {"id": "resp_test"}}
        yield {"type": "response.reasoning_text.delta", "delta": "hidden"}
        yield {"type": "response.completed", "response": {"id": "resp_test", "status": "completed"}}

    result = await process_response_stream(
        session=session,
        response_stream=_response_stream(),
        modalities=["text"],
    )

    assert result.content == ""
    assert all(_event_type(event) != "response.reasoning_text.delta" for event in session.events)


@pytest.mark.asyncio
async def test_process_response_stream_forwards_function_call_arguments() -> None:
    session = _CollectingSession()

    async def _response_stream():
        yield {"type": "response.created", "response": {"id": "resp_test"}}
        yield {
            "type": "response.output_item.added",
            "item": {
                "id": "fc_test",
                "type": "function_call",
                "call_id": "call_test",
                "name": "get_weather",
                "arguments": "",
                "status": "in_progress",
            },
        }
        yield {"type": "response.function_call_arguments.delta", "delta": '{"city"'}
        yield {"type": "response.function_call_arguments.delta", "delta": ':"金华"}'}
        yield {
            "type": "response.function_call_arguments.done",
            "arguments": '{"city":"金华"}',
        }
        yield {
            "type": "response.output_item.done",
            "item": {
                "id": "fc_test",
                "type": "function_call",
                "call_id": "call_test",
                "name": "get_weather",
                "arguments": '{"city":"金华"}',
                "status": "completed",
            },
        }
        yield {"type": "response.completed", "response": {"id": "resp_test", "status": "completed"}}

    result = await process_response_stream(
        session=session,
        response_stream=_response_stream(),
        modalities=["text"],
    )

    assert result.tool_call is not None
    assert result.tool_call.call_id == "call_test"
    assert result.tool_call.arguments == '{"city":"金华"}'
    assert [
        _event_delta(event)
        for event in session.events
        if _event_type(event) == "response.function_call_arguments.delta"
    ] == ['{"city"', ':"金华"}']
    done_event = next(
        event for event in session.events
        if _event_type(event) == "response.function_call_arguments.done"
    )
    assert _event_call_id(done_event) == "call_test"


@pytest.mark.asyncio
async def test_process_response_stream_maps_mcp_lifecycle_events() -> None:
    session = _CollectingSession()

    async def _response_stream():
        yield {"type": "response.created", "response": {"id": "resp_test"}}
        yield {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": "mcp_list",
                "type": "mcp_list_tools",
                "server_label": "srv",
                "tools": [],
            },
        }
        yield {"type": "response.mcp_list_tools.in_progress", "output_index": 0}
        yield {"type": "response.mcp_list_tools.failed", "output_index": 0}
        yield {
            "type": "response.output_item.done",
            "item": {
                "id": "mcp_list",
                "type": "mcp_list_tools",
                "server_label": "srv",
                "tools": [],
            },
        }
        yield {"type": "response.completed", "response": {"id": "resp_test", "status": "completed"}}

    await process_response_stream(
        session=session,
        response_stream=_response_stream(),
        modalities=["text"],
    )

    event_types = [_event_type(event) for event in session.events]
    assert "mcp_list_tools.in_progress" in event_types
    assert "mcp_list_tools.failed" in event_types
    assert "conversation.item.done" in event_types


@pytest.mark.asyncio
async def test_process_response_stream_marks_failed_response_done() -> None:
    session = _CollectingSession()

    async def _response_stream():
        yield {"type": "response.created", "response": {"id": "resp_test"}}
        yield {
            "type": "response.failed",
            "response": {
                "id": "resp_test",
                "status": "failed",
                "error": {"code": "upstream_error"},
            },
        }

    result = await process_response_stream(
        session=session,
        response_stream=_response_stream(),
        modalities=["text"],
    )

    assert result.failed is True
    done_event = [
        event for event in session.events
        if _event_type(event) == "response.done"
    ][-1]
    assert done_event.response.status == "failed"


@pytest.mark.asyncio
async def test_realtime_session_response_uses_prepared_turn_model_text() -> None:
    inferencer = _FakeResponsesInferencer()
    responses_session = ResponsesSession(inferencer=inferencer)
    session = RealtimeSessionState(
        responses_session=responses_session,
        response_model="gpt-4o-realtime-preview",
        writer=SimpleNamespace(send_event=_noop_async),
    )
    turn = prepare_realtime_user_turn(
        TranscriptionResult(
            transcript="给我讲个故事",
            is_final=True,
            speaker_name="migo",
        )
    )

    await session.respond_to_user(turn)

    assert inferencer.last_input == [{"role": "user", "content": turn.model_text}]

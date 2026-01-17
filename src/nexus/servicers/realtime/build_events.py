import logging
import uuid
from typing import List, Optional

import numpy as np
from openai.types.realtime import (
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ConversationItemInputAudioTranscriptionDeltaEvent,
    RealtimeResponse,
    RealtimeConversationItemFunctionCall,
    RealtimeSessionCreateRequest,
    ResponseAudioDeltaEvent,
    ResponseAudioTranscriptDeltaEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseOutputItemAddedEvent,
    SessionCreatedEvent,
    SessionUpdatedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseOutputItemAddedEvent,
)
from openai.types.realtime.conversation_item_input_audio_transcription_completed_event import (
    UsageTranscriptTextUsageTokens,
)

logger = logging.getLogger(__name__)


def build_input_audio_transcription_completed(transcript: str):
    event_id = str(uuid.uuid4())
    item_id = str(uuid.uuid4())

    event = ConversationItemInputAudioTranscriptionCompletedEvent(
        content_index=0,
        event_id=event_id,
        item_id=item_id,
        transcript=transcript,
        type="conversation.item.input_audio_transcription.completed",
        usage=UsageTranscriptTextUsageTokens(
            input_tokens=0,
            output_tokens=len(transcript),
            total_tokens=len(transcript),
            type="tokens",
        ),
    )
    return event


def build_session_created(session_id: str, model: str) -> dict:
    """构建 session.created 事件"""
    return {
        "type": "session.created",
        "session": {
            "id": session_id,
            "model": model,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
        },
    }


def build_session_updated(
    session_id: str, model: str, output_modalities: List[str]
) -> dict:
    """构建 session.updated 事件"""
    return {
        "type": "session.updated",
        "session": {
            "id": session_id,
            "model": model,
            "output_modalities": output_modalities,
        },
    }


def build_error_event(error_type: str, message: str) -> dict:
    """构建 error 事件"""
    return {
        "type": "error",
        "error": {
            "type": error_type,
            "message": message,
        },
    }


def build_response_audio_delta(
    response_id: str, item_id: str, audio_delta: str
) -> dict:
    """构建 response.output_audio.delta 事件"""
    return {
        "type": "response.output_audio.delta",
        "response_id": "0",
        "item_id": "0",
        "delta": audio_delta,
    }


def build_response_audio_done(response_id: str, item_id: str) -> dict:
    """构建 response.output_audio.done 事件"""
    return {
        "type": "response.output_audio.done",
        "response_id": "0",
        "item_id": "0",
    }


def build_response_done(response_id: str) -> dict:
    """构建 response.done 事件"""
    return {
        "type": "response.done",
        "response_id": "0",
    }


def build_response_text_delta(text_delta: str):
    """构建 response.text.delta 事件"""
    event = ResponseTextDeltaEvent(
        content_index=0,
        delta=text_delta,
        event_id=str(uuid.uuid4()),
        item_id="0",
        output_index=0,
        response_id="0",
        type="response.output_text.delta",
    )
    return event


def build_response_text_done(text: str):
    """构建 response.text.done 事件"""
    event = ResponseTextDoneEvent(
        content_index=0,
        event_id=str(uuid.uuid4()),
        item_id="0",
        output_index=0,
        response_id="0",
        text=text,
        type="response.output_text.done",
    )
    return event


def build_item_function_call(
    name: str,
    arguments: str,
    call_id: str,
) -> RealtimeConversationItemFunctionCall:
    """构建 RealtimeConversationItemFunctionCall 对象"""
    return ResponseOutputItemAddedEvent(
        event_id=str(uuid.uuid4()),
        item=RealtimeConversationItemFunctionCall(
            name=name,
            arguments=arguments,
            type="function_call",
            call_id=call_id,
        ),
        output_index=0,
        response_id="0",
        type="response.output_item.added",
    )


def build_function_call_arguments_delta(
    arguments_delta: str, call_id: str
) -> ResponseFunctionCallArgumentsDeltaEvent:
    """构建 response.function_call.arguments.delta 事件"""
    event = ResponseFunctionCallArgumentsDeltaEvent(
        event_id=str(uuid.uuid4()),
        call_id=call_id,
        item_id="0",
        output_index=0,
        response_id="0",
        delta=arguments_delta,
        type="response.function_call_arguments.delta",
    )
    return event


def build_function_call_arguments_done(
    arguments: str, call_id: str
) -> ResponseFunctionCallArgumentsDoneEvent:
    """构建 response.function_call.arguments.done 事件"""
    event = ResponseFunctionCallArgumentsDoneEvent(
        event_id=str(uuid.uuid4()),
        call_id=call_id,
        item_id="0",
        output_index=0,
        arguments=arguments,
        response_id="0",
        type="response.function_call_arguments.done",
    )
    return event

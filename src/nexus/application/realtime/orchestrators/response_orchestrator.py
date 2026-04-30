import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterable, Optional, TYPE_CHECKING

from openai.types import realtime

from nexus.infrastructure.asr import TranscriptionResult
from nexus.application.realtime.protocol.ids import conversation_id, event_id, item_id
from nexus.application.realtime.text_processing import (
    SanitizedModelOutputAccumulator,
)
from nexus.application.realtime.emitters.response_contexts import (
    AudioResponseContext,
    FunctionCallResponseContext,
    TextResponseContext,
)
from nexus.application.realtime.emitters.event_factory import (
    build_response_created_event,
    build_response_done_event,
    build_function_call_arguments_done,
)

if TYPE_CHECKING:
    from nexus.domain.realtime import RealtimeSessionState
    from nexus.infrastructure.tts import TTSBackend

logger = logging.getLogger(__name__)


def get_usage_tokens(transcript: str):
    """计算转录文本的使用 token 数"""
    # 简单按空格分词计数，实际可根据具体模型的 tokenizer 实现更精确的计数
    tokens = len(transcript.strip().split())
    usage = realtime.conversation_item_input_audio_transcription_completed_event.UsageTranscriptTextUsageTokens(
        total_tokens=tokens,
        output_tokens=0,
        input_tokens=tokens,
        type="tokens",
    )
    return usage


# ---------------------------------------------------------------------------
# TranscriptionStreamTracker – 追踪流式转写状态，从累积字符串中提取增量 delta
# ---------------------------------------------------------------------------

@dataclass
class TranscriptionStreamTracker:
    """Tracks incremental transcription state across interim ASR results.

    ASR 引擎每次返回累积后的完整字符串（如 "今天的" → "今天的天气真" → "今天的天气真好"），
    本类负责从中提取真正的增量 delta（"今天的" / "天气真" / "好"），
    以符合 OpenAI Realtime API 的 conversation.item.input_audio_transcription.delta 语义。
    """

    _previous_transcript: str = field(default="", init=False)
    _item_id: Optional[str] = field(default=None, init=False)
    _speech_started_sent: bool = field(default=False, init=False)

    @property
    def item_id(self) -> str:
        """当前语句的 item_id，首次访问时自动分配。"""
        if self._item_id is None:
            self._item_id = item_id()
        return self._item_id

    @property
    def speech_started_sent(self) -> bool:
        return self._speech_started_sent

    def mark_speech_started(self) -> None:
        self._speech_started_sent = True

    def compute_delta(self, current_transcript: str) -> str:
        """Compute the incremental delta from the previous transcript.

        If *current_transcript* starts with the previous accumulated string,
        return the new suffix.  Otherwise (ASR corrected earlier text) fall
        back to returning the full *current_transcript* and log a warning.
        """
        prev = self._previous_transcript
        if current_transcript.startswith(prev):
            delta = current_transcript[len(prev):]
        else:
            # ASR 纠正了之前的识别结果，回退到完整文本
            logger.warning(
                "ASR transcript not a prefix extension (prev=%r, cur=%r); "
                "sending full transcript as delta",
                prev,
                current_transcript,
            )
            delta = current_transcript
        self._previous_transcript = current_transcript
        return delta

    def reset(self) -> None:
        """Reset state after a final result, ready for the next utterance."""
        self._previous_transcript = ""
        self._item_id = None
        self._speech_started_sent = False


# ---------------------------------------------------------------------------
# send_transcribe_interim – 处理 is_final=False 的中间 ASR 结果
# ---------------------------------------------------------------------------

async def send_transcribe_interim(
    session: "RealtimeSessionState",
    transcription_result: TranscriptionResult,
    tracker: TranscriptionStreamTracker,
) -> None:
    """Send streaming delta events for an interim (non-final) ASR result."""

    # 首次收到 interim 结果时立即发送 speech_started（低延迟）
    if not tracker.speech_started_sent:
        if transcription_result.words:
            _, start_time, _ = transcription_result.words[0]
        else:
            start_time = 0.0
        vad_start_event = realtime.InputAudioBufferSpeechStartedEvent(
            audio_start_ms=int(start_time * 1000),
            type="input_audio_buffer.speech_started",
            event_id=event_id(),
            item_id=tracker.item_id,
        )
        await session.send_event(vad_start_event)
        tracker.mark_speech_started()

    # 计算增量 delta
    delta = tracker.compute_delta(transcription_result.transcript)
    if not delta:
        return

    delta_event = realtime.ConversationItemInputAudioTranscriptionDeltaEvent(
        event_id=event_id(),
        item_id=tracker.item_id,
        type="conversation.item.input_audio_transcription.delta",
        content_index=0,
        delta=delta,
    )
    await session.send_event(delta_event)
    logger.debug("Sent interim delta: item_id=%s, delta=%r", tracker.item_id, delta)


# ---------------------------------------------------------------------------
# send_transcribe_response – 处理 is_final=True 的最终 ASR 结果（重构后）
# ---------------------------------------------------------------------------

async def send_transcribe_response(
    session: "RealtimeSessionState",
    transcription_result: TranscriptionResult,
    tracker: Optional[TranscriptionStreamTracker] = None,
):
    """Complete the transcription event sequence for a final ASR result.

    When *tracker* is provided the function cooperates with prior interim
    deltas: it reuses the same ``item_id``, skips ``speech_started`` if
    already sent, and only emits the remaining delta.

    When *tracker* is ``None`` (backward-compat / non-interim mode) the
    function behaves like the original – sends the full transcript in a
    single delta event.
    """
    is_final = transcription_result.is_final
    if not is_final:
        logger.warning(
            "send_transcribe_response called with non-final result",
        )
        return

    transcript = transcription_result.transcript

    # Determine item_id – reuse from tracker if available
    if tracker is not None:
        response_item_id = tracker.item_id
    else:
        response_item_id = item_id()

    if transcription_result.words:
        _, start_time, end_time = transcription_result.words[0]
    else:
        start_time = end_time = 0.0

    # speech_started – only send if not already sent by interim handler
    if tracker is None or not tracker.speech_started_sent:
        vad_start_event = realtime.InputAudioBufferSpeechStartedEvent(
            audio_start_ms=int(start_time * 1000),
            type="input_audio_buffer.speech_started",
            event_id=event_id(),
            item_id=response_item_id,
        )
        await session.send_event(vad_start_event)

    # speech_stopped
    vad_stop_event = realtime.InputAudioBufferSpeechStoppedEvent(
        audio_end_ms=int(end_time * 1000),
        type="input_audio_buffer.speech_stopped",
        event_id=event_id(),
        item_id=response_item_id,
    )
    await session.send_event(vad_stop_event)

    # committed
    previous_item_id = session.register_server_conversation_item(response_item_id)
    committed_event = realtime.InputAudioBufferCommittedEvent(
        event_id=event_id(),
        item_id=response_item_id,
        type="input_audio_buffer.committed",
        previous_item_id=previous_item_id,
    )
    await session.send_event(committed_event)

    # Final delta – send remaining increment (or full transcript in legacy mode)
    if tracker is not None:
        delta = tracker.compute_delta(transcript)
    else:
        delta = transcript
    if delta:
        delta_event = realtime.ConversationItemInputAudioTranscriptionDeltaEvent(
            event_id=event_id(),
            item_id=response_item_id,
            type="conversation.item.input_audio_transcription.delta",
            content_index=0,
            delta=delta,
        )
        await session.send_event(delta_event)

    # completed
    completed_event = realtime.ConversationItemInputAudioTranscriptionCompletedEvent(
        content_index=0,
        event_id=event_id(),
        item_id=response_item_id,
        transcript=transcript,
        type="conversation.item.input_audio_transcription.completed",
        usage=get_usage_tokens(transcript),
    )
    await session.send_event(completed_event)

    item = realtime.RealtimeConversationItemUserMessage(
        content=[
            realtime.realtime_conversation_item_user_message.Content(type="input_audio")
        ],
        role="user",
        type="message",
        id=response_item_id,
        object=None,
        status="completed",
    )
    conversation_add_event = realtime.ConversationItemAdded(
        event_id=event_id(),
        item=item,
        type="conversation.item.added",
        previous_item_id=previous_item_id,
    )
    await session.send_event(conversation_add_event)
    conversation_done_event = realtime.ConversationItemDone(
        event_id=event_id(),
        item=item,
        type="conversation.item.done",
        previous_item_id=previous_item_id,
    )
    await session.send_event(conversation_done_event)

    logger.info("Sent transcription response: item_id=%s, is_final=%s", response_item_id, is_final)

    # Reset tracker for the next utterance
    if tracker is not None:
        tracker.reset()


@dataclass
class ToolCallInfo:
    """工具调用信息"""
    call_id: str
    name: str
    arguments: str


@dataclass
class ResponseStreamResult:
    """Responses 流式响应结果"""
    content: str = ""
    raw_content: str = ""
    tts_text: str = ""
    tool_call: Optional[ToolCallInfo] = None
    was_cancelled: bool = False  # 是否被打断
    response_id: str | None = None
    failed: bool = False
    
    @property
    def has_tool_call(self) -> bool:
        return self.tool_call is not None


def _modalities_or_default(modalities: Optional[list[str]]) -> list[str]:
    return list(modalities) if modalities else ["text"]


def _is_audio_mode(modalities: list[str]) -> bool:
    return "audio" in modalities


def _is_text_mode(modalities: list[str]) -> bool:
    return "text" in modalities


def _event_payload(event: Any) -> dict[str, Any]:
    if isinstance(event, dict):
        return event
    if hasattr(event, "model_dump"):
        return event.model_dump(exclude_none=True)
    return {}


def _response_id_from_event(payload: dict[str, Any], fallback: str | None = None) -> str | None:
    response = payload.get("response")
    if isinstance(response, dict):
        return response.get("id") or fallback
    return fallback


def _realtime_message_item(item: dict[str, Any], status: str = "in_progress") -> dict[str, Any]:
    return {
        "id": item.get("id") or item_id(),
        "object": "realtime.item",
        "type": "message",
        "role": "assistant",
        "status": status,
        "content": [],
    }


def _realtime_function_item(item: dict[str, Any], status: str = "in_progress") -> dict[str, Any]:
    return {
        "id": item.get("id") or item_id(),
        "object": "realtime.item",
        "type": "function_call",
        "call_id": item.get("call_id"),
        "name": item.get("name") or "",
        "arguments": item.get("arguments") or "",
        "status": status,
    }


def _realtime_passthrough_item(item: dict[str, Any]) -> dict[str, Any]:
    payload = dict(item)
    payload.setdefault("id", item_id())
    return payload


async def _send_item_added(
    session: "RealtimeSessionState",
    *,
    response_id_value: str,
    item: dict[str, Any],
) -> str | None:
    previous_item_id = session.register_server_conversation_item(item["id"])
    await session.send_event(
        {
            "type": "response.output_item.added",
            "event_id": event_id(),
            "response_id": response_id_value,
            "output_index": 0,
            "item": item,
        }
    )
    await session.send_event(
        {
            "type": "conversation.item.added",
            "event_id": event_id(),
            "previous_item_id": previous_item_id,
            "item": item,
        }
    )
    return previous_item_id


async def _send_item_done(
    session: "RealtimeSessionState",
    *,
    response_id_value: str,
    item: dict[str, Any],
    previous_item_id: str | None,
) -> None:
    await session.send_event(
        {
            "type": "response.output_item.done",
            "event_id": event_id(),
            "response_id": response_id_value,
            "output_index": 0,
            "item": item,
        }
    )
    await session.send_event(
        {
            "type": "conversation.item.done",
            "event_id": event_id(),
            "previous_item_id": previous_item_id,
            "item": item,
        }
    )


async def _send_mcp_status(
    session: "RealtimeSessionState",
    *,
    source_type: str,
    item_id_value: str | None,
) -> None:
    status_type = source_type.replace("response.", "", 1)
    await session.send_event(
        {
            "type": status_type,
            "event_id": event_id(),
            "item_id": item_id_value or item_id(),
        }
    )


async def process_response_stream(
    session: "RealtimeSessionState",
    response_stream: AsyncIterable[Any],
    *,
    modalities: Optional[list[str]] = None,
    tts_backend: Optional["TTSBackend"] = None,
    audio_output_format_type: str = "audio/pcm",
    audio_output_voice: str = "alloy",
    audio_output_speed: float = 1.0,
) -> ResponseStreamResult:
    """Bridge upstream Responses stream events into Realtime server events."""
    active_modalities = _modalities_or_default(modalities)
    audio_mode = _is_audio_mode(active_modalities)
    text_mode = _is_text_mode(active_modalities)

    result = ResponseStreamResult()
    text_ctx: Optional[TextResponseContext] = None
    audio_ctx: Optional[AudioResponseContext] = None
    func_ctx: Optional[FunctionCallResponseContext] = None

    response_id_value: str | None = None
    conversation_id_value = conversation_id()
    message_item_id: str | None = None
    function_item_id: str | None = None
    function_call_id: str | None = None
    function_name: str | None = None
    function_arguments = ""
    function_previous_item_id: str | None = None
    mcp_item_ids: dict[int, str] = {}
    mcp_previous_item_ids: dict[str, str | None] = {}
    text_finished = False
    audio_finished = False
    function_finished = False
    sanitizer = SanitizedModelOutputAccumulator()
    
    try:
        async for event in response_stream:
            if session.is_cancel_requested():
                logger.info("Responses stream cancelled due to new transcription")
                result.was_cancelled = True
                break

            payload = _event_payload(event)
            event_type = payload.get("type")

            if event_type == "response.created":
                response_id_value = _response_id_from_event(payload, response_id_value)
                result.response_id = response_id_value
                if response_id_value:
                    await session.send_event(
                        build_response_created_event(
                            response_id=response_id_value,
                            conversation_id=conversation_id_value,
                            event_id=event_id(),
                            modalities=active_modalities,
                        )
                    )
                continue

            if event_type == "response.in_progress":
                response_id_value = _response_id_from_event(payload, response_id_value)
                continue

            if event_type == "response.output_item.added":
                item = payload.get("item") or {}
                item_type = item.get("type")
                response_id_value = response_id_value or payload.get("response_id")
                if item_type == "message":
                    message_item_id = item.get("id") or item_id()
                    if audio_mode:
                        if tts_backend is None:
                            raise RuntimeError("TTS backend is not configured for audio output")
                        audio_ctx = AudioResponseContext(
                            session,
                            tts_backend=tts_backend,
                            modalities=active_modalities,
                            format_type=audio_output_format_type,
                            voice=audio_output_voice,
                            speed=audio_output_speed,
                            response_id_value=response_id_value,
                            item_id_value=message_item_id,
                            conversation_id_value=conversation_id_value,
                            emit_response_lifecycle=False,
                        )
                        await audio_ctx.__aenter__()
                    elif text_mode:
                        text_ctx = TextResponseContext(
                            session,
                            modalities=active_modalities,
                            response_id_value=response_id_value,
                            item_id_value=message_item_id,
                            conversation_id_value=conversation_id_value,
                            emit_response_lifecycle=False,
                        )
                        await text_ctx.__aenter__()
                elif item_type == "function_call":
                    function_item_id = item.get("id") or item_id()
                    function_call_id = item.get("call_id")
                    function_name = item.get("name") or ""
                    if response_id_value and function_call_id:
                        func_ctx = FunctionCallResponseContext(
                            session=session,
                            name=function_name,
                            call_id=function_call_id,
                            modalities=active_modalities,
                            response_id_value=response_id_value,
                            item_id_value=function_item_id,
                            conversation_id_value=conversation_id_value,
                            emit_response_lifecycle=False,
                            emit_arguments_done=False,
                        )
                        await func_ctx.__aenter__()
                elif item_type in {"mcp_list_tools", "mcp_call"} and response_id_value:
                    realtime_item = _realtime_passthrough_item(item)
                    previous = await _send_item_added(
                        session,
                        response_id_value=response_id_value,
                        item=realtime_item,
                    )
                    if item_type == "mcp_list_tools":
                        mcp_item_ids[payload.get("output_index", 0)] = realtime_item["id"]
                    mcp_previous_item_ids[realtime_item["id"]] = previous
                continue

            if event_type == "response.output_text.delta":
                delta = payload.get("delta") or ""
                result.raw_content += delta
                display_delta, tts_delta = sanitizer.push(delta)
                result.content = sanitizer.display_text
                result.tts_text = sanitizer.tts_text

                if audio_mode:
                    if not display_delta and not tts_delta:
                        continue
                    if audio_ctx is None:
                        if tts_backend is None:
                            raise RuntimeError("TTS backend is not configured for audio output")
                        message_item_id = message_item_id or payload.get("item_id") or item_id()
                        audio_ctx = AudioResponseContext(
                            session,
                            tts_backend=tts_backend,
                            modalities=active_modalities,
                            format_type=audio_output_format_type,
                            voice=audio_output_voice,
                            speed=audio_output_speed,
                            response_id_value=response_id_value,
                            item_id_value=message_item_id,
                            conversation_id_value=conversation_id_value,
                            emit_response_lifecycle=False,
                        )
                        await audio_ctx.__aenter__()
                    await audio_ctx.add_model_text_delta(display_delta, tts_delta=tts_delta)
                elif text_mode:
                    if not display_delta:
                        continue
                    if text_ctx is None:
                        message_item_id = message_item_id or payload.get("item_id") or item_id()
                        text_ctx = TextResponseContext(
                            session,
                            modalities=active_modalities,
                            response_id_value=response_id_value,
                            item_id_value=message_item_id,
                            conversation_id_value=conversation_id_value,
                            emit_response_lifecycle=False,
                        )
                        await text_ctx.__aenter__()
                    await text_ctx.send_text_delta(display_delta)

            elif event_type == "response.function_call_arguments.delta":
                delta = payload.get("delta") or ""
                function_arguments += delta
                if func_ctx:
                    await func_ctx.send_arguments_delta(delta)

            elif event_type == "response.function_call_arguments.done":
                function_arguments = payload.get("arguments") or function_arguments
                if function_call_id and response_id_value and function_item_id:
                    await session.send_event(
                        build_function_call_arguments_done(
                            arguments=function_arguments,
                            call_id=function_call_id,
                            item_id=function_item_id,
                            response_id=response_id_value,
                            name=function_name or payload.get("name") or "",
                        )
                    )

            elif event_type == "response.output_item.done":
                item = payload.get("item") or {}
                item_type = item.get("type")
                if item_type == "message":
                    if audio_ctx and not audio_finished:
                        await _finish_audio_context(
                            session=session,
                            audio_ctx=audio_ctx,
                            result=result,
                        )
                        audio_finished = True
                    if text_ctx and not text_finished:
                        await text_ctx.finish(cancelled=result.was_cancelled)
                        text_finished = True
                elif item_type == "function_call":
                    if func_ctx and not function_finished:
                        await func_ctx.__aexit__(None, None, None)
                        function_finished = True
                    function_call_id = item.get("call_id") or function_call_id
                    function_name = item.get("name") or function_name
                    function_arguments = item.get("arguments") or function_arguments
                    if function_call_id:
                        result.tool_call = ToolCallInfo(
                            call_id=function_call_id,
                            name=function_name or "",
                            arguments=function_arguments,
                        )
                elif item_type in {"mcp_list_tools", "mcp_call"} and response_id_value:
                    realtime_item = _realtime_passthrough_item(item)
                    previous = mcp_previous_item_ids.get(realtime_item["id"])
                    await _send_item_done(
                        session,
                        response_id_value=response_id_value,
                        item=realtime_item,
                        previous_item_id=previous,
                    )

            elif event_type in {
                "response.mcp_list_tools.in_progress",
                "response.mcp_list_tools.completed",
                "response.mcp_list_tools.failed",
            }:
                await _send_mcp_status(
                    session,
                    source_type=event_type,
                    item_id_value=payload.get("item_id")
                    or mcp_item_ids.get(payload.get("output_index", 0)),
                )

            elif event_type in {
                "response.mcp_call_arguments.delta",
                "response.mcp_call_arguments.done",
                "response.mcp_call.in_progress",
                "response.mcp_call.completed",
                "response.mcp_call.failed",
            }:
                realtime_event = dict(payload)
                realtime_event["event_id"] = event_id()
                realtime_event.setdefault("response_id", response_id_value)
                await session.send_event(realtime_event)

            elif event_type == "response.completed":
                response_id_value = _response_id_from_event(payload, response_id_value)
                result.response_id = response_id_value
                if hasattr(session, "mark_response_completed"):
                    session.mark_response_completed(response_id_value)
                if audio_ctx and not audio_finished:
                    await _finish_audio_context(
                        session=session,
                        audio_ctx=audio_ctx,
                        result=result,
                    )
                    audio_finished = True
                if text_ctx and not text_finished:
                    await text_ctx.finish(cancelled=False)
                    text_finished = True
                if func_ctx and not function_finished:
                    await func_ctx.__aexit__(None, None, None)
                    function_finished = True
                if response_id_value:
                    await session.send_event(
                        build_response_done_event(
                            response_id=response_id_value,
                            conversation_id=conversation_id_value,
                            event_id=event_id(),
                            modalities=active_modalities,
                        )
                    )

            elif event_type in {"response.failed", "response.incomplete", "response.error"}:
                result.failed = True
                response_id_value = _response_id_from_event(payload, response_id_value)
                response_error = {}
                response_payload = payload.get("response")
                if isinstance(response_payload, dict):
                    response_error = response_payload.get("error") or {}
                if audio_ctx and not audio_finished:
                    await audio_ctx.finish(failed=True)
                    audio_finished = True
                if text_ctx and not text_finished:
                    await text_ctx.finish(cancelled=True)
                    text_finished = True
                if response_id_value:
                    await session.send_event(
                        build_response_done_event(
                            response_id=response_id_value,
                            conversation_id=conversation_id_value,
                            event_id=event_id(),
                            modalities=active_modalities,
                            status="failed",
                            error_code=response_error.get("code"),
                            error_type="server_error",
                        )
                    )
    
    except asyncio.CancelledError:
        logger.info("Responses stream task was cancelled by CancelledError")
        result.was_cancelled = True
        if hasattr(response_stream, "aclose"):
            try:
                await response_stream.aclose()
            except Exception as e:
                logger.debug("Error closing responses stream: %s", e)
        raise
    
    finally:
        if audio_ctx is not None and not audio_finished:
            await _finish_audio_context(session=session, audio_ctx=audio_ctx, result=result)
        if text_ctx is not None and not text_finished:
            await text_ctx.finish(cancelled=result.was_cancelled)
        if func_ctx is not None and not function_finished:
            await func_ctx.__aexit__(None, None, None)

        if result.was_cancelled and response_id_value:
            await session.send_event(
                build_response_done_event(
                    response_id=response_id_value,
                    conversation_id=conversation_id_value,
                    event_id=event_id(),
                    modalities=active_modalities,
                    status="cancelled",
                    reason=session.get_cancel_reason()
                    if hasattr(session, "get_cancel_reason")
                    else "turn_detected",
                )
            )
    
    return result


async def _finish_audio_context(
    *,
    session: "RealtimeSessionState",
    audio_ctx: AudioResponseContext,
    result: ResponseStreamResult,
) -> None:
    audio_synthesis_error: Optional[Exception] = None
    should_synthesize = (
        not result.was_cancelled
        and not result.has_tool_call
        and bool(result.tts_text.strip())
    )
    if should_synthesize:
        try:
            await audio_ctx.synthesize_audio()
        except Exception as exc:  # pragma: no cover - defensive boundary
            audio_synthesis_error = exc
            logger.error("Audio synthesis failed: %s", exc)

    await audio_ctx.finish(
        cancelled=result.was_cancelled,
        failed=audio_synthesis_error is not None,
        error_code="audio_synthesis_failed" if audio_synthesis_error else None,
        error_type="server_error" if audio_synthesis_error else None,
    )
    if audio_synthesis_error and hasattr(session, "writer"):
        await session.writer.send_error(
            message=f"Audio synthesis failed: {audio_synthesis_error}",
            error_type="server_error",
            code="audio_synthesis_failed",
        )


async def send_text_response(session: "RealtimeSessionState", content: str):
    """发送纯文本响应（使用上下文管理器）"""
    async with TextResponseContext(session) as ctx:
        await ctx.send_text_delta(content)

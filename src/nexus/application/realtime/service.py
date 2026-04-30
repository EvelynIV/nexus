from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from openai.types.realtime import (
    RealtimeFunctionTool,
    SessionCreatedEvent,
    SessionUpdatedEvent,
)
from openai.types.realtime.realtime_tools_config_union import Mcp

from nexus.application.realtime.orchestrators.response_orchestrator import (
    process_response_stream,
)
from nexus.application.realtime.protocol import NullRealtimeReplySink, RealtimeReplySink
from nexus.application.realtime.text_processing import PreparedRealtimeUserTurn
from nexus.application.realtime.orchestrators.transcription_worker import (
    run_transcription_worker,
)
from nexus.application.realtime.protocol.tools import mcp_tool_payload
from nexus.application.realtime.protocol.ids import event_id
from nexus.domain.realtime import RealtimeSessionState
from nexus.infrastructure.asr import AsyncInferencer as ASRInferencer
from nexus.infrastructure.responses import AsyncInferencer as AsyncResponsesInferencer
from nexus.infrastructure.tts import TTSBackend
from nexus.sessions.responses_session import ResponsesSession

logger = logging.getLogger(__name__)


@dataclass
class EffectiveResponseConfig:
    modalities: list[str]
    audio_format_type: str
    audio_voice: str
    audio_speed: float


class RealtimeApplicationService:
    REALTIME_PCM_FORMAT = "audio/pcm"
    REALTIME_AUDIO_SAMPLE_RATE = 24000

    def __init__(
        self,
        grpc_addr: str,
        interim_results: bool = False,
        responses_base_url: Optional[str] = None,
        responses_api_key: Optional[str] = None,
        tts_backend: Optional[TTSBackend] = None,
    ):
        self.grpc_addr = grpc_addr
        self.interim_results = interim_results
        self.asr_inferencer = ASRInferencer(self.grpc_addr)
        self.responses_inferencer = (
            AsyncResponsesInferencer(api_key=responses_api_key, base_url=responses_base_url)
            if responses_api_key or responses_base_url
            else None
        )
        self.tts_backend = tts_backend

    async def close(self) -> None:
        if self.asr_inferencer:
            await self.asr_inferencer.close()
        if self.responses_inferencer:
            await self.responses_inferencer.close()
        if self.tts_backend:
            await self.tts_backend.close()

    def create_session(
        self,
        *,
        writer,
        output_modalities: Sequence[str],
        tools: Sequence[RealtimeFunctionTool],
        response_model: str,
        session_id: Optional[str] = None,
    ) -> RealtimeSessionState:
        if "transcribe" not in response_model.lower() and self.responses_inferencer is None:
            raise RuntimeError(
                "Responses inferencer is not configured. Set responses_api_key/responses_base_url for realtime models."
            )
        normalized_modalities = self._normalize_output_modalities(list(output_modalities or ["text"]))
        if "audio" in normalized_modalities and self.tts_backend is None:
            raise RuntimeError(
                "TTS backend is not configured. Configure tts_backend for realtime audio output."
            )
        responses_session = ResponsesSession(inferencer=self.responses_inferencer)
        return RealtimeSessionState(
            responses_session=responses_session,
            response_model=response_model,
            writer=writer,
            session_id=session_id or f"sess_{uuid.uuid4().hex}",
            output_modalities=normalized_modalities,
            tools=list(tools),
        )

    async def emit_session_created(
        self,
        session: RealtimeSessionState,
        model: str,
        sink=None,
    ) -> None:
        target = sink or session
        await target.send_event(
            SessionCreatedEvent(
                type="session.created",
                event_id=event_id(),
                session=self._session_payload(
                    session=session,
                    model=model,
                ),
            )
        )

    async def apply_session_update(
        self,
        session: RealtimeSessionState,
        update,
        *,
        model: str,
        reply_sink: RealtimeReplySink | None = None,
        emit_event: bool = True,
    ) -> None:
        reply_sink = reply_sink or NullRealtimeReplySink()
        if model:
            session.response_model = model

        try:
            self._validate_audio_input_update(update)
        except ValueError as exc:
            await reply_sink.send_error(
                message=str(exc),
                error_type="invalid_request_error",
                code="invalid_audio_input_format",
            )
            return

        output_modalities = self._get_update_field(update, "output_modalities")
        if output_modalities is not None:
            try:
                normalized_modalities = self._normalize_output_modalities(list(output_modalities))
            except ValueError as exc:
                await reply_sink.send_error(
                    message=str(exc),
                    error_type="invalid_request_error",
                    code="invalid_output_modalities",
                )
            else:
                if "audio" in normalized_modalities and self.tts_backend is None:
                    await reply_sink.send_error(
                        message=(
                            "TTS backend is not configured. "
                            "Configure tts_backend for realtime audio output."
                        ),
                        error_type="invalid_request_error",
                        code="audio_output_not_configured",
                    )
                else:
                    session.update_output_modalities(normalized_modalities)

        try:
            self._apply_audio_output_update(session, update)
        except ValueError as exc:
            await reply_sink.send_error(
                message=str(exc),
                error_type="invalid_request_error",
                code="invalid_audio_output_format",
            )
            return

        raw_tools = self._get_update_field(update, "tools")
        if raw_tools is not None:
            function_tools, mcp_tools = self._split_tools(raw_tools)
            session.tools = function_tools
            session.mcp_tools = mcp_tools

        if emit_event:
            await session.send_event(
                SessionUpdatedEvent(
                    type="session.updated",
                    event_id=event_id(),
                    session=self._session_payload(session=session, model=model),
                )
            )

    async def start_transcription_worker(
        self,
        session: RealtimeSessionState,
        auto_response_enabled: bool,
    ) -> asyncio.Task:
        return asyncio.create_task(
            run_transcription_worker(
                inferencer=self.asr_inferencer,
                session=session,
                interim_results=self.interim_results,
                auto_response_enabled=auto_response_enabled,
                response_worker=self.response_worker,
            )
        )

    async def response_worker(self, session: RealtimeSessionState, user_turn: PreparedRealtimeUserTurn) -> None:
        response_stream = await session.respond_to_user(user_turn)
        response_cfg = self._resolve_response_config(session)
        if "audio" in response_cfg.modalities:
            if self.tts_backend is None:
                raise RuntimeError(
                    "TTS backend is not configured. Configure tts_backend for realtime audio output."
                )
            self._ensure_audio_output_supported(response_cfg.audio_format_type)
        result = await process_response_stream(
            session=session,
            response_stream=response_stream,
            modalities=response_cfg.modalities,
            tts_backend=self.tts_backend,
            audio_output_format_type=response_cfg.audio_format_type,
            audio_output_voice=response_cfg.audio_voice,
            audio_output_speed=response_cfg.audio_speed,
        )

        if result.has_tool_call and result.tool_call:
            logger.info(
                "Function call sent: %s; waiting for function_call_output + response.create",
                result.tool_call.name,
            )

    async def generate_response(
        self,
        session: RealtimeSessionState,
        event=None,
        *,
        reply_sink: RealtimeReplySink | None = None,
    ) -> None:
        reply_sink = reply_sink or NullRealtimeReplySink()
        try:
            response_cfg = self._resolve_response_config(
                session=session,
                response=getattr(event, "response", None),
            )
        except ValueError as exc:
            await reply_sink.send_error(
                message=str(exc),
                error_type="invalid_request_error",
                code="invalid_output_modalities",
            )
            return

        if "audio" in response_cfg.modalities:
            if self.tts_backend is None:
                await reply_sink.send_error(
                    message=(
                        "TTS backend is not configured. "
                        "Configure tts_backend for realtime audio output."
                    ),
                    error_type="invalid_request_error",
                    code="audio_output_not_configured",
                )
                return
            try:
                self._ensure_audio_output_supported(response_cfg.audio_format_type)
            except ValueError as exc:
                await reply_sink.send_error(
                    message=str(exc),
                    error_type="invalid_request_error",
                    code="unsupported_audio_output_format",
                )
                return

        response_stream = await session.continue_conversation()
        result = await process_response_stream(
            session=session,
            response_stream=response_stream,
            modalities=response_cfg.modalities,
            tts_backend=self.tts_backend,
            audio_output_format_type=response_cfg.audio_format_type,
            audio_output_voice=response_cfg.audio_voice,
            audio_output_speed=response_cfg.audio_speed,
        )

        if result.has_tool_call and result.tool_call:
            logger.info(
                "Function call sent: %s; waiting for function_call_output + response.create",
                result.tool_call.name,
            )

    async def handle_response_create(
        self,
        session: RealtimeSessionState,
        event,
        *,
        reply_sink: RealtimeReplySink | None = None,
    ) -> None:
        asyncio.create_task(self.generate_response(session, event, reply_sink=reply_sink))

    async def handle_response_cancel(
        self,
        session: RealtimeSessionState,
        _event,
        *,
        reply_sink: RealtimeReplySink | None = None,
    ) -> None:
        del reply_sink
        session.request_cancel(reason="client_cancelled")
        task = session.get_current_response_task()
        if task and not task.done():
            task.cancel()

    async def handle_input_audio_commit(self, session: RealtimeSessionState, _event) -> None:
        # Current backend emits transcription from continuous stream; commit acts as a no-op marker.
        logger.debug("input_audio_buffer.commit received for session %s", session.session_id)

    async def close_session(self, session: RealtimeSessionState) -> None:
        del session

    def _normalize_output_modalities(self, modalities: Sequence[str]) -> list[str]:
        if not modalities:
            return ["text"]

        normalized = {str(modality).strip().lower() for modality in modalities if modality}
        if not normalized:
            return ["text"]

        unsupported = normalized - {"audio", "text"}
        if unsupported:
            raise ValueError(
                f"Unsupported output modalities: {sorted(unsupported)}. Allowed values: ['audio', 'text']"
            )

        ordered: list[str] = []
        if "audio" in normalized:
            ordered.append("audio")
        if "text" in normalized:
            ordered.append("text")
        return ordered

    def _ensure_audio_output_supported(self, format_type: str) -> None:
        if format_type != self.REALTIME_PCM_FORMAT:
            raise ValueError(
                f"Unsupported realtime audio output format '{format_type}'. "
                f"Only '{self.REALTIME_PCM_FORMAT}' is currently supported."
            )

    def _extract_format_type(self, format_config) -> Optional[str]:
        if format_config is None:
            return None
        if isinstance(format_config, str):
            return format_config
        if isinstance(format_config, dict):
            return format_config.get("type")
        return getattr(format_config, "type", None)

    def _extract_format_rate(self, format_config) -> Optional[int]:
        if format_config is None:
            return None
        if isinstance(format_config, dict):
            rate = format_config.get("rate")
        else:
            rate = getattr(format_config, "rate", None)
        if rate is None:
            return None
        return int(rate)

    def _model_to_dict(self, value) -> dict:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if hasattr(value, "model_dump"):
            return value.model_dump(exclude_none=True)
        return {}

    def _validate_audio_input_update(self, update) -> None:
        audio_config = self._get_update_field(update, "audio")
        if audio_config is None:
            return

        input_config = (
            audio_config.get("input")
            if isinstance(audio_config, dict)
            else getattr(audio_config, "input", None)
        )
        if input_config is None:
            return

        input_data = self._model_to_dict(input_config)
        format_config = input_data.get("format")
        if format_config is None:
            return

        format_type = self._extract_format_type(format_config)
        format_rate = self._extract_format_rate(format_config)

        if format_type is not None and format_type != self.REALTIME_PCM_FORMAT:
            raise ValueError(
                f"Unsupported realtime audio input format '{format_type}'. "
                f"Only '{self.REALTIME_PCM_FORMAT}' is currently supported."
            )

        if format_rate is not None and format_rate != self.REALTIME_AUDIO_SAMPLE_RATE:
            raise ValueError(
                "Unsupported realtime audio input sample rate "
                f"'{format_rate}'. Only '{self.REALTIME_AUDIO_SAMPLE_RATE}' is supported."
            )

    def _apply_audio_output_update(self, session: RealtimeSessionState, update) -> None:
        audio_config = self._get_update_field(update, "audio")
        if audio_config is None:
            return

        output_config = (
            audio_config.get("output")
            if isinstance(audio_config, dict)
            else getattr(audio_config, "output", None)
        )
        if output_config is None:
            return

        output_data = (
            output_config if isinstance(output_config, dict) else output_config.model_dump(exclude_none=True)
        )

        format_type = self._extract_format_type(output_data.get("format"))
        voice = output_data.get("voice")
        speed = output_data.get("speed")

        if format_type is not None:
            self._ensure_audio_output_supported(format_type)
        if voice is not None and session.is_audio_voice_locked() and voice != session.audio_output_voice:
            raise ValueError(
                "Audio output voice cannot be changed after the session has emitted audio."
            )

        session.update_audio_output_config(
            format_type=format_type,
            voice=voice,
            speed=speed,
        )

    @staticmethod
    def _get_update_field(update, field: str):
        if isinstance(update, dict):
            return update.get(field)
        return getattr(update, field, None)

    def _resolve_response_config(self, session: RealtimeSessionState, response=None) -> EffectiveResponseConfig:
        modalities = session.get_output_modalities()
        session_audio_cfg = session.get_audio_output_config()
        audio_format_type = session_audio_cfg["format_type"]
        audio_voice = session_audio_cfg["voice"]
        audio_speed = session_audio_cfg["speed"]

        if response is not None:
            response_modalities = (
                response.get("output_modalities")
                if isinstance(response, dict)
                else getattr(response, "output_modalities", None)
            )
            if response_modalities is not None:
                modalities = self._normalize_output_modalities(list(response_modalities))

            response_audio = (
                response.get("audio")
                if isinstance(response, dict)
                else getattr(response, "audio", None)
            )
            if response_audio is not None:
                output_cfg = (
                    response_audio.get("output")
                    if isinstance(response_audio, dict)
                    else getattr(response_audio, "output", None)
                )
                if output_cfg is not None:
                    output_data = (
                        output_cfg if isinstance(output_cfg, dict) else output_cfg.model_dump(exclude_none=True)
                    )
                    format_type = self._extract_format_type(output_data.get("format"))
                    if format_type:
                        audio_format_type = format_type
                    if output_data.get("voice"):
                        if (
                            session.is_audio_voice_locked()
                            and output_data["voice"] != session_audio_cfg["voice"]
                        ):
                            raise ValueError(
                                "Audio output voice cannot be changed after the session has emitted audio."
                            )
                        audio_voice = output_data["voice"]
                    if output_data.get("speed") is not None:
                        audio_speed = output_data["speed"]

        return EffectiveResponseConfig(
            modalities=modalities,
            audio_format_type=audio_format_type,
            audio_voice=audio_voice,
            audio_speed=audio_speed,
        )

    def _split_tools(
        self,
        raw_tools: Iterable[RealtimeFunctionTool | Mcp],
    ) -> Tuple[List[RealtimeFunctionTool], list[dict]]:
        function_tools: List[RealtimeFunctionTool] = []
        mcp_tools: list[dict] = []

        for tool in raw_tools:
            if isinstance(tool, RealtimeFunctionTool):
                function_tools.append(tool)
                continue

            if isinstance(tool, Mcp):
                mcp_tools.append(mcp_tool_payload(tool))
                continue

            payload = tool.model_dump(exclude_none=True) if hasattr(tool, "model_dump") else tool
            if isinstance(payload, dict) and payload.get("type") == "mcp":
                mcp_tools.append(mcp_tool_payload(payload))
            elif isinstance(payload, dict):
                function_tools.append(RealtimeFunctionTool(**payload))

        return function_tools, mcp_tools

    def _session_payload(
        self,
        *,
        session: RealtimeSessionState,
        model: str,
    ) -> dict:
        input_cfg = session.get_audio_input_config()
        audio_cfg = session.get_audio_output_config()
        input_format = {"type": input_cfg["format_type"]}
        if input_cfg["format_type"] == self.REALTIME_PCM_FORMAT:
            input_format["rate"] = input_cfg["sample_rate"]

        output_format = {"type": audio_cfg["format_type"]}
        if audio_cfg["format_type"] == self.REALTIME_PCM_FORMAT:
            output_format["rate"] = self.REALTIME_AUDIO_SAMPLE_RATE

        return {
            "id": session.session_id,
            "type": "realtime",
            "model": model,
            "output_modalities": session.get_output_modalities(),
            "audio": {
                "input": {
                    "format": input_format,
                },
                "output": {
                    "format": output_format,
                    "voice": audio_cfg["voice"],
                    "speed": audio_cfg["speed"],
                }
            },
            "tools": [
                *[tool.model_dump(exclude_none=True) for tool in session.tools],
                *session.mcp_tools,
            ],
        }

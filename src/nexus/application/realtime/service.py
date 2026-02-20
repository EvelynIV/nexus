from __future__ import annotations

import asyncio
import logging
from typing import Iterable, List, Optional, Sequence, Tuple

from openai.types.realtime import (
    RealtimeFunctionTool,
    SessionCreatedEvent,
    SessionUpdatedEvent,
)
from openai.types.realtime.realtime_tools_config_union import Mcp

from nexus.application.realtime.emitters.response_contexts import McpListToolsContext
from nexus.application.realtime.orchestrators.response_orchestrator import (
    process_chat_stream,
    send_tool_result_response,
)
from nexus.application.realtime.orchestrators.tool_call_orchestrator import (
    execute_mcp_tool_call,
)
from nexus.application.realtime.orchestrators.transcription_worker import (
    run_transcription_worker,
)
from nexus.application.realtime.protocol.ids import event_id
from nexus.domain.realtime import RealtimeSessionState
from nexus.infrastructure.asr import AsyncInferencer as ASRInferencer
from nexus.infrastructure.chat import AsyncInferencer as AsyncChatInferencer
from nexus.infrastructure.tts import Inferencer as TTSInferencer
from nexus.infrastructure.mcp import McpServerConfig
from nexus.sessions.chat_session import AsyncChatSession

logger = logging.getLogger(__name__)


class RealtimeApplicationService:
    def __init__(
        self,
        grpc_addr: str,
        interim_results: bool = False,
        chat_base_url: Optional[str] = None,
        chat_api_key: Optional[str] = None,
        tts_base_url: Optional[str] = None,
        tts_api_key: Optional[str] = None,
    ):
        self.grpc_addr = grpc_addr
        self.interim_results = interim_results
        self.asr_inferencer = ASRInferencer(self.grpc_addr)
        self.chat_inferencer = (
            AsyncChatInferencer(api_key=chat_api_key, base_url=chat_base_url)
            if chat_api_key
            else None
        )
        self.tts_inferencer = (
            TTSInferencer(base_url=tts_base_url, api_key=tts_api_key)
            if tts_api_key
            else None
        )

    async def close(self) -> None:
        if self.asr_inferencer:
            await self.asr_inferencer.close()
        if self.chat_inferencer:
            await self.chat_inferencer.close()

    def create_session(
        self,
        *,
        writer,
        output_modalities: Sequence[str],
        tools: Sequence[RealtimeFunctionTool],
        chat_model: str,
    ) -> RealtimeSessionState:
        if "transcribe" not in chat_model.lower() and self.chat_inferencer is None:
            raise RuntimeError(
                "Chat inferencer is not configured. Set chat_api_key/chat_base_url for realtime chat models."
            )
        chat_session = AsyncChatSession(chat_inferencer=self.chat_inferencer)
        return RealtimeSessionState(
            chat_session=chat_session,
            chat_model=chat_model,
            writer=writer,
            output_modalities=list(output_modalities),
            tools=list(tools),
        )

    async def emit_session_created(self, session: RealtimeSessionState, model: str) -> None:
        await session.send_event(
            SessionCreatedEvent(
                type="session.created",
                event_id=event_id(),
                session=self._session_payload(
                    session=session,
                    model=model,
                ),
            )
        )

    async def apply_session_update(self, session: RealtimeSessionState, update, *, model: str) -> None:
        if model:
            session.chat_model = model

        output_modalities = getattr(update, "output_modalities", None)
        if output_modalities:
            session.update_output_modalities(list(output_modalities))

        raw_tools = getattr(update, "tools", None)
        if raw_tools is not None:
            function_tools, mcp_configs = self._split_tools(raw_tools)
            session.tools = function_tools
            await self._sync_mcp_servers(session, mcp_configs)

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
        is_chat_model: bool,
    ) -> asyncio.Task:
        return asyncio.create_task(
            run_transcription_worker(
                inferencer=self.asr_inferencer,
                session=session,
                interim_results=self.interim_results,
                is_chat_model=is_chat_model,
                chat_worker=self.chat_worker,
            )
        )

    async def chat_worker(self, session: RealtimeSessionState, user_message: str) -> None:
        chat_stream = session.chat(user_message)
        result = await process_chat_stream(session, chat_stream)

        if result.has_mcp_call and result.tool_call:
            await execute_mcp_tool_call(session=session, tool_call=result.tool_call)
        elif result.has_tool_call and result.tool_call:
            logger.info(
                "Function call sent: %s; waiting for function_call_output + response.create",
                result.tool_call.name,
            )

    async def generate_response(self, session: RealtimeSessionState) -> None:
        chat_stream = session.continue_conversation()
        await send_tool_result_response(session, chat_stream)

    async def handle_response_create(self, session: RealtimeSessionState, _event) -> None:
        asyncio.create_task(self.generate_response(session))

    async def handle_response_cancel(self, session: RealtimeSessionState, _event) -> None:
        session.request_cancel(reason="client_cancelled")
        task = session.get_current_chat_task()
        if task and not task.done():
            task.cancel()

    async def handle_input_audio_commit(self, session: RealtimeSessionState, _event) -> None:
        # Current backend emits transcription from continuous stream; commit acts as a no-op marker.
        logger.debug("input_audio_buffer.commit received for session %s", session.session_id)

    async def close_session(self, session: RealtimeSessionState) -> None:
        await session.mcp_registry.close()

    def _split_tools(
        self,
        raw_tools: Iterable[RealtimeFunctionTool | Mcp],
    ) -> Tuple[List[RealtimeFunctionTool], List[McpServerConfig]]:
        function_tools: List[RealtimeFunctionTool] = []
        mcp_configs: List[McpServerConfig] = []

        for tool in raw_tools:
            if isinstance(tool, RealtimeFunctionTool):
                function_tools.append(tool)
                continue

            if isinstance(tool, Mcp):
                mcp_configs.append(McpServerConfig.from_dict(tool.model_dump(exclude_none=True)))
                continue

            payload = tool.model_dump(exclude_none=True) if hasattr(tool, "model_dump") else tool
            if isinstance(payload, dict) and payload.get("type") == "mcp":
                mcp_configs.append(McpServerConfig.from_dict(payload))
            elif isinstance(payload, dict):
                function_tools.append(RealtimeFunctionTool(**payload))

        return function_tools, mcp_configs

    async def _sync_mcp_servers(
        self,
        session: RealtimeSessionState,
        configs: Sequence[McpServerConfig],
    ) -> None:
        target_labels = {config.server_label for config in configs}
        current_labels = set(session.mcp_registry.server_labels)

        for stale_label in current_labels - target_labels:
            await session.mcp_registry.unregister_server(stale_label)

        for config in configs:
            try:
                await self._register_mcp_server(session, config)
            except Exception as exc:
                logger.error(
                    "Failed to register MCP server %s: %s",
                    config.server_label,
                    exc,
                )
                await session.writer.send_error(
                    message=f"Failed to connect MCP server '{config.server_label}': {exc}",
                    error_type="server_error",
                    code="mcp_connection_error",
                )

    async def _register_mcp_server(
        self,
        session: RealtimeSessionState,
        config: McpServerConfig,
    ) -> None:
        async with McpListToolsContext(session, config.server_label) as ctx:
            tools = await session.mcp_registry.register_server(config)

            tools_data = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                    "annotations": tool.annotations,
                }
                for tool in tools
            ]
            ctx.set_tools(tools_data)

    def _session_payload(
        self,
        *,
        session: RealtimeSessionState,
        model: str,
    ) -> dict:
        return {
            "id": session.session_id,
            "type": "realtime",
            "model": model,
            "output_modalities": session.get_output_modalities(),
            "tools": [tool.model_dump(exclude_none=True) for tool in session.get_all_tools()],
        }

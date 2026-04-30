from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator, AsyncIterable
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING, Dict, List, Optional

import numpy as np
from openai.types.realtime.realtime_function_tool import RealtimeFunctionTool

from nexus.application.realtime.protocol.tools import to_response_tools
from nexus.application.realtime.text_processing import PreparedRealtimeUserTurn
from nexus.sessions.responses_session import ResponsesSession

if TYPE_CHECKING:
    from asyncio import Task
    from nexus.application.realtime.protocol import RealtimeEventSink

logger = logging.getLogger(__name__)


@dataclass
class RealtimeSessionState:
    """Domain session state for a realtime websocket connection."""

    responses_session: ResponsesSession
    response_model: str
    writer: "RealtimeEventSink"

    session_id: str = field(default_factory=lambda: f"sess_{uuid.uuid4().hex}")
    tools: List[RealtimeFunctionTool] = field(default_factory=list)
    mcp_tools: list[dict[str, Any]] = field(default_factory=list)

    audio_input_format_type: str = "audio/pcm"
    audio_input_sample_rate: int = 24000
    asr_sample_rate: int = 16000
    output_modalities: list[str] = field(default_factory=lambda: ["text"])
    audio_output_format_type: str = "audio/pcm"
    audio_output_voice: str = "alloy"
    audio_output_speed: float = 1.0
    audio_queue: asyncio.Queue[np.ndarray] = field(default_factory=asyncio.Queue)
    audio_output_queue: asyncio.Queue[bytes | None] = field(default_factory=asyncio.Queue)
    _conversation_tail_item_id: Optional[str] = field(default=None, repr=False)
    _conversation_previous_item_ids: Dict[str, Optional[str]] = field(
        default_factory=dict,
        repr=False,
    )
    _audio_voice_locked: bool = field(default=False, repr=False)

    _current_response_task: Optional["Task"] = field(default=None, repr=False)
    _cancel_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    _cancel_reason: str = field(default="turn_detected", repr=False)

    async def send_event(self, event) -> None:
        await self.writer.send_event(event)

    def request_cancel(self, reason: str = "turn_detected") -> None:
        self._cancel_event.set()
        self._cancel_reason = reason

    def reset_cancel(self) -> None:
        self._cancel_event.clear()
        self._cancel_reason = "turn_detected"

    def is_cancel_requested(self) -> bool:
        return self._cancel_event.is_set()

    def get_cancel_reason(self) -> str:
        return self._cancel_reason

    def set_current_response_task(self, task: Optional["Task"]) -> None:
        self._current_response_task = task

    def get_current_response_task(self) -> Optional["Task"]:
        return self._current_response_task

    async def audio_iter(self) -> AsyncGenerator[np.ndarray, None]:
        while True:
            chunk = await self.audio_queue.get()
            if chunk is None:
                break
            yield chunk

    def update_output_modalities(self, modalities: List[str]) -> None:
        self.output_modalities = modalities

    def get_output_modalities(self) -> List[str]:
        return self.output_modalities.copy()

    async def push_audio_output(self, pcm_bytes: bytes) -> None:
        if pcm_bytes:
            await self.audio_output_queue.put(bytes(pcm_bytes))

    async def close_audio_output(self) -> None:
        await self.audio_output_queue.put(None)

    def update_audio_output_config(
        self,
        *,
        format_type: Optional[str] = None,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> None:
        if format_type is not None:
            self.audio_output_format_type = format_type
        if voice is not None:
            self.audio_output_voice = voice
        if speed is not None:
            self.audio_output_speed = speed

    def get_audio_output_config(self) -> dict:
        return {
            "format_type": self.audio_output_format_type,
            "voice": self.audio_output_voice,
            "speed": self.audio_output_speed,
        }

    def get_audio_input_config(self) -> dict:
        return {
            "format_type": self.audio_input_format_type,
            "sample_rate": self.audio_input_sample_rate,
        }

    def lock_audio_voice(self) -> None:
        self._audio_voice_locked = True

    def is_audio_voice_locked(self) -> bool:
        return self._audio_voice_locked

    def register_server_conversation_item(self, item_id: str) -> Optional[str]:
        """Register a server-generated item in the conversation order chain."""
        if item_id in self._conversation_previous_item_ids:
            return self._conversation_previous_item_ids[item_id]

        previous_item_id = self._conversation_tail_item_id
        self._conversation_previous_item_ids[item_id] = previous_item_id
        self._conversation_tail_item_id = item_id
        return previous_item_id

    def get_registered_previous_item_id(self, item_id: str) -> Optional[str]:
        """Return the registered predecessor for a conversation item."""
        return self._conversation_previous_item_ids.get(item_id)

    def get_response_tools(self) -> list[dict[str, Any]]:
        return to_response_tools(self.tools, self.mcp_tools)

    async def respond_to_user(self, user_turn: PreparedRealtimeUserTurn) -> AsyncIterable[Any]:
        self.responses_session.add_user_message(user_turn.model_text)
        return await self.responses_session.create_response(
            model=self.response_model,
            stream=True,
            tools=self.get_response_tools(),
        )

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        self.responses_session.add_function_call_output(tool_call_id, content)
        logger.info("Function call output queued: call_id=%s", tool_call_id)

    async def continue_conversation(self) -> AsyncIterable[Any]:
        return await self.responses_session.create_response(
            model=self.response_model,
            stream=True,
            tools=self.get_response_tools(),
        )

    def mark_response_completed(self, response_id: str | None) -> None:
        self.responses_session.mark_response_completed(response_id)

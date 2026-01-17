import logging
import queue
import uuid
from collections.abc import Generator
from dataclasses import dataclass, field
from threading import Event, Lock
from typing import List, Optional, Tuple

import numpy as np
from openai.types.chat import ChatCompletionFunctionTool
from openai.types.realtime.realtime_function_tool import RealtimeFunctionTool
from openai.types.shared import FunctionDefinition

from nexus.servicers.realtime.build_events import (
    build_function_call_arguments_delta,
    build_function_call_arguments_done,
    build_item_function_call,
)
from nexus.sessions.chat_session import ChatSession

logger = logging.getLogger(__name__)


@dataclass
class RealtimeSession:
    """实时会话状态"""

    chat_session: ChatSession
    chat_model: str
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tools: List[RealtimeFunctionTool] = field(default_factory=list)

    sample_rate: int = 16000
    output_modalities: list[str] = field(default_factory=lambda: ["text"])
    output_modalities_lock: Lock = field(default_factory=Lock)

    audio_queue: queue.Queue[np.ndarray] = field(default_factory=queue.Queue)
    result_queue: queue.Queue = field(default_factory=queue.Queue)
    tts_audio_queue: queue.Queue[Tuple[bytes, bool]] = field(
        default_factory=queue.Queue
    )

    def update_output_modalities(self, modalities: List[str]):
        with self.output_modalities_lock:
            self.output_modalities = modalities

    def get_output_modalities(self) -> List[str]:
        with self.output_modalities_lock:
            return self.output_modalities.copy()

    def chat(
        self,
        user_message: str,
    ) -> Generator[str, None, None]:
        chat_stream_resp = self.chat_session.chat(
            user_message=user_message,
            model=self.chat_model,
            stream=True,
            tools=convert_to_chat_tools(self.tools),
        )
        tool_arguments = ""
        call_id = None
        for chunk in chat_stream_resp:
            delta = chunk.choices[0].delta
            if delta.tool_calls:
                function = delta.tool_calls[0].function
                if function.name:
                    call_id = delta.tool_calls[0].id
                    yield build_item_function_call(
                        name=function.name,
                        arguments=function.arguments,
                        call_id=call_id,
                    )
                if function.arguments:
                    tool_arguments += function.arguments
                    yield build_function_call_arguments_delta(
                        function.arguments, call_id
                    )
            if delta and delta.content:
                yield delta.content
        if tool_arguments:
            logger.info(f"Function call arguments done: {tool_arguments}")
            yield build_function_call_arguments_done(tool_arguments, call_id)

    def use_tool(self, tool_call_id: str, content: str):
        tool_stream_resp = self.chat_session.use_tool(
            tool_call_id=tool_call_id,
            content=content,
            model=self.chat_model,
            stream=True,
            tools=convert_to_chat_tools(self.tools),
        )
        for chunk in tool_stream_resp:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content


def convert_to_chat_tools(
    tools: List[RealtimeFunctionTool],
) -> List[ChatCompletionFunctionTool]:
    """将 RealtimeFunctionTool 列表转换为适用于聊天的工具列表"""
    chat_tools = []
    for tool in tools:
        chat_tool = ChatCompletionFunctionTool(
            type="function",
            function=FunctionDefinition(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
            ),
        )
        chat_tools.append(chat_tool)
    return chat_tools

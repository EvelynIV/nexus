"""Tool conversion helpers for realtime orchestration."""

from __future__ import annotations

from typing import Iterable, List

from openai.types.chat import ChatCompletionFunctionTool
from openai.types.realtime.realtime_function_tool import RealtimeFunctionTool
from openai.types.shared import FunctionDefinition


def to_chat_tools(tools: Iterable[RealtimeFunctionTool]) -> List[ChatCompletionFunctionTool]:
    """Convert realtime function tools to chat-completions tool format."""
    return [
        ChatCompletionFunctionTool(
            type="function",
            function=FunctionDefinition(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
            ),
        )
        for tool in tools
    ]

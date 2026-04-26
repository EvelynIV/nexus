from __future__ import annotations

import json
import logging

from nexus.application.realtime.orchestrators.response_orchestrator import ToolCallInfo
from nexus.domain.realtime import RealtimeSessionState

logger = logging.getLogger(__name__)


def _last_assistant_reasoning_content(session: RealtimeSessionState) -> str:
    for message in reversed(session.chat_session.chat_history):
        if isinstance(message, dict):
            if message.get("role") == "assistant":
                return message.get("reasoning_content") or ""
            continue
        if getattr(message, "role", None) == "assistant":
            return getattr(message, "reasoning_content", None) or ""
    return ""


def _assistant_reasoning_kwargs(session: RealtimeSessionState) -> dict[str, str]:
    reasoning_content = _last_assistant_reasoning_content(session)
    return {"reasoning_content": reasoning_content} if reasoning_content else {}


async def execute_mcp_tool_call(
    *,
    session: RealtimeSessionState,
    tool_call: ToolCallInfo,
) -> None:
    """Execute MCP tool calls server-side and persist results to chat history."""
    tool_name = tool_call.name
    arguments_str = tool_call.arguments
    mcp_ctx = tool_call.mcp_ctx

    if not mcp_ctx:
        logger.error("MCP context not found for tool call: %s", tool_name)
        return

    try:
        arguments = json.loads(arguments_str) if arguments_str else {}
    except json.JSONDecodeError:
        arguments = {}

    try:
        output = await session.mcp_registry.call_tool(tool_name, arguments)
        mcp_ctx.set_output(output)

        assistant_msg = {
            "role": "assistant",
            "content": None,
            **_assistant_reasoning_kwargs(session),
            "tool_calls": [
                {
                    "id": mcp_ctx.item_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": arguments_str,
                    },
                }
            ],
        }
        session.chat_session.chat_history.append(assistant_msg)

        tool_msg = {
            "role": "tool",
            "tool_call_id": mcp_ctx.item_id,
            "content": output,
        }
        session.chat_session.chat_history.append(tool_msg)
    except Exception as exc:
        logger.error("MCP call %s failed: %s", tool_name, exc)
        mcp_ctx.set_error(str(exc))

        assistant_msg = {
            "role": "assistant",
            "content": None,
            **_assistant_reasoning_kwargs(session),
            "tool_calls": [
                {
                    "id": mcp_ctx.item_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": arguments_str,
                    },
                }
            ],
        }
        session.chat_session.chat_history.append(assistant_msg)

        tool_msg = {
            "role": "tool",
            "tool_call_id": mcp_ctx.item_id,
            "content": f"Error: {exc}",
        }
        session.chat_session.chat_history.append(tool_msg)

    # Explicit close triggers final MCP call events.
    await mcp_ctx.__aexit__(None, None, None)

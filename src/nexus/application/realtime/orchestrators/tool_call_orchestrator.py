from __future__ import annotations

import json
import logging

from nexus.application.realtime.orchestrators.response_orchestrator import ToolCallInfo
from nexus.domain.realtime import RealtimeSessionState

logger = logging.getLogger(__name__)


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

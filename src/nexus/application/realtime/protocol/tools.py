"""Tool conversion helpers for Responses-backed realtime orchestration."""

from __future__ import annotations

from typing import Any, Iterable

from openai.types.realtime.realtime_function_tool import RealtimeFunctionTool


def to_response_tools(
    function_tools: Iterable[RealtimeFunctionTool],
    mcp_tools: Iterable[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Convert realtime tools to Responses API tool format."""
    response_tools = [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "strict": False,
        }
        for tool in function_tools
    ]
    response_tools.extend(dict(tool) for tool in (mcp_tools or []))
    return response_tools


def mcp_tool_payload(tool: Any) -> dict[str, Any]:
    payload = tool.model_dump(exclude_none=True) if hasattr(tool, "model_dump") else dict(tool)
    return {
        key: value
        for key, value in payload.items()
        if key
        in {
            "type",
            "server_label",
            "server_url",
            "connector_id",
            "authorization",
            "headers",
            "allowed_tools",
            "require_approval",
            "server_description",
        }
    }

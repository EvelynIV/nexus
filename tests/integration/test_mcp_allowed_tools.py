from __future__ import annotations

from nexus.mcp.models import McpServerConfig


def test_mcp_allowed_tools_supports_filter_object() -> None:
    config = McpServerConfig.from_dict(
        {
            "server_label": "demo",
            "server_url": "https://example.com/mcp",
            "allowed_tools": {
                "tool_names": ["read_doc"],
                "read_only": True,
            },
        }
    )

    assert config.allows_tool(
        tool_name="read_doc",
        annotations={"readOnlyHint": True},
    )
    assert not config.allows_tool(
        tool_name="write_doc",
        annotations={"readOnlyHint": True},
    )
    assert not config.allows_tool(
        tool_name="read_doc",
        annotations={"readOnlyHint": False},
    )

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nexus.sessions.chat_session import AsyncChatSession, ChatSession


def _chunk(
    *,
    content: str | None = None,
    reasoning_content: str | None = None,
    tool_name: str | None = None,
    tool_arguments: str | None = None,
    tool_call_id: str | None = None,
    finish_reason: str | None = None,
):
    tool_calls = None
    if tool_name is not None or tool_arguments is not None or tool_call_id is not None:
        tool_calls = [
            SimpleNamespace(
                id=tool_call_id,
                function=SimpleNamespace(
                    name=tool_name,
                    arguments=tool_arguments,
                ),
            )
        ]

    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=content,
                    reasoning_content=reasoning_content,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,
            )
        ]
    )


def test_chat_session_preserves_streaming_reasoning_content_for_tool_call() -> None:
    session = ChatSession(chat_inferencer=SimpleNamespace())

    chunks = [
        _chunk(reasoning_content="需要查一下"),
        _chunk(
            tool_call_id="call_123",
            tool_name="get_project_status",
            tool_arguments='{"project"',
        ),
        _chunk(tool_arguments=':"nexus"}', finish_reason="tool_calls"),
    ]

    list(session.get_result_record_itr(chunks))

    assistant_message = session.chat_history[-1]
    assert assistant_message["reasoning_content"] == "需要查一下"
    assert assistant_message["tool_calls"][0]["id"] == "call_123"
    assert assistant_message["tool_calls"][0]["function"]["name"] == "get_project_status"
    assert assistant_message["tool_calls"][0]["function"]["arguments"] == '{"project":"nexus"}'


@pytest.mark.asyncio
async def test_async_chat_session_preserves_streaming_reasoning_content_for_tool_call() -> None:
    session = AsyncChatSession(chat_inferencer=SimpleNamespace())

    async def stream():
        yield _chunk(reasoning_content="先检查状态")
        yield _chunk(
            tool_call_id="call_456",
            tool_name="get_project_status",
            tool_arguments='{"project":"nexus"}',
            finish_reason="tool_calls",
        )

    chunks = []
    async for chunk in session.get_result_record_itr(stream()):
        chunks.append(chunk)

    assert len(chunks) == 2
    assistant_message = session.chat_history[-1]
    assert assistant_message["reasoning_content"] == "先检查状态"
    assert assistant_message["tool_calls"][0]["id"] == "call_456"


def test_replace_last_assistant_message_content_keeps_reasoning_content() -> None:
    session = ChatSession(chat_inferencer=SimpleNamespace())

    list(
        session.get_result_record_itr(
            [
                _chunk(content="原始", reasoning_content="思考内容", finish_reason="stop"),
            ]
        )
    )

    session.replace_last_assistant_message_content("展示内容")

    assistant_message = session.chat_history[-1]
    assert assistant_message["content"] == "展示内容"
    assert assistant_message["reasoning_content"] == "思考内容"
    assert "tool_calls" not in assistant_message


def test_chat_session_omits_empty_tool_calls_for_plain_assistant_message() -> None:
    session = ChatSession(chat_inferencer=SimpleNamespace())

    list(
        session.get_result_record_itr(
            [
                _chunk(content="普通回复", finish_reason="stop"),
            ]
        )
    )

    assistant_message = session.chat_history[-1]
    assert assistant_message == {
        "role": "assistant",
        "content": "普通回复",
    }

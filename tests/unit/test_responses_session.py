from __future__ import annotations

from typing import Any

import pytest

from nexus.sessions.responses_session import ResponsesSession


class FakeInferencer:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)

        async def _stream():
            if False:
                yield None

        return _stream()


@pytest.mark.asyncio
async def test_responses_session_uses_previous_response_id_across_turns() -> None:
    inferencer = FakeInferencer()
    session = ResponsesSession(inferencer=inferencer)

    session.add_user_message("第一轮")
    await session.create_response(model="deepseek-v4-flash")
    session.mark_response_completed("resp_1")

    session.add_user_message("第二轮")
    await session.create_response(model="deepseek-v4-flash")

    assert inferencer.calls[0]["previous_response_id"] is None
    assert inferencer.calls[0]["input"] == [{"role": "user", "content": "第一轮"}]
    assert inferencer.calls[1]["previous_response_id"] == "resp_1"
    assert inferencer.calls[1]["input"] == [{"role": "user", "content": "第二轮"}]


@pytest.mark.asyncio
async def test_responses_session_queues_function_call_output() -> None:
    inferencer = FakeInferencer()
    session = ResponsesSession(inferencer=inferencer, last_response_id="resp_tool")

    session.add_function_call_output("call_1", '{"ok": true}')
    await session.create_response(model="deepseek-v4-flash")

    assert inferencer.calls[0]["previous_response_id"] == "resp_tool"
    assert inferencer.calls[0]["input"] == [
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": '{"ok": true}',
        }
    ]

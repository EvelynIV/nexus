from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterable

from nexus.infrastructure.responses import AsyncInferencer

DEFAULT_SYSTEM_PROMPT = (
    "你是一个中文语音助手。"
    "你的回答必须是自然口语，适合直接朗读。"
    "只输出纯文本口语内容。"
    "严禁输出 Markdown 代码块 标题 列表 链接 emoji 表情符号 或任何装饰性符号。"
    "尽量不用标点和书面化表达。"
    "如果输入里给了当前说话人的信息，只把它当作理解上下文，不要直接复述声纹标签或识别元数据。"
)


@dataclass
class ResponsesSession:
    """Responses 会话状态。

    会话只维护 Responses 链路所需的最小状态：系统 instructions、上一次
    response id，以及下一次请求需要提交的 input items。
    """

    inferencer: AsyncInferencer
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    instructions: str = DEFAULT_SYSTEM_PROMPT
    last_response_id: str | None = None
    _pending_input: list[dict[str, Any]] = field(default_factory=list)

    def add_user_message(self, content: str) -> None:
        self._pending_input.append(
            {
                "role": "user",
                "content": content,
            }
        )

    def add_function_call_output(self, call_id: str, output: str) -> None:
        self._pending_input.append(
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            }
        )

    def add_input_item(self, item: dict[str, Any]) -> None:
        self._pending_input.append(item)

    async def create_response(
        self,
        *,
        model: str,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        stream: bool = True,
    ) -> AsyncIterable[Any]:
        input_items = self._drain_pending_input()
        try:
            return await self.inferencer.create(
                model=model,
                input=input_items,
                instructions=self.instructions,
                tools=tools or [],
                previous_response_id=self.last_response_id,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                stream=stream,
                store=True,
            )
        except Exception:
            self.restore_pending_input(input_items)
            raise

    def mark_response_completed(self, response_id: str | None) -> None:
        if response_id:
            self.last_response_id = response_id

    def restore_pending_input(self, items: list[dict[str, Any]]) -> None:
        self._pending_input = items + self._pending_input

    def _drain_pending_input(self) -> list[dict[str, Any]]:
        items = self._pending_input
        self._pending_input = []
        return items

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from openai.types.chat import (
    ChatCompletionAudioParam,
    ChatCompletionMessageParam,
    ChatCompletionToolUnionParam,
)

from nexus.infrastructure.chat import Inferencer


@dataclass
class ChatCompletionUseCase:
    base_url: str
    api_key: str

    def execute(
        self,
        *,
        messages: List[ChatCompletionMessageParam],
        model: str,
        audio: Optional[ChatCompletionAudioParam],
        tools: List[ChatCompletionToolUnionParam],
        frequency_penalty: Optional[float],
        temperature: Optional[float],
        max_tokens: Optional[int],
        stream: bool,
    ):
        inferencer = Inferencer(api_key=self.api_key, base_url=self.base_url)
        return inferencer.chat(
            messages=messages,
            model=model,
            audio=audio,
            tools=tools,
            frequency_penalty=frequency_penalty,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )

import logging
from typing import Iterator, List, Optional

from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_audio_param import ChatCompletionAudioParam
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_union_param import (
    ChatCompletionToolUnionParam,
)
from openai.types.completion_usage import CompletionUsage

logger = logging.getLogger(__name__)


class Inferencer:
    """
    Chat 推理器，封装 OpenAI 兼容的 Chat Completions API 调用。
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
    ):
        """
        初始化 Chat 推理器。

        :param base_url: API 服务器地址
        :param api_key: API 密钥
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(
        self,
        messages: List[ChatCompletionMessageParam],
        model: str,
        audio: Optional[ChatCompletionAudioParam] = None,
        tools: List[ChatCompletionToolUnionParam] = [],
        frequency_penalty: Optional[float] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ):

        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=model,
                audio=audio,
                tools=tools,
                frequency_penalty=frequency_penalty,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
            )
            return response
        except Exception as err:
            logger.error("Chat inference error: %s", err)
            return ""

    def chat_stream(
        self,
        messages: list[dict],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> Iterator[str]:
        """
        流式 Chat 推理，返回响应文本迭代器。

        :param messages: 完整的消息列表
        :param model: 模型名称
        :param temperature: 温度参数
        :param max_tokens: 最大 token 数
        :return: 生成器，逐步返回模型生成的文本片段
        """
        try:
            params = {
                "model": model,
                "messages": messages,
                "stream": True,
            }
            if temperature is not None:
                params["temperature"] = temperature
            if max_tokens is not None:
                params["max_tokens"] = max_tokens
            params.update(kwargs)

            stream = self.client.chat.completions.create(**params)

            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except Exception as err:
            logger.error("Chat stream inference error: %s", err)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """关闭客户端连接。"""
        if hasattr(self.client, "close"):
            self.client.close()

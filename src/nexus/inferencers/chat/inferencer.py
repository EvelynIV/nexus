import logging
from typing import Iterator, List, Optional, Union

from openai import OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAudioParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionToolUnionParam,
)
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

logger = logging.getLogger(__name__)


class Inferencer:
    """
    Chat 推理器，封装 OpenAI 兼容的 Chat Completions API 调用。
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
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
        max_tokens: int = 64,
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
            raise
            return ""

    def chat_stream(
        self,
        messages: List[ChatCompletionMessageParam],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: int = 64,
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
            stream_resp = self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            for chunk in stream_resp:
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

import logging
from typing import Iterator

from openai import OpenAI


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

    def chat(self, prompt: str, model: str) -> str:
        """
        非流式 Chat 推理，返回完整响应。

        :param prompt: 用户输入的文本
        :param model: 模型名称
        :return: 模型生成的完整响应文本
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content or ""
        except Exception as err:
            logger.error("Chat inference error: %s", err)
            return ""

    def chat_stream(self, prompt: str, model: str) -> Iterator[str]:
        """
        流式 Chat 推理，返回响应文本迭代器。

        :param prompt: 用户输入的文本
        :param model: 模型名称
        :return: 生成器，逐步返回模型生成的文本片段
        """
        try:
            stream = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )

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

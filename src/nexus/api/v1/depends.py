"""
API 层配置与依赖注入辅助
"""

from dataclasses import dataclass
from typing import Annotated, Optional

from fastapi import Depends

from nexus.inferencers.chat.inferencer import Inferencer as ChatInferencer


@dataclass
class ChatSettings:
    """Chat API 配置"""

    base_url: str = "http://localhost:8080/v1"
    api_key: str = "no-key"


_chat_settings: Optional[ChatSettings] = None


def configure_chat(base_url: str, api_key: str):
    """配置全局设置"""
    global _chat_settings
    _chat_settings = ChatSettings(base_url=base_url, api_key=api_key)


def get_chat_settings() -> ChatSettings:
    if _chat_settings is None:
        raise RuntimeError("Chat settings not configured. Call configure() first.")
    return _chat_settings


def get_chat_inferencer(
    settings: Annotated[ChatSettings, Depends(get_chat_settings)],
) -> ChatInferencer:
    return ChatInferencer(
        base_url=settings.base_url,
        api_key=settings.api_key,
    )

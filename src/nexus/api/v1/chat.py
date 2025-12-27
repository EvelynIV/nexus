"""
FastAPI 路由 - Chat Completions API
兼容 OpenAI Chat API 格式
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Annotated, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from nexus.inferencers.chat.inferencer import Inferencer


router = APIRouter(prefix="/chat", tags=["Chat"])

logger = logging.getLogger(__name__)


# ============== 配置 ==============


@dataclass
class ChatSettings:
    """Chat API 配置"""
    base_url: str = "http://localhost:8080/v1"
    api_key: str = "no-key"


_settings: Optional[ChatSettings] = None


def get_settings() -> ChatSettings:
    if _settings is None:
        raise RuntimeError("Chat settings not configured. Call configure() first.")
    return _settings


def configure(base_url: str, api_key: str):
    """配置全局设置"""
    global _settings
    _settings = ChatSettings(base_url=base_url, api_key=api_key)


def get_inferencer(
    settings: Annotated[ChatSettings, Depends(get_settings)],
) -> Inferencer:
    return Inferencer(
        base_url=settings.base_url,
        api_key=settings.api_key,
    )


# ============== 请求/响应模型 ==============


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "mt01"
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


# ============== API 端点 ==============


def _generate_sse_stream(
    inferencer: Inferencer,
    messages: list[dict],
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
):
    """
    生成 SSE 流式响应（兼容 OpenAI 流式格式）
    """
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    for chunk in inferencer.chat_stream(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    ):
        event_data = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": chunk},
                "finish_reason": None,
            }],
        }
        yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"

    # 发送结束 chunk
    end_data = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }],
    }
    yield f"data: {json.dumps(end_data, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    inferencer: Annotated[Inferencer, Depends(get_inferencer)],
):
    """
    创建 Chat Completion

    兼容 OpenAI Chat API 格式
    支持 stream=True 参数返回 SSE 流式响应
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    # 转换消息格式为字典列表（透明转发）
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    model = request.model

    if request.stream:
        # 流式响应
        return StreamingResponse(
            _generate_sse_stream(
                inferencer=inferencer,
                messages=messages,
                model=model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # 非流式响应
    try:
        response_text = inferencer.chat(
            messages=messages,
            model=model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        return ChatCompletionResponse(
            id=completion_id,
            created=int(time.time()),
            model=model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop",
                )
            ],
            usage=ChatCompletionUsage(),
        )
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

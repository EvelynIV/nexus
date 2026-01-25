"""
FastAPI 路由 - Chat Completions API
兼容 OpenAI Chat API 格式
"""

import json
import logging
from collections.abc import Iterable, Iterator
from typing import Annotated, List, Optional, Union

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import StreamingResponse
from openai.types.chat.chat_completion_audio_param import ChatCompletionAudioParam
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_union_param import (
    ChatCompletionToolUnionParam,
)

from nexus.inferencers.chat.inferencer import Inferencer

from .depends import get_chat_inferencer as get_inferencer

router = APIRouter(prefix="/chat", tags=["Chat"])

logger = logging.getLogger(__name__)


def get_stream_response(
    straem_response: Iterable[ChatCompletionChunk],
) -> Iterator[str]:
    for chunk in straem_response:
        chunk = chunk.model_dump()
        chunk = json.dumps(chunk, ensure_ascii=False)
        yield f"data: {chunk}\n\n"


@router.post("/completions")
async def create_chat_completion(
    inferencer: Annotated[Inferencer, Depends(get_inferencer)],
    messages: Annotated[List[ChatCompletionMessageParam], Body(..., embed=True)],
    model: Annotated[str, Body(..., embed=True)],
    audio: Annotated[Optional[ChatCompletionAudioParam], Body(embed=True)] = None,
    tools: Annotated[List[ChatCompletionToolUnionParam], Body(embed=True)] = [],
    frequency_penalty: Annotated[Optional[float], Body(embed=True)] = None,
    temperature: Annotated[Optional[float], Body(embed=True)] = None,
    max_tokens: Annotated[Optional[int], Body(embed=True)] = None,
    stream: Annotated[bool, Body(embed=True)] = False,
):
    """
    创建 Chat Completion

    兼容 OpenAI Chat API 格式
    支持 stream=True 参数返回 SSE 流式响应
    """
    logger.info(f"Request received: model={model}, len(messages)={len(messages)}")
    try:
        response = inferencer.chat(
            messages=messages,
            model=model,
            audio=audio,
            tools=tools,
            frequency_penalty=frequency_penalty,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )
        if not stream:
            return response  # 非流式直接返回
        # 处理流式返回
        if stream:
            # 流式响应
            return StreamingResponse(
                get_stream_response(response),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

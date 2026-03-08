"""HTTP interface for OpenAI-compatible chat completions."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator
from typing import Annotated, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import StreamingResponse
from openai.types.chat.chat_completion_audio_param import ChatCompletionAudioParam
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_union_param import ChatCompletionToolUnionParam

from nexus.application.container import AppContainer, get_container

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


def _stream_response(stream_response: Iterable[ChatCompletionChunk]) -> Iterator[str]:
    for chunk in stream_response:
        yield f"data: {json.dumps(chunk.model_dump(), ensure_ascii=False)}\n\n"


@router.post("/completions")
async def create_chat_completion(
    container: Annotated[AppContainer, Depends(get_container)],
    messages: Annotated[List[ChatCompletionMessageParam], Body(..., embed=True)],
    model: Annotated[str, Body(..., embed=True)],
    audio: Annotated[Optional[ChatCompletionAudioParam], Body(embed=True)] = None,
    tools: Annotated[List[ChatCompletionToolUnionParam], Body(embed=True)] = [],
    frequency_penalty: Annotated[Optional[float], Body(embed=True)] = None,
    temperature: Annotated[Optional[float], Body(embed=True)] = None,
    max_tokens: Annotated[Optional[int], Body(embed=True)] = None,
    stream: Annotated[bool, Body(embed=True)] = False,
):
    logger.info("Chat completion request model=%s messages=%s", model, len(messages))
    try:
        response = container.chat.execute(
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
            return response

        return StreamingResponse(
            _stream_response(response),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
    except Exception as exc:
        logger.error("Chat completion error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

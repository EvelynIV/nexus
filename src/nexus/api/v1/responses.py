"""HTTP interface for OpenAI-compatible Responses API."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import StreamingResponse

from nexus.application.container import AppContainer, get_container

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/responses", tags=["Responses"])


def _stream_response(stream_response: Iterable[Any]) -> Iterator[str]:
    for event in stream_response:
        payload = event.model_dump(exclude_none=True) if hasattr(event, "model_dump") else event
        event_type = payload.get("type") if isinstance(payload, dict) else None
        event_line = f"event: {event_type}\n" if event_type else ""
        yield f"{event_line}data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("")
async def create_response(
    container: Annotated[AppContainer, Depends(get_container)],
    body: Annotated[dict[str, Any], Body(...)],
):
    model = body.get("model")
    input_value = body.get("input")
    if not model:
        raise HTTPException(status_code=400, detail="model is required")
    if input_value is None:
        raise HTTPException(status_code=400, detail="input is required")

    stream = bool(body.get("stream", False))
    logger.info("Responses request model=%s stream=%s", model, stream)
    try:
        response = container.responses.execute(
            model=model,
            input=input_value,
            instructions=body.get("instructions"),
            tools=body.get("tools"),
            previous_response_id=body.get("previous_response_id"),
            temperature=body.get("temperature"),
            max_output_tokens=body.get("max_output_tokens"),
            stream=stream,
            store=body.get("store", True),
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
        logger.error("Responses error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

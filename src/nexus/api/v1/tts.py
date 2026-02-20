"""HTTP interface for OpenAI-compatible text-to-speech endpoint."""

from __future__ import annotations

import logging
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from nexus.application.container import AppContainer, get_container

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/audio", tags=["Audio"])


class CreateSpeechRequest(BaseModel):
    model: str = Field(default="tts-1", description="TTS model")
    input: str = Field(..., description="Input text", max_length=4096)
    voice: str = Field(default="alloy", description="Voice")
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3", description="Audio response format"
    )
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Playback speed")


AUDIO_MIME_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}


@router.post("/speech")
async def create_speech(
    request: CreateSpeechRequest,
    container: Annotated[AppContainer, Depends(get_container)],
):
    if not request.input.strip():
        raise HTTPException(status_code=400, detail="Input text is required")

    mime_type = AUDIO_MIME_TYPES.get(request.response_format, "audio/mpeg")

    try:
        return StreamingResponse(
            container.tts.stream_audio(
                text=request.input,
                model=request.model,
                voice=request.voice,
                response_format=request.response_format,
                speed=request.speed,
            ),
            media_type=mime_type,
            headers={
                "Content-Disposition": f'attachment; filename="speech.{request.response_format}"',
                "Transfer-Encoding": "chunked",
            },
        )
    except Exception as exc:
        logger.error("TTS error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

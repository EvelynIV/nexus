"""HTTP interface for OpenAI-compatible transcription endpoints."""

from __future__ import annotations

import io
import json
from typing import Annotated, Optional

import soundfile as sf
from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from openai.types.audio import Transcription

from nexus.application.container import AppContainer, get_container
from nexus.application.transcribe import TranscriptionBase64Request

router = APIRouter(prefix="/audio", tags=["Audio"])


def _generate_sse_stream(use_case, pcm_data: bytes, sample_rate: int, language: str):
    for text in use_case.transcribe_pcm_stream(
        pcm_data=pcm_data,
        sample_rate=sample_rate,
        language=language,
    ):
        event_data = {
            "type": "transcript.text.delta",
            "text": text,
        }
        yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/transcriptions")
async def create_transcription(
    container: Annotated[AppContainer, Depends(get_container)],
    file: Annotated[Optional[UploadFile], File()] = None,
    model: Annotated[str, Form()] = "whisper-1",
    language: Annotated[Optional[str], Form()] = None,
    stream: Annotated[bool, Form()] = False,
):
    del model
    if file is None:
        raise HTTPException(status_code=400, detail="No audio file provided")

    try:
        file_content = await file.read()
        audio_data, sample_rate = sf.read(io.BytesIO(file_content), dtype="int16")
        pcm_data = audio_data.tobytes()

        if stream:
            return StreamingResponse(
                _generate_sse_stream(
                    use_case=container.transcribe,
                    pcm_data=pcm_data,
                    sample_rate=sample_rate,
                    language=language or "",
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        result = container.transcribe.transcribe_pcm(
            pcm_data=pcm_data,
            sample_rate=sample_rate,
            language=language or "",
        )
        return Transcription(text=result.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")


@router.post("/transcriptions/base64", response_model=Transcription)
async def create_transcription_base64(
    container: Annotated[AppContainer, Depends(get_container)],
    request: TranscriptionBase64Request,
):
    try:
        result = container.transcribe.transcribe_base64(
            base64_data=request.audio,
            sample_rate=request.sample_rate,
            language=request.language or "",
            hotwords=request.hotwords,
        )
        return Transcription(text=result.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")


@router.post("/transcriptions/raw", response_model=Transcription)
async def create_transcription_raw(
    container: Annotated[AppContainer, Depends(get_container)],
    body: Annotated[bytes, Body(media_type="application/octet-stream")],
    language: str = "",
    sample_rate: int = 16000,
):
    try:
        result = container.transcribe.transcribe_pcm(
            pcm_data=body,
            sample_rate=sample_rate,
            language=language,
        )
        return Transcription(text=result.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")

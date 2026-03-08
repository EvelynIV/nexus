"""Transcribe bounded-context models."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class TranscriptionBase64Request(BaseModel):
    """Request payload for base64 PCM transcription endpoint."""

    audio: str
    model: str = "whisper-1"
    language: Optional[str] = None
    sample_rate: int = 16000
    hotwords: Optional[List[str]] = None

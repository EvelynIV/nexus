"""Transcribe application service."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Generator, Iterator, List, Optional

import numpy as np

from nexus.infrastructure.asr import Inferencer


@dataclass
class TranscribeResult:
    text: str
    language: Optional[str] = None


class TranscribeService:
    """Audio transcription service with PCM/base64 helpers."""

    def __init__(self, grpc_addr: str, interim_results: bool = False):
        self.grpc_addr = grpc_addr
        self.interim_results = interim_results

    def _pcm_to_chunks(self, pcm_data: bytes, chunk_size: int = 3200) -> Iterator[np.ndarray]:
        for i in range(0, len(pcm_data), chunk_size):
            chunk = pcm_data[i : i + chunk_size]
            if chunk:
                yield np.frombuffer(chunk, dtype=np.int16)

    def transcribe_pcm(
        self,
        pcm_data: bytes,
        sample_rate: int = 16000,
        language: str = "",
        hotwords: Optional[List[str]] = None,
    ) -> TranscribeResult:
        transcripts: List[str] = []
        with Inferencer(self.grpc_addr) as inferencer:
            audio_iter = self._pcm_to_chunks(pcm_data)
            for result in inferencer.transcribe(
                audio=audio_iter,
                sample_rate=sample_rate,
                language_code=language,
                hotwords=hotwords,
                interim_results=self.interim_results,
            ):
                if result.is_final:
                    transcripts.append(result.transcript)

        return TranscribeResult(text="".join(transcripts), language=language)

    def transcribe_pcm_stream(
        self,
        pcm_data: bytes,
        sample_rate: int = 16000,
        language: str = "",
        hotwords: Optional[List[str]] = None,
    ) -> Generator[str, None, None]:
        with Inferencer(self.grpc_addr) as inferencer:
            audio_iter = self._pcm_to_chunks(pcm_data)
            for result in inferencer.transcribe(
                audio=audio_iter,
                sample_rate=sample_rate,
                language_code=language,
                hotwords=hotwords,
                interim_results=self.interim_results,
            ):
                if result.transcript:
                    yield result.transcript

    def transcribe_base64(
        self,
        base64_data: str,
        sample_rate: int = 16000,
        language: str = "",
        hotwords: Optional[List[str]] = None,
    ) -> TranscribeResult:
        pcm_data = base64.b64decode(base64_data)
        return self.transcribe_pcm(
            pcm_data=pcm_data,
            sample_rate=sample_rate,
            language=language,
            hotwords=hotwords,
        )

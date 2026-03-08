from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .service import TranscribeService


@dataclass
class TranscribeUseCase:
    grpc_addr: str
    interim_results: bool = False

    def _service(self) -> TranscribeService:
        return TranscribeService(
            grpc_addr=self.grpc_addr,
            interim_results=self.interim_results,
        )

    def transcribe_pcm(
        self,
        *,
        pcm_data: bytes,
        sample_rate: int,
        language: str,
    ):
        return self._service().transcribe_pcm(
            pcm_data=pcm_data,
            sample_rate=sample_rate,
            language=language,
        )

    def transcribe_pcm_stream(
        self,
        *,
        pcm_data: bytes,
        sample_rate: int,
        language: str,
    ):
        return self._service().transcribe_pcm_stream(
            pcm_data=pcm_data,
            sample_rate=sample_rate,
            language=language,
        )

    def transcribe_base64(
        self,
        *,
        base64_data: str,
        sample_rate: int,
        language: str,
        hotwords: Optional[list[str]],
    ):
        return self._service().transcribe_base64(
            base64_data=base64_data,
            sample_rate=sample_rate,
            language=language,
            hotwords=hotwords,
        )

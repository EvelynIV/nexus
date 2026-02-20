from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from nexus.infrastructure.tts import Inferencer


@dataclass
class TextToSpeechUseCase:
    base_url: str
    api_key: str

    def stream_audio(
        self,
        *,
        text: str,
        model: str,
        voice: str,
        response_format: str,
        speed: float,
    ) -> Iterator[bytes]:
        inferencer = Inferencer(base_url=self.base_url, api_key=self.api_key)
        return inferencer.speech_stream(
            input=text,
            model=model,
            voice=voice,
            response_format=response_format,
            speed=speed,
        )

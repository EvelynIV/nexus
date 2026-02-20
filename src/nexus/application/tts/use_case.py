from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

from nexus.infrastructure.tts import Inferencer


@dataclass
class TextToSpeechUseCase:
    base_url: str
    api_key: str

    async def stream_audio(
        self,
        *,
        text: str,
        model: str,
        voice: str,
        response_format: str,
        speed: float,
    ) -> AsyncIterator[bytes]:
        inferencer = Inferencer(base_url=self.base_url, api_key=self.api_key)
        try:
            async for chunk in inferencer.speech_stream(
                input=text,
                model=model,
                voice=voice,
                response_format=response_format,
                speed=speed,
            ):
                yield chunk
        finally:
            await inferencer.close()

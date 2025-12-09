from __future__ import annotations

from typing import Iterable, List, Optional

import asyncio
import grpc

from ..clients.grpc.stubs import UxSpeechClient
from ..generated import ux_speech_pb2
from ..core.config import settings


def _parse_hotwords(hotwords: str) -> List[str]:
    if not hotwords:
        return []
    return [
        w.strip()
        for w in hotwords.replace(";", " ").replace(",", " ").split()
        if w.strip()
    ]


def _build_request_stream(
    audio_bytes: bytes,
    language: str = "zh-CN",
    sample_rate: int = 16000,
    interim_results: bool = False,
    hotwords: Optional[List[str]] = None,
    hotword_bias: float = 0.0,
) -> Iterable[ux_speech_pb2.StreamingRecognizeRequest]:
    """
    First yields the streaming config, then yields audio content.
    """
    config = ux_speech_pb2.StreamingRecognitionConfig(
        config=ux_speech_pb2.RecognitionConfig(
            encoding=ux_speech_pb2.RecognitionConfig.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code=language,
            enable_automatic_punctuation=True,
            hotwords=hotwords or [],
            hotword_bias=hotword_bias,
        ),
        interim_results=interim_results,
    )

    yield ux_speech_pb2.StreamingRecognizeRequest(streaming_config=config)
     # 你当前逻辑：一次性发送全部音频
    yield ux_speech_pb2.StreamingRecognizeRequest(audio_content=audio_bytes)
 # 如果未来要真·流式分片：
    # chunk_size = 4096
    # for start in range(0, len(audio_bytes), chunk_size):
    #     chunk = audio_bytes[start : start + chunk_size]
    #     yield ux_speech_pb2.StreamingRecognizeRequest(audio_content=chunk)

class TranscriptionsService:
    """
    Application service for speech-to-text gateway.
    """

    def __init__(self, default_client: UxSpeechClient | None = None) -> None:
        self._default_client = default_client

    async def transcribe(
        self,
        *,
        audio_bytes: bytes,
        language: str,
        sample_rate: int,
        interim_results: bool,
        hotwords: str,
        hotword_bias: float,
        grpc_host: str | None = None,
        grpc_port: int | None = None,
        timeout_s: float | None = None,
    ) -> str:
        # Resolve defaults from settings
        host = grpc_host or settings.default_grpc_host
        port = grpc_port or settings.default_grpc_port
        timeout = timeout_s or settings.grpc_timeout_s

        hotwords_list = _parse_hotwords(hotwords)

        # Use shared default client only when host/port not overridden
        use_shared = (
            grpc_host is None
            and grpc_port is None
            and self._default_client is not None
            and self._default_client.host == host
            and self._default_client.port == port
        )

        client = self._default_client if use_shared else UxSpeechClient(host=host, port=port)

        requests_iter = _build_request_stream(
            audio_bytes=audio_bytes,
            language=language,
            sample_rate=sample_rate,
            interim_results=interim_results,
            hotwords=hotwords_list,
            hotword_bias=hotword_bias,
        )

        final_texts: List[str] = []

        try:
            def _call():
                # 阻塞式流调用
                return client.stub.StreamingRecognize(requests_iter, timeout=timeout)

            responses = await asyncio.to_thread(_call)

            for resp in responses:
                for result in getattr(resp, "results", []) or []:
                    alt = getattr(result, "alternative", None)
                    transcript = getattr(alt, "transcript", "") if alt else ""
                    if getattr(result, "is_final", False) and transcript:
                        final_texts.append(transcript)

        finally:
            # 只关闭“临时”客户端
            if not use_shared:
                client.close()

        return " ".join(final_texts).strip()

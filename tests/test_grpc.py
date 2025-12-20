import os
import wave
from typing import Generator, Iterable, Tuple

import grpc

from nexus.protos.asr import ux_speech_pb2 as pb2
from nexus.protos.asr import ux_speech_pb2_grpc as pb2_grpc


DEFAULT_GRPC_ADDR = "192.168.1.24:50018"
DEFAULT_WAV_PATH = os.getenv(
    "UX_SPEECH_WAV_PATH",
    os.path.join("data-bin", "speaker1_a_cn_16k.wav"),
)



def wav_pcm_iter(path: str, chunk_ms: int = 100) -> Tuple[int, Iterable[bytes]]:
    wav = wave.open(path, "rb")
    if wav.getsampwidth() != 2 or wav.getnchannels() != 1:
        wav.close()
        raise ValueError("Expected 16-bit mono PCM WAV.")

    sample_rate = wav.getframerate()
    chunk_frames = max(1, int(sample_rate * (chunk_ms / 1000.0)))

    def _iter() -> Iterable[bytes]:
        try:
            while True:
                chunk = wav.readframes(chunk_frames)
                if not chunk:
                    break
                yield chunk
        finally:
            wav.close()

    return sample_rate, _iter()


def build_requests(
    audio_iter: Iterable[bytes],
    sample_rate: int,
) -> Generator[pb2.StreamingRecognizeRequest, None, None]:
    streaming_config = pb2.StreamingRecognitionConfig(
        config=pb2.RecognitionConfig(
            encoding=pb2.RecognitionConfig.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code="zh-CN",
            enable_automatic_punctuation=True,
        ),
        interim_results=True,
    )

    yield pb2.StreamingRecognizeRequest(streaming_config=streaming_config)
    for audio_chunk in audio_iter:
        yield pb2.StreamingRecognizeRequest(audio_content=audio_chunk)


def run_demo(grpc_addr: str = DEFAULT_GRPC_ADDR, wav_path: str = DEFAULT_WAV_PATH) -> None:
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    sample_rate, audio_iter = wav_pcm_iter(wav_path)

    channel = grpc.insecure_channel(grpc_addr)
    stub = pb2_grpc.UxSpeechStub(channel)

    responses = stub.StreamingRecognize(build_requests(audio_iter, sample_rate))
    for response in responses:
        for result in response.results:
            transcript = result.alternative.transcript
            if transcript:
                suffix = "" if result.is_final else " (partial)"
                print(f"{transcript}{suffix}")

    channel.close()


if __name__ == "__main__":
    run_demo()

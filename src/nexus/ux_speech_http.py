from typing import Iterable, List, Optional

import grpc
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, PlainTextResponse
import uvicorn

from . import ux_speech_pb2
from . import ux_speech_pb2_grpc
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="UxSpeech HTTP -> gRPC proxy")


def _build_request_stream(
    audio_bytes: bytes,
    language: str = "zh-CN",
    sample_rate: int = 16000,
    interim_results: bool = False,
    hotwords: Optional[List[str]] = None,
    hotword_bias: float = 0.0,
) -> Iterable[ux_speech_pb2.StreamingRecognizeRequest]:
    """Yield a stream of StreamingRecognizeRequest for the gRPC API.

    First yields the streaming config, then yields audio chunks.
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
    yield ux_speech_pb2.StreamingRecognizeRequest(audio_content=audio_bytes)
    #chunk_size = 4096
    #for start in range(0, len(audio_bytes), chunk_size):
        #chunk = audio_bytes[start : start + chunk_size]
        #yield ux_speech_pb2.StreamingRecognizeRequest(audio_content=chunk)
@app.post("/v1/audio/transcriptions")
async def transcriptions(
    model: str = Form("ux_speech_grpc_proxy"),
    file: UploadFile = File(...),
    language: str = Form("zh-CN"),
    sample_rate: int = Form(16000),
    interim_results: bool = Form(False),
    hotwords: str = Form(""),
    hotword_bias: float = Form(0.0),
    grpc_host: str = Form("39.106.1.132"),
    grpc_port: int = Form(30029),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
):
    """Mimic OpenAI speech_to_text endpoint.

    Accepts a multipart file upload and optional form params. Forwards
    audio to the local gRPC `UxSpeech` service and returns a JSON with
    the combined final transcript.
    """
    try:
        #await只能写在anync类型的函数里，因为file是uplosdfile打开的，所以支持异步
        audio_bytes = await file.read()
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"failed to read uploaded file: {e}"})

    # 整理hotwords列表
    hotwords_list = [w.strip() for w in hotwords.replace(";", " ").replace(",", " ").split() if w.strip()]
    #创建一个不使用 TLS/SSL 加密，也不做服务器证书校验的“不安全”
    channel = grpc.insecure_channel(f"{grpc_host}:{grpc_port}")
    stub = ux_speech_pb2_grpc.UxSpeechStub(channel)

    requests_iter = _build_request_stream(
        audio_bytes=audio_bytes,
        language=language,
        sample_rate=sample_rate,
        interim_results=interim_results,
        hotwords=hotwords_list,
        hotword_bias=hotword_bias,
    )

    # Collect final results
    final_texts: List[str] = []

    executor = ThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_running_loop()
    try:
        # 在线程池中执行阻塞的 gRPC 流调用
        responses = await loop.run_in_executor(
            executor, lambda: stub.StreamingRecognize(requests_iter, timeout=180)
        )

        for resp in responses:
            for result in resp.results:
                try:
                    transcript = result.alternative.transcript
                except Exception:
                    transcript = ""
                if result.is_final and transcript:
                    final_texts.append(transcript)


    except grpc.RpcError as rpc_err:
        # Map gRPC error to HTTP response
        err_message = rpc_err.details() if hasattr(rpc_err, "details") else str(rpc_err)
        channel.close()
        return JSONResponse(status_code=502, content={"error": "gRPC error", "detail": err_message})
    except Exception as e:
        channel.close()
        return JSONResponse(status_code=500, content={"error": "internal_proxy_error", "detail": str(e)})

    channel.close()

    combined = " ".join(final_texts).strip()

    fmt = (response_format or "json").lower()
    if fmt in ("text", "txt", "plain"):
        return PlainTextResponse(content=combined, status_code=200)
    elif fmt == "srt":
        # 简单把全部文本作为单个 SRT 段（若需要更精细的分段，请用 segments 列表构建）
        srt = "1\n00:00:00,000 --> 00:10:00,000\n" + combined + "\n"
        return PlainTextResponse(content=srt, media_type="text/plain", status_code=200)
    elif fmt == "vtt":
        vtt = "WEBVTT\n\n00:00.000 --> 00:10.000\n" + combined + "\n"
        return PlainTextResponse(content=vtt, media_type="text/vtt", status_code=200)
    else:
        # 默认 json
        result = {"text": combined, "model": model}
        return JSONResponse(status_code=200, content=result)


if __name__ == "__main__":
    # Run with: python ux_speech_http.py
    # Or better: uvicorn ux_speech_http:app --host 0.0.0.0 --port 8000 --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)

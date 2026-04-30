#!/usr/bin/env python3
"""Bridge an Asterisk ARI ExternalMedia SIP call to OpenAI Realtime.

Run from the repository root:

    OPENAI_API_KEY=sk-... poetry run python examples/sip/openai_realtime_phone_demo.py

This is a manual integration demo. Asterisk handles SIP and sends call audio to
this script through ARI ExternalMedia RTP. This script forwards G.711 audio to
the OpenAI Realtime API and sends the model audio back to the call.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import queue
import socket
import ssl
import struct
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal

import g711
import numpy as np
import typer
from openai import AsyncOpenAI
import websockets.sync.client

from nexus.infrastructure.audio.resampler import StreamingResampler


RTP_HEADER_LEN = 12
PCMU_PAYLOAD_TYPE = 0
PCMA_PAYLOAD_TYPE = 8
SAMPLE_RATE = 8000
REALTIME_SAMPLE_RATE = 24000
FRAME_MS = 20
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1000
G711_BYTES_PER_FRAME = SAMPLES_PER_FRAME
PCM_WIDTH_BYTES = 2


@dataclass(frozen=True)
class AriConfig:
    base_url: str
    app: str
    username: str
    password: str

    @property
    def events_ws_url(self) -> str:
        parsed = urllib.parse.urlparse(self.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        netloc = parsed.netloc
        path = parsed.path.rstrip("/")
        api_key = urllib.parse.quote(f"{self.username}:{self.password}")
        app = urllib.parse.quote(self.app)
        return f"{scheme}://{netloc}{path}/events?app={app}&api_key={api_key}"


class AriClient:
    def __init__(self, config: AriConfig) -> None:
        self.config = config

    def _request(self, path: str, *, method: str, data: bytes | None = None, **params: str) -> Any:
        url = f"{self.config.base_url.rstrip('/')}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(url, data=data, method=method)
        token = base64.b64encode(
            f"{self.config.username}:{self.config.password}".encode("utf-8")
        ).decode("ascii")
        request.add_header("Authorization", f"Basic {token}")
        with urllib.request.urlopen(request, timeout=8) as response:
            body = response.read()
        if not body:
            return {}
        return json.loads(body.decode("utf-8"))

    def post(self, path: str, **params: str) -> dict[str, Any]:
        result = self._request(path, method="POST", data=b"", **params)
        return result if isinstance(result, dict) else {}

    def get(self, path: str, **params: str) -> Any:
        return self._request(path, method="GET", **params)

    def delete(self, path: str, **params: str) -> None:
        try:
            self._request(path, method="DELETE", **params)
        except Exception:
            pass


def queue_drop_oldest(target: queue.Queue[bytes], item: bytes) -> None:
    if target.full():
        try:
            target.get_nowait()
        except queue.Empty:
            pass
    target.put_nowait(item)


def pcm16_bytes_to_float32(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def float32_to_pcm16_bytes(samples: np.ndarray) -> bytes:
    return np.clip(samples * 32768.0, -32768, 32767).astype(np.int16).tobytes()


class G711PcmTranscoder:
    def __init__(self, codec: str) -> None:
        self.codec = codec
        self._to_realtime = StreamingResampler(SAMPLE_RATE, REALTIME_SAMPLE_RATE)
        self._to_rtp = StreamingResampler(REALTIME_SAMPLE_RATE, SAMPLE_RATE)

    def reset(self) -> None:
        self._to_realtime.reset()
        self._to_rtp.reset()

    def rtp_payload_to_realtime_pcm(self, payload: bytes) -> bytes:
        if self.codec == "alaw":
            samples = g711.decode_alaw(payload)
        else:
            samples = g711.decode_ulaw(payload)
        pcm8 = float32_to_pcm16_bytes(samples)
        return self._to_realtime.process(pcm8)

    def realtime_pcm_to_rtp_payload(self, pcm24: bytes) -> bytes:
        pcm8 = self._to_rtp.process(pcm24)
        if not pcm8:
            return b""
        samples = pcm16_bytes_to_float32(pcm8)
        if self.codec == "alaw":
            return g711.encode_alaw(samples)
        return g711.encode_ulaw(samples)


class RtpAudioEndpoint:
    def __init__(
        self,
        *,
        bind_host: str,
        bind_port: int,
        codec: str,
        input_queue_size: int,
        output_queue_size: int,
    ) -> None:
        self.bind_host = bind_host
        self.bind_port = bind_port
        self.codec = codec
        self.payload_type = PCMA_PAYLOAD_TYPE if codec == "alaw" else PCMU_PAYLOAD_TYPE
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((bind_host, bind_port))
        self.remote_addr: tuple[str, int] | None = None
        self.seq = 0
        self.timestamp = 0
        self.ssrc = int(time.time()) & 0xFFFFFFFF
        self.running = threading.Event()
        self.running.set()
        self.session_active = threading.Event()
        self.input_queue: queue.Queue[bytes] = queue.Queue(maxsize=input_queue_size)
        self.output_queue: queue.Queue[bytes] = queue.Queue(maxsize=output_queue_size)
        self._threads: list[threading.Thread] = []
        self._playback_lock = threading.Lock()
        self._playback_generation = 0
        self._played_output_samples = 0

    def start(self) -> None:
        self._threads = [
            threading.Thread(target=self._recv_loop, name="rtp-recv", daemon=True),
            threading.Thread(target=self._send_loop, name="rtp-send", daemon=True),
        ]
        for thread in self._threads:
            thread.start()
        print(
            f"RTP endpoint listening on {self.bind_host}:{self.bind_port} "
            f"codec={self.codec}. Waiting for ARI ExternalMedia packets."
        )

    def stop(self) -> None:
        self.running.clear()
        self.reset_session()
        try:
            self.sock.close()
        except OSError:
            pass

    def reset_session(self) -> None:
        self.session_active.clear()
        self.remote_addr = None
        self._clear_queue(self.input_queue)
        self.clear_output_audio(reset_played=True)

    def start_session(self) -> None:
        self.reset_session()
        self.session_active.set()

    def read_input_payload(self, timeout: float = 0.2) -> bytes | None:
        try:
            return self.input_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def enqueue_output_audio(self, audio: bytes) -> None:
        if not audio:
            return
        queue_drop_oldest(self.output_queue, audio)

    def clear_output_audio(self, *, reset_played: bool = False) -> None:
        self._clear_queue(self.output_queue)
        with self._playback_lock:
            self._playback_generation += 1
            if reset_played:
                self._played_output_samples = 0

    def begin_output_audio(self) -> None:
        with self._playback_lock:
            self._played_output_samples = 0

    def played_output_audio_ms(self) -> int:
        with self._playback_lock:
            return int(self._played_output_samples * 1000 / SAMPLE_RATE)

    def playback_generation(self) -> int:
        with self._playback_lock:
            return self._playback_generation

    def _clear_queue(self, target: queue.Queue[bytes]) -> None:
        while True:
            try:
                target.get_nowait()
            except queue.Empty:
                break

    def _recv_loop(self) -> None:
        while self.running.is_set():
            try:
                packet, addr = self.sock.recvfrom(2048)
            except OSError:
                return
            if len(packet) <= RTP_HEADER_LEN:
                continue
            if not self.session_active.is_set():
                continue
            self.remote_addr = addr
            queue_drop_oldest(self.input_queue, packet[RTP_HEADER_LEN:])

    def _send_loop(self) -> None:
        pending = bytearray()
        next_send_at = time.monotonic()
        playback_generation = self.playback_generation()

        while self.running.is_set():
            current_generation = self.playback_generation()
            if current_generation != playback_generation:
                pending.clear()
                playback_generation = current_generation
                next_send_at = time.monotonic()

            if not self.session_active.is_set():
                pending.clear()
                time.sleep(0.02)
                next_send_at = time.monotonic()
                continue

            if len(pending) < G711_BYTES_PER_FRAME:
                try:
                    pending.extend(self.output_queue.get(timeout=0.02))
                except queue.Empty:
                    continue

            while len(pending) >= G711_BYTES_PER_FRAME and self.session_active.is_set():
                if self.remote_addr is None:
                    time.sleep(0.02)
                    next_send_at = time.monotonic()
                    break

                frame = bytes(pending[:G711_BYTES_PER_FRAME])
                del pending[:G711_BYTES_PER_FRAME]
                self._send_rtp_payload(frame)

                next_send_at += FRAME_MS / 1000
                delay = next_send_at - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_send_at = time.monotonic()

    def _send_rtp_payload(self, payload: bytes) -> None:
        if self.remote_addr is None:
            return
        header = struct.pack(
            "!BBHII",
            0x80,
            self.payload_type,
            self.seq,
            self.timestamp,
            self.ssrc,
        )
        try:
            self.sock.sendto(header + payload, self.remote_addr)
        except OSError:
            return
        self.seq = (self.seq + 1) & 0xFFFF
        self.timestamp = (self.timestamp + len(payload)) & 0xFFFFFFFF
        with self._playback_lock:
            self._played_output_samples += len(payload)


class ExternalMediaBridge:
    def __init__(
        self,
        *,
        ari: AriClient,
        app: str,
        external_host: str,
        codec: str,
    ) -> None:
        self.ari = ari
        self.app = app
        self.external_host = external_host
        self.codec = codec
        self.bridge_id: str | None = None
        self.external_channel_id: str | None = None
        self.call_channel_id: str | None = None

    def attach_call(self, channel_id: str) -> bool:
        if self.call_channel_id is not None:
            print(f"Already attached to channel {self.call_channel_id}; ignoring {channel_id}")
            return False
        self.call_channel_id = channel_id
        self.ari.post(f"/channels/{channel_id}/answer")
        bridge = self.ari.post("/bridges", type="mixing")
        self.bridge_id = bridge["id"]
        external = self.ari.post(
            "/channels/externalMedia",
            app=self.app,
            external_host=self.external_host,
            format=self.codec,
            direction="both",
        )
        self.external_channel_id = external["id"]
        self.ari.post(f"/bridges/{self.bridge_id}/addChannel", channel=channel_id)
        self.ari.post(
            f"/bridges/{self.bridge_id}/addChannel",
            channel=self.external_channel_id,
        )
        print(
            "Attached call to OpenAI media bridge: "
            f"call={channel_id} bridge={self.bridge_id} external={self.external_channel_id}"
        )
        return True

    def cleanup(self) -> None:
        if self.external_channel_id:
            self.ari.delete(f"/channels/{self.external_channel_id}")
        if self.bridge_id:
            self.ari.delete(f"/bridges/{self.bridge_id}")
        self.bridge_id = None
        self.external_channel_id = None
        self.call_channel_id = None

    def detach_if_related(self, channel_id: str) -> bool:
        if channel_id not in {self.call_channel_id, self.external_channel_id}:
            return False
        print(f"Channel ended; cleaning media bridge: {channel_id}")
        self.cleanup()
        return True


@dataclass(frozen=True)
class OpenAiRealtimeConfig:
    api_key: str
    base_url: str
    model: str
    voice: str
    instructions: str
    greeting: str
    audio_format: str


class OpenAiRealtimeBridge:
    def __init__(
        self,
        *,
        config: OpenAiRealtimeConfig,
        rtp: RtpAudioEndpoint,
        error_queue: queue.Queue[BaseException],
    ) -> None:
        self.config = config
        self.rtp = rtp
        self.error_queue = error_queue
        self._active = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._connection: Any | None = None
        self._response_active = False
        self._current_response_id: str | None = None
        self._current_audio_item_id: str | None = None
        self._current_audio_content_index = 0
        self._ignored_error_event_ids: set[str] = set()
        self._suppressed_response_ids: set[str] = set()
        self._transcoder = G711PcmTranscoder(rtp.codec)

    def start(self) -> None:
        self.stop()
        self._response_active = False
        self._current_response_id = None
        self._current_audio_item_id = None
        self._current_audio_content_index = 0
        self._ignored_error_event_ids.clear()
        self._suppressed_response_ids.clear()
        self._transcoder.reset()
        self._active.set()
        self._thread = threading.Thread(
            target=self._thread_main,
            name="openai-realtime",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._active.clear()
        if self._loop and self._connection:
            if not self._loop.is_closed():
                try:
                    future = asyncio.run_coroutine_threadsafe(self._connection.close(), self._loop)
                    future.result(timeout=3)
                except Exception:
                    pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        self._thread = None
        self._loop = None
        self._connection = None

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run())
        except BaseException as exc:
            try:
                self.error_queue.put_nowait(exc)
            except queue.Full:
                pass
            self.rtp.session_active.clear()

    async def _run(self) -> None:
        self._loop = asyncio.get_running_loop()
        client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )
        print(f"Connecting OpenAI Realtime model={self.config.model} voice={self.config.voice}")
        async with client.realtime.connect(
            model=self.config.model,
            websocket_connection_options={"ssl": insecure_ssl_context()},
        ) as connection:
            self._connection = connection
            await connection.send(self._build_session_update())
            if self.config.greeting:
                await connection.send(
                    {
                        "type": "response.create",
                        "response": {
                            "instructions": self.config.greeting,
                            "output_modalities": ["audio"],
                        },
                    }
                )

            sender = asyncio.create_task(self._send_phone_audio(connection))
            receiver = asyncio.create_task(self._receive_openai_events(connection))
            done, pending = await asyncio.wait(
                {sender, receiver},
                return_when=asyncio.FIRST_EXCEPTION,
            )
            for task in pending:
                task.cancel()
            for task in done:
                task.result()

    def _build_session_update(self) -> dict[str, Any]:
        realtime_audio_format = {
            "type": self.config.audio_format,
            "rate": REALTIME_SAMPLE_RATE,
        }
        return {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": self.config.model,
                "instructions": self.config.instructions,
                "output_modalities": ["audio"],
                "audio": {
                    "input": {
                        "format": realtime_audio_format,
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.5,
                            "prefix_padding_ms": 300,
                            "silence_duration_ms": 500,
                            "create_response": True,
                            "interrupt_response": True,
                        },
                    },
                    "output": {
                        "format": realtime_audio_format,
                        "voice": self.config.voice,
                    },
                },
            },
        }

    async def _send_phone_audio(self, connection: Any) -> None:
        while self._active.is_set() and self.rtp.session_active.is_set():
            payload = await asyncio.to_thread(self.rtp.read_input_payload, 0.2)
            if not payload:
                continue
            audio = await asyncio.to_thread(self._transcoder.rtp_payload_to_realtime_pcm, payload)
            if not audio:
                continue
            await connection.send(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(audio).decode("ascii"),
                }
            )

    async def _receive_openai_events(self, connection: Any) -> None:
        async for event in connection:
            if not self._active.is_set():
                break

            event_type = get_event_type(event)
            if event_type in {"response.output_audio.delta", "response.audio.delta"}:
                response_id = get_event_attr(event, "response_id")
                if isinstance(response_id, str) and response_id in self._suppressed_response_ids:
                    continue
                self._track_output_audio_delta(event)
                delta = get_event_attr(event, "delta")
                if isinstance(delta, str):
                    payload = await asyncio.to_thread(
                        self._transcoder.realtime_pcm_to_rtp_payload,
                        base64.b64decode(delta),
                    )
                    self.rtp.enqueue_output_audio(payload)
                continue

            if event_type == "error":
                event_id = get_error_client_event_id(event)
                if isinstance(event_id, str) and event_id in self._ignored_error_event_ids:
                    self._ignored_error_event_ids.discard(event_id)
                    print(f"Ignored barge-in race error: {format_event(event)}")
                    continue
                raise RuntimeError(f"OpenAI Realtime error: {format_event(event)}")

            if event_type == "response.created":
                self._response_active = True
                self._current_response_id = get_response_id(event)
            elif event_type == "response.content_part.added":
                self._track_content_part_added(event)
            elif event_type == "input_audio_buffer.speech_started":
                await self._handle_barge_in(connection)
            elif event_type == "response.done":
                response_id = get_response_id(event)
                if response_id:
                    self._suppressed_response_ids.discard(response_id)
                self._response_active = False
                self._current_response_id = None
                self._current_audio_item_id = None

            if event_type == "conversation.item.input_audio_transcription.completed":
                transcript = get_event_attr(event, "transcript")
                print(f"Realtime transcription: {transcript!r}")

            if event_type in {
                "session.created",
                "session.updated",
                "input_audio_buffer.speech_started",
                "input_audio_buffer.speech_stopped",
                "input_audio_buffer.committed",
                "conversation.item.input_audio_transcription.completed",
                "response.created",
                "response.output_audio.done",
                "response.output_audio_transcript.done",
                "response.done",
            }:
                print(f"OpenAI event: {format_event(event)}")

    def _track_output_audio_delta(self, event: Any) -> None:
        response_id = get_event_attr(event, "response_id")
        if isinstance(response_id, str):
            self._current_response_id = response_id
        item_id = get_event_attr(event, "item_id")
        content_index = get_event_attr(event, "content_index")
        if not isinstance(item_id, str):
            return
        if item_id != self._current_audio_item_id:
            self._current_audio_item_id = item_id
            self._current_audio_content_index = content_index if isinstance(content_index, int) else 0
            self.rtp.begin_output_audio()

    def _track_content_part_added(self, event: Any) -> None:
        response_id = get_event_attr(event, "response_id")
        if isinstance(response_id, str):
            self._current_response_id = response_id
        part = get_event_attr(event, "part")
        part_type = get_event_attr(part, "type")
        if part_type != "audio":
            return
        item_id = get_event_attr(event, "item_id")
        content_index = get_event_attr(event, "content_index")
        if isinstance(item_id, str):
            self._current_audio_item_id = item_id
            self._current_audio_content_index = content_index if isinstance(content_index, int) else 0
            self.rtp.begin_output_audio()

    async def _handle_barge_in(self, connection: Any) -> None:
        audio_end_ms = self.rtp.played_output_audio_ms()
        response_id = self._current_response_id
        item_id = self._current_audio_item_id
        content_index = self._current_audio_content_index
        if response_id:
            self._suppressed_response_ids.add(response_id)
        self.rtp.clear_output_audio()

        if self._response_active:
            await self._send_best_effort_barge_in_event(
                connection,
                {
                    "type": "response.cancel",
                },
            )

        if item_id and audio_end_ms > 0:
            await self._send_best_effort_barge_in_event(
                connection,
                {
                    "type": "conversation.item.truncate",
                    "item_id": item_id,
                    "content_index": content_index,
                    "audio_end_ms": audio_end_ms,
                }
            )
            print(f"Barge-in: truncated assistant audio at {audio_end_ms}ms")
        else:
            print("Barge-in: cleared queued assistant audio")

    async def _send_best_effort_barge_in_event(self, connection: Any, event: dict[str, Any]) -> None:
        event_id = f"event_barge_{uuid.uuid4().hex}"
        event["event_id"] = event_id
        self._ignored_error_event_ids.add(event_id)
        await connection.send(event)


def get_event_type(event: Any) -> str:
    if isinstance(event, dict):
        return str(event.get("type") or "")
    return str(getattr(event, "type", "") or "")


def get_event_attr(event: Any, name: str) -> Any:
    if isinstance(event, dict):
        return event.get(name)
    return getattr(event, name, None)


def get_response_id(event: Any) -> str | None:
    response_id = get_event_attr(event, "response_id")
    if isinstance(response_id, str):
        return response_id
    response = get_event_attr(event, "response")
    response_id = get_event_attr(response, "id")
    return response_id if isinstance(response_id, str) else None


def get_error_client_event_id(event: Any) -> str | None:
    error = get_event_attr(event, "error")
    event_id = get_event_attr(error, "event_id")
    return event_id if isinstance(event_id, str) else None


def event_to_dict(event: Any) -> dict[str, Any]:
    if isinstance(event, dict):
        return event
    if hasattr(event, "model_dump"):
        return event.model_dump(mode="json", exclude_none=True)
    return {"type": get_event_type(event)}


def format_event(event: Any) -> str:
    payload = event_to_dict(event)
    event_type = str(payload.get("type") or "")
    if event_type == "error":
        return json.dumps(payload, ensure_ascii=False)
    if event_type == "session.created":
        session = payload.get("session") or {}
        return f"{event_type} id={session.get('id', '<unknown>')}"
    if event_type == "session.updated":
        return event_type
    if event_type == "conversation.item.input_audio_transcription.delta":
        delta = payload.get("delta")
        return f"{event_type} delta={delta!r}"
    if event_type == "conversation.item.input_audio_transcription.completed":
        transcript = payload.get("transcript")
        return f"{event_type} transcript={transcript!r}"
    if event_type == "response.output_audio_transcript.delta":
        delta = payload.get("delta")
        return f"{event_type} delta={delta!r}"
    if event_type == "response.output_audio_transcript.done":
        transcript = payload.get("transcript")
        return f"{event_type} transcript={transcript!r}"
    if event_type == "response.done":
        response = payload.get("response") or {}
        status = response.get("status", "<unknown>") if isinstance(response, dict) else "<unknown>"
        return f"{event_type} status={status}"
    return event_type


def insecure_ssl_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


def main(
    ari_url: str = typer.Option(
        "http://127.0.0.1:8088/ari",
        "--ari-url",
        envvar="SIP_ARI_URL",
        help="Asterisk ARI base URL.",
    ),
    ari_user: str = typer.Option(
        "voicebot",
        "--ari-user",
        envvar="SIP_ARI_USER",
        help="Asterisk ARI username.",
    ),
    ari_password: str = typer.Option(
        "12345678",
        "--ari-password",
        envvar="SIP_ARI_PASSWORD",
        help="Asterisk ARI password.",
    ),
    stasis_app: str = typer.Option(
        "voicebot",
        "--app",
        envvar="SIP_STASIS_APP",
        help="Asterisk Stasis app name.",
    ),
    rtp_host: str = typer.Option(
        "127.0.0.1",
        "--rtp-host",
        envvar="SIP_RTP_HOST",
        help="Local RTP bind host passed to ARI ExternalMedia.",
    ),
    rtp_port: int = typer.Option(
        4000,
        "--rtp-port",
        envvar="SIP_RTP_PORT",
        help="Local RTP bind port passed to ARI ExternalMedia.",
    ),
    codec: Literal["ulaw", "alaw"] = typer.Option(
        "ulaw",
        "--codec",
        envvar="SIP_CODEC",
        help="Asterisk ExternalMedia codec.",
    ),
    model: str = typer.Option(
        "deepseek-v4-flash",
        "--model",
        envvar=["SIP_REALTIME_MODEL", "OPENAI_REALTIME_MODEL", "OPENAI_MODEL"],
        help="Realtime model name.",
    ),
    voice: str = typer.Option(
        "paimon",
        "--voice",
        envvar=["SIP_REALTIME_VOICE", "OPENAI_REALTIME_VOICE", "OPENAI_VOICE"],
        help="Realtime output voice.",
    ),
    instructions: str = typer.Option(
        "You are a concise phone voice assistant. Keep replies brief and natural.",
        "--instructions",
        envvar="SIP_REALTIME_INSTRUCTIONS",
        help="Realtime session instructions.",
    ),
    openai_base_url: str = typer.Option(
        "https://api.openai.com/v1",
        "--openai-base-url",
        envvar=["SIP_OPENAI_BASE_URL", "OPENAI_BASE_URL"],
        help="OpenAI-compatible API base URL.",
    ),
    greeting: str = typer.Option(
        "",
        "--greeting",
        envvar="SIP_REALTIME_GREETING",
        help="Optional initial assistant greeting instruction.",
    ),
    input_queue_size: int = typer.Option(
        128,
        "--input-queue-size",
        envvar="SIP_INPUT_QUEUE_SIZE",
        help="Maximum queued inbound RTP frames.",
    ),
    output_queue_size: int = typer.Option(
        128,
        "--output-queue-size",
        envvar="SIP_OUTPUT_QUEUE_SIZE",
        help="Maximum queued outbound RTP audio chunks.",
    ),
    hangup_existing_calls: bool = typer.Option(
        False,
        "--hangup-existing-calls",
        envvar="SIP_HANGUP_EXISTING_CALLS",
        help="Hang up existing calls in this Stasis app before waiting for a new call.",
    ),
) -> None:
    args = SimpleNamespace(
        ari_url=ari_url,
        ari_user=ari_user,
        ari_password=ari_password,
        app=stasis_app,
        rtp_host=rtp_host,
        rtp_port=rtp_port,
        codec=codec,
        model=model,
        voice=voice,
        instructions=instructions,
        openai_base_url=openai_base_url,
        greeting=greeting,
        input_queue_size=input_queue_size,
        output_queue_size=output_queue_size,
        hangup_existing_calls=hangup_existing_calls,
    )
    run_demo(args)


def check_ari_ready(ari: AriClient) -> None:
    try:
        info = ari.get("/asterisk/info")
    except Exception as exc:
        raise RuntimeError(
            "Unable to connect to Asterisk ARI. Check http.conf, ari.conf, "
            "and the configured ARI credentials."
        ) from exc
    version = ((info.get("system") or {}).get("version")) or "unknown"
    print(f"ARI connected. Asterisk version: {version}")


def find_app_call_channels(ari: AriClient, app: str) -> list[dict[str, Any]]:
    channels = ari.get("/channels")
    if not isinstance(channels, list):
        return []

    app_channels: list[dict[str, Any]] = []
    for channel in channels:
        if not isinstance(channel, dict):
            continue
        dialplan = channel.get("dialplan") or {}
        if dialplan.get("app_name") != "Stasis":
            continue
        if dialplan.get("app_data") != app:
            continue

        name = str(channel.get("name") or "")
        if name.startswith("UnicastRTP/"):
            continue
        app_channels.append(channel)
    return app_channels


def channel_label(channel: dict[str, Any]) -> str:
    channel_id = channel.get("id", "<unknown>")
    name = channel.get("name", "<unnamed>")
    created = channel.get("creationtime", "<unknown time>")
    return f"{channel_id} {name} since {created}"


def require_openai_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required. Export it before running this demo.")
    return api_key


def validate_positive(value: int, name: str) -> None:
    if value <= 0:
        raise RuntimeError(f"{name} must be greater than 0.")


def run_demo(args: SimpleNamespace) -> None:
    try:
        api_key = require_openai_api_key()
        validate_positive(args.input_queue_size, "--input-queue-size")
        validate_positive(args.output_queue_size, "--output-queue-size")
    except Exception as exc:
        print(f"Startup failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(
        "OpenAI Realtime config: "
        f"base_url={args.openai_base_url} model={args.model} voice={args.voice}"
    )

    config = AriConfig(
        base_url=args.ari_url,
        app=args.app,
        username=args.ari_user,
        password=args.ari_password,
    )
    ari = AriClient(config)
    try:
        check_ari_ready(ari)
    except Exception as exc:
        print(f"Startup failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if args.hangup_existing_calls:
        existing_channels = find_app_call_channels(ari, args.app)
        for channel in existing_channels:
            channel_id = channel.get("id")
            if not channel_id:
                continue
            print(f"Hanging up old Stasis call: {channel_label(channel)}")
            ari.delete(f"/channels/{channel_id}")
        if existing_channels:
            time.sleep(0.5)

    rtp = RtpAudioEndpoint(
        bind_host=args.rtp_host,
        bind_port=args.rtp_port,
        codec=args.codec,
        input_queue_size=args.input_queue_size,
        output_queue_size=args.output_queue_size,
    )
    bridge = ExternalMediaBridge(
        ari=ari,
        app=args.app,
        external_host=f"{args.rtp_host}:{args.rtp_port}",
        codec=args.codec,
    )
    event_error: queue.Queue[BaseException] = queue.Queue(maxsize=1)
    realtime = OpenAiRealtimeBridge(
        config=OpenAiRealtimeConfig(
            api_key=api_key,
            base_url=args.openai_base_url,
            model=args.model,
            voice=args.voice,
            instructions=args.instructions,
            greeting=args.greeting,
            audio_format="audio/pcm",
        ),
        rtp=rtp,
        error_queue=event_error,
    )

    def start_call_session(channel_id: str) -> None:
        if bridge.attach_call(channel_id):
            rtp.start_session()
            realtime.start()

    def stop_call_session() -> None:
        realtime.stop()
        rtp.reset_session()

    def event_loop() -> None:
        try:
            print(f"Connecting ARI events: {config.events_ws_url}")
            with websockets.sync.client.connect(config.events_ws_url) as ws:
                for channel in find_app_call_channels(ari, args.app):
                    channel_id = channel.get("id")
                    if not channel_id:
                        continue
                    print(f"Found existing Stasis call: {channel_label(channel)}")
                    start_call_session(str(channel_id))
                    break

                print(f"Waiting for StasisStart(app={args.app})...")
                for raw in ws:
                    event = json.loads(raw)
                    event_type = event.get("type")
                    channel = event.get("channel") or {}
                    channel_id = channel.get("id")
                    channel_name = channel.get("name", "")

                    if event_type in {"StasisEnd", "ChannelDestroyed"}:
                        if channel_id and bridge.detach_if_related(channel_id):
                            stop_call_session()
                        continue

                    if event_type != "StasisStart":
                        continue
                    if not channel_id:
                        continue
                    if channel_id == bridge.external_channel_id:
                        continue
                    print(f"StasisStart: {channel_id} {channel_name}")
                    start_call_session(channel_id)
        except BaseException as exc:
            try:
                event_error.put_nowait(exc)
            except queue.Full:
                pass
            rtp.running.clear()

    threading.Thread(target=event_loop, name="ari-events", daemon=True).start()
    rtp.start()

    try:
        while rtp.running.is_set():
            if not event_error.empty():
                exc = event_error.get_nowait()
                print(f"Demo stopped after error: {exc}", file=sys.stderr)
                raise SystemExit(1) from exc
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        realtime.stop()
        rtp.stop()
        bridge.cleanup()


if __name__ == "__main__":
    typer.run(main)

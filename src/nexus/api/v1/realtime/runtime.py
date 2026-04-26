from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
import uuid
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack
from av import AudioFrame

from nexus.application.container import AppContainer
from nexus.application.realtime.protocol import (
    BroadcastRealtimeSink,
    NullRealtimeReplySink,
    RealtimeReplySink,
    serialize_realtime_server_event,
)
from nexus.domain.realtime import RealtimeSessionState
from nexus.infrastructure.audio.resampler import StreamingResampler

from .controller import RealtimeSessionController


logger = logging.getLogger(__name__)

DEFAULT_REALTIME_MODEL = "gpt-realtime"
DEFAULT_OUTPUT_MODALITIES = ["audio", "text"]
DEFAULT_VOICE = "alloy"


class RealtimeCallError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def deep_merge_dict(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def normalize_output_modalities(value: Any, *, default_modalities: list[str] | None = None) -> list[str]:
    defaults = default_modalities or DEFAULT_OUTPUT_MODALITIES
    raw = list(value or defaults)
    normalized = {str(item).strip().lower() for item in raw if item}
    ordered: list[str] = []
    if "audio" in normalized:
        ordered.append("audio")
    if "text" in normalized:
        ordered.append("text")
    return ordered or defaults.copy()


def build_default_session_config(
    model: str = DEFAULT_REALTIME_MODEL,
    *,
    output_modalities: list[str] | None = None,
) -> dict[str, Any]:
    default_output_modalities = output_modalities or DEFAULT_OUTPUT_MODALITIES
    return {
        "type": "realtime",
        "model": model,
        "output_modalities": default_output_modalities.copy(),
        "tools": [],
        "tool_choice": "auto",
        "audio": {
            "input": {
                "format": {
                    "type": "audio/pcm",
                    "rate": 24000,
                },
                "turn_detection": {
                    "type": "server_vad",
                },
            },
            "output": {
                "format": {
                    "type": "audio/pcm",
                    "rate": 24000,
                },
                "voice": DEFAULT_VOICE,
                "speed": 1.0,
            },
        },
    }


def normalize_session_config(
    raw_session: dict[str, Any] | None,
    *,
    default_model: str = DEFAULT_REALTIME_MODEL,
    default_output_modalities: list[str] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    default_modalities = default_output_modalities or DEFAULT_OUTPUT_MODALITIES
    session = deep_merge_dict(
        build_default_session_config(
            default_model,
            output_modalities=default_modalities,
        ),
        raw_session or {},
    )
    if session.get("type", "realtime") != "realtime":
        raise ValueError("Only realtime session type is currently supported.")
    session["type"] = "realtime"
    session["model"] = session.get("model") or default_model
    session["id"] = session_id or session.get("id") or f"sess_{uuid.uuid4().hex}"
    session["object"] = "realtime.session"
    session["output_modalities"] = normalize_output_modalities(
        session.get("output_modalities"),
        default_modalities=default_modalities,
    )

    audio = session.setdefault("audio", {})
    input_cfg = audio.setdefault("input", {})
    output_cfg = audio.setdefault("output", {})
    input_cfg["format"] = deep_merge_dict(
        {"type": "audio/pcm", "rate": 24000},
        input_cfg.get("format") or {},
    )
    output_cfg["format"] = deep_merge_dict(
        {"type": "audio/pcm", "rate": 24000},
        output_cfg.get("format") or {},
    )
    output_cfg.setdefault("voice", DEFAULT_VOICE)
    output_cfg.setdefault("speed", 1.0)
    session.setdefault("tools", [])
    session.setdefault("tool_choice", "auto")
    return session


@dataclass
class ClientSecretRecord:
    value: str
    expires_at: int
    session: dict[str, Any]

    def is_expired(self) -> bool:
        return self.expires_at <= int(time.time())


class ClientSecretStore:
    def __init__(self, default_ttl_seconds: int) -> None:
        self._default_ttl_seconds = default_ttl_seconds
        self._records: dict[str, ClientSecretRecord] = {}
        self._lock = asyncio.Lock()

    async def create(
        self,
        *,
        session: dict[str, Any],
        ttl_seconds: int | None = None,
    ) -> ClientSecretRecord:
        ttl = ttl_seconds or self._default_ttl_seconds
        value = f"ek_{secrets.token_urlsafe(24)}"
        record = ClientSecretRecord(
            value=value,
            expires_at=int(time.time()) + ttl,
            session=session,
        )
        async with self._lock:
            self._purge_expired_locked()
            self._records[value] = record
        return record

    async def get(self, value: str) -> ClientSecretRecord | None:
        async with self._lock:
            self._purge_expired_locked()
            return self._records.get(value)

    def _purge_expired_locked(self) -> None:
        expired = [key for key, record in self._records.items() if record.is_expired()]
        for key in expired:
            self._records.pop(key, None)


class RealtimeDataChannelWriter:
    def __init__(self, channel) -> None:
        self._channel = channel
        self._lock = asyncio.Lock()

    async def send_event(self, event: Any) -> None:
        payload = serialize_realtime_server_event(event)
        async with self._lock:
            if self._channel.readyState != "open":
                raise RuntimeError("Realtime data channel is not open.")
            self._channel.send(payload)

    async def send_error(
        self,
        *,
        message: str,
        error_type: str = "invalid_request_error",
        code: str | None = None,
        event_ref: str | None = None,
        param: str | None = None,
    ) -> None:
        from nexus.application.realtime.protocol.server_writer import RealtimeServerWriter

        class _Proxy:
            async def send_text(self, text: str) -> None:
                if self._outer._channel.readyState != "open":
                    raise RuntimeError("Realtime data channel is not open.")
                self._outer._channel.send(text)

            def __init__(self, outer: "RealtimeDataChannelWriter") -> None:
                self._outer = outer

        proxy = _Proxy(self)
        writer = RealtimeServerWriter(proxy)
        await writer.send_error(
            message=message,
            error_type=error_type,
            code=code,
            event_ref=event_ref,
            param=param,
        )


class SessionAudioStreamTrack(MediaStreamTrack):
    kind = "audio"
    SAMPLE_RATE = 24000
    FRAME_SAMPLES = 480

    def __init__(self, session: RealtimeSessionState) -> None:
        super().__init__()
        self._session = session
        self._buffer = bytearray()
        self._pts = 0
        self._start_time: float | None = None

    async def recv(self) -> AudioFrame:
        if self.readyState != "live":
            raise MediaStreamError

        if self._start_time is None:
            self._start_time = time.monotonic()
        else:
            expected = self._start_time + (self._pts / self.SAMPLE_RATE)
            wait = expected - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)

        needed = self.FRAME_SAMPLES * 2
        while len(self._buffer) < needed:
            try:
                chunk = await asyncio.wait_for(
                    self._session.audio_output_queue.get(),
                    timeout=self.FRAME_SAMPLES / self.SAMPLE_RATE,
                )
            except asyncio.TimeoutError:
                break

            if chunk is None:
                break
            self._buffer.extend(chunk)

        if len(self._buffer) >= needed:
            frame_bytes = bytes(self._buffer[:needed])
            del self._buffer[:needed]
        elif self._buffer:
            frame_bytes = bytes(self._buffer)
            self._buffer.clear()
        else:
            frame_bytes = b"\x00" * needed

        samples = np.frombuffer(frame_bytes, dtype=np.int16)
        if samples.size < self.FRAME_SAMPLES:
            samples = np.pad(samples, (0, self.FRAME_SAMPLES - samples.size))

        frame = AudioFrame.from_ndarray(samples.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = self.SAMPLE_RATE
        frame.time_base = Fraction(1, self.SAMPLE_RATE)
        frame.pts = self._pts
        self._pts += self.FRAME_SAMPLES
        return frame


def frame_to_mono_pcm(frame: AudioFrame) -> bytes:
    data = np.asarray(frame.to_ndarray())
    channels = len(getattr(frame.layout, "channels", ()) or ())
    if channels <= 0:
        channels = 1
    is_planar = bool(getattr(frame.format, "is_planar", False))

    if data.ndim == 1:
        mono = data
    elif channels == 1:
        mono = data.reshape(-1)
    elif is_planar:
        planar = data.reshape(channels, -1).astype(np.float32)
        mono = np.mean(planar, axis=0)
    else:
        packed = data.reshape(-1)
        sample_count = min(frame.samples, packed.size // channels)
        packed = packed[: sample_count * channels].reshape(sample_count, channels).astype(np.float32)
        mono = np.mean(packed, axis=1)

    mono = np.asarray(mono)
    if np.issubdtype(mono.dtype, np.floating):
        if np.issubdtype(data.dtype, np.floating):
            mono = np.clip(mono, -1.0, 1.0) * 32767.0
        mono = np.clip(np.rint(mono), -32768, 32767).astype(np.int16)
    elif mono.dtype != np.int16:
        mono = np.clip(mono, -32768, 32767).astype(np.int16)
    return mono.tobytes()


class WebRtcCallSession:
    def __init__(
        self,
        *,
        container: AppContainer,
        call_id: str,
        session_config: dict[str, Any],
        on_close=None,
    ) -> None:
        self.container = container
        self.call_id = call_id
        self.session_config = session_config
        self.broadcaster = BroadcastRealtimeSink()
        self.peer_connection = RTCPeerConnection()
        self.session = container.realtime.create_session(
            writer=self.broadcaster,
            output_modalities=session_config.get("output_modalities", DEFAULT_OUTPUT_MODALITIES),
            tools=[],
            chat_model=session_config["model"],
            session_id=session_config["id"],
        )
        self.controller = RealtimeSessionController(
            session=self.session,
            service=container.realtime,
            model=session_config["model"],
            broadcaster=self.broadcaster,
        )
        self._audio_tasks: set[asyncio.Task] = set()
        self._close_task: asyncio.Task | None = None
        self._closed = False
        self._expiry_task: asyncio.Task | None = None
        self._on_close = on_close
        self._configure_events()

    async def start(self, sdp_offer: str) -> str:
        await self.container.realtime.apply_session_update(
            self.session,
            self.session_config,
            model=self.session_config["model"],
            reply_sink=NullRealtimeReplySink(),
            emit_event=False,
        )
        await self.controller.start()
        self.peer_connection.addTrack(SessionAudioStreamTrack(self.session))
        await self.peer_connection.setRemoteDescription(
            RTCSessionDescription(sdp=sdp_offer, type="offer")
        )
        answer = await self.peer_connection.createAnswer()
        await self.peer_connection.setLocalDescription(answer)
        return self.peer_connection.localDescription.sdp

    def start_expiry_timer(self, seconds: int) -> None:
        if self._expiry_task is None:
            self._expiry_task = asyncio.create_task(self._expire_after(seconds))

    async def attach_sideband(self, sink: RealtimeReplySink) -> None:
        await self.controller.attach_sink(sink, send_session_created=True)

    async def detach_sideband(self, sink: RealtimeReplySink) -> None:
        await self.controller.detach_sink(sink)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._expiry_task is not None:
            self._expiry_task.cancel()
        for task in list(self._audio_tasks):
            task.cancel()
        for task in list(self._audio_tasks):
            try:
                await task
            except asyncio.CancelledError:
                pass
        await self.controller.close()
        await self.peer_connection.close()
        if self._on_close is not None:
            await self._on_close(self.call_id)

    def _configure_events(self) -> None:
        @self.peer_connection.on("datachannel")
        async def on_datachannel(channel) -> None:
            if channel.label != "oai-events":
                return

            sink = RealtimeDataChannelWriter(channel)

            @channel.on("open")
            async def on_open() -> None:
                await self.controller.attach_sink(sink, send_session_created=True)

            @channel.on("close")
            async def on_close() -> None:
                await self.controller.detach_sink(sink)

            @channel.on("message")
            async def on_message(message) -> None:
                if isinstance(message, bytes):
                    message = message.decode("utf-8")
                await self.controller.enqueue_text(message, sink)

            if channel.readyState == "open":
                await self.controller.attach_sink(sink, send_session_created=True)

        @self.peer_connection.on("track")
        async def on_track(track) -> None:
            if track.kind != "audio":
                return
            task = asyncio.create_task(self._consume_audio_track(track))
            self._audio_tasks.add(task)
            task.add_done_callback(self._audio_tasks.discard)

        @self.peer_connection.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            if self.peer_connection.connectionState in {"failed", "closed", "disconnected"}:
                if self._close_task is None:
                    self._close_task = asyncio.create_task(self.close())

    async def _consume_audio_track(self, track) -> None:
        resampler: StreamingResampler | None = None
        current_rate: int | None = None
        logged_first_frame = False
        logged_downmix = False
        try:
            while True:
                frame = await track.recv()
                frame_array = np.asarray(frame.to_ndarray())
                channel_count = len(getattr(frame.layout, "channels", ()) or ())
                if channel_count <= 0:
                    channel_count = 1
                if not logged_first_frame:
                    logged_first_frame = True
                    logger.info(
                        "Inbound WebRTC audio for %s: sample_rate=%s format=%s layout=%s channels=%s ndarray_shape=%s",
                        self.call_id,
                        int(frame.sample_rate or 0),
                        getattr(frame.format, "name", "unknown"),
                        getattr(frame.layout, "name", "unknown"),
                        channel_count,
                        tuple(frame_array.shape),
                    )
                if channel_count > 1 and not logged_downmix:
                    logged_downmix = True
                    logger.info(
                        "Downmixing inbound WebRTC audio for %s: format=%s planar=%s channels=%s",
                        self.call_id,
                        getattr(frame.format, "name", "unknown"),
                        bool(getattr(frame.format, "is_planar", False)),
                        channel_count,
                    )
                pcm_bytes = frame_to_mono_pcm(frame)
                frame_rate = int(frame.sample_rate or 24000)
                if frame_rate != current_rate:
                    current_rate = frame_rate
                    if frame_rate == 24000:
                        resampler = None
                    else:
                        resampler = StreamingResampler(input_rate=frame_rate, output_rate=24000)
                if resampler is not None:
                    pcm_bytes = await resampler.aprocess(pcm_bytes)
                if not pcm_bytes:
                    continue
                await self.session.audio_queue.put(np.frombuffer(pcm_bytes, dtype=np.int16).copy())
        except MediaStreamError:
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive boundary
            logger.exception("Unhandled inbound WebRTC audio error for %s: %s", self.call_id, exc)

    async def _expire_after(self, seconds: int) -> None:
        await asyncio.sleep(seconds)
        await self.close()


class WebRtcCallRegistry:
    def __init__(self, container: AppContainer) -> None:
        self.container = container
        self._calls: dict[str, WebRtcCallSession] = {}
        self._lock = asyncio.Lock()

    async def create_call(self, *, sdp_offer: str, session_config: dict[str, Any]) -> WebRtcCallSession:
        call_id = f"rtc_{uuid.uuid4().hex}"
        call = WebRtcCallSession(
            container=self.container,
            call_id=call_id,
            session_config=session_config,
            on_close=self.remove,
        )
        async with self._lock:
            self._calls[call_id] = call
        try:
            call.start_expiry_timer(self.container.config.realtime_session_max_seconds)
            await call.start(sdp_offer)
        except Exception:
            await self.remove(call_id)
            await call.close()
            raise
        return call

    async def get(self, call_id: str) -> WebRtcCallSession | None:
        async with self._lock:
            return self._calls.get(call_id)

    async def remove(self, call_id: str) -> None:
        async with self._lock:
            self._calls.pop(call_id, None)


class RealtimeCallRegistry:
    def __init__(self, container: AppContainer) -> None:
        self.container = container
        self._webrtc = WebRtcCallRegistry(container)

    async def close(self) -> None:
        for call_id in list(self._webrtc._calls):
            call = await self._webrtc.get(call_id)
            if call is not None:
                await call.close()

    async def create_call(self, *, sdp_offer: str, session_config: dict[str, Any]) -> WebRtcCallSession:
        return await self._webrtc.create_call(sdp_offer=sdp_offer, session_config=session_config)

    async def get(self, call_id: str):
        call = await self._webrtc.get(call_id)
        if call is not None:
            return call
        return None

    async def remove(self, call_id: str) -> None:
        await self._webrtc.remove(call_id)

    async def accept_call(self, call_id: str, *, session_config: dict[str, Any]) -> None:
        del session_config
        raise RealtimeCallError(404, f"Call {call_id} not found.")

    async def reject_call(self, call_id: str, *, status_code: int = 603) -> None:
        del status_code
        raise RealtimeCallError(404, f"Call {call_id} not found.")

    async def refer_call(self, call_id: str, *, target_uri: str) -> None:
        del target_uri
        raise RealtimeCallError(404, f"Call {call_id} not found.")

    async def hangup_call(self, call_id: str) -> None:
        webrtc_call = await self._webrtc.get(call_id)
        if webrtc_call is not None:
            await webrtc_call.close()
            return
        raise RealtimeCallError(404, f"Call {call_id} not found.")


class RealtimeApiRuntime:
    def __init__(self, container: AppContainer) -> None:
        self.container = container
        self.client_secrets = ClientSecretStore(
            default_ttl_seconds=container.config.realtime_client_secret_ttl_seconds,
        )
        self.calls = RealtimeCallRegistry(container)

    def api_key_required(self) -> bool:
        return bool(self.container.config.realtime_api_key)

    def check_api_key(self, bearer_token: str | None) -> bool:
        return bool(
            bearer_token
            and self.container.config.realtime_api_key
            and secrets.compare_digest(bearer_token, self.container.config.realtime_api_key)
        )


_runtime_lock = asyncio.Lock()
_runtime: RealtimeApiRuntime | None = None
_runtime_container_id: int | None = None


async def get_realtime_api_runtime(container: AppContainer) -> RealtimeApiRuntime:
    global _runtime, _runtime_container_id
    async with _runtime_lock:
        if _runtime is None or _runtime_container_id != id(container):
            _runtime = RealtimeApiRuntime(container)
            _runtime_container_id = id(container)
        return _runtime

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import struct
import time
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any, Callable

import g711
import numpy as np
import websockets.asyncio.client

from nexus.application.container import AppContainer
from nexus.application.realtime.protocol import (
    BroadcastRealtimeSink,
    NullRealtimeReplySink,
    RealtimeReplySink,
)
from nexus.infrastructure.audio import StreamingResampler

from .controller import RealtimeSessionController


logger = logging.getLogger(__name__)

RTP_HEADER_LEN = 12
PCMU_PAYLOAD_TYPE = 0
PCMA_PAYLOAD_TYPE = 8
ASTERISK_SAMPLE_RATE = 8000
REALTIME_SAMPLE_RATE = 24000
FRAME_MS = 20
G711_BYTES_PER_FRAME = ASTERISK_SAMPLE_RATE * FRAME_MS // 1000


class AsteriskCallError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class AsteriskIngressConfig:
    enabled: bool
    ari_url: str
    ari_user: str
    ari_password: str
    stasis_app: str
    external_host: str
    rtp_port_start: int
    rtp_port_end: int
    codec: str
    realtime_webhook_url: str | None
    realtime_webhook_secret: str | None
    refer_endpoint_prefix: str | None

    @classmethod
    def from_container(cls, container: AppContainer) -> "AsteriskIngressConfig":
        config = container.config
        return cls(
            enabled=config.asterisk_ingress_enabled,
            ari_url=config.asterisk_ari_url,
            ari_user=config.asterisk_ari_user,
            ari_password=config.asterisk_ari_password,
            stasis_app=config.asterisk_stasis_app,
            external_host=config.asterisk_external_host,
            rtp_port_start=config.asterisk_rtp_port_start,
            rtp_port_end=config.asterisk_rtp_port_end,
            codec=config.asterisk_codec,
            realtime_webhook_url=config.realtime_webhook_url,
            realtime_webhook_secret=config.realtime_webhook_secret,
            refer_endpoint_prefix=config.asterisk_refer_endpoint_prefix,
        )

    @property
    def events_ws_url(self) -> str:
        parsed = urllib.parse.urlparse(self.ari_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        path = parsed.path.rstrip("/")
        api_key = urllib.parse.quote(f"{self.ari_user}:{self.ari_password}")
        app = urllib.parse.quote(self.stasis_app)
        return f"{scheme}://{parsed.netloc}{path}/events?app={app}&api_key={api_key}"

    @property
    def safe_events_ws_url(self) -> str:
        parsed = urllib.parse.urlparse(self.ari_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        path = parsed.path.rstrip("/")
        app = urllib.parse.quote(self.stasis_app)
        return f"{scheme}://{parsed.netloc}{path}/events?app={app}&api_key=<redacted>"


@dataclass
class PendingAsteriskCall:
    call_id: str
    caller_channel_id: str
    caller_channel_name: str
    created_at: float
    sip_headers: list[dict[str, str]]
    state: str = "pending"


class AriClient:
    def __init__(self, config: AsteriskIngressConfig) -> None:
        self.config = config

    async def post(self, path: str, **params: str) -> dict[str, Any]:
        result = await asyncio.to_thread(self._request, path, "POST", b"", params)
        return result if isinstance(result, dict) else {}

    async def get(self, path: str, **params: str) -> Any:
        return await asyncio.to_thread(self._request, path, "GET", None, params)

    async def delete(self, path: str, **params: str) -> None:
        try:
            await asyncio.to_thread(self._request, path, "DELETE", None, params)
        except Exception:
            logger.debug("Ignoring ARI delete failure for %s", path, exc_info=True)

    def _request(
        self,
        path: str,
        method: str,
        data: bytes | None,
        params: dict[str, str],
    ) -> Any:
        url = f"{self.config.ari_url.rstrip('/')}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(url, data=data, method=method)
        token = base64.b64encode(
            f"{self.config.ari_user}:{self.config.ari_password}".encode("utf-8")
        ).decode("ascii")
        request.add_header("Authorization", f"Basic {token}")
        with urllib.request.urlopen(request, timeout=8) as response:
            body = response.read()
        if not body:
            return {}
        return json.loads(body.decode("utf-8"))


class RtpPortPool:
    def __init__(self, start: int, end: int) -> None:
        self._available = set(range(start, end + 1))
        self._in_use: set[int] = set()
        self._lock = asyncio.Lock()

    async def acquire(self) -> int:
        async with self._lock:
            if not self._available:
                raise AsteriskCallError(503, "No RTP ports available.")
            port = min(self._available)
            self._available.remove(port)
            self._in_use.add(port)
            return port

    async def release(self, port: int | None) -> None:
        if port is None:
            return
        async with self._lock:
            if port in self._in_use:
                self._in_use.remove(port)
                self._available.add(port)


def pcm16_bytes_to_float32(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def float32_to_pcm16_bytes(samples: np.ndarray) -> bytes:
    return np.clip(samples * 32768.0, -32768, 32767).astype(np.int16).tobytes()


class G711PcmTranscoder:
    def __init__(self, codec: str) -> None:
        self.codec = codec
        self._to_realtime = StreamingResampler(ASTERISK_SAMPLE_RATE, REALTIME_SAMPLE_RATE)
        self._to_rtp = StreamingResampler(REALTIME_SAMPLE_RATE, ASTERISK_SAMPLE_RATE)

    def rtp_payload_to_realtime_pcm(self, payload: bytes) -> bytes:
        samples = g711.decode_alaw(payload) if self.codec == "alaw" else g711.decode_ulaw(payload)
        return self._to_realtime.process(float32_to_pcm16_bytes(samples))

    def realtime_pcm_to_rtp_payload(self, pcm24: bytes) -> bytes:
        pcm8 = self._to_rtp.process(pcm24)
        if not pcm8:
            return b""
        samples = pcm16_bytes_to_float32(pcm8)
        return g711.encode_alaw(samples) if self.codec == "alaw" else g711.encode_ulaw(samples)


class AsteriskRtpEndpoint(asyncio.DatagramProtocol):
    def __init__(self, *, session, codec: str) -> None:
        self.session = session
        self.codec = codec
        self.payload_type = PCMA_PAYLOAD_TYPE if codec == "alaw" else PCMU_PAYLOAD_TYPE
        self.transcoder = G711PcmTranscoder(codec)
        self.transport: asyncio.DatagramTransport | None = None
        self.remote_addr: tuple[str, int] | None = None
        self.seq = 0
        self.timestamp = 0
        self.ssrc = int(time.time()) & 0xFFFFFFFF
        self._output_task: asyncio.Task | None = None
        self._closed = asyncio.Event()

    async def start(self, *, bind_host: str, bind_port: int) -> None:
        loop = asyncio.get_running_loop()
        transport, _ = await loop.create_datagram_endpoint(
            lambda: self,
            local_addr=(bind_host, bind_port),
        )
        self.transport = transport
        self._output_task = asyncio.create_task(self._send_output_loop())

    async def close(self) -> None:
        self._closed.set()
        if self._output_task is not None:
            self._output_task.cancel()
            try:
                await self._output_task
            except asyncio.CancelledError:
                pass
        if self.transport is not None:
            self.transport.close()
            self.transport = None

    def datagram_received(self, data: bytes, addr) -> None:
        if len(data) <= RTP_HEADER_LEN:
            return
        self.remote_addr = addr
        pcm24 = self.transcoder.rtp_payload_to_realtime_pcm(data[RTP_HEADER_LEN:])
        if pcm24:
            self.session.audio_queue.put_nowait(np.frombuffer(pcm24, dtype=np.int16).copy())

    async def _send_output_loop(self) -> None:
        pending = bytearray()
        next_send_at = time.monotonic()
        while not self._closed.is_set():
            if len(pending) < G711_BYTES_PER_FRAME:
                chunk = await self.session.audio_output_queue.get()
                if chunk is None:
                    return
                pending.extend(self.transcoder.realtime_pcm_to_rtp_payload(chunk))
                if not pending:
                    continue

            while len(pending) >= G711_BYTES_PER_FRAME and not self._closed.is_set():
                if self.remote_addr is None or self.transport is None:
                    await asyncio.sleep(FRAME_MS / 1000)
                    next_send_at = time.monotonic()
                    break
                frame = bytes(pending[:G711_BYTES_PER_FRAME])
                del pending[:G711_BYTES_PER_FRAME]
                self._send_rtp_payload(frame)
                next_send_at += FRAME_MS / 1000
                delay = next_send_at - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                else:
                    next_send_at = time.monotonic()

    def _send_rtp_payload(self, payload: bytes) -> None:
        if self.transport is None or self.remote_addr is None:
            return
        header = struct.pack(
            "!BBHII",
            0x80,
            self.payload_type,
            self.seq,
            self.timestamp,
            self.ssrc,
        )
        self.transport.sendto(header + payload, self.remote_addr)
        self.seq = (self.seq + 1) & 0xFFFF
        self.timestamp = (self.timestamp + len(payload)) & 0xFFFFFFFF


class AsteriskCallSession:
    def __init__(
        self,
        *,
        container: AppContainer,
        config: AsteriskIngressConfig,
        ari: AriClient,
        pending: PendingAsteriskCall,
        rtp_port: int,
        on_close: Callable[[str, int | None], Any],
    ) -> None:
        self.container = container
        self.config = config
        self.ari = ari
        self.call_id = pending.call_id
        self.pending = pending
        self.rtp_port = rtp_port
        self.on_close = on_close
        self.broadcaster = BroadcastRealtimeSink()
        self.session = None
        self.controller: RealtimeSessionController | None = None
        self.rtp: AsteriskRtpEndpoint | None = None
        self.bridge_id: str | None = None
        self.external_channel_id: str | None = None
        self._closed = False

    async def accept(self, *, session_config: dict[str, Any]) -> None:
        if self.pending.state != "pending":
            raise AsteriskCallError(409, f"Call {self.call_id} is already {self.pending.state}.")

        try:
            self.pending.state = "accepted"
            self.session = self.container.realtime.create_session(
                writer=self.broadcaster,
                output_modalities=session_config.get("output_modalities", ["audio"]),
                tools=[],
                response_model=session_config["model"],
                session_id=session_config["id"],
            )
            self.controller = RealtimeSessionController(
                session=self.session,
                service=self.container.realtime,
                model=session_config["model"],
                broadcaster=self.broadcaster,
            )
            await self.container.realtime.apply_session_update(
                self.session,
                session_config,
                model=session_config["model"],
                reply_sink=NullRealtimeReplySink(),
                emit_event=False,
            )
            await self.controller.start()

            self.rtp = AsteriskRtpEndpoint(session=self.session, codec=self.config.codec)
            await self.rtp.start(bind_host="0.0.0.0", bind_port=self.rtp_port)
            await self.ari.post(f"/channels/{self.pending.caller_channel_id}/answer")
            bridge = await self.ari.post("/bridges", type="mixing")
            self.bridge_id = str(bridge["id"])
            external = await self.ari.post(
                "/channels/externalMedia",
                app=self.config.stasis_app,
                external_host=f"{self.config.external_host}:{self.rtp_port}",
                format=self.config.codec,
                direction="both",
            )
            self.external_channel_id = str(external["id"])
            await self.ari.post(f"/bridges/{self.bridge_id}/addChannel", channel=self.pending.caller_channel_id)
            await self.ari.post(f"/bridges/{self.bridge_id}/addChannel", channel=self.external_channel_id)

            logger.info("Accepted Asterisk call %s on RTP port %s", self.call_id, self.rtp_port)
        except Exception:
            await self.close(hangup_caller=True)
            raise

    async def reject(self, *, status_code: int = 603) -> None:
        if self.pending.state != "pending":
            raise AsteriskCallError(409, f"Call {self.call_id} is already {self.pending.state}.")
        self.pending.state = "rejected"
        await self.ari.delete(
            f"/channels/{self.pending.caller_channel_id}",
            reason=asterisk_hangup_reason(status_code),
        )
        await self.close()

    async def refer(self, *, target_uri: str) -> None:
        if not self.config.refer_endpoint_prefix:
            raise AsteriskCallError(501, "Asterisk REFER is not configured.")
        endpoint = f"{self.config.refer_endpoint_prefix}{target_uri}"
        await self.ari.post(f"/channels/{self.pending.caller_channel_id}/redirect", endpoint=endpoint)

    async def hangup(self) -> None:
        await self.ari.delete(f"/channels/{self.pending.caller_channel_id}")
        await self.close()

    async def attach_sideband(self, sink: RealtimeReplySink) -> None:
        if self.controller is None:
            raise AsteriskCallError(409, f"Call {self.call_id} is not accepted.")
        await self.controller.attach_sink(sink, send_session_created=True)

    async def detach_sideband(self, sink: RealtimeReplySink) -> None:
        if self.controller is not None:
            await self.controller.detach_sink(sink)

    async def close(self, *, hangup_caller: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        self.pending.state = "ended"
        if hangup_caller:
            await self.ari.delete(f"/channels/{self.pending.caller_channel_id}")
        if self.rtp is not None:
            await self.rtp.close()
        if self.controller is not None:
            await self.controller.close()
        if self.external_channel_id:
            await self.ari.delete(f"/channels/{self.external_channel_id}")
        if self.bridge_id:
            await self.ari.delete(f"/bridges/{self.bridge_id}")
        await self.on_close(self.call_id, self.rtp_port)

    def owns_channel(self, channel_id: str) -> bool:
        return channel_id in {self.pending.caller_channel_id, self.external_channel_id}


def sign_webhook_payload(secret: str, webhook_id: str, timestamp: int, payload: str) -> str:
    if secret.startswith("whsec_"):
        key = base64.b64decode(secret[6:])
    else:
        key = secret.encode("utf-8")
    signed_payload = f"{webhook_id}.{timestamp}.{payload}".encode("utf-8")
    signature = base64.b64encode(hmac.new(key, signed_payload, hashlib.sha256).digest()).decode("ascii")
    return f"v1,{signature}"


class AsteriskCallRegistry:
    def __init__(self, container: AppContainer) -> None:
        self.container = container
        self.config = AsteriskIngressConfig.from_container(container)
        self.ari = AriClient(self.config)
        self.port_pool = RtpPortPool(self.config.rtp_port_start, self.config.rtp_port_end)
        self._calls: dict[str, AsteriskCallSession] = {}
        self._channel_to_call_id: dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._events_task: asyncio.Task | None = None

    async def start(self) -> None:
        if not self.config.enabled or self._events_task is not None:
            return
        self._events_task = asyncio.create_task(self._event_loop())

    async def close(self) -> None:
        if self._events_task is not None:
            self._events_task.cancel()
            try:
                await self._events_task
            except asyncio.CancelledError:
                pass
            self._events_task = None
        for call in list(self._calls.values()):
            await call.close(hangup_caller=True)

    async def get(self, call_id: str) -> AsteriskCallSession | None:
        async with self._lock:
            return self._calls.get(call_id)

    async def accept_call(self, call_id: str, *, session_config: dict[str, Any]) -> None:
        call = await self.get(call_id)
        if call is None:
            raise AsteriskCallError(404, f"Call {call_id} not found.")
        await call.accept(session_config=session_config)
        if call.external_channel_id is not None:
            async with self._lock:
                self._channel_to_call_id[call.external_channel_id] = call_id

    async def reject_call(self, call_id: str, *, status_code: int = 603) -> None:
        call = await self.get(call_id)
        if call is None:
            raise AsteriskCallError(404, f"Call {call_id} not found.")
        await call.reject(status_code=status_code)

    async def refer_call(self, call_id: str, *, target_uri: str) -> None:
        call = await self.get(call_id)
        if call is None:
            raise AsteriskCallError(404, f"Call {call_id} not found.")
        await call.refer(target_uri=target_uri)

    async def hangup_call(self, call_id: str) -> None:
        call = await self.get(call_id)
        if call is None:
            raise AsteriskCallError(404, f"Call {call_id} not found.")
        await call.hangup()

    async def remove(self, call_id: str, rtp_port: int | None = None) -> None:
        async with self._lock:
            call = self._calls.pop(call_id, None)
            if call is not None:
                self._channel_to_call_id.pop(call.pending.caller_channel_id, None)
                if call.external_channel_id is not None:
                    self._channel_to_call_id.pop(call.external_channel_id, None)
        await self.port_pool.release(rtp_port)

    async def create_pending_call(self, channel: dict[str, Any]) -> AsteriskCallSession | None:
        channel_id = str(channel.get("id") or "")
        if not channel_id or str(channel.get("name") or "").startswith("UnicastRTP/"):
            return None

        async with self._lock:
            existing_id = self._channel_to_call_id.get(channel_id)
            if existing_id:
                return self._calls.get(existing_id)

        port = await self.port_pool.acquire()
        call_id = f"sip_{uuid.uuid4().hex}"
        pending = PendingAsteriskCall(
            call_id=call_id,
            caller_channel_id=channel_id,
            caller_channel_name=str(channel.get("name") or ""),
            created_at=time.time(),
            sip_headers=sip_headers_from_channel(channel),
        )
        call = AsteriskCallSession(
            container=self.container,
            config=self.config,
            ari=self.ari,
            pending=pending,
            rtp_port=port,
            on_close=self.remove,
        )
        async with self._lock:
            self._calls[call_id] = call
            self._channel_to_call_id[channel_id] = call_id

        logger.info("Asterisk incoming call pending: call_id=%s channel=%s", call_id, channel_id)
        await self.emit_incoming_webhook(pending)
        return call

    async def handle_channel_end(self, channel_id: str) -> None:
        async with self._lock:
            call_id = self._channel_to_call_id.get(channel_id)
            call = self._calls.get(call_id) if call_id else None
        if call is not None and call.owns_channel(channel_id):
            await call.close(hangup_caller=channel_id != call.pending.caller_channel_id)

    async def emit_incoming_webhook(self, pending: PendingAsteriskCall) -> None:
        event = {
            "object": "event",
            "id": f"evt_{uuid.uuid4().hex}",
            "type": "realtime.call.incoming",
            "created_at": int(pending.created_at),
            "data": {
                "call_id": pending.call_id,
                "sip_headers": pending.sip_headers,
            },
        }
        payload = json.dumps(event, separators=(",", ":"), ensure_ascii=False)

        if not self.config.realtime_webhook_url:
            logger.info("No realtime_webhook_url configured; pending call_id=%s", pending.call_id)
            return
        if not self.config.realtime_webhook_secret:
            logger.warning("No realtime_webhook_secret configured; webhook not sent for %s", pending.call_id)
            return

        webhook_id = f"wh_{uuid.uuid4().hex}"
        timestamp = int(time.time())
        headers = {
            "content-type": "application/json",
            "webhook-id": webhook_id,
            "webhook-timestamp": str(timestamp),
            "webhook-signature": sign_webhook_payload(
                self.config.realtime_webhook_secret,
                webhook_id,
                timestamp,
                payload,
            ),
        }
        try:
            await asyncio.to_thread(
                post_json_webhook,
                self.config.realtime_webhook_url,
                payload.encode("utf-8"),
                headers,
            )
        except Exception as exc:
            logger.warning("Failed to send realtime.call.incoming webhook for %s: %s", pending.call_id, exc)

    async def _event_loop(self) -> None:
        while True:
            try:
                logger.info("Connecting Asterisk ARI events: %s", self.config.safe_events_ws_url)
                async with websockets.asyncio.client.connect(self.config.events_ws_url) as websocket:
                    async for raw in websocket:
                        event = json.loads(raw)
                        event_type = event.get("type")
                        channel = event.get("channel") or {}
                        channel_id = str(channel.get("id") or "")
                        if event_type == "StasisStart":
                            await self.create_pending_call(channel)
                        elif event_type in {"StasisEnd", "ChannelDestroyed"} and channel_id:
                            await self.handle_channel_end(channel_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Asterisk ARI event loop failed: %s", exc)
                await asyncio.sleep(3)


def post_json_webhook(url: str, payload: bytes, headers: dict[str, str]) -> None:
    request = urllib.request.Request(url, data=payload, method="POST")
    for key, value in headers.items():
        request.add_header(key, value)
    with urllib.request.urlopen(request, timeout=8) as response:
        response.read()


def asterisk_hangup_reason(status_code: int) -> str:
    if status_code == 486:
        return "busy"
    if status_code in {480, 487}:
        return "no_answer"
    if status_code in {500, 503}:
        return "congestion"
    return "rejected"


def sip_headers_from_channel(channel: dict[str, Any]) -> list[dict[str, str]]:
    caller = channel.get("caller") or {}
    dialplan = channel.get("dialplan") or {}
    headers = [
        {"name": "From", "value": str(caller.get("number") or caller.get("name") or "")},
        {"name": "To", "value": str(dialplan.get("exten") or "")},
        {"name": "Call-ID", "value": str(channel.get("id") or "")},
    ]
    return [header for header in headers if header["value"]]

#!/usr/bin/env python3
"""用于 Asterisk ARI ExternalMedia 的 CLI 音频端点。

运行方式：

    poetry run python tests/test_sip/cli_external_media_demo.py

这是一个手动演示脚本，不是自动化 pytest 用例。它会让本地
麦克风/扬声器像一个简单的媒体端点一样接入 Asterisk Stasis
应用。请先运行此脚本，再发起呼叫；本地 Asterisk 实例应将
呼叫路由到 Stasis(voicebot)。
"""

from __future__ import annotations

import argparse
import audioop
import base64
import json
import queue
import socket
import struct
import sys
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

import numpy as np
import sounddevice as sd
import websockets.sync.client


RTP_HEADER_LEN = 12
PCMU_PAYLOAD_TYPE = 0
PCMA_PAYLOAD_TYPE = 8
SAMPLE_RATE = 8000
FRAME_MS = 20
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1000


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

    def post(self, path: str, **params: str) -> dict[str, Any]:
        url = f"{self.config.base_url.rstrip('/')}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(url, data=b"", method="POST")
        token = base64.b64encode(
            f"{self.config.username}:{self.config.password}".encode("utf-8")
        ).decode("ascii")
        request.add_header("Authorization", f"Basic {token}")
        with urllib.request.urlopen(request, timeout=8) as response:
            body = response.read()
        if not body:
            return {}
        return json.loads(body.decode("utf-8"))

    def get(self, path: str, **params: str) -> Any:
        url = f"{self.config.base_url.rstrip('/')}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(url, method="GET")
        token = base64.b64encode(
            f"{self.config.username}:{self.config.password}".encode("utf-8")
        ).decode("ascii")
        request.add_header("Authorization", f"Basic {token}")
        with urllib.request.urlopen(request, timeout=8) as response:
            body = response.read()
        if not body:
            return {}
        return json.loads(body.decode("utf-8"))

    def delete(self, path: str, **params: str) -> None:
        url = f"{self.config.base_url.rstrip('/')}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(url, method="DELETE")
        token = base64.b64encode(
            f"{self.config.username}:{self.config.password}".encode("utf-8")
        ).decode("ascii")
        request.add_header("Authorization", f"Basic {token}")
        try:
            urllib.request.urlopen(request, timeout=5).read()
        except Exception:
            pass


class RtpAudioEndpoint:
    def __init__(
        self,
        *,
        bind_host: str,
        bind_port: int,
        codec: str,
        input_device: str | int | None,
        output_device: str | int | None,
    ) -> None:
        self.bind_host = bind_host
        self.bind_port = bind_port
        self.codec = codec
        self.payload_type = PCMA_PAYLOAD_TYPE if codec == "alaw" else PCMU_PAYLOAD_TYPE
        self.input_device = input_device
        self.output_device = output_device
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((bind_host, bind_port))
        self.remote_addr: tuple[str, int] | None = None
        self.seq = 0
        self.timestamp = 0
        self.ssrc = int(time.time()) & 0xFFFFFFFF
        self.running = threading.Event()
        self.running.set()
        self.session_active = threading.Event()
        self.playback_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
        self.pending_playback: np.ndarray | None = None

    def start(self) -> None:
        threading.Thread(target=self._recv_loop, daemon=True).start()
        self._start_audio_streams()

    def stop(self) -> None:
        self.running.clear()
        self.sock.close()

    def reset_session(self) -> None:
        self.session_active.clear()
        self.remote_addr = None
        self.pending_playback = None
        while True:
            try:
                self.playback_queue.get_nowait()
            except queue.Empty:
                break

    def start_session(self) -> None:
        self.reset_session()
        self.session_active.set()

    def _decode_payload(self, payload: bytes) -> bytes:
        if self.codec == "alaw":
            return audioop.alaw2lin(payload, 2)
        return audioop.ulaw2lin(payload, 2)

    def _encode_payload(self, pcm16: bytes) -> bytes:
        if self.codec == "alaw":
            return audioop.lin2alaw(pcm16, 2)
        return audioop.lin2ulaw(pcm16, 2)

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
            payload = packet[RTP_HEADER_LEN:]
            pcm_bytes = self._decode_payload(payload)
            pcm = np.frombuffer(pcm_bytes, dtype=np.int16).reshape(-1, 1).copy()
            self._queue_playback(pcm)

    def _queue_playback(self, pcm: np.ndarray) -> None:
        if self.playback_queue.full():
            try:
                self.playback_queue.get_nowait()
            except queue.Empty:
                pass
        self.playback_queue.put_nowait(pcm)

    def _start_audio_streams(self) -> None:
        def input_callback(indata, frames, time_info, status) -> None:
            del frames, time_info
            if status:
                print(f"输入状态：{status}")
            self.send_pcm(np.asarray(indata[:, 0], dtype=np.int16).copy())

        def output_callback(outdata, frames, time_info, status) -> None:
            del time_info
            if status:
                print(f"输出状态：{status}")
            self._fill_output(outdata, frames)

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=SAMPLES_PER_FRAME,
            channels=1,
            dtype="int16",
            device=self.input_device,
            callback=input_callback,
        ), sd.OutputStream(
            samplerate=SAMPLE_RATE,
            blocksize=SAMPLES_PER_FRAME,
            channels=1,
            dtype="int16",
            device=self.output_device,
            callback=output_callback,
        ):
            print(
                "已就绪。请向 Asterisk Stasis(voicebot) 发起呼叫。"
                "按 Ctrl+C 停止。"
            )
            while self.running.is_set():
                time.sleep(0.2)

    def _fill_output(self, outdata, frames: int) -> None:
        outdata.fill(0)
        offset = 0
        remaining = frames
        while remaining > 0:
            chunk = self.pending_playback
            if chunk is None:
                try:
                    chunk = self.playback_queue.get_nowait()
                except queue.Empty:
                    break
            take = min(remaining, len(chunk))
            outdata[offset : offset + take, 0] = chunk[:take, 0]
            offset += take
            remaining -= take
            if take < len(chunk):
                self.pending_playback = chunk[take:].copy()
                break
            self.pending_playback = None

    def send_pcm(self, pcm: np.ndarray) -> None:
        if not self.session_active.is_set():
            return
        if self.remote_addr is None:
            return
        pcm_bytes = pcm.astype(np.int16, copy=False).tobytes()
        payload = self._encode_payload(pcm_bytes)
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
            print(f"已附加到通道 {self.call_channel_id}；忽略 {channel_id}")
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
            "已将呼叫附加到 CLI 媒体："
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
        print(f"通道已结束，正在清理媒体：{channel_id}")
        self.cleanup()
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将本地麦克风/扬声器用作 ARI ExternalMedia RTP 端点。"
    )
    parser.add_argument("--ari-url", default="http://127.0.0.1:8088/ari")
    parser.add_argument("--ari-user", default="voicebot")
    parser.add_argument("--ari-password", default="12345678")
    parser.add_argument("--app", default="voicebot")
    parser.add_argument("--rtp-host", default="127.0.0.1")
    parser.add_argument("--rtp-port", type=int, default=4000)
    parser.add_argument("--codec", choices=("ulaw", "alaw"), default="ulaw")
    parser.add_argument("--input-device", default=None)
    parser.add_argument("--output-device", default=None)
    parser.add_argument(
        "--hangup-existing-calls",
        action="store_true",
        help="启动时挂断已在同一 Stasis app 中的旧呼叫，然后等待新呼叫。",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="打印 sounddevice 设备列表并退出。",
    )
    return parser.parse_args()


def parse_device(value: str | None) -> str | int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return value


def check_ari_ready(ari: AriClient) -> None:
    try:
        info = ari.get("/asterisk/info")
    except Exception as exc:
        raise RuntimeError(
            "无法连接到 Asterisk ARI。请检查 http.conf 是否启用了 "
            "127.0.0.1:8088，且 ari.conf 中是否存在已配置的用户。"
        ) from exc
    version = ((info.get("system") or {}).get("version")) or "unknown"
    print(f"ARI 已连接。Asterisk 版本：{version}")


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


def main() -> None:
    args = parse_args()
    if args.list_devices:
        print(sd.query_devices())
        return

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
        print(f"启动失败：{exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if args.hangup_existing_calls:
        existing_channels = find_app_call_channels(ari, args.app)
        for channel in existing_channels:
            channel_id = channel.get("id")
            if not channel_id:
                continue
            print(f"挂断旧 Stasis 呼叫：{channel_label(channel)}")
            ari.delete(f"/channels/{channel_id}")
        if existing_channels:
            time.sleep(0.5)

    rtp = RtpAudioEndpoint(
        bind_host=args.rtp_host,
        bind_port=args.rtp_port,
        codec=args.codec,
        input_device=parse_device(args.input_device),
        output_device=parse_device(args.output_device),
    )
    bridge = ExternalMediaBridge(
        ari=ari,
        app=args.app,
        external_host=f"{args.rtp_host}:{args.rtp_port}",
        codec=args.codec,
    )
    event_error: queue.Queue[BaseException] = queue.Queue(maxsize=1)

    def event_loop() -> None:
        try:
            print(f"正在连接 ARI 事件：{config.events_ws_url}")
            with websockets.sync.client.connect(config.events_ws_url) as ws:
                for channel in find_app_call_channels(ari, args.app):
                    channel_id = channel.get("id")
                    if not channel_id:
                        continue
                    print(f"发现已有 Stasis 呼叫：{channel_label(channel)}")
                    if bridge.attach_call(str(channel_id)):
                        rtp.start_session()
                        break

                print(f"正在等待 StasisStart(app={args.app})...")
                for raw in ws:
                    event = json.loads(raw)
                    event_type = event.get("type")
                    channel = event.get("channel") or {}
                    channel_id = channel.get("id")
                    channel_name = channel.get("name", "")

                    if event_type in {"StasisEnd", "ChannelDestroyed"}:
                        if channel_id and bridge.detach_if_related(channel_id):
                            rtp.reset_session()
                        continue

                    if event_type != "StasisStart":
                        continue
                    if not channel_id:
                        continue
                    if channel_id == bridge.external_channel_id:
                        continue
                    print(f"StasisStart: {channel_id} {channel_name}")
                    if bridge.attach_call(channel_id):
                        rtp.start_session()
        except BaseException as exc:
            try:
                event_error.put_nowait(exc)
            except queue.Full:
                pass
            rtp.running.clear()

    threading.Thread(target=event_loop, daemon=True).start()
    try:
        rtp.start()
    except KeyboardInterrupt:
        print("\n正在停止...")
    finally:
        rtp.stop()
        bridge.cleanup()
        if not event_error.empty():
            exc = event_error.get_nowait()
            print(f"ARI 事件循环已停止：{exc}", file=sys.stderr)
            raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

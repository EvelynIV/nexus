from __future__ import annotations

import asyncio
import json
from typing import Awaitable, Callable

import httpx
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole

from .audio_io import MicrophoneAudioTrack, SpeakerPlayer
from .config import RealtimeTuiConfig
from .events import RealtimeEventProcessor
from .models import ConnectionPhase, ConversationUpdate, SessionRuntimeState


ChatCallback = Callable[[ConversationUpdate], Awaitable[None]]
TextCallback = Callable[[str], Awaitable[None]]


class CallCreateError(RuntimeError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class RealtimeWebRtcClient:
    def __init__(
        self,
        *,
        config: RealtimeTuiConfig,
        on_chat: ChatCallback,
        on_error: TextCallback,
        on_status: TextCallback,
    ) -> None:
        self._config = config
        self._on_chat = on_chat
        self._on_error = on_error
        self._on_status = on_status
        self._runtime = SessionRuntimeState()
        self._event_processor = RealtimeEventProcessor()
        self._pc: RTCPeerConnection | None = None
        self._dc = None
        self._mic: MicrophoneAudioTrack | None = None
        self._speaker: SpeakerPlayer | None = None
        self._speaker_task: asyncio.Task | None = None
        self._http: httpx.AsyncClient | None = None
        self._closing = False
        self._sink = MediaBlackhole()

    @property
    def connected(self) -> bool:
        return self._pc is not None and self._pc.connectionState not in {"closed", "failed"}

    @property
    def muted(self) -> bool:
        return self._runtime.manual_mute

    async def connect(self) -> None:
        if self.connected:
            return

        await self._close_resources(reset_status=False)
        self._event_processor.reset()
        self._runtime.phase = ConnectionPhase.CONNECTING
        self._runtime.call_id = None
        self._runtime.data_channel_open = False
        await self._publish_status("正在建立 WebRTC 连接…")

        await self._create_runtime_resources()
        offer_sdp = await self._create_offer()

        try:
            answer_sdp, call_id = await self._create_call(offer_sdp)
            self._runtime.call_id = call_id
            await self._pc.setRemoteDescription(RTCSessionDescription(sdp=answer_sdp, type="answer"))
            self._runtime.phase = ConnectionPhase.CONNECTED
            await self._publish_status("已连接")
        except Exception:
            await self._close_resources(reset_status=False)
            raise

    async def disconnect(self) -> None:
        await self._close_resources(reset_status=True)

    async def toggle_mute(self) -> None:
        self._runtime.manual_mute = not self._runtime.manual_mute
        self._apply_microphone_state()
        if self.connected:
            await self._publish_status("已连接")

    async def _create_runtime_resources(self) -> None:
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0))
        self._pc = RTCPeerConnection()
        self._speaker = SpeakerPlayer(
            device=self._config.output_device,
            on_activity_change=self._handle_remote_audio_activity,
        )
        self._mic = MicrophoneAudioTrack(device=self._config.input_device)
        self._apply_microphone_state()
        self._mic.start()
        self._pc.addTrack(self._mic)
        self._configure_peer_connection(self._pc)
        self._dc = self._pc.createDataChannel("oai-events")
        self._configure_data_channel(self._dc)

    async def _create_offer(self) -> str:
        if self._pc is None:
            raise RuntimeError("WebRTC peer connection 尚未创建。")
        offer = await self._pc.createOffer()
        await self._pc.setLocalDescription(offer)
        offer_sdp = offer.sdp or self._pc.localDescription.sdp
        if not offer_sdp:
            raise RuntimeError("WebRTC offer SDP 为空。")
        return offer_sdp

    def _configure_peer_connection(self, pc: RTCPeerConnection) -> None:
        @pc.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            state = pc.connectionState
            if state == "connected":
                self._runtime.phase = ConnectionPhase.CONNECTED
                await self._publish_status("WebRTC 已连接")
            elif state in {"failed", "disconnected"} and not self._closing:
                await self._on_error(f"连接状态异常：{state}")
                await self._close_resources(reset_status=True)
            elif state == "closed":
                self._runtime.phase = ConnectionPhase.IDLE
                await self._publish_status("未连接")

        @pc.on("track")
        def on_track(track) -> None:
            if track.kind != "audio":
                self._sink.addTrack(track)
                return
            if self._speaker_task is not None:
                self._speaker_task.cancel()
            self._speaker_task = asyncio.create_task(self._play_remote_track(track))

    def _configure_data_channel(self, dc) -> None:
        @dc.on("open")
        def on_open() -> None:
            self._runtime.data_channel_open = True
            asyncio.create_task(self._publish_status("数据通道已打开"))

        @dc.on("close")
        def on_close() -> None:
            self._runtime.data_channel_open = False
            if not self._closing:
                asyncio.create_task(self._publish_status("数据通道已关闭"))

        @dc.on("message")
        def on_message(message) -> None:
            asyncio.create_task(self._handle_message(message))

    async def _create_call(self, offer_sdp: str) -> tuple[str, str | None]:
        session = self._build_session_config(include_transcription=True)
        try:
            return await self._post_call(offer_sdp, session)
        except CallCreateError as exc:
            if exc.status_code not in {400, 422}:
                raise RuntimeError(f"创建 WebRTC 会话失败：{exc.detail}") from exc
            await self._publish_status("后端拒绝输入转写配置，重试无转写模式…")
            session = self._build_session_config(include_transcription=False)
            try:
                return await self._post_call(offer_sdp, session)
            except CallCreateError as retry_exc:
                raise RuntimeError(f"创建 WebRTC 会话失败：{retry_exc.detail}") from retry_exc

    async def _post_call(self, offer_sdp: str, session: dict) -> tuple[str, str | None]:
        if self._http is None:
            raise RuntimeError("HTTP 客户端尚未初始化。")
        response = await self._http.post(
            self._config.calls_url,
            headers={"Authorization": f"Bearer {self._config.api_key}"},
            data={"session": json.dumps(session)},
            files={"sdp": ("offer.sdp", offer_sdp, "application/sdp")},
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text.strip() or exc.response.reason_phrase
            raise CallCreateError(exc.response.status_code, detail) from exc
        return response.text, response.headers.get("location")

    def _build_session_config(self, *, include_transcription: bool) -> dict:
        session = {
            "type": "realtime",
            "model": self._config.model,
            "output_modalities": ["audio", "text"],
            "audio": {
                "input": {
                    "turn_detection": {
                        "type": "server_vad",
                        "create_response": True,
                        "interrupt_response": True,
                    }
                },
                "output": {
                    "voice": self._config.voice,
                },
            },
        }
        if include_transcription:
            session["audio"]["input"]["transcription"] = {"model": "gpt-4o-mini-transcribe"}
        return session

    async def _play_remote_track(self, track) -> None:
        if self._speaker is None:
            return
        try:
            await self._speaker.play_track(track)
        except Exception as exc:
            await self._on_error(f"播放远端音频失败：{exc}")

    async def _handle_message(self, message) -> None:
        if isinstance(message, bytes):
            message = message.decode("utf-8")
        try:
            event = json.loads(message)
        except json.JSONDecodeError as exc:
            await self._on_error(f"data channel JSON 解析失败：{exc}")
            return
        processed = self._event_processor.process(event)
        if processed.status_message:
            await self._publish_status(processed.status_message)
        if processed.error_message:
            await self._on_error(processed.error_message)
        for update in processed.chat_updates:
            await self._on_chat(update)

    def _apply_microphone_state(self) -> None:
        if self._mic is not None:
            self._mic.set_muted(self._runtime.manual_mute or self._runtime.playback_guard_active)

    async def _handle_remote_audio_activity(self, active: bool) -> None:
        self._runtime.playback_guard_active = active
        self._apply_microphone_state()
        if self.connected:
            headline = "助手回放中，已临时静音麦克风" if active else "助手回放结束，麦克风已恢复"
            await self._publish_status(headline)

    def _render_status(self, headline: str) -> str:
        if self._runtime.manual_mute:
            mic_text = "手动静音"
        elif self._runtime.playback_guard_active:
            mic_text = "回放保护"
        else:
            mic_text = "麦克风开启"
        dc_text = "dc=open" if self._runtime.data_channel_open else "dc=closed"
        call_text = self._runtime.call_id or "-"
        return (
            f"{headline} | model={self._config.model} | voice={self._config.voice} "
            f"| call={call_text} | {dc_text} | {mic_text}"
        )

    async def _publish_status(self, headline: str) -> None:
        await self._on_status(self._render_status(headline))

    async def _close_resources(self, *, reset_status: bool) -> None:
        self._closing = True
        if self._speaker_task is not None:
            self._speaker_task.cancel()
            try:
                await self._speaker_task
            except asyncio.CancelledError:
                pass
            self._speaker_task = None
        if self._speaker is not None:
            self._speaker.close()
            self._speaker = None
        if self._mic is not None:
            await self._mic.stop_track()
            self._mic = None
        self._runtime.playback_guard_active = False
        self._runtime.data_channel_open = False
        if self._dc is not None:
            self._dc = None
        if self._pc is not None:
            await self._pc.close()
            self._pc = None
        if self._http is not None:
            await self._http.aclose()
            self._http = None
        self._runtime.phase = ConnectionPhase.IDLE
        self._runtime.call_id = None
        self._event_processor.reset()
        if reset_status:
            await self._on_status("未连接")
        self._closing = False


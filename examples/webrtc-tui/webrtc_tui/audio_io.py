from __future__ import annotations

import asyncio
import logging
import queue
from collections.abc import Awaitable, Callable
from fractions import Fraction

import numpy as np
import sounddevice as sd
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack
from av import AudioFrame

logger = logging.getLogger(__name__)


class AudioFrameConverter:
    @staticmethod
    def frame_to_output(frame: AudioFrame) -> np.ndarray:
        data = np.asarray(frame.to_ndarray())
        channels = len(getattr(frame.layout, "channels", ()) or ())
        if channels <= 0:
            channels = 1
        is_planar = bool(getattr(frame.format, "is_planar", False))

        if data.ndim == 1:
            pcm = data.reshape(-1, 1)
        elif channels == 1:
            pcm = data.reshape(-1, 1)
        elif is_planar:
            pcm = data.reshape(channels, -1).T
        else:
            flat = data.reshape(-1)
            sample_count = min(frame.samples, flat.size // channels)
            pcm = flat[: sample_count * channels].reshape(sample_count, channels)

        if np.issubdtype(pcm.dtype, np.floating):
            pcm = np.clip(pcm, -1.0, 1.0)
            pcm = (pcm * 32767.0).astype(np.int16)
        elif pcm.dtype != np.int16:
            pcm = pcm.astype(np.int16)
        return pcm.copy()


class PlaybackBuffer:
    def __init__(self, max_chunks: int = 64) -> None:
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_chunks)
        self._pending_chunk: np.ndarray | None = None

    @property
    def queue(self) -> queue.Queue[np.ndarray]:
        return self._queue

    def push(self, pcm: np.ndarray) -> None:
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        self._queue.put_nowait(pcm)

    def fill(self, outdata: np.ndarray, frames: int) -> None:
        outdata.fill(0)
        remaining = frames
        offset = 0

        while remaining > 0:
            chunk = self._pending_chunk
            if chunk is None:
                try:
                    chunk = self._queue.get_nowait()
                except queue.Empty:
                    break

            take = min(remaining, len(chunk))
            outdata[offset : offset + take, : chunk.shape[1]] = chunk[:take]
            offset += take
            remaining -= take

            if take < len(chunk):
                self._pending_chunk = chunk[take:].copy()
                break

            self._pending_chunk = None

    def clear(self) -> None:
        self._pending_chunk = None
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break


class PlaybackActivityMonitor:
    def __init__(self, *, threshold: float = 256.0, silent_frames_to_idle: int = 12) -> None:
        self.threshold = threshold
        self.silent_frames_to_idle = silent_frames_to_idle
        self._active = False
        self._silent_frame_count = 0

    async def update(
        self,
        pcm: np.ndarray,
        on_change: Callable[[bool], Awaitable[None]] | None = None,
    ) -> None:
        if pcm.size == 0:
            return

        level = float(np.max(np.abs(pcm.astype(np.float32))))
        if level >= self.threshold:
            self._silent_frame_count = 0
            await self._set_active(True, on_change)
            return

        if self._active:
            self._silent_frame_count += 1
            if self._silent_frame_count >= self.silent_frames_to_idle:
                await self._set_active(False, on_change)

    async def reset(self, on_change: Callable[[bool], Awaitable[None]] | None = None) -> None:
        await self._set_active(False, on_change)

    async def _set_active(
        self,
        active: bool,
        on_change: Callable[[bool], Awaitable[None]] | None,
    ) -> None:
        if self._active == active:
            return
        self._active = active
        self._silent_frame_count = 0
        if on_change is not None:
            await on_change(active)


class MicrophoneAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(
        self,
        *,
        device: str | None = None,
        sample_rate: int = 48000,
        block_size: int = 960,
    ) -> None:
        super().__init__()
        self._device = device
        self._sample_rate = sample_rate
        self._block_size = block_size
        self._loop = asyncio.get_running_loop()
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=24)
        self._stream: sd.InputStream | None = None
        self._muted = False
        self._pts = 0

    @property
    def muted(self) -> bool:
        return self._muted

    def set_muted(self, muted: bool) -> None:
        self._muted = muted

    def start(self) -> None:
        if self._stream is not None:
            return

        def callback(indata, frames, time_info, status) -> None:
            del frames, time_info
            if status:
                logger.warning("Microphone input status: %s", status)
            pcm = np.asarray(indata[:, 0], dtype=np.int16).copy()
            if self._muted:
                pcm.fill(0)
            self._loop.call_soon_threadsafe(self._push_frame, pcm)

        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            blocksize=self._block_size,
            channels=1,
            dtype="int16",
            device=self._device,
            callback=callback,
        )
        self._stream.start()

    def _push_frame(self, pcm: np.ndarray) -> None:
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        self._queue.put_nowait(pcm)

    async def recv(self) -> AudioFrame:
        if self.readyState != "live":
            raise MediaStreamError

        pcm = await self._queue.get()
        frame = AudioFrame.from_ndarray(pcm.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = self._sample_rate
        frame.time_base = Fraction(1, self._sample_rate)
        frame.pts = self._pts
        self._pts += pcm.shape[0]
        return frame

    async def stop_track(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self.stop()


class SpeakerPlayer:
    ACTIVITY_THRESHOLD = 256.0
    SILENT_FRAMES_TO_IDLE = 12

    def __init__(
        self,
        *,
        device: str | None = None,
        on_activity_change: Callable[[bool], Awaitable[None]] | None = None,
    ) -> None:
        self._device = device
        self._on_activity_change = on_activity_change
        self._playback_buffer = PlaybackBuffer()
        self._activity_monitor = PlaybackActivityMonitor(
            threshold=self.ACTIVITY_THRESHOLD,
            silent_frames_to_idle=self.SILENT_FRAMES_TO_IDLE,
        )
        self._queue = self._playback_buffer.queue
        self._stream: sd.OutputStream | None = None
        self._closed = False
        self._logged_first_frame = False
        self._logged_packed_multichannel = False

    async def play_track(self, track: MediaStreamTrack) -> None:
        try:
            while not self._closed:
                frame = await track.recv()
                self._log_frame(frame)
                pcm = self._frame_to_output(frame)
                await self._activity_monitor.update(pcm, self._on_activity_change)
                if self._stream is None:
                    self._open_stream(
                        sample_rate=int(frame.sample_rate or 48000),
                        channels=pcm.shape[1],
                    )
                self._push_chunk(pcm)
        except MediaStreamError:
            await self._activity_monitor.reset(self._on_activity_change)
            return

    def close(self) -> None:
        self._closed = True
        self._playback_buffer.clear()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _open_stream(self, *, sample_rate: int, channels: int) -> None:
        def callback(outdata, frames, time_info, status) -> None:
            del time_info
            if status:
                logger.warning("Speaker output status: %s", status)
            self._fill_output_buffer(outdata, frames)

        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype="int16",
            device=self._device,
            callback=callback,
        )
        self._stream.start()

    def _push_chunk(self, pcm: np.ndarray) -> None:
        self._playback_buffer.push(pcm)

    @staticmethod
    def _frame_to_output(frame: AudioFrame) -> np.ndarray:
        return AudioFrameConverter.frame_to_output(frame)

    def _fill_output_buffer(self, outdata: np.ndarray, frames: int) -> None:
        self._playback_buffer.fill(outdata, frames)

    async def _update_activity(self, pcm: np.ndarray) -> None:
        await self._activity_monitor.update(pcm, self._on_activity_change)

    def _log_frame(self, frame: AudioFrame) -> None:
        if self._logged_first_frame and self._logged_packed_multichannel:
            return

        channels = len(getattr(frame.layout, "channels", ()) or ())
        if channels <= 0:
            channels = 1
        data = np.asarray(frame.to_ndarray())
        is_planar = bool(getattr(frame.format, "is_planar", False))

        if not self._logged_first_frame:
            self._logged_first_frame = True
            logger.info(
                "Remote WebRTC audio frame: sample_rate=%s format=%s layout=%s channels=%s ndarray_shape=%s",
                int(frame.sample_rate or 0),
                getattr(frame.format, "name", "unknown"),
                getattr(frame.layout, "name", "unknown"),
                channels,
                tuple(data.shape),
            )

        if channels > 1 and not is_planar and not self._logged_packed_multichannel:
            self._logged_packed_multichannel = True
            logger.info(
                "Remote WebRTC audio uses packed multichannel layout; reshaping interleaved audio to (samples, channels)"
            )

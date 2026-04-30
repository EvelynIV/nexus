from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame

from nexus.api.v1.realtime.runtime import WebRtcCallSession, frame_to_mono_pcm


def _pcm16_from_bytes(payload: bytes) -> np.ndarray:
    return np.frombuffer(payload, dtype=np.int16)


def test_frame_to_mono_pcm_keeps_s16_mono() -> None:
    samples = np.array([100, -200, 300, -400], dtype=np.int16)
    frame = AudioFrame.from_ndarray(samples.reshape(1, -1), format="s16", layout="mono")
    frame.sample_rate = 24000

    output = _pcm16_from_bytes(frame_to_mono_pcm(frame))

    assert np.array_equal(output, samples)


def test_frame_to_mono_pcm_downmixes_s16_packed_stereo() -> None:
    left = np.array([1000, 2000, 3000, 4000], dtype=np.int16)
    right = np.array([-1000, 0, 1000, 2000], dtype=np.int16)
    interleaved = np.empty(left.size * 2, dtype=np.int16)
    interleaved[0::2] = left
    interleaved[1::2] = right

    frame = AudioFrame.from_ndarray(interleaved.reshape(1, -1), format="s16", layout="stereo")
    frame.sample_rate = 48000

    output = _pcm16_from_bytes(frame_to_mono_pcm(frame))

    expected = np.rint((left.astype(np.float32) + right.astype(np.float32)) / 2.0).astype(np.int16)
    assert np.array_equal(output, expected)
    assert not np.array_equal(output, interleaved[: left.size])


def test_frame_to_mono_pcm_downmixes_s16p_stereo() -> None:
    planar = np.array(
        [
            [1200, -1200, 800, -800],
            [400, 400, -400, -400],
        ],
        dtype=np.int16,
    )
    frame = AudioFrame.from_ndarray(planar, format="s16p", layout="stereo")
    frame.sample_rate = 48000

    output = _pcm16_from_bytes(frame_to_mono_pcm(frame))

    expected = np.rint(np.mean(planar.astype(np.float32), axis=0)).astype(np.int16)
    assert np.array_equal(output, expected)


def test_frame_to_mono_pcm_converts_fltp_stereo_to_pcm16() -> None:
    planar = np.array(
        [
            [0.5, -0.5, 0.25, -0.25],
            [0.0, 0.5, -0.25, -1.0],
        ],
        dtype=np.float32,
    )
    frame = AudioFrame.from_ndarray(planar, format="fltp", layout="stereo")
    frame.sample_rate = 48000

    output = _pcm16_from_bytes(frame_to_mono_pcm(frame))

    expected_float = np.mean(planar, axis=0)
    expected = np.clip(np.rint(np.clip(expected_float, -1.0, 1.0) * 32767.0), -32768, 32767).astype(
        np.int16
    )
    assert np.array_equal(output, expected)


class _SingleFrameTrack:
    def __init__(self, frame: AudioFrame) -> None:
        self._frame = frame
        self._done = False

    async def recv(self) -> AudioFrame:
        if self._done:
            raise MediaStreamError
        self._done = True
        return self._frame


@pytest.mark.asyncio
async def test_consume_audio_track_resamples_packed_stereo_to_24k_mono() -> None:
    samples = 960
    left = np.full(samples, 1000, dtype=np.int16)
    right = np.full(samples, 3000, dtype=np.int16)
    interleaved = np.empty(samples * 2, dtype=np.int16)
    interleaved[0::2] = left
    interleaved[1::2] = right

    frame = AudioFrame.from_ndarray(interleaved.reshape(1, -1), format="s16", layout="stereo")
    frame.sample_rate = 48000

    call = WebRtcCallSession.__new__(WebRtcCallSession)
    call.call_id = "rtc_test"
    call.session = SimpleNamespace(audio_queue=asyncio.Queue())

    await WebRtcCallSession._consume_audio_track(call, _SingleFrameTrack(frame))

    queued = await asyncio.wait_for(call.session.audio_queue.get(), timeout=1)

    assert queued.dtype == np.int16
    assert queued.ndim == 1
    assert 400 <= queued.size < 960
    assert abs(float(np.mean(queued)) - 2000.0) < 32.0

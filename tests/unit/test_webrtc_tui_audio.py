from __future__ import annotations

from unittest.mock import AsyncMock
import sys
from pathlib import Path

import numpy as np
import pytest
from av import AudioFrame


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "webrtc-tui"))

from webrtc_tui.audio_io import SpeakerPlayer  # noqa: E402


def test_frame_to_output_keeps_s16_mono_as_single_channel() -> None:
    samples = np.arange(480, dtype=np.int16)
    frame = AudioFrame.from_ndarray(samples.reshape(1, -1), format="s16", layout="mono")

    pcm = SpeakerPlayer._frame_to_output(frame)

    assert pcm.shape == (480, 1)
    assert pcm.dtype == np.int16
    assert np.array_equal(pcm[:, 0], samples)


def test_frame_to_output_reshapes_packed_stereo_without_stretching_duration() -> None:
    left = np.arange(960, dtype=np.int16)
    right = left + 1000
    interleaved = np.empty(960 * 2, dtype=np.int16)
    interleaved[0::2] = left
    interleaved[1::2] = right
    frame = AudioFrame.from_ndarray(interleaved.reshape(1, -1), format="s16", layout="stereo")

    pcm = SpeakerPlayer._frame_to_output(frame)

    assert pcm.shape == (960, 2)
    assert np.array_equal(pcm[:, 0], left)
    assert np.array_equal(pcm[:, 1], right)
    assert pcm.shape != (1920, 1)


def test_frame_to_output_transposes_planar_stereo_to_samples_by_channels() -> None:
    planar = np.array(
        [
            [100, 200, 300, 400],
            [-100, -200, -300, -400],
        ],
        dtype=np.int16,
    )
    frame = AudioFrame.from_ndarray(planar, format="s16p", layout="stereo")

    pcm = SpeakerPlayer._frame_to_output(frame)

    assert pcm.shape == (4, 2)
    assert np.array_equal(pcm[:, 0], planar[0])
    assert np.array_equal(pcm[:, 1], planar[1])


def test_frame_to_output_converts_fltp_stereo_to_int16() -> None:
    planar = np.array(
        [
            [0.5, -0.5, 0.25, -0.25],
            [0.0, 0.5, -0.25, -1.0],
        ],
        dtype=np.float32,
    )
    frame = AudioFrame.from_ndarray(planar, format="fltp", layout="stereo")

    pcm = SpeakerPlayer._frame_to_output(frame)

    expected = np.clip(planar.T, -1.0, 1.0)
    expected = (expected * 32767.0).astype(np.int16)
    assert pcm.shape == (4, 2)
    assert pcm.dtype == np.int16
    assert np.array_equal(pcm, expected)


def test_fill_output_buffer_preserves_chunk_order_when_partially_consumed() -> None:
    player = SpeakerPlayer()
    chunk1 = np.array([[1], [2], [3], [4]], dtype=np.int16)
    chunk2 = np.array([[5], [6], [7], [8]], dtype=np.int16)
    player._queue.put_nowait(chunk1)
    player._queue.put_nowait(chunk2)

    first = np.zeros((3, 1), dtype=np.int16)
    player._fill_output_buffer(first, 3)
    assert first[:, 0].tolist() == [1, 2, 3]

    second = np.zeros((3, 1), dtype=np.int16)
    player._fill_output_buffer(second, 3)
    assert second[:, 0].tolist() == [4, 5, 6]

    third = np.zeros((2, 1), dtype=np.int16)
    player._fill_output_buffer(third, 2)
    assert third[:, 0].tolist() == [7, 8]


@pytest.mark.asyncio
async def test_speaker_player_activity_detection_toggles_playback_guard() -> None:
    callback = AsyncMock()
    player = SpeakerPlayer(on_activity_change=callback)

    active_chunk = np.full((480, 1), 1024, dtype=np.int16)
    silent_chunk = np.zeros((480, 1), dtype=np.int16)

    await player._update_activity(active_chunk)
    assert callback.await_count == 1
    callback.assert_awaited_with(True)

    for _ in range(player.SILENT_FRAMES_TO_IDLE):
        await player._update_activity(silent_chunk)

    assert callback.await_count == 2
    assert callback.await_args_list[1].args == (False,)

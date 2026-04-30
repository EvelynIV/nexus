from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nexus.application.realtime.service import RealtimeApplicationService


def _service_without_init() -> RealtimeApplicationService:
    return RealtimeApplicationService.__new__(RealtimeApplicationService)


def test_normalize_output_modalities_accepts_dual_modalities() -> None:
    service = _service_without_init()
    assert service._normalize_output_modalities(["audio", "text"]) == ["audio", "text"]


def test_normalize_output_modalities_accepts_single_audio_or_text() -> None:
    service = _service_without_init()

    assert service._normalize_output_modalities(["audio"]) == ["audio"]
    assert service._normalize_output_modalities(["text"]) == ["text"]


@pytest.mark.asyncio
async def test_apply_session_update_rejects_voice_change_after_audio_started() -> None:
    service = _service_without_init()
    service.tts_backend = object()

    writer = SimpleNamespace(send_error=AsyncMock())
    session = SimpleNamespace(
        session_id="sess_test",
        response_model="gpt-realtime",
        writer=writer,
        update_output_modalities=lambda modalities: modalities,
        update_audio_output_config=lambda **kwargs: kwargs,
        mcp_tools=[],
        get_output_modalities=lambda: ["audio", "text"],
        get_audio_input_config=lambda: {"format_type": "audio/pcm", "sample_rate": 24000},
        get_audio_output_config=lambda: {"format_type": "audio/pcm", "voice": "alloy", "speed": 1.0},
        tools=[],
        send_event=AsyncMock(),
        is_audio_voice_locked=lambda: True,
        audio_output_voice="alloy",
    )
    update = SimpleNamespace(
        model="gpt-realtime",
        output_modalities=["audio", "text"],
        tools=None,
        audio={
            "output": {
                "voice": "marin",
            }
        },
    )

    await service.apply_session_update(
        session,
        update,
        model="gpt-realtime",
        reply_sink=writer,
    )

    assert writer.send_error.await_count == 1
    assert session.send_event.await_count == 0


@pytest.mark.asyncio
async def test_apply_session_update_accepts_dict_audio_voice_and_modalities() -> None:
    service = _service_without_init()
    service.tts_backend = object()

    applied_modalities: list[str] = []
    applied_audio_config: list[dict] = []
    writer = SimpleNamespace(send_error=AsyncMock())
    session = SimpleNamespace(
        session_id="sess_test",
        response_model="gpt-realtime",
        writer=writer,
        update_output_modalities=lambda modalities: applied_modalities.extend(modalities),
        update_audio_output_config=lambda **kwargs: applied_audio_config.append(kwargs),
        mcp_tools=[],
        get_output_modalities=lambda: ["text"],
        get_audio_input_config=lambda: {"format_type": "audio/pcm", "sample_rate": 24000},
        get_audio_output_config=lambda: {"format_type": "audio/pcm", "voice": "alloy", "speed": 1.0},
        send_event=AsyncMock(),
        is_audio_voice_locked=lambda: False,
        audio_output_voice="alloy",
        tools=[],
    )
    update = {
        "model": "gpt-realtime",
        "output_modalities": ["audio", "text"],
        "tools": [],
        "audio": {
            "output": {
                "voice": "paimon",
            }
        },
    }

    await service.apply_session_update(
        session,
        update,
        model="gpt-realtime",
        reply_sink=writer,
    )

    assert applied_modalities == ["audio", "text"]
    assert applied_audio_config == [{"format_type": None, "voice": "paimon", "speed": None}]
    assert writer.send_error.await_count == 0
    assert session.send_event.await_count == 1

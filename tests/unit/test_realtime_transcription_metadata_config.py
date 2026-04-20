from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from nexus.application.realtime.orchestrators.transcription_worker import run_transcription_worker
from nexus.application.realtime.text_processing import PreparedRealtimeUserTurn
from nexus.infrastructure.asr import TranscriptionResult


@dataclass
class _FakeSession:
    events: list[Any] = field(default_factory=list)
    audio_input_sample_rate: int = 16000
    asr_sample_rate: int = 16000
    audio_queue: asyncio.Queue[np.ndarray | None] = field(default_factory=asyncio.Queue)
    _current_chat_task: asyncio.Task | None = None
    _conversation_tail_item_id: str | None = None
    _conversation_previous_item_ids: dict[str, str | None] = field(default_factory=dict)

    async def send_event(self, event: Any) -> None:
        self.events.append(event)

    async def audio_iter(self):
        while True:
            chunk = await self.audio_queue.get()
            if chunk is None:
                break
            yield chunk

    def get_current_chat_task(self):
        return self._current_chat_task

    def set_current_chat_task(self, task):
        self._current_chat_task = task

    def request_cancel(self, reason: str = "turn_detected") -> None:
        del reason

    def reset_cancel(self) -> None:
        return None

    def register_server_conversation_item(self, item_id: str) -> str | None:
        if item_id in self._conversation_previous_item_ids:
            return self._conversation_previous_item_ids[item_id]
        previous_item_id = self._conversation_tail_item_id
        self._conversation_previous_item_ids[item_id] = previous_item_id
        self._conversation_tail_item_id = item_id
        return previous_item_id


class _FakeInferencer:
    def __init__(self, results: list[TranscriptionResult]) -> None:
        self.results = results

    async def transcribe(self, audio, **kwargs):
        del kwargs
        async for _ in audio:
            pass
        for result in self.results:
            yield result


@pytest.mark.asyncio
async def test_transcription_worker_uses_structured_speaker_context() -> None:
    session = _FakeSession()
    await session.audio_queue.put(np.zeros(160, dtype=np.int16))
    await session.audio_queue.put(None)

    captured_turns: list[PreparedRealtimeUserTurn] = []

    async def _chat_worker(session_arg, turn: PreparedRealtimeUserTurn) -> None:
        del session_arg
        captured_turns.append(turn)

    inferencer = _FakeInferencer(
        [
            TranscriptionResult(
                transcript="来给我讲个故事",
                is_final=True,
                words=[("来给我讲个故事", 0.0, 0.3)],
                speaker_id="speaker-1",
                speaker_name="migo",
                speaker_confidence=0.56,
                language_code="zh-CN",
                language_confidence=0.98,
                metadata={"emotion": "calm"},
                speaker_changed=True,
                turn_completed=True,
            )
        ]
    )

    await run_transcription_worker(
        inferencer=inferencer,
        session=session,
        interim_results=False,
        is_chat_model=True,
        chat_worker=_chat_worker,
    )

    completed_events = [
        event for event in session.events
        if getattr(event, "type", None) == "conversation.item.input_audio_transcription.completed"
    ]
    if session.get_current_chat_task() is not None:
        await session.get_current_chat_task()

    assert completed_events[0].transcript == "来给我讲个故事"
    assert captured_turns[0].display_transcript == "来给我讲个故事"
    assert captured_turns[0].speaker_id == "speaker-1"
    assert captured_turns[0].speaker_name == "migo"
    assert captured_turns[0].speaker_confidence == pytest.approx(0.56)
    assert captured_turns[0].language_code == "zh-CN"
    assert captured_turns[0].language_confidence == pytest.approx(0.98)
    assert captured_turns[0].metadata == {"emotion": "calm"}
    assert captured_turns[0].speaker_changed is True
    assert captured_turns[0].turn_completed is True
    assert "当前说话人是migo" in captured_turns[0].model_text


@pytest.mark.asyncio
async def test_transcription_worker_skips_non_final_non_positive_end_timestamp_results() -> None:
    session = _FakeSession()
    await session.audio_queue.put(np.zeros(160, dtype=np.int16))
    await session.audio_queue.put(None)

    captured_turns: list[PreparedRealtimeUserTurn] = []

    async def _chat_worker(session_arg, turn: PreparedRealtimeUserTurn) -> None:
        del session_arg
        captured_turns.append(turn)

    inferencer = _FakeInferencer(
        [
            TranscriptionResult(
                transcript="这句要被过滤",
                is_final=False,
                words=[("这句要被过滤", 0.0, -6.25e-05)],
            )
        ]
    )

    await run_transcription_worker(
        inferencer=inferencer,
        session=session,
        interim_results=False,
        is_chat_model=True,
        chat_worker=_chat_worker,
    )

    assert session.events == []
    assert captured_turns == []
    assert session.get_current_chat_task() is None

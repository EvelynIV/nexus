from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "webrtc-tui"))

from webrtc_tui.events import RealtimeEventProcessor  # noqa: E402


def test_event_processor_accumulates_user_transcription() -> None:
    processor = RealtimeEventProcessor()

    delta = processor.process(
        {
            "type": "conversation.item.input_audio_transcription.delta",
            "item_id": "item_1",
            "delta": "你好",
        }
    )
    done = processor.process(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "item_1",
            "transcript": "你好世界",
        }
    )

    assert [(item.role, item.text, item.final) for item in delta.chat_updates] == [("user", "你好", False)]
    assert [(item.role, item.text, item.final) for item in done.chat_updates] == [("user", "你好世界", True)]


def test_event_processor_prefers_text_stream_and_uses_response_done_fallback() -> None:
    processor = RealtimeEventProcessor()

    text_delta = processor.process(
        {
            "type": "response.output_text.delta",
            "response_id": "resp_1",
            "delta": "好的",
        }
    )
    transcript_delta = processor.process(
        {
            "type": "response.output_audio_transcript.delta",
            "response_id": "resp_2",
            "delta": "收到",
        }
    )
    done = processor.process(
        {
            "type": "response.done",
        }
    )

    assert [(item.role, item.text, item.final) for item in text_delta.chat_updates] == [
        ("assistant", "好的", False)
    ]
    assert [(item.role, item.text, item.final) for item in transcript_delta.chat_updates] == [
        ("assistant", "收到", False)
    ]
    assert [(item.role, item.text, item.final) for item in done.chat_updates] == [
        ("assistant", "好的", True),
        ("assistant", "收到", True),
    ]


def test_event_processor_normalizes_error_message() -> None:
    processor = RealtimeEventProcessor()

    result = processor.process(
        {
            "type": "error",
            "error": {
                "message": "boom",
            },
        }
    )

    assert result.error_message == "boom"


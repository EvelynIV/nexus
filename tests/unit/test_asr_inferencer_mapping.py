from __future__ import annotations

from google.protobuf.duration_pb2 import Duration
import pytest

from nexus.infrastructure.asr.inferencer import _extract_transcription_result
from nexus.protos.asr import ux_speech_pb2 as pb2


def _duration(seconds: int, nanos: int = 0):
    return Duration(
        seconds=seconds,
        nanos=nanos,
    )


def test_extract_transcription_result_maps_structured_fields() -> None:
    result = pb2.StreamingRecognitionResult(
        is_final=True,
        alternative=pb2.SpeechRecognitionAlternative(
            transcript="你好，世界",
            words=[
                pb2.WordInfo(
                    word="你好",
                    start_time=_duration(0, 100_000_000),
                    end_time=_duration(0, 300_000_000),
                ),
                pb2.WordInfo(
                    word="世界",
                    start_time=_duration(0, 300_000_000),
                    end_time=_duration(0, 600_000_000),
                ),
            ],
            metadata={"emotion": "calm", "age": "adult"},
            speaker=pb2.SpeakerInfo(
                id="speaker-1",
                name="migo",
                confidence=0.56,
            ),
            language=pb2.LanguageInfo(
                code="zh-CN",
                confidence=0.98,
            ),
            speaker_changed=True,
            turn_completed=True,
        ),
    )

    extracted = _extract_transcription_result(result)

    assert extracted.transcript == "你好，世界"
    assert extracted.is_final is True
    assert extracted.words == [
        ("你好", pytest.approx(0.1), pytest.approx(0.3)),
        ("世界", pytest.approx(0.3), pytest.approx(0.6)),
    ]
    assert extracted.speaker_id == "speaker-1"
    assert extracted.speaker_name == "migo"
    assert extracted.speaker_confidence == pytest.approx(0.56)
    assert extracted.language_code == "zh-CN"
    assert extracted.language_confidence == pytest.approx(0.98)
    assert extracted.metadata == {"emotion": "calm", "age": "adult"}
    assert extracted.speaker_changed is True
    assert extracted.turn_completed is True


def test_extract_transcription_result_handles_missing_optional_fields() -> None:
    result = pb2.StreamingRecognitionResult(
        is_final=False,
        alternative=pb2.SpeechRecognitionAlternative(
            transcript="plain transcript",
        ),
    )

    extracted = _extract_transcription_result(result)

    assert extracted.transcript == "plain transcript"
    assert extracted.is_final is False
    assert extracted.words is None
    assert extracted.speaker_id is None
    assert extracted.speaker_name is None
    assert extracted.speaker_confidence is None
    assert extracted.language_code is None
    assert extracted.language_confidence is None
    assert extracted.metadata == {}
    assert extracted.speaker_changed is False
    assert extracted.turn_completed is False

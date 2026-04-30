from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from nexus.infrastructure.asr import TranscriptionResult

_WHITESPACE_PATTERN = re.compile(r"\s+")
_CJK_SPACE_PATTERN = re.compile(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])")
_TTS_DELIMITER_SPACING_PATTERN = re.compile(r"\s*([。！？；])\s*")

_SENTENCE_DELIMITER_MAP = {
    ".": "。",
    "!": "！",
    "?": "？",
    ";": "；",
    "。": "。",
    "！": "！",
    "？": "？",
    "；": "；",
}
_TTS_SENTENCE_DELIMITERS = set(_SENTENCE_DELIMITER_MAP)

_EMOJI_RANGES = (
    (0x1F300, 0x1F5FF),
    (0x1F600, 0x1F64F),
    (0x1F680, 0x1F6FF),
    (0x1F700, 0x1F77F),
    (0x1F780, 0x1F7FF),
    (0x1F800, 0x1F8FF),
    (0x1F900, 0x1F9FF),
    (0x1FA70, 0x1FAFF),
    (0x2600, 0x26FF),
    (0x2700, 0x27BF),
    (0xFE00, 0xFE0F),
)


@dataclass(frozen=True)
class PreparedRealtimeUserTurn:
    raw_transcript: str
    display_transcript: str
    model_text: str
    speaker_id: str | None = None
    speaker_name: str | None = None
    speaker_confidence: float | None = None
    language_code: str | None = None
    language_confidence: float | None = None
    metadata: dict[str, str] | None = None
    speaker_changed: bool = False
    turn_completed: bool = False


@dataclass
class SanitizedModelOutputAccumulator:
    raw_text: str = ""
    display_text: str = ""
    tts_text: str = ""

    def push(self, delta: str) -> tuple[str, str]:
        if not delta:
            return "", ""

        self.raw_text += delta
        next_display = sanitize_model_output_for_display(self.raw_text)
        next_tts = sanitize_model_output_for_tts(self.raw_text)

        display_delta = _incremental_suffix(self.display_text, next_display)
        tts_delta = _incremental_suffix(self.tts_text, next_tts)

        self.display_text = next_display
        self.tts_text = next_tts
        return display_delta, tts_delta


def prepare_realtime_user_turn(transcription_result: TranscriptionResult) -> PreparedRealtimeUserTurn:
    display_transcript = transcription_result.transcript
    speaker_label = transcription_result.speaker_name or transcription_result.speaker_id
    if speaker_label:
        model_text = (
            f"当前说话人是{speaker_label}。"
            "这只是辅助上下文，不要直接复述说话人标签。"
            f"用户说：{display_transcript}"
        )
    else:
        model_text = display_transcript

    return PreparedRealtimeUserTurn(
        raw_transcript=transcription_result.transcript,
        display_transcript=display_transcript,
        model_text=model_text,
        speaker_id=transcription_result.speaker_id,
        speaker_name=transcription_result.speaker_name,
        speaker_confidence=transcription_result.speaker_confidence,
        language_code=transcription_result.language_code,
        language_confidence=transcription_result.language_confidence,
        metadata=dict(transcription_result.metadata),
        speaker_changed=transcription_result.speaker_changed,
        turn_completed=transcription_result.turn_completed,
    )


def sanitize_model_output_for_display(text: str) -> str:
    return _sanitize_model_output(text, preserve_sentence_delimiters=False)


def sanitize_model_output_for_tts(text: str) -> str:
    return _sanitize_model_output(text, preserve_sentence_delimiters=True)


def _sanitize_model_output(text: str, *, preserve_sentence_delimiters: bool) -> str:
    if not text:
        return ""

    chunks: list[str] = []
    for char in text:
        if _is_emoji(char):
            continue
        if char in {"\r", "\t"}:
            char = " "
        elif char == "\n":
            char = "。" if preserve_sentence_delimiters else " "

        if char.isspace():
            chunks.append(" ")
            continue

        if preserve_sentence_delimiters and char in _TTS_SENTENCE_DELIMITERS:
            chunks.append(_SENTENCE_DELIMITER_MAP[char])
            continue

        category = unicodedata.category(char)
        if category.startswith(("P", "S")):
            continue

        chunks.append(char)

    result = "".join(chunks)
    result = _WHITESPACE_PATTERN.sub(" ", result)
    result = _CJK_SPACE_PATTERN.sub("", result)
    if preserve_sentence_delimiters:
        result = _TTS_DELIMITER_SPACING_PATTERN.sub(r"\1", result)
    return result.strip()


def _is_emoji(char: str) -> bool:
    codepoint = ord(char)
    return any(start <= codepoint <= end for start, end in _EMOJI_RANGES)


def _incremental_suffix(previous: str, current: str) -> str:
    if not current:
        return ""
    if not previous:
        return current
    if current.startswith(previous):
        return current[len(previous):]

    prefix_len = 0
    max_prefix = min(len(previous), len(current))
    while prefix_len < max_prefix and previous[prefix_len] == current[prefix_len]:
        prefix_len += 1
    return current[prefix_len:]

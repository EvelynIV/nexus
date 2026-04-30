from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ConnectionPhase(str, Enum):
    IDLE = "idle"
    CONNECTING = "connecting"
    CONNECTED = "connected"


@dataclass(slots=True)
class AssistantAccumulator:
    text: str = ""
    transcript: str = ""
    emitted: bool = False


@dataclass(slots=True)
class ConversationUpdate:
    role: str
    text: str
    final: bool


@dataclass(slots=True)
class ProcessedRealtimeEvent:
    status_message: str | None = None
    error_message: str | None = None
    chat_updates: list[ConversationUpdate] = field(default_factory=list)


@dataclass(slots=True)
class SessionRuntimeState:
    phase: ConnectionPhase = ConnectionPhase.IDLE
    call_id: str | None = None
    data_channel_open: bool = False
    manual_mute: bool = False
    playback_guard_active: bool = False


@dataclass(slots=True)
class ChatLine:
    role: str
    text: str


@dataclass(slots=True)
class AppViewState:
    status: str = "正在加载配置…"
    messages: list[ChatLine] = field(default_factory=list)
    pending_user: str = ""
    pending_assistant: str = ""
    errors: list[str] = field(default_factory=list)

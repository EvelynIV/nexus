from __future__ import annotations

from dataclasses import dataclass

from nexus.domain.realtime import RealtimeSessionState
from nexus.application.realtime.protocol import RealtimeReplySink


@dataclass
class RealtimeDispatchContext:
    session: RealtimeSessionState
    service: "RealtimeApplicationService"
    model: str
    reply_sink: RealtimeReplySink


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nexus.application.realtime.service import RealtimeApplicationService

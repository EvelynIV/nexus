from __future__ import annotations

from dataclasses import dataclass

from nexus.domain.realtime import RealtimeSessionState


@dataclass
class RealtimeDispatchContext:
    session: RealtimeSessionState
    service: "RealtimeApplicationService"
    model: str


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nexus.application.realtime.service import RealtimeApplicationService

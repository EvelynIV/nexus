"""Compatibility wrapper: Realtime session state moved to domain layer."""

from nexus.domain.realtime import RealtimeSessionState as RealtimeSession

__all__ = ["RealtimeSession"]

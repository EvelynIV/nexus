"""Compatibility wrapper for legacy configure/get/shutdown APIs."""

from nexus.application.container import configure, get_container, shutdown


class RealtimeServicerProxy:
    @property
    def realtime(self):
        return get_container().realtime


def get_realtime_servicer():
    return get_container().realtime


__all__ = ["configure", "shutdown", "get_realtime_servicer", "RealtimeServicerProxy"]

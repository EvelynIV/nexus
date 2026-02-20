"""Legacy compatibility module. Event routing now lives in application.realtime.dispatch."""

from nexus.application.realtime.dispatch.default_registry import build_default_registry

__all__ = ["build_default_registry"]

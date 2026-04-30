from nexus.sessions.responses_session import ResponsesSession

__all__ = ["ResponsesSession", "RealtimeSession"]


def __getattr__(name: str):
    if name == "RealtimeSession":
        from nexus.domain.realtime import RealtimeSessionState

        return RealtimeSessionState
    raise AttributeError(name)

from nexus.sessions.chat_session import AsyncChatSession, ChatSession

__all__ = ["AsyncChatSession", "ChatSession", "RealtimeSession"]


def __getattr__(name: str):
    if name == "RealtimeSession":
        from nexus.domain.realtime import RealtimeSessionState

        return RealtimeSessionState
    raise AttributeError(name)

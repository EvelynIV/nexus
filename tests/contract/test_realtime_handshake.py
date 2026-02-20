from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
from fastapi import WebSocketDisconnect
from openai.types.realtime import SessionCreatedEvent, SessionUpdatedEvent

from nexus.application.realtime.protocol.ids import event_id
from nexus.api.v1.realtime.endpoint import realtime_endpoint_worker


class FakeWebSocket:
    def __init__(self, incoming_messages: list[str]):
        self._incoming = list(incoming_messages)
        self.sent: list[dict] = []
        self.accepted = False
        self.closed = False
        self.close_code = None

    async def accept(self):
        self.accepted = True

    async def receive_text(self) -> str:
        if self._incoming:
            return self._incoming.pop(0)
        raise WebSocketDisconnect

    async def send_text(self, text: str):
        self.sent.append(json.loads(text))

    async def close(self, code: int = 1000):
        self.closed = True
        self.close_code = code


@dataclass
class DummyMcpRegistry:
    closed: bool = False

    async def close(self):
        self.closed = True


@dataclass
class DummySession:
    writer: any
    session_id: str = "sess_test"
    output_modalities: list[str] = field(default_factory=lambda: ["text"])
    audio_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    mcp_registry: DummyMcpRegistry = field(default_factory=DummyMcpRegistry)

    async def send_event(self, event):
        await self.writer.send_event(event)

    def update_output_modalities(self, modalities):
        self.output_modalities = list(modalities)

    def get_output_modalities(self):
        return list(self.output_modalities)

    def add_tool_result(self, tool_call_id: str, content: str):
        del tool_call_id, content


class DummyRealtimeService:
    def __init__(self):
        self.session: DummySession | None = None

    def create_session(self, *, writer, output_modalities, tools, chat_model):
        del tools, chat_model
        self.session = DummySession(writer=writer, output_modalities=list(output_modalities))
        return self.session

    async def emit_session_created(self, session, model):
        await session.send_event(
            SessionCreatedEvent(
                type="session.created",
                event_id=event_id(),
                session={
                    "type": "realtime",
                    "id": session.session_id,
                    "model": model,
                    "output_modalities": session.get_output_modalities(),
                },
            )
        )

    async def apply_session_update(self, session, update, *, model):
        del model
        if getattr(update, "output_modalities", None):
            session.update_output_modalities(update.output_modalities)

        await session.send_event(
            SessionUpdatedEvent(
                type="session.updated",
                event_id=event_id(),
                session={
                    "type": "realtime",
                    "id": session.session_id,
                    "model": "gpt-4o-realtime-preview",
                    "output_modalities": session.get_output_modalities(),
                },
            )
        )

    async def start_transcription_worker(self, session, is_chat_model):
        del session, is_chat_model
        return asyncio.create_task(asyncio.sleep(3600))

    async def handle_input_audio_commit(self, session, event):
        del session, event

    async def handle_response_create(self, session, event):
        del session, event

    async def handle_response_cancel(self, session, event):
        del session, event

    async def close_session(self, session):
        await session.mcp_registry.close()


@pytest.mark.asyncio
async def test_realtime_handshake_starts_with_session_created_then_updated():
    ws = FakeWebSocket(
        [
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "output_modalities": ["text"],
                    },
                }
            )
        ]
    )
    container = SimpleNamespace(realtime=DummyRealtimeService())

    await realtime_endpoint_worker(
        websocket=ws,
        model="gpt-4o-realtime-preview",
        container=container,
    )

    assert ws.accepted is True
    assert ws.sent[0]["type"] == "session.created"
    assert ws.sent[1]["type"] == "session.updated"


@pytest.mark.asyncio
async def test_realtime_requires_session_update_as_first_client_event():
    ws = FakeWebSocket([json.dumps({"type": "response.create"})])
    container = SimpleNamespace(realtime=DummyRealtimeService())

    await realtime_endpoint_worker(
        websocket=ws,
        model="gpt-4o-realtime-preview",
        container=container,
    )

    assert ws.sent[0]["type"] == "session.created"
    assert ws.sent[1]["type"] == "error"
    assert ws.closed is True
    assert ws.close_code == 1008

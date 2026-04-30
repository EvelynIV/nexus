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
        self.headers = {}

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
class DummySession:
    writer: any
    session_id: str = "sess_test"
    output_modalities: list[str] = field(default_factory=lambda: ["text"])
    audio_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    audio_output_queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    async def send_event(self, event):
        await self.writer.send_event(event)

    def update_output_modalities(self, modalities):
        self.output_modalities = list(modalities)

    def get_output_modalities(self):
        return list(self.output_modalities)

    def add_tool_result(self, tool_call_id: str, content: str):
        del tool_call_id, content

    async def close_audio_output(self):
        await self.audio_output_queue.put(None)


class DummyRealtimeService:
    def __init__(self):
        self.session: DummySession | None = None

    def create_session(self, *, writer, output_modalities, tools, response_model):
        del tools, response_model
        self.session = DummySession(writer=writer, output_modalities=list(output_modalities))
        return self.session

    async def emit_session_created(self, session, model, sink=None):
        target = sink or session
        await target.send_event(
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

    async def apply_session_update(self, session, update, *, model, reply_sink=None, emit_event=True):
        del reply_sink
        del model
        if getattr(update, "output_modalities", None):
            session.update_output_modalities(update.output_modalities)

        if emit_event:
            await session.send_event(
                SessionUpdatedEvent(
                    type="session.updated",
                    event_id=event_id(),
                    session={
                        "type": "realtime",
                        "id": session.session_id,
                        "model": "gpt-realtime",
                        "output_modalities": session.get_output_modalities(),
                    },
                )
            )

    async def start_transcription_worker(self, session, auto_response_enabled):
        del session, auto_response_enabled
        return asyncio.create_task(asyncio.sleep(3600))

    async def handle_input_audio_commit(self, session, event):
        del session, event

    async def handle_response_create(self, session, event, *, reply_sink=None):
        del session, event, reply_sink

    async def handle_response_cancel(self, session, event, *, reply_sink=None):
        del session, event, reply_sink

    async def close_session(self, session):
        del session


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        realtime_api_key=None,
        realtime_client_secret_ttl_seconds=600,
        realtime_session_max_seconds=3600,
        asterisk_ingress_enabled=False,
        asterisk_ari_url="http://127.0.0.1:8088/ari",
        asterisk_ari_user="voicebot",
        asterisk_ari_password="12345678",
        asterisk_stasis_app="nexus",
        asterisk_external_host="127.0.0.1",
        asterisk_rtp_port_start=4000,
        asterisk_rtp_port_end=4099,
        asterisk_codec="ulaw",
        realtime_webhook_url=None,
        realtime_webhook_secret=None,
        asterisk_refer_endpoint_prefix=None,
    )


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
    container = SimpleNamespace(
        realtime=DummyRealtimeService(),
        config=_config(),
    )

    await realtime_endpoint_worker(
        websocket=ws,
        model="gpt-realtime",
        call_id=None,
        container=container,
    )

    assert ws.accepted is True
    assert ws.sent[0]["type"] == "session.created"
    assert ws.sent[1]["type"] == "session.updated"


@pytest.mark.asyncio
async def test_realtime_allows_non_session_update_as_first_client_event():
    ws = FakeWebSocket([json.dumps({"type": "response.create"})])
    container = SimpleNamespace(
        realtime=DummyRealtimeService(),
        config=_config(),
    )

    await realtime_endpoint_worker(
        websocket=ws,
        model="gpt-realtime",
        call_id=None,
        container=container,
    )

    assert ws.sent[0]["type"] == "session.created"
    assert [event["type"] for event in ws.sent].count("error") == 0

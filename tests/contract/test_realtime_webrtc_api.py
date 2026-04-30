from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from nexus.api.v1.realtime import router
from nexus.api.v1.realtime.runtime import RealtimeCallError
from nexus.application.container import get_container


class FakeClientSecrets:
    def __init__(self) -> None:
        self.created: list[dict] = []
        self.records: dict[str, SimpleNamespace] = {}

    async def create(self, *, session, ttl_seconds):
        self.created.append({"session": session, "ttl_seconds": ttl_seconds})
        record = SimpleNamespace(
            value="ek_test",
            expires_at=1234567890,
            session=session,
        )
        self.records[record.value] = record
        return record

    async def get(self, value: str):
        return self.records.get(value)


class FakeCalls:
    def __init__(self) -> None:
        self.created: list[dict] = []
        self.calls: dict[str, SimpleNamespace] = {}
        self.accepted: list[dict] = []
        self.rejected: list[dict] = []
        self.referred: list[dict] = []
        self.hungup: list[str] = []
        self.next_error: RealtimeCallError | None = None

    async def create_call(self, *, sdp_offer, session_config):
        self.created.append(
            {
                "sdp_offer": sdp_offer,
                "session_config": session_config,
            }
        )
        call = SimpleNamespace(
            call_id="rtc_test",
            peer_connection=SimpleNamespace(localDescription=SimpleNamespace(sdp="v=0\nanswer")),
            controller=SimpleNamespace(enqueue_text=AsyncMock()),
            attach_sideband=AsyncMock(),
            detach_sideband=AsyncMock(),
        )
        self.calls[call.call_id] = call
        return call

    async def get(self, call_id: str):
        return self.calls.get(call_id)

    async def accept_call(self, call_id: str, *, session_config):
        if self.next_error:
            raise self.next_error
        self.accepted.append({"call_id": call_id, "session_config": session_config})

    async def reject_call(self, call_id: str, *, status_code: int = 603):
        if self.next_error:
            raise self.next_error
        self.rejected.append({"call_id": call_id, "status_code": status_code})

    async def refer_call(self, call_id: str, *, target_uri: str):
        if self.next_error:
            raise self.next_error
        self.referred.append({"call_id": call_id, "target_uri": target_uri})

    async def hangup_call(self, call_id: str):
        if self.next_error:
            raise self.next_error
        self.hungup.append(call_id)


class FakeRuntime:
    def __init__(self, *, api_key: str | None = None) -> None:
        self._api_key = api_key
        self.client_secrets = FakeClientSecrets()
        self.calls = FakeCalls()

    def api_key_required(self) -> bool:
        return self._api_key is not None

    def check_api_key(self, bearer_token: str | None) -> bool:
        return bearer_token == self._api_key


def build_client(container) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_container] = lambda: container
    return TestClient(app)


def test_create_client_secret_uses_requested_ttl_and_normalized_session(monkeypatch) -> None:
    runtime = FakeRuntime(api_key="topsecret")
    container = SimpleNamespace(
        config=SimpleNamespace(
            realtime_api_key="topsecret",
            realtime_client_secret_ttl_seconds=600,
            realtime_session_max_seconds=3600,
        )
    )
    monkeypatch.setattr("nexus.api.v1.realtime.http.get_realtime_api_runtime", AsyncMock(return_value=runtime))
    client = build_client(container)

    response = client.post(
        "/realtime/client_secrets",
        headers={"Authorization": "Bearer topsecret"},
        json={
            "expires_after": {"anchor": "created_at", "seconds": 120},
            "session": {
                "model": "gpt-realtime",
                "audio": {"output": {"voice": "marin"}},
            },
        },
    )

    assert response.status_code == 200
    assert runtime.client_secrets.created[0]["ttl_seconds"] == 120
    assert runtime.client_secrets.created[0]["session"]["type"] == "realtime"
    assert runtime.client_secrets.created[0]["session"]["audio"]["output"]["voice"] == "marin"
    assert response.json()["value"] == "ek_test"


def test_create_realtime_call_accepts_application_sdp_with_client_secret(monkeypatch) -> None:
    runtime = FakeRuntime()
    runtime.client_secrets.records["ek_valid"] = SimpleNamespace(
        value="ek_valid",
        expires_at=1234567890,
        session={
            "id": "sess_test",
            "type": "realtime",
            "object": "realtime.session",
            "model": "gpt-realtime",
            "output_modalities": ["audio", "text"],
            "tools": [],
            "tool_choice": "auto",
            "audio": {
                "input": {"format": {"type": "audio/pcm", "rate": 24000}},
                "output": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "voice": "alloy",
                    "speed": 1.0,
                },
            },
        },
    )
    container = SimpleNamespace(
        config=SimpleNamespace(
            realtime_api_key=None,
            realtime_client_secret_ttl_seconds=600,
            realtime_session_max_seconds=3600,
        )
    )
    monkeypatch.setattr("nexus.api.v1.realtime.http.get_realtime_api_runtime", AsyncMock(return_value=runtime))
    client = build_client(container)

    response = client.post(
        "/realtime/calls",
        headers={
            "Authorization": "Bearer ek_valid",
            "Content-Type": "application/sdp",
        },
        content="v=0\noffer",
    )

    assert response.status_code == 201
    assert response.headers["location"] == "/v1/realtime/calls/rtc_test"
    assert runtime.calls.created[0]["sdp_offer"] == "v=0\noffer"
    assert runtime.calls.created[0]["session_config"]["id"] == "sess_test"


def test_create_realtime_call_accepts_multipart_session(monkeypatch) -> None:
    runtime = FakeRuntime(api_key="topsecret")
    container = SimpleNamespace(
        config=SimpleNamespace(
            realtime_api_key="topsecret",
            realtime_client_secret_ttl_seconds=600,
            realtime_session_max_seconds=3600,
        )
    )
    monkeypatch.setattr("nexus.api.v1.realtime.http.get_realtime_api_runtime", AsyncMock(return_value=runtime))
    client = build_client(container)

    response = client.post(
        "/realtime/calls",
        headers={"Authorization": "Bearer topsecret"},
        files={
            "sdp": ("offer.sdp", "v=0\noffer", "application/sdp"),
            "session": (None, json.dumps({"model": "gpt-realtime", "output_modalities": ["text"]}), "application/json"),
        },
    )

    assert response.status_code == 201
    assert runtime.calls.created[0]["session_config"]["output_modalities"] == ["text"]


def test_create_realtime_call_accepts_official_json_shape(monkeypatch) -> None:
    runtime = FakeRuntime(api_key="topsecret")
    container = SimpleNamespace(
        config=SimpleNamespace(
            realtime_api_key="topsecret",
            realtime_client_secret_ttl_seconds=600,
            realtime_session_max_seconds=3600,
        )
    )
    monkeypatch.setattr("nexus.api.v1.realtime.http.get_realtime_api_runtime", AsyncMock(return_value=runtime))
    client = build_client(container)

    response = client.post(
        "/realtime/calls",
        headers={"Authorization": "Bearer topsecret"},
        json={
            "sdp": "v=0\noffer",
            "session": {"type": "realtime", "model": "gpt-realtime", "output_modalities": ["text"]},
        },
    )

    assert response.status_code == 201
    assert runtime.calls.created[0]["sdp_offer"] == "v=0\noffer"
    assert runtime.calls.created[0]["session_config"]["output_modalities"] == ["text"]


def test_accept_realtime_call_uses_official_audio_default(monkeypatch) -> None:
    runtime = FakeRuntime(api_key="topsecret")
    container = SimpleNamespace(
        config=SimpleNamespace(
            realtime_api_key="topsecret",
            realtime_client_secret_ttl_seconds=600,
            realtime_session_max_seconds=3600,
        )
    )
    monkeypatch.setattr("nexus.api.v1.realtime.http.get_realtime_api_runtime", AsyncMock(return_value=runtime))
    client = build_client(container)

    response = client.post(
        "/realtime/calls/rtc_test/accept",
        headers={"Authorization": "Bearer topsecret"},
        json={
            "type": "realtime",
            "model": "gpt-realtime",
            "instructions": "You are a support agent.",
        },
    )

    assert response.status_code == 200
    assert response.content == b""
    accepted = runtime.calls.accepted[0]
    assert accepted["call_id"] == "rtc_test"
    assert accepted["session_config"]["output_modalities"] == ["audio"]
    assert accepted["session_config"]["instructions"] == "You are a support agent."


def test_accept_realtime_call_requires_realtime_type(monkeypatch) -> None:
    runtime = FakeRuntime(api_key="topsecret")
    container = SimpleNamespace(
        config=SimpleNamespace(
            realtime_api_key="topsecret",
            realtime_client_secret_ttl_seconds=600,
            realtime_session_max_seconds=3600,
        )
    )
    monkeypatch.setattr("nexus.api.v1.realtime.http.get_realtime_api_runtime", AsyncMock(return_value=runtime))
    client = build_client(container)

    response = client.post(
        "/realtime/calls/rtc_test/accept",
        headers={"Authorization": "Bearer topsecret"},
        json={"model": "gpt-realtime"},
    )

    assert response.status_code == 400


def test_reject_realtime_call_defaults_to_603(monkeypatch) -> None:
    runtime = FakeRuntime(api_key="topsecret")
    container = SimpleNamespace(
        config=SimpleNamespace(
            realtime_api_key="topsecret",
            realtime_client_secret_ttl_seconds=600,
            realtime_session_max_seconds=3600,
        )
    )
    monkeypatch.setattr("nexus.api.v1.realtime.http.get_realtime_api_runtime", AsyncMock(return_value=runtime))
    client = build_client(container)

    response = client.post(
        "/realtime/calls/rtc_test/reject",
        headers={"Authorization": "Bearer topsecret"},
        json={},
    )

    assert response.status_code == 200
    assert response.content == b""
    assert runtime.calls.rejected == [{"call_id": "rtc_test", "status_code": 603}]


def test_refer_realtime_call_requires_sip_or_tel_uri(monkeypatch) -> None:
    runtime = FakeRuntime(api_key="topsecret")
    container = SimpleNamespace(
        config=SimpleNamespace(
            realtime_api_key="topsecret",
            realtime_client_secret_ttl_seconds=600,
            realtime_session_max_seconds=3600,
        )
    )
    monkeypatch.setattr("nexus.api.v1.realtime.http.get_realtime_api_runtime", AsyncMock(return_value=runtime))
    client = build_client(container)

    bad_response = client.post(
        "/realtime/calls/rtc_test/refer",
        headers={"Authorization": "Bearer topsecret"},
        json={"target_uri": "agent@example.com"},
    )
    good_response = client.post(
        "/realtime/calls/rtc_test/refer",
        headers={"Authorization": "Bearer topsecret"},
        json={"target_uri": "tel:+14155550123"},
    )

    assert bad_response.status_code == 400
    assert good_response.status_code == 200
    assert runtime.calls.referred == [{"call_id": "rtc_test", "target_uri": "tel:+14155550123"}]


def test_hangup_realtime_call_maps_unknown_call(monkeypatch) -> None:
    runtime = FakeRuntime(api_key="topsecret")
    runtime.calls.next_error = RealtimeCallError(404, "Call rtc_missing not found.")
    container = SimpleNamespace(
        config=SimpleNamespace(
            realtime_api_key="topsecret",
            realtime_client_secret_ttl_seconds=600,
            realtime_session_max_seconds=3600,
        )
    )
    monkeypatch.setattr("nexus.api.v1.realtime.http.get_realtime_api_runtime", AsyncMock(return_value=runtime))
    client = build_client(container)

    response = client.post(
        "/realtime/calls/rtc_missing/hangup",
        headers={"Authorization": "Bearer topsecret"},
    )

    assert response.status_code == 404


def test_reject_realtime_call_maps_state_conflict(monkeypatch) -> None:
    runtime = FakeRuntime(api_key="topsecret")
    runtime.calls.next_error = RealtimeCallError(409, "Call rtc_test is already accepted.")
    container = SimpleNamespace(
        config=SimpleNamespace(
            realtime_api_key="topsecret",
            realtime_client_secret_ttl_seconds=600,
            realtime_session_max_seconds=3600,
        )
    )
    monkeypatch.setattr("nexus.api.v1.realtime.http.get_realtime_api_runtime", AsyncMock(return_value=runtime))
    client = build_client(container)

    response = client.post(
        "/realtime/calls/rtc_test/reject",
        headers={"Authorization": "Bearer topsecret"},
        json={"status_code": 486},
    )

    assert response.status_code == 409


def test_sideband_websocket_attaches_to_existing_call(monkeypatch) -> None:
    runtime = FakeRuntime()
    fake_call = SimpleNamespace(
        controller=SimpleNamespace(enqueue_text=AsyncMock()),
        attach_sideband=AsyncMock(),
        detach_sideband=AsyncMock(),
    )
    runtime.calls.calls["rtc_test"] = fake_call
    container = SimpleNamespace(
        config=SimpleNamespace(
            realtime_api_key=None,
            realtime_client_secret_ttl_seconds=600,
            realtime_session_max_seconds=3600,
        ),
        realtime=None,
    )
    monkeypatch.setattr("nexus.api.v1.realtime.endpoint.get_realtime_api_runtime", AsyncMock(return_value=runtime))
    client = build_client(container)

    with client.websocket_connect("/realtime?call_id=rtc_test") as websocket:
        websocket.send_text('{"type":"session.update","session":{"type":"realtime"}}')

    assert fake_call.attach_sideband.await_count == 1
    assert fake_call.controller.enqueue_text.await_count == 1
    assert fake_call.detach_sideband.await_count == 1

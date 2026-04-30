from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from openai import OpenAI

from nexus.api.v1.realtime.asterisk import (
    AsteriskCallError,
    AsteriskCallRegistry,
    G711PcmTranscoder,
    G711_BYTES_PER_FRAME,
    sign_webhook_payload,
)


def _container(**overrides):
    config = SimpleNamespace(
        asterisk_ingress_enabled=False,
        asterisk_ari_url="http://127.0.0.1:8088/ari",
        asterisk_ari_user="voicebot",
        asterisk_ari_password="12345678",
        asterisk_stasis_app="nexus",
        asterisk_external_host="127.0.0.1",
        asterisk_rtp_port_start=4000,
        asterisk_rtp_port_end=4001,
        asterisk_codec="ulaw",
        realtime_webhook_url=None,
        realtime_webhook_secret=None,
        asterisk_refer_endpoint_prefix=None,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return SimpleNamespace(config=config)


class FakeAri:
    def __init__(self) -> None:
        self.deleted: list[tuple[str, dict[str, str]]] = []
        self.posted: list[tuple[str, dict[str, str]]] = []

    async def delete(self, path: str, **params: str) -> None:
        self.deleted.append((path, params))

    async def post(self, path: str, **params: str) -> dict[str, str]:
        self.posted.append((path, params))
        if path == "/bridges":
            return {"id": "bridge_test"}
        if path == "/channels/externalMedia":
            return {"id": "external_test"}
        return {}


def test_webhook_signature_matches_openai_sdk_verifier() -> None:
    secret = "local_webhook_secret"
    webhook_id = "wh_test"
    timestamp = int(time.time())
    payload = '{"object":"event","type":"realtime.call.incoming"}'

    headers = {
        "webhook-id": webhook_id,
        "webhook-timestamp": str(timestamp),
        "webhook-signature": sign_webhook_payload(secret, webhook_id, timestamp, payload),
    }

    OpenAI(api_key="dummy", webhook_secret=secret).webhooks.verify_signature(payload, headers)


@pytest.mark.asyncio
async def test_asterisk_registry_creates_pending_call_and_deduplicates_channel() -> None:
    registry = AsteriskCallRegistry(_container())
    registry.ari = FakeAri()
    channel = {
        "id": "1777.1",
        "name": "PJSIP/fxo1-00000001",
        "caller": {"number": "+15551234567"},
        "dialplan": {"exten": "s"},
    }

    first = await registry.create_pending_call(channel)
    second = await registry.create_pending_call(channel)

    assert first is not None
    assert second is first
    assert first.call_id.startswith("sip_")
    assert first.pending.sip_headers == [
        {"name": "From", "value": "+15551234567"},
        {"name": "To", "value": "s"},
        {"name": "Call-ID", "value": "1777.1"},
    ]


@pytest.mark.asyncio
async def test_reject_pending_asterisk_call_hangs_up_channel_and_releases_port() -> None:
    registry = AsteriskCallRegistry(_container())
    fake_ari = FakeAri()
    registry.ari = fake_ari
    call = await registry.create_pending_call({"id": "1777.1", "name": "PJSIP/fxo1-00000001"})
    assert call is not None

    await registry.reject_call(call.call_id, status_code=486)
    new_call = await registry.create_pending_call({"id": "1777.2", "name": "PJSIP/fxo1-00000002"})

    assert fake_ari.deleted[0] == ("/channels/1777.1", {"reason": "busy"})
    assert new_call is not None
    assert new_call.rtp_port == call.rtp_port


@pytest.mark.asyncio
async def test_refer_without_asterisk_redirect_config_returns_not_implemented() -> None:
    registry = AsteriskCallRegistry(_container())
    registry.ari = FakeAri()
    call = await registry.create_pending_call({"id": "1777.1", "name": "PJSIP/fxo1-00000001"})
    assert call is not None

    with pytest.raises(AsteriskCallError) as exc_info:
        await registry.refer_call(call.call_id, target_uri="tel:+14155550123")

    assert exc_info.value.status_code == 501


def test_g711_transcoder_round_trips_ulaw_and_alaw_to_realtime_pcm() -> None:
    for codec, silence in (("ulaw", b"\xff"), ("alaw", b"\xd5")):
        transcoder = G711PcmTranscoder(codec)
        pcm24 = transcoder.rtp_payload_to_realtime_pcm(silence * G711_BYTES_PER_FRAME)
        payload = transcoder.realtime_pcm_to_rtp_payload(pcm24)

        assert pcm24
        assert payload

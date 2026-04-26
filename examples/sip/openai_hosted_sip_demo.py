#!/usr/bin/env python3
"""Control-plane demo for OpenAI hosted SIP calls.

OpenAI hosts the SIP endpoint and media bridge in this flow. This script only
receives OpenAI webhooks, accepts/rejects calls, and optionally monitors the
Realtime session for events.
"""

from __future__ import annotations

import asyncio
import json
import os
import ssl
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any

import httpx
import typer
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from openai import AsyncOpenAI, InvalidWebhookSignatureError
from pydantic import BaseModel


DEFAULT_INSTRUCTIONS = (
    "You are a concise phone voice assistant. Keep replies brief, natural, "
    "and suitable for a live phone call."
)


@dataclass(frozen=True)
class HostedSipConfig:
    webhook_secret: str | None
    model: str
    voice: str
    instructions: str
    greeting: str
    transcription_model: str
    transcription_language: str
    verbose_deltas: bool
    tls_verify: bool


@dataclass
class ActiveCall:
    call_id: str
    status: str
    created_at: float
    sip_headers: list[dict[str, str]] = field(default_factory=list)
    last_event: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "status": self.status,
            "created_at": self.created_at,
            "last_event": self.last_event,
            "sip_headers": self.sip_headers,
        }


class ReferRequest(BaseModel):
    target_uri: str


class HostedSipApp:
    def __init__(self, config: HostedSipConfig) -> None:
        self.config = config
        http_client = None
        if not config.tls_verify:
            http_client = httpx.AsyncClient(verify=False)
        self.client = AsyncOpenAI(
            webhook_secret=config.webhook_secret,
            http_client=http_client,
        )
        self.websocket_connection_options: dict[str, Any] = {}
        if not config.tls_verify and is_https_base_url(os.environ.get("OPENAI_BASE_URL", "")):
            self.websocket_connection_options["ssl"] = ssl._create_unverified_context()
        self.calls: dict[str, ActiveCall] = {}
        self.processed_webhooks: set[str] = set()
        self.monitor_tasks: dict[str, asyncio.Task[None]] = {}
        self.lock = asyncio.Lock()

    def fastapi_app(self) -> FastAPI:
        app = FastAPI(
            title="OpenAI Hosted SIP Demo",
            description="Webhook and control-plane demo for OpenAI hosted SIP calls.",
            version="0.1.0",
        )

        @app.middleware("http")
        async def log_requests(request: Request, call_next):
            request_id = time.time_ns()
            client_host = request.client.host if request.client else "<unknown>"
            print(
                f"[req {request_id}] {request.method} {request.url.path}"
                f" from={client_host} query={request.url.query if request.url.query else ''}"
            )
            try:
                response = await call_next(request)
            except Exception as exc:
                print(f"[req {request_id}] exception={exc.__class__.__name__}:{exc}")
                raise
            print(f"[req {request_id}] status={response.status_code}")
            return response

        @app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        @app.get("/calls")
        async def list_calls() -> dict[str, list[dict[str, Any]]]:
            async with self.lock:
                return {"calls": [call.to_dict() for call in self.calls.values()]}

        @app.post("/webhook")
        async def webhook(request: Request) -> dict[str, str]:
            body = await request.body()
            body_text = body.decode("utf-8", errors="replace")
            webhook_id = request.headers.get("webhook-id") or "<missing>"
            webhook_timestamp = request.headers.get("webhook-timestamp") or "<missing>"
            webhook_signature = request.headers.get("webhook-signature") or "<missing>"
            print(
                "Incoming webhook request: "
                f"webhook-id={webhook_id} "
                f"timestamp={webhook_timestamp} "
                f"path={request.url.path}"
            )
            print(f"Incoming webhook headers: webhook-signature={webhook_signature}")
            print(f"Incoming webhook body (preview): {preview_text(body_text, 1024)}")
            headers = dict(request.headers)
            if self.config.webhook_secret:
                try:
                    event = self.client.webhooks.unwrap(body, headers)
                except InvalidWebhookSignatureError as exc:
                    print("Invalid OpenAI webhook signature.")
                    print(
                        f"  webhook-id={webhook_id} "
                        f"webhook-timestamp={webhook_timestamp}"
                    )
                    print(f"  webhook-signature={webhook_signature}")
                    print(f"  body={preview_text(body.decode('utf-8', errors='replace'), 512)}")
                    print(f"  error={exc}")
                    raise HTTPException(status_code=400, detail="Invalid webhook signature") from exc
            else:
                try:
                    event = json.loads(body_text)
                    if not isinstance(event, dict):
                        raise ValueError("webhook payload is not an object")
                except Exception as exc:
                    print(f"Failed parsing unsigned webhook body: {exc}")
                    raise HTTPException(status_code=400, detail="Invalid webhook payload") from exc

            event_type = get_event_attr(event, "type")
            event_id = str(get_event_attr(event, "id") or "")
            webhook_key = request.headers.get("webhook-id") or event_id
            print(
                f"Parsed webhook event: type={event_type} id={event_id} "
                f"webhook_key={webhook_key or '<missing>'}"
            )

            if webhook_key:
                async with self.lock:
                    if webhook_key in self.processed_webhooks:
                        print(f"Duplicate webhook ignored: webhook_key={webhook_key}")
                        return {"status": "duplicate", "event_id": event_id}
                    self.processed_webhooks.add(webhook_key)

            if event_type != "realtime.call.incoming":
                print(f"Ignored webhook event: {event_type}")
                return {"status": "ignored", "event_type": str(event_type)}

            call_id = extract_call_id(event)
            if not call_id:
                print(f"Missing call_id in webhook event: {preview_text(body_text, 300)}")
                raise HTTPException(status_code=400, detail="Missing call_id")

            sip_headers = extract_sip_headers(event)
            print_incoming_call(call_id, sip_headers)

            async with self.lock:
                existing = self.calls.get(call_id)
                if existing is None:
                    self.calls[call_id] = ActiveCall(
                        call_id=call_id,
                        status="incoming",
                        created_at=time.time(),
                        sip_headers=sip_headers,
                    )
                else:
                    existing.sip_headers = sip_headers

            print(f"Call created/updated in memory: call_id={call_id}")
            asyncio.create_task(self.accept_and_monitor_call(call_id))
            return {"status": "accepted_for_processing", "call_id": call_id}

        @app.post("/calls/{call_id}/hangup")
        async def hangup_call(call_id: str) -> dict[str, str]:
            await self.require_active_call(call_id)
            print(f"Manual hangup requested for call_id={call_id}")
            await self.client.realtime.calls.hangup(call_id)
            await self.set_call_status(call_id, "hangup_requested")
            return {"status": "hangup_requested", "call_id": call_id}

        @app.post("/calls/{call_id}/refer")
        async def refer_call(call_id: str, body: ReferRequest) -> dict[str, str]:
            await self.require_active_call(call_id)
            print(f"Manual refer requested for call_id={call_id}, target_uri={body.target_uri}")
            await self.client.realtime.calls.refer(call_id, target_uri=body.target_uri)
            await self.set_call_status(call_id, "refer_requested")
            return {
                "status": "refer_requested",
                "call_id": call_id,
                "target_uri": body.target_uri,
            }

        return app

    async def require_active_call(self, call_id: str) -> ActiveCall:
        async with self.lock:
            call = self.calls.get(call_id)
        if call is None:
            raise HTTPException(status_code=404, detail=f"Unknown call_id: {call_id}")
        return call

    async def set_call_status(self, call_id: str, status: str) -> None:
        async with self.lock:
            call = self.calls.get(call_id)
            if call is not None:
                print(f"Call status change: call_id={call_id} status={call.status} -> {status}")
                call.status = status

    async def set_call_event(self, call_id: str, event_type: str) -> None:
        async with self.lock:
            call = self.calls.get(call_id)
            if call is not None:
                call.last_event = event_type

    async def accept_and_monitor_call(self, call_id: str) -> None:
        await self.set_call_status(call_id, "accepting")
        print(f"Accept flow start: call_id={call_id}")
        try:
            await self.accept_call(call_id)
        except Exception as exc:
            await self.set_call_status(call_id, "accept_failed")
            print(f"Failed accepting hosted SIP call {call_id}: {exc.__class__.__name__}:{exc}")
            return

        await self.set_call_status(call_id, "accepted")
        print(f"Accept flow success: call_id={call_id}")
        task = asyncio.create_task(self.monitor_call(call_id))
        async with self.lock:
            self.monitor_tasks[call_id] = task

    async def accept_call(self, call_id: str) -> None:
        transcription: dict[str, str] = {"model": self.config.transcription_model}
        if self.config.transcription_language:
            transcription["language"] = self.config.transcription_language

        audio = {
            "input": {
                "transcription": transcription,
                "turn_detection": {
                    "type": "server_vad",
                    "create_response": True,
                    "interrupt_response": True,
                },
            },
            "output": {
                "voice": self.config.voice,
            },
        }

        print(
            "Accepting hosted SIP call: "
            f"call_id={call_id} model={self.config.model} voice={self.config.voice}"
        )
        response = await self.client.realtime.calls.accept(
            call_id,
            type="realtime",
            model=self.config.model,
            instructions=self.config.instructions,
            output_modalities=["audio"],
            audio=audio,
        )
        print(f"Accept API success: call_id={call_id} response={preview_object(response, 1200)}")

    async def monitor_call(self, call_id: str) -> None:
        print(f"Monitoring hosted SIP call: call_id={call_id} begin")
        try:
            print(f"Connecting realtime websocket: call_id={call_id}")
            async with self.client.realtime.connect(
                call_id=call_id,
                websocket_connection_options=self.websocket_connection_options,
            ) as connection:
                print(f"Realtime websocket connected: call_id={call_id}")
                if self.config.greeting:
                    print(f"Sending greeting: call_id={call_id}")
                    await connection.send(
                        {
                            "type": "response.create",
                            "response": {
                                "instructions": self.config.greeting,
                                "output_modalities": ["audio"],
                            },
                        }
                    )

                async for event in connection:
                    event_type = get_event_type(event)
                    print(f"Realtime stream event: call_id={call_id} event={event_type}")
                    await self.set_call_event(call_id, event_type)
                    if should_log_event(event_type, self.config.verbose_deltas):
                        print(f"Realtime call event: {format_event(event)}")
        except Exception as exc:
            await self.set_call_status(call_id, "monitor_error")
            print(
                "Hosted SIP call monitor stopped with error: "
                f"call_id={call_id} error={exc.__class__.__name__}:{exc}"
            )
            return

        await self.set_call_status(call_id, "ended")
        print(f"Hosted SIP call monitor ended: call_id={call_id}")


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required.")
    return value


def is_local_https_base_url(base_url: str) -> bool:
    parsed = urllib.parse.urlparse(base_url)
    return parsed.scheme == "https" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}


def is_https_base_url(base_url: str) -> bool:
    return urllib.parse.urlparse(base_url).scheme == "https"


def get_event_type(event: Any) -> str:
    return str(get_event_attr(event, "type") or "")


def get_event_attr(event: Any, name: str) -> Any:
    if isinstance(event, dict):
        return event.get(name)
    return getattr(event, name, None)


def event_to_dict(event: Any) -> dict[str, Any]:
    if isinstance(event, dict):
        return event
    if hasattr(event, "model_dump"):
        return event.model_dump(mode="json", exclude_none=True)
    return {"type": get_event_type(event)}


def extract_call_id(event: Any) -> str | None:
    data = get_event_attr(event, "data")
    call_id = get_event_attr(data, "call_id")
    return call_id if isinstance(call_id, str) else None


def extract_sip_headers(event: Any) -> list[dict[str, str]]:
    data = get_event_attr(event, "data")
    headers = get_event_attr(data, "sip_headers")
    if not isinstance(headers, list):
        return []

    normalized: list[dict[str, str]] = []
    for header in headers:
        name = get_event_attr(header, "name")
        value = get_event_attr(header, "value")
        if isinstance(name, str) and isinstance(value, str):
            normalized.append({"name": name, "value": value})
    return normalized


def sip_header(headers: list[dict[str, str]], name: str) -> str:
    for header in headers:
        if header["name"].lower() == name.lower():
            return header["value"]
    return "<missing>"


def print_incoming_call(call_id: str, sip_headers: list[dict[str, str]]) -> None:
    print(
        "Hosted SIP incoming call: "
        f"call_id={call_id} "
        f"from={sip_header(sip_headers, 'From')} "
        f"to={sip_header(sip_headers, 'To')} "
        f"sip_call_id={sip_header(sip_headers, 'Call-ID')}"
    )


def should_log_event(event_type: str, verbose_deltas: bool) -> bool:
    delta_events = {
        "conversation.item.input_audio_transcription.delta",
        "response.output_audio_transcript.delta",
    }
    if event_type in delta_events:
        return verbose_deltas
    return event_type in {
        "session.created",
        "session.updated",
        "input_audio_buffer.speech_started",
        "input_audio_buffer.speech_stopped",
        "input_audio_buffer.committed",
        "conversation.item.input_audio_transcription.completed",
        "response.created",
        "response.output_audio.done",
        "response.output_audio_transcript.done",
        "response.done",
        "error",
    }


def format_event(event: Any) -> str:
    payload = event_to_dict(event)
    event_type = str(payload.get("type") or "")
    if event_type == "error":
        return json.dumps(payload, ensure_ascii=False)
    if event_type == "session.created":
        session = payload.get("session") or {}
        return f"{event_type} id={session.get('id', '<unknown>')}"
    if event_type == "session.updated":
        return event_type
    if event_type == "conversation.item.input_audio_transcription.delta":
        return f"{event_type} delta={payload.get('delta')!r}"
    if event_type == "conversation.item.input_audio_transcription.completed":
        return f"{event_type} transcript={payload.get('transcript')!r}"
    if event_type == "response.output_audio_transcript.delta":
        return f"{event_type} delta={payload.get('delta')!r}"
    if event_type == "response.output_audio_transcript.done":
        return f"{event_type} transcript={payload.get('transcript')!r}"
    if event_type == "response.done":
        response = payload.get("response") or {}
        status = response.get("status", "<unknown>") if isinstance(response, dict) else "<unknown>"
        return f"{event_type} status={status}"
    return event_type


def preview_object(obj: Any, max_len: int) -> str:
    try:
        if hasattr(obj, "model_dump"):
            payload = obj.model_dump(exclude_none=True)
        elif isinstance(obj, (dict, list, str, int, float, bool)) or obj is None:
            payload = obj
        else:
            payload = str(obj)
        text = json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        text = str(obj)
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"...(<+{len(text)-max_len} chars)"


def preview_text(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"...(<+{len(text)-max_len} chars)"


def main(
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        envvar="HOSTED_SIP_HOST",
        help="Webhook/control server bind host.",
    ),
    port: int = typer.Option(
        8080,
        "--port",
        envvar="HOSTED_SIP_PORT",
        help="Webhook/control server bind port.",
    ),
    model: str = typer.Option(
        "deepseek-v4-flash",
        "--model",
        envvar="HOSTED_SIP_MODEL",
        help="Realtime model used when accepting hosted SIP calls.",
    ),
    voice: str = typer.Option(
        "paimon",
        "--voice",
        envvar="HOSTED_SIP_VOICE",
        help="Realtime voice used when accepting hosted SIP calls.",
    ),
    instructions: str = typer.Option(
        DEFAULT_INSTRUCTIONS,
        "--instructions",
        envvar="HOSTED_SIP_INSTRUCTIONS",
        help="Realtime session instructions.",
    ),
    greeting: str = typer.Option(
        "",
        "--greeting",
        envvar="HOSTED_SIP_GREETING",
        help="Optional greeting generated after the call is accepted.",
    ),
    transcription_model: str = typer.Option(
        "gpt-4o-mini-transcribe",
        "--transcription-model",
        envvar="HOSTED_SIP_TRANSCRIPTION_MODEL",
        help="Input audio transcription model.",
    ),
    transcription_language: str = typer.Option(
        "",
        "--transcription-language",
        envvar="HOSTED_SIP_TRANSCRIPTION_LANGUAGE",
        help="Optional ISO-639-1 language hint for input audio transcription.",
    ),
    webhook_secret: str = typer.Option(
        "",
        "--webhook-secret",
        envvar="HOSTED_SIP_WEBHOOK_SECRET",
        help="OpenAI-compatible webhook secret. Keep empty to accept unsigned webhooks.",
    ),
    verbose_deltas: bool = typer.Option(
        False,
        "--verbose-deltas",
        envvar="HOSTED_SIP_VERBOSE_DELTAS",
        help="Print transcription delta events.",
    ),
    tls_verify: bool | None = typer.Option(
        None,
        "--tls-verify/--no-tls-verify",
        envvar="HOSTED_SIP_TLS_VERIFY",
        help=(
            "Verify TLS for OPENAI_BASE_URL. Defaults to disabled for local HTTPS "
            "Nexus endpoints and enabled otherwise."
        ),
    ),
) -> None:
    try:
        resolved_webhook_secret = webhook_secret.strip() or os.environ.get("NEXUS_REALTIME_WEBHOOK_SECRET", "").strip()
        openai_base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
        resolved_tls_verify = tls_verify
        if resolved_tls_verify is None:
            resolved_tls_verify = not is_local_https_base_url(openai_base_url)
        config = HostedSipConfig(
            webhook_secret=resolved_webhook_secret or None,
            model=model,
            voice=voice,
            instructions=instructions,
            greeting=greeting,
            transcription_model=transcription_model,
            transcription_language=transcription_language,
            verbose_deltas=verbose_deltas,
            tls_verify=resolved_tls_verify,
        )
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc)) from exc

    print(
        "OpenAI hosted SIP demo starting: "
        f"base_url={os.environ.get('OPENAI_BASE_URL','<unknown>')} "
        f"model={config.model} voice={config.voice} "
        f"webhook=http://{host}:{port}/webhook"
    )
    print(
        "OpenAI-compatible client TLS verification: "
        + ("enabled" if config.tls_verify else "disabled")
    )
    print(
        "Webhook signature check: "
        + ("enabled" if config.webhook_secret else "disabled")
    )
    if not config.webhook_secret:
        print("Set HOSTED_SIP_WEBHOOK_SECRET or NEXUS_REALTIME_WEBHOOK_SECRET to enable verification.")
    print("Configure SIP trunk target as: sip:$PROJECT_ID@sip.api.openai.com;transport=tls")

    app = HostedSipApp(config).fastapi_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    typer.run(main)

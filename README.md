# Nexus

OpenAI-compatible ASR/Realtime/Chat/TTS server with a clean layered architecture.

## Architecture

- `src/nexus/api`: FastAPI HTTP/WebSocket entrypoints.
- `src/nexus/application`: use cases, orchestration, protocol parsing/writing, DI container.
- `src/nexus/domain`: session/domain state.
- `src/nexus/infrastructure`: adapters for OpenAI/gRPC/MCP clients.

## Realtime refactor highlights

- Inbound WebSocket events are validated with `TypeAdapter(RealtimeClientEvent)`.
- Outbound server events are validated with `TypeAdapter(RealtimeServerEvent)` before send.
- Event dispatch is registry-based (`application.realtime.dispatch`), replacing `if/elif` chains.
- Realtime worker logic is split into orchestrators (`transcription_worker`, `response_orchestrator`, `tool_call_orchestrator`).
- MCP failure paths now emit `mcp_list_tools.failed` and `response.mcp_call.failed`.
- Realtime audio contract is strict `audio/pcm` at `24000Hz` for both input and output.
- ASR path performs streaming resampling `24kHz -> 16kHz` before gRPC inference.
- GA-aligned browser WebRTC entrypoints are available at `POST /v1/realtime/client_secrets`, `POST /v1/realtime/calls`, and `wss /v1/realtime?call_id=...`.
- The legacy `wss /v1/realtime?model=...` transport remains available for direct websocket clients.
- When `NEXUS_REALTIME_API_KEY` is configured, realtime HTTP and websocket endpoints require `Authorization: Bearer ...`; otherwise local dev remains open.

## Official WebRTC Flow

1. Call `POST /v1/realtime/client_secrets` with an optional session config to mint an ephemeral `ek_...` token.
2. In the browser, create a WebRTC offer and `POST /v1/realtime/calls` with either:
   - `Authorization: Bearer ek_...` and `Content-Type: application/sdp`, or
   - `Authorization: Bearer $NEXUS_REALTIME_API_KEY` and `multipart/form-data` containing `sdp` plus optional `session`.
3. Use the returned SDP answer to finish the peer connection.
4. Open an optional sideband websocket with `wss://host/v1/realtime?call_id=rtc_...` to monitor or control the same session from the server side.

Current v1 scope:

- `session.type` only supports `realtime`
- SIP-style `accept/reject/hangup/refer` routes are not implemented
- client secrets and call state are stored in-process only

## Testing

Default automated suite:

```bash
poetry run pytest -q
```

Manual/E2E scripts live under `tests/e2e` and are excluded from default pytest runs.

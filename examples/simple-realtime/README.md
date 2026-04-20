# Simple Realtime Demo

Small React + Node.js test app for manually exercising a browser realtime flow against either Nexus or the OpenAI Realtime API.

## Features

- standalone `pnpm` app under `examples/simple-realtime`
- React frontend plus Node.js websocket backend in one app
- backend-managed connection to an OpenAI-compatible realtime upstream
- editable app websocket URL and browser model default via env
- output mode toggle between `text` and `audio`
- microphone capture with browser-side `24kHz mono PCM16` encoding
- normalized timeline for transcription and assistant responses
- raw upstream event inspector
- audio playback for backend-forwarded output deltas

## Environment

Create a local env file if you want different defaults:

```bash
cp .env.example .env.local
```

Available vars:

- `VITE_APP_WS_URL` default app websocket endpoint
- `VITE_REALTIME_MODEL` default realtime model
- `VITE_REALTIME_VOICE` default realtime voice
- `SIMPLE_REALTIME_SERVER_PORT` Node backend port
- `SIMPLE_REALTIME_UPSTREAM_BASE_URL` upstream API base URL, typically ending in `/v1`
- `SIMPLE_REALTIME_OPENAI_API_KEY` optional bearer token attached to upstream websocket requests

## Run

```bash
pnpm install
pnpm dev
```

Then open the printed Vite URL in a browser and start recording.

Example upstream modes:

```bash
# Nexus-compatible upstream without auth
SIMPLE_REALTIME_UPSTREAM_BASE_URL=http://127.0.0.1:8000/v1 \
pnpm dev

# OpenAI-compatible upstream with auth
SIMPLE_REALTIME_UPSTREAM_BASE_URL=https://api.openai.com/v1 \
SIMPLE_REALTIME_OPENAI_API_KEY=sk-... \
pnpm dev
```

## Verify

```bash
pnpm test
pnpm build
```

## Notes

- The demo sends `session.update` immediately after websocket open.
- The browser must provide a model; the backend no longer applies a fallback model when the field is empty.
- The frontend speaks an app-specific websocket protocol to the local Node backend.
- The backend speaks realtime websocket events to the configured upstream.
- Nexus and OpenAI use the same upstream URL parsing model: start from an API base URL, normalize to websocket, append `/realtime`, then add `?model=...`.
- The backend only adds `Authorization: Bearer ...` when `SIMPLE_REALTIME_OPENAI_API_KEY` is present.
- The backend forces `24kHz mono PCM16` and enables server-side VAD for turn detection and automatic responses.
- `audio` mode depends on the configured upstream supporting audio output.

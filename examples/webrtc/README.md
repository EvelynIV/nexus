# Realtime WebRTC Demo

Standalone React + Node.js demo under `examples/webrtc` for talking to a GA-compatible Realtime Runtime over WebRTC, including OpenAI official endpoints or a local Nexus server.

## Architecture

- browser UI and local token service in one `pnpm` app
- local backend exposes `GET /api/health` and `POST /api/token`
- backend calls `POST ${WEBRTC_REALTIME_BASE_URL}/v1/realtime/client_secrets`
- browser uses the returned ephemeral secret to `POST ${VITE_REALTIME_BASE_URL}/v1/realtime/calls`
- microphone input is sent as a WebRTC audio track
- session events flow through the `oai-events` data channel
- remote model audio is played directly from `RTCPeerConnection.ontrack`

## Environment

Create a local env file:

```bash
cp .env.example .env
```

Available vars:

- `WEBRTC_OPENAI_API_KEY` required server-side OpenAI API key
- `WEBRTC_REALTIME_BASE_URL` backend Realtime base URL, default `https://api.openai.com`
- `WEBRTC_SERVER_PORT` local backend port, default `8790`
- `WEBRTC_UPSTREAM_CONNECT_TIMEOUT_MS` backend timeout for OpenAI token requests, default `15000`
- `VITE_REALTIME_BASE_URL` browser-side Realtime base URL, default `https://api.openai.com`
- `VITE_REALTIME_MODEL` default browser model, default `gpt-realtime`
- `VITE_REALTIME_VOICE` default browser voice, default `alloy`
- `HTTPS_PROXY` / `HTTP_PROXY` optional proxy for the local backend when it cannot reach the upstream Realtime base URL directly

## Run

```bash
pnpm install
pnpm dev
```

Then open the printed Vite URL in a browser, connect, and toggle the microphone on when you want server VAD to listen.

To use local Nexus on `http://localhost:8000`, set both `WEBRTC_REALTIME_BASE_URL` and `VITE_REALTIME_BASE_URL` to `http://localhost:8000`.

## Verify

```bash
pnpm test
pnpm build
```

## Notes

- This app targets the current OpenAI GA Realtime WebRTC flow and can point at any compatible base URL.
- The browser never receives the long-lived API key. It only gets a short-lived client secret from `/api/token`.
- `outputMode` can be switched live through `session.update`.
- Voice changes require reconnecting because OpenAI does not allow changing the voice after audio has started in a session.
- The token service first requests `audio.input.transcription` with `gpt-4o-mini-transcribe`, then retries once without it if the model rejects that configuration.
- In restricted networks, set `HTTPS_PROXY` or `HTTP_PROXY` for the Node backend before running `pnpm dev`.

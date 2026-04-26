# Realtime SIP / Asterisk Demos

This directory contains SIP-related realtime demos. There are three different
integration shapes:

```text
Nexus native Asterisk ingress:
SIP/FXO call -> Asterisk Stasis(nexus) -> Nexus ARI ExternalMedia -> Nexus Realtime

Standalone bridge demo:
SIP/FXO call -> Asterisk Stasis(voicebot) -> Python RTP bridge -> OpenAI/Nexus Realtime

OpenAI hosted SIP control-plane demo:
SIP trunk -> OpenAI SIP endpoint -> OpenAI webhook -> Python control server
```

## OpenAI hosted SIP control demo (Nexus-compatible)

When this control server receives calls from this repository's Nexus, webhook events are emitted by Nexus as `realtime.call.incoming`.

Nexus must be configured with a webhook URL and a shared secret:

```bash
export NEXUS_REALTIME_WEBHOOK_URL=http://your-control-server:8080/webhook
export NEXUS_REALTIME_WEBHOOK_SECRET=your_shared_webhook_secret
```

Run the demo with the same secret (or rely on Nexus fallback environment variable):

```bash
export HOSTED_SIP_WEBHOOK_SECRET=your_shared_webhook_secret
# or set HOSTED_SIP_WEBHOOK_SECRET omitted; script will reuse NEXUS_REALTIME_WEBHOOK_SECRET
poetry run python examples/sip/openai_hosted_sip_demo.py --host 0.0.0.0 --port 8080
```

Use the Nexus native Asterisk ingress when you want Nexus to expose OpenAI-like
`realtime.call.incoming` + `/v1/realtime/calls/{call_id}/accept` behavior while
Asterisk remains the SIP/RTP gateway.

## Nexus Native Asterisk Ingress

Nexus does not listen on SIP port 5060. Asterisk owns SIP signaling and routes
incoming calls into `Stasis(nexus)`. Nexus listens to ARI, emits a
`realtime.call.incoming` webhook, and only answers the call after your control
server calls `/v1/realtime/calls/{call_id}/accept`.

Dialplan:

```ini
[from-fxo]
exten => s,1,NoOp(Incoming call to Nexus Asterisk ingress)
 same => n,Stasis(nexus)
 same => n,Hangup()
```

Nexus environment:

```bash
export NEXUS_ASTERISK_INGRESS_ENABLED=true
export NEXUS_ASTERISK_ARI_URL=http://127.0.0.1:8088/ari
export NEXUS_ASTERISK_ARI_USER=voicebot
export NEXUS_ASTERISK_ARI_PASSWORD=12345678
export NEXUS_ASTERISK_STASIS_APP=nexus
export NEXUS_ASTERISK_EXTERNAL_HOST=127.0.0.1
export NEXUS_ASTERISK_CODEC=ulaw
export NEXUS_ASTERISK_RTP_PORT_START=4000
export NEXUS_ASTERISK_RTP_PORT_END=4099
```

Optional OpenAI-compatible incoming-call webhook:

```bash
export NEXUS_REALTIME_WEBHOOK_URL=https://your-control-server.example/webhook
export NEXUS_REALTIME_WEBHOOK_SECRET=local_webhook_secret
```

Start Nexus:

```bash
poetry run nexus model-bin/work.yaml
```

Accept a pending SIP call manually when no webhook control server is configured:

```bash
curl -X POST "https://localhost:8000/v1/realtime/calls/$CALL_ID/accept" \
  -H "Authorization: Bearer $NEXUS_REALTIME_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "type": "realtime",
        "model": "deepseek-v4-flash",
        "instructions": "You are a concise phone assistant.",
        "output_modalities": ["audio"],
        "audio": {
          "input": {"format": {"type": "audio/pcm", "rate": 24000}},
          "output": {"format": {"type": "audio/pcm", "rate": 24000}, "voice": "paimon"}
        }
      }'
```

## Standalone RTP Bridge Demo

This manual demo connects an Asterisk SIP call to a Realtime API through ARI
ExternalMedia. It is useful for local experimentation, but it is not the native
Nexus ingress path.

Expected flow:

```text
SIP/FXO call -> Asterisk Stasis(voicebot) -> ExternalMedia RTP -> OpenAI Realtime
```

The Python script does not implement SIP. Asterisk still owns SIP signaling and
creates an RTP ExternalMedia leg. The script forwards that RTP audio to OpenAI
Realtime and sends model audio back to Asterisk.

### Asterisk Dialplan

Route the call into the `voicebot` Stasis app:

```ini
[from-fxo]
exten => s,1,NoOp(FXO incoming call to OpenAI Realtime demo)
 same => n,Answer()
 same => n,Stasis(voicebot)
 same => n,Hangup()
```

Reload and confirm:

```bash
sudo asterisk -rx "dialplan show from-fxo"
```

### Environment

Set your OpenAI API key in the shell before running the demo:

```bash
export OPENAI_API_KEY=sk-...
```

The script does not load `.env` files. If you keep values in `.env`, source them
from your shell or configure your debugger to inject them before launching the
script.

Do not commit API keys to this repository. If a key was pasted into chat,
rotate it in the OpenAI dashboard before using this demo.

Common environment variables:

```bash
export OPENAI_BASE_URL=https://localhost:8000/v1
export OPENAI_MODEL=deepseek-v4-flash
export OPENAI_VOICE=paimon
```

`SIP_OPENAI_BASE_URL`, `SIP_REALTIME_MODEL`, and `SIP_REALTIME_VOICE` are also
supported and take the same role for this demo. Run `--help` to see every
option and its environment variable.

For local HTTPS/WSS testing, a self-signed localhost certificate can live under
`model-bin`:

```text
model-bin/localhost.crt
model-bin/localhost.key
model-bin/localhost-openssl.cnf
```

`model-bin` is gitignored. The demo disables Realtime WebSocket certificate
verification so this local self-signed certificate works without installing it
into the system trust store.

Start Nexus with HTTPS/WSS:

```bash
poetry run nexus model-bin/work.yaml \
  --host 127.0.0.1 \
  --port 8000 \
  --ssl-certfile model-bin/localhost.crt \
  --ssl-keyfile model-bin/localhost.key
```

### Run

From the repository root:

```bash
poetry run python examples/sip/openai_realtime_phone_demo.py
```

Defaults:

```text
ARI URL:      http://127.0.0.1:8088/ari
ARI user:     voicebot
ARI password: 12345678
ARI app:      voicebot
RTP bind:     127.0.0.1:4000
Codec:        ulaw
OpenAI model: deepseek-v4-flash
Voice:        paimon
```

Then place a call into the SIP/FXO route. The script answers the call through
ARI, creates an ExternalMedia RTP channel, opens an OpenAI Realtime WebSocket,
and bridges G.711 audio in both directions.

On startup, the script prints the resolved Realtime base URL, model, and voice.

If an old call is already parked in the same Stasis app:

```bash
poetry run python examples/sip/openai_realtime_phone_demo.py --hangup-existing-calls
```

If your SIP leg negotiates A-law:

```bash
poetry run python examples/sip/openai_realtime_phone_demo.py --codec alaw
```

Customize the assistant:

```bash
poetry run python examples/sip/openai_realtime_phone_demo.py \
  --model deepseek-v4-flash \
  --voice paimon \
  --instructions "You are a concise phone assistant. Reply in Chinese." \
  --greeting "Greet the caller briefly and ask how you can help."
```

### Verify

Static check:

```bash
poetry run python -m py_compile examples/sip/openai_realtime_phone_demo.py
```

Manual integration checks:

- confirm the script prints `ARI connected`
- confirm the script prints `Attached call to OpenAI media bridge` after a call reaches Stasis
- confirm `OpenAI event: session.created` and `OpenAI event: session.updated`
- speak into the phone and confirm `input_audio_buffer.speech_started` / `speech_stopped`
- confirm user speech transcription prints as `Realtime transcription: '...'`
- confirm model audio is heard on the call

### Troubleshooting

- Missing `OPENAI_API_KEY`: export it in the same shell before running `poetry run`.
- ARI connection failure: verify `http.conf`, `ari.conf`, credentials, and `http://127.0.0.1:8088/ari`.
- No Stasis events: verify the dialplan enters `Stasis(voicebot)`.
- No RTP audio: check that no other process is using the selected RTP port and that Asterisk can reach `--rtp-host:--rtp-port`.
- One-way or distorted audio: make sure `--codec` matches the SIP media format, usually `ulaw` in North America/Japan or `alaw` in many EMEA setups.
- `SSL: WRONG_VERSION_NUMBER`: the client reached a plain HTTP server as WSS. Run the local OpenAI-compatible server with TLS using `model-bin/localhost.crt` and `model-bin/localhost.key`, or use a plain WebSocket endpoint if your server supports it directly.
- Realtime errors: the script prints OpenAI `error` events to stderr; check model access, API key validity, and network connectivity.

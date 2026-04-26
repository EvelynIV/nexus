# OpenAI Realtime SIP Demo

This directory contains a manual demo that connects an Asterisk SIP call to the
OpenAI Realtime API through ARI ExternalMedia.

Expected flow:

```text
SIP/FXO call -> Asterisk Stasis(voicebot) -> ExternalMedia RTP -> OpenAI Realtime
```

The Python script does not implement SIP. Asterisk still owns SIP signaling and
creates an RTP ExternalMedia leg. The script forwards that RTP audio to OpenAI
Realtime and sends model audio back to Asterisk.

## Asterisk Dialplan

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

## Environment

Set your OpenAI API key in the shell before running the demo:

```bash
export OPENAI_API_KEY=sk-...
```

Do not commit API keys to this repository. If a key was pasted into chat,
rotate it in the OpenAI dashboard before using this demo.

## Run

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
OpenAI model: gpt-realtime
Voice:        alloy
```

Then place a call into the SIP/FXO route. The script answers the call through
ARI, creates an ExternalMedia RTP channel, opens an OpenAI Realtime WebSocket,
and bridges G.711 audio in both directions.

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
  --model gpt-realtime \
  --voice alloy \
  --instructions "You are a concise phone assistant. Reply in Chinese." \
  --greeting "Greet the caller briefly and ask how you can help."
```

## Verify

Static check:

```bash
poetry run python -m py_compile examples/sip/openai_realtime_phone_demo.py
```

Manual integration checks:

- confirm the script prints `ARI connected`
- confirm the script prints `Attached call to OpenAI media bridge` after a call reaches Stasis
- confirm `OpenAI event: session.created` and `OpenAI event: session.updated`
- speak into the phone and confirm `input_audio_buffer.speech_started` / `speech_stopped`
- confirm model audio is heard on the call

## Troubleshooting

- Missing `OPENAI_API_KEY`: export it in the same shell before running `poetry run`.
- ARI connection failure: verify `http.conf`, `ari.conf`, credentials, and `http://127.0.0.1:8088/ari`.
- No Stasis events: verify the dialplan enters `Stasis(voicebot)`.
- No RTP audio: check that no other process is using the selected RTP port and that Asterisk can reach `--rtp-host:--rtp-port`.
- One-way or distorted audio: make sure `--codec` matches the SIP media format, usually `ulaw` in North America/Japan or `alaw` in many EMEA setups.
- Realtime errors: the script prints OpenAI `error` events to stderr; check model access, API key validity, and network connectivity.

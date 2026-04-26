# SIP / ARI ExternalMedia CLI demo

This directory contains manual demos for testing Asterisk SIP audio through
ARI ExternalMedia.

## CLI as a local audio endpoint

`cli_external_media_demo.py` uses your local microphone and speaker as an RTP
media endpoint. It does not implement SIP itself. Asterisk still handles SIP,
and the script only handles the RTP audio leg created by ARI ExternalMedia.

Expected flow:

```text
SIP/FXO call -> Asterisk Stasis(voicebot) -> ExternalMedia RTP -> CLI mic/speaker
```

## Asterisk dialplan

The local machine has already been configured to route FXO calls into the
`voicebot` Stasis app:

```ini
[from-fxo]
exten => s,1,NoOp(FXO incoming call to CLI ExternalMedia demo)
 same => n,Answer()
 same => n,Stasis(voicebot)
 same => n,Hangup()
```

Then reload:

```bash
sudo asterisk -rx "dialplan show from-fxo"
```

## Run

From the repo root:

```bash
poetry run python tests/test_sip/cli_external_media_demo.py
```

Then place a call into the FXO/SIP route. The Python script connects to ARI,
creates the ExternalMedia RTP leg when the call reaches `Stasis(voicebot)`,
and bridges the call audio to the local microphone and speaker.

Defaults match the local Asterisk config used during setup:

```text
ARI URL:      http://127.0.0.1:8088/ari
ARI user:     voicebot
ARI password: 12345678
ARI app:      voicebot
RTP bind:     127.0.0.1:4000
Codec:        ulaw
Audio:        8000 Hz mono, 20 ms frames
```

List audio devices:

```bash
poetry run python tests/test_sip/cli_external_media_demo.py --list-devices
```

Pick devices:

```bash
poetry run python tests/test_sip/cli_external_media_demo.py \
  --input-device 1 \
  --output-device 2
```

If your SIP leg negotiates A-law instead of u-law:

```bash
poetry run python tests/test_sip/cli_external_media_demo.py --codec alaw
```

## Notes

- The script waits for a `StasisStart` event. Start it first, then place a call
  into the Asterisk context that runs `Stasis(voicebot)`.
- If there is no audio, verify `http show status`, `pjsip show contacts`, and
  that no other process is using the selected RTP port.
- This is a manual integration demo, not part of the default pytest suite.

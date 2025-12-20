from __future__ import annotations

import ssl
import base64
import asyncio

import librosa
import numpy as np
import sounddevice as sd

from openai import AsyncOpenAI
from openai.resources.realtime.realtime import AsyncRealtimeConnection

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 3200  # bytes per chunk
CHUNK_FRAMES = CHUNK_SIZE // 2  # 16-bit mono => 2 bytes per frame


async def wait_for_enter(stop_event: asyncio.Event):
    await asyncio.to_thread(input, "Press Enter to stop recording...\n")
    stop_event.set()


async def send_audio_from_mic(
    connection: AsyncRealtimeConnection, stop_event: asyncio.Event
):
    """Stream microphone audio as PCM16 chunks."""
    loop = asyncio.get_running_loop()
    audio_queue: asyncio.Queue[np.ndarray] = asyncio.Queue()

    input_rate = int(sd.query_devices(kind="input")["default_samplerate"])

    def callback(indata: np.ndarray, frames: int, time, status):
        if status:
            print(f"[Audio Status] {status}")
        loop.call_soon_threadsafe(audio_queue.put_nowait, indata.copy())

    # Start a streaming response before sending audio
    await connection.send({"type": "response.create", "response": {}})

    print(f"Mic input rate: {input_rate} Hz, target: {SAMPLE_RATE} Hz")
    with sd.InputStream(
        channels=CHANNELS,
        samplerate=input_rate,
        dtype="float32",
        blocksize=CHUNK_FRAMES,
        callback=callback,
    ):
        stop_task = asyncio.create_task(wait_for_enter(stop_event))
        try:
            while not stop_event.is_set():
                try:
                    data = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                mono = data.squeeze()
                if input_rate != SAMPLE_RATE:
                    mono = librosa.resample(mono, orig_sr=input_rate, target_sr=SAMPLE_RATE)

                pcm16 = np.clip(mono, -1.0, 1.0)
                pcm16 = (pcm16 * 32767.0).astype(np.int16).tobytes()

                await connection.send(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(pcm16).decode("utf-8"),
                    }
                )
        finally:
            stop_task.cancel()
            try:
                await stop_task
            except asyncio.CancelledError:
                pass

    print("\nAudio sent, committing buffer...")
    await connection.send({"type": "input_audio_buffer.commit"})


async def receive_responses(connection: AsyncRealtimeConnection, done_event: asyncio.Event):
    """Receive and print realtime responses."""
    async for event in connection:
        event_type = event.type

        if event_type == "session.created":
            print(f"[Session] Created: {event.session.id}")
        elif event_type == "response.created":
            print(f"[Response] Created: {event.response.id}")
        elif event_type == "response.audio_transcript.delta":
            print(f"{event.delta}")
        elif event_type == "response.audio_transcript.done":
            print()
            print(f"[Transcript Done] {event.transcript}")
        elif event_type == "response.text.delta":
            print(f"{event.delta}")
        elif event_type == "response.text.done":
            print()
            print(f"[Text Done] {event.text}")
        elif event_type == "response.done":
            print("[Response] Done!")
            done_event.set()
            break
        elif event_type == "error":
            print(f"[Error] {event}")
            done_event.set()
            break
        else:
            print(f"[Event] {event_type}")


async def main():
    # Create insecure SSL context for local testing
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    client = AsyncOpenAI(
        base_url="https://localhost:8000/v1",
        api_key="test_api_key",
    )

    async with client.beta.realtime.connect(
        model="gpt-4o-realtime-preview",
        websocket_connection_options={"ssl": ssl_context},
    ) as connection:
        print("Connected to realtime API")

        done_event = asyncio.Event()
        stop_event = asyncio.Event()

        receive_task = asyncio.create_task(receive_responses(connection, done_event))

        await send_audio_from_mic(connection, stop_event)

        await done_event.wait()
        receive_task.cancel()
        try:
            await receive_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    asyncio.run(main())
import os

import dotenv
import httpx
import openai

dotenv.load_dotenv()

client = openai.OpenAI(
    base_url=os.getenv("TEST_BASR_URL", "http://localhost:10002/v1"),
    api_key=os.getenv("TEST_API_KEY", "dummy_api_key"),
)

audio_file_path = "data-bin/huaqiang/403369728_nb2-1-30280_left_16k.wav"

with open(audio_file_path, "rb") as audio_file:
    stream = client.audio.transcriptions.create(
        file=audio_file,
        model="gpt-4o-transcribe",
        stream=True,  # 👈 关键
        language="zh",
    )

    print("流式识别结果：")
    for event in stream:
        # 兼容 OpenAI / vLLM / FastAPI 实现
        if hasattr(event, "text") and event.text:
            print(event.text, end="", flush=True)

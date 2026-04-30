import os

import dotenv
import httpx
import openai


dotenv.load_dotenv()
client = openai.OpenAI(
    base_url=os.getenv("TEST_BASE_URL", "http://localhost:10002/v1"),
    api_key=os.getenv("TEST_API_KEY", "dummy_api_key"),
    http_client=httpx.Client(verify=False),
)

response = client.responses.create(
    model=os.getenv("TEST_MODEL", "deepseek-v4-flash"),
    instructions="你是一个有帮助的中文助手。",
    input="介绍一下你自己。",
)

print(response.output_text)

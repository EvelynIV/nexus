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

model = os.getenv("TEST_MODEL", "deepseek-v4-flash")
tools = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "获取一个地点的天气，用户应该先提供一个地点。",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "城市名，例如：北京，上海，广州",
                }
            },
            "required": ["location"],
        },
        "strict": False,
    },
]

response = client.responses.create(
    model=model,
    instructions="你必须用中文回答我",
    input="今天金华的天气怎么样？",
    tools=tools,
)
function_call = next(item for item in response.output if item.type == "function_call")

follow_up = client.responses.create(
    model=model,
    previous_response_id=response.id,
    input=[
        {
            "type": "function_call_output",
            "call_id": function_call.call_id,
            "output": "来了外星人，密密麻麻的全是外星飞船",
        }
    ],
    tools=tools,
)

print(follow_up.output_text)

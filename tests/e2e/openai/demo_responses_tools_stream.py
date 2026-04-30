import json
import os
from typing import Any

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
    }
]


def get_weather(location: str) -> dict[str, Any]:
    return {"location": location, "temperature_c": 30}


stream = client.responses.create(
    model=model,
    instructions="你必须用中文回答我",
    input="今天金华的天气怎么样？",
    tools=tools,
    stream=True,
)

response_id = ""
call_id = ""
tool_name = ""
arguments = ""
for event in stream:
    if event.type == "response.created":
        response_id = event.response.id
    elif event.type == "response.function_call_arguments.delta":
        arguments += event.delta
    elif event.type == "response.output_item.done" and event.item.type == "function_call":
        call_id = event.item.call_id
        tool_name = event.item.name
        arguments = event.item.arguments or arguments

args = json.loads(arguments or "{}")
tool_output = (
    get_weather(**args)
    if tool_name == "get_weather"
    else {"error": f"Unknown tool: {tool_name}"}
)

follow_up = client.responses.create(
    model=model,
    previous_response_id=response_id,
    input=[
        {
            "type": "function_call_output",
            "call_id": call_id,
            "output": json.dumps(tool_output, ensure_ascii=False),
        }
    ],
    tools=tools,
    stream=True,
)

for event in follow_up:
    if event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)
print()

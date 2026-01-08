import os

import dotenv
import openai

dotenv.load_dotenv()
client = openai.OpenAI(
    base_url=os.getenv("TEST_BASR_URL", "http://localhost:10002/v1"),
    api_key=os.getenv("TEST_API_KEY", "dummy_api_key"),
)

tools = [
    {
        "type": "function",
        "function": {
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
        },
    },
]


def send_messages(messages):
    response = client.chat.completions.create(
        # model="deepseek-chat",
        model="Qwen/Qwen3-8B",
        messages=messages,
        tools=tools,
    )
    return response.choices[0].message


system_prompt = "你必须用中文回答我"
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "今天金华的天气怎么样？"},
]
message = send_messages(messages)
print(f"User>\t {messages}")

tool = message.tool_calls[0]
messages.append(message)

messages.append({"role": "tool", "tool_call_id": tool.id, "content": "来了外星人，密密麻麻的全是外星飞船"})
message = send_messages(messages)
print(f"Model>\t {message.content}")

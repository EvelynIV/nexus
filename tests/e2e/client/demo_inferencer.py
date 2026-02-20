from nexus.infrastructure.chat import Inferencer

# 初始化推理器
inferencer = Inferencer(
    base_url="http://192.168.0.200:50017/v1",
    api_key="<token>",
    model="mt01"
)

# 非流式调用
print("非流式响应:")
response = inferencer.chat("讲个笑话")
print(response)

print("\n" + "="*50 + "\n")

# 流式调用
print("流式响应:")
for chunk in inferencer.chat_stream("讲个笑话"):
    print(chunk, end="", flush=True)
print("\n")

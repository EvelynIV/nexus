from nexus.infrastructure.responses import Inferencer


inferencer = Inferencer(
    base_url="http://127.0.0.1:43000/v1",
    api_key="no-key",
)

print("非流式响应:")
response = inferencer.create(
    model="deepseek-v4-flash",
    input="讲个笑话",
)
print(response)

print("\n" + "=" * 50 + "\n")

print("流式响应:")
for event in inferencer.create(
    model="deepseek-v4-flash",
    input="讲个笑话",
    stream=True,
):
    print(event)

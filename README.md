# Nexus

兼容 OpenAI 的 ASR/Realtime/Chat/TTS 服务器，采用清晰的分层架构。

## 架构

- `src/nexus/api`: FastAPI HTTP/WebSocket 入口。
- `src/nexus/application`: 用例、编排、协议解析/写入、DI 容器。
- `src/nexus/domain`: 会话/领域状态。
- `src/nexus/infrastructure`: OpenAI/gRPC/MCP 客户端适配器。

## Realtime 重构亮点

- 入站 WebSocket 事件使用 `TypeAdapter(RealtimeClientEvent)` 校验。
- 出站服务端事件在发送前使用 `TypeAdapter(RealtimeServerEvent)` 校验。
- 事件分发基于注册表（`application.realtime.dispatch`），替代 `if/elif` 链。
- Realtime worker 逻辑拆分为多个编排器（`transcription_worker`、`response_orchestrator`、`tool_call_orchestrator`）。
- MCP 失败路径现在会发出 `mcp_list_tools.failed` 和 `response.mcp_call.failed`。
- Realtime 音频契约对输入和输出都严格使用 `24000Hz` 的 `audio/pcm`。
- ASR 路径在 gRPC 推理前执行流式重采样 `24kHz -> 16kHz`。
- 与 GA 对齐的浏览器 WebRTC 入口位于 `POST /v1/realtime/client_secrets`、`POST /v1/realtime/calls` 和 `wss /v1/realtime?call_id=...`。
- 可选 Asterisk ARI ExternalMedia 电话入口：Asterisk 进 `Stasis(nexus)` 后，Nexus 生成 `sip_...` call_id、发送 `realtime.call.incoming` webhook，并通过 `/accept` 把 G.711 RTP 接到 Realtime session。
- 兼容 OpenAI 的呼叫控制接口位于 `POST /v1/realtime/calls/{call_id}/accept`、`/reject`、`/refer` 和 `/hangup`，支持 WebRTC `rtc_...` 和 Asterisk `sip_...` call。
- 旧版 `wss /v1/realtime?model=...` 传输仍可供直接 WebSocket 客户端使用。
- 配置 `NEXUS_REALTIME_API_KEY` 后，realtime HTTP 和 WebSocket 入口需要 `Authorization: Bearer ...`；否则本地开发环境保持开放。

## 官方 WebRTC 流程

1. 调用 `POST /v1/realtime/client_secrets`，可附带可选 session 配置，用于生成临时 `ek_...` token。
2. 在浏览器中创建 WebRTC offer，然后用以下任一方式调用 `POST /v1/realtime/calls`：
   - `Authorization: Bearer ek_...` 和 `Content-Type: application/sdp`，或
   - `Authorization: Bearer $NEXUS_REALTIME_API_KEY` 和 `multipart/form-data`，其中包含 `sdp` 以及可选的 `session`。
   - `Authorization: Bearer $NEXUS_REALTIME_API_KEY` 和 `application/json`，其中包含 `sdp` 以及可选的 `session`。
3. 使用返回的 SDP answer 完成 peer connection。
4. 可选打开旁路 WebSocket：`wss://host/v1/realtime?call_id=rtc_...`，用于从服务端监控或控制同一个 session。

## Asterisk 电话入口

Nexus 不直接实现 SIP 5060 协议栈；Asterisk 仍作为 SIP/RTP gateway。启用后，电话链路是：

```text
电话 / SIP trunk -> Asterisk -> Stasis(nexus) -> Nexus ARI ExternalMedia -> Realtime session
```

最小配置：

```bash
export NEXUS_ASTERISK_INGRESS_ENABLED=true
export NEXUS_ASTERISK_ARI_URL=http://127.0.0.1:8088/ari
export NEXUS_ASTERISK_ARI_USER=voicebot
export NEXUS_ASTERISK_ARI_PASSWORD=12345678
export NEXUS_ASTERISK_STASIS_APP=nexus
export NEXUS_ASTERISK_EXTERNAL_HOST=127.0.0.1
export NEXUS_ASTERISK_RTP_PORT_START=4000
export NEXUS_ASTERISK_RTP_PORT_END=4099
```

如果配置 `NEXUS_REALTIME_WEBHOOK_URL` 和 `NEXUS_REALTIME_WEBHOOK_SECRET`，来电会发送 OpenAI-compatible `realtime.call.incoming` webhook。未配置 webhook 时，Nexus 只记录 pending `sip_...` call_id，便于手动调用 `/v1/realtime/calls/{call_id}/accept` 调试。

Asterisk dialplan 不要提前 `Answer()`：

```ini
[from-fxo]
exten => s,1,NoOp(Incoming call to Nexus Asterisk ingress)
 same => n,Stasis(nexus)
 same => n,Hangup()
```

当前 v1 范围：

- `session.type` 仅支持 `realtime`
- Asterisk `refer` 需要配置本地 redirect endpoint prefix；默认返回未配置错误。
- client secrets 和 call state 仅存储在进程内

## 测试

默认自动化测试套件：

```bash
poetry run pytest -q
```

手动/E2E 脚本位于 `tests/e2e`，默认 pytest 运行会排除它们。

## 本地 HTTPS/WSS

`nexus serve` 已支持直接把证书参数传给 uvicorn。开发环境可以使用
`model-bin` 下的 localhost 自签名证书启动 HTTPS/WSS：

```bash
poetry run nexus model-bin/work.yaml \
  --host 127.0.0.1 \
  --port 8000 \
  --ssl-certfile model-bin/localhost.crt \
  --ssl-keyfile model-bin/localhost.key
```

等价环境变量：

```bash
export NEXUS_SSL_CERTFILE=model-bin/localhost.crt
export NEXUS_SSL_KEYFILE=model-bin/localhost.key
poetry run nexus model-bin/work.yaml
```

客户端侧使用：

```bash
export OPENAI_BASE_URL=https://localhost:8000/v1
```

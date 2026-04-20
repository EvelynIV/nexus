# WebRTC TUI 终端示例

这是一个尽量简单的 Python WebRTC demo，使用 `textual + sounddevice + aiortc` 直接连接 OpenAI 官方 GA Realtime WebRTC unified interface。

它只展示两类内容：

- 对话文本
- 报错和关键状态

把 `REALTIME_BASE_URL` 从 `https://api.openai.com` 换成你的 Nexus 服务地址后，这个示例也应该可以直接连到 Nexus，不需要写任何 Nexus 专属适配。

## 功能范围

- 本地 Python 进程直接创建 `RTCPeerConnection`
- 用 API key 调用 `${REALTIME_BASE_URL}/v1/realtime/calls`
- 通过 `oai-events` data channel 接收会话事件
- 用 `sounddevice` 采集麦克风并播放远端音频
- 显示用户转写、助手文本输出、助手音频转写

不包含：

- 文本输入框
- 原始事件流检查器
- 工具调用 UI
- 本地 token 服务
- `client_secrets` 流程

## 安装

这个示例不修改仓库根 `pyproject.toml`，依赖只通过 `pip` 安装到 Poetry 虚拟环境里。

```bash
cp examples/webrtc-tui/.env.example examples/webrtc-tui/.env
poetry run pip install -r examples/webrtc-tui/requirements.txt
```

Linux 上如果 `sounddevice` 缺少 PortAudio，请先安装系统依赖，例如：

```bash
sudo apt-get install portaudio19-dev
```

## 配置

编辑 `examples/webrtc-tui/.env`：

```env
REALTIME_BASE_URL=https://api.openai.com
REALTIME_API_KEY=sk-...
REALTIME_MODEL=gpt-realtime
REALTIME_VOICE=marin
REALTIME_AUDIO_INPUT_DEVICE=
REALTIME_AUDIO_OUTPUT_DEVICE=
```

说明：

- `REALTIME_BASE_URL` 默认是 OpenAI 官方地址
- 切到 Nexus 时只需要改成你的服务根地址，例如 `http://127.0.0.1:8000`
- `REALTIME_API_KEY` 是标准 Bearer API key
- 设备名留空时使用系统默认输入/输出设备

## 运行

```bash
poetry run python examples/webrtc-tui/main.py
```

快捷键：

- `c`：连接 / 断开
- `m`：麦克风静音 / 取消静音
- `q`：退出

## 连接行为

这个示例按 OpenAI 官方 GA WebRTC unified interface 组织请求：

- 请求 `POST /v1/realtime/calls`
- `Authorization: Bearer REALTIME_API_KEY`
- `multipart/form-data` 携带 `sdp` 和 `session`
- 会话默认开启 `server_vad`
- 默认请求 `gpt-4o-mini-transcribe` 用于显示用户语音转写
- 如果后端拒绝输入转写配置，会自动重试一次无转写配置

## 验证

可以先做语法检查：

```bash
poetry run python -m compileall examples/webrtc-tui
```

手工验收建议：

- 连 OpenAI 官方地址，确认能建立会话
- 把 `REALTIME_BASE_URL` 改到 Nexus，确认无需改代码即可连通
- 说话后能看到用户转写
- 助手回复时能听到音频，并看到文本或音频转写
- 网络、鉴权、设备失败时，错误区会显示原因

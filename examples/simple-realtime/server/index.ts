import http from 'node:http'
import { WebSocket, WebSocketServer } from 'ws'
import { createBannerEvent, type AppClientEvent, type RawRealtimeEvent } from '../src/lib/realtime/app-protocol'
import { loadServerConfig } from './config'
import { SessionViewModel } from './session-view'
import { buildSessionUpdateEvent, buildUpstreamSocketOptions, buildUpstreamUrl, translateClientEvent } from './upstream'

const config = loadServerConfig()

const server = http.createServer((req, res) => {
  if (req.url === '/api/health') {
    res.writeHead(200, { 'content-type': 'application/json' })
    res.end(JSON.stringify({ status: 'ok' }))
    return
  }

  res.writeHead(404, { 'content-type': 'application/json' })
  res.end(JSON.stringify({ error: 'not_found' }))
})

const wss = new WebSocketServer({ noServer: true })

wss.on('connection', (browserSocket) => {
  let upstreamSocket: WebSocket | null = null
  let view = new SessionViewModel('', 'text', config.defaultVoice)
  let closing = false

  const sendBrowser = (event: unknown) => {
    if (browserSocket.readyState === WebSocket.OPEN) {
      browserSocket.send(JSON.stringify(event))
    }
  }

  const resetView = () => {
    for (const event of view.reset()) {
      sendBrowser(event)
    }
  }

  const closeUpstream = () => {
    const socket = upstreamSocket
    upstreamSocket = null
    if (socket && socket.readyState <= WebSocket.OPEN) {
      socket.close(1000, 'browser_disconnect')
    }
  }

  const connectUpstream = (model: string, outputMode: 'text' | 'audio', voice: string) => {
    closing = false
    closeUpstream()
    view = new SessionViewModel(model, outputMode, voice)
    sendBrowser(view.setPhase('connecting'))
    sendBrowser(createBannerEvent('info', null))

    const upstream = new WebSocket(
      buildUpstreamUrl(config, model),
      buildUpstreamSocketOptions(config),
    )

    upstreamSocket = upstream
    let didOpen = false
    const connectTimeout = setTimeout(() => {
      if (didOpen || upstreamSocket !== upstream) {
        return
      }

      upstreamSocket = null
      upstream.terminate()
      resetView()
      sendBrowser(
        createBannerEvent(
          'error',
          `连接上游超时，${config.upstreamConnectTimeoutMs}ms 内未建立 WebSocket。请检查网络、代理或 API Key 配置。`,
        ),
      )
    }, config.upstreamConnectTimeoutMs)

    upstream.on('open', () => {
      didOpen = true
      clearTimeout(connectTimeout)
      const sessionUpdate = buildSessionUpdateEvent(model, outputMode, voice)
      upstream.send(JSON.stringify(sessionUpdate))
      for (const event of view.outbound(sessionUpdate)) {
        sendBrowser(event)
      }
    })

    upstream.on('message', (message) => {
      try {
        const rawEvent = JSON.parse(String(message)) as RawRealtimeEvent
        for (const event of view.inbound(rawEvent)) {
          sendBrowser(event)
        }
      } catch (error) {
        sendBrowser(createBannerEvent('error', `解析上游事件失败：${String(error)}`))
      }
    })

    upstream.on('error', (error) => {
      if (upstreamSocket !== upstream) {
        return
      }
      sendBrowser(
        createBannerEvent('error', `上游 WebSocket 发生错误：${error.message || String(error)}`),
      )
    })

    upstream.on('close', (code, reason) => {
      clearTimeout(connectTimeout)
      upstreamSocket = null
      if (closing) {
        return
      }

      resetView()
      sendBrowser(
        createBannerEvent(
          'error',
          `上游连接已关闭（${code}${reason ? `：${String(reason)}` : ''}）。`,
        ),
      )
    })
  }

  browserSocket.on('message', (message) => {
    try {
      const event = JSON.parse(String(message)) as AppClientEvent

      switch (event.type) {
        case 'app.connect':
          if (!event.model?.trim()) {
            sendBrowser(createBannerEvent('error', '模型不能为空，请先输入模型名。'))
            break
          }
          connectUpstream(event.model.trim(), event.outputMode, event.voice || config.defaultVoice)
          break
        case 'app.disconnect':
          closing = true
          closeUpstream()
          resetView()
          break
        case 'app.session.update': {
          if (!upstreamSocket || upstreamSocket.readyState !== WebSocket.OPEN) {
            sendBrowser(createBannerEvent('error', '请先建立连接，再更新会话。'))
            break
          }

          if (event.model !== undefined && !event.model.trim()) {
            sendBrowser(createBannerEvent('error', '模型不能为空，请输入模型名。'))
            break
          }

          const nextState = view.setDesiredSession(
            event.model?.trim() || view.snapshot().model,
            event.outputMode ?? view.snapshot().outputMode,
            event.voice || view.snapshot().voice,
          )
          sendBrowser(nextState)
          for (const rawEvent of translateClientEvent(
            event,
            view.snapshot().model,
            view.snapshot().outputMode,
            view.snapshot().voice,
          )) {
            upstreamSocket.send(JSON.stringify(rawEvent))
            for (const appEvent of view.outbound(rawEvent)) {
              sendBrowser(appEvent)
            }
          }
          break
        }
        case 'app.audio.append':
        case 'app.audio.commit':
        case 'app.audio.clear': {
          if (!upstreamSocket || upstreamSocket.readyState !== WebSocket.OPEN) {
            sendBrowser(createBannerEvent('error', '请先建立连接，再发送音频。'))
            break
          }

          for (const rawEvent of translateClientEvent(
            event,
            view.snapshot().model,
            view.snapshot().outputMode,
            view.snapshot().voice,
          )) {
            upstreamSocket.send(JSON.stringify(rawEvent))
            for (const appEvent of view.outbound(rawEvent)) {
              sendBrowser(appEvent)
            }
          }
          break
        }
      }
    } catch (error) {
      sendBrowser(createBannerEvent('error', `应用消息无效：${String(error)}`))
    }
  })

  browserSocket.on('close', () => {
    closing = true
    closeUpstream()
  })
})

server.on('upgrade', (request, socket, head) => {
  if (request.url !== '/ws') {
    socket.destroy()
    return
  }

  wss.handleUpgrade(request, socket, head, (wsSocket) => {
    wss.emit('connection', wsSocket, request)
  })
})

server.listen(config.port, () => {
  console.log(`[simple-realtime] backend listening on http://127.0.0.1:${config.port}`)
})

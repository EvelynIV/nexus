import { useEffect, useMemo, useReducer, useRef, useState } from 'react'
import type { Dispatch, SetStateAction } from 'react'
import {
  createBannerEvent,
  formatRecordedAt,
  type AppServerEvent,
  type ConnectionPhase,
  type InspectorEntry,
  type OutputMode,
  type RealtimeSession,
  type TimelineEntry,
} from './lib/realtime/app-protocol'
import { SessionViewModel } from './lib/realtime/session-view'
import { createInitialRealtimeState, realtimeReducer } from './lib/realtime/state'
import { WebRtcRealtimeClient } from './lib/realtime/webrtc-client'

const defaultModel = import.meta.env.VITE_REALTIME_MODEL ?? 'gpt-realtime'
const defaultVoice = import.meta.env.VITE_REALTIME_VOICE ?? 'alloy'

function phaseLabel(phase: ConnectionPhase): string {
  switch (phase) {
    case 'connecting':
      return '连接中'
    case 'connected':
      return '已连接'
    case 'recording':
      return '麦克风开启'
    default:
      return '空闲'
  }
}

function transportLabel(phase: ConnectionPhase): string {
  switch (phase) {
    case 'connecting':
      return 'WebRTC 协商中'
    case 'connected':
    case 'recording':
      return 'WebRTC 已连接'
    default:
      return '未连接'
  }
}

function inspectorDirectionLabel(direction: InspectorEntry['direction']): string {
  switch (direction) {
    case 'inbound':
      return '入站'
    case 'outbound':
      return '出站'
    default:
      return '系统'
  }
}

function describeError(error: unknown): string {
  if (error instanceof Error) {
    return error.message
  }

  return String(error)
}

function sessionHasInputTranscription(session: RealtimeSession | undefined): boolean {
  return Boolean(session?.audio?.input?.transcription)
}

interface InspectorGroup {
  id: string
  direction: InspectorEntry['direction']
  type: string
  payload: unknown
  recordedAt: string
  count: number
}

function groupInspectorEntries(entries: InspectorEntry[]): InspectorGroup[] {
  const groups: InspectorGroup[] = []

  for (const entry of entries) {
    const previous = groups[groups.length - 1]
    if (previous && previous.direction === entry.direction && previous.type === entry.type) {
      previous.count += 1
      continue
    }

    groups.push({
      id: entry.id,
      direction: entry.direction,
      type: entry.type,
      payload: entry.payload,
      recordedAt: entry.recordedAt,
      count: 1,
    })
  }

  return groups
}

function useLevelDecay(isRecording: boolean, setLevel: Dispatch<SetStateAction<number>>): void {
  useEffect(() => {
    let animationFrame = 0

    const tick = () => {
      setLevel((current) => {
        if (!isRecording && current < 0.01) {
          return 0
        }

        const next = current * (isRecording ? 0.92 : 0.7)
        return next < 0.005 ? 0 : next
      })
      animationFrame = requestAnimationFrame(tick)
    }

    animationFrame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(animationFrame)
  }, [isRecording, setLevel])
}

function App() {
  const [model, setModel] = useState(defaultModel)
  const [selectedOutputMode, setSelectedOutputMode] = useState<OutputMode>('text')
  const [voice, setVoice] = useState(defaultVoice)
  const [inputLevel, setInputLevel] = useState(0)
  const [expandedInspectorIds, setExpandedInspectorIds] = useState<Set<string>>(() => new Set())
  const [state, dispatch] = useReducer(
    realtimeReducer,
    undefined,
    () => createInitialRealtimeState('text', defaultVoice),
  )

  const clientRef = useRef<WebRtcRealtimeClient | null>(null)
  const sessionViewRef = useRef<SessionViewModel | null>(null)
  const clientClosingRef = useRef(false)

  const isConnecting = state.connectionPhase === 'connecting'
  const isConnected = state.connectionPhase === 'connected' || state.connectionPhase === 'recording'
  const isRecording = state.connectionPhase === 'recording'
  const hasActiveSession = isConnecting || isConnected
  const canEditConnectionSettings = !hasActiveSession
  const canToggleMicrophone = isConnected

  useLevelDecay(isRecording, setInputLevel)

  const stats = useMemo(() => {
    const userTurns = state.timeline.filter((entry) => entry.role === 'user').length
    const assistantTurns = state.timeline.filter((entry) => entry.role === 'assistant').length
    return {
      userTurns,
      assistantTurns,
      rawEvents: state.inspector.length,
    }
  }, [state.inspector.length, state.timeline])

  const chatEntries = useMemo(
    () => state.timeline.filter((entry) => entry.role === 'user' || entry.role === 'assistant'),
    [state.timeline],
  )
  const inspectorGroups = useMemo(() => groupInspectorEntries(state.inspector), [state.inspector])

  useEffect(() => {
    return () => {
      void closeSession({
        clearState: false,
        banner: null,
      })
    }
  }, [])

  useEffect(() => {
    if (state.model) {
      setModel(state.model)
    }
  }, [state.model])

  useEffect(() => {
    setSelectedOutputMode(state.outputMode)
  }, [state.outputMode])

  useEffect(() => {
    if (state.voice) {
      setVoice(state.voice)
    }
  }, [state.voice])

  const applyAppEvent = (event: AppServerEvent) => {
    switch (event.type) {
      case 'app.state':
        dispatch({
          type: 'app.state',
          state: event,
        })
        break
      case 'app.banner':
        dispatch(event)
        break
      case 'app.timeline.upsert':
        dispatch(event)
        break
      case 'app.trace':
        dispatch(event)
        break
    }
  }

  const applyAppEvents = (events: AppServerEvent[]) => {
    for (const event of events) {
      applyAppEvent(event)
    }
  }

  const dispatchLocalTrace = (
    direction: InspectorEntry['direction'],
    type: string,
    payload: unknown,
  ) => {
    dispatch({
      type: 'app.trace',
      entry: {
        id: `${direction}:${type}:${Date.now()}:${Math.random()}`,
        direction,
        type,
        payload,
        recordedAt: formatRecordedAt(),
      },
    })
  }

  const dispatchLocalTimeline = (
    role: TimelineEntry['role'],
    title: string,
    text: string,
    eventType: string,
  ) => {
    dispatch({
      type: 'app.timeline.upsert',
      entry: {
        id: `${role}:${eventType}:${Date.now()}:${Math.random()}`,
        role,
        mode: 'system',
        title,
        text,
        status: role === 'error' ? 'error' : 'completed',
        eventTypes: [eventType],
        updatedAt: formatRecordedAt(),
      },
    })
  }

  const closeSession = async ({
    clearState,
    banner,
  }: {
    clearState: boolean
    banner: { tone: 'info' | 'error'; message: string } | null
  }) => {
    clientClosingRef.current = true

    const client = clientRef.current
    clientRef.current = null
    sessionViewRef.current = null

    if (client) {
      await client.disconnect()
    }

    setInputLevel(0)

    if (clearState) {
      dispatch({
        type: 'session.reset',
        outputMode: selectedOutputMode,
        voice,
        banner: null,
      })
      dispatch(
        createBannerEvent(
          banner?.tone ?? 'info',
          banner?.message ?? null,
        ),
      )
    }

    clientClosingRef.current = false
  }

  const handleClientError = async (error: unknown) => {
    if (clientClosingRef.current) {
      return
    }

    const message = `WebRTC 会话异常：${describeError(error)}`
    await closeSession({
      clearState: true,
      banner: {
        tone: 'error',
        message,
      },
    })
    dispatchLocalTrace('system', 'browser.error', { message })
    dispatchLocalTimeline('error', '传输错误', message, 'browser.error')
  }

  const connect = async () => {
    await closeSession({
      clearState: true,
      banner: null,
    })

    dispatch({
      type: 'connection.phase',
      phase: 'connecting',
    })
    dispatch(createBannerEvent('info', null))
    dispatchLocalTrace('system', 'browser.connect.request', {
      model,
      outputMode: selectedOutputMode,
      voice,
    })
    dispatchLocalTimeline(
      'system',
      '开始连接',
      '正在请求 ephemeral token，并与目标 Realtime Runtime 建立 WebRTC 会话。',
      'browser.connect.request',
    )

    const sessionView = new SessionViewModel(model, selectedOutputMode, voice)
    sessionViewRef.current = sessionView
    sessionView.setPhase('connecting')

    const client = new WebRtcRealtimeClient({
      onEvent: (event) => {
        const view = sessionViewRef.current
        if (!view) {
          return
        }

        applyAppEvents(view.inbound(event))
      },
      onTrace: (direction, type, payload) => {
        dispatchLocalTrace(direction, type, payload)
      },
      onConnectionStateChange: (connectionState) => {
        dispatchLocalTrace('system', 'webrtc.connection_state', {
          state: connectionState,
        })
      },
      onInputLevel: (level) => {
        setInputLevel((current) => Math.max(level, current))
      },
      onError: (error) => {
        void handleClientError(error)
      },
    })

    clientRef.current = client

    try {
      const token = await client.connect({
        model,
        outputMode: selectedOutputMode,
        voice,
      })

      if (clientRef.current !== client) {
        await client.disconnect()
        return
      }

      dispatchLocalTrace('system', 'browser.webrtc.ready', {
        expiresAt: token.expires_at,
        hasSession: Boolean(token.session),
      })
      dispatchLocalTimeline(
        'system',
        '协商完成',
        '本地 SDP 交换已完成，等待 OpenAI 发送 session.created。',
        'browser.webrtc.ready',
      )

      if (!sessionHasInputTranscription(token.session)) {
        dispatch(
          createBannerEvent(
            'info',
            '当前模型未接受输入转写配置，时间线可能只显示助手输出。',
          ),
        )
        dispatchLocalTimeline(
          'system',
          '输入转写已降级',
          'token 服务已回退到无输入转写配置，主 WebRTC 会话仍然可用。',
          'browser.transcription.fallback',
        )
      }
    } catch (error) {
      const message = `连接失败：${describeError(error)}`
      await closeSession({
        clearState: true,
        banner: {
          tone: 'error',
          message,
        },
      })
      dispatchLocalTrace('system', 'browser.error', { message })
      dispatchLocalTimeline('error', '连接失败', message, 'browser.error')
    }
  }

  const disconnect = async () => {
    await closeSession({
      clearState: true,
      banner: null,
    })
  }

  const enableMicrophone = async () => {
    const client = clientRef.current
    const sessionView = sessionViewRef.current

    if (!client || !sessionView || !canToggleMicrophone || isRecording) {
      return
    }

    try {
      await client.setMicrophoneEnabled(true)
      applyAppEvent(sessionView.setPhase('recording'))
      dispatchLocalTrace('system', 'browser.microphone.enabled', { enabled: true })
      dispatchLocalTimeline(
        'system',
        '麦克风已开启',
        '浏览器现在会把本地音轨直接发送给 OpenAI，由服务端 VAD 自动检测话轮。',
        'browser.microphone.enabled',
      )
    } catch (error) {
      const message = `无法开启麦克风：${describeError(error)}`
      dispatch(createBannerEvent('error', message))
      dispatchLocalTrace('system', 'browser.error', { message })
      dispatchLocalTimeline('error', '麦克风错误', message, 'browser.error')
    }
  }

  const disableMicrophone = async () => {
    const client = clientRef.current
    const sessionView = sessionViewRef.current

    if (!client || !sessionView || !isRecording) {
      return
    }

    try {
      await client.setMicrophoneEnabled(false)
      applyAppEvent(sessionView.setPhase('connected'))
      dispatchLocalTrace('system', 'browser.microphone.enabled', { enabled: false })
      dispatchLocalTimeline(
        'system',
        '麦克风已关闭',
        '浏览器已停止发送本地音轨，当前会话继续保持连接。',
        'browser.microphone.disabled',
      )
    } catch (error) {
      const message = `无法关闭麦克风：${describeError(error)}`
      dispatch(createBannerEvent('error', message))
      dispatchLocalTrace('system', 'browser.error', { message })
      dispatchLocalTimeline('error', '麦克风错误', message, 'browser.error')
    }
  }

  const handleOutputModeChange = (mode: OutputMode) => {
    setSelectedOutputMode(mode)
    dispatch({
      type: 'output.mode',
      outputMode: mode,
    })

    const client = clientRef.current
    if (!client || !isConnected) {
      return
    }

    try {
      client.updateSession({ outputMode: mode })
      dispatchLocalTimeline(
        'system',
        '请求切换输出模式',
        `已向 OpenAI 发送 session.update，目标输出模式为 ${mode}。`,
        'session.update',
      )
    } catch (error) {
      const message = `无法更新会话：${describeError(error)}`
      dispatch(createBannerEvent('error', message))
      dispatchLocalTrace('system', 'browser.error', { message })
      dispatchLocalTimeline('error', '会话更新失败', message, 'browser.error')
    }
  }

  const toggleOutputMode = () => {
    handleOutputModeChange(selectedOutputMode === 'text' ? 'audio' : 'text')
  }

  const latestBanner = state.banner

  const toggleInspectorEntry = (id: string) => {
    setExpandedInspectorIds((current) => {
      const next = new Set(current)
      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }
      return next
    })
  }

  return (
    <div className="shell">
      <div className="shell__glow shell__glow--left" />
      <div className="shell__glow shell__glow--right" />

      <main className="app">
        <header className="topbar card">
          <div className="topbar__intro">
            <p className="eyebrow">OpenAI Realtime</p>
            <h1>WebRTC 控制台</h1>
            <p className="topbar__text">
              浏览器直接连到 OpenAI 官方 Realtime Runtime。这个本地后端只负责签发 ephemeral
              token，聊天时间线和原始事件都在当前页面里观察。
            </p>
          </div>
          <div className="topbar__stats">
            <StatChip label="当前状态" value={phaseLabel(state.connectionPhase)} />
            <StatChip label="传输" value={transportLabel(state.connectionPhase)} />
            <StatChip label="对话轮次" value={`${stats.userTurns} / ${stats.assistantTurns}`} />
            <StatChip label="原始事件" value={String(stats.rawEvents)} />
          </div>
        </header>

        {latestBanner ? (
          <section className={`banner banner--${latestBanner.tone}`}>
            <strong>{latestBanner.tone === 'error' ? '实时错误' : '提示'}</strong>
            <span>{latestBanner.message}</span>
          </section>
        ) : null}

        <section className="workspace">
          <section className="card panel panel--controls">
            <div className="panel__header">
              <div>
                <p className="eyebrow">控制区</p>
                <h2>连接与采集</h2>
              </div>
            </div>

            <div className="control-stack">
              <section className="subpanel">
                <div className="subpanel__header">
                  <strong>会话连接</strong>
                  <span className={`status-pill status-pill--${state.connectionPhase}`}>
                    {transportLabel(state.connectionPhase)}
                  </span>
                </div>

                <label className="field">
                  <span>模型</span>
                  <input
                    value={model}
                    onChange={(event) => setModel(event.target.value)}
                    disabled={canEditConnectionSettings ? false : true}
                    placeholder="gpt-realtime"
                  />
                </label>

                <label className="field">
                  <span>音色</span>
                  <input
                    value={voice}
                    onChange={(event) => setVoice(event.target.value)}
                    disabled={canEditConnectionSettings ? false : true}
                    placeholder="alloy / ash / marin"
                    spellCheck={false}
                  />
                </label>

                <div className="field">
                  <span>输出模式</span>
                  <OutputModeToggle
                    mode={selectedOutputMode}
                    disabled={isConnecting}
                    onClick={toggleOutputMode}
                  />
                </div>

                <div className="actions actions--tight actions--single">
                  <button
                    className={`button ${hasActiveSession ? 'button--secondary' : 'button--primary'}`}
                    onClick={hasActiveSession ? disconnect : connect}
                    disabled={!hasActiveSession && (!model.trim() || !voice.trim())}
                  >
                    {hasActiveSession ? '断开连接' : '连接'}
                  </button>
                </div>

                <dl className="meta-list meta-list--compact">
                  <div>
                    <dt>会话 ID</dt>
                    <dd>{state.sessionId ?? '尚未建立'}</dd>
                  </div>
                </dl>
              </section>

              <section className="subpanel">
                <div className="subpanel__header">
                  <strong>麦克风</strong>
                  <span className="mono">{Math.round(inputLevel * 100)}%</span>
                </div>

                <div className="meter">
                  <div className="meter__track">
                    <div
                      className="meter__fill"
                      style={{ width: `${Math.max(4, Math.min(100, inputLevel * 100))}%` }}
                    />
                  </div>
                  <div className="meter__labels">
                    <span>本地音轨电平</span>
                    <strong>{isRecording ? 'LIVE' : 'MUTED'}</strong>
                  </div>
                </div>

                <div className="actions actions--stacked">
                  <button
                    className="button button--accent"
                    onClick={enableMicrophone}
                    disabled={!canToggleMicrophone || isRecording}
                  >
                    开启麦克风
                  </button>
                  <button
                    className="button button--secondary"
                    onClick={disableMicrophone}
                    disabled={!isRecording}
                  >
                    关闭麦克风
                  </button>
                </div>

                <ul className="notes notes--compact">
                  <li>浏览器直接把本地音轨送到 OpenAI 官方 WebRTC Runtime。</li>
                  <li>服务端 VAD 自动检测开始说话、停止说话和话轮提交。</li>
                  <li>助手音频不再走 PCM chunk 播放，而是由远端媒体流直接播放。</li>
                </ul>
              </section>
            </div>
          </section>

          <section className="card panel panel--timeline">
            <div className="panel__header">
              <div>
                <p className="eyebrow">对话</p>
                <h2>会话过程</h2>
              </div>
            </div>

            <div className="chat-panel">
              {chatEntries.length === 0 ? (
                <EmptyState
                  title="暂无对话内容"
                  detail="连接后开启麦克风说话。若输入转写可用，用户发言会同步显示在这里。"
                />
              ) : (
                chatEntries.map((entry) => <ChatMessage key={entry.id} entry={entry} />)
              )}
            </div>
          </section>

          <section className="card panel panel--inspector">
            <div className="panel__header">
              <div>
                <p className="eyebrow">检查器</p>
                <h2>原始事件流</h2>
              </div>
            </div>

            <div className="inspector">
              {state.inspector.length === 0 ? (
                <EmptyState
                  title="尚未捕获任何数据包"
                  detail="这里会显示 OpenAI Realtime data channel 事件，以及本地的连接和更新诊断。"
                />
              ) : (
                inspectorGroups.map((entry) => (
                  <InspectorCard
                    key={entry.id}
                    entry={entry}
                    expanded={expandedInspectorIds.has(entry.id)}
                    onToggle={() => toggleInspectorEntry(entry.id)}
                  />
                ))
              )}
            </div>
          </section>
        </section>
      </main>
    </div>
  )
}

function StatChip({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat-chip">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function OutputModeToggle({
  mode,
  disabled,
  onClick,
}: {
  mode: OutputMode
  disabled: boolean
  onClick: () => void
}) {
  const isAudio = mode === 'audio'
  return (
    <button
      type="button"
      className={`mode-toggle ${isAudio ? 'mode-toggle--audio' : 'mode-toggle--text'}`}
      disabled={disabled}
      onClick={onClick}
    >
      <strong>{isAudio ? '音频' : '文本'}</strong>
    </button>
  )
}

function ChatMessage({ entry }: { entry: TimelineEntry }) {
  return (
    <article className={`chat-message chat-message--${entry.role}`}>
      <div className="chat-message__meta">
        <span className="chat-message__role">{entry.role === 'user' ? '你' : '助手'}</span>
        <span className="mono">{entry.updatedAt}</span>
      </div>
      <div className="chat-message__bubble">
        <div>
          <p className="chat-message__title">{entry.title}</p>
          <p className="chat-message__body">{entry.text || '等待流式内容...'}</p>
        </div>
      </div>
    </article>
  )
}

function InspectorCard({
  entry,
  expanded,
  onToggle,
}: {
  entry: InspectorGroup
  expanded: boolean
  onToggle: () => void
}) {
  return (
    <article className={`inspector-card inspector-card--${entry.direction}`}>
      <div className="inspector-card__header">
        <div className="inspector-card__summary">
          <span className="mono">{inspectorDirectionLabel(entry.direction)}</span>
          <strong>
            {entry.type}
            {entry.count > 1 ? ` * ${entry.count}` : ''}
          </strong>
          <span className="mono">{entry.recordedAt}</span>
        </div>
        <button type="button" className="collapse-button" onClick={onToggle}>
          {expanded ? '收起' : '展开'}
        </button>
      </div>
      {expanded ? <pre>{JSON.stringify(entry.payload, null, 2)}</pre> : null}
    </article>
  )
}

function EmptyState({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="empty-state">
      <strong>{title}</strong>
      <p>{detail}</p>
    </div>
  )
}

export default App

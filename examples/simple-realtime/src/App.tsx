import { useEffect, useMemo, useReducer, useRef, useState } from 'react'
import type { Dispatch, SetStateAction } from 'react'
import { MicrophoneRecorder } from './lib/audio/recorder'
import { PcmAudioPlayer } from './lib/audio/player'
import {
  buildAppWebSocketUrl,
  createBannerEvent,
  createConnectEvent,
  createSessionUpdateCommand,
  formatRecordedAt,
  type AppClientEvent,
  type AppServerEvent,
  type ConnectionPhase,
  type InspectorEntry,
  type OutputMode,
  type TimelineEntry,
} from './lib/realtime/app-protocol'
import {
  createInitialRealtimeState,
  realtimeReducer,
  type BannerState,
} from './lib/realtime/state'

const defaultWsUrl = import.meta.env.VITE_APP_WS_URL ?? '/ws'
const defaultModel = import.meta.env.VITE_REALTIME_MODEL ?? ''
const defaultVoice = import.meta.env.VITE_REALTIME_VOICE ?? 'alloy'

function phaseLabel(phase: ConnectionPhase): string {
  switch (phase) {
    case 'connecting':
      return '连接中'
    case 'connected':
      return '已就绪'
    case 'recording':
      return '录音中'
    default:
      return '空闲'
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
  const [appWsUrl, setAppWsUrl] = useState(defaultWsUrl)
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

  const socketRef = useRef<WebSocket | null>(null)
  const recorderRef = useRef<MicrophoneRecorder | null>(null)
  const playerRef = useRef<PcmAudioPlayer | null>(null)
  const clientClosingRef = useRef(false)

  const isConnected = state.connectionPhase === 'connected' || state.connectionPhase === 'recording'
  const isRecording = state.connectionPhase === 'recording'
  const canReconnect = state.connectionPhase !== 'recording'
  const canClearBuffer = isConnected && !isRecording
  const canToggleMode = isConnected && !isRecording

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

  const getPlayer = () => {
    if (!playerRef.current) {
      playerRef.current = new PcmAudioPlayer()
    }
    return playerRef.current
  }

  const stopRecorder = async (flush = true) => {
    const recorder = recorderRef.current
    recorderRef.current = null

    if (recorder) {
      await recorder.stop({ flush })
    }
    setInputLevel(0)
  }

  const resetPlayer = async () => {
    const player = playerRef.current
    playerRef.current = null
    if (player) {
      await player.reset()
    }
  }

  const closeSession = async ({
    clearState,
    banner,
  }: {
    clearState: boolean
    banner: BannerState | null
  }) => {
    clientClosingRef.current = true

    await stopRecorder(false)
    await resetPlayer()

    const socket = socketRef.current
    socketRef.current = null

    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: 'app.disconnect' } satisfies AppClientEvent))
    }

    if (socket && socket.readyState <= WebSocket.OPEN) {
      socket.close(1000, 'client_disconnect')
    }

    if (clearState) {
      dispatch({
        type: 'session.reset',
        outputMode: selectedOutputMode,
        voice,
        banner,
      })
      dispatch(createBannerEvent(banner?.tone ?? 'info', banner?.message ?? null))
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
        id: `${direction}:${type}:${Date.now()}`,
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
        id: `${role}:${eventType}:${Date.now()}`,
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

  const sendCommand = (payload: AppClientEvent): boolean => {
    const socket = socketRef.current
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      dispatch(createBannerEvent('error', 'WebSocket 连接尚未打开。'))
      dispatchLocalTrace('system', 'browser.error', { message: 'WebSocket 连接尚未打开。' })
      dispatchLocalTimeline('error', '传输错误', 'WebSocket 连接尚未打开。', 'browser.error')
      return false
    }

    socket.send(JSON.stringify(payload))
    return true
  }

  const connect = async () => {
    await closeSession({
      clearState: true,
      banner: null,
    })
    clientClosingRef.current = false

    dispatch({
      type: 'connection.phase',
      phase: 'connecting',
    })

    try {
      await getPlayer().prime()
      const url = buildAppWebSocketUrl(appWsUrl)
      const socket = new WebSocket(url)
      socketRef.current = socket

      socket.onopen = () => {
        sendCommand(createConnectEvent(model, selectedOutputMode, voice))
      }

      socket.onmessage = (message) => {
        try {
          const event = JSON.parse(String(message.data)) as AppServerEvent

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
            case 'app.audio.delta':
              void getPlayer().enqueueBase64(event.chunkBase64)
              break
            case 'app.audio.reset':
              void (async () => {
                await resetPlayer()
                await getPlayer().prime()
              })()
              break
            default:
              dispatch(createBannerEvent('error', `未知的服务端消息：${JSON.stringify(event)}`))
          }
        } catch (error) {
          dispatch(createBannerEvent('error', `解析收到的事件失败：${String(error)}`))
          dispatchLocalTrace('system', 'browser.error', { message: String(error) })
          dispatchLocalTimeline(
            'error',
            '解析错误',
            `解析收到的事件失败：${String(error)}`,
            'browser.error',
          )
        }
      }

      socket.onerror = () => {
        dispatch(createBannerEvent('error', 'WebSocket 传输层报告了错误。'))
        dispatchLocalTrace('system', 'browser.error', { message: 'WebSocket 传输层报告了错误。' })
      }

      socket.onclose = async (event) => {
        socketRef.current = null
        await stopRecorder(false)
        await resetPlayer()

        if (clientClosingRef.current) {
          clientClosingRef.current = false
          return
        }

        dispatch({
          type: 'session.reset',
          outputMode: selectedOutputMode,
          voice,
          banner: null,
        })
        const messageText = `连接已关闭（${event.code}${event.reason ? `：${event.reason}` : ''}）。`
        dispatch(createBannerEvent('error', messageText))
        dispatchLocalTrace('system', 'browser.error', { message: messageText })
        dispatchLocalTimeline('error', '连接已关闭', messageText, 'browser.error')
      }
    } catch (error) {
      await closeSession({
        clearState: true,
        banner: {
          tone: 'error',
          message: `连接失败：${String(error)}`,
        },
      })
    }
  }

  const disconnect = async () => {
    await closeSession({
      clearState: true,
      banner: null,
    })
  }

  const startRecording = async () => {
    if (!isConnected || isRecording) {
      return
    }

    try {
      await getPlayer().prime()
      const recorder = new MicrophoneRecorder({
        onFrame: ({ base64 }) => {
          sendCommand({
            type: 'app.audio.append',
            audioBase64: base64,
          })
        },
        onLevel: (level) => {
          setInputLevel((current) => Math.max(level, current))
        },
      })

      await recorder.start()
      recorderRef.current = recorder

      dispatch({
        type: 'connection.phase',
        phase: 'recording',
      })
      dispatchLocalTimeline('system', '开始录音', '麦克风音频流已启动。', 'client.recording.started')
    } catch (error) {
      await stopRecorder(false)
      const messageText = `无法启动麦克风：${String(error)}`
      dispatch(createBannerEvent('error', messageText))
      dispatchLocalTrace('system', 'browser.error', { message: String(error) })
      dispatchLocalTimeline('error', '麦克风错误', messageText, 'browser.error')
    }
  }

  const stopRecording = async () => {
    if (!isRecording) {
      return
    }

    await stopRecorder(true)
    sendCommand({ type: 'app.audio.commit' })
    dispatch({
      type: 'connection.phase',
      phase: 'connected',
    })
    dispatchLocalTimeline(
      'system',
      '停止录音',
      '麦克风音频流已停止，缓冲区已提交，等待服务端 VAD 完成这轮输入。',
      'client.recording.stopped',
    )
  }

  const clearBuffer = () => {
    if (!canClearBuffer) {
      return
    }

    sendCommand({ type: 'app.audio.clear' })
    dispatchLocalTimeline('system', '已清空缓冲区', '已向服务端发送 app.audio.clear。', 'client.buffer.cleared')
  }

  const handleOutputModeChange = (mode: OutputMode) => {
    setSelectedOutputMode(mode)
    dispatch({
      type: 'output.mode',
      outputMode: mode,
    })

    if (canToggleMode) {
      if (mode === 'audio') {
        void getPlayer().prime()
      }
      sendCommand(createSessionUpdateCommand(model, mode, voice))
    }
  }

  const toggleOutputMode = () => {
    handleOutputModeChange(selectedOutputMode === 'text' ? 'audio' : 'text')
  }

  const handleVoiceChange = (nextVoice: string) => {
    setVoice(nextVoice)
    if (!canToggleMode) {
      return
    }
    sendCommand(createSessionUpdateCommand(model, selectedOutputMode, nextVoice))
  }

  const latestBanner = state.banner
  const transportLabel = isConnected ? 'WebSocket 已连接' : '未连接'
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
            <p className="eyebrow">实时测试台</p>
            <h1>实时控制台</h1>
            <p className="topbar__text">
              连接应用后端、推送麦克风音频，并在同一屏内观察会话时间线与上游原始事件流。
            </p>
          </div>
          <div className="topbar__stats">
            <StatChip label="当前状态" value={phaseLabel(state.connectionPhase)} />
            <StatChip label="连接" value={transportLabel} />
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
                </div>

                <label className="field">
                  <span>模型</span>
                  <input
                    value={model}
                    onChange={(event) => setModel(event.target.value)}
                    disabled={!canReconnect}
                    placeholder="gpt-realtime"
                  />
                </label>

                <label className="field">
                  <span>音色</span>
                  <input
                    value={voice}
                    onChange={(event) => handleVoiceChange(event.target.value)}
                    placeholder="alloy / paimon / rita"
                    spellCheck={false}
                  />
                </label>

                <div className="field">
                  <span>输出模式</span>
                  <OutputModeToggle
                    mode={selectedOutputMode}
                    disabled={isRecording}
                    onClick={toggleOutputMode}
                  />
                </div>

                <div className="actions actions--tight actions--single">
                  <button
                    className={`button ${isConnected ? 'button--secondary' : 'button--primary'}`}
                    onClick={isConnected ? disconnect : connect}
                    disabled={isConnected ? !isConnected : !canReconnect}
                  >
                    {isConnected ? '断开连接' : '连接'}
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
                  <strong>麦克风采集</strong>
                  <span className="mono">{Math.round(inputLevel * 100)}%</span>
                </div>

                <div className="meter">
                  <div className="meter__track">
                    <div
                      className="meter__fill"
                      style={{ width: `${Math.max(4, Math.min(100, inputLevel * 100))}%` }}
                    />
                  </div>
                </div>

                <div className="actions actions--stacked">
                  <button
                    className="button button--accent"
                    onClick={startRecording}
                    disabled={!isConnected || isRecording}
                  >
                    开始录音
                  </button>
                  <button className="button button--secondary" onClick={stopRecording} disabled={!isRecording}>
                    停止并提交
                  </button>
                  <button className="button button--secondary" onClick={clearBuffer} disabled={!canClearBuffer}>
                    清空缓冲区
                  </button>
                </div>

                <ul className="notes notes--compact">
                  <li>AudioWorklet 采集浏览器麦克风音频。</li>
                  <li>发送前重采样到 24kHz 单声道 PCM16。</li>
                  <li>服务端 VAD 自动分段并触发响应。</li>
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
                  detail="连接后开始说话，用户和助手的文本内容会在这里以聊天面板形式显示。"
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
                  detail="这里会按时间倒序显示出站控制消息和入站服务端事件。"
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

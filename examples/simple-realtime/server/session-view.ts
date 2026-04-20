import {
  createBannerEvent,
  formatRecordedAt,
  summarizeInspectorPayload,
  type AppServerEvent,
  type AppStatePayload,
  type ConnectionPhase,
  type InspectorEntry,
  type OutputMode,
  type RawRealtimeEvent,
  type TimelineEntry,
} from '../src/lib/realtime/app-protocol'

function extractResponseId(event: RawRealtimeEvent): string | undefined {
  if (typeof event.response_id === 'string') {
    return event.response_id
  }

  const response = event.response
  if (response && typeof response === 'object' && typeof (response as { id?: unknown }).id === 'string') {
    return (response as { id: string }).id
  }

  return undefined
}

export class SessionViewModel {
  private sequence = 0
  private state: AppStatePayload
  private readonly userEntryIds = new Map<string, string>()
  private readonly assistantEntryIds = new Map<string, string>()
  private readonly timeline = new Map<string, TimelineEntry>()

  constructor(model: string, outputMode: OutputMode, voice = 'alloy') {
    this.state = {
      phase: 'idle',
      sessionId: null,
      model,
      outputMode,
      voice,
    }
  }

  snapshot(): AppStatePayload {
    return { ...this.state }
  }

  setDesiredSession(model: string, outputMode: OutputMode, voice: string): AppServerEvent {
    this.state.model = model
    this.state.outputMode = outputMode
    this.state.voice = voice
    return this.stateEvent()
  }

  setPhase(phase: ConnectionPhase): AppServerEvent {
    this.state.phase = phase
    return this.stateEvent()
  }

  reset(): AppServerEvent[] {
    this.userEntryIds.clear()
    this.assistantEntryIds.clear()
    this.timeline.clear()
    this.state.phase = 'idle'
    this.state.sessionId = null
    return [this.stateEvent(), createBannerEvent('info', null), { type: 'app.audio.reset' }]
  }

  outbound(rawEvent: RawRealtimeEvent): AppServerEvent[] {
    const events: AppServerEvent[] = [this.trace('outbound', rawEvent.type, summarizeInspectorPayload(rawEvent))]

    if (rawEvent.type === 'input_audio_buffer.append' && this.state.phase !== 'recording') {
      this.state.phase = 'recording'
      events.push(this.stateEvent())
    }

    if (rawEvent.type === 'input_audio_buffer.commit' && this.state.phase !== 'connected') {
      this.state.phase = 'connected'
      events.push(this.stateEvent())
      events.push(
        this.systemTimeline(
          '输入已提交',
          '麦克风缓冲区已提交到上游会话。',
          rawEvent.type,
        ),
      )
    }

    if (rawEvent.type === 'input_audio_buffer.clear') {
      events.push(
        this.systemTimeline(
          '缓冲区已清空',
          '后端已清空缓存的麦克风输入。',
          rawEvent.type,
        ),
      )
    }

    return events
  }

  inbound(rawEvent: RawRealtimeEvent): AppServerEvent[] {
    const events: AppServerEvent[] = [this.trace('inbound', rawEvent.type, summarizeInspectorPayload(rawEvent))]

    switch (rawEvent.type) {
      case 'session.created': {
        const session = rawEvent.session as
          | { id?: string; model?: string; output_modalities?: string[]; audio?: { output?: { voice?: string } } }
          | undefined
        this.state.sessionId = session?.id ?? null
        this.state.model = session?.model ?? this.state.model
        this.state.phase = 'connected'
        if (session?.output_modalities?.[0] === 'audio') {
          this.state.outputMode = 'audio'
        }
        if (session?.audio?.output?.voice) {
          this.state.voice = session.audio.output.voice
        }
        events.push(createBannerEvent('info', null))
        events.push(this.stateEvent())
        events.push(
          this.systemTimeline(
            '会话已就绪',
            `已连接到 ${session?.id ?? '未知会话'}，当前模型 ${session?.model ?? this.state.model}。`,
            rawEvent.type,
          ),
        )
        return events
      }
      case 'session.updated': {
        const session = rawEvent.session as
          | { model?: string; output_modalities?: string[]; audio?: { output?: { voice?: string } } }
          | undefined
        if (session?.model) {
          this.state.model = session.model
        }
        if (session?.output_modalities?.[0] === 'audio') {
          this.state.outputMode = 'audio'
        } else if (session?.output_modalities?.[0] === 'text') {
          this.state.outputMode = 'text'
        }
        if (session?.audio?.output?.voice) {
          this.state.voice = session.audio.output.voice
        }
        events.push(createBannerEvent('info', null))
        events.push(this.stateEvent())
        events.push(
          this.systemTimeline(
            '会话已更新',
            `已切换到${this.state.outputMode === 'audio' ? '音频' : '文本'}模式，模型 ${this.state.model}，音色 ${this.state.voice}。`,
            rawEvent.type,
          ),
        )
        return events
      }
      case 'input_audio_buffer.speech_started':
        this.state.phase = 'recording'
        events.push(this.stateEvent())
        events.push(
          this.systemTimeline(
            '开始说话',
            '上游语音检测已开启新的输入片段。',
            rawEvent.type,
          ),
        )
        return events
      case 'input_audio_buffer.speech_stopped':
        events.push(
          this.systemTimeline(
            '停止说话',
            '语音检测已标记当前话语结束。',
            rawEvent.type,
          ),
        )
        return events
      case 'input_audio_buffer.committed':
        this.state.phase = 'connected'
        events.push(this.stateEvent())
        events.push(this.upsertCommittedUserTurn(rawEvent))
        events.push(
          this.systemTimeline(
            '输入已提交',
            '上游已提交当前麦克风缓冲区。',
            rawEvent.type,
          ),
        )
        return events
      case 'conversation.item.input_audio_transcription.delta': {
        const itemId = typeof rawEvent.item_id === 'string' ? rawEvent.item_id : 'unknown-input'
        const entryId = this.resolveUserEntryId(itemId)
        events.push(
          this.upsertTimeline({
            id: entryId,
            role: 'user',
            mode: 'input',
            title: '实时转写',
            text: typeof rawEvent.delta === 'string' ? rawEvent.delta : '',
            status: 'streaming',
            eventType: rawEvent.type,
          }, 'append'),
        )
        return events
      }
      case 'conversation.item.input_audio_transcription.completed': {
        const itemId = typeof rawEvent.item_id === 'string' ? rawEvent.item_id : 'unknown-input'
        const entryId = this.resolveUserEntryId(itemId)
        events.push(
          this.upsertTimeline({
            id: entryId,
            role: 'user',
            mode: 'input',
            title: '最终转写',
            text: typeof rawEvent.transcript === 'string' ? rawEvent.transcript : '',
            status: 'completed',
            eventType: rawEvent.type,
          }),
        )
        return events
      }
      case 'response.created': {
        const responseId = extractResponseId(rawEvent)
        if (!responseId) {
          return events
        }
        const entryId = this.assistantEntryIds.get(responseId) ?? `assistant:${responseId}`
        this.assistantEntryIds.set(responseId, entryId)
        events.push({ type: 'app.audio.reset' })
        events.push(
          this.upsertTimeline({
            id: entryId,
            role: 'assistant',
            mode: this.state.outputMode,
            title: '助手响应',
            text: '',
            status: 'streaming',
            eventType: rawEvent.type,
          }),
        )
        return events
      }
      case 'response.output_text.delta': {
        const responseId = extractResponseId(rawEvent) ?? 'text-response'
        const entryId = this.assistantEntryIds.get(responseId) ?? `assistant:${responseId}`
        this.assistantEntryIds.set(responseId, entryId)
        events.push(
          this.upsertTimeline({
            id: entryId,
            role: 'assistant',
            mode: 'text',
            title: '助手文本',
            text: typeof rawEvent.delta === 'string' ? rawEvent.delta : '',
            status: 'streaming',
            eventType: rawEvent.type,
          }, 'append'),
        )
        return events
      }
      case 'response.output_text.done': {
        const responseId = extractResponseId(rawEvent) ?? 'text-response'
        const entryId = this.assistantEntryIds.get(responseId) ?? `assistant:${responseId}`
        this.assistantEntryIds.set(responseId, entryId)
        events.push(
          this.upsertTimeline({
            id: entryId,
            role: 'assistant',
            mode: 'text',
            title: '助手文本',
            text: typeof rawEvent.text === 'string' ? rawEvent.text : '',
            status: 'streaming',
            eventType: rawEvent.type,
          }),
        )
        return events
      }
      case 'response.output_audio.delta': {
        const responseId = extractResponseId(rawEvent) ?? 'audio-response'
        const entryId = this.assistantEntryIds.get(responseId) ?? `assistant:${responseId}`
        this.assistantEntryIds.set(responseId, entryId)
        events.push({
          type: 'app.audio.delta',
          responseId,
          chunkBase64: typeof rawEvent.delta === 'string' ? rawEvent.delta : '',
        })
        events.push(
          this.upsertTimeline({
            id: entryId,
            role: 'assistant',
            mode: 'audio',
            title: '助手音频',
            text: '',
            status: 'streaming',
            eventType: rawEvent.type,
          }),
        )
        return events
      }
      case 'response.output_audio_transcript.delta': {
        const responseId = extractResponseId(rawEvent) ?? 'audio-response'
        const entryId = this.assistantEntryIds.get(responseId) ?? `assistant:${responseId}`
        this.assistantEntryIds.set(responseId, entryId)
        events.push(
          this.upsertTimeline({
            id: entryId,
            role: 'assistant',
            mode: 'audio',
            title: '助手音频转写',
            text: typeof rawEvent.delta === 'string' ? rawEvent.delta : '',
            status: 'streaming',
            eventType: rawEvent.type,
          }, 'append'),
        )
        return events
      }
      case 'response.output_audio_transcript.done': {
        const responseId = extractResponseId(rawEvent) ?? 'audio-response'
        const entryId = this.assistantEntryIds.get(responseId) ?? `assistant:${responseId}`
        this.assistantEntryIds.set(responseId, entryId)
        events.push(
          this.upsertTimeline({
            id: entryId,
            role: 'assistant',
            mode: 'audio',
            title: '助手音频转写',
            text: typeof rawEvent.transcript === 'string' ? rawEvent.transcript : '',
            status: 'streaming',
            eventType: rawEvent.type,
          }),
        )
        return events
      }
      case 'response.done': {
        const responseId = extractResponseId(rawEvent)
        if (!responseId) {
          return events
        }
        const entryId = this.assistantEntryIds.get(responseId) ?? `assistant:${responseId}`
        this.assistantEntryIds.set(responseId, entryId)
        const existing = this.timeline.get(entryId)
        events.push(
          this.upsertTimeline({
            id: entryId,
            role: 'assistant',
            mode: existing?.mode ?? this.state.outputMode,
            title: '助手响应完成',
            text: existing?.text ?? '',
            status: 'completed',
            eventType: rawEvent.type,
          }),
        )
        return events
      }
      case 'error': {
        const error = rawEvent.error as { message?: string; code?: string } | undefined
        const message = error?.message ?? 'Unknown realtime error.'
        events.push(createBannerEvent('error', message))
        events.push(
          this.errorTimeline(
            '服务端错误',
            error?.code ? `${message} (${error.code})` : message,
            rawEvent.type,
          ),
        )
        return events
      }
      default:
        return events
    }
  }

  private nextId(prefix: string): string {
    this.sequence += 1
    return `${prefix}:${this.sequence}`
  }

  private stateEvent(): AppServerEvent {
    return {
      type: 'app.state',
      ...this.state,
    }
  }

  private trace(
    direction: InspectorEntry['direction'],
    type: string,
    payload: unknown,
  ): Extract<AppServerEvent, { type: 'app.trace' }> {
    return {
      type: 'app.trace',
      entry: {
        id: this.nextId(`trace:${direction}`),
        direction,
        type,
        payload,
        recordedAt: formatRecordedAt(),
      },
    }
  }

  private systemTimeline(
    title: string,
    text: string,
    eventType: string,
  ): Extract<AppServerEvent, { type: 'app.timeline.upsert' }> {
    return {
      type: 'app.timeline.upsert',
      entry: {
        id: this.nextId('system'),
        role: 'system',
        mode: 'system',
        title,
        text,
        status: 'completed',
        eventTypes: [eventType],
        updatedAt: formatRecordedAt(),
      },
    }
  }

  private errorTimeline(
    title: string,
    text: string,
    eventType: string,
  ): Extract<AppServerEvent, { type: 'app.timeline.upsert' }> {
    return {
      type: 'app.timeline.upsert',
      entry: {
        id: this.nextId('error'),
        role: 'error',
        mode: 'system',
        title,
        text,
        status: 'error',
        eventTypes: [eventType],
        updatedAt: formatRecordedAt(),
      },
    }
  }

  private upsertCommittedUserTurn(
    rawEvent: RawRealtimeEvent,
  ): Extract<AppServerEvent, { type: 'app.timeline.upsert' }> {
    const itemId = typeof rawEvent.item_id === 'string' ? rawEvent.item_id : null
    const entryId = itemId ? this.resolveUserEntryId(itemId) : this.nextId('user:pending')
    return this.upsertTimeline({
      id: entryId,
      role: 'user',
      mode: 'input',
      title: '实时转写',
      text: '',
      status: 'streaming',
      eventType: rawEvent.type,
    })
  }

  private resolveUserEntryId(itemId: string): string {
    const existingEntryId = this.userEntryIds.get(itemId)
    if (existingEntryId) {
      return existingEntryId
    }

    const nextEntryId = `user:${itemId}`
    this.userEntryIds.set(itemId, nextEntryId)
    return nextEntryId
  }

  private upsertTimeline(
    entry: Omit<TimelineEntry, 'updatedAt' | 'eventTypes'> & { eventType: string },
    mode: 'append' | 'replace' = 'replace',
  ): Extract<AppServerEvent, { type: 'app.timeline.upsert' }> {
    const existing = this.timeline.get(entry.id)
    const nextEntry: TimelineEntry = existing
      ? {
          ...existing,
          ...entry,
          text: mode === 'append' ? `${existing.text}${entry.text}` : entry.text,
          eventTypes: existing.eventTypes.includes(entry.eventType)
            ? existing.eventTypes
            : [...existing.eventTypes, entry.eventType],
          updatedAt: formatRecordedAt(),
        }
      : {
          ...entry,
          eventTypes: [entry.eventType],
          updatedAt: formatRecordedAt(),
        }

    this.timeline.set(entry.id, nextEntry)
    return {
      type: 'app.timeline.upsert',
      entry: nextEntry,
    }
  }
}

import { describe, expect, it } from 'vitest'
import type { AppServerEvent } from '../src/lib/realtime/app-protocol'
import { SessionViewModel } from './session-view'

describe('SessionViewModel', () => {
  it('turns transcription events into one user timeline entry', () => {
    const view = new SessionViewModel('gpt-realtime', 'text', 'alloy')

    view.inbound({
      type: 'conversation.item.input_audio_transcription.delta',
      item_id: 'item-1',
      delta: 'hello ',
    })
    const events = view.inbound({
      type: 'conversation.item.input_audio_transcription.completed',
      item_id: 'item-1',
      transcript: 'hello world',
    })

    const timelineEvent = events.find((event: AppServerEvent) => event.type === 'app.timeline.upsert')
    expect(timelineEvent).toMatchObject({
      entry: {
        id: 'user:item-1',
        text: 'hello world',
        status: 'completed',
      },
    })
  })

  it('keeps assistant text aggregated across deltas and done', () => {
    const view = new SessionViewModel('gpt-realtime', 'text', 'alloy')

    view.inbound({ type: 'response.created', response: { id: 'resp-1' } })
    view.inbound({ type: 'response.output_text.delta', response_id: 'resp-1', delta: 'hello ' })
    const events = view.inbound({ type: 'response.done', response: { id: 'resp-1' } })

    const timelineEvent = events.find((event: AppServerEvent) => event.type === 'app.timeline.upsert')
    expect(timelineEvent).toMatchObject({
      entry: {
        id: 'assistant:resp-1',
        status: 'completed',
        text: 'hello ',
      },
    })
  })

  it('emits audio delta messages separately from trace events', () => {
    const view = new SessionViewModel('gpt-realtime', 'audio', 'alloy')

    const events = view.inbound({
      type: 'response.output_audio.delta',
      response_id: 'resp-audio',
      delta: 'ZmFrZQ==',
    })

    expect(events.some((event: AppServerEvent) => event.type === 'app.audio.delta')).toBe(true)
    expect(events.some((event: AppServerEvent) => event.type === 'app.trace')).toBe(true)
  })

  it('keeps the user transcription ahead of the assistant response for the same turn', () => {
    const view = new SessionViewModel('gpt-realtime', 'text', 'alloy')

    const committedEvents = view.inbound({ type: 'input_audio_buffer.committed', item_id: 'item-1' })
    const placeholderEvent = committedEvents.find(
      (event: AppServerEvent) => event.type === 'app.timeline.upsert' && event.entry.role === 'user',
    )

    const assistantEvents = view.inbound({ type: 'response.created', response: { id: 'resp-1' } })
    const assistantEvent = assistantEvents.find(
      (event: AppServerEvent) => event.type === 'app.timeline.upsert' && event.entry.role === 'assistant',
    )

    const transcriptionEvents = view.inbound({
      type: 'conversation.item.input_audio_transcription.completed',
      item_id: 'item-1',
      transcript: 'hello world',
    })
    const transcriptionEvent = transcriptionEvents.find(
      (event: AppServerEvent) => event.type === 'app.timeline.upsert' && event.entry.role === 'user',
    )

    expect(placeholderEvent).toMatchObject({
      entry: {
        role: 'user',
        status: 'streaming',
      },
    })
    expect(assistantEvent).toMatchObject({
      entry: {
        role: 'assistant',
      },
    })
    expect(transcriptionEvent).toMatchObject({
      entry: {
        id: 'user:item-1',
        role: 'user',
        text: 'hello world',
        status: 'completed',
      },
    })
    expect((transcriptionEvent as Extract<AppServerEvent, { type: 'app.timeline.upsert' }>).entry.id).toBe(
      (placeholderEvent as Extract<AppServerEvent, { type: 'app.timeline.upsert' }>).entry.id,
    )
    expect((assistantEvent as Extract<AppServerEvent, { type: 'app.timeline.upsert' }>).entry.id).not.toBe(
      (placeholderEvent as Extract<AppServerEvent, { type: 'app.timeline.upsert' }>).entry.id,
    )
  })

  it('uses committed item_id to bind the user turn before transcription arrives', () => {
    const view = new SessionViewModel('gpt-realtime', 'text', 'alloy')

    const committedEvents = view.inbound({ type: 'input_audio_buffer.committed', item_id: 'item-42' })
    const committedEvent = committedEvents.find(
      (event: AppServerEvent) => event.type === 'app.timeline.upsert' && event.entry.role === 'user',
    )

    expect(committedEvent).toMatchObject({
      entry: {
        id: 'user:item-42',
        role: 'user',
        status: 'streaming',
      },
    })
  })

  it('falls back to a pending-style user entry when committed has no item_id', () => {
    const view = new SessionViewModel('gpt-realtime', 'text', 'alloy')

    const committedEvents = view.inbound({ type: 'input_audio_buffer.committed' })
    const committedEvent = committedEvents.find(
      (event: AppServerEvent) => event.type === 'app.timeline.upsert' && event.entry.role === 'user',
    )

    expect(committedEvent).toMatchObject({
      entry: {
        id: expect.stringMatching(/^user:pending:/),
        role: 'user',
        status: 'streaming',
      },
    })
  })

  it('tracks audio voice from session updates', () => {
    const view = new SessionViewModel('gpt-realtime', 'audio', 'alloy')

    const events = view.inbound({
      type: 'session.updated',
      session: {
        model: 'gpt-realtime',
        output_modalities: ['audio'],
        audio: {
          output: {
            voice: 'paimon',
          },
        },
      },
    })

    const stateEvent = events.find((event: AppServerEvent) => event.type === 'app.state')
    expect(stateEvent).toMatchObject({
      voice: 'paimon',
    })
  })
})

import { describe, expect, it } from 'vitest'
import { buildSessionUpdateEvent, buildUpstreamHeaders, buildUpstreamSocketOptions, buildUpstreamUrl, translateClientEvent } from './upstream'
import type { ServerConfig } from './config'

const nexusConfig: ServerConfig = {
  port: 8787,
  defaultVoice: 'alloy',
  upstreamBaseUrl: 'http://127.0.0.1:8000/v1',
  openAiApiKey: null,
  upstreamProxyUrl: null,
  upstreamConnectTimeoutMs: 15000,
}

const openAiConfig: ServerConfig = {
  ...nexusConfig,
  upstreamBaseUrl: 'https://api.openai.com/v1',
  openAiApiKey: 'sk-test',
}

describe('upstream helpers', () => {
  it('builds nexus upstream urls', () => {
    expect(buildUpstreamUrl(nexusConfig, 'test-model')).toBe('ws://127.0.0.1:8000/v1/realtime?model=test-model')
  })

  it('builds openai upstream urls and auth headers', () => {
    expect(buildUpstreamUrl(openAiConfig, 'gpt-realtime')).toBe('wss://api.openai.com/v1/realtime?model=gpt-realtime')
    expect(buildUpstreamHeaders(openAiConfig)).toEqual({
      Authorization: 'Bearer sk-test',
    })
  })

  it('uses a proxy agent when proxy env is configured', () => {
    const options = buildUpstreamSocketOptions({
      ...openAiConfig,
      upstreamProxyUrl: 'http://127.0.0.1:7890',
    })

    expect(options.agent).toBeDefined()
    expect(options.headers).toEqual({
      Authorization: 'Bearer sk-test',
    })
  })

  it('normalizes websocket base urls with the same realtime path rules', () => {
    expect(
      buildUpstreamUrl(
        {
          ...nexusConfig,
          upstreamBaseUrl: 'ws://127.0.0.1:8000/v1',
        },
        'test-model',
      ),
    ).toBe('ws://127.0.0.1:8000/v1/realtime?model=test-model')
  })

  it('tolerates base urls that already point at /realtime', () => {
    expect(
      buildUpstreamUrl(
        {
          ...openAiConfig,
          upstreamBaseUrl: 'wss://api.openai.com/v1/realtime',
        },
        'gpt-realtime',
      ),
    ).toBe('wss://api.openai.com/v1/realtime?model=gpt-realtime')
  })

  it('translates commit into a single commit event in server vad mode', () => {
    expect(translateClientEvent({ type: 'app.audio.commit' }, 'gpt-realtime', 'text', 'alloy')).toEqual([
      { type: 'input_audio_buffer.commit' },
    ])
  })

  it('builds the fixed session update payload with server vad enabled', () => {
    const event = buildSessionUpdateEvent('gpt-realtime', 'audio', 'paimon')
    expect(event).toMatchObject({
      type: 'session.update',
      session: {
        model: 'gpt-realtime',
        output_modalities: ['audio'],
        audio: {
          input: {
            transcription: {
              model: 'gpt-4o-mini-transcribe',
            },
            turn_detection: {
              type: 'server_vad',
              create_response: true,
              interrupt_response: true,
            },
          },
          output: {
            voice: 'paimon',
          },
        },
      },
    })
  })
})

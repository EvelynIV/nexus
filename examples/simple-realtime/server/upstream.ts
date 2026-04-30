import { HttpsProxyAgent } from 'https-proxy-agent'
import type { ServerConfig } from './config'
import type { ClientOptions } from 'ws'
import type { AppClientEvent, OutputMode, RawRealtimeEvent } from '../src/lib/realtime/app-protocol'

function normalizeWebSocketUrl(rawUrl: string): URL {
  const url = new URL(rawUrl)

  if (url.protocol === 'http:') {
    url.protocol = 'ws:'
  } else if (url.protocol === 'https:') {
    url.protocol = 'wss:'
  }

  return url
}

function joinRealtimePath(pathname: string): string {
  const trimmedPath = pathname.replace(/\/+$/, '')

  if (!trimmedPath || trimmedPath === '/') {
    return '/realtime'
  }

  if (trimmedPath.endsWith('/realtime')) {
    return trimmedPath
  }

  return `${trimmedPath}/realtime`
}

export function buildUpstreamUrl(config: ServerConfig, model: string): string {
  const url = normalizeWebSocketUrl(config.upstreamBaseUrl)
  url.pathname = joinRealtimePath(url.pathname)
  url.searchParams.set('model', model)
  return url.toString()
}

export function buildUpstreamHeaders(config: ServerConfig): Record<string, string> | undefined {
  if (!config.openAiApiKey) {
    return undefined
  }

  return {
    Authorization: `Bearer ${config.openAiApiKey!}`,
  }
}

export function buildUpstreamSocketOptions(config: ServerConfig): ClientOptions {
  return {
    headers: buildUpstreamHeaders(config),
    agent: config.upstreamProxyUrl ? new HttpsProxyAgent(config.upstreamProxyUrl) : undefined,
  }
}

export function buildSessionUpdateEvent(model: string, outputMode: OutputMode, voice: string): RawRealtimeEvent {
  return {
    type: 'session.update',
    session: {
      type: 'realtime',
      model,
      output_modalities: [outputMode],
      audio: {
        input: {
          format: {
            type: 'audio/pcm',
            rate: 24000,
          },
          transcription: {
            model: 'gpt-4o-mini-transcribe',
          },
          turn_detection: {
            type: 'server_vad',
            threshold: 0.5,
            prefix_padding_ms: 300,
            silence_duration_ms: 500,
            create_response: true,
            interrupt_response: true,
          },
        },
        output: {
          format: {
            type: 'audio/pcm',
            rate: 24000,
          },
          voice,
          speed: 1,
        },
      },
    },
  }
}

export function translateClientEvent(
  event: AppClientEvent,
  currentModel: string,
  currentOutputMode: OutputMode,
  currentVoice: string,
): RawRealtimeEvent[] {
  switch (event.type) {
    case 'app.connect':
      return [buildSessionUpdateEvent(event.model || currentModel, event.outputMode, event.voice || currentVoice)]
    case 'app.session.update':
      return [
        buildSessionUpdateEvent(
          event.model || currentModel,
          event.outputMode ?? currentOutputMode,
          event.voice || currentVoice,
        ),
      ]
    case 'app.audio.append':
      return [
        {
          type: 'input_audio_buffer.append',
          audio: event.audioBase64,
        },
      ]
    case 'app.audio.commit':
      return [{ type: 'input_audio_buffer.commit' }]
    case 'app.audio.clear':
      return [{ type: 'input_audio_buffer.clear' }]
    case 'app.disconnect':
      return []
  }

  throw new Error(`Unsupported client event: ${(event as { type: string }).type}`)
}

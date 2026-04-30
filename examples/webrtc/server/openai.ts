import { ProxyAgent } from 'undici'
import type { OutputMode } from '../src/lib/realtime/app-protocol'

export interface TokenRequest {
  model: string
  outputMode: OutputMode
  voice: string
}

export interface RealtimeSession {
  type?: string
  model?: string
  output_modalities?: string[]
  audio?: {
    input?: {
      transcription?: unknown | null
      [key: string]: unknown
    }
    output?: {
      voice?: string
      [key: string]: unknown
    }
    [key: string]: unknown
  }
  [key: string]: unknown
}

export interface ClientSecretResponse {
  value: string
  expires_at: number
  session?: RealtimeSession
  [key: string]: unknown
}

interface ClientSecretRequest {
  expires_after: {
    anchor: 'created_at'
    seconds: number
  }
  session: Record<string, unknown>
}

export type FetchLike = typeof fetch

export class OpenAiRequestError extends Error {
  readonly status: number
  readonly detail: string

  constructor(status: number, detail: string) {
    super(detail || `OpenAI request failed with status ${status}.`)
    this.name = 'OpenAiRequestError'
    this.status = status
    this.detail = detail
  }
}

export class OpenAiNetworkError extends Error {
  readonly status: number
  readonly detail: string

  constructor(status: number, detail: string) {
    super(detail)
    this.name = 'OpenAiNetworkError'
    this.status = status
    this.detail = detail
  }
}

interface CreateClientSecretOptions {
  apiKey: string
  realtimeBaseUrl: string
  fetchImpl?: FetchLike
  upstreamProxyUrl?: string | null
  upstreamConnectTimeoutMs?: number
}

function describeNetworkFailure(
  error: unknown,
  options: {
    upstreamProxyUrl: string | null
    upstreamConnectTimeoutMs: number
  },
): OpenAiNetworkError {
  const normalizedError = error instanceof Error ? error : new Error(String(error))
  const isTimeout = normalizedError.name === 'TimeoutError' || normalizedError.name === 'AbortError'

  if (isTimeout) {
    return new OpenAiNetworkError(
      504,
      `OpenAI token request timed out after ${options.upstreamConnectTimeoutMs}ms`,
    )
  }

  if (options.upstreamProxyUrl) {
    return new OpenAiNetworkError(
      502,
      `OpenAI token request failed via proxy (${options.upstreamProxyUrl}): ${normalizedError.message}`,
    )
  }

  return new OpenAiNetworkError(
    502,
    `OpenAI token request failed; set HTTPS_PROXY/HTTP_PROXY if needed: ${normalizedError.message}`,
  )
}

export function parseTokenRequest(payload: unknown): TokenRequest {
  if (!payload || typeof payload !== 'object') {
    throw new Error('Invalid token request body.')
  }

  const maybePayload = payload as Record<string, unknown>
  const model = typeof maybePayload.model === 'string' ? maybePayload.model.trim() : ''
  const voice = typeof maybePayload.voice === 'string' ? maybePayload.voice.trim() : ''
  const outputMode = maybePayload.outputMode

  if (!model) {
    throw new Error('Model is required.')
  }

  if (!voice) {
    throw new Error('Voice is required.')
  }

  if (outputMode !== 'text' && outputMode !== 'audio') {
    throw new Error('outputMode must be "text" or "audio".')
  }

  return {
    model,
    outputMode,
    voice,
  }
}

export function buildClientSecretRequest(
  request: TokenRequest,
  includeInputTranscription = true,
): ClientSecretRequest {
  const session: Record<string, unknown> = {
    type: 'realtime',
    model: request.model,
    output_modalities: [request.outputMode],
    audio: {
      input: {
        turn_detection: {
          type: 'server_vad',
          create_response: true,
          interrupt_response: true,
        },
        ...(includeInputTranscription
          ? {
              transcription: {
                model: 'gpt-4o-mini-transcribe',
              },
            }
          : {}),
      },
      output: {
        voice: request.voice,
      },
    },
  }

  return {
    expires_after: {
      anchor: 'created_at',
      seconds: 600,
    },
    session,
  }
}

async function readErrorDetail(response: Response): Promise<string> {
  const contentType = response.headers.get('content-type') ?? ''

  if (contentType.includes('application/json')) {
    const payload = (await response.json()) as
      | { error?: { message?: string } | string; message?: string }
      | null
    if (typeof payload?.error === 'string') {
      return payload.error
    }
    if (payload?.error && typeof payload.error === 'object' && typeof payload.error.message === 'string') {
      return payload.error.message
    }
    if (typeof payload?.message === 'string') {
      return payload.message
    }
    return JSON.stringify(payload)
  }

  return (await response.text()) || `OpenAI request failed with status ${response.status}.`
}

async function postClientSecret(
  request: ClientSecretRequest,
  options: {
    apiKey: string
    realtimeBaseUrl: string
    fetchImpl: FetchLike
    upstreamProxyUrl: string | null
    upstreamConnectTimeoutMs: number
  },
): Promise<ClientSecretResponse> {
  const proxyAgent = options.upstreamProxyUrl ? new ProxyAgent(options.upstreamProxyUrl) : null

  try {
    const fetchInit: RequestInit = {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${options.apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
      signal: AbortSignal.timeout(options.upstreamConnectTimeoutMs),
      dispatcher: (proxyAgent ?? undefined) as RequestInit['dispatcher'],
    }

    const response = await options.fetchImpl(
      `${options.realtimeBaseUrl.replace(/\/$/, '')}/v1/realtime/client_secrets`,
      fetchInit,
    )

    if (!response.ok) {
      throw new OpenAiRequestError(response.status, await readErrorDetail(response))
    }

    return (await response.json()) as ClientSecretResponse
  } catch (error) {
    if (error instanceof OpenAiRequestError) {
      throw error
    }

    throw describeNetworkFailure(error, {
      upstreamProxyUrl: options.upstreamProxyUrl,
      upstreamConnectTimeoutMs: options.upstreamConnectTimeoutMs,
    })
  } finally {
    await proxyAgent?.close()
  }
}

export async function createClientSecret(
  request: TokenRequest,
  options: CreateClientSecretOptions,
): Promise<ClientSecretResponse> {
  const fetchImpl = options.fetchImpl ?? fetch
  const upstreamProxyUrl = options.upstreamProxyUrl ?? null
  const upstreamConnectTimeoutMs = options.upstreamConnectTimeoutMs ?? 15000

  try {
    return await postClientSecret(buildClientSecretRequest(request, true), {
      apiKey: options.apiKey,
      realtimeBaseUrl: options.realtimeBaseUrl,
      fetchImpl,
      upstreamProxyUrl,
      upstreamConnectTimeoutMs,
    })
  } catch (error) {
    if (!(error instanceof OpenAiRequestError)) {
      throw error
    }

    if (error.status !== 400 && error.status !== 422) {
      throw error
    }

    return postClientSecret(buildClientSecretRequest(request, false), {
      apiKey: options.apiKey,
      realtimeBaseUrl: options.realtimeBaseUrl,
      fetchImpl,
      upstreamProxyUrl,
      upstreamConnectTimeoutMs,
    })
  }
}

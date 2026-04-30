import { config as loadDotEnv } from 'dotenv'

loadDotEnv()

export interface ServerConfig {
  port: number
  openAiApiKey: string
  realtimeBaseUrl: string
  upstreamProxyUrl: string | null
  upstreamConnectTimeoutMs: number
}

function parsePort(rawPort: string | undefined): number {
  const fallback = 8790

  if (!rawPort) {
    return fallback
  }

  const parsed = Number(rawPort)
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`Invalid WEBRTC_SERVER_PORT: ${rawPort}`)
  }

  return parsed
}

function parseTimeout(rawTimeout: string | undefined): number {
  const fallback = 15000

  if (!rawTimeout) {
    return fallback
  }

  const parsed = Number(rawTimeout)
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`Invalid WEBRTC_UPSTREAM_CONNECT_TIMEOUT_MS: ${rawTimeout}`)
  }

  return parsed
}

function parseBaseUrl(rawBaseUrl: string | undefined): string {
  const fallback = 'https://api.openai.com'
  const value = rawBaseUrl?.trim() || fallback

  try {
    const url = new URL(value)
    return url.toString().replace(/\/$/, '')
  } catch {
    throw new Error(`Invalid WEBRTC_REALTIME_BASE_URL: ${rawBaseUrl}`)
  }
}

export function loadServerConfig(env: NodeJS.ProcessEnv = process.env): ServerConfig {
  const config: ServerConfig = {
    port: parsePort(env.WEBRTC_SERVER_PORT),
    openAiApiKey: env.WEBRTC_OPENAI_API_KEY ?? '',
    realtimeBaseUrl: parseBaseUrl(env.WEBRTC_REALTIME_BASE_URL),
    upstreamProxyUrl: env.HTTPS_PROXY ?? env.HTTP_PROXY ?? null,
    upstreamConnectTimeoutMs: parseTimeout(env.WEBRTC_UPSTREAM_CONNECT_TIMEOUT_MS),
  }

  if (!config.openAiApiKey) {
    throw new Error('WEBRTC_OPENAI_API_KEY is required')
  }

  return config
}

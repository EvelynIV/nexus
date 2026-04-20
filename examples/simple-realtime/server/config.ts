import { config as loadDotEnv } from 'dotenv'

loadDotEnv()

export interface ServerConfig {
  port: number
  defaultVoice: string
  upstreamBaseUrl: string
  openAiApiKey: string | null
  upstreamProxyUrl: string | null
  upstreamConnectTimeoutMs: number
}

function parsePort(rawPort: string | undefined): number {
  const fallback = 8787

  if (!rawPort) {
    return fallback
  }

  const parsed = Number(rawPort)
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`Invalid SIMPLE_REALTIME_SERVER_PORT: ${rawPort}`)
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
    throw new Error(`Invalid SIMPLE_REALTIME_UPSTREAM_CONNECT_TIMEOUT_MS: ${rawTimeout}`)
  }

  return parsed
}

export function loadServerConfig(env: NodeJS.ProcessEnv = process.env): ServerConfig {
  const config: ServerConfig = {
    port: parsePort(env.SIMPLE_REALTIME_SERVER_PORT),
    defaultVoice: env.VITE_REALTIME_VOICE ?? 'alloy',
    upstreamBaseUrl: env.SIMPLE_REALTIME_UPSTREAM_BASE_URL ?? '',
    openAiApiKey: env.SIMPLE_REALTIME_OPENAI_API_KEY ?? null,
    upstreamProxyUrl: env.HTTPS_PROXY ?? env.HTTP_PROXY ?? null,
    upstreamConnectTimeoutMs: parseTimeout(env.SIMPLE_REALTIME_UPSTREAM_CONNECT_TIMEOUT_MS),
  }

  if (!config.upstreamBaseUrl) {
    throw new Error('SIMPLE_REALTIME_UPSTREAM_BASE_URL is required')
  }

  return config
}

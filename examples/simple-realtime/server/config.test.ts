import { describe, expect, it } from 'vitest'
import { loadServerConfig } from './config'

describe('loadServerConfig', () => {
  it('requires upstream base url', () => {
    expect(() =>
      loadServerConfig({
        SIMPLE_REALTIME_SERVER_PORT: '8787',
      }),
    ).toThrow('SIMPLE_REALTIME_UPSTREAM_BASE_URL')
  })

  it('accepts a config without api key', () => {
    expect(() =>
      loadServerConfig({
        SIMPLE_REALTIME_SERVER_PORT: '8787',
        SIMPLE_REALTIME_UPSTREAM_BASE_URL: 'https://api.openai.com/v1',
      }),
    ).not.toThrow()
  })

  it('returns defaults when config is valid', () => {
    const config = loadServerConfig({
      SIMPLE_REALTIME_SERVER_PORT: '8787',
      SIMPLE_REALTIME_UPSTREAM_BASE_URL: 'http://127.0.0.1:8000/v1',
    })

    expect(config.defaultVoice).toBe('alloy')
    expect(config.upstreamBaseUrl).toBe('http://127.0.0.1:8000/v1')
    expect(config.upstreamConnectTimeoutMs).toBe(15000)
  })

  it('keeps an optional api key when provided', () => {
    const config = loadServerConfig({
      SIMPLE_REALTIME_SERVER_PORT: '8787',
      SIMPLE_REALTIME_UPSTREAM_BASE_URL: 'https://api.openai.com/v1',
      SIMPLE_REALTIME_OPENAI_API_KEY: 'sk-test',
      HTTPS_PROXY: 'http://127.0.0.1:7890',
    })

    expect(config.openAiApiKey).toBe('sk-test')
    expect(config.upstreamProxyUrl).toBe('http://127.0.0.1:7890')
  })

  it('keeps a default voice when provided', () => {
    const config = loadServerConfig({
      SIMPLE_REALTIME_SERVER_PORT: '8787',
      SIMPLE_REALTIME_UPSTREAM_BASE_URL: 'http://127.0.0.1:8000/v1',
      VITE_REALTIME_VOICE: 'paimon',
    })

    expect(config.defaultVoice).toBe('paimon')
  })
})

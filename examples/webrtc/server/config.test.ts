import { describe, expect, it } from 'vitest'
import { loadServerConfig } from './config'

describe('loadServerConfig', () => {
  it('requires an OpenAI API key', () => {
    expect(() =>
      loadServerConfig({
        WEBRTC_SERVER_PORT: '8790',
      }),
    ).toThrow('WEBRTC_OPENAI_API_KEY')
  })

  it('returns defaults when config is valid', () => {
    const config = loadServerConfig({
      WEBRTC_OPENAI_API_KEY: 'sk-test',
    })

    expect(config.port).toBe(8790)
    expect(config.openAiApiKey).toBe('sk-test')
    expect(config.realtimeBaseUrl).toBe('https://api.openai.com')
    expect(config.upstreamProxyUrl).toBeNull()
    expect(config.upstreamConnectTimeoutMs).toBe(15000)
  })

  it('keeps an explicit port when provided', () => {
    const config = loadServerConfig({
      WEBRTC_SERVER_PORT: '9900',
      WEBRTC_OPENAI_API_KEY: 'sk-test',
    })

    expect(config.port).toBe(9900)
  })

  it('reads proxy settings from the environment', () => {
    const config = loadServerConfig({
      WEBRTC_OPENAI_API_KEY: 'sk-test',
      HTTPS_PROXY: 'http://127.0.0.1:7890',
    })

    expect(config.upstreamProxyUrl).toBe('http://127.0.0.1:7890')
  })

  it('reads an explicit realtime base url', () => {
    const config = loadServerConfig({
      WEBRTC_OPENAI_API_KEY: 'sk-test',
      WEBRTC_REALTIME_BASE_URL: 'http://127.0.0.1:8000/',
    })

    expect(config.realtimeBaseUrl).toBe('http://127.0.0.1:8000')
  })

  it('rejects invalid upstream timeout values', () => {
    expect(() =>
      loadServerConfig({
        WEBRTC_OPENAI_API_KEY: 'sk-test',
        WEBRTC_UPSTREAM_CONNECT_TIMEOUT_MS: '0',
      }),
    ).toThrow('WEBRTC_UPSTREAM_CONNECT_TIMEOUT_MS')
  })
})

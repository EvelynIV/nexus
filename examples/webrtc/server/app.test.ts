import type { AddressInfo } from 'node:net'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createAppServer } from './app'
import type { ServerConfig } from './config'

const baseConfig: ServerConfig = {
  port: 0,
  openAiApiKey: 'sk-test',
  realtimeBaseUrl: 'https://api.openai.com',
  upstreamProxyUrl: null,
  upstreamConnectTimeoutMs: 15000,
}

const servers: Array<ReturnType<typeof createAppServer>> = []

async function startServer(fetchImpl?: typeof fetch): Promise<{
  baseUrl: string
  close: () => Promise<void>
}> {
  const server = createAppServer(baseConfig, fetchImpl ? { fetchImpl } : {})
  servers.push(server)

  await new Promise<void>((resolve) => {
    server.listen(0, '127.0.0.1', resolve)
  })

  const address = server.address() as AddressInfo

  return {
    baseUrl: `http://127.0.0.1:${address.port}`,
    close: () =>
      new Promise<void>((resolve, reject) => {
        server.close((error) => {
          if (error) {
            reject(error)
            return
          }
          resolve()
        })
      }),
  }
}

afterEach(async () => {
  await Promise.all(
    servers.splice(0).map(
      (server) =>
        new Promise<void>((resolve) => {
          server.close(() => resolve())
        }),
    ),
  )
})

describe('createAppServer', () => {
  it('exposes a health endpoint', async () => {
    const { baseUrl } = await startServer()

    const response = await fetch(`${baseUrl}/api/health`)

    expect(response.status).toBe(200)
    await expect(response.json()).resolves.toEqual({ status: 'ok' })
  })

  it('proxies token creation requests to OpenAI', async () => {
    const fetchImpl = vi.fn(async () => {
      return new Response(
        JSON.stringify({
          value: 'ek_test',
          expires_at: 123,
          session: {
            type: 'realtime',
          },
        }),
        {
          status: 200,
          headers: {
            'content-type': 'application/json',
          },
        },
      )
    })

    const { baseUrl } = await startServer(fetchImpl)
    const response = await fetch(`${baseUrl}/api/token`, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
      },
      body: JSON.stringify({
        model: 'gpt-realtime',
        outputMode: 'audio',
        voice: 'alloy',
      }),
    })

    expect(response.status).toBe(200)
    await expect(response.json()).resolves.toMatchObject({
      value: 'ek_test',
      expires_at: 123,
    })
    expect(fetchImpl).toHaveBeenCalledTimes(1)
  })

  it('rejects invalid request payloads', async () => {
    const { baseUrl } = await startServer()
    const response = await fetch(`${baseUrl}/api/token`, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
      },
      body: JSON.stringify({
        model: '',
        outputMode: 'audio',
        voice: 'alloy',
      }),
    })

    expect(response.status).toBe(400)
    await expect(response.json()).resolves.toEqual({
      error: 'Model is required.',
    })
  })

  it('surfaces explicit upstream network failures', async () => {
    const fetchImpl = vi.fn(async () => {
      throw new Error('fetch failed')
    })

    const { baseUrl } = await startServer(fetchImpl)
    const response = await fetch(`${baseUrl}/api/token`, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
      },
      body: JSON.stringify({
        model: 'gpt-realtime',
        outputMode: 'audio',
        voice: 'alloy',
      }),
    })

    expect(response.status).toBe(502)
    await expect(response.json()).resolves.toEqual({
      error: 'OpenAI token request failed; set HTTPS_PROXY/HTTP_PROXY if needed: fetch failed',
    })
  })
})

import { describe, expect, it, vi } from 'vitest'
import {
  buildClientSecretRequest,
  createClientSecret,
  OpenAiNetworkError,
  OpenAiRequestError,
  parseTokenRequest,
} from './openai'

describe('parseTokenRequest', () => {
  it('normalizes a valid request payload', () => {
    expect(
      parseTokenRequest({
        model: ' gpt-realtime ',
        outputMode: 'audio',
        voice: ' alloy ',
      }),
    ).toEqual({
      model: 'gpt-realtime',
      outputMode: 'audio',
      voice: 'alloy',
    })
  })

  it('rejects invalid output modes', () => {
    expect(() =>
      parseTokenRequest({
        model: 'gpt-realtime',
        outputMode: 'video',
        voice: 'alloy',
      }),
    ).toThrow('outputMode')
  })
})

describe('buildClientSecretRequest', () => {
  it('includes VAD, voice, TTL, and transcription by default', () => {
    const payload = buildClientSecretRequest(
      {
        model: 'gpt-realtime',
        outputMode: 'audio',
        voice: 'alloy',
      },
      true,
    )

    expect(payload.expires_after.seconds).toBe(600)
    expect(payload.session.type).toBe('realtime')
    expect(payload.session.model).toBe('gpt-realtime')
    expect(payload.session.output_modalities).toEqual(['audio'])
    expect(payload.session.audio).toEqual({
      input: {
        turn_detection: {
          type: 'server_vad',
          create_response: true,
          interrupt_response: true,
        },
        transcription: {
          model: 'gpt-4o-mini-transcribe',
        },
      },
      output: {
        voice: 'alloy',
      },
    })
  })

  it('omits input transcription when fallback mode is used', () => {
    const payload = buildClientSecretRequest(
      {
        model: 'gpt-realtime',
        outputMode: 'text',
        voice: 'alloy',
      },
      false,
    )

    expect(payload.session.output_modalities).toEqual(['text'])
    expect(payload.session.audio).toEqual({
      input: {
        turn_detection: {
          type: 'server_vad',
          create_response: true,
          interrupt_response: true,
        },
      },
      output: {
        voice: 'alloy',
      },
    })
  })
})

describe('createClientSecret', () => {
  it('forwards the OpenAI auth header and request body', async () => {
    const fetchImpl = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
      const requestInit = init as RequestInit & { dispatcher?: unknown }

      expect(requestInit.headers).toEqual({
        Authorization: 'Bearer sk-test',
        'Content-Type': 'application/json',
      })
      expect(requestInit.signal).toBeInstanceOf(AbortSignal)
      expect(requestInit.dispatcher).toBeUndefined()

      const body = JSON.parse(String(requestInit.body)) as {
        session: { output_modalities: string[]; audio: { output: { voice: string } } }
      }
      expect(body.session.output_modalities).toEqual(['audio'])
      expect(body.session.audio.output.voice).toBe('alloy')

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

    const secret = await createClientSecret(
      {
        model: 'gpt-realtime',
        outputMode: 'audio',
        voice: 'alloy',
      },
      {
        apiKey: 'sk-test',
        realtimeBaseUrl: 'https://api.openai.com',
        fetchImpl,
        upstreamConnectTimeoutMs: 15000,
      },
    )

    expect(secret.value).toBe('ek_test')
    expect(fetchImpl).toHaveBeenCalledTimes(1)
  })

  it('retries without input transcription when OpenAI rejects the first request', async () => {
    const fetchImpl = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            error: {
              message: 'audio.input.transcription is not supported for this model',
            },
          }),
          {
            status: 400,
            headers: {
              'content-type': 'application/json',
            },
          },
        ),
      )
      .mockImplementationOnce(async (_url: string | URL | Request, init?: RequestInit) => {
        const requestInit = init as RequestInit & { dispatcher?: unknown }
        expect(requestInit.dispatcher).toBeUndefined()

        const body = JSON.parse(String(requestInit.body)) as {
          session: { audio: { input: Record<string, unknown> } }
        }

        expect(body.session.audio.input.transcription).toBeUndefined()

        return new Response(
          JSON.stringify({
            value: 'ek_retry',
            expires_at: 456,
            session: {
              audio: {
                input: {
                  transcription: null,
                },
              },
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

    const secret = await createClientSecret(
      {
        model: 'gpt-realtime',
        outputMode: 'audio',
        voice: 'alloy',
      },
      {
        apiKey: 'sk-test',
        realtimeBaseUrl: 'https://api.openai.com',
        fetchImpl,
        upstreamConnectTimeoutMs: 15000,
      },
    )

    expect(secret.value).toBe('ek_retry')
    expect(fetchImpl).toHaveBeenCalledTimes(2)
  })

  it('surfaces non-retryable OpenAI failures', async () => {
    await expect(
      createClientSecret(
        {
          model: 'gpt-realtime',
          outputMode: 'audio',
          voice: 'alloy',
        },
        {
          apiKey: 'sk-test',
          realtimeBaseUrl: 'https://api.openai.com',
          upstreamConnectTimeoutMs: 15000,
          fetchImpl: vi.fn(async () => {
            return new Response(
              JSON.stringify({
                error: {
                  message: 'bad auth',
                },
              }),
              {
                status: 401,
                headers: {
                  'content-type': 'application/json',
                },
              },
            )
          }),
        },
      ),
    ).rejects.toBeInstanceOf(OpenAiRequestError)
  })

  it('attaches a proxy dispatcher when a proxy url is configured', async () => {
    const fetchImpl = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
      const requestInit = init as RequestInit & { dispatcher?: unknown }
      expect(requestInit.dispatcher).toBeTruthy()

      return new Response(
        JSON.stringify({
          value: 'ek_test',
          expires_at: 123,
        }),
        {
          status: 200,
          headers: {
            'content-type': 'application/json',
          },
        },
      )
    })

    const secret = await createClientSecret(
      {
        model: 'gpt-realtime',
        outputMode: 'audio',
        voice: 'alloy',
      },
      {
        apiKey: 'sk-test',
        realtimeBaseUrl: 'https://api.openai.com',
        fetchImpl,
        upstreamProxyUrl: 'http://127.0.0.1:7890',
        upstreamConnectTimeoutMs: 15000,
      },
    )

    expect(secret.value).toBe('ek_test')
  })

  it('wraps timeout failures with an explicit upstream timeout message', async () => {
    const timeoutError = Object.assign(new Error('The operation was aborted due to timeout'), {
      name: 'TimeoutError',
    })

    await expect(
      createClientSecret(
        {
          model: 'gpt-realtime',
          outputMode: 'audio',
          voice: 'alloy',
        },
        {
          apiKey: 'sk-test',
          realtimeBaseUrl: 'https://api.openai.com',
          upstreamConnectTimeoutMs: 15000,
          fetchImpl: vi.fn(async () => {
            throw timeoutError
          }),
        },
      ),
    ).rejects.toMatchObject({
      status: 504,
      message: 'OpenAI token request timed out after 15000ms',
    })
  })

  it('wraps non-timeout network failures with actionable guidance', async () => {
    await expect(
      createClientSecret(
        {
          model: 'gpt-realtime',
          outputMode: 'audio',
          voice: 'alloy',
        },
        {
          apiKey: 'sk-test',
          realtimeBaseUrl: 'https://api.openai.com',
          upstreamConnectTimeoutMs: 15000,
          fetchImpl: vi.fn(async () => {
            throw new Error('fetch failed')
          }),
        },
      ),
    ).rejects.toMatchObject({
      status: 502,
      message: 'OpenAI token request failed; set HTTPS_PROXY/HTTP_PROXY if needed: fetch failed',
    })
  })
})

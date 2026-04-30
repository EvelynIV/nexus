import http from 'node:http'
import type { ServerConfig } from './config'
import {
  createClientSecret,
  OpenAiNetworkError,
  OpenAiRequestError,
  parseTokenRequest,
  type FetchLike,
} from './openai'

function sendJson(
  res: http.ServerResponse,
  statusCode: number,
  payload: unknown,
): void {
  res.writeHead(statusCode, {
    'content-type': 'application/json',
  })
  res.end(JSON.stringify(payload))
}

async function readBody(req: http.IncomingMessage): Promise<string> {
  const chunks: Buffer[] = []

  for await (const chunk of req) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk))
  }

  return Buffer.concat(chunks).toString('utf8')
}

export function createAppServer(
  config: ServerConfig,
  options: {
    fetchImpl?: FetchLike
  } = {},
): http.Server {
  const fetchImpl = options.fetchImpl ?? fetch

  return http.createServer(async (req, res) => {
    const url = new URL(req.url ?? '/', 'http://127.0.0.1')

    if (req.method === 'GET' && url.pathname === '/api/health') {
      sendJson(res, 200, { status: 'ok' })
      return
    }

    if (req.method === 'POST' && url.pathname === '/api/token') {
      try {
        const body = await readBody(req)
        const request = parseTokenRequest(JSON.parse(body || '{}'))
        const secret = await createClientSecret(request, {
          apiKey: config.openAiApiKey,
          realtimeBaseUrl: config.realtimeBaseUrl,
          fetchImpl,
          upstreamProxyUrl: config.upstreamProxyUrl,
          upstreamConnectTimeoutMs: config.upstreamConnectTimeoutMs,
        })
        sendJson(res, 200, secret)
      } catch (error) {
        if (error instanceof SyntaxError) {
          sendJson(res, 400, { error: 'Invalid JSON body.' })
          return
        }

        if (error instanceof OpenAiRequestError) {
          sendJson(res, error.status, { error: error.message })
          return
        }

        if (error instanceof OpenAiNetworkError) {
          sendJson(res, error.status, { error: error.message })
          return
        }

        if (error instanceof Error) {
          sendJson(res, 400, { error: error.message })
          return
        }

        sendJson(res, 500, { error: 'Unknown server error.' })
      }
      return
    }

    sendJson(res, 404, { error: 'not_found' })
  })
}

import http from 'node:http'
import { createAppServer } from './app'
import { loadServerConfig } from './config'

const config = loadServerConfig()
const server = createAppServer(config)

server.listen(config.port, () => {
  console.log(`[webrtc-demo] backend listening on http://127.0.0.1:${config.port}`)
})

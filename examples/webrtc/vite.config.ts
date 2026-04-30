import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const serverPort = env.WEBRTC_SERVER_PORT || '8790'
  const realtimeBaseUrl = env.VITE_REALTIME_BASE_URL || 'https://api.openai.com'

  return {
    plugins: [react()],
    // Keep server-only variables out of the browser bundle.
    envPrefix: ['VITE_'],
    server: {
      proxy: {
        '/api': {
          target: `http://127.0.0.1:${serverPort}`,
        },
        '/v1': {
          target: realtimeBaseUrl,
          changeOrigin: true,
          secure: false,
        },
      },
    },
    test: {
      environment: 'node',
      globals: true,
      include: ['src/**/*.test.ts', 'server/**/*.test.ts'],
    },
  }
})

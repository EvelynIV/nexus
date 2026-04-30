import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const serverPort = env.SIMPLE_REALTIME_SERVER_PORT || '8787'

  return {
    plugins: [react()],
    envPrefix: ['VITE_', 'SIMPLE_REALTIME_'],
    server: {
      proxy: {
        '/ws': {
          target: `ws://127.0.0.1:${serverPort}`,
          ws: true,
        },
        '/api': {
          target: `http://127.0.0.1:${serverPort}`,
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

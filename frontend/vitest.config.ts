import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: './vitest.setup.ts',
    css: false,
  },
  resolve: {
    alias: {
      // PRD-261 A-10a: the theme SDK by the name a theme imports it by (tsconfig.json paths too).
      '@automatos/theme-sdk': path.resolve(__dirname, 'lib/theme-sdk/index.ts'),
      '@': path.resolve(__dirname, '.'),
    },
  },
})

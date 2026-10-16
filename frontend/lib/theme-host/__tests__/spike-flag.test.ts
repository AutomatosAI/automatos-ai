/**
 * PRD-261 A-01 (D3) — the theme-host spike is a build-time switch. next.config.js inlines
 * AUTOMATOS_THEME_HOST_SPIKE as '1' or '0', and no file that builds or deploys a shipped image
 * names the variable, so every shipped build has the spike off.
 */
import { readFileSync } from 'fs'
import { createRequire } from 'module'
import path from 'path'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { isThemeHostSpikeOn } from '../spike-flag'

const FLAG = 'AUTOMATOS_THEME_HOST_SPIKE'
const load = createRequire(import.meta.url)
const CONFIG = load.resolve('../../../next.config.js')
const REPO_ROOT = path.resolve(__dirname, '../../../..')
const SHIPPED_BUILD_FILES = [
  'frontend/Dockerfile',
  'docker-compose.yml',
  'docker-compose.dev.yml',
  'docker-compose.images.yml',
  'envs/frontend.defaults',
  'railway.json',
  '.github/workflows/images.yml',
]

function configEnvWith(value: string | undefined): Record<string, string> {
  vi.stubEnv(FLAG, value)
  delete load.cache[CONFIG]
  const config = load(CONFIG) as { env?: Record<string, string> }
  return config.env ?? {}
}

afterEach(() => {
  vi.unstubAllEnvs()
  delete load.cache[CONFIG]
})

describe('PRD-261 A-01 — the spike flag', () => {
  it("is on only for exactly '1'", () => {
    vi.stubEnv(FLAG, undefined)
    expect(isThemeHostSpikeOn()).toBe(false)
    vi.stubEnv(FLAG, '1')
    expect(isThemeHostSpikeOn()).toBe(true)
    vi.stubEnv(FLAG, 'true')
    expect(isThemeHostSpikeOn()).toBe(false)
    vi.stubEnv(FLAG, '0')
    expect(isThemeHostSpikeOn()).toBe(false)
  })

  it("is inlined by next.config.js as '0' unless the build sets '1'", () => {
    expect(configEnvWith(undefined)[FLAG]).toBe('0')
    expect(configEnvWith('true')[FLAG]).toBe('0')
    expect(configEnvWith('1')[FLAG]).toBe('1')
  })

  it.each(SHIPPED_BUILD_FILES)('is never named by %s', (file) => {
    expect(readFileSync(path.join(REPO_ROOT, file), 'utf8')).not.toContain(FLAG)
  })
})

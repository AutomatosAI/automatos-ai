// @vitest-environment node
/**
 * PRD-277 US-505, PRD-279 US-402 — the enterprise web server checks the settings it reads at run
 * time when it starts, and stops on a bad one with a line naming the setting, never its value.
 * register() (instrumentation.ts) runs the check in the enterprise build's Node runtime only: the
 * local and saas builds, and the Edge runtime, never reach it.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

import { enterpriseStartUpProblem } from '@/lib/enterprise/start-up'
import { START_UP_REFUSED_EXIT_CODE, stopUnlessEnterpriseReady } from '@/lib/enterprise/start-up-node'

const READY = { MEDIA_ORIGINS: 'https://media.example.test', BACKEND_INTERNAL_URL: 'http://api.internal.test:8000' }

afterEach(() => {
  vi.unstubAllEnvs()
  vi.restoreAllMocks()
  vi.doUnmock('@/lib/enterprise/start-up-node')
  vi.resetModules()
})

describe('enterpriseStartUpProblem', () => {
  it('passes good settings, and no media origin at all', () => {
    expect(enterpriseStartUpProblem(READY)).toBeNull()
    expect(enterpriseStartUpProblem({ ...READY, MEDIA_ORIGINS: undefined })).toBeNull()
    expect(enterpriseStartUpProblem({ ...READY, BACKEND_INTERNAL_URL: 'https://api.internal.test' })).toBeNull()
  })

  it('refuses a media origin the policy may not name, naming the setting only', () => {
    const problem = enterpriseStartUpProblem({ ...READY, MEDIA_ORIGINS: 'https://media.example.test/files' })
    expect(problem).toContain('MEDIA_ORIGINS')
    expect(problem).not.toContain('media.example.test')
  })

  it.each([
    ['unset', undefined],
    ['blank', ' '],
    ['not an address', 'api-internal'],
    ['another scheme', 'ftp://api.internal.test'],
    ['an address with a user name', 'http://user:pw@api.internal.test'],
  ])('refuses BACKEND_INTERNAL_URL %s, naming the setting only', (_why, value) => {
    const problem = enterpriseStartUpProblem({ ...READY, BACKEND_INTERNAL_URL: value })
    expect(problem).toContain('BACKEND_INTERNAL_URL')
    if (value) expect(problem).not.toContain(value.trim())
  })
})

describe('stopUnlessEnterpriseReady', () => {
  it('does nothing when the settings are good', () => {
    const exit = vi.fn()
    stopUnlessEnterpriseReady(READY, exit)
    expect(exit).not.toHaveBeenCalled()
  })

  it('logs one line naming the setting, never its value, and stops the server', () => {
    const logged = vi.spyOn(console, 'error').mockImplementation(() => {})
    const exit = vi.fn()
    stopUnlessEnterpriseReady({ ...READY, MEDIA_ORIGINS: "'unsafe-inline'" }, exit)

    expect(exit).toHaveBeenCalledWith(START_UP_REFUSED_EXIT_CODE)
    expect(logged).toHaveBeenCalledTimes(1)
    expect(String(logged.mock.calls[0][0])).toContain('MEDIA_ORIGINS')
    expect(String(logged.mock.calls[0][0])).not.toContain('unsafe-inline')
  })
})

describe('register() runs the check in the enterprise Node server only', () => {
  async function registerAs(edition: string, runtime: string) {
    const stop = vi.fn()
    // Mocked by its resolved file, the same module instrumentation.ts imports by a relative path.
    vi.doMock('@/lib/enterprise/start-up-node', () => ({ stopUnlessEnterpriseReady: stop }))
    vi.stubEnv('NEXT_PUBLIC_AUTH_EDITION', edition)
    vi.stubEnv('NEXT_RUNTIME', runtime)
    vi.stubEnv('OTEL_ENABLED', '')
    const { register } = await import('../../../instrumentation')
    await register()
    return stop
  }

  it('in enterprise, on Node', async () => {
    expect(await registerAs('enterprise', 'nodejs')).toHaveBeenCalledTimes(1)
  })

  it.each([
    ['enterprise', 'edge'],
    ['local', 'nodejs'],
    ['saas', 'nodejs'],
  ])('not in %s on %s', async (edition, runtime) => {
    expect(await registerAs(edition, runtime)).not.toHaveBeenCalled()
  })
})

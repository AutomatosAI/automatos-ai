// @vitest-environment node
/**
 * PRD-260 D6 — hello plays once per browser session: sessionStorage remembers
 * it, and where storage is blocked it plays once per page load.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

import { blockedStorage, memoryStorage } from '@/lib/looks/__tests__/helpers'

afterEach(() => {
  vi.restoreAllMocks()
  vi.resetModules()
})

describe('claimFirstOpen', () => {
  it('says yes once per session, and remembers it', async () => {
    const { claimFirstOpen, HELLO_KEY } = await import('@/lib/mascot/session')
    const session = memoryStorage()
    expect(claimFirstOpen(() => session)).toBe(true)
    expect(session.getItem(HELLO_KEY)).not.toBeNull()
    expect(claimFirstOpen(() => session)).toBe(false)
    expect(claimFirstOpen(() => memoryStorage({ [HELLO_KEY]: '1' }))).toBe(false)
  })

  it('says yes once per page load where storage is blocked', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    const { claimFirstOpen } = await import('@/lib/mascot/session')
    expect(claimFirstOpen(blockedStorage)).toBe(true)
    expect(claimFirstOpen(blockedStorage)).toBe(false)
    expect(warn).toHaveBeenCalled()
  })
})

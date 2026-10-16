/**
 * PRD-261 A-01 (§6) — `?safe_mode=themes` turns theme loading off for the tab and is remembered in
 * the tab's sessionStorage. A storage that throws never breaks it: the query still works.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

import { SAFE_MODE_STORAGE_KEY } from '../constants'
import { isSafeModeRequested, safeSessionStorage, type SafeModeStorage } from '../safe-mode'

function memoryStorage(): SafeModeStorage & { readonly items: Map<string, string> } {
  const items = new Map<string, string>()
  return {
    items,
    getItem: (key: string) => items.get(key) ?? null,
    setItem: (key: string, value: string) => {
      items.set(key, value)
    },
  }
}

const THROWING_STORAGE: SafeModeStorage = {
  getItem: () => {
    throw new DOMException('blocked', 'SecurityError')
  },
  setItem: () => {
    throw new DOMException('blocked', 'SecurityError')
  },
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe('PRD-261 A-01 — isSafeModeRequested', () => {
  it('is on for ?safe_mode=themes and remembers it for the tab', () => {
    const storage = memoryStorage()
    expect(isSafeModeRequested('?safe_mode=themes', storage)).toBe(true)
    expect(storage.items.get(SAFE_MODE_STORAGE_KEY)).toBe('1')
  })

  it('stays on without the query once remembered', () => {
    const storage = memoryStorage()
    storage.setItem(SAFE_MODE_STORAGE_KEY, '1')
    expect(isSafeModeRequested('', storage)).toBe(true)
  })

  it('is off for any other value, or none', () => {
    expect(isSafeModeRequested('?safe_mode=other', memoryStorage())).toBe(false)
    expect(isSafeModeRequested('?mode=themes', memoryStorage())).toBe(false)
    expect(isSafeModeRequested('', memoryStorage())).toBe(false)
    expect(isSafeModeRequested('', null)).toBe(false)
  })

  it('still honours the query when the storage throws, and never throws itself', () => {
    vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    expect(isSafeModeRequested('?safe_mode=themes', THROWING_STORAGE)).toBe(true)
    expect(isSafeModeRequested('', THROWING_STORAGE)).toBe(false)
    expect(isSafeModeRequested('?safe_mode=themes', null)).toBe(true)
  })
})

describe('PRD-261 A-01 — safeSessionStorage', () => {
  it("returns this tab's sessionStorage", () => {
    expect(safeSessionStorage()).toBe(window.sessionStorage)
  })

  it('returns null when reading window.sessionStorage itself throws', () => {
    vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    vi.spyOn(window, 'sessionStorage', 'get').mockImplementation(() => {
      throw new DOMException('blocked', 'SecurityError')
    })
    expect(safeSessionStorage()).toBeNull()
  })
})

/**
 * PRD-261 A-04c — the failure ledger: per theme, inside a ten-minute window, in a storage that may
 * throw. Only the theme's own reasons count; `auth` (A-08b) never does.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  FAILURE_LEDGER_KEY,
  FAILURE_WINDOW_MS,
  FAILURES_BEFORE_REVERT,
  clearThemeFailures,
  countsTowardRevert,
  recordThemeFailure,
  safeLocalStorage,
  shouldRevert,
  type LedgerStorage,
} from '../failure-ledger'
import type { ThemeFailureReason } from '../types'

const THEME = 'theme:install:00000000-0000-4000-8000-0000000000a1'
const OTHER_THEME = 'theme:install:00000000-0000-4000-8000-0000000000a2'
const T0 = 1_700_000_000_000
const MINUTE_MS = 60_000

function memoryStorage(): LedgerStorage & { readonly items: Map<string, string> } {
  const items = new Map<string, string>()
  return {
    items,
    getItem: (key: string) => items.get(key) ?? null,
    setItem: (key: string, value: string) => {
      items.set(key, value)
    },
    removeItem: (key: string) => {
      items.delete(key)
    },
  }
}

function blocked(): never {
  throw new DOMException('blocked', 'SecurityError')
}

const THROWING_STORAGE: LedgerStorage = { getItem: blocked, setItem: blocked, removeItem: blocked }

function stored(storage: ReturnType<typeof memoryStorage>): unknown {
  return JSON.parse(storage.items.get(FAILURE_LEDGER_KEY) ?? 'null')
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe('PRD-261 A-04c — which failures count', () => {
  const counted: ThemeFailureReason[] = ['render', 'load', 'register', 'look', 'engines']
  const ignored: ThemeFailureReason[] = ['auth', 'manifest', 'revoked', 'opt-in']

  it.each(counted)('%s is the theme\'s own failure and counts', (reason) => {
    expect(countsTowardRevert(reason)).toBe(true)
  })

  it.each(ignored)('%s never counts', (reason) => {
    expect(countsTowardRevert(reason)).toBe(false)
  })

  it('reverts on the third failure, not before', () => {
    expect(FAILURES_BEFORE_REVERT).toBe(3)
    expect(shouldRevert(2)).toBe(false)
    expect(shouldRevert(3)).toBe(true)
    expect(shouldRevert(4)).toBe(true)
  })
})

describe('PRD-261 A-04c — the ten-minute window', () => {
  it('counts three failures inside ten minutes', () => {
    const storage = memoryStorage()
    expect(recordThemeFailure(THEME, T0, storage)).toBe(1)
    expect(recordThemeFailure(THEME, T0 + 4 * MINUTE_MS, storage)).toBe(2)
    expect(recordThemeFailure(THEME, T0 + 9 * MINUTE_MS, storage)).toBe(3)
  })

  it('forgets a failure ten minutes old', () => {
    expect(FAILURE_WINDOW_MS).toBe(10 * MINUTE_MS)
    const storage = memoryStorage()
    recordThemeFailure(THEME, T0, storage)
    recordThemeFailure(THEME, T0 + 5 * MINUTE_MS, storage)
    expect(recordThemeFailure(THEME, T0 + 10 * MINUTE_MS, storage)).toBe(2)
    expect(stored(storage)).toEqual({ [THEME]: [T0 + 5 * MINUTE_MS, T0 + 10 * MINUTE_MS] })
  })

  it('three failures spread over more than ten minutes never reach three', () => {
    const storage = memoryStorage()
    const counts = [0, 6, 12, 18].map((minute) => recordThemeFailure(THEME, T0 + minute * MINUTE_MS, storage))
    expect(counts).toEqual([1, 2, 2, 2])
  })

  it('drops a time after now (a clock moved back)', () => {
    const storage = memoryStorage()
    recordThemeFailure(THEME, T0 + 5 * MINUTE_MS, storage)
    expect(recordThemeFailure(THEME, T0, storage)).toBe(1)
  })

  it('keeps each theme apart, and prunes every theme\'s old failures', () => {
    const storage = memoryStorage()
    recordThemeFailure(OTHER_THEME, T0, storage)
    recordThemeFailure(THEME, T0 + MINUTE_MS, storage)
    expect(recordThemeFailure(THEME, T0 + 2 * MINUTE_MS, storage)).toBe(2)
    recordThemeFailure(THEME, T0 + 11 * MINUTE_MS, storage)
    expect(stored(storage)).toEqual({ [THEME]: [T0 + 2 * MINUTE_MS, T0 + 11 * MINUTE_MS] })
  })
})

describe('PRD-261 A-04c — clearing and storage', () => {
  it('clears one theme and keeps the others; the key goes when none is left', () => {
    const storage = memoryStorage()
    recordThemeFailure(THEME, T0, storage)
    recordThemeFailure(OTHER_THEME, T0, storage)
    clearThemeFailures(THEME, storage)
    expect(stored(storage)).toEqual({ [OTHER_THEME]: [T0] })
    clearThemeFailures(OTHER_THEME, storage)
    expect(storage.items.has(FAILURE_LEDGER_KEY)).toBe(false)
  })

  it('reads a value it did not write as an empty ledger, and keeps the well-formed entries', () => {
    const storage = memoryStorage()
    storage.setItem(FAILURE_LEDGER_KEY, 'not json')
    vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    expect(recordThemeFailure(THEME, T0, storage)).toBe(1)
    storage.setItem(FAILURE_LEDGER_KEY, JSON.stringify({ [THEME]: [T0, 'x'], [OTHER_THEME]: [T0] }))
    expect(recordThemeFailure(OTHER_THEME, T0 + MINUTE_MS, storage)).toBe(2)
    expect(stored(storage)).toEqual({ [OTHER_THEME]: [T0, T0 + MINUTE_MS] })
  })

  it('a storage that throws counts each failure alone and never throws', () => {
    vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    expect(recordThemeFailure(THEME, T0, THROWING_STORAGE)).toBe(1)
    expect(recordThemeFailure(THEME, T0 + MINUTE_MS, THROWING_STORAGE)).toBe(1)
    expect(() => clearThemeFailures(THEME, THROWING_STORAGE)).not.toThrow()
  })

  it('with no storage at all, each failure counts alone', () => {
    expect(recordThemeFailure(THEME, T0, null)).toBe(1)
    expect(recordThemeFailure(THEME, T0, null)).toBe(1)
    expect(() => clearThemeFailures(THEME, null)).not.toThrow()
  })

  it('safeLocalStorage is null where reading localStorage throws', () => {
    vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    vi.spyOn(window, 'localStorage', 'get').mockImplementation(blocked)
    expect(safeLocalStorage()).toBeNull()
  })
})

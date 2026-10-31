// @vitest-environment node
/**
 * PRD-261 A-19d1 — ported from lib/looks/__tests__/storage.test.ts to the generic look runtime, on
 * hello-a09b's data: a theme's looks in localStorage, every read through the theme's checks
 * (spec.ts checkLook), every access in try/catch (blocked storage → the theme's default look, nothing
 * kept, and the helpers say so: M98). Keys are scoped per user and workspace (PRD-260 F5); what a
 * browser kept under the unscoped keys moves to the first account once (lib/look-kit/scope.ts
 * moveUnscopedOnce), never deleted, for a key set that names a marker.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { lookScope } from '@/lib/look-kit/scope'
import {
  loadStoredLooks,
  lookKeys,
  lookTheme,
  parseActiveLook,
  parseSavedLooks,
  readPrepaintLook,
  readStoredLooks,
  rememberLookScope,
  writeActiveLook,
  writeSavedLooks,
} from '@/lib/theme-host/looks/storage'

import { blockedStorage, DUSK, fullStorage, HELLO, HELLO_RULES, HELLO_THEME_ID, INK, memoryStorage, myLook, PAPER } from './hello-looks'

const A = lookScope('ws-1', 'user_a')
const B = lookScope('ws-1', 'user_b')
const KEYS = HELLO.keys
const MOVED_MARKER = `${KEYS.saved}.moved-to`
/** hello's looks with a marker: what the unscoped keys hold moves once. */
const MOVING = lookTheme(HELLO_THEME_ID, HELLO_RULES, { ...KEYS, moved: MOVED_MARKER })

beforeEach(() => {
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('the storage keys', () => {
  it("are the theme's, and carry the workspace and the user (F5)", () => {
    expect(lookKeys(A, KEYS)).toEqual({
      saved: `automatos-theme.${HELLO_THEME_ID}.looks:ws-1:user_a`,
      active: `automatos-theme.${HELLO_THEME_ID}.look:ws-1:user_a`,
    })
    expect(lookKeys(lookScope('ws-local', null), KEYS).saved).toBe(`automatos-theme.${HELLO_THEME_ID}.looks:ws-local:local`)
  })
})

describe('parseActiveLook', () => {
  it('reads a stored look, preset or saved', () => {
    expect(parseActiveLook(HELLO_RULES, JSON.stringify(DUSK))).toEqual(DUSK)
    expect(parseActiveLook(HELLO_RULES, JSON.stringify(myLook()))).toEqual(myLook())
  })

  it("gives null for nothing, broken JSON or a look outside the theme's knobs, with a warning", () => {
    expect(parseActiveLook(HELLO_RULES, null)).toBeNull()
    expect(console.warn).not.toHaveBeenCalled()
    expect(parseActiveLook(HELLO_RULES, '{not json')).toBeNull()
    expect(parseActiveLook(HELLO_RULES, JSON.stringify({ ...PAPER, accent: '#FF0000' }))).toBeNull()
    expect(parseActiveLook(HELLO_RULES, JSON.stringify('paper'))).toBeNull()
    expect(console.warn).toHaveBeenCalledTimes(3)
  })
})

describe('parseSavedLooks', () => {
  it('keeps the valid looks in order', () => {
    const second = myLook({ id: 'my-look-2', name: 'My look 2' })
    expect(parseSavedLooks(HELLO_RULES, JSON.stringify([myLook(), second]))).toEqual([myLook(), second])
  })

  it('skips invalid entries, preset ids and repeated ids', () => {
    const raw = JSON.stringify([myLook(), { ...myLook(), glow: 99 }, { ...INK }, myLook({ name: 'Again' }), 7])
    expect(parseSavedLooks(HELLO_RULES, raw)).toEqual([myLook()])
    expect(console.warn).toHaveBeenCalledTimes(4)
  })

  it('gives an empty list for nothing, broken JSON or a non-list', () => {
    expect(parseSavedLooks(HELLO_RULES, null)).toEqual([])
    expect(parseSavedLooks(HELLO_RULES, '[oops')).toEqual([])
    expect(parseSavedLooks(HELLO_RULES, JSON.stringify(myLook()))).toEqual([])
  })
})

describe('reading and writing', () => {
  it('round-trips the active look and the saved looks', () => {
    const store = memoryStorage()
    expect(writeActiveLook(() => store, A, KEYS, DUSK)).toBe(true)
    expect(writeSavedLooks(() => store, A, KEYS, [myLook()])).toBe(true)
    expect(readStoredLooks(() => store, A, HELLO)).toEqual({ scope: A, active: DUSK, saved: [myLook()], persisted: true })
  })

  it("keeps each account's looks apart: another user, or the same user in another workspace, reads none of them", () => {
    const store = memoryStorage()
    writeActiveLook(() => store, A, KEYS, DUSK)
    writeSavedLooks(() => store, A, KEYS, [myLook()])
    expect(readStoredLooks(() => store, B, HELLO)).toEqual({ scope: B, active: null, saved: [], persisted: true })
    expect(readStoredLooks(() => store, lookScope('ws-2', 'user_a'), HELLO).saved).toEqual([])
  })

  it('reads nothing, with a warning, when storage is blocked', () => {
    expect(readStoredLooks(blockedStorage, A, HELLO)).toEqual({ scope: A, active: null, saved: [], persisted: false })
    expect(console.warn).toHaveBeenCalledTimes(1)
  })

  it('reads nothing when getItem itself throws', () => {
    const throwing = memoryStorage()
    throwing.getItem = () => {
      throw new DOMException('The operation is insecure.', 'SecurityError')
    }
    expect(readStoredLooks(() => throwing, A, HELLO)).toEqual({ scope: A, active: null, saved: [], persisted: false })
  })

  it('warns instead of throwing when storage refuses a write', () => {
    const full = fullStorage()
    expect(() => writeActiveLook(() => full, A, KEYS, PAPER)).not.toThrow()
    expect(() => writeSavedLooks(blockedStorage, A, KEYS, [])).not.toThrow()
    expect(console.warn).toHaveBeenCalledTimes(2)
  })

  it('says false when storage refuses a write, so the page can tell the user (M98)', () => {
    expect(writeActiveLook(fullStorage, A, KEYS, PAPER)).toBe(false)
    expect(writeSavedLooks(fullStorage, A, KEYS, [myLook()])).toBe(false)
    expect(writeActiveLook(blockedStorage, A, KEYS, PAPER)).toBe(false)
    expect(writeSavedLooks(blockedStorage, A, KEYS, [])).toBe(false)
  })
})

describe('the looks a browser kept under the unscoped keys (move-once, lib/look-kit/scope.ts)', () => {
  function upgradedEntries(): Readonly<Record<string, string>> {
    return { [KEYS.saved]: JSON.stringify([myLook()]), [KEYS.active]: JSON.stringify(myLook()) }
  }

  function upgradedBrowser(): Storage {
    return memoryStorage(upgradedEntries())
  }

  it('move to the first account that reads them, once, and are never deleted', () => {
    const store = upgradedBrowser()
    expect(loadStoredLooks(() => store, A, MOVING)).toEqual({ scope: A, active: myLook(), saved: [myLook()], persisted: true })
    expect(store.getItem(MOVED_MARKER)).toBe(A)
    expect(loadStoredLooks(() => store, B, MOVING)).toEqual({ scope: B, active: null, saved: [], persisted: true })
    expect(store.getItem(KEYS.saved)).toBe(JSON.stringify([myLook()]))
    expect(store.getItem(KEYS.active)).toBe(JSON.stringify(myLook()))
  })

  it('never overwrite looks the account already has', () => {
    const store = upgradedBrowser()
    writeSavedLooks(() => store, A, KEYS, [])
    expect(loadStoredLooks(() => store, A, MOVING)).toMatchObject({ active: myLook(), saved: [] })
  })

  it('leave nothing marked when there is nothing to move', () => {
    const store = memoryStorage()
    expect(loadStoredLooks(() => store, A, MOVING).saved).toEqual([])
    expect(store.getItem(MOVED_MARKER)).toBeNull()
  })

  it("still read the account's looks when the move is refused", () => {
    const full = fullStorage(upgradedEntries())
    expect(loadStoredLooks(() => full, A, MOVING)).toEqual({ scope: A, active: null, saved: [], persisted: true })
    expect(loadStoredLooks(blockedStorage, A, MOVING).persisted).toBe(false)
  })

  it("stay where they are for an installed theme's keys, which name no marker", () => {
    const store = upgradedBrowser()
    expect(KEYS.moved).toBeNull()
    expect(loadStoredLooks(() => store, A, HELLO)).toEqual({ scope: A, active: null, saved: [], persisted: true })
    expect(store.getItem(MOVED_MARKER)).toBeNull()
  })
})

describe('the pre-paint pointer', () => {
  it('names the scope whose active look the next page load paints', () => {
    const store = memoryStorage()
    expect(rememberLookScope(() => store, A, KEYS)).toBe(true)
    expect(store.getItem(KEYS.pointer)).toBe(A)
    expect(rememberLookScope(blockedStorage, A, KEYS)).toBe(false)
  })
})

describe('the pre-paint look (painted until the account is known)', () => {
  it("is the pointer scope's active look, else the unscoped one, else the theme's default", () => {
    const store = memoryStorage({ [KEYS.active]: JSON.stringify(myLook()), [lookKeys(A, KEYS).active]: JSON.stringify(DUSK) })
    expect(readPrepaintLook(() => store, HELLO)).toEqual(myLook())
    store.setItem(KEYS.pointer, A)
    expect(readPrepaintLook(() => store, HELLO)).toEqual(DUSK)
    store.setItem(KEYS.pointer, B)
    expect(readPrepaintLook(() => store, HELLO)).toBe(PAPER)
    expect(readPrepaintLook(() => memoryStorage(), HELLO)).toBe(PAPER)
    expect(readPrepaintLook(() => memoryStorage({ [KEYS.active]: '{not json' }), HELLO)).toBe(PAPER)
    expect(readPrepaintLook(blockedStorage, HELLO)).toBe(PAPER)
  })

  it('only reads: nothing moves and nothing is written', () => {
    const store = memoryStorage({ [KEYS.saved]: JSON.stringify([myLook()]), [KEYS.active]: JSON.stringify(myLook()) })
    const setItem = vi.spyOn(store, 'setItem')
    readPrepaintLook(() => store, MOVING)
    expect(setItem).not.toHaveBeenCalled()
    expect(store.getItem(MOVED_MARKER)).toBeNull()
  })
})

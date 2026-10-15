// @vitest-environment node
/**
 * PRD-260 — looks in localStorage: every read through validateLook, every
 * access in try/catch (blocked storage → the default look, nothing kept, and
 * the helpers say so: M98). Keys are scoped per user and workspace (F5); what
 * a browser kept before moves to the first account once, never deleted.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { DAYLIGHT, EMBER } from '@/lib/looks/presets'
import { lookScope } from '@/lib/looks/scope'
import {
  ACTIVE_LOOK_KEY,
  LOOK_SCOPE_KEY,
  SAVED_LOOKS_KEY,
  UNSCOPED_LOOKS_MOVED_KEY,
  loadStoredLooks,
  lookKeys,
  parseActiveLook,
  parseSavedLooks,
  readStoredLooks,
  rememberLookScope,
  writeActiveLook,
  writeSavedLooks,
} from '@/lib/looks/storage'

import { blockedStorage, memoryStorage, myLook } from './helpers'

const A = lookScope('ws-1', 'user_a')
const B = lookScope('ws-1', 'user_b')

function fullStorage(): Storage {
  const full = memoryStorage()
  full.setItem = () => {
    throw new DOMException('The quota has been exceeded.', 'QuotaExceededError')
  }
  return full
}

beforeEach(() => {
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('the storage keys', () => {
  it('are the ones in the PRD', () => {
    expect(SAVED_LOOKS_KEY).toBe('automatos-play.looks')
    expect(ACTIVE_LOOK_KEY).toBe('automatos-play.look')
  })

  it('carry the workspace and the user (F5)', () => {
    expect(lookKeys(A)).toEqual({ saved: 'automatos-play.looks:ws-1:user_a', active: 'automatos-play.look:ws-1:user_a' })
    expect(lookKeys(lookScope('ws-local', null)).saved).toBe('automatos-play.looks:ws-local:local')
  })
})

describe('parseActiveLook', () => {
  it('reads a stored look, preset or saved', () => {
    expect(parseActiveLook(JSON.stringify(DAYLIGHT))).toEqual(DAYLIGHT)
    expect(parseActiveLook(JSON.stringify(myLook()))).toEqual(myLook())
  })

  it('gives null for nothing, broken JSON or a look outside the catalogue, with a warning', () => {
    expect(parseActiveLook(null)).toBeNull()
    expect(console.warn).not.toHaveBeenCalled()
    expect(parseActiveLook('{not json')).toBeNull()
    expect(parseActiveLook(JSON.stringify({ ...EMBER, accent: '#FF0000' }))).toBeNull()
    expect(parseActiveLook(JSON.stringify('ember'))).toBeNull()
    expect(console.warn).toHaveBeenCalledTimes(3)
  })
})

describe('parseSavedLooks', () => {
  it('keeps the valid looks in order', () => {
    const second = myLook({ id: 'my-look-2', name: 'My look 2' })
    expect(parseSavedLooks(JSON.stringify([myLook(), second]))).toEqual([myLook(), second])
  })

  it('skips invalid entries, preset ids and repeated ids', () => {
    const raw = JSON.stringify([myLook(), { ...myLook(), glow: 99 }, { ...DAYLIGHT }, myLook({ name: 'Again' }), 7])
    expect(parseSavedLooks(raw)).toEqual([myLook()])
    expect(console.warn).toHaveBeenCalledTimes(4)
  })

  it('gives an empty list for nothing, broken JSON or a non-list', () => {
    expect(parseSavedLooks(null)).toEqual([])
    expect(parseSavedLooks('[oops')).toEqual([])
    expect(parseSavedLooks(JSON.stringify(myLook()))).toEqual([])
  })
})

describe('reading and writing', () => {
  it('round-trips the active look and the saved looks', () => {
    const store = memoryStorage()
    expect(writeActiveLook(() => store, A, DAYLIGHT)).toBe(true)
    expect(writeSavedLooks(() => store, A, [myLook()])).toBe(true)
    expect(readStoredLooks(() => store, A)).toEqual({ scope: A, active: DAYLIGHT, saved: [myLook()], persisted: true })
  })

  it("keeps each account's looks apart: another user in the same browser reads none of them", () => {
    const store = memoryStorage()
    writeActiveLook(() => store, A, DAYLIGHT)
    writeSavedLooks(() => store, A, [myLook()])
    expect(readStoredLooks(() => store, B)).toEqual({ scope: B, active: null, saved: [], persisted: true })
    expect(readStoredLooks(() => store, lookScope('ws-2', 'user_a')).saved).toEqual([])
  })

  it('reads nothing, with a warning, when storage is blocked', () => {
    expect(readStoredLooks(blockedStorage, A)).toEqual({ scope: A, active: null, saved: [], persisted: false })
    expect(console.warn).toHaveBeenCalledTimes(1)
  })

  it('reads nothing when getItem itself throws', () => {
    const throwing = memoryStorage()
    throwing.getItem = () => {
      throw new DOMException('The operation is insecure.', 'SecurityError')
    }
    expect(readStoredLooks(() => throwing, A)).toEqual({ scope: A, active: null, saved: [], persisted: false })
  })

  it('warns instead of throwing when storage refuses a write', () => {
    const full = fullStorage()
    expect(() => writeActiveLook(() => full, A, EMBER)).not.toThrow()
    expect(() => writeSavedLooks(blockedStorage, A, [])).not.toThrow()
    expect(console.warn).toHaveBeenCalledTimes(2)
  })

  it('says false when storage refuses a write, so the page can tell the user (M98)', () => {
    expect(writeActiveLook(fullStorage, A, EMBER)).toBe(false)
    expect(writeSavedLooks(fullStorage, A, [myLook()])).toBe(false)
    expect(writeActiveLook(blockedStorage, A, EMBER)).toBe(false)
    expect(writeSavedLooks(blockedStorage, A, [])).toBe(false)
  })
})

describe('the looks a browser kept before F5', () => {
  function upgradedBrowser(): Storage {
    return memoryStorage({
      [SAVED_LOOKS_KEY]: JSON.stringify([myLook()]),
      [ACTIVE_LOOK_KEY]: JSON.stringify(myLook()),
    })
  }

  it('move to the first account that reads them, once, and are never deleted', () => {
    const store = upgradedBrowser()
    expect(loadStoredLooks(() => store, A)).toEqual({ scope: A, active: myLook(), saved: [myLook()], persisted: true })
    expect(store.getItem(UNSCOPED_LOOKS_MOVED_KEY)).toBe(A)
    expect(loadStoredLooks(() => store, B)).toEqual({ scope: B, active: null, saved: [], persisted: true })
    expect(store.getItem(SAVED_LOOKS_KEY)).toBe(JSON.stringify([myLook()]))
    expect(store.getItem(ACTIVE_LOOK_KEY)).toBe(JSON.stringify(myLook()))
  })

  it("never overwrite looks the account already has", () => {
    const store = upgradedBrowser()
    writeSavedLooks(() => store, A, [])
    expect(loadStoredLooks(() => store, A)).toMatchObject({ active: myLook(), saved: [] })
  })

  it('leave nothing marked when there is nothing to move', () => {
    const store = memoryStorage()
    expect(loadStoredLooks(() => store, A).saved).toEqual([])
    expect(store.getItem(UNSCOPED_LOOKS_MOVED_KEY)).toBeNull()
  })

  it("still read the account's looks when the move is refused", () => {
    const full = upgradedBrowser()
    full.setItem = () => {
      throw new DOMException('The quota has been exceeded.', 'QuotaExceededError')
    }
    expect(loadStoredLooks(() => full, A)).toEqual({ scope: A, active: null, saved: [], persisted: true })
    expect(loadStoredLooks(blockedStorage, A).persisted).toBe(false)
  })
})

describe('the pre-paint pointer', () => {
  it("names the scope whose active look the next page load paints", () => {
    const store = memoryStorage()
    expect(rememberLookScope(() => store, A)).toBe(true)
    expect(store.getItem(LOOK_SCOPE_KEY)).toBe(A)
    expect(rememberLookScope(blockedStorage, A)).toBe(false)
  })
})

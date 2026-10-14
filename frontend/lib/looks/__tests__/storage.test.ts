// @vitest-environment node
/**
 * PRD-260 — looks in localStorage: every read through validateLook, every
 * access in try/catch (blocked storage → the default look, nothing kept).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { DAYLIGHT, EMBER } from '@/lib/looks/presets'
import {
  ACTIVE_LOOK_KEY,
  SAVED_LOOKS_KEY,
  parseActiveLook,
  parseSavedLooks,
  readStoredLooks,
  writeActiveLook,
  writeSavedLooks,
} from '@/lib/looks/storage'

import { blockedStorage, memoryStorage, myLook } from './helpers'

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
    writeActiveLook(() => store, DAYLIGHT)
    writeSavedLooks(() => store, [myLook()])
    expect(readStoredLooks(() => store)).toEqual({ active: DAYLIGHT, saved: [myLook()] })
  })

  it('reads nothing, with a warning, when storage is blocked', () => {
    expect(readStoredLooks(blockedStorage)).toEqual({ active: null, saved: [] })
    expect(console.warn).toHaveBeenCalledTimes(1)
  })

  it('reads nothing when getItem itself throws', () => {
    const throwing = memoryStorage()
    throwing.getItem = () => {
      throw new DOMException('The operation is insecure.', 'SecurityError')
    }
    expect(readStoredLooks(() => throwing)).toEqual({ active: null, saved: [] })
  })

  it('warns instead of throwing when storage refuses a write', () => {
    const full = memoryStorage()
    full.setItem = () => {
      throw new DOMException('The quota has been exceeded.', 'QuotaExceededError')
    }
    expect(() => writeActiveLook(() => full, EMBER)).not.toThrow()
    expect(() => writeSavedLooks(blockedStorage, [])).not.toThrow()
    expect(console.warn).toHaveBeenCalledTimes(2)
  })
})

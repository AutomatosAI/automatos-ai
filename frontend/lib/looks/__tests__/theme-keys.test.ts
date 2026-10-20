// @vitest-environment node
/**
 * PRD-261 A-09b — the look storage is theme-neutral: an installed theme keeps its looks under
 * `automatos-theme.<themeId>.*`, builtin Play under its PRD-260 `automatos-play.*` keys (so no
 * look moves before A-18), and the boot paints from the same keys the store writes. The server's
 * answer goes through the same checks as a stored one.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { DAYLIGHT, EMBER } from '@/lib/looks/presets'
import { lookScope } from '@/lib/looks/scope'
import {
  ACTIVE_LOOK_KEY,
  LOOK_SCOPE_KEY,
  SAVED_LOOKS_KEY,
  UNSCOPED_LOOKS_MOVED_KEY,
  activeLookOf,
  bootLookStorage,
  hasLookMark,
  installedLookKeys,
  installedLookTheme,
  loadStoredLooks,
  lookKeys,
  lookMarkKey,
  readPrepaintLook,
  readStoredLooks,
  rememberLookScope,
  savedLooksOf,
  setLookMark,
  writeActiveLook,
  writeSavedLooks,
} from '@/lib/theme-host/builtin/play-look-storage'
import { BUILTIN_PLAY_BOOT } from '@/lib/theme-host/builtin/play-boot'
import {
  BUILTIN_PLAY_LOOK_KEYS,
  BUILTIN_PLAY_LOOK_THEME,
  BUILTIN_PLAY_THEME_ID,
} from '@/lib/theme-host/builtin/play-look-keys'

import { blockedStorage, memoryStorage, myLook } from './helpers'

const SCOPE = lookScope('7d3c2a10-1b2c-4d5e-8f90-123456789abc', 'user_a')
const HELLO = installedLookKeys('acme/hello')

beforeEach(() => {
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe("builtin Play's keys", () => {
  it('are its PRD-260 keys, unchanged, and its server id is its package id', () => {
    expect(BUILTIN_PLAY_LOOK_KEYS).toEqual({
      saved: 'automatos-play.looks',
      active: 'automatos-play.look',
      pointer: 'automatos-play.look.current',
      moved: 'automatos-play.looks.moved-to',
    })
    expect(BUILTIN_PLAY_THEME_ID).toBe('automatos/play')
    expect(BUILTIN_PLAY_LOOK_THEME).toEqual({ themeId: 'automatos/play', keys: BUILTIN_PLAY_LOOK_KEYS })
  })

  it("are the storage module's defaults and its PRD-260 names", () => {
    expect([SAVED_LOOKS_KEY, ACTIVE_LOOK_KEY, LOOK_SCOPE_KEY, UNSCOPED_LOOKS_MOVED_KEY]).toEqual([
      'automatos-play.looks',
      'automatos-play.look',
      'automatos-play.look.current',
      'automatos-play.looks.moved-to',
    ])
    expect(lookKeys(SCOPE)).toEqual(lookKeys(SCOPE, BUILTIN_PLAY_LOOK_KEYS))
    expect(lookKeys(SCOPE).active).toBe(`automatos-play.look:${SCOPE}`)
  })

  it("are the keys builtin Play's boot paints from", () => {
    expect(BUILTIN_PLAY_BOOT.lookStorage).toEqual({
      pointerKey: 'automatos-play.look.current',
      scopedPrefix: 'automatos-play.look:',
      fallbackKey: 'automatos-play.look',
    })
    expect(BUILTIN_PLAY_BOOT.lookStorage).toEqual(bootLookStorage(BUILTIN_PLAY_LOOK_KEYS))
  })
})

describe("an installed theme's keys", () => {
  it('are automatos-theme.<themeId>.looks, .look and .look.current, and nothing moves', () => {
    expect(HELLO).toEqual({
      saved: 'automatos-theme.acme/hello.looks',
      active: 'automatos-theme.acme/hello.look',
      pointer: 'automatos-theme.acme/hello.look.current',
      moved: null,
    })
    expect(installedLookTheme('acme/hello')).toEqual({ themeId: 'acme/hello', keys: HELLO })
    expect(lookKeys(SCOPE, HELLO)).toEqual({
      saved: `automatos-theme.acme/hello.looks:${SCOPE}`,
      active: `automatos-theme.acme/hello.look:${SCOPE}`,
    })
    expect(bootLookStorage(HELLO)).toEqual({
      pointerKey: 'automatos-theme.acme/hello.look.current',
      scopedPrefix: 'automatos-theme.acme/hello.look:',
      fallbackKey: 'automatos-theme.acme/hello.look',
    })
  })

  it("keep that theme's looks apart from Play's in the same browser and scope", () => {
    const store = memoryStorage()
    writeActiveLook(() => store, SCOPE, DAYLIGHT, HELLO)
    writeSavedLooks(() => store, SCOPE, [myLook()], HELLO)
    expect(readStoredLooks(() => store, SCOPE, HELLO)).toEqual({ scope: SCOPE, active: DAYLIGHT, saved: [myLook()], persisted: true })
    expect(readStoredLooks(() => store, SCOPE)).toEqual({ scope: SCOPE, active: null, saved: [], persisted: true })
    expect(store.getItem(lookKeys(SCOPE).active)).toBeNull()
  })

  it("never take Play's pre-F5 looks: the first read moves nothing", () => {
    const store = memoryStorage({ [SAVED_LOOKS_KEY]: JSON.stringify([myLook()]), [ACTIVE_LOOK_KEY]: JSON.stringify(myLook()) })
    expect(loadStoredLooks(() => store, SCOPE, HELLO)).toEqual({ scope: SCOPE, active: null, saved: [], persisted: true })
    expect(store.getItem(UNSCOPED_LOOKS_MOVED_KEY)).toBeNull()
  })

  it('have their own pre-paint pointer', () => {
    const store = memoryStorage()
    writeActiveLook(() => store, SCOPE, DAYLIGHT, HELLO)
    expect(rememberLookScope(() => store, SCOPE, HELLO)).toBe(true)
    expect(store.getItem(HELLO.pointer)).toBe(SCOPE)
    expect(store.getItem(LOOK_SCOPE_KEY)).toBeNull()
    expect(readPrepaintLook(() => store, HELLO)).toEqual(DAYLIGHT)
    expect(readPrepaintLook(() => store)).toEqual(EMBER)
  })
})

describe('the server marks', () => {
  it('sit beside the saved looks, per scope and theme, and say false when storage is blocked', () => {
    const store = memoryStorage()
    expect(lookMarkKey(SCOPE, BUILTIN_PLAY_LOOK_KEYS, 'pending')).toBe(`automatos-play.looks.pending:${SCOPE}`)
    expect(lookMarkKey(SCOPE, HELLO, 'synced')).toBe(`automatos-theme.acme/hello.looks.synced:${SCOPE}`)
    expect(setLookMark(() => store, SCOPE, HELLO, 'pending', true)).toBe(true)
    expect(hasLookMark(() => store, SCOPE, HELLO, 'pending')).toBe(true)
    expect(hasLookMark(() => store, SCOPE, BUILTIN_PLAY_LOOK_KEYS, 'pending')).toBe(false)
    expect(setLookMark(() => store, SCOPE, HELLO, 'pending', false)).toBe(true)
    expect(hasLookMark(() => store, SCOPE, HELLO, 'pending')).toBe(false)
    expect(setLookMark(blockedStorage, SCOPE, HELLO, 'synced', true)).toBe(false)
    expect(hasLookMark(blockedStorage, SCOPE, HELLO, 'synced')).toBe(false)
  })
})

describe("the server's looks", () => {
  it('go through the same checks as stored ones', () => {
    expect(activeLookOf(DAYLIGHT, 'the server')).toEqual(DAYLIGHT)
    expect(activeLookOf({ ...EMBER, accent: '#FF0000' }, 'the server')).toBeNull()
    expect(savedLooksOf([myLook(), { ...myLook(), glow: 99 }, DAYLIGHT], 'the server')).toEqual([myLook()])
    expect(savedLooksOf({ looks: [] }, 'the server')).toEqual([])
    expect(console.warn).toHaveBeenCalledTimes(4)
  })
})

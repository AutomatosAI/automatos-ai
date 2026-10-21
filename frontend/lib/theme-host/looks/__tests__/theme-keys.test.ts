// @vitest-environment node
/**
 * PRD-261 A-19d1 — ported from lib/looks/__tests__/theme-keys.test.ts to the generic look runtime:
 * a theme keeps its looks under `automatos-theme.<themeId>.*`, per account and workspace (PRD-260
 * F5), the keys the backend's boot names for the boot script (orchestrator serve.py look_storage;
 * both sides held to one shared fixture), so the boot paints from the keys the store writes. Two
 * themes in one browser and scope keep their looks apart, and the server's answer goes through the
 * same checks as a stored one.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { lookScope } from '@/lib/look-kit/scope'
import { servedLooksOf } from '@/lib/theme-host/looks/server'
import {
  activeLookOf,
  bootLookStorage,
  hasLookMark,
  INSTALLED_LOOK_KEY_PREFIX,
  loadStoredLooks,
  lookKeys,
  lookMarkKey,
  lookTheme,
  readPrepaintLook,
  readStoredLooks,
  rememberLookScope,
  savedLooksOf,
  setLookMark,
  themeLookKeys,
  writeActiveLook,
  writeSavedLooks,
} from '@/lib/theme-host/looks/storage'

import { blockedStorage, DUSK, HELLO, HELLO_RULES, HELLO_THEME_ID, INK, memoryStorage, myLook, PAPER, sharedLookStorage } from './hello-looks'

const SCOPE = lookScope('7d3c2a10-1b2c-4d5e-8f90-123456789abc', 'user_a')
const OTHER_ID = 'acme/hello'
/** Another installed theme with the same look rules: only its id, so its keys, differ. */
const OTHER = lookTheme(OTHER_ID, HELLO_RULES)

beforeEach(() => {
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe("an installed theme's keys", () => {
  it("are the ones the backend's boot names for the boot script (serve.py look_storage)", () => {
    const shared = sharedLookStorage()
    expect(bootLookStorage(themeLookKeys(shared.themeId))).toEqual(shared.lookStorage)
  })

  it('are automatos-theme.<themeId>.looks, .look and .look.current, and nothing moves', () => {
    expect(INSTALLED_LOOK_KEY_PREFIX).toBe('automatos-theme.')
    expect(themeLookKeys(OTHER_ID)).toEqual({
      saved: 'automatos-theme.acme/hello.looks',
      active: 'automatos-theme.acme/hello.look',
      pointer: 'automatos-theme.acme/hello.look.current',
      moved: null,
    })
    expect(OTHER).toEqual({ themeId: OTHER_ID, keys: themeLookKeys(OTHER_ID), rules: HELLO_RULES })
    expect(HELLO.keys).toEqual(themeLookKeys(HELLO_THEME_ID))
    expect(lookKeys(SCOPE, OTHER.keys)).toEqual({
      saved: `automatos-theme.acme/hello.looks:${SCOPE}`,
      active: `automatos-theme.acme/hello.look:${SCOPE}`,
    })
    expect(bootLookStorage(OTHER.keys)).toEqual({
      pointerKey: 'automatos-theme.acme/hello.look.current',
      scopedPrefix: 'automatos-theme.acme/hello.look:',
      fallbackKey: 'automatos-theme.acme/hello.look',
    })
  })

  it('carry the account and the workspace: each pair of them reads only its own looks (F5)', () => {
    const store = memoryStorage()
    const otherUser = lookScope('7d3c2a10-1b2c-4d5e-8f90-123456789abc', 'user_b')
    const otherWorkspace = lookScope('0b1c2d3e-4f50-4617-8899-aabbccddeeff', 'user_a')
    writeActiveLook(() => store, SCOPE, HELLO.keys, DUSK)
    expect(new Set([SCOPE, otherUser, otherWorkspace].map((scope) => lookKeys(scope, HELLO.keys).active)).size).toBe(3)
    expect(readStoredLooks(() => store, otherUser, HELLO).active).toBeNull()
    expect(readStoredLooks(() => store, otherWorkspace, HELLO).active).toBeNull()
    expect(readStoredLooks(() => store, SCOPE, HELLO).active).toEqual(DUSK)
  })

  it("keep one theme's looks apart from another's in the same browser and scope", () => {
    const store = memoryStorage()
    writeActiveLook(() => store, SCOPE, OTHER.keys, DUSK)
    writeSavedLooks(() => store, SCOPE, OTHER.keys, [myLook()])
    expect(readStoredLooks(() => store, SCOPE, OTHER)).toEqual({ scope: SCOPE, active: DUSK, saved: [myLook()], persisted: true })
    expect(readStoredLooks(() => store, SCOPE, HELLO)).toEqual({ scope: SCOPE, active: null, saved: [], persisted: true })
    expect(store.getItem(lookKeys(SCOPE, HELLO.keys).active)).toBeNull()
  })

  it("never take another theme's unscoped looks: the first read moves nothing", () => {
    const store = memoryStorage({ [HELLO.keys.saved]: JSON.stringify([myLook()]), [HELLO.keys.active]: JSON.stringify(myLook()) })
    expect(loadStoredLooks(() => store, SCOPE, OTHER)).toEqual({ scope: SCOPE, active: null, saved: [], persisted: true })
    expect(loadStoredLooks(() => store, SCOPE, HELLO)).toEqual({ scope: SCOPE, active: null, saved: [], persisted: true })
  })

  it('have their own pre-paint pointer', () => {
    const store = memoryStorage()
    writeActiveLook(() => store, SCOPE, OTHER.keys, INK)
    expect(rememberLookScope(() => store, SCOPE, OTHER.keys)).toBe(true)
    expect(store.getItem(OTHER.keys.pointer)).toBe(SCOPE)
    expect(store.getItem(HELLO.keys.pointer)).toBeNull()
    expect(readPrepaintLook(() => store, OTHER)).toEqual(INK)
    expect(readPrepaintLook(() => store, HELLO)).toBe(PAPER)
  })
})

describe('the server marks', () => {
  it('sit beside the saved looks, per scope and theme, and say false when storage is blocked', () => {
    const store = memoryStorage()
    expect(lookMarkKey(SCOPE, HELLO.keys, 'pending')).toBe(`automatos-theme.${HELLO_THEME_ID}.looks.pending:${SCOPE}`)
    expect(lookMarkKey(SCOPE, OTHER.keys, 'synced')).toBe(`automatos-theme.acme/hello.looks.synced:${SCOPE}`)
    expect(setLookMark(() => store, SCOPE, OTHER.keys, 'pending', true)).toBe(true)
    expect(hasLookMark(() => store, SCOPE, OTHER.keys, 'pending')).toBe(true)
    expect(hasLookMark(() => store, SCOPE, HELLO.keys, 'pending')).toBe(false)
    expect(setLookMark(() => store, SCOPE, OTHER.keys, 'pending', false)).toBe(true)
    expect(hasLookMark(() => store, SCOPE, OTHER.keys, 'pending')).toBe(false)
    expect(setLookMark(blockedStorage, SCOPE, OTHER.keys, 'synced', true)).toBe(false)
    expect(hasLookMark(blockedStorage, SCOPE, OTHER.keys, 'synced')).toBe(false)
  })
})

describe("the server's looks", () => {
  it('go through the same checks as stored ones', () => {
    expect(activeLookOf(HELLO_RULES, DUSK, 'the server')).toEqual(DUSK)
    expect(activeLookOf(HELLO_RULES, { ...PAPER, accent: '#FF0000' }, 'the server')).toBeNull()
    expect(savedLooksOf(HELLO_RULES, [myLook(), { ...myLook(), glow: 99 }, DUSK], 'the server')).toEqual([myLook()])
    expect(savedLooksOf(HELLO_RULES, { looks: [] }, 'the server')).toEqual([])
    expect(console.warn).toHaveBeenCalledTimes(4)
  })

  it("are read from the looks answer only, and say when the server holds none", () => {
    const answer = { themeId: HELLO_THEME_ID, looks: [myLook()], active: DUSK, defaultLook: PAPER, dropped: 0 }
    expect(servedLooksOf(HELLO_RULES, answer)).toEqual({ active: DUSK, saved: [myLook()], empty: false })
    expect(servedLooksOf(HELLO_RULES, { ...answer, looks: [], active: null })).toEqual({ active: null, saved: [], empty: true })
    expect(servedLooksOf(HELLO_RULES, { detail: 'Something went wrong.' })).toBeNull()
    expect(servedLooksOf(HELLO_RULES, null)).toBeNull()
  })
})

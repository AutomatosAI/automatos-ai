// @vitest-environment node
/**
 * PRD-261 A-19d1 — one-time, while the old look runtime is still in core (it leaves with W-A8, and
 * A-19f2 deletes this file with it): the generic runtime keeps an installed theme's looks under
 * exactly the keys the old runtime's installedLookKeys names, per account and workspace (F5), with
 * the same server marks and the same boot keys. So the looks a member already has under an installed
 * theme are the ones the generic runtime reads when A-19d2 switches installed themes onto it: nothing
 * moves. theme-keys.test.ts holds the same keys to their literal names and to the backend's fixture.
 */
import { describe, expect, it } from 'vitest'

import { lookScope } from '@/lib/look-kit/scope'
import * as oldRuntime from '@/lib/theme-host/builtin/play-look-storage'
import { bootLookStorage, lookKeys, lookMarkKey, themeLookKeys } from '@/lib/theme-host/looks/storage'

import { HELLO_THEME_ID } from './hello-looks'

const THEME_IDS: readonly string[] = [HELLO_THEME_ID, 'acme/hello', 'a/b']
const SCOPES: readonly string[] = [
  lookScope('7d3c2a10-1b2c-4d5e-8f90-123456789abc', 'user_a'),
  lookScope('7d3c2a10-1b2c-4d5e-8f90-123456789abc', 'user_b'),
  lookScope('0b1c2d3e-4f50-4617-8899-aabbccddeeff', 'user_a'),
  lookScope(null, null),
]

describe("the generic runtime's keys equal the old runtime's installedLookKeys", () => {
  it.each(THEME_IDS)('for %s: the bases, the boot keys, and every scoped key and mark', (themeId) => {
    const keys = themeLookKeys(themeId)
    const old = oldRuntime.installedLookKeys(themeId)
    expect(keys).toEqual(old)
    expect(bootLookStorage(keys)).toEqual(oldRuntime.bootLookStorage(old))
    SCOPES.forEach((scope) => {
      expect(lookKeys(scope, keys)).toEqual(oldRuntime.lookKeys(scope, old))
      expect(lookMarkKey(scope, keys, 'pending')).toBe(oldRuntime.lookMarkKey(scope, old, 'pending'))
      expect(lookMarkKey(scope, keys, 'synced')).toBe(oldRuntime.lookMarkKey(scope, old, 'synced'))
    })
  })
})

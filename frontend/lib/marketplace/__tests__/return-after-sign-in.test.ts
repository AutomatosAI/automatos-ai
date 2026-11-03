/**
 * PRD-270 US-270-09a (§5.7, K21) — the way back to a Marketplace item after sign-in.
 *
 * - only '/marketplace' with its tab, item and add query is ever kept, rebuilt from those three:
 *   another site, '//', another path, a scheme, a backslash, a fragment, an overlong value or any
 *   other parameter is dropped (no open redirect);
 * - the sign-in and sign-up links carry it encoded as `redirect_url`, and an unsafe one becomes
 *   '/marketplace';
 * - it is remembered in this tab for 30 minutes and taken once; an expired, tampered or malformed
 *   store gives nothing and is forgotten; storage errors never throw.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  MARKETPLACE_PATH,
  RETURN_PATH_STORAGE_KEY,
  RETURN_PATH_TTL_MS,
  type ReturnPathStorage,
  marketplaceItemPath,
  remember,
  safeReturnPath,
  signInUrl,
  signUpUrl,
  take,
} from '../return-after-sign-in'

const ARCADE = '/marketplace?tab=themes&item=automatos.arcade&add=1'
const NOW = 1_800_000_000_000
const store = () => window.sessionStorage
const stored = () => store().getItem(RETURN_PATH_STORAGE_KEY)

beforeEach(() => {
  store().clear()
})

afterEach(() => {
  store().clear()
  vi.restoreAllMocks()
})

describe('safeReturnPath', () => {
  it.each([
    [ARCADE, ARCADE],
    [MARKETPLACE_PATH, MARKETPLACE_PATH],
    ['/marketplace?', MARKETPLACE_PATH],
    ['/marketplace?tab=agents', '/marketplace?tab=agents'],
    ['/marketplace?add=1&item=automatos.arcade&tab=themes', ARCADE],
    [`${ARCADE}&next=https://evil.example&redirect_url=//evil.example`, ARCADE],
    ['/marketplace?tab=themes&add=yes', '/marketplace?tab=themes'],
    ['/marketplace?tab=skills&item=pdf%20tools', '/marketplace?tab=skills&item=pdf+tools'],
    ['/marketplace?@evil.example', MARKETPLACE_PATH],
    [`/marketplace?tab=agents&item=${'w'.repeat(256)}`, `/marketplace?tab=agents&item=${'w'.repeat(256)}`],
    [`/marketplace?tab=agents&item=${'w'.repeat(257)}&add=1`, '/marketplace?tab=agents&add=1'],
    [`/marketplace?tab=${'t'.repeat(257)}&item=writer`, '/marketplace?item=writer'],
  ])('keeps %s as %s', (given, kept) => {
    expect(safeReturnPath(given)).toBe(kept)
  })

  it.each([
    'https://evil.example',
    'https://evil.example/marketplace?tab=themes',
    'http:/marketplace',
    '//evil.example',
    '//evil.example/marketplace',
    '/\\evil.example',
    '/chat',
    '/',
    '/marketplace/publish',
    '/marketplace/developer?tab=themes',
    '/marketplace/',
    '/marketplace\\x',
    '/marketplace\\\\x',
    '/marketplace?tab=themes\\x',
    '/marketplace%2F..%2Fchat',
    '/marketplace#top',
    '/marketplace?tab=themes#item',
    '/marketplace?tab=the mes',
    '/marketplace?tab=themes\n',
    '/marketplaces',
    '/marketplace.evil.example',
    ' /marketplace',
    'marketplace',
    'javascript:alert(1)',
    'javascript:/marketplace',
    'data:text/html,hi',
    '',
  ])('drops %s', (given) => {
    expect(safeReturnPath(given)).toBeNull()
  })

  it('drops what is not a string', () => {
    for (const value of [null, undefined, 42, {}, [ARCADE]]) {
      expect(safeReturnPath(value)).toBeNull()
    }
  })
})

describe('the links', () => {
  it('opens the item on its tab with the add dialog ready', () => {
    expect(marketplaceItemPath('themes', 'automatos.arcade')).toBe(ARCADE)
    expect(marketplaceItemPath('recipes', 'weekly report')).toBe('/marketplace?tab=recipes&item=weekly+report&add=1')
  })

  it('sign-in and sign-up carry the return path, encoded, as redirect_url', () => {
    expect(signInUrl(ARCADE)).toBe(`/sign-in?redirect_url=${encodeURIComponent(ARCADE)}`)
    expect(signUpUrl(ARCADE)).toBe(`/sign-up?redirect_url=${encodeURIComponent(ARCADE)}`)
  })

  it('an unsafe return path rides as /marketplace, never as given', () => {
    expect(signInUrl('https://evil.example')).toBe('/sign-in?redirect_url=%2Fmarketplace')
    expect(signUpUrl('//evil.example')).toBe('/sign-up?redirect_url=%2Fmarketplace')
    expect(signInUrl(`${ARCADE}&next=//evil.example`)).toBe(`/sign-in?redirect_url=${encodeURIComponent(ARCADE)}`)
  })
})

describe('remember and take', () => {
  it('takes the remembered path once, then forgets it', () => {
    remember(ARCADE, store(), NOW)
    expect(take(store(), NOW + 1)).toBe(ARCADE)
    expect(stored()).toBeNull()
    expect(take(store(), NOW + 2)).toBeNull()
  })

  it("uses this tab's sessionStorage and the clock by default", () => {
    remember(ARCADE)
    expect(stored()).not.toBeNull()
    expect(take()).toBe(ARCADE)
    expect(take()).toBeNull()
  })

  it('stores the rebuilt path and its 30-minute expiry, nothing else', () => {
    remember(`${ARCADE}&next=//evil.example`, store(), NOW)
    expect(JSON.parse(stored() ?? 'null')).toEqual({ path: ARCADE, expiresAt: NOW + RETURN_PATH_TTL_MS })
  })

  it('keeps the path for 30 minutes and gives nothing after, forgetting it', () => {
    remember(ARCADE, store(), NOW)
    expect(take(store(), NOW + RETURN_PATH_TTL_MS - 1)).toBe(ARCADE)

    remember(ARCADE, store(), NOW)
    expect(take(store(), NOW + RETURN_PATH_TTL_MS)).toBeNull()
    expect(stored()).toBeNull()
  })

  it('a later path replaces the earlier one', () => {
    remember(ARCADE, store(), NOW)
    remember('/marketplace?tab=agents&item=writer&add=1', store(), NOW + 5)
    expect(take(store(), NOW + 10)).toBe('/marketplace?tab=agents&item=writer&add=1')
  })

  it('never remembers an unsafe path, and keeps the safe one it had', () => {
    remember(ARCADE, store(), NOW)
    for (const unsafe of ['https://evil.example', '//evil.example', '/chat', 'javascript:alert(1)']) {
      remember(unsafe, store(), NOW + 1)
    }
    expect(take(store(), NOW + 2)).toBe(ARCADE)

    remember('/chat', store(), NOW)
    expect(stored()).toBeNull()
  })

  it.each([
    ['another site', JSON.stringify({ path: 'https://evil.example', expiresAt: NOW + 1000 })],
    ['another page', JSON.stringify({ path: '/chat', expiresAt: NOW + 1000 })],
    ['an expiry past 30 minutes', JSON.stringify({ path: ARCADE, expiresAt: NOW + RETURN_PATH_TTL_MS + 1 })],
    ['an expiry that is text', JSON.stringify({ path: ARCADE, expiresAt: String(NOW + 1000) })],
    ['no expiry', JSON.stringify({ path: ARCADE })],
    ['no path', JSON.stringify({ expiresAt: NOW + 1000 })],
    ['an array', JSON.stringify([ARCADE, NOW + 1000])],
    ['null', 'null'],
    ['a bare path', ARCADE],
    ['broken JSON', '{"path":'],
  ])('a store holding %s gives nothing and is forgotten', (_name, raw) => {
    vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    store().setItem(RETURN_PATH_STORAGE_KEY, raw)
    expect(take(store(), NOW)).toBeNull()
    expect(stored()).toBeNull()
  })

  it('storage errors are logged, never thrown, and nothing is remembered', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    const broken: ReturnPathStorage = {
      getItem: () => {
        throw new Error('blocked')
      },
      setItem: () => {
        throw new Error('full')
      },
      removeItem: () => {
        throw new Error('blocked')
      },
    }
    expect(() => remember(ARCADE, broken, NOW)).not.toThrow()
    expect(take(broken, NOW)).toBeNull()
    expect(warn).toHaveBeenCalledTimes(2)

    expect(() => remember(ARCADE, null, NOW)).not.toThrow()
    expect(take(null, NOW)).toBeNull()
  })
})

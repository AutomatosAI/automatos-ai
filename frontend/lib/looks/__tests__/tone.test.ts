/**
 * PRD-260 K4 — the tone a look's ground sets, and its cookie: parsed
 * strictly, written and cleared by the LookProvider. PRD-261 A-03: the server
 * reads it into the theme boot under Play only (lib/theme-host/__tests__/boot.test.ts).
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

import { GROUND_KEYS } from '@/lib/looks/catalogue'
import { PLAY_TONE_COOKIE, parsePlayTone, toneOfGround } from '@/lib/looks/tone'
import { clearPlayToneCookie, writePlayToneCookie } from '@/lib/theme-host/builtin/play-tone-cookie'

function toneCookie(): string | undefined {
  return document.cookie.split('; ').find((entry) => entry.startsWith(`${PLAY_TONE_COOKIE}=`))?.split('=')[1]
}

afterEach(() => {
  vi.restoreAllMocks()
  document.cookie = `${PLAY_TONE_COOKIE}=; Max-Age=0; Path=/`
})

describe('toneOfGround', () => {
  it('Paper is light; Graphite, Obsidian and Night are dark', () => {
    expect(Object.fromEntries(GROUND_KEYS.map((key) => [key, toneOfGround(key)]))).toEqual({
      graphite: 'dark',
      obsidian: 'dark',
      indigo: 'dark',
      paper: 'light',
    })
  })
})

describe('the tone cookie', () => {
  it("parses only 'dark' and 'light'", () => {
    expect(parsePlayTone('dark')).toBe('dark')
    expect(parsePlayTone('light')).toBe('light')
    for (const value of ['Dark', 'system', '', ' dark', 'dark; Path=/', 'toString', null, undefined]) {
      expect({ value, tone: parsePlayTone(value) }).toEqual({ value, tone: undefined })
    }
  })

  it('is written and cleared on the whole site', () => {
    writePlayToneCookie('light')
    expect(toneCookie()).toBe('light')
    writePlayToneCookie('dark')
    expect(toneCookie()).toBe('dark')
    clearPlayToneCookie()
    expect(toneCookie()).toBeUndefined()
  })

  it('clearing writes nothing when there is no tone cookie (a Classic or Studio load)', () => {
    const write = vi.spyOn(Document.prototype, 'cookie', 'set')
    clearPlayToneCookie()
    expect(write).not.toHaveBeenCalled()
  })
})

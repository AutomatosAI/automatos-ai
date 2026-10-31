/**
 * PRD-260 P260-FX-27 (M88) — one cookie writer: the style cookie and the theme
 * tone cookie are written by lib/cookies.ts, so their flags can't drift apart.
 * PRD-261 A-19d2: the host's look runtime writes the tone cookie for an
 * installed theme whose tone follows its look (lib/theme-host/looks/tone-cookie.ts;
 * its own cases are lib/theme-host/looks/__tests__/tone-cookie.test.ts's).
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

import { applyUiStyleToDocument } from '@/contexts/ui-style-context'
import { writeCookie } from '@/lib/cookies'
import { HELLO_PACKAGE_ENTRY, INSTALLED_HELLO_STYLE } from '@/lib/theme-host/__tests__/installed-entry-fixture'
import { THEME_TONE_COOKIE, themeToneCookieValue, writeThemeToneCookie } from '@/lib/theme-host/looks/tone-cookie'
import { STYLE_HTML_CLASS_NAMES, UI_STYLE_COOKIE, UI_STYLE_COOKIE_MAX_AGE_S } from '@/lib/ui-style'

const TEST_COOKIE = 'fx27-cookie'
const TEST_VALUE = 'on'
const MAX_AGE_S = 60
const THEME_ID = HELLO_PACKAGE_ENTRY.themeId

/** The flags after the name=value pair. */
const flagsOf = (line: unknown) => String(line).split('; ').slice(1).join('; ')

afterEach(() => {
  vi.restoreAllMocks()
  for (const name of [TEST_COOKIE, UI_STYLE_COOKIE, THEME_TONE_COOKIE]) document.cookie = `${name}=; Max-Age=0; Path=/`
  document.documentElement.classList.remove(...STYLE_HTML_CLASS_NAMES)
})

describe('writeCookie', () => {
  it('writes the cookie for the whole site, Lax, with its Max-Age, and no Secure on an http page', () => {
    const write = vi.spyOn(Document.prototype, 'cookie', 'set')
    writeCookie(TEST_COOKIE, TEST_VALUE, MAX_AGE_S)
    expect(write).toHaveBeenCalledWith(`${TEST_COOKIE}=${TEST_VALUE}; Path=/; Max-Age=${MAX_AGE_S}; SameSite=Lax`)
    expect(document.cookie).toContain(`${TEST_COOKIE}=${TEST_VALUE}`)
  })

  it('the style cookie and the theme tone cookie carry the same flags', () => {
    const write = vi.spyOn(Document.prototype, 'cookie', 'set')
    applyUiStyleToDocument(INSTALLED_HELLO_STYLE)
    writeThemeToneCookie(THEME_ID, 'dark')
    const lines = write.mock.calls.map(([line]) => String(line))
    expect(lines.map((line) => line.split(';')[0])).toEqual([
      `${UI_STYLE_COOKIE}=${INSTALLED_HELLO_STYLE}`,
      `${THEME_TONE_COOKIE}=${themeToneCookieValue(THEME_ID, 'dark')}`,
    ])
    const expected = `Path=/; Max-Age=${UI_STYLE_COOKIE_MAX_AGE_S}; SameSite=Lax`
    expect(lines.map(flagsOf)).toEqual([expected, expected])
  })
})

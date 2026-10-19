/**
 * PRD-260 P260-FX-27 (M88) — one cookie writer: the style cookie and Play's
 * tone cookie are written by lib/cookies.ts, so their flags can't drift apart.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

import { applyUiStyleToDocument } from '@/contexts/ui-style-context'
import { writeCookie } from '@/lib/cookies'
import { PLAY_TONE_COOKIE, writePlayToneCookie } from '@/lib/looks/tone'
import { BUILTIN_PLAY_STYLE } from '@/lib/theme-host/builtin/play-entry'
import { STYLE_HTML_CLASS_NAMES, UI_STYLE_COOKIE, UI_STYLE_COOKIE_MAX_AGE_S } from '@/lib/ui-style'

const TEST_COOKIE = 'fx27-cookie'
const MAX_AGE_S = 60

/** The flags after the name=value pair. */
const flagsOf = (line: unknown) => String(line).split('; ').slice(1).join('; ')

afterEach(() => {
  vi.restoreAllMocks()
  for (const name of [TEST_COOKIE, UI_STYLE_COOKIE, PLAY_TONE_COOKIE]) document.cookie = `${name}=; Max-Age=0; Path=/`
  document.documentElement.classList.remove(...STYLE_HTML_CLASS_NAMES)
})

describe('writeCookie', () => {
  it('writes the cookie for the whole site, Lax, with its Max-Age, and no Secure on an http page', () => {
    const write = vi.spyOn(Document.prototype, 'cookie', 'set')
    writeCookie(TEST_COOKIE, 'play', MAX_AGE_S)
    expect(write).toHaveBeenCalledWith(`${TEST_COOKIE}=play; Path=/; Max-Age=${MAX_AGE_S}; SameSite=Lax`)
    expect(document.cookie).toContain(`${TEST_COOKIE}=play`)
  })

  it('the style cookie and the tone cookie carry the same flags', () => {
    const write = vi.spyOn(Document.prototype, 'cookie', 'set')
    applyUiStyleToDocument(BUILTIN_PLAY_STYLE)
    writePlayToneCookie('dark')
    const lines = write.mock.calls.map(([line]) => String(line))
    expect(lines.map((line) => line.split(';')[0])).toEqual([`${UI_STYLE_COOKIE}=${BUILTIN_PLAY_STYLE}`, `${PLAY_TONE_COOKIE}=dark`])
    const expected = `Path=/; Max-Age=${UI_STYLE_COOKIE_MAX_AGE_S}; SameSite=Lax`
    expect(lines.map(flagsOf)).toEqual([expected, expected])
  })
})

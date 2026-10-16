/**
 * PRD-259 S-004b — a skin as a style: its base drives the studio class and isStudio (so no
 * layout changes), choosing it writes both cookies, data-skin, the one <style> element and
 * the cache; leaving it removes all three; a single-tone skin locks the tone and leaving it
 * restores the user's own; and every fallback trigger shows the base with a notice, once.
 * P259-W1-RVW-2: the lock rides in the tone cookie, so it holds from the first paint, before
 * (or without) the installed list.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import { readFileSync } from 'fs'
import path from 'path'

const api = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn() }))
vi.mock('@/lib/api-client', () => ({ apiClient: { get: api.get, post: api.post } }))
const toasts = vi.hoisted(() => ({ info: vi.fn(), error: vi.fn(), success: vi.fn() }))
vi.mock('sonner', () => ({ toast: toasts }))
const cssFault = vi.hoisted(() => ({ on: false }))
vi.mock('@/lib/skins/css', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/lib/skins/css')>()
  return {
    ...actual,
    skinToCss: (skin: unknown, id: string) => {
      if (cssFault.on) throw new actual.SkinCssError('forced by the test')
      return actual.skinToCss(skin, id)
    },
  }
})

import { CASES, OTHER_SKIN_ID, SKIN_ID, SkinHarness, StyleProbe, THEME_STORAGE_KEY, installedItem } from '@/components/__tests__/prd259-skin-harness'
import { SKIN_CSS_CACHE_KEY } from '@/lib/skins/cache'
import { SKIN_STYLE_ELEMENT_ID } from '@/lib/skins/prepaint'
import {
  APPEARANCE_DEFAULTS_KEY,
  APPEARANCE_DEFAULTS_VERSION,
  STUDIO_HTML_CLASS,
  UI_SKIN_BASE_COOKIE,
  UI_SKIN_TONE_COOKIE,
  UI_STYLE_COOKIE,
  type BaseStyle,
  type UiStyle,
} from '@/lib/ui-style'

const html = () => document.documentElement
const styleElement = () => document.getElementById(SKIN_STYLE_ELEMENT_ID)
const cachedId = () => (JSON.parse(window.localStorage.getItem(SKIN_CSS_CACHE_KEY) ?? 'null') as { id?: string } | null)?.id
const click = (label: string) => act(() => { screen.getByText(label).click() })

function clearCookie(name: string) {
  document.cookie = `${name}=; Max-Age=0; Path=/`
}

beforeEach(() => {
  html().className = ''
  html().removeAttribute('data-skin')
  styleElement()?.remove()
  clearCookie(UI_STYLE_COOKIE)
  clearCookie(UI_SKIN_BASE_COOKIE)
  clearCookie(UI_SKIN_TONE_COOKIE)
  window.localStorage.clear()
  window.localStorage.setItem(APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION)
  api.get.mockReset()
  Object.values(toasts).forEach((fn) => fn.mockClear())
  cssFault.on = false
})
afterEach(cleanup)

interface StartAs {
  initialStyle?: UiStyle
  initialSkinBase?: BaseStyle
}

async function loaded(skins: unknown[], props: StartAs = {}) {
  api.get.mockResolvedValue(skins)
  render(<SkinHarness {...props}><StyleProbe /></SkinHarness>)
  click('load skins')
  await waitFor(() => expect(screen.getByTestId('skins').textContent).toMatch(/^ready:/))
}

describe('a skin as a style', () => {
  it('a Studio-based skin is Studio for isStudio from the first render (the server-read base)', () => {
    api.get.mockReturnValue(new Promise(() => undefined))
    render(<SkinHarness initialStyle={`skin:${SKIN_ID}`} initialSkinBase="studio"><StyleProbe /></SkinHarness>)
    expect(screen.getByTestId('studio').textContent).toBe('true')
  })

  it('a Classic-based skin is not Studio: isStudio false, no studio class', async () => {
    await loaded([installedItem(CASES.classicLightOnly, SKIN_ID)], { initialStyle: 'studio' })
    click('go skin')
    expect(screen.getByTestId('studio').textContent).toBe('false')
    expect(html().classList.contains(STUDIO_HTML_CLASS)).toBe(false)
    expect(document.cookie).toContain(`${UI_SKIN_BASE_COOKIE}=classic`)
  })

  it('choosing a skin writes both cookies, the class from the base, data-skin, the style element and the cache', async () => {
    await loaded([installedItem(CASES.twoTone, SKIN_ID)], { initialStyle: 'classic' })
    click('go skin')
    expect(screen.getByTestId('style').textContent).toBe(`skin:${SKIN_ID}`)
    expect(screen.getByTestId('studio').textContent).toBe('true')
    expect(html().classList.contains(STUDIO_HTML_CLASS)).toBe(true)
    expect(document.cookie).toContain(`${UI_STYLE_COOKIE}=skin:${SKIN_ID}`)
    expect(document.cookie).toContain(`${UI_SKIN_BASE_COOKIE}=studio`)
    expect(html().getAttribute('data-skin')).toBe(SKIN_ID)
    expect(styleElement()?.textContent).toContain(`html:not(.dark)[data-skin="${SKIN_ID}"]{`)
    expect(cachedId()).toBe(SKIN_ID)
  })

  it('choosing Classic or Studio removes data-skin, the style element and the cache entry', async () => {
    await loaded([installedItem(CASES.twoTone, SKIN_ID)])
    click('go skin')
    click('go classic')
    expect(html().hasAttribute('data-skin')).toBe(false)
    expect(styleElement()).toBeNull()
    expect(window.localStorage.getItem(SKIN_CSS_CACHE_KEY)).toBeNull()
    expect(html().classList.contains(STUDIO_HTML_CLASS)).toBe(false)
    expect(document.cookie).toContain(`${UI_STYLE_COOKIE}=classic`)
  })

  it('opening on a skin loads the list at once and repairs the stylesheet', async () => {
    api.get.mockResolvedValue([installedItem(CASES.twoTone, SKIN_ID)])
    render(<SkinHarness initialStyle={`skin:${SKIN_ID}`} initialSkinBase="studio"><StyleProbe /></SkinHarness>)
    await waitFor(() => expect(styleElement()?.textContent).toContain(SKIN_ID))
    expect(api.get).toHaveBeenCalledWith('/api/skins/installed')
    expect(cachedId()).toBe(SKIN_ID)
  })
})

describe('the tone lock', () => {
  it('a dark-only skin forces Dark; back to Studio restores the stored light', async () => {
    window.localStorage.setItem(THEME_STORAGE_KEY, 'light')
    await loaded([installedItem(CASES.darkOnly, SKIN_ID)], { initialStyle: 'studio' })
    await waitFor(() => expect(html().classList.contains('light')).toBe(true))
    click('go skin')
    await waitFor(() => expect(html().classList.contains('dark')).toBe(true))
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe('light')
    click('go studio')
    await waitFor(() => expect(html().classList.contains('light')).toBe(true))
    expect(html().classList.contains('dark')).toBe(false)
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe('light')
  })

  it('a light-only skin forces Light (decision K4)', async () => {
    window.localStorage.setItem(THEME_STORAGE_KEY, 'dark')
    await loaded([installedItem(CASES.classicLightOnly, SKIN_ID)], { initialStyle: 'studio' })
    click('go skin')
    await waitFor(() => expect(html().classList.contains('light')).toBe(true))
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe('dark')
  })

  it('Providers lifts the lock into ThemeProvider forcedTheme', () => {
    const src = readFileSync(path.resolve(__dirname, '..', '..', 'components', 'providers.tsx'), 'utf8')
    expect(src).toContain('forcedTheme={lockedTone}')
    expect(src).toContain('onToneLock={setLockedTone}')
  })
})

describe('the tone lock from the first paint (the tone cookie)', () => {
  const lockedSkin = `skin:${SKIN_ID}` as const
  const toneCookie = (tone: string) => `${UI_SKIN_TONE_COOKIE}=${tone}`

  /** What the server had: the style cookie on a dark-only skin, and its tone cookie. */
  function serverCookies() {
    document.cookie = `${UI_STYLE_COOKIE}=${lockedSkin}; Path=/`
    document.cookie = `${toneCookie('dark')}; Path=/`
  }

  function mountLocked() {
    render(
      <SkinHarness initialStyle={lockedSkin} initialSkinBase="studio" initialLockedTone="dark">
        <StyleProbe />
      </SkinHarness>,
    )
  }

  it('(a) the locked Dark holds before the installed list resolves, over a stored light', () => {
    window.localStorage.setItem(THEME_STORAGE_KEY, 'light')
    api.get.mockReturnValue(new Promise(() => undefined))
    mountLocked()
    expect(screen.getByTestId('skins').textContent).not.toMatch(/^ready:/)
    expect(html().classList.contains('dark')).toBe(true)
    expect(html().classList.contains('light')).toBe(false)
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe('light')
  })

  it('(b) a failed list load leaves Dark forced and the stored light alone', async () => {
    window.localStorage.setItem(THEME_STORAGE_KEY, 'light')
    serverCookies()
    api.get.mockRejectedValue(new Error('offline'))
    mountLocked()
    await waitFor(() => expect(screen.getByTestId('skins').textContent).toMatch(/^error:/))
    expect(html().classList.contains('dark')).toBe(true)
    expect(html().classList.contains('light')).toBe(false)
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe('light')
    expect(document.cookie).toContain(toneCookie('dark'))
  })

  it('(c) choosing Studio afterwards clears the tone cookie and restores the stored light', async () => {
    window.localStorage.setItem(THEME_STORAGE_KEY, 'light')
    serverCookies()
    api.get.mockRejectedValue(new Error('offline'))
    mountLocked()
    await waitFor(() => expect(screen.getByTestId('skins').textContent).toMatch(/^error:/))
    click('go studio')
    await waitFor(() => expect(html().classList.contains('light')).toBe(true))
    expect(html().classList.contains('dark')).toBe(false)
    expect(document.cookie).not.toContain(`${UI_SKIN_TONE_COOKIE}=`)
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe('light')
  })

  it('(d) the fallback clears the tone cookie and the lock', async () => {
    window.localStorage.setItem(THEME_STORAGE_KEY, 'light')
    serverCookies()
    api.get.mockResolvedValue([])
    mountLocked()
    await waitFor(() => expect(screen.getByTestId('style').textContent).toBe('studio'))
    await waitFor(() => expect(html().classList.contains('light')).toBe(true))
    expect(document.cookie).not.toContain(`${UI_SKIN_TONE_COOKIE}=`)
    expect(toasts.info).toHaveBeenCalledTimes(1)
  })

  it('a single-tone skin writes its tone cookie; a two-tone skin clears it', async () => {
    await loaded([installedItem(CASES.darkOnly, SKIN_ID), installedItem(CASES.twoTone, OTHER_SKIN_ID)], { initialStyle: 'studio' })
    click('go skin')
    expect(document.cookie).toContain(toneCookie('dark'))
    click('go other skin')
    expect(document.cookie).not.toContain(`${UI_SKIN_TONE_COOKIE}=`)
  })

  it('the server reads the tone cookie and Providers starts forcedTheme from it', () => {
    const read = (...parts: string[]) => readFileSync(path.resolve(__dirname, '..', '..', ...parts), 'utf8')
    const layout = read('app', 'layout.tsx')
    expect(layout).toContain('lockedToneFor(styleCookie, store.get(UI_SKIN_TONE_COOKIE)?.value)')
    expect(layout).toContain('initialLockedTone={lockedTone}')
    const providers = read('components', 'providers.tsx')
    expect(providers).toContain('useState<SkinTone | undefined>(initialLockedTone)')
    expect(providers).toContain('initialLockedTone={initialLockedTone}')
  })

  it('the list resolving the same dark-only skin keeps the lock and rewrites the cookie', async () => {
    window.localStorage.setItem(THEME_STORAGE_KEY, 'light')
    api.get.mockResolvedValue([installedItem(CASES.darkOnly, SKIN_ID)])
    mountLocked()
    await waitFor(() => expect(styleElement()?.textContent).toContain(SKIN_ID))
    expect(html().classList.contains('dark')).toBe(true)
    expect(document.cookie).toContain(toneCookie('dark'))
  })
})

describe('the fallback', () => {
  function expectBase(base: 'studio' | 'classic') {
    expect(screen.getByTestId('style').textContent).toBe(base)
    expect(html().hasAttribute('data-skin')).toBe(false)
    expect(styleElement()).toBeNull()
    expect(window.localStorage.getItem(SKIN_CSS_CACHE_KEY)).toBeNull()
    expect(document.cookie).toContain(`${UI_STYLE_COOKIE}=${base}`)
  }

  it('a skin that is not installed here: the base, the cookie rewritten, the cache cleared, one notice', async () => {
    window.localStorage.setItem(SKIN_CSS_CACHE_KEY, JSON.stringify({ id: SKIN_ID, version: '1.0.0', css: 'x' }))
    api.get.mockResolvedValue([installedItem(CASES.twoTone, OTHER_SKIN_ID)])
    render(<SkinHarness initialStyle={`skin:${SKIN_ID}`} initialSkinBase="classic"><StyleProbe /></SkinHarness>)
    await waitFor(() => expect(screen.getByTestId('style').textContent).toBe('classic'))
    expectBase('classic')
    expect(toasts.info).toHaveBeenCalledTimes(1)
    expect(toasts.info).toHaveBeenCalledWith("The skin isn't available here, so Classic is showing.")
    click('go skin')
    expectBase('classic')
    expect(toasts.info).toHaveBeenCalledTimes(1)
  })

  it('no base cookie: Studio shows', async () => {
    api.get.mockResolvedValue([])
    render(<SkinHarness initialStyle={`skin:${SKIN_ID}`}><StyleProbe /></SkinHarness>)
    await waitFor(() => expect(screen.getByTestId('style').textContent).toBe('studio'))
    expectBase('studio')
    expect(toasts.info).toHaveBeenCalledWith("The skin isn't available here, so Studio is showing.")
  })

  it('a skin that fails validateSkin: the base, with its name in the notice', async () => {
    const tampered = installedItem(CASES.twoTone, SKIN_ID, { tokens: { light: { background: '0 0% 0%; } body{display:none' } } })
    await loaded([tampered], { initialStyle: 'studio' })
    click('go skin')
    expectBase('studio')
    expect(toasts.info).toHaveBeenCalledWith("Studio, as a skin isn't available here, so Studio is showing.")
  })

  it('a skin skinToCss refuses: the base, with a notice', async () => {
    await loaded([installedItem(CASES.twoTone, SKIN_ID)], { initialStyle: 'classic' })
    cssFault.on = true
    click('go skin')
    expectBase('classic')
    expect(toasts.info).toHaveBeenCalledWith("Studio, as a skin isn't available here, so Classic is showing.")
  })

  it('a failed list load keeps the cached skin (no fallback on a network blip)', async () => {
    api.get.mockRejectedValue(new Error('offline'))
    render(<SkinHarness initialStyle={`skin:${SKIN_ID}`} initialSkinBase="studio"><StyleProbe /></SkinHarness>)
    await waitFor(() => expect(screen.getByTestId('skins').textContent).toMatch(/^error:/))
    expect(screen.getByTestId('style').textContent).toBe(`skin:${SKIN_ID}`)
    expect(toasts.info).not.toHaveBeenCalled()
  })
})

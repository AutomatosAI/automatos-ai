/**
 * PRD-260 K3, K4 — the LookProvider puts the look on <html> only while the
 * style is play, forces the tone its ground sets (through ThemeProvider's
 * forcedTheme, as components/providers.tsx wires it), and leaves nothing
 * behind when the style leaves Play: no property, no motion mark, no tone
 * cookie, and the user's own stored tone back.
 */
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import { useState, type ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { LookProvider, useLook } from '@/components/looks/look-provider'
import { ThemeProvider } from '@/components/theme-provider'
import { UiStyleProvider, useUiStyle } from '@/contexts/ui-style-context'
import { MOTION_ATTRIBUTE, MOTION_VAR, lookDocumentVars } from '@/lib/looks/document'
import { DAYLIGHT, DEFAULT_LOOK, OBSIDIAN } from '@/lib/looks/presets'
import { ACTIVE_LOOK_KEY, SAVED_LOOKS_KEY, type StorageSource } from '@/lib/looks/storage'
import { PLAY_TONE_COOKIE, type PlayTone } from '@/lib/looks/tone'
import type { Look } from '@/lib/looks/types'
import { APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION, type UiStyle } from '@/lib/ui-style'
import { blockedStorage, clearRoot, lookNamesLeft, myLook, rootVars } from '@/lib/looks/__tests__/helpers'

/** next-themes' storage key in components/providers.tsx: the user's own tone. */
const TONE_STORAGE_KEY = 'automatos-theme'
const PROVIDERS = path.resolve(__dirname, '../../providers.tsx')

/** The appearance providers exactly as components/providers.tsx nests them. */
function Appearance({ style, storage, children }: { style: UiStyle; storage?: StorageSource; children: ReactNode }) {
  const [playTone, setPlayTone] = useState<PlayTone | undefined>(undefined)
  return (
    <ThemeProvider
      attribute="class"
      defaultTheme="dark"
      enableSystem
      themes={['light', 'dark']}
      storageKey={TONE_STORAGE_KEY}
      disableTransitionOnChange
      forcedTheme={playTone}
    >
      <UiStyleProvider initialStyle={style}>
        <LookProvider storage={storage} onToneChange={setPlayTone}>{children}</LookProvider>
      </UiStyleProvider>
    </ThemeProvider>
  )
}

function Probe() {
  const { look, saved, tone, reducedMotion, setLook, saveLook, deleteLook } = useLook()
  const { setStyle } = useUiStyle()
  const pick = (next: Look) => () => setLook(next)
  return (
    <div>
      <p>Active: {look.name}</p>
      <p>Saved: {saved.map((entry) => entry.name).join(', ') || 'none'}</p>
      <p>Tone: {tone ?? 'not forced'}</p>
      <p>Motion: {reducedMotion ? 'reduced' : 'full'}</p>
      <button type="button" onClick={pick(DAYLIGHT)}>Use Daylight</button>
      <button type="button" onClick={pick(OBSIDIAN)}>Use Obsidian</button>
      <button type="button" onClick={() => saveLook(myLook())}>Save mine</button>
      <button type="button" onClick={() => deleteLook(myLook().id)}>Delete mine</button>
      <button type="button" onClick={() => setStyle('play')}>Go Play</button>
      <button type="button" onClick={() => setStyle('studio')}>Go Studio</button>
      <button type="button" onClick={() => setStyle('classic')}>Go Classic</button>
    </div>
  )
}

function renderApp(style: UiStyle, storage?: StorageSource) {
  return render(<Appearance style={style} storage={storage}><Probe /></Appearance>)
}

function click(name: string): void {
  act(() => screen.getByRole('button', { name }).click())
}

function storeActive(look: Look): void {
  window.localStorage.setItem(ACTIVE_LOOK_KEY, JSON.stringify(look))
}

function toneCookie(): string | undefined {
  return document.cookie.split('; ').find((entry) => entry.startsWith(`${PLAY_TONE_COOKIE}=`))?.split('=')[1]
}

function htmlTone(): string {
  const classes = document.documentElement.classList
  return ['light', 'dark'].filter((tone) => classes.contains(tone)).join(' ')
}

function stubReducedMotion(matches: boolean): void {
  vi.stubGlobal('matchMedia', (query: string) => ({
    matches: query.includes('reduced-motion') ? matches : false,
    media: query,
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
    addListener: () => undefined,
    removeListener: () => undefined,
  }))
}

beforeEach(() => {
  window.localStorage.clear()
  // A browser already moved to the current defaults, so the provider keeps the given style.
  window.localStorage.setItem(APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION)
  document.cookie = `${PLAY_TONE_COOKIE}=; Max-Age=0; Path=/`
  clearRoot()
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  window.localStorage.clear()
  clearRoot()
})

describe('LookProvider under Play', () => {
  it('puts the stored look on <html>: Play variables and Studio tokens', async () => {
    storeActive(myLook())
    renderApp('play')
    expect(await screen.findByText('Active: My look 1')).toBeInTheDocument()
    await waitFor(() => expect(rootVars()).toEqual(lookDocumentVars(myLook())))
  })

  it('applies the default look when nothing is stored, and writes nothing', async () => {
    renderApp('play')
    await waitFor(() => expect(rootVars()).toEqual(lookDocumentVars(DEFAULT_LOOK)))
    expect(window.localStorage.getItem(ACTIVE_LOOK_KEY)).toBeNull()
  })

  it('switches the variables at once and keeps the choice', async () => {
    renderApp('play')
    click('Use Daylight')
    await waitFor(() => expect(rootVars()).toEqual(lookDocumentVars(DAYLIGHT)))
    expect(JSON.parse(window.localStorage.getItem(ACTIVE_LOOK_KEY) ?? 'null')).toEqual(DAYLIGHT)
  })

  it('works on the default look with storage blocked, and still switches looks', async () => {
    renderApp('play', blockedStorage)
    await waitFor(() => expect(rootVars()).toEqual(lookDocumentVars(DEFAULT_LOOK)))
    click('Use Obsidian')
    await waitFor(() => expect(rootVars()).toEqual(lookDocumentVars(OBSIDIAN)))
    expect(console.warn).toHaveBeenCalled()
  })

  it('saves and deletes the user\'s looks', async () => {
    renderApp('play')
    click('Save mine')
    expect(screen.getByText('Saved: My look 1')).toBeInTheDocument()
    expect(JSON.parse(window.localStorage.getItem(SAVED_LOOKS_KEY) ?? 'null')).toEqual([myLook()])
    click('Delete mine')
    expect(screen.getByText('Saved: none')).toBeInTheDocument()
  })
})

describe('LookProvider — the tone follows the ground (K4)', () => {
  it('a Paper look forces light, Obsidian forces dark, and back on Studio the stored tone returns', async () => {
    window.localStorage.setItem(TONE_STORAGE_KEY, 'dark')
    storeActive(DAYLIGHT)
    renderApp('play')
    await waitFor(() => expect(htmlTone()).toBe('light'))
    expect(screen.getByText('Tone: light')).toBeInTheDocument()
    expect(toneCookie()).toBe('light')

    click('Use Obsidian')
    await waitFor(() => expect(htmlTone()).toBe('dark'))
    expect(toneCookie()).toBe('dark')

    click('Use Daylight')
    await waitFor(() => expect(htmlTone()).toBe('light'))

    click('Go Studio')
    await waitFor(() => expect(htmlTone()).toBe('dark'))
    expect(screen.getByText('Tone: not forced')).toBeInTheDocument()
    expect(window.localStorage.getItem(TONE_STORAGE_KEY)).toBe('dark')
  })

  it('never overwrites the user\'s stored tone while it forces one', async () => {
    window.localStorage.setItem(TONE_STORAGE_KEY, 'light')
    storeActive(OBSIDIAN)
    renderApp('play')
    await waitFor(() => expect(htmlTone()).toBe('dark'))
    expect(window.localStorage.getItem(TONE_STORAGE_KEY)).toBe('light')
    click('Go Classic')
    await waitFor(() => expect(htmlTone()).toBe('light'))
  })

  it('is wired the same way in components/providers.tsx', () => {
    const src = readFileSync(PROVIDERS, 'utf8')
    expect(src).toContain('forcedTheme={playTone}')
    expect(src).toContain('<LookProvider onToneChange={setPlayTone}>')
    expect(src).toContain('useState<PlayTone | undefined>(initialPlayTone)')
  })
})

describe('LookProvider — leaving Play leaves nothing behind (K3)', () => {
  it('play → studio removes every look property, the motion mark and the tone cookie', async () => {
    storeActive(myLook())
    renderApp('play')
    await waitFor(() => expect(rootVars()).toEqual(lookDocumentVars(myLook())))
    expect(document.documentElement).toHaveAttribute(MOTION_ATTRIBUTE)
    expect(toneCookie()).toBe('dark')

    click('Go Studio')
    await waitFor(() => expect(lookNamesLeft()).toEqual([]))
    expect(document.documentElement.style.getPropertyValue(MOTION_VAR)).toBe('')
    expect(document.documentElement).not.toHaveAttribute(MOTION_ATTRIBUTE)
    expect(toneCookie()).toBeUndefined()
  })

  it('studio → play applies the stored look and its tone at once (no pre-paint ran)', async () => {
    storeActive(DAYLIGHT)
    renderApp('studio')
    expect(lookNamesLeft()).toEqual([])
    click('Go Play')
    await waitFor(() => expect(rootVars()).toEqual(lookDocumentVars(DAYLIGHT)))
    await waitFor(() => expect(htmlTone()).toBe('light'))
  })

  it('sets nothing under Studio or Classic', async () => {
    storeActive(myLook())
    for (const style of ['studio', 'classic'] as const) {
      renderApp(style)
      expect(await screen.findByText('Active: My look 1')).toBeInTheDocument()
      expect({ style, left: lookNamesLeft() }).toEqual({ style, left: [] })
      expect(document.documentElement).not.toHaveAttribute(MOTION_ATTRIBUTE)
      expect(toneCookie()).toBeUndefined()
      cleanup()
    }
  })

  it('takes its variables off <html> when it unmounts', async () => {
    const { unmount } = renderApp('play')
    await waitFor(() => expect(rootVars()).toEqual(lookDocumentVars(DEFAULT_LOOK)))
    unmount()
    expect(lookNamesLeft()).toEqual([])
    expect(document.documentElement).not.toHaveAttribute(MOTION_ATTRIBUTE)
  })
})

describe('LookProvider — motion', () => {
  it("reduced motion sets data-pl-motion='reduce' and --pl-motion 0", async () => {
    stubReducedMotion(true)
    renderApp('play')
    expect(screen.getByText('Motion: reduced')).toBeInTheDocument()
    await waitFor(() => expect(document.documentElement).toHaveAttribute(MOTION_ATTRIBUTE, 'reduce'))
    expect(document.documentElement.style.getPropertyValue(MOTION_VAR)).toBe('0')
  })

  it("full motion sets data-pl-motion='full' and --pl-motion 1", async () => {
    stubReducedMotion(false)
    renderApp('play')
    await waitFor(() => expect(document.documentElement).toHaveAttribute(MOTION_ATTRIBUTE, 'full'))
    expect(document.documentElement.style.getPropertyValue(MOTION_VAR)).toBe('1')
  })
})

describe('useLook outside a provider', () => {
  it('reads the default look and warns instead of changing anything', () => {
    function Bare() {
      const { look, setLook } = useLook()
      return <button type="button" onClick={() => setLook(DAYLIGHT)}>{look.name}</button>
    }
    render(<Bare />)
    click(DEFAULT_LOOK.name)
    expect(screen.getByRole('button', { name: DEFAULT_LOOK.name })).toBeInTheDocument()
    expect(console.warn).toHaveBeenCalledWith(expect.stringContaining('needs a LookProvider'))
  })
})

/**
 * PRD-260 K3, K4 — the LookProvider puts the look on <html> only while the
 * style is play, forces the tone its ground sets (through ThemeProvider's
 * forcedTheme, as components/providers.tsx wires it), and leaves nothing
 * behind when the style leaves Play: no property, no motion mark, no tone
 * cookie, and the user's own stored tone back. A change another tab stores
 * arrives here, the tone with it, and is never written back (M25).
 */
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import { useState, type ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { LookProvider, useLook } from '@/lib/theme-host/builtin/play-look-provider'
import { ThemeProvider } from '@/components/theme-provider'
import { UiStyleProvider, useUiStyle } from '@/contexts/ui-style-context'
import { MOTION_ATTRIBUTE, MOTION_VAR, lookDocumentVars } from '@/lib/looks/document'
import { SAVED_SUB, asNewLook } from '@/lib/looks/editor'
import { DAYLIGHT, DEFAULT_LOOK, OBSIDIAN } from '@/lib/looks/presets'
import { DEFAULT_LOOK_SCOPE } from '@/lib/looks/scope'
import { LOOK_SCOPE_KEY, lookKeys, type StorageSource } from '@/lib/theme-host/builtin/play-look-storage'
import { PLAY_TONE_COOKIE, type PlayTone } from '@/lib/looks/tone'
import type { Look } from '@/lib/looks/types'
import { BUILTIN_PLAY_STYLE } from '@/lib/theme-host/builtin/play-entry'
import { APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION, type UiStyle } from '@/lib/ui-style'
import { blockedStorage, clearRoot, lookNamesLeft, myLook, rootVars } from '@/lib/looks/__tests__/helpers'

/** next-themes' storage key in components/providers.tsx: the user's own tone. */
const TONE_STORAGE_KEY = 'automatos-theme'
const PROVIDERS = path.resolve(__dirname, '../../providers.tsx')
/** PRD-261 A-13c: the host component that mounts this provider (components/providers.tsx mounts it). */
const LOOK_RUNTIME = path.resolve(__dirname, '../../theme-host/theme-look-runtime.tsx')
/** No AccountLookScope above the provider here, so the looks are kept under the default scope (F5). */
const KEYS = lookKeys(DEFAULT_LOOK_SCOPE)

/** The appearance providers exactly as components/providers.tsx nests them. */
function Appearance({ style, storage, enabled, children }: { style: UiStyle; storage?: StorageSource; enabled?: boolean; children: ReactNode }) {
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
        <LookProvider storage={storage} onToneChange={setPlayTone} enabled={enabled}>{children}</LookProvider>
      </UiStyleProvider>
    </ThemeProvider>
  )
}

function Probe() {
  const { look, saved, tone, reducedMotion, setLook, saveLook, deleteLook } = useLook()
  const saveAsNew = () => {
    const result = saveLook(asNewLook(look, saved))
    if (result.ok) setLook(result.look)
  }
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
      <button type="button" onClick={saveAsNew}>Save as new look</button>
      <button type="button" onClick={() => setStyle(BUILTIN_PLAY_STYLE)}>Go Play</button>
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
  window.localStorage.setItem(KEYS.active, JSON.stringify(look))
}

function storedJson(key: string): unknown {
  return JSON.parse(window.localStorage.getItem(key) ?? 'null')
}

/** What another tab does: writes the key itself, and the browser tells this tab with a storage event. */
function storeFromAnotherTab(key: string, value: unknown): void {
  window.localStorage.setItem(key, JSON.stringify(value))
  act(() => {
    window.dispatchEvent(new StorageEvent('storage', { key }))
  })
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
    renderApp(BUILTIN_PLAY_STYLE)
    expect(await screen.findByText('Active: My look 1')).toBeInTheDocument()
    await waitFor(() => expect(rootVars()).toEqual(lookDocumentVars(myLook())))
  })

  it('applies the default look when nothing is stored, and writes nothing', async () => {
    renderApp(BUILTIN_PLAY_STYLE)
    await waitFor(() => expect(rootVars()).toEqual(lookDocumentVars(DEFAULT_LOOK)))
    expect(window.localStorage.getItem(KEYS.active)).toBeNull()
  })

  it('switches the variables at once and keeps the choice', async () => {
    renderApp(BUILTIN_PLAY_STYLE)
    click('Use Daylight')
    await waitFor(() => expect(rootVars()).toEqual(lookDocumentVars(DAYLIGHT)))
    expect(JSON.parse(window.localStorage.getItem(KEYS.active) ?? 'null')).toEqual(DAYLIGHT)
  })

  it('works on the default look with storage blocked, and still switches looks', async () => {
    renderApp(BUILTIN_PLAY_STYLE, blockedStorage)
    await waitFor(() => expect(rootVars()).toEqual(lookDocumentVars(DEFAULT_LOOK)))
    click('Use Obsidian')
    await waitFor(() => expect(rootVars()).toEqual(lookDocumentVars(OBSIDIAN)))
    expect(console.warn).toHaveBeenCalled()
  })

  it('saves and deletes the user\'s looks', async () => {
    renderApp(BUILTIN_PLAY_STYLE)
    click('Save mine')
    expect(screen.getByText('Saved: My look 1')).toBeInTheDocument()
    expect(JSON.parse(window.localStorage.getItem(KEYS.saved) ?? 'null')).toEqual([myLook()])
    click('Delete mine')
    expect(screen.getByText('Saved: none')).toBeInTheDocument()
  })
})

describe('LookProvider switched off (PRD-261 A-13c: the theme has no look runtime)', () => {
  it('paints nothing and forces no tone, even under Play', async () => {
    window.localStorage.setItem(TONE_STORAGE_KEY, 'dark')
    storeActive(DAYLIGHT)
    render(<Appearance style={BUILTIN_PLAY_STYLE} enabled={false}><Probe /></Appearance>)
    expect(await screen.findByText('Active: Daylight')).toBeInTheDocument()
    expect(screen.getByText('Tone: not forced')).toBeInTheDocument()
    await waitFor(() => expect(htmlTone()).toBe('dark'))
    expect(lookNamesLeft()).toEqual([])
    expect(toneCookie()).toBeUndefined()
  })
})

describe('LookProvider — the tone follows the ground (K4)', () => {
  it('a Paper look forces light, Obsidian forces dark, and back on Studio the stored tone returns', async () => {
    window.localStorage.setItem(TONE_STORAGE_KEY, 'dark')
    storeActive(DAYLIGHT)
    renderApp(BUILTIN_PLAY_STYLE)
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
    renderApp(BUILTIN_PLAY_STYLE)
    await waitFor(() => expect(htmlTone()).toBe('dark'))
    expect(window.localStorage.getItem(TONE_STORAGE_KEY)).toBe('light')
    click('Go Classic')
    await waitFor(() => expect(htmlTone()).toBe('light'))
  })

  // PRD-261 A-13c moved the mount: components/providers.tsx mounts ThemeLookRuntime with the same tone
  // setter, and ThemeLookRuntime mounts this provider (at every width), switched on while the theme declares looks.
  // These two pin the same wiring through both files; theme-look-runtime.test.tsx renders it.
  it('is wired the same way in components/providers.tsx, through the host look runtime', () => {
    const src = readFileSync(PROVIDERS, 'utf8')
    const runtime = readFileSync(LOOK_RUNTIME, 'utf8')
    // At every width, as this file's harness renders it: the look paints below 1024 px too (K4, P260-SEAM).
    expect(src).toContain('forcedTheme={playTone}')
    expect(src).toContain('<ThemeLookRuntime onToneChange={setPlayTone}>')
    expect(src).toContain('useState<ThemeTone | undefined>(initialTone)')
    expect(runtime).toContain('<LookProvider onToneChange={onToneChange} enabled={mounted}>')
  })

  it('reads the account inside the WorkspaceProvider in components/providers.tsx (F5)', () => {
    const inOrder = (src: string, order: readonly string[]) => {
      const at = order.map((tag) => src.indexOf(tag))
      expect(at.every((index) => index >= 0), order.join(' ')).toBe(true)
      expect([...at].sort((a, b) => a - b)).toEqual(at)
    }
    inOrder(readFileSync(PROVIDERS, 'utf8'), [
      '<WorkspaceProvider>',
      '<ThemeLookRuntime onToneChange={setPlayTone}>',
      '</ThemeLookRuntime>',
      '</WorkspaceProvider>',
    ])
    inOrder(readFileSync(LOOK_RUNTIME, 'utf8'), [
      '<AccountLookScope enabled={mounted}>',
      '<LookProvider onToneChange={onToneChange} enabled={mounted}>',
      '</LookProvider>',
      '</AccountLookScope>',
    ])
  })
})

describe('LookProvider — another tab (M25)', () => {
  const theirs = myLook({ id: 'look-1', name: 'My look 1', sub: SAVED_SUB })

  it("takes the saved looks another tab stored, so Save as new look keeps both under different ids", async () => {
    renderApp(BUILTIN_PLAY_STYLE)
    await waitFor(() => expect(rootVars()).toEqual(lookDocumentVars(DEFAULT_LOOK)))
    storeFromAnotherTab(KEYS.saved, [theirs])
    expect(screen.getByText('Saved: My look 1')).toBeInTheDocument()

    click('Save as new look')
    expect(screen.getByText('Saved: My look 1, My look 2')).toBeInTheDocument()
    const kept = storedJson(KEYS.saved) as Look[]
    expect(kept.map((look) => look.id)).toEqual(['look-1', 'look-2'])
    expect(kept[0]).toEqual(theirs)
  })

  it('follows the active look another tab stored, and the tone cookie with it', async () => {
    renderApp(BUILTIN_PLAY_STYLE)
    await waitFor(() => expect(toneCookie()).toBe('dark'))
    storeFromAnotherTab(KEYS.active, DAYLIGHT)
    expect(screen.getByText('Active: Daylight')).toBeInTheDocument()
    await waitFor(() => expect(rootVars()).toEqual(lookDocumentVars(DAYLIGHT)))
    expect(toneCookie()).toBe('light')
    await waitFor(() => expect(htmlTone()).toBe('light'))
  })

  it('never writes back what it took, so two tabs cannot ping-pong', async () => {
    renderApp(BUILTIN_PLAY_STYLE)
    await waitFor(() => expect(rootVars()).toEqual(lookDocumentVars(DEFAULT_LOOK)))
    const setItem = vi.spyOn(Storage.prototype, 'setItem')
    storeFromAnotherTab(KEYS.active, OBSIDIAN)
    storeFromAnotherTab(KEYS.saved, [theirs])
    await waitFor(() => expect(rootVars()).toEqual(lookDocumentVars(OBSIDIAN)))
    const keys = setItem.mock.calls.map(([key]) => key)
    expect(keys.filter((key) => key === KEYS.active || key === KEYS.saved)).toEqual([KEYS.active, KEYS.saved])
  })

  it("ignores another key's event, and takes a cleared storage as nothing stored", async () => {
    storeActive(DAYLIGHT)
    renderApp(BUILTIN_PLAY_STYLE)
    expect(await screen.findByText('Active: Daylight')).toBeInTheDocument()
    storeFromAnotherTab(lookKeys('ws-2:user-b').active, OBSIDIAN)
    expect(screen.getByText('Active: Daylight')).toBeInTheDocument()

    window.localStorage.removeItem(KEYS.active)
    act(() => {
      window.dispatchEvent(new StorageEvent('storage', { key: null }))
    })
    expect(screen.getByText('Active: Ember')).toBeInTheDocument()
  })
})

describe('LookProvider — the pre-paint pointer (F5)', () => {
  it("points the next page load at this account's active look, only under Play", async () => {
    renderApp('studio')
    expect(await screen.findByText('Active: Ember')).toBeInTheDocument()
    expect(window.localStorage.getItem(LOOK_SCOPE_KEY)).toBeNull()
    click('Go Play')
    await waitFor(() => expect(window.localStorage.getItem(LOOK_SCOPE_KEY)).toBe(DEFAULT_LOOK_SCOPE))
  })
})

describe('LookProvider — leaving Play leaves nothing behind (K3)', () => {
  it('play → studio removes every look property, the motion mark and the tone cookie', async () => {
    storeActive(myLook())
    renderApp(BUILTIN_PLAY_STYLE)
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
    const { unmount } = renderApp(BUILTIN_PLAY_STYLE)
    await waitFor(() => expect(rootVars()).toEqual(lookDocumentVars(DEFAULT_LOOK)))
    unmount()
    expect(lookNamesLeft()).toEqual([])
    expect(document.documentElement).not.toHaveAttribute(MOTION_ATTRIBUTE)
  })
})

describe('LookProvider — motion', () => {
  it("reduced motion sets data-pl-motion='reduce' and --pl-motion 0", async () => {
    stubReducedMotion(true)
    renderApp(BUILTIN_PLAY_STYLE)
    expect(screen.getByText('Motion: reduced')).toBeInTheDocument()
    await waitFor(() => expect(document.documentElement).toHaveAttribute(MOTION_ATTRIBUTE, 'reduce'))
    expect(document.documentElement.style.getPropertyValue(MOTION_VAR)).toBe('0')
  })

  it("full motion sets data-pl-motion='full' and --pl-motion 1", async () => {
    stubReducedMotion(false)
    renderApp(BUILTIN_PLAY_STYLE)
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

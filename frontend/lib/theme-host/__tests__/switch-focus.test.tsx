/**
 * PRD-261 G3 (P260-FX-22, M21) — 'Switch to Play' onto Play's installed package reloads the page
 * (PA-13), so the note leaves word in this tab's sessionStorage just before the reload, and the
 * next load takes it once (theme-slot-page-switch.test.tsx proves the heading then takes focus).
 *
 * - The word: the page's path under one key, read and cleared together; another path's word is
 *   cleared and ignored; blocked storage never throws.
 * - setStyle runs a switch's `beforeReload` after the cookie is written and just before the
 *   reload, and never for a switch made in place.
 * - useChooseStyle hands it on: after the server stored the pick, and not when it refused it.
 * - The note's Switch to Play leaves the word when it picks the install, and none for builtin Play.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, render, screen } from '@testing-library/react'
import { useEffect } from 'react'

const tone = vi.hoisted(() => ({ theme: 'dark', setTheme: vi.fn() }))
vi.mock('next-themes', () => ({ useTheme: () => tone }))
const ws = vi.hoisted(() => ({ id: 'ws-1' as string | null }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspaceOptional: () => (ws.id === null ? null : { workspace: { id: ws.id } }),
  useWorkspace: () => ({ workspace: ws.id === null ? null : { id: ws.id } }),
}))
const toasts = vi.hoisted(() => ({ error: vi.fn(), success: vi.fn() }))
vi.mock('sonner', () => ({ toast: toasts }))

import { UiStyleProvider, useUiStyle } from '@/contexts/ui-style-context'
import { apiClient, type InstalledTheme } from '@/lib/api-client'
import { PlayOnlyNote, SWITCH_TO_PLAY } from '@/lib/theme-host/builtin/play-only-route'
import { BUILTIN_PLAY_STYLE } from '@/lib/theme-host/builtin/play-entry'
import { forgetChecksInThisDocumentForTests, useChooseStyle } from '@/lib/theme-host/choice-sync'
import {
  SWITCH_FOCUS_STORAGE_KEY,
  rememberHeadingFocusAfterReload,
  takeHeadingFocusAfterReload,
  type SwitchFocusStorage,
} from '@/lib/theme-host/switch-focus'
import { APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION, UI_STYLE_COOKIE, type UiStyle } from '@/lib/ui-style'

const LOOKS_PATH = '/looks'
const PLAY_INSTALL_ID = '44444444-4444-4444-8444-444444444444'
const PLAY_INSTALL_STYLE = `theme:install:${PLAY_INSTALL_ID}` as UiStyle

const PLAY_INSTALL: InstalledTheme = {
  installId: PLAY_INSTALL_ID, themeId: 'automatos/play', name: 'Play', packageId: '22222222-2222-4222-8222-222222222222',
  version: '1.0.0', ownerType: 'first_party', reviewed: true, private: false, isDefault: false,
  pinnedVersion: null, previousPackageId: null, pendingPackageId: null, pendingVersion: null,
  installedAt: null, optIn: { required: false, wording: '', packageId: '' }, revoked: null, needsUpdate: false,
}

const word = () => window.sessionStorage.getItem(SWITCH_FOCUS_STORAGE_KEY)

/** A reload that notes what the tab had kept for the next load, and the cookie, at that moment. */
function recordingReload() {
  const seen: { word: string | null; cookie: string }[] = []
  const reload = vi.fn(() => {
    seen.push({ word: word(), cookie: document.cookie })
  })
  return { reload, seen }
}

/** Lets the PUT's promise and the effects after it run. */
async function settle() {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0))
  })
}

beforeEach(() => {
  forgetChecksInThisDocumentForTests()
  document.cookie = `${UI_STYLE_COOKIE}=; Max-Age=0; Path=/`
  window.localStorage.clear()
  window.localStorage.setItem(APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION)
  window.sessionStorage.clear()
  window.history.replaceState(null, '', LOOKS_PATH)
  ws.id = 'ws-1'
  toasts.error.mockClear()
})
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  window.sessionStorage.clear()
})

describe('the word left for the next load', () => {
  it("is this page's path, taken once: read and cleared together", () => {
    rememberHeadingFocusAfterReload()
    expect(word()).toBe(LOOKS_PATH)
    expect(takeHeadingFocusAfterReload()).toBe(true)
    expect(word()).toBeNull()
    expect(takeHeadingFocusAfterReload()).toBe(false)
  })

  it("another page's word is cleared and ignored", () => {
    rememberHeadingFocusAfterReload('/looks/style')
    expect(takeHeadingFocusAfterReload(LOOKS_PATH)).toBe(false)
    expect(word()).toBeNull()
  })

  it('with no word, or no storage, nothing is asked', () => {
    expect(takeHeadingFocusAfterReload()).toBe(false)
    rememberHeadingFocusAfterReload(LOOKS_PATH, null)
    expect(takeHeadingFocusAfterReload(LOOKS_PATH, null)).toBe(false)
  })

  it('blocked storage is logged, never thrown', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    const blocked: SwitchFocusStorage = {
      getItem: () => { throw new Error('blocked') },
      setItem: () => { throw new Error('blocked') },
      removeItem: () => { throw new Error('blocked') },
    }
    expect(() => rememberHeadingFocusAfterReload(LOOKS_PATH, blocked)).not.toThrow()
    expect(takeHeadingFocusAfterReload(LOOKS_PATH, blocked)).toBe(false)
    expect(warn).toHaveBeenCalledTimes(2)
  })
})

function SetStyleProbe({ beforeReload }: { beforeReload: () => void }) {
  const { setStyle } = useUiStyle()
  return (
    <div>
      <button onClick={() => setStyle(PLAY_INSTALL_STYLE, { beforeReload })}>to the install</button>
      <button onClick={() => setStyle('classic', { beforeReload })}>to classic</button>
    </div>
  )
}

describe('setStyle carries a switch across the reload (PA-13)', () => {
  it('runs beforeReload after the cookie is written and just before the reload', () => {
    const { reload, seen } = recordingReload()
    const atBefore: { cookie: string; reloads: number }[] = []
    const beforeReload = vi.fn(() => {
      atBefore.push({ cookie: document.cookie, reloads: reload.mock.calls.length })
      rememberHeadingFocusAfterReload()
    })
    render(<UiStyleProvider initialStyle="studio" reloadPage={reload}><SetStyleProbe beforeReload={beforeReload} /></UiStyleProvider>)
    act(() => { screen.getByText('to the install').click() })
    expect(atBefore).toEqual([{ cookie: expect.stringContaining(`${UI_STYLE_COOKIE}=${PLAY_INSTALL_STYLE}`), reloads: 0 }])
    expect(seen).toEqual([{ word: LOOKS_PATH, cookie: expect.stringContaining(PLAY_INSTALL_STYLE) }])
  })

  it('never runs it for a switch made in place', () => {
    const reload = vi.fn()
    const beforeReload = vi.fn()
    render(<UiStyleProvider initialStyle="studio" reloadPage={reload}><SetStyleProbe beforeReload={beforeReload} /></UiStyleProvider>)
    act(() => { screen.getByText('to classic').click() })
    expect(beforeReload).not.toHaveBeenCalled()
    expect(reload).not.toHaveBeenCalled()
  })
})

function ChooseProbe({ beforeReload }: { beforeReload: () => void }) {
  const choose = useChooseStyle()
  return <button onClick={() => choose(PLAY_INSTALL_STYLE, { beforeReload })}>pick the install</button>
}

describe('useChooseStyle hands the switch on', () => {
  it('to the reload, once the server stored the pick', async () => {
    const put = vi.spyOn(apiClient, 'putThemeChoice').mockResolvedValue({})
    const { reload } = recordingReload()
    const beforeReload = vi.fn()
    render(<UiStyleProvider initialStyle="studio" reloadPage={reload}><ChooseProbe beforeReload={beforeReload} /></UiStyleProvider>)
    act(() => { screen.getByText('pick the install').click() })
    expect(beforeReload).not.toHaveBeenCalled()
    await settle()
    expect(put).toHaveBeenCalledWith(PLAY_INSTALL_ID)
    expect(beforeReload).toHaveBeenCalledTimes(1)
    expect(reload).toHaveBeenCalledTimes(1)
  })

  it('not when the server refused it: no word, no reload', async () => {
    vi.spyOn(apiClient, 'putThemeChoice').mockRejectedValue(Object.assign(new Error('No.'), { status: 403 }))
    const reload = vi.fn()
    const beforeReload = vi.fn()
    render(<UiStyleProvider initialStyle="studio" reloadPage={reload}><ChooseProbe beforeReload={beforeReload} /></UiStyleProvider>)
    act(() => { screen.getByText('pick the install').click() })
    await settle()
    expect(beforeReload).not.toHaveBeenCalled()
    expect(reload).not.toHaveBeenCalled()
  })
})

/** The workspace's installs, as ThemeChoiceSync hands them to the style context. */
function SeedInstalls({ rows }: { rows: readonly InstalledTheme[] }) {
  const { setInstalledThemes } = useUiStyle()
  useEffect(() => {
    setInstalledThemes(rows)
  }, [rows, setInstalledThemes])
  return null
}

function renderNote(rows: readonly InstalledTheme[], reload: () => void) {
  return render(
    <UiStyleProvider initialStyle="studio" reloadPage={reload}>
      <SeedInstalls rows={rows} />
      <PlayOnlyNote title="Make it yours" note="Make it yours is part of the Play theme." offerSwitch />
    </UiStyleProvider>,
  )
}

describe("the note's Switch to Play (G3)", () => {
  it("onto Play's install leaves word for this page just before the reload", async () => {
    ws.id = null
    const { reload, seen } = recordingReload()
    renderNote([PLAY_INSTALL], reload)
    await settle()
    act(() => { screen.getByRole('button', { name: SWITCH_TO_PLAY }).click() })
    expect(seen).toEqual([{ word: LOOKS_PATH, cookie: expect.stringContaining(PLAY_INSTALL_STYLE) }])
  })

  it('onto builtin Play switches in place and leaves no word', async () => {
    ws.id = null
    const reload = vi.fn()
    renderNote([], reload)
    await settle()
    act(() => { screen.getByRole('button', { name: SWITCH_TO_PLAY }).click() })
    expect(reload).not.toHaveBeenCalled()
    expect(document.cookie).toContain(`${UI_STYLE_COOKIE}=${BUILTIN_PLAY_STYLE}`)
    expect(word()).toBeNull()
  })
})

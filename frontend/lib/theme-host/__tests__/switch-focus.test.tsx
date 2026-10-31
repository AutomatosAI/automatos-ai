/**
 * PRD-261 G3 (P260-FX-22, M21) — a switch onto an installed theme reloads the page (PA-13), so the
 * switch can leave word in this tab's sessionStorage just before the reload, and the next load
 * takes it once: the page's heading then takes the focus that dropped to <body>.
 *
 * - The word: the page's path under one key, read and cleared together; another path's word is
 *   cleared and ignored; blocked storage never throws.
 * - setStyle runs a switch's `beforeReload` after the cookie is written and just before the
 *   reload, and never for a switch made in place.
 * - useChooseStyle hands it on: after the server stored the pick, not when it refused it, and
 *   never for a pick applied in place (a Classic pick).
 * - The next load of /looks (ThemeSlotPage, as app/looks/page.tsx draws it with
 *   THEME_HEADING_SELECTOR): the installed theme's screen heading takes the focus, or the core's
 *   note (components/theme-host/no-look-editor-note.tsx, the same mark on its region) when the
 *   theme has no screen for the page; with no word, focus stays where it is.
 *   components/theme-host/__tests__/theme-slot-page-switch.test.tsx holds the late-loading screen.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, render, screen } from '@testing-library/react'
import type { ComponentType, ReactNode } from 'react'

const tone = vi.hoisted(() => ({ theme: 'dark', setTheme: vi.fn() }))
vi.mock('next-themes', () => ({ useTheme: () => tone }))
const ws = vi.hoisted(() => ({ id: 'ws-1' as string | null }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspaceOptional: () => (ws.id === null ? null : { workspace: { id: ws.id } }),
  useWorkspace: () => ({ workspace: ws.id === null ? null : { id: ws.id } }),
}))
const toasts = vi.hoisted(() => ({ error: vi.fn(), success: vi.fn() }))
vi.mock('sonner', () => ({ toast: toasts }))
/** The theme's frame around the page: its content alone here. */
vi.mock('@/components/layout/main-layout', () => ({
  MainLayout: ({ children }: { readonly children?: ReactNode }) => <div data-testid="main-layout">{children}</div>,
}))

import { NoLookEditorNote, LOOKS_NOTE, LOOKS_TITLE, THEME_HEADING_ATTRIBUTE, THEME_HEADING_SELECTOR, WIDER_SCREEN_NOTE } from '@/components/theme-host/no-look-editor-note'
import { ThemeHostContext, type ThemeHostContextValue } from '@/components/theme-host/theme-host-context'
import { ThemeSlotPage } from '@/components/theme-host/theme-slot-page'
import { UiStyleProvider, useUiStyle } from '@/contexts/ui-style-context'
import { apiClient } from '@/lib/api-client'
import { forgetChecksInThisDocumentForTests, useChooseStyle } from '@/lib/theme-host/choice-sync'
import type { SlotId } from '@/lib/theme-host/registry-types'
import {
  SWITCH_FOCUS_STORAGE_KEY,
  rememberHeadingFocusAfterReload,
  takeHeadingFocusAfterReload,
  type SwitchFocusStorage,
} from '@/lib/theme-host/switch-focus'
import { APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION, UI_STYLE_COOKIE, type UiStyle } from '@/lib/ui-style'

import { fixtureThemeHost } from './fixture-theme-entry'

const LOOKS_PATH = '/looks'
const LOOK_EDITOR_SLOT: SlotId = 'lookEditor@1'
const INSTALL_ID = '44444444-4444-4444-8444-444444444444'
const INSTALL_STYLE = `theme:install:${INSTALL_ID}` as UiStyle
/** The installed theme's look editor heading. */
const SCREEN_HEADING = 'Make it yours.'
/** The mark and the focusability a theme's screen puts on its h1, as the core's note does on its region. */
const SCREEN_HEADING_MARK = { [THEME_HEADING_ATTRIBUTE]: true, tabIndex: -1 } as const

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
      <button onClick={() => setStyle(INSTALL_STYLE, { beforeReload })}>to the install</button>
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
    expect(atBefore).toEqual([{ cookie: expect.stringContaining(`${UI_STYLE_COOKIE}=${INSTALL_STYLE}`), reloads: 0 }])
    expect(seen).toEqual([{ word: LOOKS_PATH, cookie: expect.stringContaining(INSTALL_STYLE) }])
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
  return (
    <div>
      <button onClick={() => choose(INSTALL_STYLE, { beforeReload })}>pick the install</button>
      <button onClick={() => choose('classic', { beforeReload })}>pick classic</button>
    </div>
  )
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
    expect(put).toHaveBeenCalledWith(INSTALL_ID)
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

  it('never for a pick applied in place (Classic): no word, no reload', async () => {
    const put = vi.spyOn(apiClient, 'putThemeChoice').mockResolvedValue({})
    const reload = vi.fn()
    const beforeReload = vi.fn(() => rememberHeadingFocusAfterReload())
    render(<UiStyleProvider initialStyle="studio" reloadPage={reload}><ChooseProbe beforeReload={beforeReload} /></UiStyleProvider>)
    act(() => { screen.getByText('pick classic').click() })
    await settle()
    expect(put).toHaveBeenCalledWith('classic')
    expect(beforeReload).not.toHaveBeenCalled()
    expect(reload).not.toHaveBeenCalled()
    expect(word()).toBeNull()
  })
})

/** The installed theme's look editor: its h1 carries the heading mark. */
function ThemeLookEditor() {
  return <h1 {...SCREEN_HEADING_MARK}>{SCREEN_HEADING}</h1>
}

/** The theme host for an installed theme, its look editor registered or (null) none. */
function hostWith(lookEditor: ComponentType | null): ThemeHostContextValue {
  const registered: Array<[SlotId, unknown]> = lookEditor === null ? [] : [[LOOK_EDITOR_SLOT, lookEditor]]
  return fixtureThemeHost('active', { slots: new Map<SlotId, unknown>(registered) })
}

/** /looks on the next load, as app/looks/page.tsx draws it, under the theme host's value. */
function looksPage(host: ThemeHostContextValue) {
  return render(
    <ThemeHostContext.Provider value={host}>
      <ThemeSlotPage
        slot={LOOK_EDITOR_SLOT}
        note={<NoLookEditorNote title={LOOKS_TITLE} note={LOOKS_NOTE} />}
        narrowNote={<NoLookEditorNote title={LOOKS_TITLE} note={WIDER_SCREEN_NOTE} />}
        headingSelector={THEME_HEADING_SELECTOR}
      />
    </ThemeHostContext.Provider>,
  )
}

const markedHeading = () => document.querySelector<HTMLElement>(THEME_HEADING_SELECTOR)

describe('the next load hands focus to the heading the switch left word for (G3)', () => {
  it("the installed theme's screen heading, after a switch from this page reloaded into it", () => {
    const { reload, seen } = recordingReload()
    const beforeReload = () => rememberHeadingFocusAfterReload()
    render(<UiStyleProvider initialStyle="studio" reloadPage={reload}><SetStyleProbe beforeReload={beforeReload} /></UiStyleProvider>)
    act(() => { screen.getByText('to the install').click() })
    expect(seen).toEqual([{ word: LOOKS_PATH, cookie: expect.stringContaining(INSTALL_STYLE) }])
    cleanup()

    looksPage(hostWith(ThemeLookEditor))
    expect(screen.getByRole('heading', { level: 1, name: SCREEN_HEADING })).toHaveFocus()
    expect(markedHeading()).toBe(screen.getByRole('heading', { level: 1, name: SCREEN_HEADING }))
    expect(word()).toBeNull()
  })

  it("the core's note when the theme switched to has no screen for the page: its region carries the mark", () => {
    rememberHeadingFocusAfterReload(LOOKS_PATH)
    looksPage(hostWith(null))
    const note = markedHeading()
    expect(note).toHaveAttribute('aria-label', LOOKS_TITLE)
    expect(note).toHaveTextContent(LOOKS_NOTE)
    expect(note).toHaveFocus()
    expect(word()).toBeNull()
  })

  it('a load with no word leaves focus where it is, on the note and on the screen', () => {
    looksPage(hostWith(null))
    expect(markedHeading()).not.toHaveFocus()
    cleanup()
    looksPage(hostWith(ThemeLookEditor))
    expect(markedHeading()).not.toHaveFocus()
    expect(document.activeElement).toBe(document.body)
  })
})

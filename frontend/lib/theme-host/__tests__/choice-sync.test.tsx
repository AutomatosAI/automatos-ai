/**
 * PRD-261 A-02c (PA-12, R16) — the server holds the theme choice; the style cookie caches it.
 *
 * - A pick stores the choice ('classic', 'studio', an install by its id); Classic and Studio
 *   apply at once (undone if the PUT fails), an installed one after it.
 * - The check: one GET /api/themes/active per workspace and tab; a differing server choice moves
 *   the tab, an installed one with one reload at most (sessionStorage guard; none without it). A
 *   server choice this build does not know (no theme is compiled in, A-19f1) moves nothing.
 * - The appearance-defaults step never overrides a server choice, and never stores one.
 * - A server Studio moves a tab only off an installed theme (off Classic it may mean "no
 *   choice"); safe mode asks nothing; a pick made while the check is in flight wins, and so does
 *   the host's revert to Studio (A-04c), so its answer never reloads back into the failing theme.
 * - The entries: installs join as theme:install:<id>, 'Private · not reviewed' when unreviewed;
 *   Play's package is an install like any other, and a withdrawn install is not offered.
 * - M1: the installs are fetched only once the menu or ⌘K wants them, never in safe mode. M2: with
 *   no stored server choice (`source` default or fallback), a tab on its own style (not Studio)
 *   stores it instead of following the workspace default.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactNode } from 'react'

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
import {
  CHOICE_CHECK_KEY_PREFIX,
  CHOICE_NOT_SAVED,
  ThemeChoiceSync,
  choiceOfStyle,
  forgetChecksInThisDocumentForTests,
  reconcileTarget,
  storesTabStyle,
  styleChoices,
  styleOfChoice,
  themeChoiceEntries,
  useChooseStyle,
  wantInstalledThemes,
} from '@/lib/theme-host/choice-sync'
import { SAFE_MODE_PARAM, SAFE_MODE_VALUE } from '@/lib/theme-host/constants'
import { APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION, UI_STYLE_COOKIE, type UiStyle } from '@/lib/ui-style'

const INSTALL_ID = '11111111-1111-4111-8111-111111111111'
const OTHER_INSTALL_ID = '44444444-4444-4444-8444-444444444444'
const INSTALLED_STYLE = `theme:install:${INSTALL_ID}` as UiStyle
const OTHER_INSTALLED_STYLE = `theme:install:${OTHER_INSTALL_ID}` as UiStyle
const PRIVATE_NOT_REVIEWED = 'Private · not reviewed'
/** A server choice this build does not know: no theme is compiled into the app (A-19f1). */
const UNKNOWN_CHOICE = 'builtin:nope'

function installRow(overrides: Partial<InstalledTheme> = {}): InstalledTheme {
  return {
    installId: INSTALL_ID, themeId: 'acme/garden', name: 'Garden', packageId: '22222222-2222-4222-8222-222222222222',
    version: '1.0.0', ownerType: 'workspace', reviewed: false, private: true, isDefault: false,
    pinnedVersion: null, previousPackageId: null, pendingPackageId: null, pendingVersion: null,
    installedAt: null, optIn: { required: false, wording: '', packageId: '' }, revoked: null, needsUpdate: false,
    signature: { state: 'unsigned', keyId: null }, ...overrides,
  }
}

/** The Design Kit's Play package, installed in the workspace. */
const INSTALLED_PLAY = installRow({ installId: OTHER_INSTALL_ID, themeId: 'automatos/play', name: 'Play', reviewed: true })

function activeAnswer(choice: string, theme: unknown = null) {
  return { choice, theme }
}

/** The answer of a backend that says where the choice came from (A-07b `source`). */
function sourcedAnswer(choice: string, source: 'choice' | 'default' | 'fallback', theme: unknown = null) {
  return { choice, source, theme }
}

const seenDefaults = () => window.localStorage.setItem(APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION)
const checkRecord = (workspaceId: string) => window.sessionStorage.getItem(`${CHOICE_CHECK_KEY_PREFIX}${workspaceId}`)

function Probe() {
  const { style } = useUiStyle()
  const choose = useChooseStyle()
  return (
    <div>
      <span data-testid="style">{style}</span>
      <button onClick={() => choose('classic')}>pick classic</button>
      <button onClick={() => choose('studio')}>pick studio</button>
      <button onClick={() => choose(INSTALLED_STYLE)}>pick garden</button>
    </div>
  )
}

function mount(initialStyle: UiStyle, reload: () => void, extra?: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <UiStyleProvider initialStyle={initialStyle} reloadPage={reload}>
        <ThemeChoiceSync />
        <Probe />
        {extra}
      </UiStyleProvider>
    </QueryClientProvider>,
  )
}

/** Lets the GET's promise and the effects after it run. */
async function settle() {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0))
  })
}

const styleShown = () => screen.getByTestId('style').textContent

beforeEach(() => {
  forgetChecksInThisDocumentForTests()
  document.cookie = `${UI_STYLE_COOKIE}=; Max-Age=0; Path=/`
  window.localStorage.clear()
  window.sessionStorage.clear()
  window.history.replaceState(null, '', '/agents')
  ws.id = 'ws-1'
  tone.setTheme.mockClear()
  toasts.error.mockClear()
  vi.spyOn(apiClient, 'getInstalledThemes').mockResolvedValue({ installs: [] })
})
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('choice ↔ style', () => {
  it('sends an install by its id, Classic and Studio as themselves', () => {
    expect(choiceOfStyle(INSTALLED_STYLE)).toBe(INSTALL_ID)
    expect(choiceOfStyle('classic')).toBe('classic')
    expect(choiceOfStyle('studio')).toBe('studio')
  })

  it("reads the server's choices back, and none it does not know", () => {
    expect(styleOfChoice(INSTALL_ID)).toBe(INSTALLED_STYLE)
    expect(styleOfChoice('classic')).toBe('classic')
    expect(styleOfChoice('studio')).toBe('studio')
    for (const unknown of [UNKNOWN_CHOICE, 'acme-unknown', 'NOT-A-UUID', 'ABCDEF01-1111-4111-8111-111111111111', '']) {
      expect(styleOfChoice(unknown)).toBeNull()
    }
    expect(styleOfChoice(null)).toBeNull()
  })

  it('moves to a differing server choice; a server Studio only off an installed theme; an unknown one never', () => {
    expect(reconcileTarget('studio', activeAnswer('classic'))).toBe('classic')
    expect(reconcileTarget('classic', activeAnswer(INSTALL_ID, {}))).toBe(INSTALLED_STYLE)
    expect(reconcileTarget(INSTALLED_STYLE, activeAnswer(OTHER_INSTALL_ID, {}))).toBe(OTHER_INSTALLED_STYLE)
    expect(reconcileTarget('classic', activeAnswer('classic'))).toBeNull()
    expect(reconcileTarget(INSTALLED_STYLE, activeAnswer(INSTALL_ID, {}))).toBeNull()
    expect(reconcileTarget('classic', activeAnswer('studio'))).toBeNull()
    expect(reconcileTarget(INSTALLED_STYLE, activeAnswer('studio'))).toBe('studio')
    for (const current of ['classic', 'studio', INSTALLED_STYLE] as const) {
      expect(reconcileTarget(current, activeAnswer(UNKNOWN_CHOICE))).toBeNull()
    }
    expect(reconcileTarget('classic', { choice: 42 })).toBeNull()
    expect(reconcileTarget('classic', null)).toBeNull()
  })
})

describe('the entries the menu and ⌘K offer', () => {
  it('lists an install as theme:install:<id>, "Private · not reviewed" when nobody reviewed it', () => {
    const entries = themeChoiceEntries([installRow(), installRow({ installId: OTHER_INSTALL_ID, name: 'Dune', reviewed: true })])
    expect(entries.map((entry) => [entry.style, entry.name, entry.kind, entry.note])).toEqual([
      [INSTALLED_STYLE, 'Garden', 'installed', PRIVATE_NOT_REVIEWED],
      [OTHER_INSTALLED_STYLE, 'Dune', 'installed', null],
    ])
  })

  it('offers no withdrawn install and none with a malformed id; a nameless one by its theme id', () => {
    const withdrawn = installRow({ revoked: { level: 'disable', reason: 'withdrawn' } })
    const malformed = installRow({ installId: 'not-a-uuid' })
    const nameless = installRow({ installId: OTHER_INSTALL_ID, name: null })
    expect(themeChoiceEntries([withdrawn, malformed, nameless]).map((entry) => entry.name)).toEqual(['acme/garden'])
  })

  it("lists Play's package and another publisher's play as installs like any other, and no withdrawn one", () => {
    const lookalike = installRow({ themeId: 'someone/play', name: 'Play?' })
    const withdrawn = { ...INSTALLED_PLAY, revoked: { level: 'disable' as const, reason: 'withdrawn' } }
    expect(themeChoiceEntries([INSTALLED_PLAY, lookalike, withdrawn]).map((entry) => [entry.style, entry.name, entry.kind])).toEqual([
      [OTHER_INSTALLED_STYLE, 'Play', 'installed'],
      [INSTALLED_STYLE, 'Play?', 'installed'],
    ])
  })

  it('⌘K adds Classic and Studio before the themes', () => {
    expect(styleChoices([]).map((entry) => entry.name)).toEqual(['Classic', 'Studio'])
    expect(styleChoices([installRow()]).map((entry) => [entry.style, entry.name, entry.kind])).toEqual([
      ['classic', 'Classic', 'core'],
      ['studio', 'Studio', 'core'],
      [INSTALLED_STYLE, 'Garden', 'installed'],
    ])
  })
})

describe('the server choice check (once per workspace and tab)', () => {
  it('moves the tab to a differing Classic choice in place, and records the check', async () => {
    seenDefaults()
    const reload = vi.fn()
    const active = vi.spyOn(apiClient, 'getActiveTheme').mockResolvedValue(activeAnswer('classic'))
    mount('studio', reload)
    await settle()
    expect(styleShown()).toBe('classic')
    expect(document.cookie).toContain(`${UI_STYLE_COOKIE}=classic`)
    expect(reload).not.toHaveBeenCalled()
    expect(active).toHaveBeenCalledTimes(1)
    expect(checkRecord('ws-1')).toBe('checked')
  })

  it('a server choice this build does not know moves nothing, and records the check', async () => {
    seenDefaults()
    vi.spyOn(apiClient, 'getActiveTheme').mockResolvedValue(activeAnswer(UNKNOWN_CHOICE))
    for (const style of ['classic', INSTALLED_STYLE] as const) {
      forgetChecksInThisDocumentForTests()
      window.sessionStorage.clear()
      const reload = vi.fn()
      mount(style, reload)
      await settle()
      expect(styleShown()).toBe(style)
      expect(reload).not.toHaveBeenCalled()
      expect(checkRecord('ws-1')).toBe('checked')
      cleanup()
    }
    expect(document.cookie).not.toContain(`${UI_STYLE_COOKIE}=`)
  })

  it('reloads into an installed server choice once, and never again in the tab', async () => {
    seenDefaults()
    const reload = vi.fn()
    const active = vi.spyOn(apiClient, 'getActiveTheme').mockResolvedValue(activeAnswer(INSTALL_ID, {}))
    mount('studio', reload)
    await settle()
    expect(reload).toHaveBeenCalledTimes(1)
    expect(document.cookie).toContain(`${UI_STYLE_COOKIE}=${INSTALLED_STYLE}`)
    expect(checkRecord('ws-1')).toBe('reloaded')

    // The next document in the tab, even one where the cookie did not stick: no GET, no reload.
    cleanup()
    forgetChecksInThisDocumentForTests()
    mount('studio', reload)
    await settle()
    expect(active).toHaveBeenCalledTimes(1)
    expect(reload).toHaveBeenCalledTimes(1)
  })

  it('without sessionStorage it writes the cookie but never reloads (no guard, no loop)', async () => {
    seenDefaults()
    const session = window.sessionStorage
    const realSetItem = Storage.prototype.setItem
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(function (this: Storage, key: string, value: string) {
      if (this === session) throw new Error('blocked')
      realSetItem.call(this, key, value)
    })
    const reload = vi.fn()
    vi.spyOn(apiClient, 'getActiveTheme').mockResolvedValue(activeAnswer(INSTALL_ID, {}))
    mount('studio', reload)
    await settle()
    expect(reload).not.toHaveBeenCalled()
    expect(document.cookie).toContain(`${UI_STYLE_COOKIE}=${INSTALLED_STYLE}`)
    expect(styleShown()).toBe('studio')
  })

  it('asks once per workspace and tab: a remount does not ask again, another workspace does', async () => {
    seenDefaults()
    const active = vi.spyOn(apiClient, 'getActiveTheme').mockResolvedValue(activeAnswer('studio'))
    mount('studio', vi.fn())
    await settle()
    cleanup()
    mount('studio', vi.fn())
    await settle()
    expect(active).toHaveBeenCalledTimes(1)
    cleanup()
    ws.id = 'ws-2'
    mount('studio', vi.fn())
    await settle()
    expect(active).toHaveBeenCalledTimes(2)
  })

  it('the defaults step never overrides a server choice, and stores none', async () => {
    // A browser that has not seen the defaults: the step moves it to Studio + Dark at mount.
    const put = vi.spyOn(apiClient, 'putThemeChoice').mockResolvedValue({})
    vi.spyOn(apiClient, 'getActiveTheme').mockResolvedValue(activeAnswer('classic'))
    mount('classic', vi.fn())
    await settle()
    expect(tone.setTheme).toHaveBeenCalledWith('dark')
    expect(styleShown()).toBe('classic')
    expect(document.cookie).toContain(`${UI_STYLE_COOKIE}=classic`)
    expect(put).not.toHaveBeenCalled()
  })

  it('a server Studio leaves a Classic tab on Classic (a pick made before PA-12)', async () => {
    seenDefaults()
    vi.spyOn(apiClient, 'getActiveTheme').mockResolvedValue(activeAnswer('studio'))
    mount('classic', vi.fn())
    await settle()
    expect(styleShown()).toBe('classic')
    expect(checkRecord('ws-1')).toBe('checked')
  })

  it('asks nothing in safe mode, or outside a workspace', async () => {
    seenDefaults()
    const active = vi.spyOn(apiClient, 'getActiveTheme').mockResolvedValue(activeAnswer('classic'))
    window.history.replaceState(null, '', `/agents?${SAFE_MODE_PARAM}=${SAFE_MODE_VALUE}`)
    mount('studio', vi.fn())
    await settle()
    cleanup()
    window.sessionStorage.clear()
    window.history.replaceState(null, '', '/agents')
    ws.id = null
    mount('studio', vi.fn())
    await settle()
    expect(active).not.toHaveBeenCalled()
  })

  it('a pick made while the check is in flight wins over its answer', async () => {
    seenDefaults()
    let answer: (value: unknown) => void = () => {}
    vi.spyOn(apiClient, 'getActiveTheme').mockReturnValue(new Promise((resolve) => { answer = resolve }))
    vi.spyOn(apiClient, 'putThemeChoice').mockResolvedValue({})
    mount('classic', vi.fn())
    await settle()
    act(() => { screen.getByText('pick studio').click() })
    await act(async () => { answer(activeAnswer('classic')) })
    await settle()
    expect(styleShown()).toBe('studio')
  })

  it("drops an answer that lands after the host's revert to Studio, which is no pick (A-04c)", async () => {
    seenDefaults()
    let answer: (value: unknown) => void = () => {}
    vi.spyOn(apiClient, 'getActiveTheme').mockReturnValue(new Promise((resolve) => { answer = resolve }))
    const reload = vi.fn()
    function Revert() {
      const { setStyle } = useUiStyle()
      return <button onClick={() => setStyle('studio')}>revert</button>
    }
    mount(INSTALLED_STYLE, reload, <Revert />)
    await settle()
    act(() => { screen.getByText('revert').click() })
    await act(async () => { answer(activeAnswer(INSTALL_ID)) })
    await settle()
    expect(styleShown()).toBe('studio')
    expect(reload).not.toHaveBeenCalled()
  })

  it('a failed check changes nothing', async () => {
    seenDefaults()
    vi.spyOn(console, 'warn').mockImplementation(() => {})
    vi.spyOn(apiClient, 'getActiveTheme').mockRejectedValue(Object.assign(new Error('Not Found'), { status: 404 }))
    mount(INSTALLED_STYLE, vi.fn())
    await settle()
    expect(styleShown()).toBe(INSTALLED_STYLE)
    expect(checkRecord('ws-1')).toBeNull()
  })

  it('hands the workspace’s installs to the style context, fetched only once wanted (M1)', async () => {
    seenDefaults()
    vi.spyOn(apiClient, 'getActiveTheme').mockResolvedValue(activeAnswer('studio'))
    const installed = vi.spyOn(apiClient, 'getInstalledThemes').mockResolvedValue({ installs: [installRow()] })
    function Installed() {
      return <span data-testid="installed">{useUiStyle().installedThemes.map((row) => row.name).join(',')}</span>
    }
    mount('studio', vi.fn(), <Installed />)
    await settle()
    expect(installed).not.toHaveBeenCalled()
    expect(screen.getByTestId('installed').textContent).toBe('')
    act(() => wantInstalledThemes())
    await waitFor(() => expect(screen.getByTestId('installed').textContent).toBe('Garden'))
    expect(installed).toHaveBeenCalledTimes(1)
  })

  it('never fetches the installs in safe mode, even when wanted', async () => {
    seenDefaults()
    const installed = vi.spyOn(apiClient, 'getInstalledThemes').mockResolvedValue({ installs: [installRow()] })
    window.history.replaceState(null, '', `/agents?${SAFE_MODE_PARAM}=${SAFE_MODE_VALUE}`)
    mount('studio', vi.fn())
    act(() => wantInstalledThemes())
    await settle()
    expect(installed).not.toHaveBeenCalled()
  })
})

describe('a cookie-only style with no stored server choice (M2)', () => {
  it('stores the tab style only when the server holds no choice and the tab is on its own style', () => {
    expect(storesTabStyle('classic', sourcedAnswer(INSTALL_ID, 'default'))).toBe(true)
    expect(storesTabStyle('classic', sourcedAnswer('studio', 'fallback'))).toBe(true)
    expect(storesTabStyle('classic', sourcedAnswer(UNKNOWN_CHOICE, 'default'))).toBe(true)   // a default this build does not know
    expect(storesTabStyle(INSTALLED_STYLE, sourcedAnswer(OTHER_INSTALL_ID, 'default'))).toBe(true)
    expect(storesTabStyle('studio', sourcedAnswer(INSTALL_ID, 'default'))).toBe(false)     // the build default follows
    expect(storesTabStyle('classic', sourcedAnswer(INSTALL_ID, 'choice'))).toBe(false)     // the member's own choice wins
    expect(storesTabStyle(INSTALLED_STYLE, sourcedAnswer(INSTALL_ID, 'default'))).toBe(false) // already the default
    expect(storesTabStyle('classic', activeAnswer(INSTALL_ID))).toBe(false)                // a backend with no source
  })

  it('a workspace default does not override a Classic tab: Classic is stored, no reload', async () => {
    seenDefaults()
    vi.spyOn(apiClient, 'getActiveTheme').mockResolvedValue(sourcedAnswer(INSTALL_ID, 'default'))
    const put = vi.spyOn(apiClient, 'putThemeChoice').mockResolvedValue({})
    const reload = vi.fn()
    mount('classic', reload)
    await settle()
    expect(put).toHaveBeenCalledWith('classic')
    expect(styleShown()).toBe('classic')
    expect(reload).not.toHaveBeenCalled()
    expect(checkRecord('ws-1')).toBe('checked')
  })

  it("an installed tab with no stored choice stores its install, and is not reloaded off it", async () => {
    seenDefaults()
    vi.spyOn(apiClient, 'getActiveTheme').mockResolvedValue(sourcedAnswer('studio', 'fallback'))
    const put = vi.spyOn(apiClient, 'putThemeChoice').mockResolvedValue({})
    mount(INSTALLED_STYLE, vi.fn())
    await settle()
    expect(put).toHaveBeenCalledWith(INSTALL_ID)
    expect(styleShown()).toBe(INSTALLED_STYLE)
  })

  it('a tab on Studio follows the workspace default and stores nothing', async () => {
    seenDefaults()
    vi.spyOn(apiClient, 'getActiveTheme').mockResolvedValue(sourcedAnswer(INSTALL_ID, 'default'))
    const put = vi.spyOn(apiClient, 'putThemeChoice').mockResolvedValue({})
    const reload = vi.fn()
    mount('studio', reload)
    await settle()
    expect(put).not.toHaveBeenCalled()
    expect(reload).toHaveBeenCalledTimes(1)
  })

  it('a refused store (members may not choose) follows the server as before', async () => {
    seenDefaults()
    vi.spyOn(console, 'warn').mockImplementation(() => {})
    vi.spyOn(apiClient, 'getActiveTheme').mockResolvedValue(sourcedAnswer('classic', 'default'))
    const put = vi.spyOn(apiClient, 'putThemeChoice').mockRejectedValue(Object.assign(new Error('Forbidden'), { status: 403 }))
    const reload = vi.fn()
    mount(INSTALLED_STYLE, reload)
    await settle()
    await settle()
    expect(put).toHaveBeenCalledWith(INSTALL_ID)
    expect(styleShown()).toBe('classic')
    expect(reload).not.toHaveBeenCalled()
  })

  it("the member's own stored choice is followed, never overwritten", async () => {
    seenDefaults()
    vi.spyOn(apiClient, 'getActiveTheme').mockResolvedValue(sourcedAnswer('classic', 'choice'))
    const put = vi.spyOn(apiClient, 'putThemeChoice').mockResolvedValue({})
    mount(INSTALLED_STYLE, vi.fn())
    await settle()
    expect(put).not.toHaveBeenCalled()
    expect(styleShown()).toBe('classic')
  })
})

describe('choosing stores the choice on the server', () => {
  it('selecting Classic or Studio applies it at once and sends it as itself', async () => {
    seenDefaults()
    vi.spyOn(apiClient, 'getActiveTheme').mockResolvedValue(activeAnswer('studio'))
    const put = vi.spyOn(apiClient, 'putThemeChoice').mockResolvedValue({})
    mount('studio', vi.fn())
    await settle()
    act(() => { screen.getByText('pick classic').click() })
    expect(styleShown()).toBe('classic')
    await settle()
    expect(put).toHaveBeenLastCalledWith('classic')
    act(() => { screen.getByText('pick studio').click() })
    expect(styleShown()).toBe('studio')
    await settle()
    expect(put).toHaveBeenLastCalledWith('studio')
    expect(put).toHaveBeenCalledTimes(2)
  })

  it('an installed theme is stored first, then the page reloads into it', async () => {
    seenDefaults()
    vi.spyOn(apiClient, 'getActiveTheme').mockResolvedValue(activeAnswer('studio'))
    let stored: (value: unknown) => void = () => {}
    const put = vi.spyOn(apiClient, 'putThemeChoice').mockReturnValue(new Promise((resolve) => { stored = resolve }))
    const reload = vi.fn()
    mount('studio', reload)
    await settle()
    act(() => { screen.getByText('pick garden').click() })
    expect(put).toHaveBeenCalledWith(INSTALL_ID)
    expect(reload).not.toHaveBeenCalled()
    await act(async () => { stored({}) })
    await settle()
    expect(reload).toHaveBeenCalledTimes(1)
    expect(document.cookie).toContain(`${UI_STYLE_COOKIE}=${INSTALLED_STYLE}`)
  })

  it("a refusal shows the server's reason and does not reload", async () => {
    seenDefaults()
    vi.spyOn(apiClient, 'getActiveTheme').mockResolvedValue(activeAnswer('studio'))
    const refusal = Object.assign(new Error('Members of this workspace may not choose a theme.'), { status: 403 })
    vi.spyOn(apiClient, 'putThemeChoice').mockRejectedValue(refusal)
    const reload = vi.fn()
    mount('studio', reload)
    await settle()
    act(() => { screen.getByText('pick garden').click() })
    await settle()
    expect(reload).not.toHaveBeenCalled()
    expect(toasts.error).toHaveBeenCalledWith(CHOICE_NOT_SAVED, { description: 'Members of this workspace may not choose a theme.' })
  })

  it('a refused Classic pick puts back the style and cookie the tab had', async () => {
    seenDefaults()
    vi.spyOn(apiClient, 'getActiveTheme').mockResolvedValue(activeAnswer('studio'))
    vi.spyOn(apiClient, 'putThemeChoice').mockRejectedValue(Object.assign(new Error('No.'), { status: 403 }))
    mount('studio', vi.fn())
    await settle()
    act(() => { screen.getByText('pick classic').click() })
    expect(styleShown()).toBe('classic')
    await settle()
    expect(styleShown()).toBe('studio')
    expect(document.cookie).toContain(`${UI_STYLE_COOKIE}=studio`)
  })

  it('outside a workspace a pick only applies the style', async () => {
    seenDefaults()
    ws.id = null
    const put = vi.spyOn(apiClient, 'putThemeChoice').mockResolvedValue({})
    mount('studio', vi.fn())
    act(() => { screen.getByText('pick classic').click() })
    await settle()
    expect(styleShown()).toBe('classic')
    expect(put).not.toHaveBeenCalled()
  })
})

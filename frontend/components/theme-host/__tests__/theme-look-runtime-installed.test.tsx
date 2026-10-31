/**
 * PRD-261 W-A6 — an installed theme keeps its first paint after hydration. While the host loads or
 * runs it, the look runtime leaves <html> as the root layout's boot painted it from the theme's
 * prepaint table, also when no look runtime mounts for it (its boot carried no entry); a failed
 * load and safe mode take that paint off, as Studio shows instead.
 *
 * PRD-261 A-19d2, A-19f1 — an installed theme with looks runs on the generic look runtime
 * (lib/theme-host/looks), from its own boot on the page: the SDK's looks area is available, the look
 * paints from the theme's table, <html> gets data-theme-motion and --theme-motion, and the look's tone
 * follows the table's tone rule into `automatos-theme-tone` and forcedTheme. Any installed theme with
 * looks gets it, whatever its class; here hello-a09b (lib/theme-host/looks/__tests__/hello-looks.ts).
 *
 * PRD-261 A-08c — a page that got the base boot (the server's boot fetch missed its 300 ms or was
 * refused) asks GET /api/themes/boot itself, once for the style context and the look runtime both,
 * and the look paints from the answer. Every other test here answers that request with the base boot
 * (no theme to run), so none of them reaches the network.
 */
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/lib/auth-hooks', () => ({
  useUser: () => ({ isLoaded: true, isSignedIn: true, user: { id: 'user_a' } }),
}))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspaceOptional: () => ({ workspace: { id: 'ws-1' }, isLoading: false, error: null, refreshWorkspace: async () => undefined }),
}))

import { useUiStyle } from '@/contexts/ui-style-context'
import { apiClient } from '@/lib/api-client'
import { runThemeBoot } from '@/lib/theme-host/__tests__/boot-harness'
import { baseThemeBoot } from '@/lib/theme-host/boot'
import { HELLO_PACKAGE_ENTRY, INSTALLED_HELLO_STYLE } from '@/lib/theme-host/__tests__/installed-entry-fixture'
import { installedThemeEntry } from '@/lib/theme-host/installed-entry'
import { HELLO_ENTRY, HELLO_RULES, helloThemeBoot, PAPER, putPageBoot } from '@/lib/theme-host/looks/__tests__/hello-looks'
import { clearRoot, cookieValue, dropCookies, htmlTone, lookNamesLeft, motionMarks, rootVars } from '@/lib/theme-host/looks/__tests__/look-test-helpers'
import { paint } from '@/lib/theme-host/looks/paint'
import { THEME_TONE_COOKIE, themeToneCookieValue } from '@/lib/theme-host/looks/tone-cookie'
import { installIdOfStyle } from '@/lib/theme-host/registry'
import type { ThemeEntry } from '@/lib/theme-host/registry-types'
import type { ThemeHostStatus } from '@/lib/theme-host/types'
import { useLook } from '@/lib/theme-sdk/looks'
import { APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION, UI_STYLE_COOKIE, type UiStyle } from '@/lib/ui-style'

import { HELLO_LOOKS_STYLE, HELLO_LOOKS_THEME, LookAppearance, USER_DEFAULT_TONE } from './look-runtime-harness'

/** An installed theme's style whose boot carried no entry: no look runtime mounts for it. */
const INSTALLED_STYLE: UiStyle = 'theme:install:0b9a3c3e-6a1d-4f9e-9a43-0c1f2d3e4f50'
const TABLE = HELLO_RULES.table
/** A motion mark no theme-neutral runtime sets (Play's own name): the generic runtime never puts it on <html>. */
const NOT_THEME_NEUTRAL_MOTION_ATTRIBUTE = 'data-pl-motion'
const NO_MOTION_MARKS = { attribute: null, variable: '' }
/** The cookies these tests may leave: the look's tone, and the style a switch in place writes. */
const COOKIES_SET_HERE: readonly string[] = [THEME_TONE_COOKIE, UI_STYLE_COOKIE]

function Probe() {
  const { available, tone } = useLook()
  return (
    <div>
      <p>Looks: {available ? 'available' : 'none'}</p>
      <p>Tone: {tone ?? 'not forced'}</p>
    </div>
  )
}

/** The style menu's switch, made in place for a style that isn't an installed theme. */
function StyleSwitch({ to }: { readonly to: UiStyle }) {
  const { setStyle } = useUiStyle()
  return <button type="button" onClick={() => setStyle(to)}>Switch to {to}</button>
}

function app(style: UiStyle, status: ThemeHostStatus, entry: ThemeEntry | null = null) {
  return <LookAppearance style={style} status={status} entry={entry}><Probe /><StyleSwitch to="studio" /></LookAppearance>
}

function renderApp(style: UiStyle, status: ThemeHostStatus, entry: ThemeEntry | null = null) {
  return render(app(style, status, entry))
}

let removeBoot: () => void = () => undefined

/** What the root layout's boot does before any page code runs (hello-a09b's default look on <html>), and its JSON element left on the page. */
function bootThePage(): void {
  const boot = helloThemeBoot()
  runThemeBoot(boot)
  removeBoot = putPageBoot(boot)
}

beforeEach(() => {
  window.localStorage.clear()
  // A browser already moved to the current defaults, so the provider keeps the given style.
  window.localStorage.setItem(APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION)
  dropCookies(COOKIES_SET_HERE)
  clearRoot()
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
  // The browser's own boot fetch (lib/theme-host/boot-client.ts) answers the base boot: no theme to run.
  vi.spyOn(apiClient, 'getThemeBoot').mockResolvedValue(baseThemeBoot(INSTALLED_STYLE))
})

afterEach(() => {
  cleanup()
  removeBoot()
  removeBoot = () => undefined
  vi.restoreAllMocks()
  window.localStorage.clear()
  dropCookies(COOKIES_SET_HERE)
  clearRoot()
})

describe('PRD-261 W-A6 — an installed theme keeps the first paint the boot put on <html>', () => {
  it.each<ThemeHostStatus>(['loading', 'active'])('host %s, no registry entry: every look variable the boot painted stays after mount', async (status) => {
    runThemeBoot(helloThemeBoot())
    renderApp(INSTALLED_STYLE, status)
    expect(await screen.findByText('Looks: none')).toBeInTheDocument()
    expect(rootVars()).toEqual(paint(TABLE, PAPER))
    expect(screen.getByText('Tone: not forced')).toBeInTheDocument()
    expect(cookieValue(THEME_TONE_COOKIE)).toBeUndefined()
  })

  it('a load that fails takes the first paint off, as Studio shows instead', async () => {
    bootThePage()
    const { rerender } = renderApp(HELLO_LOOKS_STYLE, 'active', HELLO_LOOKS_THEME)
    expect(await screen.findByText('Looks: available')).toBeInTheDocument()
    expect(rootVars()).toEqual(paint(TABLE, PAPER))
    rerender(app(HELLO_LOOKS_STYLE, 'failed', HELLO_LOOKS_THEME))
    expect(await screen.findByText('Looks: none')).toBeInTheDocument()
    await waitFor(() => expect(lookNamesLeft()).toEqual([]))
    expect(motionMarks()).toEqual(NO_MOTION_MARKS)
    expect(cookieValue(THEME_TONE_COOKIE)).toBeUndefined()
  })

  it.each<ThemeHostStatus>(['safe-mode', 'failed'])(
    'host %s from the start, before it released the entry: no look runtime, the first paint off, no tone forced',
    async (status) => {
      bootThePage()
      // The host's release reaches the style context a commit later; here the entry still stands.
      renderApp(HELLO_LOOKS_STYLE, status, HELLO_LOOKS_THEME)
      expect(await screen.findByText('Looks: none')).toBeInTheDocument()
      await waitFor(() => expect(lookNamesLeft()).toEqual([]))
      expect(screen.getByText('Tone: not forced')).toBeInTheDocument()
      expect(cookieValue(THEME_TONE_COOKIE)).toBeUndefined()
      expect(motionMarks()).toEqual(NO_MOTION_MARKS)
      await waitFor(() => expect(htmlTone()).toBe(USER_DEFAULT_TONE))
    },
  )
})

describe('PRD-261 A-19d2 — an installed theme with looks runs on the generic look runtime, from its own boot', () => {
  it.each<ThemeHostStatus>(['loading', 'active'])(
    'host %s: the looks are available, the look paints from its table and its tone rule forces the tone',
    async (status) => {
      bootThePage()
      renderApp(HELLO_LOOKS_STYLE, status, HELLO_LOOKS_THEME)
      expect(await screen.findByText('Looks: available')).toBeInTheDocument()
      expect(await screen.findByText('Tone: light')).toBeInTheDocument()
      await waitFor(() => expect(cookieValue(THEME_TONE_COOKIE)).toBe(themeToneCookieValue(HELLO_ENTRY.themeId, 'light')))
      await waitFor(() => expect(htmlTone()).toBe('light'))
      expect(rootVars()).toEqual(paint(TABLE, PAPER))
    },
  )

  it('marks motion with data-theme-motion and --theme-motion, never a theme\'s own motion mark', async () => {
    removeBoot = putPageBoot(helloThemeBoot())
    renderApp(HELLO_LOOKS_STYLE, 'active', HELLO_LOOKS_THEME)
    await waitFor(() => expect(motionMarks()).toEqual({ attribute: 'full', variable: '1' }))
    expect(document.documentElement.hasAttribute(NOT_THEME_NEUTRAL_MOTION_ATTRIBUTE)).toBe(false)
  })

  it("left in place for Studio: its table's variables and the motion marks come off once the host stops", async () => {
    bootThePage()
    const { rerender } = renderApp(HELLO_LOOKS_STYLE, 'active', HELLO_LOOKS_THEME)
    await waitFor(() => expect(rootVars()).toEqual(paint(TABLE, PAPER)))
    await waitFor(() => expect(motionMarks().attribute).toBe('full'))

    // A switch away from an installed theme is made in place (only one TO an installed theme reloads),
    // and the host's status trails the style by a commit: it still runs the theme here.
    fireEvent.click(screen.getByRole('button', { name: 'Switch to studio' }))
    // UiStyleProvider reads its props once, at mount, as from the root layout: only the host's status changes here.
    rerender(app('studio', 'off'))
    expect(await screen.findByText('Looks: none')).toBeInTheDocument()
    await waitFor(() => expect(rootVars()).toEqual({}))
    expect(motionMarks()).toEqual(NO_MOTION_MARKS)
    expect(cookieValue(THEME_TONE_COOKIE)).toBeUndefined()
    expect(screen.getByText('Tone: not forced')).toBeInTheDocument()
  })

  it("a theme with looks whose boot on the page and fetched by the browser carries none: no looks, the first paint kept, no tone forced", async () => {
    // The boot script painted the first frame; no boot element of this theme is left on the page.
    runThemeBoot(helloThemeBoot())
    const entry = installedThemeEntry(INSTALLED_HELLO_STYLE, HELLO_PACKAGE_ENTRY)
    renderApp(INSTALLED_HELLO_STYLE, 'active', entry)
    expect(await screen.findByText('Looks: none')).toBeInTheDocument()
    expect(rootVars()).toEqual(paint(TABLE, PAPER))
    expect(screen.getByText('Tone: not forced')).toBeInTheDocument()
    expect(cookieValue(THEME_TONE_COOKIE)).toBeUndefined()
  })
})

// boot-client.ts's page source keeps one answer per install for the module's life (this file): no
// other test here may send HELLO_LOOKS_STYLE's install through the browser's fetch, or this one reads
// that test's answer. The tests above all carry a page boot that names the theme, so none fetches it.
describe('PRD-261 A-08c — the server missed its boot budget: the browser fetches the boot and the look still paints', () => {
  it("a page with the base boot: the theme's entry and look come from one GET /api/themes/boot, and the look's tone is forced", async () => {
    removeBoot = putPageBoot(baseThemeBoot(HELLO_LOOKS_STYLE))
    const boot = vi.spyOn(apiClient, 'getThemeBoot').mockResolvedValue(helloThemeBoot())
    renderApp(HELLO_LOOKS_STYLE, 'active')
    expect(await screen.findByText('Looks: available')).toBeInTheDocument()
    await waitFor(() => expect(rootVars()).toEqual(paint(TABLE, PAPER)))
    expect(await screen.findByText('Tone: light')).toBeInTheDocument()
    expect(boot.mock.calls).toEqual([[installIdOfStyle(HELLO_LOOKS_STYLE)]])
  })
})

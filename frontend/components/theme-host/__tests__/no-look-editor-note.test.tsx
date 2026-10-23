/**
 * PRD-261 A-19f1 — /looks (Make it yours) and /looks/style (the style sheet) are pages only a theme
 * draws, through its lookEditor@1 and styleSheet@1 slots (components/theme-host/theme-slot-page.tsx).
 * Under Classic and Studio, under an installed theme that failed (Studio shows) and under one that
 * registered no such slot, the core's note shows instead (components/theme-host/no-look-editor-note.tsx):
 * the page's title in the shared PageHeader, a line that points at the Appearance menu, and no switch
 * of any kind, so no button. The note is a region named by the title that carries the theme heading
 * mark (`[data-theme-heading]`) with tabIndex -1, the page's one focus anchor whichever shows. An
 * installed theme with the slot (the fixture theme, PRD-261 A-19b) renders its screen instead, full
 * bleed inside MainLayout; below the theme's width the page says it opens on a wider screen; while
 * the host still loads the theme, neither shows.
 *
 * W-A8: in a workspace, the note reads the workspace's installs. When none can be chosen (none
 * installed, or only a withdrawn one: the Appearance menu lists Classic and Studio alone), it says
 * how to get a theme (Marketplace › Themes) instead; with one to choose, and before the installs
 * are known, it keeps the page's line. Outside a workspace it asks for nothing.
 *
 * Replaces components/looks/__tests__/play-only-route.test.tsx in the core test ledger. The real
 * pages render under the real style provider and slot hooks, with the theme host's value as its
 * provider gives it; MainLayout (the frame) is stood in for.
 */
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, render, screen, within } from '@testing-library/react'
import type { ComponentType, ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('next-themes', () => ({ useTheme: () => ({ theme: 'dark', setTheme: () => undefined }) }))
/** The workspace the page is in: none (an isolated mount) unless a test names one. */
const ws = vi.hoisted(() => ({ id: null as string | null }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspaceOptional: () => (ws.id === null ? null : { workspace: { id: ws.id } }),
  useWorkspace: () => ({ workspace: ws.id === null ? null : { id: ws.id } }),
}))
vi.mock('@/components/layout/main-layout', () => ({
  MainLayout: ({ children, fullBleed }: { children: ReactNode; fullBleed?: boolean }) => (
    <div data-testid="main-layout" data-full-bleed={fullBleed ? 'true' : 'false'}>
      {children}
    </div>
  ),
}))

import LooksPage from '@/app/looks/page'
import LookStylePage from '@/app/looks/style/page'
import { UiStyleProvider } from '@/contexts/ui-style-context'
import { apiClient, type InstalledTheme } from '@/lib/api-client'
import { PHONE_WIDTH_PX, stubWindowWidth, type WindowWidth } from '@/hooks/__tests__/window-width'
import {
  FIXTURE_THEME_ENTRY,
  FIXTURE_THEME_STYLE,
  FIXTURE_THEME_TEST_IDS,
  fixtureThemeHost,
} from '@/lib/theme-host/__tests__/fixture-theme-entry'
import type { SlotId } from '@/lib/theme-host/registry-types'
import { APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION, type UiStyle } from '@/lib/ui-style'

import {
  LOOKS_NOTE,
  LOOKS_NO_THEME_NOTE,
  LOOKS_TITLE,
  STYLE_SHEET_NOTE,
  STYLE_SHEET_NO_THEME_NOTE,
  STYLE_SHEET_TITLE,
  THEME_HEADING_ATTRIBUTE,
  THEME_HEADING_SELECTOR,
  WIDER_SCREEN_NOTE,
} from '../no-look-editor-note'
import { ThemeHostContext, type ThemeHostContextValue } from '../theme-host-context'

interface PageCase {
  readonly Page: ComponentType
  readonly title: string
  readonly note: string
  /** The page's line while the workspace has no theme to choose. */
  readonly noThemeNote: string
  /** The test id of the fixture theme's screen in the page's slot. */
  readonly screenTestId: string
}

const PAGES: ReadonlyArray<readonly [string, PageCase]> = [
  [
    '/looks (lookEditor@1)',
    { Page: LooksPage, title: LOOKS_TITLE, note: LOOKS_NOTE, noThemeNote: LOOKS_NO_THEME_NOTE, screenTestId: FIXTURE_THEME_TEST_IDS.lookEditor },
  ],
  [
    '/looks/style (styleSheet@1)',
    {
      Page: LookStylePage, title: STYLE_SHEET_TITLE, note: STYLE_SHEET_NOTE, noThemeNote: STYLE_SHEET_NO_THEME_NOTE,
      screenTestId: FIXTURE_THEME_TEST_IDS.styleSheet,
    },
  ],
]

const CORE_STYLES: readonly UiStyle[] = ['classic', 'studio']

/** A phone-wide window, while a test sets one. */
let phone: WindowWidth | null = null

/**
 * The page under `style` with the theme host's value: Classic and Studio have no entry (the host is
 * off); the fixture theme's style has the entry the root layout builds from its boot.
 */
function renderPage({ Page }: PageCase, style: UiStyle, host: ThemeHostContextValue) {
  const entry = style === FIXTURE_THEME_STYLE ? FIXTURE_THEME_ENTRY : null
  return render(
    <ThemeHostContext.Provider value={host}>
      <UiStyleProvider initialStyle={style} installedTheme={entry}>
        <Page />
      </UiStyleProvider>
    </ThemeHostContext.Provider>,
  )
}

function layout(): HTMLElement {
  return screen.getByTestId('main-layout')
}

/** The core's note: the page's title and `line` in a region named by the title, the heading mark, no switch. */
function expectNote({ title, screenTestId }: PageCase, line: string): void {
  const region = screen.getByRole('region', { name: title })
  expect(region).toHaveAttribute(THEME_HEADING_ATTRIBUTE)
  expect(region).toHaveAttribute('tabindex', '-1')
  expect(document.querySelector(THEME_HEADING_SELECTOR)).toBe(region)
  // The note wears the core's frame, the shared PageHeader (PRD-244, app/__tests__/page-frames.test.ts).
  expect(within(region).getByRole('heading', { level: 1, name: title })).toBe(within(region).getByTestId('page-title'))
  expect(within(region).getByText(line)).toBeInTheDocument()
  expect(screen.queryByRole('button')).toBeNull()
  expect(screen.queryByText(/switch to/i)).toBeNull()
  expect(screen.queryByTestId(screenTestId)).toBeNull()
  expect(layout()).toHaveAttribute('data-full-bleed', 'false')
}

beforeEach(() => {
  window.localStorage.clear()
  // A browser already moved to the current defaults, so the provider keeps the given style.
  window.localStorage.setItem(APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION)
})

afterEach(() => {
  cleanup()
  phone?.restore()
  phone = null
  window.localStorage.clear()
  ws.id = null
  vi.restoreAllMocks()
})

describe.each(PAGES)('%s', (_path, page) => {
  it.each(CORE_STYLES)('under %s shows the note: the title, the line, no switch', (style) => {
    renderPage(page, style, fixtureThemeHost('off'))
    expectNote(page, page.note)
  })

  it('under an installed theme that failed to load (Studio shows) shows the note', () => {
    renderPage(page, FIXTURE_THEME_STYLE, fixtureThemeHost('failed'))
    expectNote(page, page.note)
  })

  it('under an installed theme that registered no such slot shows the note', () => {
    renderPage(page, FIXTURE_THEME_STYLE, fixtureThemeHost('active', { slots: new Map<SlotId, unknown>() }))
    expectNote(page, page.note)
  })

  it("under an installed theme with the slot renders the theme's screen instead, full bleed, and no note", () => {
    renderPage(page, FIXTURE_THEME_STYLE, fixtureThemeHost())
    expect(within(layout()).getByTestId(page.screenTestId)).toBeInTheDocument()
    expect(layout()).toHaveAttribute('data-full-bleed', 'true')
    expect(screen.queryByRole('region', { name: page.title })).toBeNull()
    expect(screen.queryByText(page.note)).toBeNull()
  })

  it("below the theme's width says it opens on a wider screen, with no screen and no switch", () => {
    phone = stubWindowWidth(PHONE_WIDTH_PX)
    renderPage(page, FIXTURE_THEME_STYLE, fixtureThemeHost())
    expectNote(page, WIDER_SCREEN_NOTE)
    expect(screen.queryByText(page.note)).toBeNull()
  })

  it('while the host still loads the installed theme, shows neither the note nor a screen', () => {
    renderPage(page, FIXTURE_THEME_STYLE, fixtureThemeHost('loading'))
    expect(screen.queryByRole('region', { name: page.title })).toBeNull()
    expect(screen.queryByTestId(page.screenTestId)).toBeNull()
  })
})

/** An install of the workspace's, as GET /api/themes/installed lists it. */
function installRow(overrides: Partial<InstalledTheme> = {}): InstalledTheme {
  return {
    installId: '11111111-1111-4111-8111-111111111111', themeId: 'acme/garden', name: 'Garden',
    packageId: '22222222-2222-4222-8222-222222222222', version: '1.0.0', ownerType: 'workspace', reviewed: true,
    private: false, isDefault: false, pinnedVersion: null, previousPackageId: null, pendingPackageId: null,
    pendingVersion: null, installedAt: null, optIn: { required: false, wording: '', packageId: '' }, revoked: null,
    needsUpdate: false, ...overrides,
  }
}

/** Lets the installs' request answer and the render after it run. */
async function settle(): Promise<void> {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0))
  })
}

/** The page in workspace ws-1, under Studio, with `installs` as the workspace's installs. */
function renderInWorkspace(page: PageCase, installs: readonly InstalledTheme[]) {
  ws.id = 'ws-1'
  const listed = vi.spyOn(apiClient, 'getInstalledThemes').mockResolvedValue({ installs: [...installs] })
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <QueryClientProvider client={client}>
      <ThemeHostContext.Provider value={fixtureThemeHost('off')}>
        <UiStyleProvider initialStyle="studio">
          <page.Page />
        </UiStyleProvider>
      </ThemeHostContext.Provider>
    </QueryClientProvider>,
  )
  return listed
}

describe.each(PAGES)('%s in a workspace (W-A8)', (_path, page) => {
  it('with no theme installed says how to get one: Marketplace › Themes, then the Appearance menu', async () => {
    const listed = renderInWorkspace(page, [])
    expect(await screen.findByText(page.noThemeNote)).toBeInTheDocument()
    expectNote(page, page.noThemeNote)
    expect(page.noThemeNote).toContain('Marketplace › Themes')
    expect(screen.queryByText(page.note)).toBeNull()
    expect(listed).toHaveBeenCalledTimes(1)
  })

  it('with only a withdrawn install (the menu offers none) says how to get one', async () => {
    renderInWorkspace(page, [installRow({ revoked: { level: 'disable', reason: 'withdrawn' } })])
    expect(await screen.findByText(page.noThemeNote)).toBeInTheDocument()
  })

  it("with a theme to choose keeps the page's line, before and after the installs are known", async () => {
    const listed = renderInWorkspace(page, [installRow()])
    expectNote(page, page.note)
    await vi.waitFor(() => expect(listed).toHaveBeenCalledTimes(1))
    await settle()
    expectNote(page, page.note)
    expect(screen.queryByText(page.noThemeNote)).toBeNull()
  })

  it('outside a workspace asks for no installs and keeps the line', () => {
    const listed = vi.spyOn(apiClient, 'getInstalledThemes')
    renderPage(page, 'studio', fixtureThemeHost('off'))
    expectNote(page, page.note)
    expect(listed).not.toHaveBeenCalled()
  })
})

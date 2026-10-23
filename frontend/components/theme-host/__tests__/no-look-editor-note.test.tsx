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
 * Replaces components/looks/__tests__/play-only-route.test.tsx in the core test ledger. The real
 * pages render under the real style provider and slot hooks, with the theme host's value as its
 * provider gives it; MainLayout (the frame) is stood in for.
 */
import { cleanup, render, screen, within } from '@testing-library/react'
import type { ComponentType, ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('next-themes', () => ({ useTheme: () => ({ theme: 'dark', setTheme: () => undefined }) }))
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
  LOOKS_TITLE,
  STYLE_SHEET_NOTE,
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
  /** The test id of the fixture theme's screen in the page's slot. */
  readonly screenTestId: string
}

const PAGES: ReadonlyArray<readonly [string, PageCase]> = [
  ['/looks (lookEditor@1)', { Page: LooksPage, title: LOOKS_TITLE, note: LOOKS_NOTE, screenTestId: FIXTURE_THEME_TEST_IDS.lookEditor }],
  [
    '/looks/style (styleSheet@1)',
    { Page: LookStylePage, title: STYLE_SHEET_TITLE, note: STYLE_SHEET_NOTE, screenTestId: FIXTURE_THEME_TEST_IDS.styleSheet },
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

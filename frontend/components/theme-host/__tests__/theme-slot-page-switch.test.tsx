/**
 * PRD-261 G3 (P260-FX-22, M21) — ThemeSlotPage after a switch onto an installed theme. The switch
 * reloads the page (PA-13), so this load never saw the page that made it; the switch left word in
 * this tab's sessionStorage just before the reload (switch-focus.test.tsx), and here the installed
 * theme's screen heading takes the focus once it renders, however late the host loads the theme. A
 * load with no word (or another page's) leaves focus alone, and the word is taken once.
 */
import React from 'react'
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SlotId } from '@/lib/theme-host/registry-types'
import type { ThemeHostStatus } from '@/lib/theme-host/types'

vi.mock('@/hooks/use-studio-theme', () => ({
  useIsStudio: () => true,
  useActiveTheme: () => null,
  useThemeLayout: () => 'theme',
}))
vi.mock('@/components/layout/main-layout', () => ({
  MainLayout: ({ children }: { children: React.ReactNode }) => <div data-testid="main-layout">{children}</div>,
}))

import { SWITCH_FOCUS_STORAGE_KEY, rememberHeadingFocusAfterReload } from '@/lib/theme-host/switch-focus'

import { ThemeHostContext, type ThemeHostContextValue } from '../theme-host-context'
import { ThemeSlotPage } from '../theme-slot-page'

const LOOKS_PATH = '/looks'
/** The mark the theme's screen puts on its h1 (a theme's own; the core's is THEME_HEADING_SELECTOR). */
const HEADING_SELECTOR = '[data-test-screen-heading]'

function InstalledLookEditor() {
  return (
    <h1 tabIndex={-1} data-test-screen-heading>
      Make it yours.
    </h1>
  )
}

function Frame({ children }: { children?: React.ReactNode }) {
  return <>{children}</>
}

/** The theme host as its provider gives it for an installed theme with a look editor. */
function installedHost(status: ThemeHostStatus): ThemeHostContextValue {
  const active = status === 'active'
  return {
    status,
    reason: null,
    frame: active ? Frame : null,
    slots: active ? new Map<SlotId, unknown>([['lookEditor@1', InstalledLookEditor]]) : null,
    reportRenderFailure: vi.fn(),
  }
}

function page(status: ThemeHostStatus) {
  return (
    <ThemeHostContext.Provider value={installedHost(status)}>
      <ThemeSlotPage
        slot="lookEditor@1"
        note={<p>This style has no look to edit.</p>}
        narrowNote={<p>It opens on a wider screen.</p>}
        headingSelector={HEADING_SELECTOR}
      />
    </ThemeHostContext.Provider>
  )
}

const heading = () => screen.getByRole('heading', { level: 1, name: 'Make it yours.' })

beforeEach(() => {
  window.sessionStorage.clear()
  window.history.replaceState(null, '', LOOKS_PATH)
})
afterEach(() => {
  cleanup()
  window.sessionStorage.clear()
})

describe('ThemeSlotPage after a switch reloaded the page into an installed theme (G3)', () => {
  it("hands focus to the installed theme's screen heading, and takes the word once", () => {
    rememberHeadingFocusAfterReload(LOOKS_PATH)
    render(page('active'))
    expect(heading()).toHaveFocus()
    expect(window.sessionStorage.getItem(SWITCH_FOCUS_STORAGE_KEY)).toBeNull()
  })

  it('waits for the host to load the theme, then focuses the heading', () => {
    rememberHeadingFocusAfterReload(LOOKS_PATH)
    const { rerender } = render(page('loading'))
    expect(screen.queryByRole('heading')).not.toBeInTheDocument()
    expect(document.activeElement).toBe(document.body)
    rerender(page('active'))
    expect(heading()).toHaveFocus()
  })

  it('a load with no word leaves focus where it is', () => {
    render(page('active'))
    expect(heading()).not.toHaveFocus()
    expect(document.activeElement).toBe(document.body)
  })

  it("another page's word is cleared and does not move focus here", () => {
    rememberHeadingFocusAfterReload('/looks/style')
    render(page('active'))
    expect(heading()).not.toHaveFocus()
    expect(window.sessionStorage.getItem(SWITCH_FOCUS_STORAGE_KEY)).toBeNull()
  })
})

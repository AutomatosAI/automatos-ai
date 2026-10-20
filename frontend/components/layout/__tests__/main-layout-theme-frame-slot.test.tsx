/**
 * PRD-261 A-01 — MainLayout's Studio desktop chrome goes through the theme frame slot. With no
 * theme host (every build with the spike off) the shell is exactly as before: `.sh-shell`, the
 * full-bleed main, or the padded main with its max-width wrapper. With an active frame, the frame
 * wraps the page instead of the shell; while the host loads an installed theme, nothing draws (the
 * page never mounts in the shell and again in the frame, M69). A theme's frame comes from the
 * theme host, never from its registry entry: the fixture theme (PRD-261 A-19b) stands in for one
 * whose registrations the host handed over; below the theme's width MainLayout draws the core's
 * chrome before it reaches the frame slot.
 */
import React from 'react'
import { render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ThemeHostContextValue } from '@/components/theme-host/theme-host-context'
import type { ThemeEntry, ThemeLayout } from '@/lib/theme-host/registry-types'

const host = vi.hoisted(() => ({ current: null as ThemeHostContextValue | null }))
// The Studio style by default: no active theme entry, the fallback layout. Under an installed
// theme's style the registry has no entry for it yet (A-02c), so it reads the same.
const theme = vi.hoisted(() => ({ entry: null as ThemeEntry | null, layout: 'fallback' as ThemeLayout }))

vi.mock('next/navigation', () => ({ usePathname: () => '/agents', useRouter: () => ({ push: vi.fn(), replace: vi.fn() }) }))
vi.mock('framer-motion', () => ({
  motion: { div: ({ children, className }: { children?: React.ReactNode; className?: string }) => <div className={className}>{children}</div> },
}))
vi.mock('../sidebar', () => ({ Sidebar: () => null }))
vi.mock('../mobile-sidebar', () => ({ MobileSidebar: () => null }))
vi.mock('../header', () => ({ Header: () => null }))
vi.mock('../studio-sidebar', () => ({ StudioSidebar: () => null }))
vi.mock('../studio-header', () => ({ StudioHeader: () => null }))
vi.mock('../../chatbot/chat-widget', () => ({ AutoWidget: () => <aside data-testid="auto-widget" /> }))
vi.mock('@/components/ui/sheet', () => ({
  Sheet: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
  SheetContent: () => null,
}))
// A desktop width by default, in the Studio style (the core's chrome, not another style's frame).
const viewport = vi.hoisted(() => ({ tabletOrBelow: false }))
vi.mock('@/hooks/use-mobile', () => ({ useIsTabletOrBelow: () => viewport.tabletOrBelow, useIsMobile: () => false }))
vi.mock('@/hooks/use-studio-theme', () => ({ useIsStudio: () => true, useActiveTheme: () => theme.entry, useThemeLayout: () => theme.layout }))

import { ThemeHostContext } from '@/components/theme-host/theme-host-context'
import { FIXTURE_THEME_ENTRY, FIXTURE_THEME_TEST_IDS, fixtureThemeHost } from '@/lib/theme-host/__tests__/fixture-theme-entry'
import type { FrameProps } from '@/lib/theme-host/types'

import { MainLayout } from '../main-layout'

/** An installed theme's entry, once the registry has one (A-02c): the theme layout, its frame from the host only. */
const INSTALLED_ENTRY: ThemeEntry = { ...FIXTURE_THEME_ENTRY, ref: 'install:00000000-0000-4000-8000-00000000e2e1' }
/** The fixture theme's own frame: drawn only where the host hands its registrations over. */
const FIXTURE_FRAME = FIXTURE_THEME_TEST_IDS.frame

/** MainLayout under the theme host's current value (none: no provider, as in an isolated mount). */
function renderLayout(props: { fullBleed?: boolean } = {}) {
  const layout = (
    <MainLayout {...props}>
      <div data-testid="page" />
    </MainLayout>
  )
  return render(host.current === null ? layout : <ThemeHostContext.Provider value={host.current}>{layout}</ThemeHostContext.Provider>)
}

function renderPage() {
  return renderLayout()
}

function HelloFrame({ children }: { children?: React.ReactNode }) {
  return <section data-testid="theme-frame">{children}</section>
}

/** A frame that shows the two frame@1 props MainLayout hands it (plan §2.2). */
function PropsFrame({ children, fullBleed, autoWidget }: FrameProps) {
  return (
    <section data-testid="theme-frame" data-full-bleed={String(fullBleed)}>
      {children}
      {autoWidget}
    </section>
  )
}

afterEach(() => {
  host.current = null
  viewport.tabletOrBelow = false
  theme.entry = null
  theme.layout = 'fallback'
})

describe('MainLayout at desktop width in the Studio style, through the theme frame slot', () => {
  it('draws the Studio shell with no theme host', () => {
    const { container } = renderPage()
    expect(container.querySelector('.sh-shell.safe-bottom')).not.toBeNull()
    expect(screen.queryByTestId('theme-frame')).toBeNull()
  })

  it('keeps the full-bleed main for fullBleed pages', () => {
    const { container } = renderLayout({ fullBleed: true })
    const main = container.querySelector('.sh-shell main.sh-fullbleed')
    expect(main).not.toBeNull()
    expect(main!.querySelector('[data-testid="page"]')).not.toBeNull()
  })

  it('keeps the padded main and its max-width wrapper for framed pages', () => {
    const { container } = renderPage()
    const main = container.querySelector('.sh-shell main')
    expect(main?.className).toContain('px-4')
    expect(main?.querySelector('.max-w-\\[1720px\\] [data-testid="page"]')).not.toBeNull()
  })

  it("draws the active theme's frame around the page instead of the shell", () => {
    host.current = { status: 'active', reason: null, frame: HelloFrame, slots: null, reportRenderFailure: vi.fn() }
    const { container } = renderPage()
    expect(screen.getByTestId('theme-frame')).toContainElement(screen.getByTestId('page'))
    expect(container.querySelector('.sh-shell')).toBeNull()
  })

  it("hands the active theme's frame the full-bleed flag and Auto's widget", () => {
    host.current = { status: 'active', reason: null, frame: PropsFrame, slots: null, reportRenderFailure: vi.fn() }
    renderLayout({ fullBleed: true })
    const frame = screen.getByTestId('theme-frame')
    expect(frame).toHaveAttribute('data-full-bleed', 'true')
    expect(frame).toContainElement(screen.getByTestId('auto-widget'))
  })

  it('draws nothing, neither the shell nor the page, while the host loads an installed theme (M69)', () => {
    host.current = { status: 'loading', reason: null, frame: null, slots: null, reportRenderFailure: vi.fn() }
    const { container } = renderPage()
    expect(container.innerHTML).toBe('')
  })

  it("draws an installed theme's frame through the slot, never one its entry names, under the theme layout", () => {
    theme.entry = INSTALLED_ENTRY
    theme.layout = 'theme'
    host.current = { status: 'active', reason: null, frame: HelloFrame, slots: null, reportRenderFailure: vi.fn() }
    renderPage()
    expect(screen.getByTestId('theme-frame')).toContainElement(screen.getByTestId('page'))
    expect(screen.queryByTestId(FIXTURE_FRAME)).toBeNull()
  })

  it("draws nothing while an installed theme with a layout loads, never a frame its entry names", () => {
    theme.entry = INSTALLED_ENTRY
    theme.layout = 'theme'
    host.current = { status: 'loading', reason: null, frame: null, slots: null, reportRenderFailure: vi.fn() }
    const { container } = renderPage()
    expect(container.innerHTML).toBe('')
  })

  it("draws the theme's frame its host handed over at the theme layout, and not another", async () => {
    theme.entry = FIXTURE_THEME_ENTRY
    theme.layout = 'theme'
    host.current = fixtureThemeHost()
    renderPage()
    expect(await screen.findByTestId(FIXTURE_FRAME)).toContainElement(screen.getByTestId('page'))
    expect(screen.queryByTestId('theme-frame')).toBeNull()
  })

  it("draws the theme's frame through the frame slot: the page inside it keeps its own boundary", async () => {
    theme.entry = FIXTURE_THEME_ENTRY
    theme.layout = 'theme'
    host.current = fixtureThemeHost()
    const { container } = renderPage()
    // The frame slot's own page boundary adds no element: the page sits directly in the frame.
    const frame = await screen.findByTestId(FIXTURE_FRAME)
    expect(screen.getByTestId('page').parentElement).toBe(frame)
    expect(container.querySelector('.sh-shell')).toBeNull()
  })

  it("draws the core's chrome, never the theme's frame, under the theme below its width (D7)", () => {
    viewport.tabletOrBelow = true
    theme.entry = FIXTURE_THEME_ENTRY
    theme.layout = 'fallback'
    host.current = fixtureThemeHost()
    renderPage()
    expect(screen.getByTestId('page')).toBeInTheDocument()
    expect(screen.queryByTestId(FIXTURE_FRAME)).toBeNull()
  })

  it("draws nothing under the theme while the width is not known (M69)", () => {
    theme.entry = FIXTURE_THEME_ENTRY
    theme.layout = 'pending'
    host.current = fixtureThemeHost()
    const { container } = renderPage()
    expect(container.innerHTML).toBe('')
  })
})

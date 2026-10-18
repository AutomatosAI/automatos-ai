/**
 * PRD-261 A-01 — MainLayout's Studio desktop chrome goes through the theme frame slot. With no
 * theme host (every build with the spike off) the shell is exactly as before: `.sh-shell`, the
 * full-bleed main, or the padded main with its max-width wrapper. With an active frame, the frame
 * wraps the page instead of the shell; while the host loads an installed theme, nothing draws (the
 * page never mounts in the shell and again in the frame, M69). Only builtin Play's entry draws
 * Play's frame; an installed theme's goes through the slot.
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
// A desktop width, in the Studio style (the core's chrome, not another style's frame).
vi.mock('@/hooks/use-mobile', () => ({ useIsTabletOrBelow: () => false, useIsMobile: () => false }))
vi.mock('@/hooks/use-studio-theme', () => ({ useIsStudio: () => true, useActiveTheme: () => theme.entry, useThemeLayout: () => theme.layout }))
vi.mock('@/components/theme-host/theme-host-context', () => ({ useThemeHost: () => host.current }))
// Builtin Play's frame (play-frame.test.tsx renders the real one through MainLayout).
vi.mock('@/components/play/shell/play-frame-lazy', () => ({
  PlayFrame: ({ children }: { children?: React.ReactNode }) => <section data-testid="play-frame">{children}</section>,
}))

import { BUILTIN_PLAY_ENTRY } from '@/lib/theme-host/builtin/play-entry'
import type { FrameProps } from '@/lib/theme-host/types'

import { MainLayout } from '../main-layout'

/** An installed theme's entry, once the registry has one (A-02c): the theme layout, no builtin frame. */
const INSTALLED_ENTRY: ThemeEntry = { ...BUILTIN_PLAY_ENTRY, ref: 'install:00000000-0000-4000-8000-00000000e2e1', kind: 'installed' }

function renderPage() {
  return render(
    <MainLayout>
      <div data-testid="page" />
    </MainLayout>,
  )
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
  theme.entry = null
  theme.layout = 'fallback'
})

describe('MainLayout at desktop width in the Studio style, through the theme frame slot', () => {
  it('draws the Studio shell with no theme host', () => {
    const { container } = render(
      <MainLayout>
        <div data-testid="page" />
      </MainLayout>,
    )
    expect(container.querySelector('.sh-shell.safe-bottom')).not.toBeNull()
    expect(screen.queryByTestId('theme-frame')).toBeNull()
  })

  it('keeps the full-bleed main for fullBleed pages', () => {
    const { container } = render(
      <MainLayout fullBleed>
        <div data-testid="page" />
      </MainLayout>,
    )
    const main = container.querySelector('.sh-shell main.sh-fullbleed')
    expect(main).not.toBeNull()
    expect(main!.querySelector('[data-testid="page"]')).not.toBeNull()
  })

  it('keeps the padded main and its max-width wrapper for framed pages', () => {
    const { container } = render(
      <MainLayout>
        <div data-testid="page" />
      </MainLayout>,
    )
    const main = container.querySelector('.sh-shell main')
    expect(main?.className).toContain('px-4')
    expect(main?.querySelector('.max-w-\\[1720px\\] [data-testid="page"]')).not.toBeNull()
  })

  it("draws the active theme's frame around the page instead of the shell", () => {
    host.current = { status: 'active', reason: null, frame: HelloFrame, reportRenderFailure: vi.fn() }
    const { container } = render(
      <MainLayout>
        <div data-testid="page" />
      </MainLayout>,
    )
    expect(screen.getByTestId('theme-frame')).toContainElement(screen.getByTestId('page'))
    expect(container.querySelector('.sh-shell')).toBeNull()
  })

  it("hands the active theme's frame the full-bleed flag and Auto's widget", () => {
    host.current = { status: 'active', reason: null, frame: PropsFrame, reportRenderFailure: vi.fn() }
    render(
      <MainLayout fullBleed>
        <div data-testid="page" />
      </MainLayout>,
    )
    const frame = screen.getByTestId('theme-frame')
    expect(frame).toHaveAttribute('data-full-bleed', 'true')
    expect(frame).toContainElement(screen.getByTestId('auto-widget'))
  })

  it('draws nothing, neither the shell nor the page, while the host loads an installed theme (M69)', () => {
    host.current = { status: 'loading', reason: null, frame: null, reportRenderFailure: vi.fn() }
    const { container } = renderPage()
    expect(container.innerHTML).toBe('')
  })

  it("draws an installed theme's frame through the slot, never builtin Play's, under the theme layout", () => {
    theme.entry = INSTALLED_ENTRY
    theme.layout = 'theme'
    host.current = { status: 'active', reason: null, frame: HelloFrame, reportRenderFailure: vi.fn() }
    renderPage()
    expect(screen.getByTestId('theme-frame')).toContainElement(screen.getByTestId('page'))
    expect(screen.queryByTestId('play-frame')).toBeNull()
  })

  it("draws nothing while an installed theme with a layout loads, never builtin Play's frame", () => {
    theme.entry = INSTALLED_ENTRY
    theme.layout = 'theme'
    host.current = { status: 'loading', reason: null, frame: null, reportRenderFailure: vi.fn() }
    const { container } = renderPage()
    expect(container.innerHTML).toBe('')
  })

  it("draws builtin Play's frame under Play at the theme layout, and not the slot's", () => {
    theme.entry = BUILTIN_PLAY_ENTRY
    theme.layout = 'theme'
    renderPage()
    expect(screen.getByTestId('play-frame')).toContainElement(screen.getByTestId('page'))
    expect(screen.queryByTestId('theme-frame')).toBeNull()
  })
})

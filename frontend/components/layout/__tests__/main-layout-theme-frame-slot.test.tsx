/**
 * PRD-261 A-01 — MainLayout's Studio desktop chrome goes through the theme frame slot. With no
 * theme host (every build with the spike off) the shell is exactly as before: `.sh-shell`, the
 * full-bleed main, or the padded main with its max-width wrapper. With an active frame, the frame
 * wraps the page instead of the shell.
 */
import React from 'react'
import { render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ThemeHostContextValue } from '@/components/theme-host/theme-host-context'

const host = vi.hoisted(() => ({ current: null as ThemeHostContextValue | null }))

vi.mock('next/navigation', () => ({ usePathname: () => '/agents', useRouter: () => ({ push: vi.fn(), replace: vi.fn() }) }))
vi.mock('framer-motion', () => ({
  motion: { div: ({ children, className }: { children?: React.ReactNode; className?: string }) => <div className={className}>{children}</div> },
}))
vi.mock('../sidebar', () => ({ Sidebar: () => null }))
vi.mock('../mobile-sidebar', () => ({ MobileSidebar: () => null }))
vi.mock('../header', () => ({ Header: () => null }))
vi.mock('../studio-sidebar', () => ({ StudioSidebar: () => null }))
vi.mock('../studio-header', () => ({ StudioHeader: () => null }))
vi.mock('../../chatbot/chat-widget', () => ({ AutoWidget: () => null }))
vi.mock('@/components/ui/sheet', () => ({
  Sheet: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
  SheetContent: () => null,
}))
// A desktop width, in the Studio style (the core's chrome, not another style's frame).
vi.mock('@/hooks/use-mobile', () => ({ useIsTabletOrBelow: () => false, useIsMobile: () => false }))
vi.mock('@/hooks/use-studio-theme', () => ({ useIsStudio: () => true, useActiveTheme: () => null, useThemeLayout: () => 'fallback' }))
vi.mock('@/components/theme-host/theme-host-context', () => ({ useThemeHost: () => host.current }))

import { MainLayout } from '../main-layout'

function HelloFrame({ children }: { children?: React.ReactNode }) {
  return <section data-testid="theme-frame">{children}</section>
}

afterEach(() => {
  host.current = null
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
})

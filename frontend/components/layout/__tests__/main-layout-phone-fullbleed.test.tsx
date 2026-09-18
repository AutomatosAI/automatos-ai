/**
 * PRD-246 follow-up — below 1024 the Studio pages render inside the classic chrome,
 * and they are built for the Studio shell's contract: a definite-height column they
 * fill and scroll inside. `fullBleed` must mean the same thing here (Gerard's phone
 * pass, 2026-09-18: the chat shell measured 65px tall and nothing scrolled).
 */
import React from 'react'
import { render } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

vi.mock('next/navigation', () => ({ usePathname: () => '/chat', useRouter: () => ({ push: vi.fn(), replace: vi.fn() }) }))
vi.mock('framer-motion', () => ({
  motion: { div: ({ children, className }: { children?: React.ReactNode; className?: string }) => <div className={className}>{children}</div> },
}))
vi.mock('../sidebar', () => ({ Sidebar: () => null }))
vi.mock('../mobile-sidebar', () => ({ MobileSidebar: () => null }))
vi.mock('../header', () => ({ Header: () => <header data-testid="phone-header" /> }))
vi.mock('../studio-sidebar', () => ({ StudioSidebar: () => null }))
vi.mock('../studio-header', () => ({ StudioHeader: () => null }))
vi.mock('../../chatbot/chat-widget', () => ({ AutoWidget: () => null }))
vi.mock('@/components/ui/sheet', () => ({
  Sheet: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
  SheetContent: () => null,
}))
vi.mock('@/hooks/use-mobile', () => ({ useIsTabletOrBelow: () => true, useIsMobile: () => true }))
vi.mock('@/hooks/use-studio-theme', () => ({ useIsStudio: () => true }))

import { MainLayout } from '../main-layout'

describe('MainLayout on a phone (Studio style, classic chrome)', () => {
  it('fullBleed: a definite-height column the page fills and scrolls inside', () => {
    const { container } = render(
      <MainLayout fullBleed>
        <div data-testid="page" />
      </MainLayout>,
    )
    const main = container.querySelector('main[data-bleed="phone"]')
    expect(main).not.toBeNull()
    expect(main!.className).toContain('flex-1')
    expect(main!.className).toContain('min-h-0')
    expect(main!.className).not.toContain('px-4')
    expect(container.firstElementChild!.className).toContain('h-[100dvh]')
  })

  it('a framed page keeps the padded, document-scrolling main', () => {
    const { container } = render(
      <MainLayout>
        <div data-testid="page" />
      </MainLayout>,
    )
    const main = container.querySelector('main')!
    expect(main.getAttribute('data-bleed')).toBeNull()
    expect(main.className).toContain('px-4')
    expect(container.firstElementChild!.className).toContain('min-h-screen')
  })
})

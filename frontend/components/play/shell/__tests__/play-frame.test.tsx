/**
 * PRD-260 T-004 — MainLayout picks the chrome: Play's frame under Play at >= 1024 px
 * (D7), the Studio shell under Studio, and below 1024 px the fallback chrome it
 * renders today, whatever the style. The frame keeps the core's header utilities
 * (the theme menu among them), honours fullBleed as the Studio shell does, and
 * keeps the floating Auto widget. Roles, attributes and text only (CSS modules are
 * proxies under css: false).
 *
 * P260-FX-23 (M17): MainLayout loads the frame through its lazy slot, so a
 * Classic or Studio page never downloads it; under Play the frame arrives a
 * moment after the first render, so each test waits for it (findBy*).
 *
 * P260-FX-25 (M69): MainLayout reads Play's width from useThemeLayout (stood in
 * for here; play-layout-phone.test.tsx runs the real one) and, under Play
 * while it isn't known (the server, hydration), renders nothing, so the page
 * never mounts in one chrome and again in the other.
 */
import React from 'react'
import { fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

const state = vi.hoisted(() => ({ play: true, studio: true, narrow: false, widthUnknown: false, pathname: '/command-center' }))

vi.mock('next/navigation', () => ({
  usePathname: () => state.pathname,
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
}))
vi.mock('framer-motion', () => ({
  motion: { div: ({ children, className }: { children?: React.ReactNode; className?: string }) => <div className={className}>{children}</div> },
}))
vi.mock('@/hooks/use-mobile', () => ({ useIsTabletOrBelow: () => state.narrow, useIsMobile: () => state.narrow }))
vi.mock('@/hooks/use-studio-theme', async () => {
  const { BUILTIN_PLAY_ENTRY } = await import('@/lib/theme-host/builtin/play-entry')
  return {
    useIsStudio: () => state.studio,
    useActiveTheme: () => (state.play ? BUILTIN_PLAY_ENTRY : null),
    useThemeLayout: () => {
      if (!state.play || state.narrow) return 'fallback'
      return state.widthUnknown ? 'pending' : 'theme'
    },
  }
})
// GET /health answers, so the frame shows (edges.test.tsx covers the asleep screen in its place).
vi.mock('@/hooks/use-backend-health', () => ({ useBackendHealth: () => ({ connection: 'connected', check: vi.fn() }) }))

// The chrome's parts: the sidebars have their own tests.
vi.mock('@/components/play/shell/play-sidebar', () => ({ PlaySidebar: () => <aside data-testid="play-sidebar" /> }))
vi.mock('@/components/layout/studio-sidebar', () => ({ StudioSidebar: () => <aside data-testid="studio-sidebar" /> }))
vi.mock('@/components/layout/sidebar', () => ({ Sidebar: () => null }))
vi.mock('@/components/layout/mobile-sidebar', () => ({ MobileSidebar: () => null }))
vi.mock('@/components/layout/header', () => ({ Header: () => <header data-testid="fallback-header" /> }))
vi.mock('@/components/ui/sheet', () => ({
  Sheet: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
  SheetContent: () => null,
}))
vi.mock('@/components/chatbot/chat-widget', () => ({
  AutoWidget: ({ currentPage, visible }: { currentPage: string; visible: boolean }) => (
    <div data-testid="auto-widget" data-page={currentPage} data-visible={String(visible)} />
  ),
}))

// The header utilities the frame keeps (StudioHeaderUtilities renders them).
vi.mock('@/components/ui/theme-toggle', () => ({ ThemeToggle: () => <button type="button">Toggle theme</button> }))
vi.mock('@/components/notifications/notification-bell', () => ({ NotificationBell: () => <button type="button">Notifications</button> }))
vi.mock('@/components/auth/profile-menu', () => ({ ProfileMenu: () => <button type="button">Profile</button> }))
vi.mock('@/components/auth/user-profile-button', () => ({ UserProfileButton: () => <button type="button">Profile</button> }))

import { MainLayout } from '@/components/layout/main-layout'
import { GLOBAL_SEARCH_OPEN_EVENT } from '@/lib/global-search-event'

function renderLayout(props: { fullBleed?: boolean } = {}) {
  return render(
    <MainLayout {...props}>
      <div data-testid="page">The page</div>
    </MainLayout>,
  )
}

// The frame's chunk, fetched once as a Play page's first render would, so each
// findBy waits on MainLayout's render, not on the first transform of the module.
beforeAll(async () => {
  await import('@/components/play/shell/play-frame')
})

afterEach(() => {
  Object.assign(state, { play: true, studio: true, narrow: false, widthUnknown: false, pathname: '/command-center' })
})

describe('MainLayout under Play at desktop width', () => {
  it("renders Play's frame around the page, not the Studio shell", async () => {
    const { container } = renderLayout()
    expect(await screen.findByTestId('play-sidebar')).toBeInTheDocument()
    expect(within(screen.getByRole('main')).getByTestId('page')).toHaveTextContent('The page')
    expect(screen.queryByTestId('studio-sidebar')).toBeNull()
    expect(container.querySelector('.sh-shell')).toBeNull()
    expect(screen.queryByTestId('fallback-header')).toBeNull()
  })

  it("keeps the core's header utilities: the theme menu, notifications, profile and Help", async () => {
    renderLayout()
    const header = await screen.findByRole('banner')
    expect(within(header).getByRole('button', { name: 'Toggle theme' })).toBeInTheDocument()
    expect(within(header).getByRole('button', { name: 'Notifications' })).toBeInTheDocument()
    expect(within(header).getByRole('button', { name: 'Profile' })).toBeInTheDocument()
    expect(within(header).getByRole('button', { name: 'Help & documentation' })).toBeInTheDocument()
  })

  it('opens the global search from the header', async () => {
    const opened = vi.fn()
    window.addEventListener(GLOBAL_SEARCH_OPEN_EVENT, opened)
    renderLayout()
    fireEvent.click(await screen.findByRole('button', { name: 'Open search (⌘K)' }))
    window.removeEventListener(GLOBAL_SEARCH_OPEN_EVENT, opened)
    expect(opened).toHaveBeenCalledTimes(1)
  })

  it('fullBleed: the page fills the main column itself, unwrapped', async () => {
    renderLayout({ fullBleed: true })
    expect((await screen.findByTestId('page')).parentElement).toBe(screen.getByRole('main'))
  })

  it('a framed page sits in a capped wrapper inside the scrolling main', async () => {
    renderLayout()
    const wrapper = (await screen.findByTestId('page')).parentElement
    expect(wrapper).not.toBe(screen.getByRole('main'))
    expect(wrapper?.parentElement).toBe(screen.getByRole('main'))
  })

  it('keeps the floating Auto widget as the Studio shell has it: on every page but the chat', async () => {
    renderLayout()
    expect(await screen.findByTestId('auto-widget')).toHaveAttribute('data-page', 'activity')
    expect(screen.getByTestId('auto-widget')).toHaveAttribute('data-visible', 'true')
  })

  it('hides the Auto widget on the chat', async () => {
    state.pathname = '/chat'
    renderLayout({ fullBleed: true })
    expect(await screen.findByTestId('auto-widget')).toHaveAttribute('data-visible', 'false')
  })
})

describe('MainLayout under the other styles and widths', () => {
  it('under Studio renders the Studio shell exactly as before', () => {
    state.play = false
    const { container } = renderLayout()
    expect(container.querySelector('.sh-shell.safe-bottom')).not.toBeNull()
    expect(screen.getByTestId('studio-sidebar')).toBeInTheDocument()
    expect(screen.queryByTestId('play-sidebar')).toBeNull()
    expect(screen.getByTestId('page')).toBeInTheDocument()
    expect(screen.getByTestId('auto-widget')).toBeInTheDocument()
  })

  it('under Play below 1024 px renders the existing fallback chrome', () => {
    state.narrow = true
    renderLayout()
    expect(screen.getByTestId('fallback-header')).toBeInTheDocument()
    expect(screen.queryByTestId('play-sidebar')).toBeNull()
    expect(screen.queryByTestId('studio-sidebar')).toBeNull()
    expect(screen.getByTestId('page')).toBeInTheDocument()
  })

  it('under Play while the width is not known renders nothing: neither chrome, nor the page (M69)', () => {
    state.widthUnknown = true
    const { container } = renderLayout()
    expect(container).toBeEmptyDOMElement()
    expect(screen.queryByTestId('page')).toBeNull()
    expect(screen.queryByTestId('auto-widget')).toBeNull()
  })

  it('under Classic renders the fallback chrome, never the frame', () => {
    Object.assign(state, { play: false, studio: false })
    renderLayout()
    expect(screen.getByTestId('fallback-header')).toBeInTheDocument()
    expect(screen.queryByTestId('play-sidebar')).toBeNull()
  })
})

/**
 * PRD-260 M70 — the floating Auto widget names the page it floats over. On Play's
 * own pages (Conversations, Make it yours at /looks and its style sheet at
 * /looks/style) MainLayout used to fall through to 'dashboard', so the open
 * widget said "Ask me anything about Dashboard" and a bug report sent page
 * 'dashboard'. The real AutoWidget renders here; its data sources are stubbed.
 */
import React from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const state = vi.hoisted(() => ({ pathname: '/looks' }))

vi.mock('next/navigation', () => ({
  usePathname: () => state.pathname,
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
}))
vi.mock('framer-motion', () => ({
  motion: { div: ({ children, className }: { children?: React.ReactNode; className?: string }) => <div className={className}>{children}</div> },
  AnimatePresence: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
}))
vi.mock('@/hooks/use-mobile', () => ({ useIsTabletOrBelow: () => false, useIsMobile: () => false }))
// Under Play at a laptop's width: Play's frame (P260-FX-25, PRD-261 A-02b: useThemeLayout is MainLayout's width source).
vi.mock('@/hooks/use-studio-theme', () => ({ useIsStudio: () => true, useThemeLayout: () => 'theme' }))

// Play's frame: the page and the widget MainLayout hands it (play-frame.test.tsx covers the frame).
vi.mock('@/components/play/shell/play-frame', () => ({
  PlayFrame: ({ children, autoWidget }: { children?: React.ReactNode; autoWidget?: React.ReactNode }) => (
    <>
      {children}
      {autoWidget}
    </>
  ),
}))
vi.mock('@/components/layout/sidebar', () => ({ Sidebar: () => null }))
vi.mock('@/components/layout/mobile-sidebar', () => ({ MobileSidebar: () => null }))
vi.mock('@/components/layout/header', () => ({ Header: () => null }))
vi.mock('@/components/layout/studio-sidebar', () => ({ StudioSidebar: () => null }))
vi.mock('@/components/layout/studio-header', () => ({ StudioHeader: () => null }))

// The widget's sources: no user, no conversation yet (the draft), nothing streaming.
vi.mock('@/lib/auth-hooks', () => ({ useUser: () => ({ user: null }) }))
vi.mock('@/hooks/use-bug-report-api', () => ({ useSubmitBugReport: () => ({ mutate: vi.fn() }) }))
vi.mock('@/lib/chat/hooks', () => ({
  useChat: () => ({ messages: [], isLoading: false, sendMessage: vi.fn(), stop: vi.fn() }),
}))
vi.mock('@/lib/page-context', () => ({ usePageContext: () => ({ page: 'unknown-page', route: state.pathname }) }))
vi.mock('@/contexts/role-context', () => ({ useSystemRole: () => ({ isAdmin: false }) }))
vi.mock('@/lib/chat/api', () => ({
  getChat: vi.fn(),
  getChatHistory: vi.fn(() => Promise.resolve([])),
  getChatMessages: vi.fn(),
}))
vi.mock('@/hooks/use-chat-session', () => ({ useChatSessionHydration: () => ({ key: 'k', hydrated: true }) }))
vi.mock('@/stores/chat-session-store', async () => {
  const { EMPTY_CHAT_SESSION } = await import('@/lib/chat/chat-session')
  // Stable actions: the widget's effects depend on them.
  const store = {
    session: EMPTY_CHAT_SESSION,
    openChat: () => {},
    newDraft: () => {},
    closeTab: () => {},
    chatIdAssigned: () => {},
    setTitles: () => {},
  }
  return { useChatSessionStore: (select: (s: typeof store) => unknown) => select(store) }
})

import { MainLayout } from '@/components/layout/main-layout'

async function openWidgetOn(pathname: string) {
  state.pathname = pathname
  render(
    <MainLayout>
      <div data-testid="page" />
    </MainLayout>,
  )
  fireEvent.click(await screen.findByTitle('Ask Auto'))
  return screen.findByText(/^Ask me anything about/)
}

afterEach(() => {
  state.pathname = '/looks'
})

describe("the floating Auto widget on Play's pages (PRD-260 M70)", () => {
  it.each(['/looks', '/looks/style'])("says 'Make it yours' on %s, not 'Dashboard'", async (pathname) => {
    const greeting = await openWidgetOn(pathname)
    expect(greeting).toHaveTextContent('Ask me anything about Make it yours, or tell me what you need help with.')
    expect(screen.queryByText(/Dashboard/)).toBeNull()
  })

  it("says 'Chat' on Conversations, not 'Dashboard'", async () => {
    const greeting = await openWidgetOn('/conversations')
    expect(greeting).toHaveTextContent('Ask me anything about Chat, or tell me what you need help with.')
    expect(screen.queryByText(/Dashboard/)).toBeNull()
  })

  it("still says 'Dashboard' on a page with no name of its own (the queries above can see it)", async () => {
    const greeting = await openWidgetOn('/docs')
    expect(greeting).toHaveTextContent('Ask me anything about Dashboard, or tell me what you need help with.')
  })
})

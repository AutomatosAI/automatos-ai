/**
 * PRD-260 M70 — the floating Auto widget names the page it floats over. On Play's
 * own pages (Conversations, Make it yours at /looks and its style sheet at
 * /looks/style) MainLayout used to fall through to 'dashboard', so the open
 * widget said "Ask me anything about Dashboard" and a bug report sent page
 * 'dashboard'. The real AutoWidget renders here; its data sources are stubbed.
 * The pages are drawn under a theme at a laptop's width, in its frame: the fixture theme
 * (PRD-261 A-19b), whose frame the theme host hands MainLayout with the widget.
 *
 * PRD-261 W-A8: MainLayout draws no widget on /chat and the pages under it, where the chat page is
 * Auto already, and draws it everywhere else (builtin Play's phone-layout test held this until it
 * left core; only core's MainLayout decides it).
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
// Under a theme at a laptop's width: its frame (P260-FX-25, PRD-261 A-02b: useThemeLayout is MainLayout's width
// source). The frame is the fixture theme's, from the theme host: the page and the widget MainLayout hands it.
vi.mock('@/hooks/use-studio-theme', async () => {
  const { FIXTURE_THEME_ENTRY } = await import('@/lib/theme-host/__tests__/fixture-theme-entry')
  return { useIsStudio: () => true, useActiveTheme: () => FIXTURE_THEME_ENTRY, useThemeLayout: () => 'theme' }
})
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
import { ThemeHostContext } from '@/components/theme-host/theme-host-context'
import { fixtureThemeHost } from '@/lib/theme-host/__tests__/fixture-theme-entry'

/** The widget's floating button, by its title. */
const WIDGET_BUTTON = 'Ask Auto'

/** MainLayout on `pathname`, around a page, in the fixture theme's frame. */
function renderLayoutOn(pathname: string) {
  state.pathname = pathname
  render(
    <ThemeHostContext.Provider value={fixtureThemeHost()}>
      <MainLayout>
        <div data-testid="page" />
      </MainLayout>
    </ThemeHostContext.Provider>,
  )
}

async function openWidgetOn(pathname: string) {
  renderLayoutOn(pathname)
  fireEvent.click(await screen.findByTitle(WIDGET_BUTTON))
  return screen.findByText(/^Ask me anything about/)
}

afterEach(() => {
  state.pathname = '/looks'
})

describe("the floating Auto widget on a theme's pages (PRD-260 M70)", () => {
  it.each(['/looks', '/looks/style'])("says 'Make it yours' on %s, not 'Dashboard'", async (pathname) => {
    const greeting = await openWidgetOn(pathname)
    expect(greeting).toHaveTextContent('Ask me anything about Make it yours, or tell me what you need help with.')
    expect(screen.queryByText(/Dashboard/)).toBeNull()
  })

  // PRD-261 A-04b, A-19f1: Conversations is a theme's route at /x/conversations (Play's package serves it); the old
  // address is the not-found page now, and the widget still names neither path the dashboard.
  it.each(['/conversations', '/x/conversations'])("says 'Chat' on Conversations (%s), not 'Dashboard'", async (pathname) => {
    const greeting = await openWidgetOn(pathname)
    expect(greeting).toHaveTextContent('Ask me anything about Chat, or tell me what you need help with.')
    expect(screen.queryByText(/Dashboard/)).toBeNull()
  })

  it("still says 'Dashboard' on a page with no name of its own (the queries above can see it)", async () => {
    const greeting = await openWidgetOn('/docs')
    expect(greeting).toHaveTextContent('Ask me anything about Dashboard, or tell me what you need help with.')
  })
})

describe('the floating Auto widget is hidden on /chat (PRD-261 W-A8)', () => {
  // The frame draws the page and the widget together, so once the page is there a widget would be too.
  it.each(['/chat', '/chat/c-42'])('draws no widget on %s, the page still drawn', async (pathname) => {
    renderLayoutOn(pathname)
    expect(await screen.findByTestId('page')).toBeInTheDocument()
    expect(screen.queryByTitle(WIDGET_BUTTON)).toBeNull()
    expect(screen.queryByText(/^Ask me anything about/)).toBeNull()
  })

  it.each(['/', '/docs', '/command-center', '/x/conversations'])('draws the widget on %s', async (pathname) => {
    renderLayoutOn(pathname)
    expect(await screen.findByTitle(WIDGET_BUTTON)).toBeInTheDocument()
    expect(screen.getByTestId('page')).toBeInTheDocument()
  })
})

/**
 * PRD-260 P260-FX-25 (M69) — one width source for Play. The five places that
 * pick Play's version or the core's (MainLayout, ThemeSlot through the
 * chat layout, the 404, the Play-only pages /looks and /looks/style, and
 * Conversations) read useThemeLayout, which knows the width on a browser
 * mount's first render. useIsTabletOrBelow starts false, so under Play on a
 * phone each of them used to mount Play's version for one render before the
 * width flipped it: Play's chat opened a linked conversation and the event
 * lane, Play's frame polled /health and asked for Recent, Conversations asked
 * for the list, and the style sheet drew Auto's 12 slots.
 *
 * Here the real switches, layouts and Play screens render in a phone-wide
 * window (window-width.ts) under the real UiStyleProvider; the slots are
 * Play's screens as they are once their chunks have loaded (a later visit,
 * when they render in the same pass), and the side effects are spies. A
 * laptop-wide run of the chat proves the spies see Play when it does mount.
 * On the server, where no width is known, the same trees render nothing of
 * Play. Roles, attributes and text only (CSS modules are proxies).
 */
import React, { type ReactNode } from 'react'
import { renderToString } from 'react-dom/server'
import { render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const nav = vi.hoisted(() => ({
  pathname: '/chat',
  search: '',
  router: { replace: vi.fn(), push: vi.fn(), prefetch: vi.fn() },
}))
/** Play's side effects, each a spy: none may run on a phone. */
const spies = vi.hoisted(() => ({
  openChat: vi.fn(),
  eventLane: vi.fn(),
  health: vi.fn(() => ({ connection: 'connected', check: () => undefined })),
  recentChats: vi.fn(() => []),
  chatHistory: vi.fn(async () => []),
  mascot: vi.fn(),
  lookEditor: vi.fn(),
}))

vi.mock('next/navigation', () => ({
  usePathname: () => nav.pathname,
  useSearchParams: () => new URLSearchParams(nav.search),
  useRouter: () => nav.router,
}))
vi.mock('@/lib/chat/api', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/lib/chat/api')>()),
  getChatHistory: spies.chatHistory,
  getChat: vi.fn(),
  getChatMessages: vi.fn(),
}))
// The session store as a stand-in: hydrated, nothing open; opening a conversation is the spy.
vi.mock('@/stores/chat-session-store', () => {
  const state = {
    hydrated: true,
    session: { activeChatId: null, openChatIds: [], draftOpen: false },
    openChat: spies.openChat,
    newDraft: vi.fn(),
    closeTab: vi.fn(),
    setTitles: vi.fn(),
  }
  const useChatSessionStore = Object.assign((select: (s: typeof state) => unknown) => select(state), {
    getState: () => state,
  })
  return { useChatSessionStore }
})
vi.mock('@/hooks/use-chat-session', () => ({
  useChatSessionHydration: () => ({ key: 'k', hydrated: true }),
  useChatScope: () => 'k',
}))
vi.mock('@/hooks/use-board-event-stream', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/hooks/use-board-event-stream')>()),
  useBoardEventStream: spies.eventLane,
}))
vi.mock('@/hooks/use-backend-health', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/hooks/use-backend-health')>()),
  useBackendHealth: spies.health,
}))
vi.mock('@/hooks/use-recent-chats', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/hooks/use-recent-chats')>()),
  useRecentChats: spies.recentChats,
}))
vi.mock('@/components/mascot/auto-mascot', () => ({
  AutoMascot: ({ state }: { state: string }) => {
    spies.mascot(state)
    return <div data-testid="auto" data-state={state} />
  },
}))
vi.mock('@/components/looks/look-editor', () => ({
  LookEditor: () => {
    spies.lookEditor()
    return <aside aria-label="Edit the look" />
  },
}))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: null, isLoading: false }),
  useWorkspaceOptional: () => null,
}))
vi.mock('@/components/local/first-run-nudge', () => ({ FirstRunNudge: () => null }))
vi.mock('@/components/chatbot/discussion-bar', () => ({ DiscussionBar: () => null }))

// The core's fallback chrome, its parts left out (their own tests cover them).
vi.mock('@/components/layout/sidebar', () => ({ Sidebar: () => null }))
vi.mock('@/components/layout/mobile-sidebar', () => ({ MobileSidebar: () => null }))
vi.mock('@/components/layout/header', () => ({ Header: () => <header data-testid="fallback-header" /> }))
vi.mock('@/components/layout/studio-sidebar', () => ({ StudioSidebar: () => null }))
vi.mock('@/components/layout/studio-header', () => ({ StudioHeader: () => null, StudioHeaderUtilities: () => null }))
vi.mock('@/components/chatbot/chat-widget', () => ({ AutoWidget: () => null }))
vi.mock('@/components/ui/sheet', () => ({
  Sheet: ({ children }: { children?: ReactNode }) => <>{children}</>,
  SheetContent: () => null,
}))

// Play's slots as they are once their chunks have loaded: the screen renders in the same pass.
// The chat's slot (builtin Play's, lib/theme-host/builtin/play-chat-lazy.tsx: the real PlayChat in
// MainLayout, full bleed, PRD-261 A-14a) reads them from a holder the imports below fill. Its factory
// may not import them itself: MainLayout reaches this slot through the builtin registrations
// (theme-slot-registrations -> builtin/slots -> play-slots, PRD-261 A-04b), so a factory awaiting
// MainLayout waits on the factory: the file never loads.
const loadedChat = vi.hoisted(() => ({ PlayChat: null as null | (() => React.ReactElement | null) }))
vi.mock('@/lib/theme-host/builtin/play-chat-lazy', () => ({
  PlayChat: () => {
    const Screen = loadedChat.PlayChat
    if (Screen === null) throw new Error('play-chat has not loaded: the test imports it below')
    return <Screen />
  },
}))
vi.mock('@/components/play/shell/play-frame-lazy', async () => ({
  PlayFrame: (await import('@/components/play/shell/play-frame')).PlayFrame,
}))
vi.mock('@/components/play/edges/lost-screen-lazy', async () => ({
  LostScreen: (await import('@/components/play/edges/lost-screen')).LostScreen,
}))

import ChatLayout from '@/app/chat/layout'
import LooksPage from '@/app/looks/page'
import LookStylePage from '@/app/looks/style/page'
import NotFound from '@/app/not-found'
import { MainLayout } from '@/components/layout/main-layout'
import { WIDER_SCREEN_NOTE } from '@/lib/theme-host/builtin/play-only-route'
import { MASCOT_LABEL } from '@/components/looks/style/style-mascots'
import { OPENING, PlayChat } from '@/components/play/chat/play-chat'
import { PlayConversations } from '@/components/play/conversations/play-conversations'
import { UiStyleProvider } from '@/contexts/ui-style-context'
import { LAPTOP_WIDTH_PX, PHONE_WIDTH_PX, stubWindowWidth, type WindowWidth } from '@/hooks/__tests__/window-width'
import { BUILTIN_PLAY_STYLE } from '@/lib/theme-host/builtin/play-entry'
import { APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION } from '@/lib/ui-style'

loadedChat.PlayChat = () => (
  <MainLayout fullBleed>
    <PlayChat />
  </MainLayout>
)

const CHAT_PAGE = 'The core chat page'
const STUDIO_PAGE = 'A Studio page Play has no design for'
const PLAIN_404 = 'This page could not be found.'

let width: WindowWidth | null = null

/** Under Play, with a query client (Conversations' list and Recent read through react-query). */
function underPlay(ui: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return (
    <QueryClientProvider client={client}>
      <UiStyleProvider initialStyle={BUILTIN_PLAY_STYLE}>{ui}</UiStyleProvider>
    </QueryClientProvider>
  )
}

function chatRoute() {
  return (
    <ChatLayout>
      <p>{CHAT_PAGE}</p>
    </ChatLayout>
  )
}

/** No spy heard Play: nothing of Play mounted, not even for a render. */
function expectNothingOfPlay() {
  expect(spies.eventLane, "Play's chat holds the event lane open").not.toHaveBeenCalled()
  expect(spies.openChat, "Play's chat opens the linked conversation").not.toHaveBeenCalled()
  expect(spies.health, "Play's frame asks /health").not.toHaveBeenCalled()
  expect(spies.recentChats, "Play's sidebar asks for Recent").not.toHaveBeenCalled()
  expect(spies.chatHistory, 'Conversations or the cold start asks for the history').not.toHaveBeenCalled()
  expect(spies.mascot, 'Auto shows on a Play screen').not.toHaveBeenCalled()
  expect(spies.lookEditor, "Make it yours' editor").not.toHaveBeenCalled()
}

beforeEach(() => {
  // A browser already moved to the current defaults, so the provider keeps Play.
  window.localStorage.setItem(APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION)
  Object.assign(nav, { pathname: '/chat', search: '' })
})

afterEach(() => {
  width?.restore()
  width = null
  window.localStorage.clear()
  vi.clearAllMocks()
})

describe('under Play on a phone, nothing of Play mounts, not even for a render (M69)', () => {
  beforeEach(() => {
    width = stubWindowWidth(PHONE_WIDTH_PX)
  })

  it("the chat layout shows the core's chat: the linked conversation is not opened, the event lane and Play's frame never start", () => {
    nav.search = 'chatId=c1'
    render(underPlay(chatRoute()))
    expect(screen.getByText(CHAT_PAGE)).toBeInTheDocument()
    expect(nav.router.replace).not.toHaveBeenCalled()
    expectNothingOfPlay()
  })

  it("MainLayout keeps the fallback chrome: Play's frame never polls /health or asks for Recent", () => {
    nav.pathname = '/command-center'
    render(
      underPlay(
        <MainLayout>
          <p>{STUDIO_PAGE}</p>
        </MainLayout>,
      ),
    )
    expect(screen.getByTestId('fallback-header')).toBeInTheDocument()
    expect(screen.getByText(STUDIO_PAGE)).toBeInTheDocument()
    expectNothingOfPlay()
  })

  it("/looks/style says it opens on a wider screen, and Auto's 12 mascot slots are never drawn", () => {
    // No LookProvider: the look reads as already read, as on any page after the first.
    nav.pathname = '/looks/style'
    render(underPlay(<LookStylePage />))
    expect(screen.getByText(WIDER_SCREEN_NOTE)).toBeInTheDocument()
    expect(screen.queryByRole('region', { name: MASCOT_LABEL })).toBeNull()
    expectNothingOfPlay()
  })

  it('/looks says so too, and the editor aside never mounts', () => {
    nav.pathname = '/looks'
    render(underPlay(<LooksPage />))
    expect(screen.getByText(WIDER_SCREEN_NOTE)).toBeInTheDocument()
    expect(screen.queryByRole('complementary', { name: 'Edit the look' })).toBeNull()
    expectNothingOfPlay()
  })

  it('Conversations sends the browser to /chat without asking for the list', () => {
    nav.pathname = '/conversations'
    render(underPlay(<PlayConversations />))
    expect(nav.router.replace).toHaveBeenCalledWith('/chat')
    expect(screen.queryByRole('searchbox')).toBeNull()
    expectNothingOfPlay()
  })

  it('the 404 is the plain page, Auto never lost on it', () => {
    nav.pathname = '/nowhere'
    render(underPlay(<NotFound />))
    expect(screen.getByText(PLAIN_404)).toBeInTheDocument()
    expectNothingOfPlay()
  })
})

describe('under Play on a laptop, the same trees are Play’s (the spies see Play when it mounts)', () => {
  beforeEach(() => {
    width = stubWindowWidth(LAPTOP_WIDTH_PX)
  })

  it("the chat layout is Play's chat in Play's frame, opening the linked conversation", () => {
    nav.search = 'chatId=c1'
    render(underPlay(chatRoute()))
    expect(screen.getByText(OPENING)).toBeInTheDocument()
    expect(screen.queryByText(CHAT_PAGE)).toBeNull()
    expect(spies.eventLane).toHaveBeenCalled()
    expect(spies.health).toHaveBeenCalled()
    expect(spies.recentChats).toHaveBeenCalled()
    expect(spies.openChat).toHaveBeenCalledWith('c1')
  })
})

describe('under Play on the server, where no width is known, the five render nothing of Play', () => {
  it("the chat layout renders neither Play's chat nor the core's page", () => {
    nav.search = 'chatId=c1'
    const html = renderToString(underPlay(chatRoute()))
    expect(html).not.toContain(CHAT_PAGE)
    expect(html).not.toContain(OPENING)
    expectNothingOfPlay()
  })

  it('MainLayout, the Play-only pages, Conversations and the 404 render nothing', () => {
    const pages: ReadonlyArray<readonly [string, ReactNode]> = [
      ['/command-center', <MainLayout key="layout">{STUDIO_PAGE}</MainLayout>],
      ['/looks/style', <LookStylePage key="style" />],
      ['/looks', <LooksPage key="looks" />],
      ['/conversations', <PlayConversations key="conversations" />],
      ['/nowhere', <NotFound key="404" />],
    ]
    for (const [pathname, page] of pages) {
      nav.pathname = pathname
      expect(renderToString(underPlay(page)), pathname).toBe('')
    }
    expectNothingOfPlay()
  })
})

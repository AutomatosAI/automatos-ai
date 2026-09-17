/**
 * PRD-237 S2 — the chat page resumes the active conversation.
 *
 * Heavy children are stubbed; the session store and its model are real, so
 * these prove the wiring: stored pointer → that chat; stored draft → a new
 * chat with no fetch; nothing stored → the most recent conversation; a deep
 * link wins and is then dropped from the URL; a reply still in flight
 * server-side reaches <Chat> as `initialAwaitingReply`.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

const nav = vi.hoisted(() => ({ search: '', push: vi.fn(), replace: vi.fn() }))
vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: nav.push, replace: nav.replace }),
  useSearchParams: () => new URLSearchParams(nav.search),
  usePathname: () => '/chat',
}))
vi.mock('@/components/layout/main-layout', () => ({
  MainLayout: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}))
vi.mock('@/hooks/use-page-api', () => ({ usePageAPI: () => {} }))
vi.mock('@/hooks/use-mobile', () => ({ useIsMobile: () => false, useIsTabletOrBelow: () => false }))
vi.mock('@/hooks/use-studio-theme', () => ({ useIsStudio: () => false }))
vi.mock('@/components/local/first-run-nudge', () => ({ FirstRunNudge: () => null }))
vi.mock('@/components/chatbot/sidebar', () => ({ AppSidebar: () => <div data-testid="history" /> }))
vi.mock('@/components/chatbot/studio-chat-shell', () => ({
  StudioChatShell: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}))
vi.mock('@/components/chatbot/chat', () => ({
  Chat: (props: { id: string; initialMessages?: unknown[]; initialAwaitingReply?: boolean }) => (
    <div
      data-testid="chat"
      data-id={props.id}
      data-count={props.initialMessages?.length ?? 0}
      data-awaiting={String(Boolean(props.initialAwaitingReply))}
    />
  ),
}))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspaceOptional: () => ({ workspace: { id: 'ws-1' }, isLoading: false }),
}))
vi.mock('@/lib/auth-hooks', () => ({
  useUser: () => ({ user: { id: 'u-1' }, isLoaded: true }),
  useAuth: () => ({ isLoaded: true, getToken: async () => null }),
}))
const api = vi.hoisted(() => ({
  getChat: vi.fn(),
  getChatMessages: vi.fn(),
  getChatHistory: vi.fn(),
  getChatSession: vi.fn(),
  putChatSession: vi.fn(),
  cancelChatTurn: vi.fn(),
}))
vi.mock('@/lib/chat/api', () => api)

import ChatPage from '@/app/chat/page'
import { EMPTY_CHAT_SESSION, chatSessionKey, saveChatSession, withChatOpened, withDraft } from '@/lib/chat/chat-session'
import { useChatSessionStore } from '@/stores/chat-session-store'

const KEY = chatSessionKey('ws-1', 'u-1')
const row = (id: string, extra: Record<string, unknown> = {}) => ({
  id,
  userId: 'u-1',
  title: `Chat ${id}`,
  createdAt: '2026-09-07T10:00:00Z',
  updatedAt: '2026-09-07T10:00:00Z',
  visibility: 'private',
  ...extra,
})

beforeEach(() => {
  localStorage.clear()
  vi.clearAllMocks()
  nav.search = ''
  useChatSessionStore.setState({ key: null, hydrated: false, session: EMPTY_CHAT_SESSION })
  api.getChatSession.mockResolvedValue({ activeChatId: null, draftOpen: false, openChatIds: [], lastReadAt: {}, updatedAt: null })
  api.putChatSession.mockResolvedValue(undefined)
  api.getChatHistory.mockResolvedValue([])
  api.getChat.mockImplementation(async (id: string) => row(id))
  api.getChatMessages.mockResolvedValue([{ id: 'm1', role: 'user', content: 'hi', parts: [] }])
})

afterEach(cleanup)

describe('ChatPage resume', () => {
  it('reopens the stored active conversation with its messages', async () => {
    saveChatSession(KEY, withChatOpened(EMPTY_CHAT_SESSION, 'c1', Date.now()))
    render(<ChatPage />)
    const chat = await screen.findByTestId('chat')
    await waitFor(() => expect(chat).toHaveAttribute('data-id', 'c1'))
    expect(chat).toHaveAttribute('data-count', '1')
    expect(api.getChat).toHaveBeenCalledWith('c1')
    expect(api.getChatMessages).toHaveBeenCalledWith('c1')
    expect(api.getChatHistory).not.toHaveBeenCalledWith(1)
  })

  it('a stored draft (New Chat, then reload) stays a new chat — nothing fetched', async () => {
    saveChatSession(KEY, withDraft(withChatOpened(EMPTY_CHAT_SESSION, 'c1', 1), 2))
    render(<ChatPage />)
    const chat = await screen.findByTestId('chat')
    expect(chat).toHaveAttribute('data-id', '')
    expect(api.getChatMessages).not.toHaveBeenCalled()
    expect(api.getChatHistory).not.toHaveBeenCalledWith(1)
    expect(screen.getByRole('tab', { selected: true })).toHaveTextContent('New chat')
  })

  it('a device with nothing stored lands on the most recent conversation', async () => {
    api.getChatHistory.mockImplementation(async (limit: number) => (limit === 1 ? [row('recent')] : [row('recent'), row('older')]))
    render(<ChatPage />)
    await waitFor(() => expect(screen.getByTestId('chat')).toHaveAttribute('data-id', 'recent'))
    expect(api.getChatHistory).toHaveBeenCalledWith(1)
  })

  it('a ?chatId= deep link opens that conversation and is dropped from the URL', async () => {
    saveChatSession(KEY, withChatOpened(EMPTY_CHAT_SESSION, 'c1', Date.now()))
    nav.search = 'chatId=deep&mode=plan'
    render(<ChatPage />)
    await waitFor(() => expect(screen.getByTestId('chat')).toHaveAttribute('data-id', 'deep'))
    expect(nav.replace).toHaveBeenCalledWith('/chat?mode=plan')
    expect(screen.getAllByRole('tab')).toHaveLength(2) // c1 stays open, deep is added
  })

  it('a reply still being produced server-side shows as awaiting', async () => {
    saveChatSession(KEY, withChatOpened(EMPTY_CHAT_SESSION, 'busy', Date.now()))
    api.getChat.mockImplementation(async (id: string) => row(id, { turnInFlight: true }))
    render(<ChatPage />)
    await waitFor(() => expect(screen.getByTestId('chat')).toHaveAttribute('data-awaiting', 'true'))
  })

  it('a stale pointer (conversation gone) drops its tab instead of hanging', async () => {
    saveChatSession(KEY, withChatOpened(withChatOpened(EMPTY_CHAT_SESSION, 'ok', 1), 'gone', 2))
    api.getChat.mockImplementation(async (id: string) => {
      if (id === 'gone') throw new Error('404')
      return row(id)
    })
    render(<ChatPage />)
    await waitFor(() => expect(screen.getByTestId('chat')).toHaveAttribute('data-id', 'ok'))
    expect(useChatSessionStore.getState().session.openChatIds).toEqual(['ok'])
  })
})

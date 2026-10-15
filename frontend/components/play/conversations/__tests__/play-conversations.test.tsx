/**
 * PRD-260 T-006 — Play's Conversations page over a mocked lib/chat/api
 * getChatHistory, ported from Automatos Play's
 * tests/conversations/ConversationsScreen.test.tsx: grouping with a fixed now,
 * each row's title, time and newest line, the title search, New chat (a draft,
 * then /chat), a row's link (/chat?chatId=<id>), the empty, loading and error
 * states; and the page sending anyone not on Play at >= 1024 px to /chat.
 * Roles, attributes and text only (CSS modules are proxies under css: false).
 * "Now" is pinned to a local date, so it passes in any time zone.
 */
import React from 'react'
import { cleanup, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const nav = vi.hoisted(() => ({ push: vi.fn(), replace: vi.fn() }))
vi.mock('next/navigation', () => ({
  useRouter: () => nav,
  usePathname: () => '/conversations',
}))

const store = vi.hoisted(() => ({ newDraft: vi.fn() }))
vi.mock('@/stores/chat-session-store', () => ({
  useChatSessionStore: (select: (state: typeof store) => unknown) => select(store),
}))
// useStartChat binds the chat session store itself (P260-FX-03): hydrated here.
vi.mock('@/hooks/use-chat-session', () => ({ useChatSessionHydration: () => ({ key: 'k', hydrated: true }) }))

const chatApi = vi.hoisted(() => ({ getChatHistory: vi.fn() }))
vi.mock('@/lib/chat/api', () => chatApi)

const style = vi.hoisted(() => ({ isPlay: true, narrow: false }))
vi.mock('@/hooks/use-studio-theme', () => ({ useIsPlay: () => style.isPlay, useIsStudio: () => true }))
vi.mock('@/hooks/use-mobile', () => ({ useIsTabletOrBelow: () => style.narrow, useIsMobile: () => style.narrow }))

vi.mock('@/components/mascot/auto-mascot', () => ({
  AutoMascot: ({ state }: { state: string }) => <div data-testid="auto" data-state={state} />,
}))

import { CONVERSATION_LIMIT } from '@/hooks/use-conversation-list'
import {
  CONVERSATIONS_TITLE,
  LIST_FAILED,
  NEW_CHAT,
  NO_CONVERSATIONS,
  PlayConversations,
  SEARCH_LABEL,
} from '../play-conversations'

const NOW = new Date(2026, 9, 14, 15, 30) // Wed 14 Oct 2026, 15:30

function at(day: number, hour: number, minute = 0): string {
  return new Date(2026, 9, day, hour, minute).toISOString()
}

function row(id: string, title: string, createdAt: string, updatedAt: string, lastMessagePreview: string | null) {
  return { id, userId: 'u1', title, createdAt, updatedAt, visibility: 'private', lastMessagePreview, kind: 'user' }
}

const HISTORY = [
  row('c-laptop', 'Laptop for uni', at(14, 9), at(14, 15, 28), 'Done! Three laptops make the cut.'),
  row('c-chem', 'Revision plan for chemistry', at(13, 9), at(13, 18), 'Week 1 covers bonding and moles.'),
  row('c-bake', 'Instagram captions for the bake sale', at(12, 9), at(12, 10), null),
  row('c-cafe', 'Cover letter for the café job', at(1, 9), at(1, 10), 'I tightened the opening line.'),
]

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const user = userEvent.setup()
  render(
    <QueryClientProvider client={client}>
      <PlayConversations now={NOW} />
    </QueryClientProvider>,
  )
  return { user }
}

function group(name: string) {
  return within(screen.getByRole('region', { name }))
}

beforeEach(() => {
  chatApi.getChatHistory.mockResolvedValue(HISTORY)
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  vi.restoreAllMocks()
  style.isPlay = true
  style.narrow = false
})

describe('PlayConversations — the list', () => {
  it('groups the history into Today, This week and Earlier, asked for once through getChatHistory', async () => {
    renderPage()
    expect(screen.getByRole('status')).toHaveTextContent('Loading your conversations')
    await screen.findByRole('region', { name: 'Today' })

    expect(screen.getAllByRole('heading', { level: 2 }).map((heading) => heading.textContent)).toEqual([
      'Today',
      'This week',
      'Earlier',
    ])
    expect(group('Today').getAllByRole('link')).toHaveLength(1)
    expect(group('This week').getAllByRole('link')).toHaveLength(2)
    expect(group('Earlier').getByRole('link')).toHaveAccessibleName(/Cover letter for the café job/)
    expect(chatApi.getChatHistory).toHaveBeenCalledTimes(1)
    expect(chatApi.getChatHistory).toHaveBeenCalledWith(CONVERSATION_LIMIT)
    expect(screen.getByRole('heading', { level: 1, name: 'Conversations' })).toBeInTheDocument()
    expect(screen.getByTestId('auto')).toHaveAttribute('data-state', 'idle')
  })

  it("shows each conversation's title, relative time and newest line", async () => {
    renderPage()
    const laptop = await screen.findByRole('link', { name: /Laptop for uni/ })
    expect(laptop).toHaveTextContent('2 min')
    expect(laptop).toHaveTextContent('Done! Three laptops make the cut.')
    expect(laptop).not.toHaveTextContent('Auto:')
    expect(group('This week').getByRole('link', { name: /Revision plan/ })).toHaveTextContent('Yesterday')
    expect(group('This week').getByRole('link', { name: /bake sale/ })).toHaveTextContent('Mon')
    expect(group('Earlier').getByRole('link')).toHaveTextContent('1 Oct')
  })

  it('opens a conversation in the chat by its deep link', async () => {
    renderPage()
    await screen.findByRole('region', { name: 'Today' })
    expect(group('Today').getByRole('link')).toHaveAttribute('href', '/chat?chatId=c-laptop')
    expect(screen.getByRole('link', { name: /café job/ })).toHaveAttribute('href', '/chat?chatId=c-cafe')
  })

  it('filters by title as you type, and says when nothing matches', async () => {
    const { user } = renderPage()
    await screen.findByRole('link', { name: /Laptop for uni/ })
    const search = screen.getByRole('searchbox', { name: 'Search conversations' })

    await user.type(search, 'LAPTOP')
    expect(screen.getAllByRole('link')).toHaveLength(1)
    expect(screen.getByRole('link', { name: /Laptop for uni/ })).toBeInTheDocument()
    expect(screen.queryByRole('region', { name: 'This week' })).not.toBeInTheDocument()

    await user.clear(search)
    await user.type(search, 'lisbon')
    expect(screen.queryByRole('link')).not.toBeInTheDocument()
    expect(screen.getByText(/No conversation titles match “lisbon”/)).toBeInTheDocument()

    await user.clear(search)
    expect(screen.getAllByRole('region')).toHaveLength(3)
  })

  it("heads the list's own column with the title and New chat, over the search box, as in the Play app (P260-FX-14)", async () => {
    renderPage()
    await screen.findByRole('region', { name: 'Today' })
    const search = screen.getByRole('searchbox', { name: SEARCH_LABEL })
    const column = search.parentElement as HTMLElement
    const heading = within(column).getByRole('heading', { level: 1, name: CONVERSATIONS_TITLE })
    const newChat = within(column).getByRole('button', { name: NEW_CHAT })
    const titleRow = column.firstElementChild
    expect(titleRow).toContainElement(heading)
    expect(titleRow).toContainElement(newChat)
    expect(titleRow?.nextElementSibling).toBe(search)
  })

  it('starts a new chat: a fresh draft, then the chat', async () => {
    const { user } = renderPage()
    await screen.findByRole('region', { name: 'Today' })
    await user.click(screen.getByRole('button', { name: 'New chat' }))
    expect(store.newDraft).toHaveBeenCalledTimes(1)
    expect(nav.push).toHaveBeenCalledWith('/chat')
  })
})

describe('PlayConversations — empty and failed', () => {
  it('says so when there are no conversations yet', async () => {
    chatApi.getChatHistory.mockResolvedValue([])
    renderPage()
    expect(await screen.findByText(NO_CONVERSATIONS)).toBeInTheDocument()
    expect(screen.getByText(/No conversations yet/)).toBeInTheDocument()
    expect(screen.queryByRole('region')).not.toBeInTheDocument()
  })

  it('shows a clear error with no internals when the history fails, then loads on Try again', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    chatApi.getChatHistory
      .mockRejectedValueOnce(new Error('HTTP 502: upstream connect error at http://10.0.0.7:8000'))
      .mockResolvedValue(HISTORY)
    const { user } = renderPage()

    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent(LIST_FAILED)
    expect(alert).not.toHaveTextContent(/502|10\.0\.0\.7|upstream/)
    expect(console.error).toHaveBeenCalled()

    await user.click(within(alert).getByRole('button', { name: 'Try again' }))
    expect(await screen.findByRole('link', { name: /Laptop for uni/ })).toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    expect(chatApi.getChatHistory).toHaveBeenCalledTimes(2)
  })
})

describe('PlayConversations — only under Play at >= 1024 px', () => {
  it('sends the browser to /chat when the style is not Play, showing nothing', () => {
    style.isPlay = false
    renderPage()
    expect(nav.replace).toHaveBeenCalledWith('/chat')
    expect(screen.queryByRole('heading', { name: 'Conversations' })).not.toBeInTheDocument()
    expect(chatApi.getChatHistory).not.toHaveBeenCalled()
  })

  it('sends the browser to /chat below 1024 px, where Play pages do not render (D7)', () => {
    style.narrow = true
    renderPage()
    expect(nav.replace).toHaveBeenCalledWith('/chat')
    expect(screen.queryByRole('searchbox')).not.toBeInTheDocument()
  })

  it('stays put under Play on a desktop', async () => {
    renderPage()
    await screen.findByRole('region', { name: 'Today' })
    expect(nav.replace).not.toHaveBeenCalled()
  })
})

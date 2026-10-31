/**
 * PRD-237 — a cold start gives way to a chat the person started themselves.
 *
 * On a device with nothing stored the page asks for the newest conversation and,
 * while that answer is on its way, already shows a blank chat with the composer.
 * Only an EMPTY draft gives way: once the person types there, or sends from it,
 * the late answer must not replace their chat. Untouched, the blank chat still
 * lands on the newest conversation, and with no history it becomes a new chat.
 *
 * The page, the session store, the real composer (multimodal-input.tsx) and the
 * page's chat hook (use-studio-chat.ts) are real; `Chat` is a stand-in that wires
 * the composer and a one-tap prompt to that hook as chat.tsx does.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
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
vi.mock('@/components/chatbot/auto-now-rail', () => ({ AutoNowRail: () => null }))
vi.mock('@/components/chatbot/auto-now-pill', () => ({ AutoNowPill: () => null }))
vi.mock('@/hooks/use-auto-now', () => ({ useAutoNowOpen: () => [false, () => {}] }))
vi.mock('@/components/chatbot/sidebar', () => ({ AppSidebar: () => <div data-testid="history" /> }))
vi.mock('@/components/chatbot/studio-chat-shell', () => ({
  StudioChatShell: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}))
// The composer's agent picker loads agents through react-query; not this test's concern.
vi.mock('@/components/chatbot/agent-selector', () => ({ AgentSelector: () => null }))
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

// The stream under the page's chat hook: a send is recorded, nothing goes out.
const turn = vi.hoisted(() => ({ send: vi.fn() }))
vi.mock('@/lib/chat/use-chat-with-receipts', () => ({
  useChatWithReceipts: () => ({ sendMessage: turn.send, errorCode: null, messages: [], reload: vi.fn() }),
}))
vi.mock('@/hooks/use-billing-summary', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/hooks/use-billing-summary')>()
  return { ...actual, useBillingSummary: () => ({ data: undefined }) }
})

// As chat.tsx wires them: the composer and the welcome screen's one-tap prompt both send through the page's chat hook.
const words = vi.hoisted(() => ({ prompt: 'Where did we leave off?' }))
vi.mock('@/components/chatbot/chat', async () => {
  const { MultimodalInput } = await import('@/components/chatbot/multimodal-input')
  const { useStudioChat } = await import('@/lib/chat/use-studio-chat')
  function Chat({ id }: { id: string }) {
    const chat = useStudioChat({ id })
    return (
      <div data-testid="chat" data-id={id}>
        <button type="button" onClick={() => void chat.sendMessage(words.prompt)}>
          {words.prompt}
        </button>
        <MultimodalInput chatId={id} status="ready" stop={() => {}} sendMessage={chat.sendMessage} selectedVisibilityType="private" />
      </div>
    )
  }
  return { Chat }
})

import ChatPage from '@/app/chat/page'
import { EMPTY_CHAT_SESSION } from '@/lib/chat/chat-session'
import { useChatSessionStore } from '@/stores/chat-session-store'

const PLACEHOLDER = 'Send a message...'
const PROMPT = words.prompt
const WORDS = 'Draft the launch note for Friday'

interface HistoryRow {
  id: string
  title: string
}

const row = (id: string): HistoryRow => ({ id, title: `Chat ${id}` })

/** The cold start's question (the newest conversation) waits for the test to answer it. */
function newestConversationPending() {
  let answer: (rows: HistoryRow[]) => void = () => {}
  const pending = new Promise<HistoryRow[]>((resolve) => {
    answer = resolve
  })
  // getChatHistory(1) is the cold start; the tab titles' read (20 rows) is answered at once.
  api.getChatHistory.mockImplementation(async (limit: number) => (limit === 1 ? pending : []))
  return {
    /** The answer lands, and everything it would set going settles. */
    arrives: async (rows: HistoryRow[]) => {
      await act(async () => {
        answer(rows)
        await pending
      })
      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0))
      })
    },
  }
}

/** The page is up on a fresh device: the blank chat is on screen and the cold start is asking. */
async function blankChatWhileAsking(): Promise<HTMLElement> {
  render(<ChatPage />)
  const box = await screen.findByPlaceholderText(PLACEHOLDER)
  await waitFor(() => expect(api.getChatHistory).toHaveBeenCalledWith(1))
  expect(screen.getByTestId('chat')).toHaveAttribute('data-id', '')
  return box
}

const session = () => useChatSessionStore.getState().session

beforeEach(() => {
  localStorage.clear()
  vi.clearAllMocks()
  nav.search = ''
  useChatSessionStore.setState({ key: null, hydrated: false, session: EMPTY_CHAT_SESSION })
  api.getChatSession.mockResolvedValue({ activeChatId: null, draftOpen: false, openChatIds: [], lastReadAt: {}, updatedAt: null })
  api.putChatSession.mockResolvedValue(undefined)
  api.getChat.mockImplementation(async (id: string) => ({ ...row(id), visibility: 'private' }))
  api.getChatMessages.mockResolvedValue([])
})

afterEach(cleanup)

describe('ChatPage cold start — only an empty draft gives way', () => {
  it('the person typed: the newest conversation arriving late opens nothing, and the words stay', async () => {
    const newest = newestConversationPending()
    const box = await blankChatWhileAsking()

    fireEvent.change(box, { target: { value: WORDS } })
    await newest.arrives([row('recent')])

    expect(session()).toMatchObject({ activeChatId: null, draftOpen: true, openChatIds: [] })
    expect(api.getChat).not.toHaveBeenCalledWith('recent')
    expect(screen.getByTestId('chat')).toHaveAttribute('data-id', '')
    expect(screen.getByPlaceholderText(PLACEHOLDER)).toHaveValue(WORDS)
  })

  it('the person sent (a one-tap prompt, nothing typed): their chat stays, and takes the name the server gives it', async () => {
    const newest = newestConversationPending()
    await blankChatWhileAsking()

    fireEvent.click(screen.getByRole('button', { name: PROMPT }))
    expect(turn.send).toHaveBeenCalledWith(PROMPT)
    await newest.arrives([row('recent')])

    expect(session()).toMatchObject({ activeChatId: null, draftOpen: true, openChatIds: [] })
    expect(api.getChat).not.toHaveBeenCalledWith('recent')
    expect(screen.getByTestId('chat')).toHaveAttribute('data-id', '')

    // The stream names the new chat: that conversation is the one on screen, and the newest one was never opened.
    act(() => useChatSessionStore.getState().chatIdAssigned('mine'))
    await waitFor(() => expect(screen.getByTestId('chat')).toHaveAttribute('data-id', 'mine'))
    expect(session().openChatIds).toEqual(['mine'])
  })

  it('untouched: the blank chat still gives way to the newest conversation', async () => {
    const newest = newestConversationPending()
    await blankChatWhileAsking()

    await newest.arrives([row('recent')])

    await waitFor(() => expect(screen.getByTestId('chat')).toHaveAttribute('data-id', 'recent'))
    expect(session()).toMatchObject({ activeChatId: 'recent', draftOpen: false, openChatIds: ['recent'] })
  })

  it('untouched with no conversations yet: the blank chat becomes a new chat', async () => {
    const newest = newestConversationPending()
    await blankChatWhileAsking()

    await newest.arrives([])

    expect(session()).toMatchObject({ activeChatId: null, draftOpen: true, openChatIds: [] })
    expect(screen.getByTestId('chat')).toHaveAttribute('data-id', '')
    expect(screen.getByRole('tab', { selected: true })).toHaveTextContent('New chat')
    expect(turn.send).not.toHaveBeenCalled()
  })
})

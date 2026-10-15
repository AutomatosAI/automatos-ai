/**
 * PRD-260 §4, D5, D6 — Play's chat on the core's data. The chat hook
 * (useChatWithReceipts) and the session store are mocked; the conversation
 * loading, Auto's reducer (useAutoMascot) and the markdown renderer are real.
 * Auto himself is a stand-in that shows his state and caption (his clips have
 * their own tests). Roles and text only (CSS modules are proxies).
 */
import React from 'react'
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { Chat, ChatMessage, ToolCall } from '@/types'

const nav = vi.hoisted(() => ({ search: '', replace: vi.fn(), push: vi.fn() }))
const chat = vi.hoisted(() => ({
  options: null as null | Record<string, any>,
  status: 'idle' as 'idle' | 'streaming' | 'error',
  isLoading: false,
  messages: null as null | unknown[],
  sendMessage: vi.fn(),
  stop: vi.fn(),
  reload: vi.fn(),
}))
const api = vi.hoisted(() => ({ getChat: vi.fn(), getChatMessages: vi.fn(), getChatHistory: vi.fn() }))
const toastError = vi.hoisted(() => vi.fn())

vi.mock('next/navigation', () => ({
  usePathname: () => '/chat',
  useSearchParams: () => new URLSearchParams(nav.search),
  useRouter: () => nav,
}))
vi.mock('sonner', () => ({ toast: { error: toastError } }))
vi.mock('@/lib/chat/api', () => api)
vi.mock('@/lib/page-context', () => ({ usePageContext: () => ({ page: 'chat', route: '/chat' }) }))
vi.mock('@/hooks/use-chat-session', () => ({ useChatSessionHydration: () => ({ key: 'k', hydrated: true }) }))
vi.mock('@/hooks/use-backend-health', () => ({ useBackendHealth: () => ({ connection: 'connected', check: vi.fn() }) }))
// PlayChat holds the chat_changed SSE lane open (P260-FX-02); no stream and no QueryClient here.
vi.mock('@/hooks/use-board-event-stream', () => ({ useBoardEventStream: vi.fn() }))
vi.mock('@/components/layout/main-layout', () => ({
  MainLayout: ({ children }: { children?: React.ReactNode }) => <div data-testid="frame">{children}</div>,
}))
vi.mock('@/components/local/first-run-nudge', () => ({ FirstRunNudge: () => null }))
// T-005b: the chat shows the core's trial banner off the workspace snapshot; no snapshot here.
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: null }),
  useWorkspaceOptional: () => null,
}))
vi.mock('@/components/mascot/auto-mascot', () => ({
  AutoMascot: ({ state, caption, detail }: { state: string; caption?: string; detail?: string }) => (
    <div data-testid="auto" data-state={state}>
      {caption} {detail}
    </div>
  ),
}))
vi.mock('@/lib/chat/use-chat-with-receipts', () => ({
  useChatWithReceipts: (options: Record<string, any>) => {
    chat.options = options
    return {
      messages: chat.messages ?? options.initialMessages,
      setMessages: vi.fn(),
      sendMessage: chat.sendMessage,
      reload: chat.reload,
      stop: chat.stop,
      status: chat.status,
      isLoading: chat.isLoading,
      awaitingReply: false,
      usage: undefined,
      errorCode: null,
    }
  },
}))
vi.mock('@/stores/chat-session-store', async () => {
  const { create } = await import('zustand')
  type Session = { activeChatId: string | null; draftOpen: boolean; openChatIds: string[] }
  const open = (session: Session, id: string): Session => ({
    ...session,
    activeChatId: id,
    draftOpen: false,
    openChatIds: session.openChatIds.includes(id) ? session.openChatIds : [...session.openChatIds, id],
  })
  const store = create<any>((set) => ({
    hydrated: true,
    session: { activeChatId: null, draftOpen: true, openChatIds: [] },
    openChat: vi.fn((id: string) => set((s: { session: Session }) => ({ session: open(s.session, id) }))),
    chatIdAssigned: vi.fn((id: string) => set((s: { session: Session }) => ({ session: open(s.session, id) }))),
    newDraft: vi.fn(() => set((s: { session: Session }) => ({ session: { ...s.session, activeChatId: null, draftOpen: true } }))),
    closeTab: vi.fn((id: string) =>
      set((s: { session: Session }) => ({
        session: { ...s.session, activeChatId: null, draftOpen: true, openChatIds: s.session.openChatIds.filter((x) => x !== id) },
      })),
    ),
    setTitles: vi.fn(),
  }))
  return { useChatSessionStore: store }
})

import { PlayChat } from '@/components/play/chat/play-chat'
import { EXAMPLES, HELLO_TITLE } from '@/components/play/chat/first-open'
import { PLACEHOLDER } from '@/components/play/chat/message-box'
import { useBoardEventStream } from '@/hooks/use-board-event-stream'
import { EMPTY_CHAT_SESSION } from '@/lib/chat/chat-session'
import { HELLO_KEY } from '@/lib/mascot/session'
import { useChatSessionStore } from '@/stores/chat-session-store'

// --- core fixtures: the ChatMessage shape of types/chat.ts --------------------------------

function userMessage(id: string, text: string): ChatMessage {
  return { id, role: 'user', content: text, parts: [{ type: 'text', text }] } as unknown as ChatMessage
}

function autoMessage(id: string, text: string, extra: Partial<ChatMessage> = {}): ChatMessage {
  return { id, role: 'assistant', content: text, parts: [{ type: 'text', text }], ...extra } as unknown as ChatMessage
}

function storedChat(id: string, title: string): Chat {
  return { id, title, userId: 'u1', createdAt: '', updatedAt: '', visibility: 'private', turnInFlight: false }
}

const STORED = [userMessage('m1', 'What should I look for in a laptop?'), autoMessage('m2', 'Aim for **16 GB of memory**.')]

function setSession(pointer: { activeChatId: string | null; draftOpen: boolean; openChatIds: string[] }) {
  useChatSessionStore.setState({ session: { ...EMPTY_CHAT_SESSION, ...pointer } })
}

function openStoredChat(messages: ChatMessage[] = STORED) {
  setSession({ activeChatId: 'c1', draftOpen: false, openChatIds: ['c1'] })
  api.getChat.mockResolvedValue(storedChat('c1', 'Laptop for uni'))
  api.getChatMessages.mockResolvedValue(messages)
}

const auto = () => screen.getByTestId('auto')
const box = () => screen.getByRole('textbox', { name: PLACEHOLDER })

beforeEach(() => {
  window.sessionStorage.setItem(HELLO_KEY, '1') // hello already played this session
  setSession({ activeChatId: null, draftOpen: true, openChatIds: [] })
  Object.values(api).forEach((fn) => fn.mockReset())
  api.getChatHistory.mockResolvedValue([])
})

afterEach(() => {
  vi.clearAllMocks()
  window.sessionStorage.clear()
  Object.assign(chat, { options: null, status: 'idle', isLoading: false, messages: null })
  nav.search = ''
})

describe('the chat_changed lane', () => {
  it("opens the core's SSE lane, as Studio's chat does, so an awaited reply and a background post merge in", () => {
    render(<PlayChat />)
    expect(useBoardEventStream).toHaveBeenCalledWith(true)
  })
})

describe('first open (an empty chat)', () => {
  it("says 'Hi, I'm Auto.' with Play's example chips", () => {
    render(<PlayChat />)
    expect(screen.getByRole('heading', { name: HELLO_TITLE })).toBeInTheDocument()
    const chips = within(screen.getByRole('list', { name: 'Examples to try' })).getAllByRole('button')
    expect(chips.map((chip) => chip.textContent)).toEqual([...EXAMPLES])
    expect(screen.queryByRole('list', { name: 'Conversation' })).toBeNull()
  })

  it('a chip fills the message box', () => {
    render(<PlayChat />)
    fireEvent.click(screen.getByRole('button', { name: EXAMPLES[1] }))
    expect(box()).toHaveValue(EXAMPLES[1])
  })

  it('plays hello once per browser session', () => {
    window.sessionStorage.removeItem(HELLO_KEY)
    const first = render(<PlayChat />)
    expect(auto()).toHaveAttribute('data-state', 'hello')
    first.unmount()
    render(<PlayChat />)
    expect(auto()).not.toHaveAttribute('data-state', 'hello')
  })

  it('a cold start on this device lands on the newest conversation', async () => {
    setSession({ activeChatId: null, draftOpen: false, openChatIds: [] })
    api.getChatHistory.mockResolvedValue([storedChat('c1', 'Laptop for uni')])
    api.getChat.mockResolvedValue(storedChat('c1', 'Laptop for uni'))
    api.getChatMessages.mockResolvedValue(STORED)
    render(<PlayChat />)
    expect(await screen.findByText('What should I look for in a laptop?')).toBeInTheDocument()
    expect(api.getChatHistory).toHaveBeenCalledWith(1)
    expect(screen.queryByRole('heading', { name: HELLO_TITLE })).toBeNull()
  })
})

describe('a stored conversation', () => {
  it('renders its messages, loaded through getChat and getChatMessages', async () => {
    openStoredChat()
    render(<PlayChat />)
    expect(screen.getByRole('status')).toHaveTextContent('Opening this conversation…')
    const list = await screen.findByRole('list', { name: 'Conversation' })
    expect(within(list).getByText('What should I look for in a laptop?')).toBeInTheDocument()
    expect(within(list).getByRole('article', { name: 'Auto said' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Laptop for uni')
    expect(api.getChat).toHaveBeenCalledWith('c1')
    expect(api.getChatMessages).toHaveBeenCalledWith('c1')
    expect(chat.options?.id).toBe('c1')
    expect(chat.options?.initialMessages).toEqual(STORED)
  })

  it("renders Auto's reply as markdown, with no raw HTML and images as their alt text", async () => {
    openStoredChat([
      userMessage('m1', 'Hi'),
      autoMessage('m2', 'Aim for **16 GB**. <img src="https://evil.example/x.png" onerror="alert(1)"> ![a chart](https://evil.example/c.png)'),
    ])
    const { container } = render(<PlayChat />)
    const reply = await screen.findByRole('article', { name: 'Auto said' })
    expect(within(reply).getByText('16 GB').tagName).toBe('STRONG')
    expect(within(reply).getByText('a chart')).toBeInTheDocument()
    expect(container.querySelector('img')).toBeNull()
  })

  it('a ?chatId= deep link opens that conversation and leaves the URL', async () => {
    setSession({ activeChatId: null, draftOpen: false, openChatIds: [] })
    nav.search = 'chatId=c2'
    api.getChat.mockResolvedValue(storedChat('c2', 'Bake sale'))
    api.getChatMessages.mockResolvedValue([userMessage('m1', 'Captions for the bake sale')])
    render(<PlayChat />)
    expect(await screen.findByText('Captions for the bake sale')).toBeInTheDocument()
    expect(useChatSessionStore.getState().openChat).toHaveBeenCalledWith('c2')
    expect(nav.replace).toHaveBeenCalledWith('/chat')
    expect(api.getChat).toHaveBeenCalledWith('c2')
    expect(api.getChatHistory).not.toHaveBeenCalled()
  })

  it('opens the same deep link again once the URL was cleaned (Recent, New chat, Recent)', async () => {
    api.getChat.mockResolvedValue(storedChat('c2', 'Bake sale'))
    api.getChatMessages.mockResolvedValue([userMessage('m1', 'Captions for the bake sale')])
    nav.search = 'chatId=c2'
    const view = render(<PlayChat />)
    await screen.findByText('Captions for the bake sale')
    nav.search = ''
    view.rerender(<PlayChat />)
    act(() => useChatSessionStore.getState().newDraft())
    expect(await screen.findByRole('heading', { name: HELLO_TITLE })).toBeInTheDocument()
    nav.search = 'chatId=c2'
    view.rerender(<PlayChat />)
    expect(await screen.findByText('Captions for the bake sale')).toBeInTheDocument()
    expect(useChatSessionStore.getState().openChat).toHaveBeenCalledTimes(2)
  })

  it("closes a conversation that can't be loaded, with a toast", async () => {
    openStoredChat()
    api.getChat.mockRejectedValue(new Error('404'))
    const quiet = vi.spyOn(console, 'error').mockImplementation(() => undefined)
    render(<PlayChat />)
    await waitFor(() => expect(useChatSessionStore.getState().closeTab).toHaveBeenCalledWith('c1'))
    expect(toastError).toHaveBeenCalledWith('That conversation is no longer available')
    expect(await screen.findByRole('heading', { name: HELLO_TITLE })).toBeInTheDocument()
    quiet.mockRestore()
  })
})

describe('sending and stopping', () => {
  it('Enter sends through the chat hook; Shift+Enter is a newline', () => {
    render(<PlayChat />)
    fireEvent.change(box(), { target: { value: 'Plan my week' } })
    fireEvent.keyDown(box(), { key: 'Enter', shiftKey: true })
    expect(chat.sendMessage).not.toHaveBeenCalled()
    fireEvent.keyDown(box(), { key: 'Enter' })
    expect(chat.sendMessage).toHaveBeenCalledWith('Plan my week')
    expect(box()).toHaveValue('')
  })

  it('keeps the words when a turn is still ending', () => {
    chat.isLoading = true
    render(<PlayChat />)
    fireEvent.change(box(), { target: { value: 'Again' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send' }))
    expect(chat.sendMessage).not.toHaveBeenCalled()
    expect(box()).toHaveValue('Again')
  })

  it("shows Stop only while streaming, and Stop calls the hook's stop()", async () => {
    openStoredChat()
    const view = render(<PlayChat />)
    await screen.findByRole('list', { name: 'Conversation' })
    expect(screen.queryByRole('button', { name: 'Stop' })).toBeNull()
    chat.status = 'streaming'
    view.rerender(<PlayChat />)
    expect(screen.queryByRole('button', { name: 'Send' })).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Stop' }))
    expect(chat.stop).toHaveBeenCalledTimes(1)
  })

  it("a new chat's id goes to the session store, and the screen keeps it (no reload)", () => {
    render(<PlayChat />)
    act(() => chat.options?.onChatIdUpdate('new-1'))
    expect(useChatSessionStore.getState().chatIdAssigned).toHaveBeenCalledWith('new-1')
    expect(chat.options?.id).toBe('new-1')
    expect(api.getChat).not.toHaveBeenCalled()
    expect(screen.queryByText('Opening this conversation…')).toBeNull()
  })
})

describe("Auto's column", () => {
  it('a status change to streaming puts Auto in thinking', async () => {
    openStoredChat()
    const view = render(<PlayChat />)
    await screen.findByRole('list', { name: 'Conversation' })
    expect(auto()).toHaveAttribute('data-state', 'idle')
    chat.status = 'streaming'
    view.rerender(<PlayChat />)
    expect(auto()).toHaveAttribute('data-state', 'thinking')
  })

  it("leaves thinking when the reply's first words stream in (T-003b), and a tool after them still wins", async () => {
    openStoredChat()
    const view = render(<PlayChat />)
    await screen.findByRole('list', { name: 'Conversation' })
    const asked = userMessage('m3', 'And the battery?')
    chat.status = 'streaming'
    chat.messages = [...STORED, asked, autoMessage('m4', '')]
    view.rerender(<PlayChat />)
    expect(auto()).toHaveAttribute('data-state', 'thinking')
    chat.messages = [...STORED, asked, autoMessage('m4', 'Aim for')]
    view.rerender(<PlayChat />)
    expect(auto()).toHaveAttribute('data-state', 'idle')
    act(() => chat.options?.onData({ type: 'tool-start', data: { toolCallId: 't1', toolName: 'query_database', input: {} } }))
    expect(auto()).toHaveAttribute('data-state', 'working')
  })

  it("follows the chat's tool parts, with the search query in his caption", async () => {
    const running: ToolCall = { toolCallId: 't1', toolName: 'platform_web_search', state: 'running', input: { query: 'budget laptops' } }
    openStoredChat([userMessage('m1', 'Shortlist?'), autoMessage('m2', '', { toolCalls: [running] })])
    chat.status = 'streaming'
    render(<PlayChat />)
    await screen.findByRole('list', { name: 'Conversation' })
    act(() => chat.options?.onData({ type: 'tool-start', data: { toolCallId: 't1', toolName: 'platform_web_search', input: running.input } }))
    expect(auto()).toHaveAttribute('data-state', 'searching')
    expect(auto()).toHaveTextContent('budget laptops')
  })

  it('listens while you type', async () => {
    openStoredChat()
    render(<PlayChat />)
    await screen.findByRole('list', { name: 'Conversation' })
    fireEvent.focus(box())
    expect(auto()).toHaveAttribute('data-state', 'listening')
    fireEvent.blur(box())
    expect(auto()).toHaveAttribute('data-state', 'idle')
  })
})

// T-007b: a Deliverable's "Ask Auto about this" links to /chat?draft=<words> (lib/play/draft.ts).
describe('a ?draft= link (Ask Auto about this)', () => {
  const ASK = 'About my Deliverable "Laptop shortlist": '

  it('opens a new chat with the words waiting in the message box, and leaves the URL', async () => {
    openStoredChat()
    nav.search = `draft=${encodeURIComponent(ASK)}`
    render(<PlayChat />)
    expect(await screen.findByRole('textbox', { name: PLACEHOLDER })).toHaveValue(ASK)
    expect(screen.getByRole('heading', { name: HELLO_TITLE })).toBeInTheDocument()
    expect(useChatSessionStore.getState().newDraft).toHaveBeenCalled()
    expect(nav.replace).toHaveBeenCalledWith('/chat')
    expect(chat.sendMessage).not.toHaveBeenCalled()
    expect(api.getChat).not.toHaveBeenCalled()
    expect(api.getChatHistory).not.toHaveBeenCalled()
  })

  it('puts the words in only once: a later new chat starts empty', async () => {
    nav.search = `draft=${encodeURIComponent(ASK)}`
    const view = render(<PlayChat />)
    expect(await screen.findByRole('textbox', { name: PLACEHOLDER })).toHaveValue(ASK)
    nav.search = ''
    view.rerender(<PlayChat />)
    api.getChat.mockResolvedValue(storedChat('c1', 'Laptop for uni'))
    api.getChatMessages.mockResolvedValue(STORED)
    act(() => useChatSessionStore.getState().openChat('c1'))
    expect(await screen.findByText('What should I look for in a laptop?')).toBeInTheDocument()
    act(() => useChatSessionStore.getState().newDraft())
    expect(await screen.findByRole('heading', { name: HELLO_TITLE })).toBeInTheDocument()
    expect(box()).toHaveValue('')
  })

  it('a ?chatId= link wins over the words, and drops them with itself', async () => {
    setSession({ activeChatId: null, draftOpen: false, openChatIds: [] })
    nav.search = `chatId=c2&draft=${encodeURIComponent(ASK)}`
    api.getChat.mockResolvedValue(storedChat('c2', 'Bake sale'))
    api.getChatMessages.mockResolvedValue([userMessage('m1', 'Captions for the bake sale')])
    render(<PlayChat />)
    expect(await screen.findByText('Captions for the bake sale')).toBeInTheDocument()
    expect(nav.replace).toHaveBeenCalledWith('/chat')
    expect(box()).toHaveValue('')
  })
})

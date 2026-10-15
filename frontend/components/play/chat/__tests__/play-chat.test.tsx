/**
 * PRD-260 §4, D5, D6 — Play's chat on the core's data. The chat hook
 * (useChatWithReceipts) and the session store are mocked; the conversation
 * loading, the session hydration (useChatSessionHydration, over a stand-in
 * identity and workspace), Play's asleep gate, Auto's reducer (useAutoMascot)
 * and the markdown renderer are real. Auto himself is a stand-in that shows his
 * state and caption (his clips have their own tests). Roles and text only (CSS
 * modules are proxies).
 */
import React from 'react'
import { act, fireEvent, render as renderUi, screen, waitFor, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
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
  /** How many chat screens have mounted the hook (a new key mounts it again). */
  mounts: 0,
}))
const api = vi.hoisted(() => ({ getChat: vi.fn(), getChatMessages: vi.fn(), getChatHistory: vi.fn() }))
const toastError = vi.hoisted(() => vi.fn())
const ids = vi.hoisted(() => ({ user: 'u1' }))
/** GET /health as Play's frame and the chat read it (useBackendHealth); tests switch it to put Automatos to sleep. */
const health = vi.hoisted(() => ({ connection: 'connected' as 'checking' | 'connected' | 'down', check: vi.fn() }))
/** The workspace provider's value (useWorkspaceOptional); null is no provider. */
const scope = vi.hoisted(() => ({ workspace: null as null | { workspace: { id: string } | null; isLoading: boolean } }))

vi.mock('next/navigation', () => ({
  usePathname: () => '/chat',
  useSearchParams: () => new URLSearchParams(nav.search),
  useRouter: () => nav,
}))
vi.mock('sonner', () => ({ toast: { error: toastError } }))
vi.mock('@/lib/chat/api', () => api)
vi.mock('@/lib/page-context', () => ({ usePageContext: () => ({ page: 'chat', route: '/chat' }) }))
// The session hydration is real (P260-FX-07): the identity is a stand-in, the local edition's shapes (lib/auth-hooks.ts).
vi.mock('@/lib/auth-hooks', () => ({
  useAuth: () => ({ isLoaded: true, isSignedIn: true, getToken: async () => null }),
  useUser: () => ({ isLoaded: true, isSignedIn: true, user: { id: ids.user } }),
  useSession: () => ({ isLoaded: true, isSignedIn: true, session: null }),
  useOrganization: () => ({ isLoaded: true, organization: null, membership: null }),
  useClerk: () => ({ signOut: async () => undefined }),
}))
vi.mock('@/hooks/use-backend-health', () => ({
  useBackendHealth: () => ({ connection: health.connection, check: health.check }),
}))
// PlayChat holds the chat_changed SSE lane open (P260-FX-02); no stream here. The QueryClient is real (render, below).
vi.mock('@/hooks/use-board-event-stream', () => ({ useBoardEventStream: vi.fn() }))
// Play's frame as MainLayout renders it under Play (play-frame.tsx): the real asleep gate around the page, the rest left out.
vi.mock('@/components/layout/main-layout', async () => {
  const { AsleepGate } = await import('@/components/play/edges/asleep-gate')
  return {
    MainLayout: ({ children }: { children?: React.ReactNode }) => (
      <div data-testid="frame">
        <AsleepGate>{children}</AsleepGate>
      </div>
    ),
  }
})
vi.mock('@/components/local/first-run-nudge', () => ({ FirstRunNudge: () => null }))
// T-005b: the chat shows the core's trial banner off the workspace snapshot; no snapshot here unless a test sets one.
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: null }),
  useWorkspaceOptional: () => scope.workspace,
}))
vi.mock('@/components/mascot/auto-mascot', () => ({
  AutoMascot: ({ state, caption, detail }: { state: string; caption?: string; detail?: string }) => (
    <div data-testid="auto" data-state={state}>
      {caption} {detail}
    </div>
  ),
}))
vi.mock('@/lib/chat/use-chat-with-receipts', async () => {
  const { useState } = await import('react')
  return {
    useChatWithReceipts: (options: Record<string, any>) => {
      // Counted once per chat screen: a new key mounts the hook again.
      useState(() => (chat.mounts += 1))
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
  }
})
vi.mock('@/stores/chat-session-store', async () => {
  const { create } = await import('zustand')
  const { chatSessionKey } = await import('@/lib/chat/chat-session')
  type Session = { activeChatId: string | null; draftOpen: boolean; openChatIds: string[] }
  const open = (session: Session, id: string): Session => ({
    ...session,
    activeChatId: id,
    draftOpen: false,
    openChatIds: session.openChatIds.includes(id) ? session.openChatIds : [...session.openChatIds, id],
  })
  const store = create<any>((set, get) => ({
    key: chatSessionKey(null, ids.user),
    hydrated: true,
    session: { activeChatId: null, draftOpen: true, openChatIds: [] },
    // As the real store: a key already hydrated is kept as it is; a new one is bound (the browser copy is the stand-in's session).
    hydrate: vi.fn(async (key: string) => {
      if (get().key !== key || !get().hydrated) set({ key, hydrated: true })
    }),
    reload: vi.fn(),
    touchChat: vi.fn(),
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

import { OPENING, PlayChat } from '@/components/play/chat/play-chat'
import { EXAMPLES, HELLO_TITLE } from '@/components/play/chat/first-open'
import { PLACEHOLDER } from '@/components/play/chat/message-box'
import { ASLEEP_TITLE } from '@/components/play/edges/asleep-screen'
import { useBoardEventStream } from '@/hooks/use-board-event-stream'
import { CONVERSATION_LIMIT, useConversationList } from '@/hooks/use-conversation-list'
import { useStartChat } from '@/hooks/use-start-chat'
import { chatSessionKey, EMPTY_CHAT_SESSION } from '@/lib/chat/chat-session'
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

/** Play's New chat (the sidebar's, through useStartChat), beside the chat. */
function NewChatButton() {
  const { startChat, ready } = useStartChat()
  return (
    <button type="button" onClick={startChat} disabled={!ready}>
      New chat
    </button>
  )
}

const auto = () => screen.getByTestId('auto')
const box = () => screen.getByRole('textbox', { name: PLACEHOLDER })

/** The root providers' fresh window (components/providers.tsx): within it, nothing is asked again on its own. */
const ROOT_STALE_MS = 60_000

/**
 * Under a real QueryClient with the root's staleTime, as components/providers.tsx
 * gives it: the chat asks the history again when a turn ends (P260-FX-10). A
 * rerender keeps the same client.
 */
function render(ui: React.ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { staleTime: ROOT_STALE_MS, retry: false } } })
  const Root = ({ children }: { children: React.ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>
  return renderUi(ui, { wrapper: Root })
}

/** Play's Conversations list beside the chat (as Recent sits beside it in the sidebar): the history's titles, in order. */
function HistoryTitles() {
  const list = useConversationList()
  return <p data-testid="history">{list.status === 'loaded' ? list.chats.map((c) => c.title).join(', ') : list.status}</p>
}

beforeEach(() => {
  window.sessionStorage.setItem(HELLO_KEY, '1') // hello already played this session
  useChatSessionStore.setState({ key: chatSessionKey(null, ids.user), hydrated: true })
  setSession({ activeChatId: null, draftOpen: true, openChatIds: [] })
  Object.values(api).forEach((fn) => fn.mockReset())
  api.getChatHistory.mockResolvedValue([])
})

afterEach(() => {
  vi.clearAllMocks()
  window.sessionStorage.clear()
  Object.assign(chat, { options: null, status: 'idle', isLoading: false, messages: null, mounts: 0 })
  nav.search = ''
  health.connection = 'connected'
  scope.workspace = null
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
    // The asleep gate keeps its own live region in the page from the start (P260-FX-22), so ask for this line by its text.
    expect(screen.getByText(OPENING)).toHaveAttribute('role', 'status')
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
    // The same chat screen all along: its stream carries on.
    expect(chat.mounts).toBe(1)
  })
})

// P260-FX-03: the id a turn names is adopted once, for the new chat on screen; New chat always shows a fresh one.
describe('New chat and reopening a conversation', () => {
  const WEEK = [userMessage('m1', 'Plan my week, please'), autoMessage('m2', 'Here is your week.')]

  it('after New chat, the conversation the new chat became opens from the server, not as an empty first open (M03)', async () => {
    render(<PlayChat />)
    act(() => chat.options?.onChatIdUpdate('new-1'))
    act(() => useChatSessionStore.getState().newDraft())
    expect(await screen.findByRole('heading', { name: HELLO_TITLE })).toBeInTheDocument()
    api.getChat.mockResolvedValue(storedChat('new-1', 'Weekly plan'))
    api.getChatMessages.mockResolvedValue(WEEK)
    act(() => useChatSessionStore.getState().openChat('new-1'))
    const list = await screen.findByRole('list', { name: 'Conversation' })
    expect(within(list).getByText('Plan my week, please')).toBeInTheDocument()
    expect(api.getChat).toHaveBeenCalledWith('new-1')
    expect(screen.queryByRole('heading', { name: HELLO_TITLE })).toBeNull()
  })

  it("reopening a stored conversation after New chat loads it again, though a later turn named it (M03)", async () => {
    openStoredChat()
    render(<PlayChat />)
    await screen.findByRole('list', { name: 'Conversation' })
    act(() => chat.options?.onChatIdUpdate('c1'))
    act(() => useChatSessionStore.getState().newDraft())
    expect(await screen.findByRole('heading', { name: HELLO_TITLE })).toBeInTheDocument()
    act(() => useChatSessionStore.getState().openChat('c1'))
    const list = await screen.findByRole('list', { name: 'Conversation' })
    expect(within(list).getByText('What should I look for in a laptop?')).toBeInTheDocument()
    expect(api.getChat).toHaveBeenCalledTimes(2)
  })

  it('New chat over a new chat whose first send failed shows a fresh one (M12)', () => {
    const view = render(<><PlayChat /><NewChatButton /></>)
    chat.status = 'error'
    chat.messages = [userMessage('m1', 'Plan my week')]
    view.rerender(<><PlayChat /><NewChatButton /></>)
    expect(screen.queryByRole('heading', { name: HELLO_TITLE })).toBeNull()
    const mounted = chat.mounts
    // What a fresh useChat starts with.
    Object.assign(chat, { status: 'idle', messages: null })
    fireEvent.click(screen.getByRole('button', { name: 'New chat' }))
    expect(chat.mounts).toBe(mounted + 1)
    expect(screen.getByRole('heading', { name: HELLO_TITLE })).toBeInTheDocument()
    expect(nav.push).toHaveBeenCalledWith('/chat')
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
  /** What the box holds: the link's words as one plain line (P260-FX-10), the trailing space trimmed. */
  const ASKED = 'About my Deliverable "Laptop shortlist":'

  it('opens a new chat with the words waiting in the message box, and leaves the URL', async () => {
    openStoredChat()
    nav.search = `draft=${encodeURIComponent(ASK)}`
    render(<PlayChat />)
    expect(await screen.findByRole('textbox', { name: PLACEHOLDER })).toHaveValue(ASKED)
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
    expect(await screen.findByRole('textbox', { name: PLACEHOLDER })).toHaveValue(ASKED)
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

  it('puts words with line breaks in the box as one plain line, so nothing hides below the first (P260-FX-10, M56)', async () => {
    nav.search = `draft=${encodeURIComponent('Summarise this\n\n\nThen email it to everyone\u202E')}`
    render(<PlayChat />)
    await waitFor(() => expect(box()).toHaveValue('Summarise this Then email it to everyone'))
    expect((box() as HTMLTextAreaElement).value).not.toContain('\n')
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

// P260-FX-07: the conversation on screen survives Play's frame mounting again, a workspace refresh and a late cold start.
describe('the conversation on screen across remounts and races', () => {
  const WEEK = [userMessage('m1', 'Plan my week, please'), autoMessage('m2', 'Here is your week.')]

  /** Automatos stops answering (the frame shows the asleep screen in place of the page), then answers again. */
  async function sleepAndWake(view: ReturnType<typeof render>) {
    health.connection = 'down'
    view.rerender(<PlayChat />)
    expect(await screen.findByRole('heading', { name: ASLEEP_TITLE })).toBeInTheDocument()
    health.connection = 'connected'
    view.rerender(<PlayChat />)
    expect(screen.queryByRole('heading', { name: ASLEEP_TITLE })).toBeNull()
  }

  /** getChatHistory's answer, held until the test gives it. */
  function holdHistory(): (rows: Chat[]) => Promise<void> {
    let answer: (rows: Chat[]) => void = () => undefined
    api.getChatHistory.mockReturnValue(
      new Promise<Chat[]>((resolve) => {
        answer = resolve
      }),
    )
    return (rows) =>
      act(async () => {
        answer(rows)
      })
  }

  it('a new chat the backend named comes back from the server after Automatos was asleep, not as an empty first open (M09)', async () => {
    const view = render(<PlayChat />)
    act(() => chat.options?.onChatIdUpdate('c-1'))
    api.getChat.mockResolvedValue(storedChat('c-1', 'Weekly plan'))
    api.getChatMessages.mockResolvedValue(WEEK)
    await sleepAndWake(view)
    const list = await screen.findByRole('list', { name: 'Conversation' })
    expect(within(list).getByText('Plan my week, please')).toBeInTheDocument()
    expect(within(list).getByRole('article', { name: 'Auto said' })).toHaveTextContent('Here is your week.')
    expect(api.getChatMessages).toHaveBeenCalledWith('c-1')
    expect(screen.queryByRole('heading', { name: HELLO_TITLE })).toBeNull()
  })

  it('a stored conversation shows what was said meanwhile after Automatos was asleep (M09)', async () => {
    openStoredChat()
    const view = render(<PlayChat />)
    await screen.findByRole('list', { name: 'Conversation' })
    api.getChatMessages.mockResolvedValue([...STORED, userMessage('m3', 'And the battery?'), autoMessage('m4', 'Ten hours at least.')])
    await sleepAndWake(view)
    expect(await screen.findByText('And the battery?')).toBeInTheDocument()
    expect(api.getChatMessages).toHaveBeenCalledTimes(2)
  })

  it('a refresh of the same workspace (a key saved from the trial banner) keeps the chat on screen: never Opening, one chat screen (M09)', () => {
    const workspace = { id: 'ws-1' }
    scope.workspace = { workspace, isLoading: false }
    useChatSessionStore.setState({ key: chatSessionKey(workspace.id, ids.user) })
    const view = render(<PlayChat />)
    act(() => chat.options?.onChatIdUpdate('c-1'))
    expect(screen.queryByText(OPENING)).toBeNull()
    scope.workspace = { workspace, isLoading: true }
    view.rerender(<PlayChat />)
    expect(screen.queryByText(OPENING)).toBeNull()
    scope.workspace = { workspace, isLoading: false }
    view.rerender(<PlayChat />)
    expect(screen.queryByText(OPENING)).toBeNull()
    expect(chat.mounts).toBe(1)
    expect(chat.options?.id).toBe('c-1')
    expect(api.getChat).not.toHaveBeenCalled()
  })

  it('a Recent link opens the next conversation without mounting the previous one again on the way', async () => {
    openStoredChat()
    const view = render(<PlayChat />)
    await screen.findByRole('list', { name: 'Conversation' })
    api.getChat.mockResolvedValue(storedChat('c2', 'Bake sale'))
    api.getChatMessages.mockResolvedValue([userMessage('m5', 'Captions for the bake sale')])
    nav.search = 'chatId=c2'
    view.rerender(<PlayChat />)
    expect(await screen.findByText('Captions for the bake sale')).toBeInTheDocument()
    // One chat screen per conversation: c1's, then c2's.
    expect(chat.mounts).toBe(2)
  })

  it('New chat pressed before the cold start reads the newest conversation stays a new chat (M10)', async () => {
    setSession({ activeChatId: null, draftOpen: false, openChatIds: [] })
    const answer = holdHistory()
    render(<PlayChat />)
    expect(screen.getByText(OPENING)).toBeInTheDocument()
    act(() => useChatSessionStore.getState().newDraft())
    expect(await screen.findByRole('heading', { name: HELLO_TITLE })).toBeInTheDocument()
    await answer([storedChat('c1', 'Laptop for uni')])
    expect(useChatSessionStore.getState().openChat).not.toHaveBeenCalled()
    expect(useChatSessionStore.getState().session.activeChatId).toBeNull()
    expect(screen.getByRole('heading', { name: HELLO_TITLE })).toBeInTheDocument()
    expect(api.getChat).not.toHaveBeenCalled()
  })

  it('a conversation opened before the cold start answers stays on screen (M10)', async () => {
    setSession({ activeChatId: null, draftOpen: false, openChatIds: [] })
    const answer = holdHistory()
    api.getChat.mockResolvedValue(storedChat('c2', 'Bake sale'))
    api.getChatMessages.mockResolvedValue([userMessage('m1', 'Captions for the bake sale')])
    render(<PlayChat />)
    act(() => useChatSessionStore.getState().openChat('c2'))
    expect(await screen.findByText('Captions for the bake sale')).toBeInTheDocument()
    await answer([storedChat('c1', 'Laptop for uni')])
    expect(useChatSessionStore.getState().session.activeChatId).toBe('c2')
    expect(api.getChat).not.toHaveBeenCalledWith('c1')
    expect(screen.getByText('Captions for the bake sale')).toBeInTheDocument()
  })

  it('a blank ?chatId= is no link: the first open shows, never Opening, and it leaves the URL (M11)', async () => {
    nav.search = 'chatId='
    render(<PlayChat />)
    expect(await screen.findByRole('heading', { name: HELLO_TITLE })).toBeInTheDocument()
    expect(screen.queryByText(OPENING)).toBeNull()
    expect(nav.replace).toHaveBeenCalledWith('/chat')
    expect(useChatSessionStore.getState().openChat).not.toHaveBeenCalled()
  })

  it('a blank ?chatId= on a cold start lands on the newest conversation (M11)', async () => {
    setSession({ activeChatId: null, draftOpen: false, openChatIds: [] })
    nav.search = 'chatId=%20'
    api.getChatHistory.mockResolvedValue([storedChat('c1', 'Laptop for uni')])
    api.getChat.mockResolvedValue(storedChat('c1', 'Laptop for uni'))
    api.getChatMessages.mockResolvedValue(STORED)
    render(<PlayChat />)
    expect(await screen.findByText('What should I look for in a laptop?')).toBeInTheDocument()
    expect(api.getChatHistory).toHaveBeenCalledWith(1)
    expect(screen.queryByText(OPENING)).toBeNull()
  })
})

// P260-FX-10 (M44): the root QueryClient keeps a list fresh for a minute; a turn's end asks the history again.
describe('the history after a turn', () => {
  const NEWER = [storedChat('c2', 'Bake sale'), storedChat('c1', 'Laptop for uni')]
  const page = () => (
    <>
      <PlayChat />
      <HistoryTitles />
    </>
  )

  /** A stored conversation on screen, its history listed beside it. */
  async function renderWithHistory() {
    openStoredChat()
    api.getChatHistory.mockResolvedValue([storedChat('c1', 'Laptop for uni')])
    const view = render(page())
    await screen.findByRole('list', { name: 'Conversation' })
    await waitFor(() => expect(screen.getByTestId('history')).toHaveTextContent('Laptop for uni'))
    return view
  }

  it('asks the history again when the reply ends, so Conversations and Recent show the turn', async () => {
    const view = await renderWithHistory()
    api.getChatHistory.mockResolvedValue(NEWER)
    chat.status = 'streaming'
    view.rerender(page())
    expect(api.getChatHistory).toHaveBeenCalledTimes(1)
    chat.status = 'idle'
    view.rerender(page())
    await waitFor(() => expect(screen.getByTestId('history')).toHaveTextContent('Bake sale, Laptop for uni'))
    expect(api.getChatHistory).toHaveBeenCalledTimes(2)
    expect(api.getChatHistory).toHaveBeenLastCalledWith(CONVERSATION_LIMIT)
  })

  it('a turn that fails asks again too; a render with the status unchanged does not', async () => {
    const view = await renderWithHistory()
    view.rerender(page())
    expect(api.getChatHistory).toHaveBeenCalledTimes(1)
    chat.status = 'streaming'
    view.rerender(page())
    chat.status = 'error'
    view.rerender(page())
    await waitFor(() => expect(api.getChatHistory).toHaveBeenCalledTimes(2))
  })
})

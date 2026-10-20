/**
 * PRD-261 A-19e1 — the SDK's chat (lib/theme-sdk/chat.ts) on the hello theme's chat page
 * (support/hello-chat-page.tsx, the page of e2e/browser/fixtures/hello-a11b/entry.js) over the REAL chat
 * controller, conversation loading (lib/chat-controller/conversation.ts), session store and New chat. The
 * stream hook (useChatWithReceipts) is a stand-in, as the PRD-260 chat tests stood it in
 * (lib/chat-controller/__tests__/controller-cases.test.ts runs the real one); so are the chat API, the
 * identity, the router and the toasts. The page never sends.
 *
 * It re-homes the core assertions of components/play/chat/__tests__/play-chat.test.tsx, which leaves core
 * with builtin Play (W-A8; the kit keeps its Play-rendering assertions, B-05c): a cold start, a stored
 * conversation, a deep link, one that can't be loaded, a new chat the backend names, New chat and reopening
 * (P260-FX-03, M03), New chat's fresh draft (M12), the page mounting again and the races of P260-FX-07
 * (M09-M11), and a theme's draft link (Play's `?draft=`). The history asked again after a turn is
 * controller-cases.test.ts; the host's cards and what a stored reply carries are
 * chat-controller-cards-cases.test.tsx.
 */
import { act, fireEvent, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { Chat } from '@/types'

const nav = vi.hoisted(() => ({ search: '', replace: vi.fn(), push: vi.fn() }))
const chat = vi.hoisted(() => ({
  options: null as null | Record<string, any>,
  status: 'idle' as 'idle' | 'streaming' | 'error',
  messages: null as null | unknown[],
  /** How many chat screens mounted the stream (a new key mounts it again). */
  mounts: 0,
}))
const api = vi.hoisted(() => ({ getChat: vi.fn(), getChatMessages: vi.fn(), getChatHistory: vi.fn(), putChatSession: vi.fn() }))
const toastError = vi.hoisted(() => vi.fn())

vi.mock('next/navigation', () => ({
  usePathname: () => '/chat',
  useSearchParams: () => new URLSearchParams(nav.search),
  useRouter: () => nav,
}))
vi.mock('sonner', () => ({ toast: { error: toastError, success: vi.fn(), info: vi.fn() } }))
vi.mock('@/lib/chat/api', () => api)
vi.mock('@/lib/page-context', () => ({ usePageContext: () => ({ page: 'chat', route: '/chat' }) }))
// The session hydration is real: the identity is a stand-in, the local edition's shapes (lib/auth-hooks.ts).
vi.mock('@/lib/auth-hooks', () => ({
  useAuth: () => ({ isLoaded: true, isSignedIn: true, getToken: async () => null }),
  useUser: () => ({ isLoaded: true, isSignedIn: true, user: { id: 'u1' } }),
  useSession: () => ({ isLoaded: true, isSignedIn: true, session: null }),
  useOrganization: () => ({ isLoaded: true, organization: null, membership: null }),
  useClerk: () => ({ signOut: async () => undefined }),
}))
vi.mock('@/components/workspace-provider', () => ({ useWorkspace: () => ({ workspace: null }), useWorkspaceOptional: () => null }))
vi.mock('@/lib/chat/use-chat-with-receipts', async () => {
  const { useState } = await import('react')
  return {
    useChatWithReceipts: (options: Record<string, any>) => {
      useState(() => (chat.mounts += 1))
      chat.options = options
      const { status } = chat
      return {
        messages: chat.messages ?? options.initialMessages,
        setMessages: vi.fn(),
        sendMessage: vi.fn(),
        reload: vi.fn(),
        stop: vi.fn(),
        status,
        isLoading: status === 'streaming',
        awaitingReply: false,
        usage: undefined,
        errorCode: null,
      }
    },
  }
})

import { CONVERSATION_GONE } from '@/lib/chat-controller/view-model'
import { CHAT_HREF } from '@/lib/ticket-links'
import { useChatSessionStore } from '@/stores/chat-session-store'

import {
  DRAFT_LINK,
  forgetPage,
  HelloChatPage,
  message,
  NEW_CHAT,
  NEW_CHAT_LABEL,
  NewChatButton,
  NOTHING_OPEN,
  on,
  OPENING,
  renderPage,
  seen,
  serve,
  sessionNow,
  setSession,
  shownChat,
  storedChat,
} from './support/hello-chat-page'

const LAPTOP = 'What should I look for in a laptop?'
const STORED = [message('m1', 'user', LAPTOP), message('m2', 'assistant', 'Aim for 16 GB of memory.')]
const BAKE = 'Captions for the bake sale'
const DRAFT_WORDS = 'About my Deliverable'

const store = () => useChatSessionStore.getState()
/** The backend names the new chat on screen, as useChat's onChatIdUpdate hands it on. */
const nameNewChat = (chatId: string) => act(() => chat.options?.onChatIdUpdate(chatId))

/** The cold start's getChatHistory, held until the test answers it with these rows. */
function holdHistory(): (rows: Chat[]) => Promise<void> {
  let answer: (rows: Chat[]) => void = () => undefined
  api.getChatHistory.mockReturnValue(new Promise<Chat[]>((resolve) => { answer = resolve }))
  return (rows) => act(async () => answer(rows))
}

beforeEach(() => {
  Object.values(api).forEach((fn) => fn.mockReset())
  api.getChatHistory.mockResolvedValue([])
  setSession(NEW_CHAT)
})

afterEach(() => {
  vi.clearAllMocks()
  window.localStorage.clear()
  Object.assign(chat, { options: null, status: 'idle', messages: null, mounts: 0 })
  forgetPage()
  nav.search = ''
})

describe('the conversation on screen', () => {
  it('a cold start on this device lands on the newest conversation', async () => {
    setSession(NOTHING_OPEN)
    api.getChatHistory.mockResolvedValue([storedChat('c1', 'Laptop for uni')])
    serve('c1', 'Laptop for uni', STORED)
    renderPage()
    expect(screen.getByText(OPENING)).toBeInTheDocument()
    expect(await screen.findByText(LAPTOP)).toBeInTheDocument()
    expect(api.getChatHistory).toHaveBeenCalledWith(1)
    expect((await shownChat('c1')).title).toBe('Laptop for uni')
  })

  it('a stored conversation is opening while it loads, then shows what getChat and getChatMessages gave, on a stream opened on it', async () => {
    setSession(on('c1'))
    serve('c1', 'Laptop for uni', STORED)
    renderPage()
    expect(screen.getByText(OPENING)).toBeInTheDocument()
    expect(await screen.findByRole('heading', { level: 1 })).toHaveTextContent('Laptop for uni')
    expect(screen.getByText(LAPTOP)).toBeInTheDocument()
    expect(api.getChat).toHaveBeenCalledWith('c1')
    expect(api.getChatMessages).toHaveBeenCalledWith('c1')
    expect(chat.options).toMatchObject({ id: 'c1', initialMessages: STORED })
    expect(api.getChatHistory).not.toHaveBeenCalled()
  })

  it('a ?chatId= deep link opens that conversation and leaves the URL; the same link opens it again once cleaned', async () => {
    setSession(NOTHING_OPEN)
    nav.search = 'chatId=c2'
    serve('c2', 'Bake sale', [message('m1', 'user', BAKE)])
    const view = renderPage()
    expect(await screen.findByText(BAKE)).toBeInTheDocument()
    expect(nav.replace).toHaveBeenCalledWith(CHAT_HREF)
    expect(api.getChatHistory).not.toHaveBeenCalled()
    nav.search = ''
    view.rerender(<HelloChatPage />)
    act(() => store().newDraft())
    await shownChat(null)
    nav.search = 'chatId=c2'
    view.rerender(<HelloChatPage />)
    await shownChat('c2')
    expect(api.getChat).toHaveBeenCalledTimes(2)
  })

  it("closes a conversation that can't be loaded, with a toast, and shows a new chat", async () => {
    setSession(on('c1'))
    api.getChat.mockRejectedValue(new Error('404'))
    api.getChatMessages.mockResolvedValue(STORED)
    const quiet = vi.spyOn(console, 'error').mockImplementation(() => undefined)
    renderPage()
    await waitFor(() => expect(toastError).toHaveBeenCalledWith(CONVERSATION_GONE))
    await shownChat(null)
    expect(sessionNow().openChatIds).toEqual([])
    quiet.mockRestore()
  })

  it('a new chat the backend names stays on screen: the same chat screen, its id in the session, nothing loaded', async () => {
    renderPage()
    const draft = await shownChat(null)
    nameNewChat('new-1')
    expect(sessionNow().activeChatId).toBe('new-1')
    const named = await shownChat('new-1')
    expect(named.conversation.key).toBe(draft.conversation.key)
    expect(chat.options?.id).toBe('new-1')
    expect(chat.mounts).toBe(1)
    expect(api.getChat).not.toHaveBeenCalled()
    expect(screen.queryByText(OPENING)).toBeNull()
  })
})

// P260-FX-03: the id a turn names is adopted once, for the new chat on screen; New chat always shows a fresh one.
describe('New chat and opening a conversation again', () => {
  it('after New chat, the conversation the new chat became opens from the server, not empty (M03)', async () => {
    renderPage()
    await shownChat(null)
    nameNewChat('new-1')
    act(() => store().newDraft())
    await shownChat(null)
    serve('new-1', 'Weekly plan', [message('m1', 'user', 'Plan my week, please')])
    act(() => store().openChat('new-1'))
    expect(await screen.findByText('Plan my week, please')).toBeInTheDocument()
    expect(api.getChat).toHaveBeenCalledWith('new-1')
  })

  it('a stored conversation opened again after New chat loads again, though a later turn named it (M03)', async () => {
    setSession(on('c1'))
    serve('c1', 'Laptop for uni', STORED)
    renderPage()
    await shownChat('c1')
    nameNewChat('c1')
    act(() => store().newDraft())
    await shownChat(null)
    act(() => store().openChat('c1'))
    expect(await screen.findByText(LAPTOP)).toBeInTheDocument()
    expect(api.getChat).toHaveBeenCalledTimes(2)
  })

  it("New chat's fresh draft: pressed over a new chat whose first send failed, it shows a fresh one (M12)", async () => {
    const page = () => (
      <>
        <HelloChatPage />
        <NewChatButton />
      </>
    )
    const view = renderPage(page())
    const first = await shownChat(null)
    Object.assign(chat, { status: 'error', messages: [message('m1', 'user', 'Plan my week')] })
    view.rerender(page())
    const mounted = chat.mounts
    // What a fresh useChat starts with.
    Object.assign(chat, { status: 'idle', messages: null })
    fireEvent.click(screen.getByRole('button', { name: NEW_CHAT_LABEL }))
    await waitFor(() => expect(seen.chat?.conversation.key).not.toBe(first.conversation.key))
    expect(chat.mounts).toBe(mounted + 1)
    expect(seen.chat?.messages).toEqual([])
    expect(nav.push).toHaveBeenCalledWith(CHAT_HREF)
  })
})

// P260-FX-07: the conversation on screen survives the page mounting again, a Recent link and a late cold start.
describe('the conversation on screen across mounts and races', () => {
  it('the page mounting again loads its conversation afresh: a new chat the backend named, and what was said meanwhile (M09)', async () => {
    const first = renderPage()
    await shownChat(null)
    nameNewChat('c-1')
    first.unmount()
    serve('c-1', 'Weekly plan', [message('m1', 'user', 'Plan my week, please'), message('m2', 'assistant', 'Here is your week.')])
    const second = renderPage()
    expect(await screen.findByText('Here is your week.')).toBeInTheDocument()
    second.unmount()
    api.getChatMessages.mockResolvedValue([message('m3', 'user', 'And the battery?')])
    renderPage()
    expect(await screen.findByText('And the battery?')).toBeInTheDocument()
    expect(api.getChatMessages).toHaveBeenCalledTimes(2)
    expect(api.getChatMessages).toHaveBeenCalledWith('c-1')
  })

  it('a Recent link opens the next conversation without mounting the previous one again on the way', async () => {
    setSession(on('c1'))
    serve('c1', 'Laptop for uni', STORED)
    const view = renderPage()
    await screen.findByText(LAPTOP)
    serve('c2', 'Bake sale', [message('m5', 'user', BAKE)])
    nav.search = 'chatId=c2'
    view.rerender(<HelloChatPage />)
    expect(await screen.findByText(BAKE)).toBeInTheDocument()
    expect(chat.mounts).toBe(2)
  })

  it('New chat pressed before the cold start reads the newest conversation stays a new chat (M10)', async () => {
    setSession(NOTHING_OPEN)
    const answer = holdHistory()
    renderPage()
    expect(screen.getByText(OPENING)).toBeInTheDocument()
    act(() => store().newDraft())
    const draft = await shownChat(null)
    await answer([storedChat('c1', 'Laptop for uni')])
    expect(sessionNow().activeChatId).toBeNull()
    expect(api.getChat).not.toHaveBeenCalled()
    expect(seen.chat?.conversation.chatId).toBeNull()
    expect(seen.chat?.conversation.key).toBe(draft.conversation.key)
    expect(screen.queryByText(OPENING)).toBeNull()
  })

  it("a cold start's late answer replaces neither a new chat nor a conversation opened meanwhile (M10)", async () => {
    setSession(NOTHING_OPEN)
    const answer = holdHistory()
    serve('c2', 'Bake sale', [message('m1', 'user', BAKE)])
    renderPage()
    expect(screen.getByText(OPENING)).toBeInTheDocument()
    act(() => store().newDraft())
    await shownChat(null)
    act(() => store().openChat('c2'))
    await shownChat('c2')
    await answer([storedChat('c1', 'Laptop for uni')])
    expect(sessionNow().activeChatId).toBe('c2')
    expect(api.getChat).not.toHaveBeenCalledWith('c1')
    expect(screen.getByText(BAKE)).toBeInTheDocument()
  })

  it('a blank ?chatId= is no link: it leaves the URL and a new chat shows, or on a cold start the newest conversation (M11)', async () => {
    nav.search = 'chatId='
    const first = renderPage()
    await shownChat(null)
    expect(nav.replace).toHaveBeenCalledWith(CHAT_HREF)
    expect(screen.queryByText(OPENING)).toBeNull()
    first.unmount()
    setSession(NOTHING_OPEN)
    nav.search = 'chatId=%20'
    api.getChatHistory.mockResolvedValue([storedChat('c1', 'Laptop for uni')])
    serve('c1', 'Laptop for uni', STORED)
    renderPage()
    await shownChat('c1')
    expect(api.getChatHistory).toHaveBeenCalledWith(1)
  })
})

describe("a theme's draft link (Play's ?draft=, a Deliverable's Ask Auto about this)", () => {
  it('opens a new chat with its words waiting, once, and leaves the URL; nothing is loaded or sent', async () => {
    setSession(on('c1'))
    nav.search = `draft=${encodeURIComponent(DRAFT_WORDS)}`
    const view = renderPage(<HelloChatPage draftLink={DRAFT_LINK} />)
    await shownChat(null)
    expect(seen.onScreen?.draft?.text).toBe(DRAFT_WORDS)
    expect(sessionNow().activeChatId).toBeNull()
    expect(nav.replace).toHaveBeenCalledWith(CHAT_HREF)
    expect(api.getChat).not.toHaveBeenCalled()
    expect(api.getChatHistory).not.toHaveBeenCalled()
    // The theme put the words in its box: a later new chat starts empty.
    act(() => seen.onScreen?.takeDraft())
    nav.search = ''
    view.rerender(<HelloChatPage draftLink={DRAFT_LINK} />)
    serve('c1', 'Laptop for uni', STORED)
    act(() => store().openChat('c1'))
    await shownChat('c1')
    act(() => store().newDraft())
    await shownChat(null)
    expect(seen.onScreen?.draft).toBeNull()
  })

  it('a ?chatId= link wins over the words, and drops them with itself', async () => {
    setSession(NOTHING_OPEN)
    nav.search = `chatId=c2&draft=${encodeURIComponent(DRAFT_WORDS)}`
    serve('c2', 'Bake sale', [message('m1', 'user', BAKE)])
    renderPage(<HelloChatPage draftLink={DRAFT_LINK} />)
    await shownChat('c2')
    expect(await screen.findByText(BAKE)).toBeInTheDocument()
    expect(nav.replace).toHaveBeenCalledWith(CHAT_HREF)
    expect(seen.onScreen?.draft).toBeNull()
  })
})

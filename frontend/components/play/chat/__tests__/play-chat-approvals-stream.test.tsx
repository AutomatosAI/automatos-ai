/**
 * PRD-260 T-005c — approvals in Play's chat through the core's REAL chat hook:
 * useChatWithReceipts and useChat (lib/chat/hooks.ts) read a stubbed /api/chat
 * stream, so the step from useChat's `d:` tool-data line to the card on screen
 * is exercised, not mocked. And a card outlives its reply: useChat drops the
 * reply on Stop (its catch removes the placeholder on an AbortError) and Try
 * again (reload) slices it off, yet the grant still waits on the server, so the
 * card stays on screen under "Approvals still open" until it is answered or
 * closed, and Auto stops listening then. Settling a card mid-turn returns Auto
 * to the tool still running. And the page holds the core's SSE lane open
 * (useBoardEventStream, real, under a QueryClient as the root providers give
 * it): a reply the server finishes detached (the page opened mid-turn) merges
 * in when the lane says chat_changed. Only the network (fetch), the identity,
 * the approval hooks (as the core's widget tests mock them) and Auto's stand-in
 * are mocked. Roles and text only.
 */
import React from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { Chat, ChatMessage } from '@/types'

const api = vi.hoisted(() => ({ getChat: vi.fn(), getChatMessages: vi.fn(), getChatHistory: vi.fn(), cancelChatTurn: vi.fn() }))
const grants = vi.hoisted(() => ({ grant: vi.fn(), deny: vi.fn() }))

vi.mock('next/navigation', () => ({
  usePathname: () => '/chat',
  useSearchParams: () => new URLSearchParams(''),
  useRouter: () => ({ replace: vi.fn(), push: vi.fn() }),
}))
vi.mock('sonner', () => ({ toast: { error: vi.fn(), success: vi.fn(), info: vi.fn() } }))
vi.mock('@/lib/chat/api', () => api)
// useChat reads the identity here; the local edition's shapes (lib/auth-hooks.ts), with no token.
vi.mock('@/lib/auth-hooks', () => ({
  useAuth: () => ({ isLoaded: true, isSignedIn: true, getToken: async () => null }),
  useUser: () => ({ isLoaded: true, isSignedIn: true, user: null }),
  useSession: () => ({ isLoaded: true, isSignedIn: true, session: null }),
  useOrganization: () => ({ isLoaded: true, organization: null, membership: null }),
  useClerk: () => ({ signOut: async () => undefined }),
}))
vi.mock('@/lib/page-context', () => ({ usePageContext: () => ({ page: 'chat', route: '/chat' }) }))
vi.mock('@/hooks/use-chat-session', () => ({ useChatSessionHydration: () => ({ key: 'k', hydrated: true }) }))
vi.mock('@/hooks/use-backend-health', () => ({ useBackendHealth: () => ({ connection: 'connected', check: vi.fn() }) }))
vi.mock('@/components/layout/main-layout', () => ({
  MainLayout: ({ children }: { children?: React.ReactNode }) => <div data-testid="frame">{children}</div>,
}))
vi.mock('@/components/local/first-run-nudge', () => ({ FirstRunNudge: () => null }))
vi.mock('@/components/mascot/auto-mascot', () => ({
  AutoMascot: ({ state, caption, detail }: { state: string; caption?: string; detail?: string }) => (
    <div data-testid="auto" data-state={state}>
      {caption} {detail}
    </div>
  ),
}))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: null }),
  useWorkspaceOptional: () => null,
}))
vi.mock('@/hooks/use-approval-grants', () => ({
  useGrantApproval: () => ({ isLoading: false, mutateAsync: grants.grant }),
  useDenyApproval: () => ({ isLoading: false, mutateAsync: grants.deny }),
}))
vi.mock('@/hooks/use-missions-api', () => ({
  useApproveMission: () => ({ isLoading: false, mutateAsync: vi.fn() }),
  useRejectMission: () => ({ isLoading: false, mutateAsync: vi.fn() }),
  useUpdateMissionPlan: () => ({ isLoading: false, mutateAsync: vi.fn() }),
  useCreateMission: () => ({ mutate: vi.fn(), isLoading: false }),
  useMission: () => ({ data: undefined, isLoading: true }),
}))
vi.mock('@/stores/chat-session-store', async () => {
  const { create } = await import('zustand')
  const store = create<any>(() => ({
    hydrated: true,
    session: { activeChatId: 'c1', draftOpen: false, openChatIds: ['c1'] },
    openChat: vi.fn(),
    chatIdAssigned: vi.fn(),
    newDraft: vi.fn(),
    closeTab: vi.fn(),
    setTitles: vi.fn(),
  }))
  return { useChatSessionStore: store }
})

import { APPROVALS_LABEL, LOOSE_APPROVALS_LABEL } from '@/components/play/chat/approval-cards'
import { MISSION_APPROVAL_TITLE, TOOL_APPROVAL_TITLE } from '@/components/play/chat/approvals'
import { AWAITING_ANSWER } from '@/components/play/chat/caption'
import { PLACEHOLDER } from '@/components/play/chat/message-box'
import { PlayChat } from '@/components/play/chat/play-chat'
import { TRY_AGAIN } from '@/components/play/chat/snag-box'
import { apiClient } from '@/lib/api-client'
import { HELLO_KEY } from '@/lib/mascot/session'

// --- the wire: useChat's data stream (`0:` words, `d:` data, `e:` a failure), one stream per request -----------

interface Writable {
  readonly write: (text: string) => void
  readonly end: () => void
}

interface Turn {
  readonly push: (line: string) => void
  readonly end: () => void
}

const turns: Turn[] = []
/** The SSE lanes useBoardEventStream opened (GET /api/v1/tasks/stream), newest last. */
const lanes: Writable[] = []

/** A response body the test writes into; an abort (Stop, an unmount) fails the read as a browser's does. */
function writableResponse(signal: AbortSignal | null | undefined): { response: Response; body: Writable } {
  const encoder = new TextEncoder()
  let controller: ReadableStreamDefaultController<Uint8Array> | null = null
  const stream = new ReadableStream<Uint8Array>({
    start(c) {
      controller = c
    },
  })
  signal?.addEventListener('abort', () => controller?.error(new DOMException('The user aborted a request.', 'AbortError')))
  const body = { write: (text: string) => controller?.enqueue(encoder.encode(text)), end: () => controller?.close() }
  return { response: { ok: true, status: 200, headers: { get: () => null }, body: stream } as unknown as Response, body }
}

/** A /api/chat response body the test writes line by line. */
function chatResponse(signal: AbortSignal | null | undefined): Response {
  const { response, body } = writableResponse(signal)
  turns.push({ push: (line) => body.write(`${line}\n`), end: body.end })
  return response
}

const chatEndpoint = vi.fn(async (url: string, init?: RequestInit) => {
  if (url !== '/api/chat') throw new Error(`unexpected request: ${url}`)
  return chatResponse(init?.signal)
})

const laneEndpoint = vi.fn(async (_url: string, init?: RequestInit) => {
  const { response, body } = writableResponse(init?.signal)
  lanes.push(body)
  return response
})

/** The network: the SSE lane to its own stream, everything else to the chat endpoint. */
function network(url: string, init?: RequestInit): Promise<Response> {
  const lane = `${apiClient.getBaseUrl()}/api/v1/tasks/stream`
  return url === lane ? laneEndpoint(url, init) : chatEndpoint(url, init)
}

const dataLine = (type: string, data: Record<string, unknown>) => `d:${JSON.stringify({ type, data })}`
const wordsLine = (text: string) => `0:${JSON.stringify(text)}`

/** Write lines into the newest request's stream, and let useChat read them. */
async function stream(...lines: string[]) {
  await act(async () => {
    lines.forEach((line) => turns[turns.length - 1].push(line))
    await new Promise((resolve) => setTimeout(resolve, 0))
  })
}

async function endStream() {
  await act(async () => {
    turns[turns.length - 1].end()
    await new Promise((resolve) => setTimeout(resolve, 0))
  })
}

// --- core fixtures: types/chat.ts' shapes, result_formatter.py's payloads ----------------------------------------

const message = (id: string, role: 'user' | 'assistant', text: string) =>
  ({ id, role, content: text, parts: [{ type: 'text', text }] }) as unknown as ChatMessage
const STORED_CHAT: Chat = { id: 'c1', title: 'Tidy the shelf', userId: 'u1', createdAt: '', updatedAt: '', visibility: 'private', turnInFlight: false }
const ASK = 'Delete the old brief?'
const TOOL_APPROVAL = {
  grant_id: 42,
  action: 'platform_delete_document',
  message: 'This action (destructive) requires confirmation.',
  permission_level: 'destructive',
  params: { document_id: 7 },
  risk_tier: 'human_in_the_loop',
  requires_approval: true,
}
const MISSION_APPROVAL = { mission_id: 'm-1', goal: 'Refund the customer', state: 'awaiting_approval', tasks: [{ title: 'Issue refund' }] }
const APPROVAL_LINE = dataLine('tool-data', { tool_approval: TOOL_APPROVAL })

const auto = () => screen.getByTestId('auto')
const replies = () => screen.getAllByRole('article', { name: 'Auto said' })
const looseCards = () => screen.getByRole('region', { name: LOOSE_APPROVALS_LABEL })
const click = (button: HTMLElement) => act(async () => {
  fireEvent.click(button)
})

/** PlayChat under a QueryClient, as the root providers (components/providers.tsx) give it: the lane needs one. */
function renderChat() {
  return render(
    <QueryClientProvider client={new QueryClient()}>
      <PlayChat />
    </QueryClientProvider>,
  )
}

/** Open the stored chat and send ASK through the message box: useChat posts to /api/chat. */
async function openAndSend() {
  renderChat()
  await screen.findByRole('list', { name: 'Conversation' })
  const box = screen.getByRole('textbox', { name: PLACEHOLDER })
  fireEvent.change(box, { target: { value: ASK } })
  fireEvent.keyDown(box, { key: 'Enter' })
  await waitFor(() => expect(chatEndpoint).toHaveBeenCalledTimes(1))
  await waitFor(() => expect(auto()).toHaveAttribute('data-state', 'thinking'))
}

beforeEach(() => {
  window.sessionStorage.setItem(HELLO_KEY, '1') // hello already played this session
  Object.values(api).forEach((fn) => fn.mockReset())
  api.getChat.mockResolvedValue(STORED_CHAT)
  api.getChatMessages.mockResolvedValue([message('m1', 'user', 'Hello'), message('m2', 'assistant', 'Hi.')])
  api.getChatHistory.mockResolvedValue([])
  api.cancelChatTurn.mockResolvedValue(undefined)
  vi.stubGlobal('fetch', network)
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.clearAllMocks()
  turns.length = 0
  lanes.length = 0
  window.sessionStorage.clear()
})

describe("useChat's d: tool-data line → the card inside the reply being written", () => {
  it("a tool_approval and a mission_approval on the wire become the core's cards in the streaming reply; Auto listens", async () => {
    await openAndSend()
    await stream(wordsLine('That needs your OK first.'), APPROVAL_LINE, dataLine('tool-data', { mission_approval: MISSION_APPROVAL }))
    const writing = replies()[replies().length - 1]
    expect(writing).toHaveAttribute('aria-busy', 'true')
    const cards = within(writing).getByRole('region', { name: APPROVALS_LABEL })
    expect(within(cards).getByText(TOOL_APPROVAL_TITLE)).toBeInTheDocument()
    expect(within(cards).getByText(MISSION_APPROVAL_TITLE)).toBeInTheDocument()
    expect(within(cards).getByText('document_id')).toBeInTheDocument()
    expect(auto()).toHaveAttribute('data-state', 'listening')
    await endStream()
    expect(auto()).toHaveAttribute('data-state', 'listening')
    expect(auto()).toHaveTextContent(AWAITING_ANSWER)
  })
})

describe('a card outlives its reply', () => {
  it('Stop drops the reply, the card stays under the conversation and answering it returns Auto to idle', async () => {
    grants.grant.mockResolvedValue({ grant: { id: 42, status: 'revoked', details: { executed_result: { success: true } } } })
    await openAndSend()
    await stream(APPROVAL_LINE)
    await click(screen.getByRole('button', { name: 'Stop' }))
    await waitFor(() => expect(api.cancelChatTurn).toHaveBeenCalledWith('c1'))
    await waitFor(() => expect(replies()).toHaveLength(1)) // the stopped reply is gone, as useChat drops it
    expect(screen.queryAllByRole('region', { name: APPROVALS_LABEL })).toHaveLength(0)
    expect(within(looseCards()).getByText(TOOL_APPROVAL_TITLE)).toBeInTheDocument()
    expect(auto()).toHaveAttribute('data-state', 'listening')
    expect(auto()).toHaveTextContent(AWAITING_ANSWER)

    await click(within(looseCards()).getByRole('button', { name: /approve & run/i }))
    expect(grants.grant).toHaveBeenCalledWith(42)
    await waitFor(() => expect(within(looseCards()).getByRole('status')).toHaveTextContent(/the action ran/i))
    await waitFor(() => expect(auto()).toHaveAttribute('data-state', 'idle'))
  })

  it('Try again replaces the failed reply; the card stays once, its X closes it and Auto idles', async () => {
    await openAndSend()
    await stream(APPROVAL_LINE, 'e:{"message":"The model stopped answering."}')
    await endStream()
    await click(within(replies()[replies().length - 1]).getByRole('button', { name: TRY_AGAIN }))
    await waitFor(() => expect(chatEndpoint).toHaveBeenCalledTimes(2))
    await stream(wordsLine('Here it is again.'))
    await endStream()
    expect(screen.queryAllByRole('region', { name: APPROVALS_LABEL })).toHaveLength(0)
    expect(screen.getAllByText(TOOL_APPROVAL_TITLE)).toHaveLength(1)
    expect(auto()).toHaveAttribute('data-state', 'listening')

    await click(within(looseCards()).getByRole('button', { name: 'Close' }))
    expect(screen.queryByRole('region', { name: LOOSE_APPROVALS_LABEL })).toBeNull()
    await waitFor(() => expect(auto()).toHaveAttribute('data-state', 'idle'))
  })
})

describe('a card settled while the turn still runs', () => {
  it('returns Auto to the tool still running (its start was outranked by the card), then idle when it ends', async () => {
    grants.deny.mockResolvedValue({ grant: { id: 42, status: 'denied' } })
    await openAndSend()
    await stream(APPROVAL_LINE, dataLine('tool-start', { toolCallId: 't1', toolName: 'platform_web_search', input: { query: 'old brief' } }))
    expect(auto()).toHaveAttribute('data-state', 'listening')
    const writing = replies()[replies().length - 1]
    await click(within(within(writing).getByRole('region', { name: APPROVALS_LABEL })).getByRole('button', { name: /deny/i }))
    expect(grants.deny).toHaveBeenCalledWith(42)
    await waitFor(() => expect(auto()).toHaveAttribute('data-state', 'searching'))
    await stream(dataLine('tool-end', { toolCallId: 't1', toolName: 'platform_web_search', success: true }))
    expect(auto()).toHaveAttribute('data-state', 'idle')
    await endStream()
    expect(auto()).toHaveAttribute('data-state', 'idle')
  })
})

describe('the chat_changed lane', () => {
  it('a reply the server finishes detached merges in when the lane says the chat changed: Send comes back, Auto idles', async () => {
    const asked = message('m3', 'user', ASK)
    api.getChat.mockResolvedValue({ ...STORED_CHAT, turnInFlight: true })
    api.getChatMessages.mockResolvedValue([message('m1', 'user', 'Hello'), message('m2', 'assistant', 'Hi.'), asked])
    renderChat()
    await screen.findByRole('list', { name: 'Conversation' })
    expect(screen.getByRole('button', { name: 'Stop' })).toBeInTheDocument()
    await waitFor(() => expect(auto()).toHaveAttribute('data-state', 'thinking'))
    await waitFor(() => expect(lanes).toHaveLength(1))

    api.getChatMessages.mockResolvedValue([message('m1', 'user', 'Hello'), message('m2', 'assistant', 'Hi.'), asked, message('m4', 'assistant', 'Here it is.')])
    await act(async () => {
      lanes[0].write('event: chat_changed\ndata: {"chat_id":"c1"}\n\n')
      await new Promise((resolve) => setTimeout(resolve, 0))
    })
    expect(await screen.findByText('Here it is.')).toBeInTheDocument()
    expect(api.getChatMessages).toHaveBeenCalledTimes(2)
    expect(screen.getByRole('button', { name: 'Send' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Stop' })).toBeNull()
    await waitFor(() => expect(auto()).toHaveAttribute('data-state', 'idle'))
    expect(chatEndpoint).not.toHaveBeenCalled()
  })
})

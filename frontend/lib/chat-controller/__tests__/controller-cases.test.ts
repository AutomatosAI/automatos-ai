/**
 * PRD-261 A-19e1 — the chat controller (lib/chat-controller/controller.ts) through the core's REAL
 * stream hook: useChatWithReceipts and useChat (lib/chat/hooks.ts) read a stubbed /api/chat body, so
 * every case goes from the wire to the view model a theme renders, with no theme around it. Only the
 * network (fetch), the chat API, the identity, the page context, the session store and the toasts are
 * stand-ins.
 *
 * It re-homes the core assertions of builtin Play's chat tests, which leave core with Play (W-A8; the
 * kit keeps their Play-rendering assertions, B-05c):
 * - components/play/chat/__tests__/play-chat-approvals-stream.test.tsx: Stop keeps the reply being
 *   written (P260-FX-30: its words, a tool call still running, a card it raised; the next reply streams
 *   after it) while Studio's hook alone still drops it; the cards a `d:` line raises, filed in wire order
 *   under the reply being written; a card outliving the reply Try again replaced; a card settled
 *   mid-turn; a reply the server finished detached merging in on chat_changed;
 * - play-chat-turn.test.tsx: a reply that failed in words and Try again, a send the server refused and a
 *   reply blocked mid-stream (a spent trial), a Deliverable part with a bad id;
 * - play-chat.test.tsx: the history lists asked again when a turn ends (P260-FX-10, M44);
 * - play-chat-approvals.test.tsx: parts that only look like approvals (approvals.ts itself:
 *   lib/chat/__tests__/approvals.test.ts).
 * The conversation on screen, the host's cards and a stored reply: lib/theme-sdk/__tests__/
 * chat-controller-cases.test.tsx and chat-controller-cards-cases.test.tsx.
 */
import { createElement, type ReactNode } from 'react'
import { QueryClient, QueryClientProvider, useQuery } from '@tanstack/react-query'
import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ChatMessage } from '@/types'

const api = vi.hoisted(() => ({ getChatMessages: vi.fn(), getChatHistory: vi.fn(), cancelChatTurn: vi.fn() }))
const session = vi.hoisted(() => ({ chatIdAssigned: vi.fn() }))

vi.mock('sonner', () => ({ toast: { error: vi.fn(), success: vi.fn(), info: vi.fn() } }))
vi.mock('@/lib/chat/api', () => api)
// useChat reads the identity here; the local edition's shape (lib/auth-hooks.ts), with no token.
vi.mock('@/lib/auth-hooks', () => ({ useAuth: () => ({ isLoaded: true, isSignedIn: true, getToken: async () => null }) }))
vi.mock('@/lib/page-context', () => ({ usePageContext: () => ({ page: 'chat', route: '/chat' }) }))
vi.mock('@/stores/chat-session-store', () => ({ useChatSessionStore: { getState: () => session } }))

import { CHAT_HISTORY_QUERY_PREFIX } from '@/hooks/use-refresh-history-on-turn-end'
import type { FiledApproval } from '@/lib/chat/approvals'
import { messageText } from '@/lib/chat/message-text'
import { useChatWithReceipts } from '@/lib/chat/use-chat-with-receipts'
import { useChatApprovalsStore } from '@/lib/chat-controller/approvals'
import { useChatController } from '@/lib/chat-controller/controller'
import type { ChatConversationView, ChatTurnEvent } from '@/lib/chat-controller/view-model'
import { TRIAL_EXHAUSTED_CODE } from '@/lib/trial'

// --- the wire: useChat's data stream (`0:` words, `d:` data, `e:` a failure), one stream per request ------------

interface Turn {
  readonly push: (line: string) => void
  readonly end: () => void
}

/** The body useChat posts: the words of the message are its first part's text. */
type ChatPost = { readonly message?: { readonly parts?: ReadonlyArray<{ readonly text?: string }> } }

type Refusal = { readonly status: number; readonly body: string }

const CHAT_PATH = '/api/chat'
/** What useBoardEventStream fans the lane's chat_changed out as (hooks/use-board-event-stream.ts). */
const CHAT_CHANGED_EVENT = 'automatos:chat-changed'
const turns: Turn[] = []
const posts: ChatPost[] = []
const wire: { refusal: Refusal | null } = { refusal: null }

/** A response body the test writes into; an abort (Stop, an unmount) fails the read as a browser's does. */
function streamResponse(signal: AbortSignal | null | undefined): Response {
  const encoder = new TextEncoder()
  let controller: ReadableStreamDefaultController<Uint8Array> | null = null
  const body = new ReadableStream<Uint8Array>({
    start(c) {
      controller = c
    },
  })
  signal?.addEventListener('abort', () => controller?.error(new DOMException('The user aborted a request.', 'AbortError')))
  turns.push({ push: (line) => controller?.enqueue(encoder.encode(`${line}\n`)), end: () => controller?.close() })
  return { ok: true, status: 200, headers: { get: () => null }, body } as unknown as Response
}

function refusedResponse({ status, body }: Refusal): Response {
  return { ok: false, status, headers: { get: () => null }, text: async () => body } as unknown as Response
}

const chatEndpoint = vi.fn(async (url: string, init?: RequestInit): Promise<Response> => {
  if (url !== CHAT_PATH) throw new Error(`unexpected request: ${url}`)
  posts.push(JSON.parse(String(init?.body)) as ChatPost)
  return wire.refusal ? refusedResponse(wire.refusal) : streamResponse(init?.signal)
})

const tick = () => new Promise<void>((resolve) => setTimeout(resolve, 0))
const dataLine = (type: string, data: Record<string, unknown>) => `d:${JSON.stringify({ type, data })}`
const wordsLine = (text: string) => `0:${JSON.stringify(text)}`

/** Write lines into the newest request's stream, and let useChat read them; endStream closes it. */
async function stream(...lines: string[]) {
  await act(async () => {
    lines.forEach((line) => turns[turns.length - 1].push(line))
    await tick()
  })
}

const endStream = () => act(async () => {
  turns[turns.length - 1].end()
  await tick()
})

// --- core fixtures: types/chat.ts' shapes, result_formatter.py's payloads -------------------------------------------

const CHAT_ID = 'c1'
const message = (id: string, role: 'user' | 'assistant', text: string) =>
  ({ id, role, content: text, parts: [{ type: 'text', text }] }) as unknown as ChatMessage
const HELLO = message('m1', 'user', 'Hello')
const HI = message('m2', 'assistant', 'Hi.')
const STORED: ChatConversationView = { key: CHAT_ID, chatId: CHAT_ID, title: null, messages: [HELLO, HI], awaitingReply: false }
const ASK = 'Delete the old brief?'
const PARTIAL = 'The keeper climbed the stairs'
const AGAIN = 'Here it is again.'
const TURN_ERROR = 'The model stopped answering.'
const ERROR_LINE = `e:${JSON.stringify({ message: TURN_ERROR, code: 'timeout' })}`
const TOOL_APPROVAL = { grant_id: 42, action: 'platform_delete_document', params: { document_id: 7 }, risk_tier: 'human_in_the_loop' }
const MISSION_APPROVAL = { mission_id: 'm-1', goal: 'Refund the customer', task_count: 1, tasks: [{ title: 'Issue refund' }] }
const APPROVAL_LINE = dataLine('tool-data', { tool_approval: TOOL_APPROVAL })
const MISSION_LINE = dataLine('tool-data', { mission_approval: MISSION_APPROVAL })
const SEARCH_START = dataLine('tool-start', { toolCallId: 't1', toolName: 'platform_web_search', input: { query: 'old brief' } })

// --- the controller under the root providers' QueryClient (components/providers.tsx), a history list beside it -----

/** The root providers' fresh window: within it, nothing is asked again on its own. */
const ROOT_STALE_MS = 60_000
const HISTORY_KEY = [...CHAT_HISTORY_QUERY_PREFIX, 'controller-cases']
const HISTORY_LIMIT = 20

function renderController(view: ChatConversationView = STORED) {
  const events: string[] = []
  const client = new QueryClient({ defaultOptions: { queries: { staleTime: ROOT_STALE_MS, retry: false } } })
  const wrapper = ({ children }: { readonly children: ReactNode }) => createElement(QueryClientProvider, { client }, children)
  const onEvent = (event: ChatTurnEvent) => events.push(event.type)
  const onData = (part: unknown) => events.push(`theme:${(part as { type?: string }).type}`)
  const hook = renderHook(
    () => {
      // A chat-history list beside the chat, as Conversations and Recent sit beside it (P260-FX-10).
      useQuery({ queryKey: HISTORY_KEY, queryFn: () => api.getChatHistory(HISTORY_LIMIT) })
      return useChatController({ view, onEvent, onData })
    },
    { wrapper },
  )
  return { ...hook, events }
}

type Hook = ReturnType<typeof renderController>

const newest = (hook: Hook): ChatMessage => hook.result.current.messages[hook.result.current.messages.length - 1]
const texts = (messages: readonly ChatMessage[]) => messages.map(messageText)
const statuses = (cards: readonly FiledApproval[]) => cards.map((card) => `${card.key}:${card.status}`)
const turnEvents = (hook: Hook) => hook.events.filter((event) => !event.startsWith('theme:'))

/** Send through the controller: useChat posts to /api/chat. */
async function ask(hook: Hook, text: string) {
  const before = posts.length
  act(() => {
    expect(hook.result.current.send(text)).toBe(true)
  })
  await waitFor(() => expect(posts).toHaveLength(before + 1))
}

/** Stop; the cancel reaches the server; useChat's catch has run (it drops the reply it was writing). */
async function stop(hook: Hook) {
  act(() => hook.result.current.stop())
  await waitFor(() => expect(api.cancelChatTurn).toHaveBeenCalledWith(CHAT_ID))
  await act(tick)
}

/** The turn fails after these lines; Try again asks the same words again and its reply replaces the failed one. */
async function failThenTryAgain(hook: Hook, ...lines: string[]) {
  await stream(...lines, ERROR_LINE)
  await endStream()
  act(() => {
    expect(hook.result.current.retry()).toBe(true)
  })
  await waitFor(() => expect(posts).toHaveLength(2))
  await stream(wordsLine(AGAIN))
  await endStream()
}

beforeEach(() => {
  Object.values(api).forEach((fn) => fn.mockReset())
  api.getChatHistory.mockResolvedValue([])
  api.cancelChatTurn.mockResolvedValue(undefined)
  session.chatIdAssigned.mockReset()
  useChatApprovalsStore.setState({ filed: [] })
  vi.stubGlobal('fetch', chatEndpoint)
})

afterEach(() => {
  vi.unstubAllGlobals()
  chatEndpoint.mockClear()
  turns.length = 0
  posts.length = 0
  wire.refusal = null
})

describe('Stop keeps the reply being written (P260-FX-30, decision F4)', () => {
  it('keeps the words so far once useChat drops the reply, cancels the turn, and the next reply streams after it', async () => {
    const hook = renderController()
    await ask(hook, ASK)
    await stream(wordsLine(PARTIAL))
    const writing = newest(hook)
    await stop(hook)
    const { messages, liveMessages, isStopped, streaming } = hook.result.current
    expect(liveMessages.map((shown) => shown.id)).not.toContain(writing.id)
    expect(texts(messages)).toEqual(['Hello', 'Hi.', ASK, PARTIAL])
    expect(isStopped(writing.id)).toBe(true)
    expect(streaming).toBe(false)
    await ask(hook, 'A shorter one, please')
    await stream(wordsLine('Here is a short one.'))
    expect(hook.result.current.streaming).toBe(true)
    await endStream()
    const shown = hook.result.current.messages
    expect(texts(shown)).toEqual(['Hello', 'Hi.', ASK, PARTIAL, 'A shorter one, please', 'Here is a short one.'])
    expect(shown.filter((reply) => hook.result.current.isStopped(reply.id)).map(messageText)).toEqual([PARTIAL])
  })

  it('keeps the empty reply as stopped when Stop comes before the first word', async () => {
    const hook = renderController()
    await ask(hook, ASK)
    const writing = newest(hook)
    await stop(hook)
    expect(hook.result.current.messages.map((shown) => shown.id)).toEqual([HELLO.id, HI.id, expect.any(String), writing.id])
    expect(hook.result.current.isStopped(writing.id)).toBe(true)
  })

  it('keeps its tool call still running and the card it raised inside it; answering the card ends the wait', async () => {
    const hook = renderController()
    await ask(hook, ASK)
    await stream(SEARCH_START, APPROVAL_LINE)
    const writing = newest(hook)
    await stop(hook)
    const kept = hook.result.current.messages.find((shown) => shown.id === writing.id)
    expect(kept?.toolCalls?.map((call) => call.state)).toEqual(['running'])
    expect(statuses(hook.result.current.approvals.of(writing.id))).toEqual(['tool-42:waiting'])
    expect(hook.result.current.approvals.loose).toEqual([])
    expect(hook.result.current.approvals.waiting).toBe(true)
    act(() => hook.result.current.approvals.answered('tool-42'))
    expect(hook.result.current.approvals.waiting).toBe(false)
    expect(turnEvents(hook)).toEqual(['approval-waiting', 'approval-settled'])
  })

  it("leaves Studio's hook as it is: useChatWithReceipts on its own still drops the reply on Stop", async () => {
    const { result } = renderHook(() => useChatWithReceipts({ id: CHAT_ID, initialMessages: [] }))
    act(() => {
      void result.current.sendMessage(ASK)
    })
    await waitFor(() => expect(posts).toHaveLength(1))
    await stream(wordsLine(PARTIAL))
    expect(result.current.messages.map((shown) => shown.role)).toEqual(['user', 'assistant'])
    act(() => result.current.stop())
    await waitFor(() => expect(api.cancelChatTurn).toHaveBeenCalledWith(CHAT_ID))
    await act(tick)
    expect(result.current.messages.map((shown) => shown.role)).toEqual(['user'])
    expect(result.current.status).toBe('idle')
  })
})

describe('Try again after a failed turn', () => {
  it('marks a reply that failed in words, tells the theme once, and asks the same words again in its place', async () => {
    const hook = renderController()
    await ask(hook, ASK)
    await stream(wordsLine(PARTIAL), ERROR_LINE)
    await endStream()
    const failed = newest(hook)
    expect(failed.error).toEqual({ message: TURN_ERROR, code: 'timeout' })
    expect(hook.result.current.streaming).toBe(false)
    expect(turnEvents(hook)).toEqual(['turn-failed'])
    act(() => {
      expect(hook.result.current.retry()).toBe(true)
    })
    await waitFor(() => expect(posts).toHaveLength(2))
    expect(posts[1].message?.parts?.[0]?.text).toBe(ASK)
    await stream(wordsLine(AGAIN))
    await endStream()
    expect(texts(hook.result.current.messages)).toEqual(['Hello', 'Hi.', ASK, AGAIN])
    expect(hook.result.current.messages.map((shown) => shown.id)).not.toContain(failed.id)
    expect(turnEvents(hook)).toEqual(['turn-failed'])
  })

  it('a send the server refuses (a spent trial) leaves no reply and says why; Try again asks again', async () => {
    wire.refusal = { status: 402, body: JSON.stringify({ detail: { code: TRIAL_EXHAUSTED_CODE } }) }
    const hook = renderController()
    await ask(hook, ASK)
    await waitFor(() => expect(hook.result.current.status).toBe('error'))
    expect(hook.result.current.errorCode).toBe(TRIAL_EXHAUSTED_CODE)
    expect(texts(hook.result.current.messages)).toEqual(['Hello', 'Hi.', ASK])
    wire.refusal = null
    act(() => {
      expect(hook.result.current.retry()).toBe(true)
    })
    await waitFor(() => expect(posts).toHaveLength(2))
    expect(posts[1].message?.parts?.[0]?.text).toBe(ASK)
  })

  it('a reply the trial blocks mid-stream carries its failure and says trial_exhausted', async () => {
    const hook = renderController()
    await ask(hook, ASK)
    await stream(`e:${JSON.stringify({ message: 'Trial used up.', code: TRIAL_EXHAUSTED_CODE })}`)
    await endStream()
    expect(newest(hook).error).toEqual({ message: 'Trial used up.', code: TRIAL_EXHAUSTED_CODE })
    expect(hook.result.current.errorCode).toBe(TRIAL_EXHAUSTED_CODE)
  })
})

describe('approvals on the wire', () => {
  it('files each card a `d:` line raises under the reply being written, in wire order, the theme reading each part first', async () => {
    const hook = renderController()
    await ask(hook, ASK)
    await stream(wordsLine('That needs your OK first.'), APPROVAL_LINE, MISSION_LINE)
    const writing = newest(hook)
    expect(hook.result.current.streaming).toBe(true)
    expect(statuses(hook.result.current.approvals.of(writing.id))).toEqual(['tool-42:waiting', 'mission-m-1:waiting'])
    expect(hook.events).toEqual(['theme:tool-data', 'approval-waiting', 'theme:tool-data', 'approval-waiting'])
    await endStream()
    expect(hook.result.current.approvals.waiting).toBe(true)
    expect(statuses(hook.result.current.approvals.of(writing.id))).toHaveLength(2)
  })

  it('tells the theme a card settled while the turn still runs, and the turn carries on', async () => {
    const hook = renderController()
    await ask(hook, ASK)
    await stream(APPROVAL_LINE, SEARCH_START)
    act(() => hook.result.current.approvals.answered('tool-42'))
    expect(hook.result.current.streaming).toBe(true)
    expect(turnEvents(hook)).toEqual(['approval-waiting', 'approval-settled'])
    await endStream()
    expect(hook.result.current.approvals.waiting).toBe(false)
  })

  it('a card whose reply Try again replaced is loose: answered, it stays there answered; closed, it goes', async () => {
    const hook = renderController()
    await ask(hook, ASK)
    await failThenTryAgain(hook, APPROVAL_LINE, MISSION_LINE)
    expect(texts(hook.result.current.messages)).toEqual(['Hello', 'Hi.', ASK, AGAIN])
    expect(statuses(hook.result.current.approvals.loose)).toEqual(['tool-42:waiting', 'mission-m-1:waiting'])
    act(() => hook.result.current.approvals.answered('tool-42'))
    expect(statuses(hook.result.current.approvals.loose)).toEqual(['tool-42:answered', 'mission-m-1:waiting'])
    expect(hook.result.current.approvals.waiting).toBe(true)
    act(() => hook.result.current.approvals.closed('mission-m-1'))
    expect(statuses(hook.result.current.approvals.loose)).toEqual(['tool-42:answered'])
    expect(hook.result.current.approvals.waiting).toBe(false)
    expect(turnEvents(hook)).toEqual(['approval-waiting', 'approval-waiting', 'turn-failed', 'approval-settled'])
  })

  it('files nothing for a part that only looks like one: bad ids, a tool-start, other tool data, a bad Deliverable id', async () => {
    const hook = renderController()
    await ask(hook, ASK)
    await stream(
      dataLine('tool-data', { tool_approval: { grant_id: '../1', action: 'x' }, mission_approval: { mission_id: '../../admin', goal: 'x' } }),
      dataLine('tool-start', { toolCallId: 't9', toolName: 'platform_web_search', tool_approval: TOOL_APPROVAL }),
      dataLine('tool-data', { documents: [{ id: 'd1' }] }),
      dataLine('tool-data', { generated_document: { deliverable_id: '../../admin?x=1', title: 'Sneaky' } }),
    )
    const writing = newest(hook)
    expect(hook.result.current.approvals.of(writing.id)).toEqual([])
    expect(hook.result.current.approvals.waiting).toBe(false)
    expect(hook.result.current.deliverablesOf(writing.id)).toEqual([])
    expect(turnEvents(hook)).toEqual([])
  })
})

describe("a turn's end and what the server sends besides", () => {
  it('asks every chat-history list again when a reply ends and when a send fails; a render with the status unchanged does not', async () => {
    const hook = renderController()
    await waitFor(() => expect(api.getChatHistory).toHaveBeenCalledTimes(1))
    hook.rerender()
    await ask(hook, ASK)
    await stream(wordsLine(PARTIAL))
    expect(api.getChatHistory).toHaveBeenCalledTimes(1)
    await endStream()
    await waitFor(() => expect(api.getChatHistory).toHaveBeenCalledTimes(2))
    expect(api.getChatHistory).toHaveBeenLastCalledWith(HISTORY_LIMIT)
    wire.refusal = { status: 500, body: '' }
    await ask(hook, 'Again?')
    await waitFor(() => expect(api.getChatHistory).toHaveBeenCalledTimes(3))
  })

  it('merges a reply the server finished detached (the page opened mid-turn) when the lane says the chat changed', async () => {
    const asked = message('m3', 'user', ASK)
    const hook = renderController({ ...STORED, messages: [HELLO, HI, asked], awaitingReply: true })
    expect(hook.result.current.streaming).toBe(true)
    api.getChatMessages.mockResolvedValue([HELLO, HI, asked, message('m4', 'assistant', 'Here it is.')])
    await act(async () => {
      window.dispatchEvent(new CustomEvent(CHAT_CHANGED_EVENT, { detail: { chat_id: CHAT_ID } }))
      await tick()
    })
    await waitFor(() => expect(texts(hook.result.current.messages)).toEqual(['Hello', 'Hi.', ASK, 'Here it is.']))
    expect(api.getChatMessages).toHaveBeenCalledWith(CHAT_ID)
    expect(hook.result.current.streaming).toBe(false)
    expect(posts).toEqual([])
  })
})

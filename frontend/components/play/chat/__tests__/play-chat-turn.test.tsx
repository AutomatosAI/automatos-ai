/**
 * PRD-260 §4 — a turn's details in Play's chat, on the core's data: the tool
 * lines from the reply's ToolCall entries, the receipts in receiptLine's words,
 * the Deliverable card, the snag box with Try again (the hook's reload()), the
 * core's TrialExhaustedBanner for a spent trial, and the core's
 * MissionSuggestionCard for a mission-suggestion part. The chat hook
 * (useChatWithReceipts) and the session store are mocked; the conversation
 * loading, Auto's reducer (useAutoMascot) and the core's cards are real. Auto
 * himself is a stand-in that shows his state. Roles and text only (CSS
 * modules are proxies).
 */
import React from 'react'
import { act, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { Chat, ChatMessage, Receipt, ToolCall } from '@/types'

const chat = vi.hoisted(() => ({
  options: null as null | Record<string, any>,
  status: 'idle' as 'idle' | 'streaming' | 'error',
  errorCode: null as string | null,
  messages: null as null | unknown[],
  sendMessage: vi.fn(),
  stop: vi.fn(),
  reload: vi.fn(),
}))
const api = vi.hoisted(() => ({ getChat: vi.fn(), getChatMessages: vi.fn(), getChatHistory: vi.fn() }))
const missions = vi.hoisted(() => ({ mutate: vi.fn() }))

vi.mock('next/navigation', () => ({
  usePathname: () => '/chat',
  useSearchParams: () => new URLSearchParams(''),
  useRouter: () => ({ replace: vi.fn(), push: vi.fn() }),
}))
vi.mock('sonner', () => ({ toast: { error: vi.fn(), success: vi.fn() } }))
vi.mock('@/lib/chat/api', () => api)
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
// The core banner's own dependencies: no workspace snapshot, and its power-up card as a stand-in.
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: null }),
  useWorkspaceOptional: () => null,
}))
vi.mock('@/components/onboarding/power-up-card', () => ({ PowerUpCard: () => <div data-testid="power-up-card" /> }))
vi.mock('@/hooks/use-missions-api', () => ({
  useCreateMission: () => ({ mutate: missions.mutate, isLoading: false }),
  useMission: () => ({ data: undefined, isLoading: true }),
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
      isLoading: chat.status === 'streaming',
      awaitingReply: false,
      usage: undefined,
      errorCode: chat.errorCode,
    }
  },
}))
vi.mock('@/stores/chat-session-store', async () => {
  const { create } = await import('zustand')
  const store = create<any>((set) => ({
    hydrated: true,
    session: { activeChatId: 'c1', draftOpen: false, openChatIds: ['c1'] },
    openChat: vi.fn(),
    chatIdAssigned: vi.fn(),
    newDraft: vi.fn(),
    closeTab: vi.fn(),
    setTitles: vi.fn(),
    reset: () => set({ session: { activeChatId: 'c1', draftOpen: false, openChatIds: ['c1'] } }),
  }))
  return { useChatSessionStore: store }
})

import { DeliverableCard } from '@/components/play/chat/deliverable-card'
import { RECEIPTS_TITLE } from '@/components/play/chat/receipts'
import { SNAG_TITLE, TRY_AGAIN, TURN_FAILED } from '@/components/play/chat/snag-box'
import { MISSION_CREATED_LABEL, SUGGESTION_LABEL } from '@/components/play/chat/mission-suggestion'
import { TOOL_LINES_LABEL } from '@/components/play/chat/tool-lines'
import { PlayChat } from '@/components/play/chat/play-chat'
import { NO_ACTIONS, NOTHING_ON_THE_BOARD, receiptLine } from '@/lib/chat/receipts'
import { HELLO_KEY } from '@/lib/mascot/session'
import { TRIAL_EXHAUSTED_CODE } from '@/lib/trial'
import { useChatSessionStore } from '@/stores/chat-session-store'
import { useMissionStore } from '@/stores/mission-store'

// --- core fixtures: the ChatMessage, ToolCall and Receipt shapes of types/chat.ts ----------

function userMessage(id: string, text: string): ChatMessage {
  return { id, role: 'user', content: text, parts: [{ type: 'text', text }] } as unknown as ChatMessage
}

function autoMessage(id: string, text: string, extra: Partial<ChatMessage> = {}): ChatMessage {
  return { id, role: 'assistant', content: text, parts: [{ type: 'text', text }], ...extra } as unknown as ChatMessage
}

function receipt(fields: Partial<Receipt>): Receipt {
  return { action: 'platform_update_task', kind: 'write', status: 'done', subject: '', effect: '', link: null, reason: null, ...fields }
}

const STORED_CHAT: Chat = { id: 'c1', title: 'Laptop for uni', userId: 'u1', createdAt: '', updatedAt: '', visibility: 'private', turnInFlight: false }
const QUESTION = userMessage('m1', 'Shortlist three laptops?')

function openChat(messages: ChatMessage[]) {
  api.getChat.mockResolvedValue(STORED_CHAT)
  api.getChatMessages.mockResolvedValue(messages)
}

async function renderChat(messages: ChatMessage[]) {
  openChat(messages)
  const view = render(<PlayChat />)
  await screen.findByRole('list', { name: 'Conversation' })
  return view
}

const auto = () => screen.getByTestId('auto')
const reply = (): HTMLElement => {
  const replies = screen.getAllByRole('article', { name: 'Auto said' })
  return replies[replies.length - 1]
}
const data = (part: unknown) => act(() => chat.options?.onData(part))

beforeEach(() => {
  window.sessionStorage.setItem(HELLO_KEY, '1') // hello already played this session
  ;(useChatSessionStore.getState() as { reset: () => void }).reset()
  useMissionStore.setState({ activePlanningMissionId: null })
  Object.values(api).forEach((fn) => fn.mockReset())
  api.getChatHistory.mockResolvedValue([])
})

afterEach(() => {
  vi.clearAllMocks()
  window.sessionStorage.clear()
  Object.assign(chat, { options: null, status: 'idle', errorCode: null, messages: null })
})

describe('tool lines', () => {
  const calls: ToolCall[] = [
    { toolCallId: 't1', toolName: 'platform_web_search', state: 'completed', summary: '3 results', durationMs: 2140 },
    { toolCallId: 't2', toolName: 'platform_send_email', state: 'error', error: 'the mail server timed out' },
    { toolCallId: 't3', toolName: 'platform_update_agent', state: 'completed', waiting: true, summary: 'card raised' },
    { toolCallId: 't4', toolName: 'platform_web_search', state: 'running', input: { query: 'budget laptops' } },
  ]

  it("renders the reply's tool calls as lines in their states: done, failed, waiting for you, running", async () => {
    chat.status = 'streaming'
    await renderChat([QUESTION, autoMessage('m2', '', { toolCalls: calls })])
    const lines = within(within(reply()).getByRole('list', { name: TOOL_LINES_LABEL })).getAllByRole('listitem')
    expect(lines.map((line) => line.textContent)).toEqual([
      '✓Searched the web · 3 results · 2 s',
      "✕Send email didn't work · the mail server timed out",
      '…Waiting for you · card raised',
      "Searching the web: 'budget laptops'",
    ])
  })

  it('a call still running on a finished reply reads as stopped, never a spinner forever', async () => {
    await renderChat([QUESTION, autoMessage('m2', 'Here is what I found.', { toolCalls: [calls[3]] })])
    const lines = within(reply()).getByRole('list', { name: TOOL_LINES_LABEL })
    expect(lines).toHaveTextContent('Stopped: searching the web')
  })
})

describe('receipts', () => {
  const done = receipt({ subject: '#0422', effect: 'moved to Done', link: '/activity?ticket=422' })
  const refused = receipt({ action: 'platform_send_email', status: 'refused', subject: 'Letter to Maya', reason: 'no mail connection' })

  it("render in receiptLine's words, a done one linked to its page", async () => {
    const receipts = [done, refused]
    await renderChat([QUESTION, autoMessage('m2', 'Done.', { parts: [{ type: 'text', text: 'Done.' }, { type: 'receipts', receipts }] })])
    const section = within(reply()).getByRole('region', { name: RECEIPTS_TITLE })
    const items = within(section).getAllByRole('listitem')
    expect(items.map((item) => item.textContent?.slice(1))).toEqual(receipts.map(receiptLine))
    expect(within(items[0]).getByRole('link', { name: receiptLine(done) })).toHaveAttribute('href', '/activity?ticket=422')
    expect(within(items[1]).queryByRole('link')).toBeNull()
    expect(section).not.toHaveTextContent(NOTHING_ON_THE_BOARD)
  })

  it('a turn that ran nothing says so; one that moved nothing on the board says that', async () => {
    await renderChat([
      QUESTION,
      autoMessage('m2', 'Hi.', { receipts: [] }),
      userMessage('m3', 'And now?'),
      autoMessage('m4', 'Read it.', { receipts: [receipt({ kind: 'read', subject: 'Brief', effect: 'looked up' })] }),
    ])
    const [first, second] = screen.getAllByRole('region', { name: RECEIPTS_TITLE })
    expect(first).toHaveTextContent(NO_ACTIONS)
    expect(second).toHaveTextContent(NOTHING_ON_THE_BOARD)
  })

  it('a done receipt makes Auto celebrate once the turn finishes, not before', async () => {
    const view = await renderChat([QUESTION, autoMessage('m2', '')])
    chat.status = 'streaming'
    view.rerender(<PlayChat />)
    data({ type: 'tool-data', data: { receipts: { receipts: [done], above: [] } } })
    expect(auto()).not.toHaveAttribute('data-state', 'celebrate')
    chat.status = 'idle'
    view.rerender(<PlayChat />)
    expect(auto()).toHaveAttribute('data-state', 'celebrate')
  })
})

describe('the Deliverable card', () => {
  it('a generated_document tool-data part renders the card, opening /deliverables?deliverable=<id>, and Auto celebrates', async () => {
    chat.status = 'streaming'
    await renderChat([QUESTION, autoMessage('m2', '')])
    data({ type: 'tool-data', data: { generated_document: { deliverable_id: 'd-42', title: 'Laptop shortlist', format: 'pdf' } } })
    const card = within(reply()).getByRole('article', { name: 'Deliverable: Laptop shortlist' })
    expect(within(card).getByRole('link', { name: 'Open Laptop shortlist' })).toHaveAttribute('href', '/deliverables?deliverable=d-42')
    expect(card).toHaveTextContent('Document · just now')
    expect(auto()).toHaveAttribute('data-state', 'celebrate')
  })

  it('a bad id renders no link', async () => {
    chat.status = 'streaming'
    await renderChat([QUESTION, autoMessage('m2', '')])
    data({ type: 'tool-data', data: { generated_document: { deliverable_id: '../../admin?x=1', title: 'Sneaky' } } })
    expect(within(reply()).queryByRole('link')).toBeNull()
    expect(screen.queryByRole('article', { name: 'Deliverable: Sneaky' })).toBeNull()
    render(<DeliverableCard deliverable={{ id: 'a b/../c', title: 'Odd', kind: 'document' }} />)
    const odd = screen.getByRole('article', { name: 'Deliverable: Odd' })
    expect(within(odd).queryByRole('link')).toBeNull()
  })

  it("a done receipt that links a Deliverable renders its card (the core's open_url shape)", async () => {
    const linked = receipt({ action: 'generate_document', subject: 'Budget memo', effect: 'made', link: '/deliverables?deliverable=d-7' })
    await renderChat([QUESTION, autoMessage('m2', 'Made it.', { receipts: [linked] })])
    const card = within(reply()).getByRole('article', { name: 'Deliverable: Budget memo' })
    expect(within(card).getByRole('link', { name: 'Open Budget memo' })).toHaveAttribute('href', '/deliverables?deliverable=d-7')
  })
})

describe('the snag box', () => {
  it("a reply that failed shows Play's snag box; Try again calls the hook's reload(), and Auto plays oops", async () => {
    chat.status = 'streaming'
    const view = await renderChat([QUESTION, autoMessage('m2', '')])
    chat.messages = [QUESTION, autoMessage('m2', '', { error: { message: 'The model took too long to answer.', code: 'timeout' } })]
    chat.status = 'idle'
    view.rerender(<PlayChat />)
    const snag = within(reply()).getByRole('region', { name: SNAG_TITLE })
    expect(snag).toHaveTextContent('The model took too long to answer.')
    expect(auto()).toHaveAttribute('data-state', 'oops')
    fireEvent.click(within(snag).getByRole('button', { name: TRY_AGAIN }))
    expect(chat.reload).toHaveBeenCalledTimes(1)
  })

  it("a turn that failed before Auto's reply began (status 'error') shows the snag box under the conversation", async () => {
    chat.status = 'error'
    await renderChat([QUESTION])
    const snag = screen.getByRole('region', { name: SNAG_TITLE })
    expect(snag).toHaveTextContent(TURN_FAILED)
    fireEvent.click(within(snag).getByRole('button', { name: TRY_AGAIN }))
    expect(chat.reload).toHaveBeenCalledTimes(1)
    expect(auto()).toHaveAttribute('data-state', 'oops')
  })

  it('only the newest reply gets Try again', async () => {
    const failed = (id: string) => autoMessage(id, '', { error: { message: 'It failed.' } })
    await renderChat([QUESTION, failed('m2'), userMessage('m3', 'Again?'), failed('m4')])
    const snags = screen.getAllByRole('region', { name: SNAG_TITLE })
    expect(snags).toHaveLength(2)
    expect(within(snags[0]).queryByRole('button', { name: TRY_AGAIN })).toBeNull()
    expect(within(snags[1]).getByRole('button', { name: TRY_AGAIN })).toBeInTheDocument()
  })
})

describe('a spent trial', () => {
  it("trial_exhausted renders the core's TrialExhaustedBanner, not a snag", async () => {
    chat.status = 'error'
    chat.errorCode = TRIAL_EXHAUSTED_CODE
    await renderChat([QUESTION])
    expect(screen.getByTestId('trial-exhausted-banner')).toHaveTextContent('Your trial credit is used up')
    expect(screen.queryByRole('region', { name: SNAG_TITLE })).toBeNull()
  })

  it('a reply blocked mid-stream by the trial shows the banner instead of its snag box', async () => {
    chat.errorCode = TRIAL_EXHAUSTED_CODE
    await renderChat([QUESTION, autoMessage('m2', '', { error: { message: 'Trial used up.', code: TRIAL_EXHAUSTED_CODE } })])
    expect(screen.getByTestId('trial-exhausted-banner')).toBeInTheDocument()
    expect(screen.queryByRole('region', { name: SNAG_TITLE })).toBeNull()
  })
})

describe('a Mission suggestion', () => {
  const suggestion = { type: 'mission-suggestion', data: { goal: 'Research ten competitors and compare pricing', complexity: 'organ', agent_id: 3 } }

  it("a mission-suggestion part renders the core's MissionSuggestionCard with the goal, once the turn is over", async () => {
    chat.status = 'streaming'
    const view = await renderChat([QUESTION, autoMessage('m2', 'That is a big one.')])
    data(suggestion)
    expect(screen.queryByRole('region', { name: SUGGESTION_LABEL })).toBeNull()
    chat.status = 'idle'
    view.rerender(<PlayChat />)
    const card = screen.getByRole('region', { name: SUGGESTION_LABEL })
    expect(card).toHaveTextContent('Research ten competitors and compare pricing')
    expect(card).toHaveTextContent('Multi-step')
    fireEvent.click(within(card).getByRole('button', { name: /Launch Mission/ }))
    expect(missions.mutate).toHaveBeenCalledWith(
      {
        goal: 'Research ten competitors and compare pricing',
        config: {
          source: 'chat',
          chat_id: 'c1',
          context_messages: [
            { role: 'user', content: 'Shortlist three laptops?' },
            { role: 'assistant', content: 'That is a big one.' },
          ],
        },
      },
      expect.anything(),
    )
  })

  it('respects the activePlanningMissionId gate: a Mission being planned shows its card instead', async () => {
    await renderChat([QUESTION, autoMessage('m2', 'That is a big one.')])
    data(suggestion)
    expect(screen.getByRole('region', { name: SUGGESTION_LABEL })).toBeInTheDocument()
    act(() => useMissionStore.setState({ activePlanningMissionId: 'mission-1' }))
    expect(screen.queryByRole('region', { name: SUGGESTION_LABEL })).toBeNull()
    expect(screen.getByRole('region', { name: MISSION_CREATED_LABEL })).toBeInTheDocument()
  })

  it('a new message clears the suggestion, as chat.tsx does', async () => {
    await renderChat([QUESTION, autoMessage('m2', 'That is a big one.')])
    data(suggestion)
    const box = screen.getByRole('textbox')
    fireEvent.change(box, { target: { value: 'No, just a quick answer' } })
    fireEvent.keyDown(box, { key: 'Enter' })
    expect(chat.sendMessage).toHaveBeenCalledWith('No, just a quick answer')
    expect(screen.queryByRole('region', { name: SUGGESTION_LABEL })).toBeNull()
  })
})

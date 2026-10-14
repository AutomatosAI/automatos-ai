/**
 * PRD-260 T-005c — approvals in Play's chat, as the core chat shows them: a
 * turn's tool-data `tool_approval` and `mission_approval` (chat.tsx's 'Action
 * approval needed' and 'Mission plan — approval needed' widgets) render inside
 * Auto's reply as the core's own ToolApprovalWidget and MissionApprovalWidget,
 * whose Approve & run, Deny and Reject call the core's hooks exactly as in
 * Studio (mocked here at the hook, as the core's widget tests do), and Auto
 * listens while a card waits for the owner. The chat hook (useChatWithReceipts)
 * and the session store are mocked; the conversation loading, Auto's reducer,
 * usePlayTurn and the core's widgets are real. Roles and text only.
 */
import React from 'react'
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { toast } from 'sonner'

import type { Chat, ChatMessage } from '@/types'

const chat = vi.hoisted(() => ({
  options: null as null | Record<string, any>,
  status: 'idle' as 'idle' | 'streaming' | 'error',
  messages: null as null | unknown[],
  sendMessage: vi.fn(),
  stop: vi.fn(),
  reload: vi.fn(),
}))
const api = vi.hoisted(() => ({ getChat: vi.fn(), getChatMessages: vi.fn(), getChatHistory: vi.fn() }))
const grants = vi.hoisted(() => ({ grant: vi.fn(), deny: vi.fn() }))
const missions = vi.hoisted(() => ({ approve: vi.fn(), reject: vi.fn(), updatePlan: vi.fn(), create: vi.fn() }))

vi.mock('next/navigation', () => ({
  usePathname: () => '/chat',
  useSearchParams: () => new URLSearchParams(''),
  useRouter: () => ({ replace: vi.fn(), push: vi.fn() }),
}))
vi.mock('sonner', () => ({ toast: { error: vi.fn(), success: vi.fn(), info: vi.fn() } }))
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
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: null }),
  useWorkspaceOptional: () => null,
}))
// The core's approval hooks, as the core's own widget tests mock them: the widgets call these and nothing else.
vi.mock('@/hooks/use-approval-grants', () => ({
  useGrantApproval: () => ({ isLoading: false, mutateAsync: grants.grant }),
  useDenyApproval: () => ({ isLoading: false, mutateAsync: grants.deny }),
}))
vi.mock('@/hooks/use-missions-api', () => ({
  useApproveMission: () => ({ isLoading: false, mutateAsync: missions.approve }),
  useRejectMission: () => ({ isLoading: false, mutateAsync: missions.reject }),
  useUpdateMissionPlan: () => ({ isLoading: false, mutateAsync: missions.updatePlan }),
  useCreateMission: () => ({ mutate: missions.create, isLoading: false }),
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
      errorCode: null,
    }
  },
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

import { APPROVALS_LABEL } from '@/components/play/chat/approval-cards'
import {
  approvalsInPart,
  fileApprovals,
  looseApprovals,
  MISSION_APPROVAL_TITLE,
  TOOL_APPROVAL_TITLE,
  withStatus,
} from '@/components/play/chat/approvals'
import { AWAITING_ANSWER } from '@/components/play/chat/caption'
import { PlayChat } from '@/components/play/chat/play-chat'
import { HELLO_KEY } from '@/lib/mascot/session'

// --- core fixtures: the ChatMessage shape of types/chat.ts, the payloads of result_formatter.py ----------

function userMessage(id: string, text: string): ChatMessage {
  return { id, role: 'user', content: text, parts: [{ type: 'text', text }] } as unknown as ChatMessage
}

function autoMessage(id: string, text: string): ChatMessage {
  return { id, role: 'assistant', content: text, parts: [{ type: 'text', text }] } as unknown as ChatMessage
}

const STORED_CHAT: Chat = { id: 'c1', title: 'Tidy the shelf', userId: 'u1', createdAt: '', updatedAt: '', visibility: 'private', turnInFlight: false }
const QUESTION = userMessage('m1', 'Delete the old brief?')

const TOOL_APPROVAL = {
  grant_id: 42,
  action: 'platform_delete_document',
  message: 'This action (destructive) requires confirmation.',
  permission_level: 'destructive',
  params: { document_id: 7 },
  risk_class: 'destructive',
  risk_tier: 'human_in_the_loop',
  oversight_rationale: 'Destructive: deletes data. A human must approve before it runs.',
  requires_approval: true,
}
const MISSION_APPROVAL = {
  mission_id: 'm-1',
  goal: 'Refund the customer and email them',
  state: 'awaiting_approval',
  task_count: 2,
  tasks: [
    { title: 'Issue refund', agent_role: 'ops', sequence: 1 },
    { title: 'Send email', agent_role: 'comms', sequence: 2 },
  ],
  risk_class: 'external_side_effect',
  risk_tier: 'human_in_the_loop',
  oversight_rationale: 'External side-effect: acts on a third-party system.',
  requires_approval: true,
}
const toolData = (data: Record<string, unknown>) => ({ type: 'tool-data', data })

async function renderTurn() {
  api.getChat.mockResolvedValue(STORED_CHAT)
  api.getChatMessages.mockResolvedValue([QUESTION, autoMessage('m2', '')])
  chat.status = 'streaming'
  const view = render(<PlayChat />)
  await screen.findByRole('list', { name: 'Conversation' })
  return view
}

/** The turn ends: the chat goes back to idle with Auto's words in the reply. */
function finishTurn(view: ReturnType<typeof render>) {
  chat.status = 'idle'
  chat.messages = [QUESTION, autoMessage('m2', 'That needs your OK first.')]
  view.rerender(<PlayChat />)
}

const auto = () => screen.getByTestId('auto')
const reply = (): HTMLElement => {
  const replies = screen.getAllByRole('article', { name: 'Auto said' })
  return replies[replies.length - 1]
}
const cards = () => within(reply()).getByRole('region', { name: APPROVALS_LABEL })
const data = (part: unknown) => act(() => chat.options?.onData(part))
const click = (button: HTMLElement) => act(async () => {
  fireEvent.click(button)
})

beforeEach(() => {
  window.sessionStorage.setItem(HELLO_KEY, '1') // hello already played this session
  Object.values(api).forEach((fn) => fn.mockReset())
  api.getChatHistory.mockResolvedValue([])
})

afterEach(() => {
  vi.clearAllMocks()
  window.sessionStorage.clear()
  Object.assign(chat, { options: null, status: 'idle', messages: null })
})

describe('an action that needs approval (tool_approval)', () => {
  it("renders the core's 'Action approval needed' card inside the reply: the action, its oversight and its params", async () => {
    await renderTurn()
    data(toolData({ tool_approval: TOOL_APPROVAL }))
    const card = cards()
    expect(within(card).getByText(TOOL_APPROVAL_TITLE)).toBeInTheDocument()
    expect(within(card).getAllByText('platform_delete_document').length).toBeGreaterThan(0)
    expect(within(card).getByText(/deletes data/)).toBeInTheDocument()
    expect(within(card).getByText('document_id')).toBeInTheDocument()
    expect(within(card).getByText('7')).toBeInTheDocument()
  })

  it("Approve & run calls Studio's handler (useGrantApproval with the grant id); Auto listens until then", async () => {
    grants.grant.mockResolvedValue({ grant: { id: 42, status: 'revoked', details: { executed_result: { success: true } } } })
    const view = await renderTurn()
    data(toolData({ tool_approval: TOOL_APPROVAL }))
    expect(auto()).toHaveAttribute('data-state', 'listening')
    finishTurn(view)
    expect(auto()).toHaveAttribute('data-state', 'listening')
    expect(auto()).toHaveTextContent(AWAITING_ANSWER)

    await click(within(cards()).getByRole('button', { name: /approve & run/i }))
    expect(grants.grant).toHaveBeenCalledWith(42)
    expect(grants.deny).not.toHaveBeenCalled()
    await waitFor(() => expect(within(cards()).getByRole('status')).toHaveTextContent(/the action ran/i))
    await waitFor(() => expect(auto()).toHaveAttribute('data-state', 'idle'))
  })

  it("Deny calls Studio's handler (useDenyApproval with the grant id), and Auto stops listening", async () => {
    grants.deny.mockResolvedValue({ grant: { id: 42, status: 'denied' } })
    const view = await renderTurn()
    data(toolData({ tool_approval: TOOL_APPROVAL }))
    finishTurn(view)
    await click(within(cards()).getByRole('button', { name: /deny/i }))
    expect(grants.deny).toHaveBeenCalledWith(42)
    expect(grants.grant).not.toHaveBeenCalled()
    await waitFor(() => expect(within(cards()).getByRole('status')).toHaveTextContent(/will not run/i))
    await waitFor(() => expect(auto()).toHaveAttribute('data-state', 'idle'))
  })

  it('a failed approve keeps the card waiting, and Auto listening', async () => {
    grants.grant.mockRejectedValue(new Error('403'))
    const view = await renderTurn()
    data(toolData({ tool_approval: TOOL_APPROVAL }))
    finishTurn(view)
    await click(within(cards()).getByRole('button', { name: /approve & run/i }))
    await waitFor(() => expect(toast.error).toHaveBeenCalledWith('Failed to approve the action'))
    expect(within(cards()).getByRole('button', { name: /approve & run/i })).toBeInTheDocument()
    expect(auto()).toHaveAttribute('data-state', 'listening')
  })

  it('closing the card with its X takes it away and ends the wait', async () => {
    const view = await renderTurn()
    data(toolData({ tool_approval: TOOL_APPROVAL }))
    finishTurn(view)
    await click(within(cards()).getByRole('button', { name: 'Close' }))
    expect(within(reply()).queryByRole('region', { name: APPROVALS_LABEL })).toBeNull()
    await waitFor(() => expect(auto()).toHaveAttribute('data-state', 'idle'))
  })
})

describe('a Mission plan that needs approval (mission_approval)', () => {
  it("renders the core's plan card: the goal, its tasks in order and the oversight", async () => {
    await renderTurn()
    data(toolData({ mission_approval: MISSION_APPROVAL }))
    const card = cards()
    expect(within(card).getByText(MISSION_APPROVAL_TITLE)).toBeInTheDocument()
    expect(within(card).getByText('Refund the customer and email them')).toBeInTheDocument()
    expect(within(card).getByText(/2 tasks/)).toBeInTheDocument()
    expect(within(card).getAllByRole('listitem').map((item) => item.textContent)).toEqual([
      expect.stringContaining('Issue refund'),
      expect.stringContaining('Send email'),
    ])
    expect(within(card).getByText(/acts on a third-party system/)).toBeInTheDocument()
  })

  it("Approve & run calls Studio's handler (useApproveMission); the card leaves the reply as it leaves Studio's canvas", async () => {
    missions.approve.mockResolvedValue({ mission: { id: 'm-1' } })
    const view = await renderTurn()
    data(toolData({ mission_approval: MISSION_APPROVAL }))
    finishTurn(view)
    expect(auto()).toHaveAttribute('data-state', 'listening')
    await click(within(cards()).getByRole('button', { name: /approve & run/i }))
    expect(missions.approve).toHaveBeenCalledWith({ id: 'm-1', body: {} })
    await waitFor(() => expect(within(reply()).queryByRole('region', { name: APPROVALS_LABEL })).toBeNull())
    await waitFor(() => expect(auto()).toHaveAttribute('data-state', 'idle'))
  })

  it("Reject calls Studio's handler (useRejectMission with the reason chat.tsx's card sends)", async () => {
    missions.reject.mockResolvedValue({ mission: { id: 'm-1' } })
    const view = await renderTurn()
    data(toolData({ mission_approval: MISSION_APPROVAL }))
    finishTurn(view)
    await click(within(cards()).getByRole('button', { name: /reject/i }))
    expect(missions.reject).toHaveBeenCalledWith({ id: 'm-1', body: { reason: 'Rejected from chat' } })
    expect(missions.approve).not.toHaveBeenCalled()
    await waitFor(() => expect(auto()).toHaveAttribute('data-state', 'idle'))
  })

  it('Auto listens until the last waiting card is settled', async () => {
    missions.reject.mockResolvedValue({ mission: { id: 'm-1' } })
    const view = await renderTurn()
    data(toolData({ mission_approval: MISSION_APPROVAL, tool_approval: TOOL_APPROVAL }))
    finishTurn(view)
    await click(within(cards()).getByRole('button', { name: /reject/i }))
    await waitFor(() => expect(within(cards()).queryByText(MISSION_APPROVAL_TITLE)).toBeNull())
    expect(auto()).toHaveAttribute('data-state', 'listening')
    await click(within(cards()).getByRole('button', { name: 'Close' }))
    await waitFor(() => expect(auto()).toHaveAttribute('data-state', 'idle'))
  })
})

describe('parts that are not approvals', () => {
  it('render nothing and leave Auto as he was', async () => {
    await renderTurn()
    expect(auto()).toHaveAttribute('data-state', 'thinking')
    data(toolData({ tool_approval: { grant_id: '../1', action: 'x' }, mission_approval: { mission_id: '../../admin', goal: 'x' } }))
    data({ type: 'tool-start', data: { tool_approval: TOOL_APPROVAL } })
    data(toolData({ documents: [{ id: 'd1' }] }))
    expect(within(reply()).queryByRole('region', { name: APPROVALS_LABEL })).toBeNull()
    expect(auto()).toHaveAttribute('data-state', 'thinking')
  })
})

describe('approvals.ts: the part read as chat.tsx reads it', () => {
  it("builds chat.tsx's widget data, with the oversight fields the backend sends, each checked", () => {
    const [mission, tool] = approvalsInPart(
      toolData({ mission_approval: { ...MISSION_APPROVAL, task_count: undefined, cost_estimate_usd: 1.5, risk_tier: 'odd' }, tool_approval: TOOL_APPROVAL }),
    )
    expect(mission).toMatchObject({ kind: 'mission', key: 'mission-m-1' })
    expect(mission.data).toMatchObject({ mission_id: 'm-1', task_count: 2, cost_estimate_usd: 1.5, risk_tier: undefined })
    expect(tool).toMatchObject({ kind: 'tool', key: 'tool-42', data: { grant_id: 42, params: { document_id: 7 }, risk_tier: 'human_in_the_loop' } })
    expect(approvalsInPart(toolData({ tool_approval: { ...TOOL_APPROVAL, grant_id: '17' } }))[0]).toMatchObject({ key: 'tool-17' })
  })

  it('files a card once, under the reply that raised it; a closed card stays closed', () => {
    const at = new Date(0)
    const raised = approvalsInPart(toolData({ tool_approval: TOOL_APPROVAL }))
    const filed = fileApprovals([], raised, 'm2', at)
    expect(filed).toEqual([{ ...raised[0], replyId: 'm2', status: 'waiting', filedAt: at }])
    expect(fileApprovals(filed, raised, 'm4', at)).toBe(filed)
    const closed = withStatus(filed, 'tool-42', 'closed')
    expect(closed[0].status).toBe('closed')
    expect(withStatus(closed, 'tool-42', 'answered')).toBe(closed)
    expect(withStatus(filed, 'tool-9', 'answered')).toBe(filed)
  })

  it('a card whose reply left the conversation is loose until it is closed (play-chat-approvals-stream.test.tsx: Stop, Try again)', () => {
    const filed = fileApprovals([], approvalsInPart(toolData({ tool_approval: TOOL_APPROVAL })), 'm2', new Date(0))
    expect(looseApprovals(filed, new Set(['m1', 'm2']))).toEqual([])
    expect(looseApprovals(filed, new Set(['m1']))).toEqual(filed)
    expect(looseApprovals(withStatus(filed, 'tool-42', 'answered'), new Set(['m1']))).toHaveLength(1)
    expect(looseApprovals(withStatus(filed, 'tool-42', 'closed'), new Set(['m1']))).toEqual([])
  })
})

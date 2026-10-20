/**
 * PRD-261 A-19e1 — the SDK's chat (lib/theme-sdk/chat.ts) on the hello theme's chat page
 * (support/hello-chat-page.tsx) with the host's own cards placed: ToolApproval and MissionApproval
 * (the host's widgets, whose Approve & run, Deny and Reject call the host's hooks, mocked here at the
 * hook as the widgets' own tests mock them) and MissionSuggestionCard, over the REAL chat controller,
 * approvals store, conversation loading and session store. The stream hook (useChatWithReceipts) is a
 * stand-in, as the PRD-260 chat tests stood it in; lib/chat-controller/__tests__/controller-cases.test.ts
 * runs the real one. The page never sends.
 *
 * It re-homes the core assertions of builtin Play's chat tests, which leave core with Play (W-A8; the
 * kit keeps their Play-rendering assertions, B-05c):
 * - components/play/chat/__tests__/play-chat-approvals.test.tsx: approve and deny via the host cards,
 *   the wait lasting until the last card settles, a card outliving the chat screen (P260-FX-08, M46);
 * - play-chat-turn.test.tsx: a stored reply's receipts in the host's words, a receipt linking a
 *   Deliverable by the core's absolute open_url, a failure the server saved, a background post, one card
 *   per ticket, and the Mission suggestion on the host's card.
 */
import { act, fireEvent, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ChatMessage, Receipt, TaskCardData } from '@/types'

const chat = vi.hoisted(() => ({
  options: null as null | Record<string, any>,
  status: 'idle' as 'idle' | 'streaming' | 'error',
  messages: null as null | unknown[],
}))
const api = vi.hoisted(() => ({ getChat: vi.fn(), getChatMessages: vi.fn(), getChatHistory: vi.fn(), putChatSession: vi.fn() }))
const toastError = vi.hoisted(() => vi.fn())
const grants = vi.hoisted(() => ({ grant: vi.fn(), deny: vi.fn() }))
const missions = vi.hoisted(() => ({ approve: vi.fn(), reject: vi.fn(), create: vi.fn() }))

vi.mock('next/navigation', () => ({
  usePathname: () => '/chat',
  useSearchParams: () => new URLSearchParams(''),
  useRouter: () => ({ replace: vi.fn(), push: vi.fn() }),
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
// The host cards' own hooks, as the widgets' own tests mock them: the cards call these and nothing else.
vi.mock('@/hooks/use-approval-grants', () => ({
  useGrantApproval: () => ({ isLoading: false, mutateAsync: grants.grant }),
  useDenyApproval: () => ({ isLoading: false, mutateAsync: grants.deny }),
}))
vi.mock('@/hooks/use-missions-api', () => ({
  useApproveMission: () => ({ isLoading: false, mutateAsync: missions.approve }),
  useRejectMission: () => ({ isLoading: false, mutateAsync: missions.reject }),
  useUpdateMissionPlan: () => ({ isLoading: false, mutateAsync: vi.fn() }),
  useCreateMission: () => ({ mutate: missions.create, isLoading: false }),
  useMission: () => ({ data: undefined, isLoading: true }),
}))
vi.mock('@/lib/chat/use-chat-with-receipts', () => ({
  useChatWithReceipts: (options: Record<string, any>) => {
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
}))

import { MISSION_APPROVAL_TITLE, TOOL_APPROVAL_TITLE } from '@/lib/chat/approvals'
import { useChatApprovalsStore } from '@/lib/chat-controller/approvals'
import { useMissionStore } from '@/stores/mission-store'

import { movedSomethingOnTheBoard, receiptLine, receiptLink } from '../chat'
import {
  cardOf,
  click,
  firstWords,
  forgetPage,
  HelloChatPage,
  message,
  NEW_CHAT,
  on,
  renderPage,
  seen,
  serve,
  setSession,
  shownChat,
  statusesOf,
  storedReply,
} from './support/hello-chat-page'

const QUESTION = message('q1', 'user', 'Delete the old brief?')
const NEEDS_OK = 'That needs your OK first.'
/** The server's copy of the reply that raised the card: its own id, not the one useChat gave it while streaming. */
const STORED_REPLY = message('m2-stored', 'assistant', NEEDS_OK)
const TOOL_APPROVAL = { grant_id: 42, action: 'platform_delete_document', params: { document_id: 7 }, risk_tier: 'human_in_the_loop' }
const MISSION_APPROVAL = { mission_id: 'm-1', goal: 'Refund the customer', task_count: 1, tasks: [{ title: 'Issue refund' }] }
const RAN = { grant: { id: 42, status: 'revoked', details: { executed_result: { success: true } } } }
const APPROVE = { name: /approve & run/i }
const CLOSE = { name: 'Close' }

type View = ReturnType<typeof renderPage>

/** The theme hands a stream part on, as useChat's onData does. */
const raise = (data: Record<string, unknown>) => act(() => chat.options?.onData({ type: 'tool-data', data }))

/** A stored conversation whose reply 'm2' is being written. */
async function renderTurn(): Promise<View> {
  setSession(on('c1'))
  serve('c1', 'Tidy the shelf', [QUESTION, message('m2', 'assistant', '')])
  chat.status = 'streaming'
  const view = renderPage()
  await shownChat('c1')
  await screen.findByText(firstWords(QUESTION))
  return view
}

/** The turn ends with Auto's words in the reply. */
function finishTurn(view: View) {
  Object.assign(chat, { status: 'idle', messages: [QUESTION, message('m2', 'assistant', NEEDS_OK)] })
  view.rerender(<HelloChatPage />)
}

/** A turn that raised these cards, ended. */
async function raisedAndEnded(data: Record<string, unknown>): Promise<View> {
  const view = await renderTurn()
  raise(data)
  finishTurn(view)
  return view
}

/** Leave the chat, then open `chatId` again: the page mounts afresh and loads the conversation from the server. */
async function openAgain(view: View, chatId: string, messages: ChatMessage[]): Promise<View> {
  view.unmount()
  forgetPage()
  Object.assign(chat, { status: 'idle', messages: null })
  setSession(on(chatId))
  serve(chatId, 'Tidy the shelf', messages)
  const again = renderPage()
  await shownChat(chatId)
  await screen.findByText(firstWords(messages[0]))
  return again
}

beforeEach(() => {
  Object.values(api).forEach((fn) => fn.mockReset())
  api.getChatHistory.mockResolvedValue([])
  setSession(NEW_CHAT)
  useChatApprovalsStore.setState({ filed: [] })
  useMissionStore.setState({ activePlanningMissionId: null })
})

afterEach(() => {
  vi.clearAllMocks()
  window.localStorage.clear()
  Object.assign(chat, { options: null, status: 'idle', messages: null })
  forgetPage()
})

describe('approve and deny via the host cards', () => {
  it('Approve & run calls the grant (useGrantApproval with its id); the card shows it ran and nothing waits any more', async () => {
    grants.grant.mockResolvedValue(RAN)
    await raisedAndEnded({ tool_approval: TOOL_APPROVAL })
    expect(seen.chat?.approvalWaiting).toBe(true)
    const card = await waitFor(() => cardOf(TOOL_APPROVAL_TITLE))
    await click(within(card).getByRole('button', APPROVE))
    expect(grants.grant).toHaveBeenCalledWith(42)
    expect(grants.deny).not.toHaveBeenCalled()
    await waitFor(() => expect(within(card).getByRole('status')).toHaveTextContent(/the action ran/i))
    expect(card).toHaveAttribute('data-status', 'answered')
    expect(seen.chat?.approvalWaiting).toBe(false)
  })

  it('a failed approve keeps the card waiting; Deny calls useDenyApproval with the grant id; the X takes the card away', async () => {
    grants.grant.mockRejectedValue(new Error('403'))
    grants.deny.mockResolvedValue({ grant: { id: 42, status: 'denied' } })
    await raisedAndEnded({ tool_approval: TOOL_APPROVAL })
    const card = await waitFor(() => cardOf(TOOL_APPROVAL_TITLE))
    await click(within(card).getByRole('button', APPROVE))
    await waitFor(() => expect(toastError).toHaveBeenCalledWith('Failed to approve the action'))
    expect(card).toHaveAttribute('data-status', 'waiting')
    expect(seen.chat?.approvalWaiting).toBe(true)
    await click(within(card).getByRole('button', { name: /deny/i }))
    expect(grants.deny).toHaveBeenCalledWith(42)
    await waitFor(() => expect(within(card).getByRole('status')).toHaveTextContent(/will not run/i))
    expect(card).toHaveAttribute('data-status', 'answered')
    await click(within(card).getByRole('button', CLOSE))
    expect(screen.queryByText(TOOL_APPROVAL_TITLE)).toBeNull()
  })

  it("the Mission card's Approve & run and Reject are the host's; the wait lasts until the last card settles", async () => {
    missions.approve.mockResolvedValue({ mission: { id: 'm-1' } })
    missions.reject.mockResolvedValue({ mission: { id: 'm-2' } })
    await raisedAndEnded({ mission_approval: MISSION_APPROVAL, tool_approval: TOOL_APPROVAL })
    await click(within(await waitFor(() => cardOf(MISSION_APPROVAL_TITLE))).getByRole('button', APPROVE))
    expect(missions.approve).toHaveBeenCalledWith({ id: 'm-1', body: {} })
    // The Mission card closes itself after a decision, as it leaves Studio's canvas.
    await waitFor(() => expect(screen.queryByText(MISSION_APPROVAL_TITLE)).toBeNull())
    expect(seen.chat?.approvalWaiting).toBe(true)
    await click(within(cardOf(TOOL_APPROVAL_TITLE)).getByRole('button', CLOSE))
    await waitFor(() => expect(seen.chat?.approvalWaiting).toBe(false))

    raise({ mission_approval: { ...MISSION_APPROVAL, mission_id: 'm-2' } })
    await click(within(await waitFor(() => cardOf(MISSION_APPROVAL_TITLE))).getByRole('button', { name: /reject/i }))
    expect(missions.reject).toHaveBeenCalledWith({ id: 'm-2', body: { reason: 'Rejected from chat' } })
    expect(missions.approve).toHaveBeenCalledTimes(1)
    await waitFor(() => expect(seen.chat?.approvalWaiting).toBe(false))
  })
})

describe('a card outlives the chat screen (P260-FX-08, M46)', () => {
  it('opened again, the conversation shows the waiting card after it, and Approve & run still calls grant(42)', async () => {
    grants.grant.mockResolvedValue(RAN)
    const view = await raisedAndEnded({ tool_approval: TOOL_APPROVAL })
    await openAgain(view, 'c1', [QUESTION, STORED_REPLY])
    expect(statusesOf(seen.chat?.looseApprovals)).toEqual(['tool-42:waiting'])
    expect(seen.chat?.approvalWaiting).toBe(true)
    await click(within(cardOf(TOOL_APPROVAL_TITLE)).getByRole('button', APPROVE))
    expect(grants.grant).toHaveBeenCalledWith(42)
    await waitFor(() => expect(statusesOf(seen.chat?.looseApprovals)).toEqual(['tool-42:answered']))
    expect(seen.chat?.approvalWaiting).toBe(false)
  })

  it('a reply back with the same id holds its card again; another conversation never shows it; a closed card stays closed', async () => {
    const view = await raisedAndEnded({ tool_approval: TOOL_APPROVAL })
    const back = await openAgain(view, 'c1', [QUESTION, message('m2', 'assistant', NEEDS_OK)])
    expect(statusesOf(seen.chat?.messages[1].approvals)).toEqual(['tool-42:waiting'])
    expect(seen.chat?.looseApprovals).toEqual([])
    const other = await openAgain(back, 'c2', [message('m5', 'user', 'Something else')])
    expect(screen.queryByText(TOOL_APPROVAL_TITLE)).toBeNull()
    expect(seen.chat?.approvalWaiting).toBe(false)
    const again = await openAgain(other, 'c1', [QUESTION, STORED_REPLY])
    await click(within(cardOf(TOOL_APPROVAL_TITLE)).getByRole('button', CLOSE))
    await openAgain(again, 'c1', [QUESTION, STORED_REPLY])
    expect(screen.queryByText(TOOL_APPROVAL_TITLE)).toBeNull()
  })

  it("a new chat's card goes with the name the backend gives it: that conversation shows it again, a new chat doesn't", async () => {
    Object.assign(chat, { status: 'streaming', messages: [QUESTION, message('m2', 'assistant', '')] })
    const view = renderPage()
    await shownChat(null)
    await screen.findByText(firstWords(QUESTION))
    raise({ tool_approval: TOOL_APPROVAL })
    act(() => chat.options?.onChatIdUpdate('c9'))
    await waitFor(() => expect(seen.chat?.messages[1].approvals.map((card) => card.chatId)).toEqual(['c9']))
    const named = await openAgain(view, 'c9', [QUESTION, STORED_REPLY])
    expect(statusesOf(seen.chat?.looseApprovals)).toEqual(['tool-42:waiting'])
    named.unmount()
    setSession(NEW_CHAT)
    Object.assign(chat, { status: 'streaming', messages: [QUESTION, message('m7', 'assistant', '')] })
    renderPage()
    await shownChat(null)
    expect(screen.queryByText(TOOL_APPROVAL_TITLE)).toBeNull()
  })
})

function receipt(fields: Partial<Receipt>): Receipt {
  return { action: 'platform_update_task', kind: 'write', status: 'done', subject: '', effect: '', link: null, reason: null, ...fields }
}

const ticket = (fields: Partial<TaskCardData>): TaskCardData =>
  ({ id: 92, number: '#0042', title: 'Write the landing page', status: 'in_progress', assigned_agent: 'Bob', ...fields }) as TaskCardData

/** A stored conversation with these messages on screen, and its chat. */
async function renderStored(messages: ChatMessage[]) {
  setSession(on('c1'))
  serve('c1', 'Laptop for uni', messages)
  renderPage()
  const shown = await shownChat('c1')
  await screen.findByText(firstWords(messages[0]))
  return shown
}

describe('what a stored reply carries', () => {
  it("its receipts in the host's words, linked only when done and in this app; an empty list is a turn that ran nothing", async () => {
    const done = receipt({ subject: '#0422', effect: 'moved to Done', link: '/activity?ticket=422' })
    const refused = receipt({ action: 'platform_send_email', status: 'refused', subject: 'Letter to Maya', reason: 'no mail connection' })
    const read = receipt({ kind: 'read', subject: 'Brief', effect: 'looked up' })
    const shown = await renderStored([
      QUESTION,
      message('m2', 'assistant', 'Done.', { parts: [{ type: 'text', text: 'Done.' }, { type: 'receipts', receipts: [done, refused] }] }),
      message('m3', 'assistant', 'Hi.', { receipts: [] }),
      message('m4', 'assistant', 'Read it.', { receipts: [read] }),
    ])
    const [asked, both, none, readOnly] = shown.messages
    expect(both.receipts?.map(receiptLine)).toEqual(['#0422: moved to Done', 'Letter to Maya: tried, refused: no mail connection'])
    expect(both.receipts?.map(receiptLink)).toEqual(['/activity?ticket=422', null])
    expect(movedSomethingOnTheBoard(both.receipts ?? [])).toBe(true)
    expect(none.receipts).toEqual([])
    expect(movedSomethingOnTheBoard(readOnly.receipts ?? [])).toBe(false)
    expect(asked.receipts).toBeNull()
  })

  it("a done receipt linking a Deliverable by the core's absolute open_url lists it by its checked id, never the receipt's own URL", async () => {
    const link = 'https://app.example/deliverables?tab=outputs&deliverable=d-7'
    const made = receipt({ action: 'generate_document', subject: 'Budget memo', effect: 'made', link })
    const shown = await renderStored([QUESTION, message('m2', 'assistant', 'Made it.', { receipts: [made] })])
    expect(shown.messages[1].linkedDeliverables).toEqual([{ id: 'd-7', title: 'Budget memo', kind: 'document' }])
    expect(receiptLink(made)).toBeNull()
  })

  it('a failure the server saved is its words without the ⚠️, never an answer; a background post names who wrote it; one card per ticket', async () => {
    const shown = await renderStored([
      QUESTION,
      storedReply('m2', '⚠️ The model is unavailable', { origin: 'turn_error', label: 'Error' }),
      storedReply('m3', 'Watched it. 8.4/10, passed.', { origin: 'watcher', label: 'Auto · background' }),
      message('m4', 'assistant', 'Filed it.', {
        parts: [{ type: 'text', text: 'Filed it.' }, { type: 'task_card', card: ticket({ status: 'assigned' }) }],
        taskCards: [ticket({})],
      }),
    ])
    const [asked, saved, posted, filed] = shown.messages
    expect(saved).toMatchObject({ savedError: 'The model is unavailable', provenance: null })
    expect(posted).toMatchObject({ savedError: null, provenance: 'Auto · background', text: 'Watched it. 8.4/10, passed.' })
    expect(filed.taskCards.map((card) => `${card.number}:${card.status}`)).toEqual(['#0042:in_progress'])
    expect(asked.taskCards).toEqual([])
  })

  it("the turn's Mission suggestion on the host's card: Launch sends the goal, the conversation and its recent words", async () => {
    await renderStored([message('q1', 'user', 'Shortlist three laptops?'), message('m2', 'assistant', 'That is a big one.')])
    const goal = 'Research ten competitors and compare pricing'
    act(() => chat.options?.onData({ type: 'mission-suggestion', data: { goal, complexity: 'organ', agent_id: 3 } }))
    expect(await screen.findByText(goal)).toBeInTheDocument()
    expect(screen.getByText('Multi-step')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /Launch Mission/ }))
    const context = [{ role: 'user', content: 'Shortlist three laptops?' }, { role: 'assistant', content: 'That is a big one.' }]
    expect(missions.create).toHaveBeenCalledWith({ goal, config: { source: 'chat', chat_id: 'c1', context_messages: context } }, expect.anything())
    act(() => useMissionStore.setState({ activePlanningMissionId: 'mission-1' }))
    expect(seen.chat?.planningMissionId).toBe('mission-1')
  })
})

/**
 * PRD-268 US-C20c (§6.11.4, K10) — the approval view on the host's two approval cards.
 *
 * A card that carries `approval_view` shows who asked, the action, its target, the exact preview
 * and the rule that asked, with the rule before the buttons and named by the card's
 * aria-describedby. A card without one is exactly today's. While the card, or a control inside
 * it, has focus, key 1 refuses (Deny, Reject) and 3 allows once (Approve & run); 2 does nothing,
 * and nothing answers without focus. The buttons keep today's words and Always allow is nowhere.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

const calls = vi.hoisted(() => ({
  grant: vi.fn(),
  deny: vi.fn(),
  approve: vi.fn(),
  reject: vi.fn(),
  updatePlan: vi.fn(),
}))
const toast = vi.hoisted(() => ({ success: vi.fn(), error: vi.fn(), info: vi.fn() }))
vi.mock('sonner', () => ({ toast }))
vi.mock('@/hooks/use-approval-grants', () => ({
  useGrantApproval: () => ({ isLoading: false, mutateAsync: calls.grant }),
  useDenyApproval: () => ({ isLoading: false, mutateAsync: calls.deny }),
}))
vi.mock('@/hooks/use-missions-api', () => ({
  useApproveMission: () => ({ isLoading: false, mutateAsync: calls.approve }),
  useRejectMission: () => ({ isLoading: false, mutateAsync: calls.reject }),
  useUpdateMissionPlan: () => ({ isLoading: false, mutateAsync: calls.updatePlan }),
}))

import { MissionApprovalWidget } from '../MissionApprovalWidget'
import { ToolApprovalWidget } from '../ToolApprovalWidget'
import { ASKED_BY_LABEL } from '../approval-view-parts'
import type { MissionApprovalWidgetData, ToolApprovalWidgetData, WidgetMetadata } from '../types'
import type { ApprovalView } from '@/lib/proactive/types'

const metadata: WidgetMetadata = { source: { type: 'tool', name: 'platform_send_email' }, createdAt: new Date() }
const TOOL_TITLE = 'Action approval needed'
const MISSION_TITLE = 'Mission plan — approval needed'
const SEND_RULE = 'Ask before anything that sends to other people or acts in another app'
const PLAN_RULE = 'Ask before any Mission or task runs'
const PREVIEW = 'to: aisha@example.com\nsubject: March invoice'
const TODAYS_BODY_CLASS = 'flex flex-col gap-3 p-3'
const ALWAYS_ALLOW = /always allow/i
/** The card's body, named by its title and the action. */
const TOOL_CARD = `${TOOL_TITLE}: Send an email`
const MISSION_CARD = `${MISSION_TITLE}: Refund the customer and email them`

const TOOL_VIEW: ApprovalView = {
  grant_id: '42',
  asked_by: { kind: 'auto', name: 'Auto' },
  action: { verb: 'send', summary: 'Send an email' },
  target: { label: 'aisha@example.com', detail: 'to' },
  preview: { kind: 'text', body: PREVIEW },
  rule: { id: 'confirm.external_side_effect', sentence: SEND_RULE },
  answers: ['refuse', 'allow_once'],
}
const MISSION_VIEW: ApprovalView = {
  grant_id: 'm-1',
  asked_by: { kind: 'auto', name: 'Auto' },
  action: { verb: 'run', summary: 'Refund the customer and email them' },
  target: { label: 'M-12' },
  preview: { kind: 'text', body: '1. Issue refund\n2. Send email' },
  rule: { id: 'approval_policy.always_ask', sentence: PLAN_RULE },
  answers: ['refuse', 'allow_once'],
}

function toolData(view?: ApprovalView): ToolApprovalWidgetData {
  return {
    grant_id: 42,
    action: 'platform_send_email',
    message: 'This action (external_side_effect) requires confirmation.',
    params: { to: 'aisha@example.com', subject: 'March invoice' },
    ...(view ? { approval_view: view } : {}),
  }
}

function missionData(view?: ApprovalView): MissionApprovalWidgetData {
  return {
    mission_id: 'm-1',
    goal: 'Refund the customer and email them',
    task_count: 2,
    tasks: [
      { title: 'Issue refund', agent_role: 'ops', sequence: 1 },
      { title: 'Send email', agent_role: 'comms', sequence: 2 },
    ],
    ...(view ? { approval_view: view } : {}),
  }
}

function renderTool(view?: ApprovalView) {
  return render(<ToolApprovalWidget id="w-tool" title={TOOL_TITLE} data={toolData(view)} metadata={metadata} />)
}

function renderMission(view?: ApprovalView) {
  return render(<MissionApprovalWidget id="w-mission" title={MISSION_TITLE} data={missionData(view)} metadata={metadata} />)
}

/** `earlier` comes before `later` in the document. */
function expectBefore(earlier: Element, later: Element): void {
  expect(earlier.compareDocumentPosition(later) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
}

beforeEach(() => {
  for (const call of Object.values(calls)) call.mockReset().mockResolvedValue({})
  for (const shown of Object.values(toast)) shown.mockReset()
})

describe('the approval view on the tool card', () => {
  it('shows who asked, the action, the target, the exact preview and the rule, the rule read before the buttons', () => {
    const { container } = renderTool(TOOL_VIEW)
    const card = screen.getByRole('group', { name: TOOL_CARD })
    expect(within(card).getByText(`${ASKED_BY_LABEL} Auto · send`)).toBeInTheDocument()
    expect(within(card).getByText('Send an email')).toBeInTheDocument()
    expect(within(card).getByText(/→ aisha@example\.com · to/)).toBeInTheDocument()
    const preview = container.querySelector('pre')
    expect(preview?.textContent).toBe(PREVIEW)
    const rule = within(card).getByText(SEND_RULE)
    expect(card.getAttribute('aria-describedby')).toBe(rule.id)
    expect(card).toHaveAccessibleDescription(SEND_RULE)
    expectBefore(rule, screen.getByRole('button', { name: /Approve & run/ }))
    expectBefore(rule, screen.getByRole('button', { name: /Deny/ }))
  })

  it('renders the preview as text, never as markup', () => {
    const body = '<img src="x" onerror="alert(1)"> **bold**'
    const { container } = renderTool({ ...TOOL_VIEW, preview: { kind: 'markdown', body } })
    expect(container.querySelector('pre')?.textContent).toBe(body)
    expect(container.querySelector('img')).toBeNull()
    expect(container.querySelector('strong')).toBeNull()
  })

  it("is today's card without a view: no parts, no group, no focus stop", () => {
    const { container } = renderTool()
    expect(screen.queryByRole('group')).toBeNull()
    expect(screen.queryByText(new RegExp(ASKED_BY_LABEL))).toBeNull()
    expect(container.querySelector('pre')).toBeNull()
    expect(container.querySelector('[tabindex="0"]')).toBeNull()
    expect(container.querySelector(`[class="${TODAYS_BODY_CLASS}"]`)).not.toBeNull()
    expect(screen.getByText('to')).toBeInTheDocument()
  })

  it("drops a view whose grant_id is another card's, and shows today's card", () => {
    renderTool({ ...TOOL_VIEW, grant_id: '43' })
    expect(screen.queryByRole('group')).toBeNull()
    expect(screen.queryByText(SEND_RULE)).toBeNull()
    expect(screen.getByRole('button', { name: /Approve & run/ })).toBeInTheDocument()
  })

  it('answers 1 with Deny and 3 with Approve & run while the card has focus; 2 does nothing', async () => {
    const user = userEvent.setup()
    renderTool(TOOL_VIEW)
    screen.getByRole('group', { name: TOOL_CARD }).focus()
    await user.keyboard('2')
    expect(calls.grant).not.toHaveBeenCalled()
    expect(calls.deny).not.toHaveBeenCalled()
    await user.keyboard('3')
    expect(calls.grant).toHaveBeenCalledWith(42)
    expect(calls.deny).not.toHaveBeenCalled()
  })

  it('answers 1 with Deny while a button inside the card has focus', async () => {
    const user = userEvent.setup()
    renderTool(TOOL_VIEW)
    screen.getByRole('button', { name: /Approve & run/ }).focus()
    await user.keyboard('1')
    expect(calls.deny).toHaveBeenCalledWith(42)
    expect(calls.grant).not.toHaveBeenCalled()
  })

  it('answers nothing without focus, and nothing with a modifier held', async () => {
    const user = userEvent.setup()
    renderTool(TOOL_VIEW)
    await user.keyboard('1')
    await user.keyboard('3')
    screen.getByRole('group', { name: TOOL_CARD }).focus()
    await user.keyboard('{Control>}3{/Control}')
    await user.keyboard('{Meta>}1{/Meta}')
    expect(calls.grant).not.toHaveBeenCalled()
    expect(calls.deny).not.toHaveBeenCalled()
  })

  it('answers no key the view does not offer: a view with no answers left takes neither 1 nor 3', async () => {
    const user = userEvent.setup()
    renderTool({ ...TOOL_VIEW, answers: [] })
    screen.getByRole('group', { name: TOOL_CARD }).focus()
    await user.keyboard('1')
    await user.keyboard('3')
    expect(calls.grant).not.toHaveBeenCalled()
    expect(calls.deny).not.toHaveBeenCalled()
    expect(screen.getByRole('button', { name: 'Approve & run' })).not.toHaveAttribute('aria-keyshortcuts')
  })

  it('announces each key on its button with a view, and nothing without one', () => {
    const first = renderTool(TOOL_VIEW)
    expect(screen.getByRole('button', { name: 'Approve & run' })).toHaveAttribute('aria-keyshortcuts', '3')
    expect(screen.getByRole('button', { name: 'Deny' })).toHaveAttribute('aria-keyshortcuts', '1')
    first.unmount()
    renderTool()
    expect(screen.getByRole('button', { name: 'Approve & run' })).not.toHaveAttribute('aria-keyshortcuts')
    expect(screen.getByRole('button', { name: 'Deny' })).not.toHaveAttribute('aria-keyshortcuts')
  })

  it('keeps the words: Approve & run and Deny, and no Always allow anywhere', () => {
    renderTool({ ...TOOL_VIEW, answers: ['refuse', 'allow_once'] })
    expect(screen.getByRole('button', { name: 'Approve & run' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Deny' })).toBeInTheDocument()
    expect(screen.getAllByRole('button').filter((button) => ALWAYS_ALLOW.test(button.textContent ?? ''))).toEqual([])
    expect(screen.queryByText(ALWAYS_ALLOW)).toBeNull()
  })
})

describe('the approval view on the Mission card', () => {
  it('shows who asked, the goal, its number, the steps and the policy that held it, before the buttons', () => {
    renderMission(MISSION_VIEW)
    const card = screen.getByRole('group', { name: MISSION_CARD })
    expect(within(card).getByText(`${ASKED_BY_LABEL} Auto · run`)).toBeInTheDocument()
    expect(within(card).getByText(/→ M-12/)).toBeInTheDocument()
    expect(card.querySelector('pre')?.textContent).toBe('1. Issue refund\n2. Send email')
    const rule = within(card).getByText(PLAN_RULE)
    expect(card).toHaveAccessibleDescription(PLAN_RULE)
    expectBefore(rule, screen.getByRole('button', { name: /Approve & run/ }))
    expectBefore(rule, screen.getByRole('button', { name: /Reject/ }))
  })

  it("is today's card without a view", () => {
    const { container } = renderMission()
    expect(screen.queryByRole('group')).toBeNull()
    expect(container.querySelector('pre')).toBeNull()
    expect(container.querySelector('[tabindex="0"]')).toBeNull()
    expect(container.querySelector(`[class="${TODAYS_BODY_CLASS}"]`)).not.toBeNull()
    expect(screen.getByRole('textbox', { name: 'Agent for task 1' })).toHaveValue('ops')
  })

  it('answers 1 with Reject and 3 with Approve & run while the card has focus', async () => {
    const user = userEvent.setup()
    const first = renderMission(MISSION_VIEW)
    screen.getByRole('group', { name: MISSION_CARD }).focus()
    await user.keyboard('1')
    expect(calls.reject).toHaveBeenCalledWith({ id: 'm-1', body: { reason: 'Rejected from chat' } })
    expect(calls.approve).not.toHaveBeenCalled()
    first.unmount()

    renderMission(MISSION_VIEW)
    screen.getByRole('group', { name: MISSION_CARD }).focus()
    await user.keyboard('3')
    expect(calls.approve).toHaveBeenCalledWith({ id: 'm-1', body: {} })
  })

  it("never answers a key typed into a task's agent box", async () => {
    const user = userEvent.setup()
    renderMission(MISSION_VIEW)
    await user.click(screen.getByRole('textbox', { name: 'Agent for task 1' }))
    await user.keyboard('13')
    expect(calls.reject).not.toHaveBeenCalled()
    expect(calls.approve).not.toHaveBeenCalled()
    expect(screen.getByRole('textbox', { name: 'Agent for task 1' })).toHaveValue('ops13')
  })

  it('answers nothing without focus, and keeps the words Approve & run and Reject', async () => {
    const user = userEvent.setup()
    renderMission(MISSION_VIEW)
    await user.keyboard('3')
    await user.keyboard('1')
    expect(calls.approve).not.toHaveBeenCalled()
    expect(calls.reject).not.toHaveBeenCalled()
    expect(screen.getByRole('button', { name: 'Approve & run' })).toHaveAttribute('aria-keyshortcuts', '3')
    expect(screen.getByRole('button', { name: 'Reject' })).toHaveAttribute('aria-keyshortcuts', '1')
    expect(screen.queryByText(ALWAYS_ALLOW)).toBeNull()
  })
})

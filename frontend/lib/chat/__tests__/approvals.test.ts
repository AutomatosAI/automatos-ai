/**
 * PRD-260 T-005c, PRD-261 A-19e1 — lib/chat/approvals.ts, the approval cards read off a stream part as
 * chat.tsx reads them (its 'Action approval needed' and 'Mission plan — approval needed' widgets'
 * data, each field checked because it comes off the wire), filed under the reply that raised them,
 * and loose once that reply leaves the conversation. Pure. PRD-268 US-C20c: each card's approval_view
 * (ApprovalView@1) is checked the same way, and a view that isn't what it says never costs the card.
 *
 * Re-homed from the approvals.ts cases of components/play/chat/__tests__/play-chat-approvals.test.tsx,
 * which leaves core with builtin Play (W-A8): what they test is the host's, so core keeps it.
 */
import { describe, expect, it } from 'vitest'

import { approvalsInPart, approvalViewOf, fileApprovals, looseApprovals, withStatus } from '@/lib/chat/approvals'

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

describe('approvals.ts: the part read as chat.tsx reads it', () => {
  it("builds chat.tsx's widget data from one part, the Mission first, with the oversight fields the backend sends, each checked", () => {
    const [mission, tool] = approvalsInPart(
      toolData({ mission_approval: { ...MISSION_APPROVAL, task_count: undefined, cost_estimate_usd: 1.5, risk_tier: 'odd' }, tool_approval: TOOL_APPROVAL }),
    )
    expect(mission).toMatchObject({ kind: 'mission', key: 'mission-m-1' })
    expect(mission.data).toMatchObject({ mission_id: 'm-1', task_count: 2, cost_estimate_usd: 1.5, risk_tier: undefined })
    expect(tool).toMatchObject({ kind: 'tool', key: 'tool-42', data: { grant_id: 42, params: { document_id: 7 }, risk_tier: 'human_in_the_loop' } })
    expect(approvalsInPart(toolData({ tool_approval: { ...TOOL_APPROVAL, grant_id: '17' } }))[0]).toMatchObject({ key: 'tool-17' })
  })

  it('raises nothing for a part that is not what it says: bad ids, another part type, no data', () => {
    expect(approvalsInPart(toolData({ tool_approval: { grant_id: '../1', action: 'x' }, mission_approval: { mission_id: '../../admin', goal: 'x' } }))).toEqual([])
    expect(approvalsInPart({ type: 'tool-start', data: { tool_approval: TOOL_APPROVAL } })).toEqual([])
    expect(approvalsInPart(toolData({ documents: [{ id: 'd1' }] }))).toEqual([])
    expect(approvalsInPart({ type: 'tool-data' })).toEqual([])
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

  it('a card whose reply left the conversation is loose until it is closed, answered or not', () => {
    const filed = fileApprovals([], approvalsInPart(toolData({ tool_approval: TOOL_APPROVAL })), 'm2', new Date(0))
    expect(looseApprovals(filed, new Set(['m1', 'm2']))).toEqual([])
    expect(looseApprovals(filed, new Set(['m1']))).toEqual(filed)
    expect(looseApprovals(withStatus(filed, 'tool-42', 'answered'), new Set(['m1']))).toHaveLength(1)
    expect(looseApprovals(withStatus(filed, 'tool-42', 'closed'), new Set(['m1']))).toEqual([])
  })
})

/** PRD-268 US-C20c: ApprovalView@1 as the server sends it (approval_view.py), on the tool card. */
const VIEW = {
  grant_id: '42',
  asked_by: { kind: 'auto', name: 'Auto' },
  action: { verb: 'delete', summary: 'Delete the document' },
  target: { label: '7', detail: 'document_id' },
  preview: { kind: 'text', body: 'document_id: 7' },
  rule: { id: 'confirm.destructive', sentence: "Ask before anything that deletes or can't be undone" },
  answers: ['refuse', 'allow_once'],
  expires_at: '2026-10-29T12:00:00Z',
}
const toolWith = (view: unknown) => approvalsInPart(toolData({ tool_approval: { ...TOOL_APPROVAL, approval_view: view } }))

describe('approvals.ts: the approval view checked off the wire (PRD-268 US-C20c)', () => {
  it('keeps a view that is what it says, on the tool card and on the Mission card', () => {
    expect(toolWith(VIEW)[0].data).toMatchObject({ grant_id: 42, approval_view: VIEW })
    const planView = { ...VIEW, grant_id: 'm-1', target: { label: 'M-12' } }
    const [mission] = approvalsInPart(toolData({ mission_approval: { ...MISSION_APPROVAL, approval_view: planView } }))
    expect(mission.data).toMatchObject({ mission_id: 'm-1', approval_view: { grant_id: 'm-1', target: { label: 'M-12' } } })
  })

  it('leaves a card without a view exactly as before: no approval_view key at all', () => {
    const [tool] = approvalsInPart(toolData({ tool_approval: TOOL_APPROVAL }))
    expect('approval_view' in tool.data).toBe(false)
  })

  it("drops the view, never the card, when its grant_id isn't an id or isn't this card's", () => {
    for (const grantId of ['../42', '42; drop', '', '43', 42.5, null]) {
      const [card] = toolWith({ ...VIEW, grant_id: grantId })
      expect(card).toMatchObject({ kind: 'tool', key: 'tool-42' })
      expect('approval_view' in card.data).toBe(false)
    }
    const [mission] = approvalsInPart(toolData({ mission_approval: { ...MISSION_APPROVAL, approval_view: { ...VIEW, grant_id: 'm-2' } } }))
    expect('approval_view' in mission.data).toBe(false)
  })

  it('drops the view, never the card, when who asked, the action or the rule is not what it says', () => {
    const broken = [
      'a view',
      [VIEW],
      { ...VIEW, asked_by: { kind: 'someone', name: 'Auto' } },
      { ...VIEW, asked_by: { kind: 'auto' } },
      { ...VIEW, action: { verb: 'delete', summary: 7 } },
      { ...VIEW, action: undefined },
      { ...VIEW, rule: { id: 'confirm.destructive' } },
      { ...VIEW, rule: 'Ask before anything' },
    ]
    for (const view of broken) {
      const [card] = toolWith(view)
      expect(card.key).toBe('tool-42')
      expect('approval_view' in card.data).toBe(false)
    }
  })

  it('leaves out a target or a preview that is not what it says, and keeps the rest of the view', () => {
    const [card] = toolWith({ ...VIEW, target: { label: 7 }, preview: { kind: 'html', body: '<b>x</b>' }, expires_at: 9 })
    const view = card.kind === 'tool' ? card.data.approval_view : undefined
    expect(view).toMatchObject({ rule: VIEW.rule, action: VIEW.action })
    expect(view && 'target' in view).toBe(false)
    expect(view && 'preview' in view).toBe(false)
    expect(view && 'expires_at' in view).toBe(false)
  })

  it('keeps only the answers a person may give: never Always allow (K10)', () => {
    expect(approvalViewOf({ ...VIEW, answers: ['refuse', 'always_allow', 'allow_once', 3] }, '42')?.answers).toEqual(['refuse', 'allow_once'])
    expect(approvalViewOf({ ...VIEW, answers: 'allow_once' }, '42')?.answers).toEqual([])
  })

  it('reads a grant_id sent as a number as its string', () => {
    expect(approvalViewOf({ ...VIEW, grant_id: 42 }, '42')?.grant_id).toBe('42')
  })
})

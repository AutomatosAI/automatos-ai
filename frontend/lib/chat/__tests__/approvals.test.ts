/**
 * PRD-260 T-005c, PRD-261 A-19e1 — lib/chat/approvals.ts, the approval cards read off a stream part as
 * chat.tsx reads them (its 'Action approval needed' and 'Mission plan — approval needed' widgets'
 * data, each field checked because it comes off the wire), filed under the reply that raised them,
 * and loose once that reply leaves the conversation. Pure.
 *
 * Re-homed from the approvals.ts cases of components/play/chat/__tests__/play-chat-approvals.test.tsx,
 * which leaves core with builtin Play (W-A8): what they test is the host's, so core keeps it.
 */
import { describe, expect, it } from 'vitest'

import { approvalsInPart, fileApprovals, looseApprovals, withStatus } from '@/lib/chat/approvals'

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

/**
 * F162 (c) — the approval card re-staffs the task it shows, not its step.
 */
import { describe, expect, it } from 'vitest'
import { roleEdit, withPlanPicks } from '../MissionApprovalWidget/role-edit'

describe('roleEdit', () => {
  it('names a side-by-side task by its temp_id', () => {
    const task = { title: 'Draft Club Member Email for Christmas Box', sequence: 1, temp_id: 'task_2' }
    expect(roleEdit(task, 1, 'CLUB SECRETARY')).toEqual({ temp_id: 'task_2', agent_role: 'CLUB SECRETARY' })
  })

  it('falls back to the step number on a card without temp_ids', () => {
    expect(roleEdit({ title: 'Draft the page', sequence: 3 }, 3, 'OPS')).toEqual({ sequence_number: 3, agent_role: 'OPS' })
  })
})

describe('withPlanPicks', () => {
  const card = [
    { title: 'Split the box for the club', sequence: 1, temp_id: 'task_1', agent_role: 'writer', match_agent: 'TRACKER' },
    { title: 'Draft Club Member Email', sequence: 1, temp_id: 'task_2', agent_role: 'writer', match_agent: 'NEWSROOM' },
    { title: 'An older card task', sequence: 2, agent_role: 'writer', match_agent: 'WRITER' },
  ]

  it('shows the edited task’s new pick and leaves its siblings as they were', () => {
    const plan = { tasks: [
      { temp_id: 'task_1', agent_role: 'writer', match_agent: 'TRACKER' },
      { temp_id: 'task_2', agent_role: 'CLUB SECRETARY', match_agent: 'CLUB SECRETARY', match_is_override: true },
    ] }
    const [first, second, third] = withPlanPicks(card, plan)
    expect(second).toMatchObject({ agent_role: 'CLUB SECRETARY', match_agent: 'CLUB SECRETARY', match_is_override: true })
    expect(first).toMatchObject({ match_agent: 'TRACKER' })
    expect(third).toBe(card[2])
  })

  it('keeps the card as it was when the answer has no plan', () => {
    expect(withPlanPicks(card, null)).toEqual(card)
  })
})

/** PRD-244 review — an activity row opens the thing itself: the ticket on the board, the report a routine produced. */
import { describe, it, expect } from 'vitest'
import { rowHref } from '@/components/command-center/activity-tab'

const base = { id: 'x', name: '', status: 'completed', started_at: null, completed_at: null, duration_seconds: null, agent: null, agents: [], summary: '', source_id: null, source_url: null, trigger: null, error_message: null } as const

describe('rowHref', () => {
  it('a task opens the ticket on the board (the board reads ?task_id=)', () => {
    expect(rowHref({ ...base, type: 'task', source_id: '121' } as any)).toBe('/command-center?tab=board&task_id=121')
  })
  it('a routine opens the report it produced when there is one', () => {
    expect(rowHref({ ...base, type: 'routine', source_url: '/deliverables/explorer?path=reports%2Fops.md', agent: { id: 9, name: 'OPS' } } as any)).toBe('/deliverables/explorer?path=reports%2Fops.md')
  })
  it('a routine without a report opens the agent on its Reports panel', () => {
    expect(rowHref({ ...base, type: 'routine', source_url: '/agents/9', agent: { id: 9, name: 'OPS' } } as any)).toBe('/agents?agent=9&panel=reports')
  })
  it('an orchestrator routine with no report has nowhere to go', () => {
    expect(rowHref({ ...base, type: 'routine' } as any)).toBeNull()
  })
})

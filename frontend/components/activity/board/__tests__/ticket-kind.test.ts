/**
 * PRD-252 R4 — tickets you can tell apart: number, type and title on every
 * surface. Every card said TASK; two tickets with one title looked the same.
 */
import { describe, it, expect } from 'vitest'
import { readFileSync } from 'fs'
import path from 'path'
import { numberedTitle, runsInSession, ticketKind } from '../ticket-kind'

const read = (rel: string) => readFileSync(path.resolve(__dirname, '..', '..', '..', '..', rel), 'utf8')

describe('ticketKind', () => {
  it('reads the type from what filed the ticket, as the PRD table says', () => {
    for (const source of ['user', 'chat', 'agent', 'agent_output', 'activity', undefined]) expect(ticketKind(source)).toBe('Task')
    expect(ticketKind('recipe')).toBe('Playbook')
    for (const source of ['mission', 'orchestration_task', 'orchestration']) expect(ticketKind(source)).toBe('Mission')
    expect(ticketKind('heartbeat')).toBe('Routine')
  })

  it('marks a ticket a Claude Code session runs', () => {
    expect(runsInSession({ runtime_ref: { runtime: 'cli' } })).toBe(true)
    expect(runsInSession({ runtime_ref: null })).toBe(false)
  })

  it('names a ticket by number and title, and by title alone without a number', () => {
    expect(numberedTitle('#0042', 'Price list')).toBe('#0042 · Price list')
    expect(numberedTitle(null, 'Price list')).toBe('Price list')
  })

  it('every surface that names a ticket shows its number', () => {
    expect(read('components/command-center/board-tab.tsx')).toContain('{task.number && <span className="num">{task.number}</span>}')
    expect(read('components/activity/board/board-card.tsx')).toContain('{task.number && ')
    expect(read('components/activity/board/board-task-viewer.tsx')).toContain('{task.number && ')
    expect(read('components/activity/widgets/needs-you-widget.tsx')).toContain('numberedTitle(r.number')
    for (const feed of ['components/command-center/activity-tab.tsx', 'components/activity/activity-feed.tsx',
      'components/activity/command-center-history.tsx', 'components/activity/widgets/activity-widget.tsx']) {
      expect(read(feed), feed).toContain('numberedTitle(')
    }
    expect(read('components/chatbot/task-card.tsx')).toContain('Ticket {card.number ?? card.id}')
  })
})

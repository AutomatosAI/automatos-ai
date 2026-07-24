import { describe, it, expect } from 'vitest'
import { readFileSync } from 'fs'
import path from 'path'

// PRD-221 S13/S14 — source contracts for the feed card's progress line and
// follow-up-task action, and that Auto's Read is mounted in BOTH shells.

const read = (p: string) =>
  readFileSync(path.resolve(__dirname, '..', '..', '..', p), 'utf8')

describe('Auto\'s Read mounted in both Command Centre shells', () => {
  it('classic ActivityPage imports and renders AutosRead', () => {
    const src = read('components/activity/activity-page.tsx')
    expect(src).toContain('autos-read')
    expect(src).toContain('<AutosRead')
  })

  it('studio summary tab imports and renders AutosRead', () => {
    const src = read('components/command-center/summary-tab.tsx')
    expect(src).toContain('autos-read')
    expect(src).toContain('<AutosRead')
  })
})

describe('feed card progress line + follow-up action', () => {
  const src = read('components/activity/activity-feed-item.tsx')

  it('renders last_progress with a needs-attention emphasis path', () => {
    expect(src).toContain('item.last_progress')
    expect(src).toContain('requires_attention')
    // both the attention and normal treatments exist
    expect(src).toContain('text-warning')
  })

  it('shows the follow-up action only for failed / needs-attention items', () => {
    expect(src).toContain('needsAttention')
    expect(src).toContain("item.status === 'failed'")
    expect(src).toContain('last_progress?.requires_attention')
    expect(src).toContain('Create follow-up task')
  })

  it('creates the board task with activity source references', () => {
    expect(src).toContain("apiClient.post('/api/v1/tasks'")
    expect(src).toContain("source_type: 'activity'")
    expect(src).toContain('source_id: `${item.type}:${item.source_id ?? item.id}`')
  })
})

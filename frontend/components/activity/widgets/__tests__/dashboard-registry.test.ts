/** PRD-244 review batch 2 — the Summary's widget set: six by default, three opt-in, the eight superseded widgets gone. */
import { describe, it, expect } from 'vitest'
import { existsSync, readFileSync } from 'fs'
import path from 'path'

const DIR = path.resolve(__dirname, '..')
const src = readFileSync(path.join(DIR, 'command-centre-dashboard.tsx'), 'utf8')
const registry = src.slice(src.indexOf('const WIDGET_REGISTRY'), src.indexOf('const ALL_IDS'))
const entries = [...registry.matchAll(/id: '([a-z-]+)'[^\n]*defaultVisible: (true|false)/g)].map((m) => ({ id: m[1], visible: m[2] === 'true' }))

describe('Command Centre widget registry', () => {
  it('six widgets are visible by default, three opt-in', () => {
    expect(entries.filter((e) => e.visible).map((e) => e.id)).toEqual(['needs-you', 'activity', 'board-glance', 'schedule', 'agent-reports', 'cost-tracker'])
    expect(entries.filter((e) => !e.visible).map((e) => e.id)).toEqual(['agent-performance', 'playbook-metrics', 'self-learning'])
    expect(src).toContain('const DEFAULT_HIDDEN: string[] = WIDGET_REGISTRY.filter((w) => !w.defaultVisible)')
  })
  it('every registry id renders', () => {
    for (const e of entries) expect(src, e.id).toContain(`case '${e.id}':`)
  })
  it('the superseded widgets are deleted, not left lying around', () => {
    for (const w of ['active-now', 'recent-activity', 'decisions-needed', 'approval-gates', 'status-overview', 'priority-breakdown', 'types-of-work', 'team-workload']) {
      expect(existsSync(path.join(DIR, `${w}-widget.tsx`)), w).toBe(false)
      expect(src).not.toContain(`'./${w}-widget'`)
    }
  })
  it('saved layouts start from the new defaults once (v5 key, no legacy read)', () => {
    expect(src).toContain("'automatos:command-centre-v5'")
    expect(src).not.toContain('LEGACY_STORAGE_KEY')
  })
})

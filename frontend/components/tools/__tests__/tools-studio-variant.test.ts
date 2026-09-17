/** PRD-244 W5c — the Tools dashboard's Studio frame: same state, grid and modals; only the frame differs. */
import { describe, it, expect } from 'vitest'
import { readFileSync } from 'fs'
import path from 'path'

const src = readFileSync(path.resolve(__dirname, '..', 'tools-dashboard.tsx'), 'utf8')

describe('ToolsDashboard variants', () => {
  it('the Studio frame is the editorial page: head, cc-stats, cc-toolbar, one shared grid and shared modals', () => {
    const studio = src.slice(src.indexOf("if (variant === 'studio')"), src.indexOf('return (\n    <div className="space-y-6">'))
    for (const s of ['className="cc-page"', 'className="cc-h1"', 'className="cc-stats"', 'className="cc-toolbar"', '{applicationsGrid}', '{modals}', '<IntegrationsDisabledCard']) {
      expect(studio, s).toContain(s)
    }
    expect(studio).not.toContain('<PageHeader')
    expect(studio).not.toContain('<StatsBar')
  })

  it('the Classic frame is unchanged in kind and mounts the same grid and modals', () => {
    const classic = src.slice(src.indexOf('return (\n    <div className="space-y-6">'))
    for (const s of ['<PageHeader', '<StatsBar', '<Tabs value={activeTab}', '{applicationsGrid}', '{modals}']) expect(classic, s).toContain(s)
    expect(src.match(/<ToolCard\b/g)).toHaveLength(1) // one grid definition, mounted by both
    expect(src.match(/<ToolDetailsModal\b/g)).toHaveLength(1)
  })
})

import { readFileSync } from 'fs'
import path from 'path'
import { describe, expect, it } from 'vitest'

const FRONTEND = path.resolve(__dirname, '../../..')
const LEGACY_COPY = ['reusable recipes', 'Activity · live run', 'reusable recipe']
const SURFACES = [
  ['components/playbooks/PlaybooksPanel.tsx', 'Playbooks · reusable steps'],
  ['app/activity/execution/page.tsx', 'Command Center · live run'],
  ['lib/glossary.ts', 'A reusable set of steps for a Mission.'],
] as const

describe('canonical Playbook copy', () => {
  it.each(SURFACES)('%s uses the agreed wording without legacy copy', (file, expected) => {
    const source = readFileSync(path.join(FRONTEND, file), 'utf8')

    expect(source).toContain(expected)
    for (const legacy of LEGACY_COPY) {
      expect(source).not.toContain(legacy)
    }
  })
})

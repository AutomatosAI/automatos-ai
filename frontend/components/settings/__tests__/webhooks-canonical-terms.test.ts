import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import path from 'node:path'

const FRONTEND = path.resolve(__dirname, '../../..')
const LEGACY_COPY = ['Recipe Webhooks', 'Each recipe with a', 'individual recipes', "'Edit Recipe'"]
const SURFACES = [
  ['components/settings/WebhooksSettingsTab.tsx', [
    'Playbook Webhooks', 'Each playbook with a', 'individual playbooks',
  ]],
  ['components/activity/execution-detail.tsx', ["'Edit Playbook'"]],
] as const

describe('canonical Playbook wording in webhooks and run detail', () => {
  it.each(SURFACES)('%s uses Playbook wording without legacy visible phrases', (file, expected) => {
    const source = readFileSync(path.join(FRONTEND, file), 'utf8')

    for (const copy of expected) expect(source).toContain(copy)
    for (const legacy of LEGACY_COPY) expect(source).not.toContain(legacy)
  })
})

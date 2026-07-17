import { describe, it, expect } from 'vitest'
import { readFileSync } from 'fs'
import path from 'path'
import { getPageEntry } from '@/lib/generated/page-manifest'

// PRD-221 S7 — the widget empty state renders the current page's quick-prompt
// chips from the manifest mirror; admin-only prompts are hidden for members.

const visibleFor = (key: string, isAdmin: boolean) =>
  (getPageEntry(key)?.quick_prompts ?? []).filter((qp) => !qp.admin_only || isAdmin)

describe('quick-prompt chip data', () => {
  it('command-center offers member-visible starter prompts', () => {
    const chips = visibleFor('command-center', false)
    expect(chips.length).toBeGreaterThan(0)
    expect(chips.map((c) => c.text)).toContain('What needs my attention?')
  })

  it('hides admin-only prompts for members, shows them for admins', () => {
    const member = visibleFor('team', false).map((c) => c.text)
    const admin = visibleFor('team', true).map((c) => c.text)
    expect(member).not.toContain('Invite a teammate')
    expect(admin).toContain('Invite a teammate')
  })
})

describe('widget chip wiring (source contract)', () => {
  it('renders manifest quick prompts as chips wired to sendMessage', () => {
    const src = readFileSync(
      path.resolve(__dirname, '..', 'chat-widget.tsx'),
      'utf8',
    )
    expect(src).toContain('getPageEntry')
    expect(src).toContain('useSystemRole')
    expect(src).toContain('quickPrompts')
    expect(src).toContain('sendMessage(qp.text)')
    // admin filter present
    expect(src).toContain('!qp.admin_only || isAdmin')
  })
})

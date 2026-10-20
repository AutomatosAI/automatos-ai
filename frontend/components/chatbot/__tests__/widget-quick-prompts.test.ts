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
  // The empty thread lives in widget-empty-state.tsx (PRD-264: split out of
  // chat-widget.tsx for the code-shape limit); chat-widget.tsx still picks the
  // prompts and hands it sendMessage.
  const read = (file: string) => readFileSync(path.resolve(__dirname, '..', file), 'utf8')

  it('renders manifest quick prompts as chips wired to sendMessage', () => {
    const widget = read('chat-widget.tsx')
    expect(widget).toContain('getPageEntry')
    expect(widget).toContain('useSystemRole')
    expect(widget).toContain('quickPrompts={quickPrompts}')
    expect(widget).toContain('onSend={sendMessage}')
    // admin filter present
    expect(widget).toContain('!qp.admin_only || isAdmin')

    const emptyState = read('widget-empty-state.tsx')
    expect(emptyState).toContain('quickPrompts.map')
    expect(emptyState).toContain('send(qp.text)')
    expect(emptyState).toContain('if (!isLoading) onSend(text)')
  })
})

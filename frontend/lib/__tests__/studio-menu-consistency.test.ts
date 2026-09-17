/**
 * PRD-244 W0 — the Studio menu config tells the truth: every page tab is a
 * `?tab=` its page reads, the primary count matches the file's own claim, and
 * the role gates mirror the classic rail.
 */
import { describe, it, expect } from 'vitest'

import {
  STUDIO_MENU_FOOTER,
  STUDIO_MENU_PRIMARY,
  STUDIO_PAGE_TABS,
  resolveActiveMenuId,
  slugifyTab,
} from '@/lib/studio-menu'
import { DELIVERABLE_TABS } from '@/lib/deliverables/tabs'

describe('studio menu consistency', () => {
  it('has eleven primary items with unique ids', () => {
    expect(STUDIO_MENU_PRIMARY).toHaveLength(11)
    expect(new Set(STUDIO_MENU_PRIMARY.map((m) => m.id)).size).toBe(11)
  })

  it('registers page tabs only for menu ids that exist', () => {
    const ids = new Set(STUDIO_MENU_PRIMARY.map((m) => m.id))
    for (const key of Object.keys(STUDIO_PAGE_TABS)) expect(ids.has(key)).toBe(true)
    expect(STUDIO_PAGE_TABS).not.toHaveProperty('assign') // the hub composes its own tabs
    expect(STUDIO_PAGE_TABS).not.toHaveProperty('agents') // W5a: the Studio page composes its own
  })

  it('links every deliverables tab to a value the page reads', () => {
    expect(STUDIO_PAGE_TABS.deliv.map(slugifyTab)).toEqual([...DELIVERABLE_TABS])
  })

  it('keeps execution detail pages under Command Centre', () => {
    expect(resolveActiveMenuId('/activity/execution/42')).toBe('cmd')
    expect(resolveActiveMenuId('/command-center')).toBe('cmd')
  })

  it('gates Workspace Admin and Settings by role, like the classic rail', () => {
    expect(STUDIO_MENU_PRIMARY.find((m) => m.id === 'admin')?.requiredRole).toBe('admin')
    expect(STUDIO_MENU_FOOTER.find((m) => m.id === 'settings')?.requiredRole).toBe('admin')
    expect(STUDIO_MENU_PRIMARY.filter((m) => m.requiredRole)).toHaveLength(1)
  })
})

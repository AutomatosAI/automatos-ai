/**
 * PRD-244 — the Studio menu config tells the truth: the primary count matches
 * the file's own claim, the role gates mirror the classic rail, and there is no
 * header sub-nav config any more (W5b): every Studio page composes its own tabs.
 */
import { describe, it, expect } from 'vitest'

import * as menu from '@/lib/studio-menu'

const { STUDIO_MENU_FOOTER, STUDIO_MENU_PRIMARY, resolveActiveMenuId } = menu

describe('studio menu consistency', () => {
  it('has eleven primary items with unique ids', () => {
    expect(STUDIO_MENU_PRIMARY).toHaveLength(11)
    expect(new Set(STUDIO_MENU_PRIMARY.map((m) => m.id)).size).toBe(11)
  })

  it('carries no header sub-nav config — every Studio page composes its own tabs', () => {
    expect(Object.keys(menu)).not.toContain('STUDIO_PAGE_TABS')
    expect(Object.keys(menu)).not.toContain('slugifyTab')
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

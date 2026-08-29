/**
 * PRD-222 W2·S1b (US-024) — nav exposure gating logic.
 */
import { describe, it, expect } from 'vitest'
import { isNavItemVisible } from '@/lib/nav-exposure'

const basic = {
  plan: 'basic',
  families: { codegraph: false, nl2sql: false, team: false, voice: false },
  marketplace_depth: 1,
  nav: { analytics: false, team: false },
}
const business = {
  plan: 'business',
  families: { codegraph: true, nl2sql: true, team: true, voice: true },
  marketplace_depth: 3,
  nav: { analytics: true, team: true },
}

describe('isNavItemVisible', () => {
  it('always shows an item with no requiredExposure', () => {
    expect(isNavItemVisible(undefined, basic)).toBe(true)
    expect(isNavItemVisible(undefined, business)).toBe(true)
  })

  it('hides a gated item when the tier disables its nav key (basic)', () => {
    expect(isNavItemVisible('analytics', basic)).toBe(false)
    expect(isNavItemVisible('team', basic)).toBe(false)
  })

  it('shows a gated item when the tier enables its nav key (business)', () => {
    expect(isNavItemVisible('analytics', business)).toBe(true)
    expect(isNavItemVisible('team', business)).toBe(true)
  })

  it('fails open when exposure is unknown (loading / solo session)', () => {
    expect(isNavItemVisible('analytics', null)).toBe(true)
    expect(isNavItemVisible('analytics', undefined)).toBe(true)
  })
})

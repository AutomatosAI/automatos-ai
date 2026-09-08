import { describe, expect, it } from 'vitest'
import { addShellTab, canAddTab, closeTab, initialTabs, nextActive } from './terminal-tabs'

describe('terminal tabs (PRD-239 S7b)', () => {
  it('starts with the session for a Runtime Canvas, a terminal otherwise', () => {
    expect(initialTabs(true)[0]).toMatchObject({ kind: 'session', label: 'Session' })
    expect(initialTabs(false)[0]).toMatchObject({ kind: 'terminal', label: 'Terminal' })
  })

  it('adds numbered shells up to the host limit', () => {
    let tabs = initialTabs(true)
    tabs = addShellTab(tabs, 3)
    tabs = addShellTab(tabs, 3)
    expect(tabs.map((t) => t.label)).toEqual(['Session', 'Shell 1', 'Shell 2'])
    expect(canAddTab(tabs, 3)).toBe(false)
    expect(addShellTab(tabs, 3)).toBe(tabs)
    expect(canAddTab(initialTabs(false), undefined)).toBe(true) // default cap 4
  })

  it('never closes the first tab and picks a sensible neighbour', () => {
    let tabs = addShellTab(addShellTab(initialTabs(true), 4), 4)
    const [first, s1, s2] = tabs
    expect(closeTab(tabs, first.id)).toBe(tabs)
    expect(nextActive(tabs, s2.id, s2.id)).toBe(s1.id)
    expect(nextActive(tabs, first.id, s2.id)).toBe(first.id)
    tabs = closeTab(tabs, s1.id)
    expect(tabs.map((t) => t.id)).toEqual([first.id, s2.id])
  })
})

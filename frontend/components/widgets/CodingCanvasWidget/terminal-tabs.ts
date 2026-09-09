/**
 * PRD-239 S7b — several terminals in one Canvas. Pure tab model, no React.
 *
 * The first tab is the session (a Runtime Canvas) or a plain terminal; every
 * added tab is a plain shell in the same folder. The host caps how many
 * terminals it serves at once (its `max_terminals` capability, 4 by default).
 */
export interface TerminalTab {
  id: string
  kind: 'session' | 'terminal' | 'shell'
  label: string
}

export const DEFAULT_MAX_TERMINALS = 4

export function initialTabs(runtime: boolean): TerminalTab[] {
  return [{ id: 't1', kind: runtime ? 'session' : 'terminal', label: runtime ? 'Session' : 'Terminal' }]
}

export function canAddTab(tabs: TerminalTab[], maxTerminals: number | null | undefined): boolean {
  return tabs.length < Math.max(1, maxTerminals ?? DEFAULT_MAX_TERMINALS)
}

export function addShellTab(tabs: TerminalTab[], maxTerminals: number | null | undefined): TerminalTab[] {
  if (!canAddTab(tabs, maxTerminals)) return tabs
  const n = tabs.filter((t) => t.kind === 'shell').length + 1
  const id = `t${Date.now().toString(36)}${tabs.length}`
  return [...tabs, { id, kind: 'shell', label: `Shell ${n}` }]
}

/** Closing a tab never leaves the strip empty: the first tab stays. */
export function closeTab(tabs: TerminalTab[], id: string): TerminalTab[] {
  if (tabs.length <= 1 || tabs[0].id === id) return tabs
  return tabs.filter((t) => t.id !== id)
}

/** The tab to show after `closed` went away. */
export function nextActive(tabs: TerminalTab[], active: string, closed: string): string {
  if (active !== closed) return active
  const idx = tabs.findIndex((t) => t.id === closed)
  const remaining = tabs.filter((t) => t.id !== closed)
  return (remaining[Math.max(0, idx - 1)] ?? remaining[0]).id
}

/**
 * #942 — the pure half of the agent page's "Session tools" picker.
 *
 * Night 9b (F329): a Business Analyst moved to the CLI runtime answered "There is no database
 * query tool here … someone needs to run a SELECT", because a session only had a fixed core of
 * platform tools and nothing on the agent page said so. The backend now lets the owner choose
 * tool groups per agent and reports what the workspace has that the sessions cannot reach; these
 * helpers read that answer defensively (an older backend carries none of it) and build the
 * owner's selection. Nothing here fetches or renders.
 */

import type {
  SessionToolGapEntry,
  SessionToolGroup,
  SessionToolGroups,
  WorkspaceCapabilityGap,
} from '@/types/session-tools'

/** The tools every session has whatever groups are on (`services/session_tools.py`). Shown when the backend's own list has not answered. */
export const SESSION_CORE_TOOLS: readonly string[] = [
  'board_summary',
  'list_tasks',
  'update_ticket',
  'submit_report',
  'ask_human',
  'composio_execute',
  'search_knowledge',
  'search_documents',
  'record_memory',
  'read_step_file',
]

function str(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function strings(value: unknown): string[] {
  return Array.isArray(value) ? value.map(str).filter(Boolean) : []
}

function normalizeGroup(entry: unknown): SessionToolGroup[] {
  const row = (entry ?? {}) as Record<string, unknown>
  const id = str(row.id)
  if (!id) return []
  return [{ id, label: str(row.label) || id, description: str(row.description), tools: strings(row.tools) }]
}

/**
 * The agent detail's `session_tool_groups`, or null when the backend predates #942 (or sent
 * something unreadable) — the picker then does not render. Pure.
 */
export function normalizeSessionToolGroups(raw: unknown): SessionToolGroups | null {
  if (!raw || typeof raw !== 'object') return null
  const src = raw as Record<string, unknown>
  const available = Array.isArray(src.available) ? src.available.flatMap(normalizeGroup) : []
  if (!available.length) return null
  const known = new Set(available.map((g) => g.id))
  return {
    enabled: strings(src.enabled).filter((id) => known.has(id)),
    is_default: src.is_default === true,
    available,
  }
}

/** The workspace capabilities a session cannot reach (`kind: "workspace"`), messageless rows dropped. Pure. */
export function workspaceGaps(gaps: SessionToolGapEntry[] | null | undefined): WorkspaceCapabilityGap[] {
  if (!Array.isArray(gaps)) return []
  return gaps.flatMap((gap) => {
    if (gap?.kind !== 'workspace' || !str(gap.message)) return []
    return [{ ...gap, message: str(gap.message), group: str(gap.group) }]
  })
}

/** The group a gap's one-click fix turns on, or '' when it offers none. Pure. */
export function gapFixGroup(gap: WorkspaceCapabilityGap): string {
  return str(gap.fix?.enable_group)
}

/**
 * The tools that stay on whatever the owner picks: the session-mode settings' list minus every
 * group's tools (that list may name them all), else the known core. Pure.
 */
export function alwaysOnTools(sessionToolNames: string[], groups: SessionToolGroups): string[] {
  const grouped = new Set(groups.available.flatMap((g) => g.tools))
  const core = sessionToolNames.filter((name) => !grouped.has(name))
  return core.length ? core : [...SESSION_CORE_TOOLS]
}

/** The selection with one group turned on or off, in the backend's display order. Pure. */
export function toggleGroup(groups: SessionToolGroups, selection: string[], id: string, on: boolean): string[] {
  const chosen = new Set(selection)
  if (on) chosen.add(id)
  else chosen.delete(id)
  return groups.available.map((g) => g.id).filter((groupId) => chosen.has(groupId))
}

/** A stored or edited group list as the form keeps it: strings only, trimmed; undefined when it is not a list. Pure. */
export function sessionGroupsField(raw: unknown): string[] | undefined {
  return Array.isArray(raw) ? strings(raw) : undefined
}

/** `GET /api/agents/{id}`, with `?groups=` to preview a selection the owner has not saved yet. Pure. */
export function agentDetailPath(agentId: number, groups: string[] | null): string {
  const base = `/api/agents/${agentId}`
  return groups === null ? base : `${base}?groups=${groups.map(encodeURIComponent).join(',')}`
}

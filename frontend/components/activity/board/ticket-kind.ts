/**
 * PRD-252 R4 — tickets you can tell apart: number, type and title.
 *
 * Every card said TASK, and a mission step only added a small tag, so two
 * tickets with the same title looked the same. A ticket's number (#0042, a
 * mission step's #0051.3) comes from the API; its type comes from what filed it,
 * as the PRD's table says, and a mark shows when a Claude Code session runs it.
 */
export type TicketKind = 'Task' | 'Playbook' | 'Mission' | 'Routine'

// source_type → the type a person reads; anything else filed a Task.
const KINDS: Record<string, TicketKind> = {
  recipe: 'Playbook',
  playbook: 'Playbook',
  orchestration: 'Mission',
  orchestration_task: 'Mission',
  mission: 'Mission',
  heartbeat: 'Routine',
  routine: 'Routine',
}

export function ticketKind(sourceType: string | null | undefined): TicketKind {
  return KINDS[sourceType ?? ''] ?? 'Task'
}

/** A Claude Code session runs this ticket (its claim stamps the runtime). */
export function runsInSession(task: { runtime_ref?: { runtime?: string } | null }): boolean {
  return task.runtime_ref?.runtime === 'cli'
}

/** "#0042 · Price list": how a list names a ticket; the title alone when it has no number. */
export function numberedTitle(number: string | null | undefined, title: string): string {
  return number ? `${number} · ${title}` : title
}

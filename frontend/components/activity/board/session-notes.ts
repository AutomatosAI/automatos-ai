/**
 * F094 (night 5) — the notes on a Claude Code session ticket.
 *
 * `runtime_ref.session_notes` holds what the session said while it worked (its
 * update_ticket tool), what the operator wrote, and the mission's verdict when
 * it stopped waiting for a step or was cancelled while the step ran. The ticket
 * showed none of them.
 */

export interface SessionNote {
  note: string
  by?: string
  at?: string
}

export function sessionNotes(ref: Record<string, any> | null | undefined): SessionNote[] {
  const raw = ref?.session_notes
  if (!Array.isArray(raw)) return []
  return raw
    .filter((n): n is Record<string, unknown> =>
      !!n && typeof n === 'object' && typeof n.note === 'string' && n.note.trim() !== '')
    .map((n) => ({
      note: String(n.note),
      by: typeof n.by === 'string' ? n.by : undefined,
      at: typeof n.at === 'string' ? n.at : undefined,
    }))
}

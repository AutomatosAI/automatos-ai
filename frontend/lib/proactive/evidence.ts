/**
 * PRD-268 §6.5.1 (US-W01a) — where a suggestion's evidence opens in the app.
 *
 * A check-in cites the reads it made as `{type, id}` refs (W1's US-C03b: services/proactive/signals.py
 * and reads.py; a failing streak cites its runs, failing_streaks.py). A `ticket` names a board number
 * ("0846", "#0846"), while a `task` or `board_task` names the ticket's row id, which the board opens
 * directly. Every other kind opens the page that lists it, by the same routes the bell uses
 * (components/notifications/notification-bell.tsx linkFor). A kind the app has no page for is quoted,
 * never linked.
 *
 * The refs are data the check-in read: an id that isn't a plain identifier opens nothing, and every id
 * that reaches a path is encoded.
 */
import type { EvidenceRef } from './types'
import { chatHref, DELIVERABLE_PARAM, DELIVERABLES_HREF, missionHref, ticketHref } from '@/lib/ticket-links'

/** The ref type whose id is a board number (modules/proactive/provenance.py TICKET). */
export const TICKET_REF = 'ticket'
/** A board number as a ref holds it: "#0846", "0846", or a mission step's "#0051.3" (its card's number and its place). */
const BOARD_NUMBER = /^#?(\d{1,9})(\.\d{1,4})?$/
const STEP = /\.\d+$/
/** A ticket's row id. */
const ROW_ID = /^\d{1,12}$/
/** What a ref's id may be to reach a path (services/proactive/reads.py MAX_ID_LENGTH). */
const PLAIN_ID = /^[A-Za-z0-9_.:-]{1,64}$/
/** The board number's digits (lib/ticket-number.ts). */
const NUMBER_DIGITS = 4
/** Ids up to this long are shown whole; a longer one (a UUID) shows its first eight characters. */
const SHORT_ID = 12

type HrefOf = (id: string) => string | null

interface EvidenceKind {
  readonly label: string
  readonly href: HrefOf
  /** False where the id means nothing to the person: a ticket's row id (never shown as its number, F414), a workspace. */
  readonly showId?: false
}

const always = (href: string): HrefOf => () => href
const rowTicket: HrefOf = (id) => (ROW_ID.test(id) ? ticketHref(id) : null)

/** Each kind the app can open, with its link's words. */
const EVIDENCE_KINDS: Readonly<Record<string, EvidenceKind>> = Object.freeze({
  task: { label: 'Ticket', href: rowTicket, showId: false },
  board_task: { label: 'Ticket', href: rowTicket, showId: false },
  report: { label: 'Report', href: (id) => `/command-center?tab=feed&report=${encodeURIComponent(id)}` },
  run: { label: 'Check-in run', href: always('/command-center?tab=feed') },
  heartbeat: { label: 'Heartbeat', href: always('/command-center?tab=feed') },
  document: { label: 'Document', href: always('/documents') },
  deliverable: { label: 'Deliverable', href: (id) => `${DELIVERABLES_HREF}?${DELIVERABLE_PARAM}=${encodeURIComponent(id)}` },
  mission: { label: 'Mission', href: missionHref },
  chat: { label: 'Chat', href: chatHref },
  watch: { label: 'Watch', href: always('/command-center?tab=watchlist') },
  grant: { label: 'Approval', href: always('/command-center?tab=governance') },
  scheduled_task: { label: 'Scheduled task', href: always('/command-center?tab=calendar') },
  credits: { label: 'AI credits', href: always('/settings?tab=billing'), showId: false },
  agent: { label: 'Agent', href: (id) => `/agents?agent=${encodeURIComponent(id)}` },
})

function kindOf(ref: EvidenceRef): string {
  return ref.type.trim().toLowerCase()
}

function capitalised(words: string): string {
  return words ? words[0].toUpperCase() + words.slice(1) : words
}

function hasKind(kind: string): boolean {
  return Object.prototype.hasOwnProperty.call(EVIDENCE_KINDS, kind)
}

/** A `ticket` ref's board number as the board writes it, "#0846" or a step's "#0051.3"; null for anything else. */
export function boardNumberOf(ref: EvidenceRef): string | null {
  if (kindOf(ref) !== TICKET_REF) return null
  const match = BOARD_NUMBER.exec(ref.id.trim())
  return match ? `#${match[1].padStart(NUMBER_DIGITS, '0')}${match[2] ?? ''}` : null
}

/** The card a board number belongs to: a step's mission card ("#0051" for "#0051.3"), else the number itself. */
export function cardNumberOf(number: string): string {
  return number.replace(STEP, '')
}

/** Where a ref opens, or null: a ticket's board number is resolved on the board (evidence-links.tsx). */
export function evidenceHref(ref: EvidenceRef): string | null {
  const kind = kindOf(ref)
  const id = ref.id.trim()
  if (!hasKind(kind) || !PLAIN_ID.test(id)) return null
  return EVIDENCE_KINDS[kind].href(id)
}

/** The link's words: "Ticket #0846", "Report 12", "Mission 3f2a9c1e"; a kind the app has no page for, in words. */
export function evidenceLabel(ref: EvidenceRef): string {
  const number = boardNumberOf(ref)
  if (number) return `Ticket ${number}`
  const kind = kindOf(ref)
  const known = hasKind(kind) ? EVIDENCE_KINDS[kind] : null
  const label = known ? known.label : capitalised(kind.replace(/_/g, ' '))
  const id = ref.id.trim()
  if (known?.showId === false || !id) return label
  return `${label} ${id.length > SHORT_ID ? id.slice(0, 8) : id}`
}

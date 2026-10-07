/**
 * PRD-256 US-001 — what Auto did this turn, written by the platform from the calls
 * that ran, never by the model.
 *
 * Live, the backend sends one frame after the tool loop:
 * `d:{"type":"receipts","data":{"receipts":[…],"model":…}}`. Reloaded, the saved
 * message carries the same list as its `receipts` part. A message from before
 * PRD-256 has neither and renders as it always did.
 *
 * US-002: the same frame carries `above`, the lines the platform puts above the reply
 * text (a write that was refused, work said done that was not). Saved, they are the top
 * of the reply's text, so only the live message needs them.
 */
import type { ChatMessage, MessagePart, Receipt } from '@/types'

export const NO_ACTIONS = 'No actions in this turn.'
export const NOTHING_ON_THE_BOARD = 'No actions on the board.'
export const REFUSED_PREFIX = 'tried, refused'
export const SKIPPED_PREFIX = 'not run'
const NO_REASON = 'no reason given'
const STATUSES: ReadonlySet<string> = new Set(['done', 'refused', 'skipped'])
/** A card on the board is named by its number, as the board shows it (#0422). */
const CARD_NUMBER = /^#\d/

type ReceiptsPart = Extract<MessagePart, { type: 'receipts' }>

function isReceipt(value: unknown): value is Receipt {
  const r = value as Partial<Receipt> | null
  return Boolean(r) && typeof r?.action === 'string' && typeof r?.status === 'string' && STATUSES.has(r.status)
}

/** The receipts a `receipts` frame's data carries; undefined for a frame without a list. */
export function receiptsFromFrame(data: unknown): Receipt[] | undefined {
  const list = (data as { receipts?: unknown } | null | undefined)?.receipts
  return Array.isArray(list) ? list.filter(isReceipt) : undefined
}

/** The lines above the reply a `receipts` frame carries; undefined when it has none. */
export function aboveFromFrame(data: unknown): string[] | undefined {
  const list = (data as { above?: unknown } | null | undefined)?.above
  if (!Array.isArray(list)) return undefined
  const lines = list.filter((line): line is string => typeof line === 'string' && line.trim() !== '')
  return lines.length > 0 ? lines : undefined
}

/** What a `receipts` frame sets on the live message: its receipts and the lines above the reply. */
export function liveReceipts(data: unknown): Pick<ChatMessage, 'receipts' | 'receiptsAbove'> {
  return { receipts: receiptsFromFrame(data), receiptsAbove: aboveFromFrame(data) }
}

/** A message's receipts: live from the frame, else from its saved part; undefined before PRD-256. */
export function receiptsOf(message: Pick<ChatMessage, 'receipts' | 'parts'>): Receipt[] | undefined {
  if (message.receipts) return message.receipts
  const part = message.parts?.find((p): p is ReceiptsPart => p.type === 'receipts')
  return part && Array.isArray(part.receipts) ? part.receipts.filter(isReceipt) : undefined
}

/** One receipt in plain words: "#0422: moved to Done"; a call that did nothing names what it was
 * for, never an effect it did not have: "Letter to Maya: tried, refused: <reason>". */
export function receiptLine(r: Receipt): string {
  if (r.status === 'done') return [r.subject, r.effect].filter(Boolean).join(': ')
  const prefix = r.status === 'refused' ? REFUSED_PREFIX : SKIPPED_PREFIX
  const what = r.subject || r.effect
  return `${what ? `${what}: ` : ''}${prefix}: ${r.reason || NO_REASON}`
}

/** Whether a done write touched a card on the board (named by its number). */
export function movedSomethingOnTheBoard(receipts: Receipt[]): boolean {
  return receipts.some((r) => r.kind === 'write' && r.status === 'done' && CARD_NUMBER.test(r.subject))
}

/** An internal page only: a receipt's link is a path in this app, never another site. */
export function safeLink(link: string | null): string | null {
  return link && link.startsWith('/') && !link.startsWith('//') ? link : null
}

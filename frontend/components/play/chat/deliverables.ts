import { liveReceiptsFromToolData } from '@/lib/chat/receipts'
import { DELIVERABLE_PARAM, DELIVERABLES_HREF } from '@/lib/ticket-links'
import type { Receipt } from '@/types'

/**
 * PRD-260 §4, K8 — which Deliverables a turn made, for the inline card in the
 * chat, ported from Automatos Play's lib/chat/deliverables.ts. Two sources:
 * - the document tool's tool-data part, {generated_document: {deliverable_id,
 *   title, format, …}} (format_for_frontend in
 *   orchestrator/modules/tools/formatting/result_formatter.py);
 * - a done receipt whose link names a Deliverable (/deliverables/<id>, or the
 *   core's open_url, deliverable_open_url in generate_document_tool.py:
 *   FRONTEND_URL + /deliverables?tab=outputs&deliverable=<id>, so absolute
 *   wherever FRONTEND_URL is set). Receipts don't link a Deliverable yet
 *   (consumers/chatbot/receipts.py _link names a card or an agent only:
 *   decision F8, backend), so today only the live tool-data part brings a
 *   card, and a reopened conversation's reply shows none.
 * Both come off the wire, so only the id is kept, checked against
 * DELIVERABLE_ID; the link is rebuilt on the core's own page, relative
 * (deliverableHref), whatever origin the receipt named. A Deliverable's page
 * is the core's /deliverables?deliverable=<id> (DeliverableDeepLink's shape).
 * Pure.
 */

export type DeliverableKind = 'document' | 'image' | 'video'

export type DeliverableRef = {
  readonly id: string
  readonly title: string
  readonly kind: DeliverableKind
}

const DELIVERABLE_ID = /^[A-Za-z0-9_-]{1,128}$/
const DELIVERABLE_PAGE = /^\/deliverables\/([^/]+)\/?$/
const DELIVERABLES_LIST = /^\/deliverables\/?$/
/** Only for resolving relative links; never fetched. */
const LINK_BASE = 'http://play.invalid'
/** An absolute receipt link: the open_url with FRONTEND_URL in front. Any other scheme is no link. */
const ABSOLUTE_LINK = /^https?:\/\//i
const GENERATED_DOCUMENT = 'generated_document'

const IMAGE_FORMATS: ReadonlySet<string> = new Set(['png', 'jpg', 'jpeg', 'gif', 'webp', 'svg'])
const VIDEO_FORMATS: ReadonlySet<string> = new Set(['mp4', 'webm', 'mov'])
export const DEFAULT_KIND: DeliverableKind = 'document'
export const UNTITLED = 'Your Deliverable'

/** The kind in words, for the card's type badge. */
export const KIND_WORDS: Readonly<Record<DeliverableKind, string>> = {
  document: 'Document',
  image: 'Image',
  video: 'Video',
}

type Fields = Readonly<Record<string, unknown>>

function recordOf(value: unknown): Fields | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? (value as Fields) : null
}

function textOf(fields: Fields, key: string): string | null {
  const value = fields[key]
  return typeof value === 'string' && value.trim() ? value.trim() : null
}

export function isDeliverableId(id: string): boolean {
  return DELIVERABLE_ID.test(id)
}

/** The core's page for one Deliverable (K8). */
export function deliverableHref(id: string): string {
  return `${DELIVERABLES_HREF}?${DELIVERABLE_PARAM}=${encodeURIComponent(id)}`
}

/** The generated document's format as a shelf kind: the core files non-media documents as "document". */
export function kindOfFormat(format: string | null): DeliverableKind {
  const lower = (format ?? '').toLowerCase()
  if (IMAGE_FORMATS.has(lower)) return 'image'
  if (VIDEO_FORMATS.has(lower)) return 'video'
  return DEFAULT_KIND
}

/** The id as sent: a string, or a whole number; anything else is no id. */
function idOf(fields: Fields): string | null {
  const value = fields.deliverable_id
  if (typeof value === 'number' && Number.isSafeInteger(value)) return String(value)
  return typeof value === 'string' ? value : null
}

/** The Deliverable in a tool-data part's data, or null when it names none (or a bad id). */
export function deliverableFromToolData(data: unknown): DeliverableRef | null {
  const made = recordOf(recordOf(data)?.[GENERATED_DOCUMENT])
  if (made === null) return null
  const id = idOf(made)
  if (id === null || !isDeliverableId(id)) return null
  return {
    id,
    title: textOf(made, 'title') ?? textOf(made, 'filename') ?? UNTITLED,
    kind: kindOfFormat(textOf(made, 'format')),
  }
}

function parseLink(link: string): URL | null {
  try {
    return new URL(link, LINK_BASE)
  } catch (error) {
    console.warn('[play-chat] a receipt link that is not a URL:', link, error)
    return null
  }
}

/** "/deliverables/<id>": the id, still to be checked. The URL parser leaves it percent-encoded. */
function idOnPage(url: URL): string | null {
  const raw = DELIVERABLE_PAGE.exec(url.pathname)?.[1]
  if (!raw) return null
  try {
    return decodeURIComponent(raw)
  } catch (error) {
    console.warn('[play-chat] a Deliverable link with a broken id:', raw, error)
    return null
  }
}

/** A path in the app ("/…", not "//host"), or an absolute http(s) link. */
function isReceiptLink(link: string): boolean {
  return ABSOLUTE_LINK.test(link) || (link.startsWith('/') && !link.startsWith('//'))
}

/**
 * The Deliverable id a receipt link names (a path, or the absolute open_url), else null. Only the
 * checked id leaves here: the card links to deliverableHref(id), never to the receipt's own URL.
 */
export function deliverableIdFromLink(link: string | null): string | null {
  if (!link || !isReceiptLink(link)) return null
  const url = parseLink(link)
  if (url === null) return null
  const id = idOnPage(url) ?? (DELIVERABLES_LIST.test(url.pathname) ? url.searchParams.get(DELIVERABLE_PARAM) : null)
  return id && isDeliverableId(id) ? id : null
}

/** The Deliverables the done receipts link to (none until receipts link Deliverables: decision F8). */
export function deliverablesFromReceipts(receipts: readonly Receipt[] | undefined): DeliverableRef[] {
  return (receipts ?? []).flatMap((receipt) => {
    const id = receipt.status === 'done' ? deliverableIdFromLink(receipt.link) : null
    return id ? [{ id, title: receipt.subject.trim() || UNTITLED, kind: DEFAULT_KIND }] : []
  })
}

/**
 * The Deliverables one of useChat's onData parts announces, live: a 'tool-data'
 * part's generated_document, and the done receipts of the receipts frame it
 * carries (read by lib/chat/receipts.ts, as the reply reads them). None for any
 * other part.
 */
export function deliverablesInPart(part: unknown): DeliverableRef[] {
  const forwarded = recordOf(part)
  if (forwarded?.type !== 'tool-data') return []
  const made = deliverableFromToolData(forwarded.data)
  const linked = deliverablesFromReceipts(liveReceiptsFromToolData(part)?.receipts)
  return made ? [made, ...linked] : linked
}

/** `known` with the new ones from `found` after it; one per id, the first one seen wins. Never mutates. */
export function mergeDeliverables(
  known: readonly DeliverableRef[],
  found: readonly DeliverableRef[],
): readonly DeliverableRef[] {
  const fresh = found.filter(
    (ref, index) => !known.some((seen) => seen.id === ref.id) && found.findIndex((other) => other.id === ref.id) === index,
  )
  return fresh.length > 0 ? [...known, ...fresh] : known
}

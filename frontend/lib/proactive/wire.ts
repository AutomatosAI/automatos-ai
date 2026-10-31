/**
 * PRD-268 §6.11 (US-C21b) — the proactive answers, checked off the wire before anything draws them.
 *
 * The hooks (hooks/use-suggestions.ts, use-brief.ts, use-readiness.ts) read every answer through
 * these. An id is checked before it reaches a request path, a row that isn't what it says is left
 * out (the rest are kept), and a readiness fix that isn't an app path loses its link. Unknown
 * fields pass through untouched; nothing here mutates what it reads. A suggestion's title and
 * detail are model-written data: these readers keep them as text, and the surfaces draw them as
 * text, never markup.
 */
import { isDisplayState } from './display-state'
import { BRIEF_PART, BRIEF_SECTIONS } from './types'
import type {
  Brief,
  BriefCopy,
  BriefMessageRef,
  BriefOpened,
  BriefCountedSection,
  BriefRow,
  BriefSection,
  EvidenceRef,
  ProactiveRefusal,
  Readiness,
  ReadinessItem,
  ReadinessNeed,
  Suggestion,
} from './types'

type Fields = Readonly<Record<string, unknown>>

/** A suggestion's, a chat's and a message's id: the server's UUIDs, nothing else in a path. */
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i
const READINESS_STATES: ReadonlySet<string> = new Set(['ready', 'not_checked', 'needs_setup'])
const READINESS_NEEDS: ReadonlySet<string> = new Set(['brief', 'voice', 'push', 'check_in'])
const OPENS_TYPES: ReadonlySet<string> = new Set(['ticket', 'mission', 'question', 'chat', 'deliverable', 'suggestion', 'settings'])
const ACTION_VERBS: ReadonlySet<string> = new Set(['start', 'answer', 'open', 'connect', 'hide'])
/** A fix opens an app path: `/` then neither `/` nor `\`, and no control character or backslash anywhere
 * (a browser drops tabs and newlines, so `/\t/host` would be read as `//host`). */
const APP_PATH = /^\/(?![/\\])[^\u0000-\u001f\u007f\\]*$/

/** The value as a JSON object's fields, or null when it is not one. */
export function recordOf(value: unknown): Fields | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? (value as Fields) : null
}

/** The value when it is text, else undefined. */
export function textOf(value: unknown): string | undefined {
  return typeof value === 'string' ? value : undefined
}

function listOf(value: unknown): readonly unknown[] {
  return Array.isArray(value) ? value : []
}

/** True for an id the proactive routes and the chat routes take. */
export function isProactiveId(value: unknown): value is string {
  return typeof value === 'string' && UUID.test(value)
}

function evidenceOf(value: unknown): EvidenceRef[] {
  return listOf(value)
    .map(recordOf)
    .filter((ref): ref is Fields => ref !== null && typeof ref.type === 'string' && typeof ref.id === 'string')
    .map((ref) => ({ type: ref.type as string, id: ref.id as string }))
}

/** One suggestion as the list writes it, or null without its id, its title and what Start does. */
export function suggestionOf(value: unknown): Suggestion | null {
  const fields = recordOf(value)
  const action = recordOf(fields?.proposed_action)
  if (fields === null || !isProactiveId(fields.id) || typeof fields.title !== 'string' || typeof action?.kind !== 'string') {
    return null
  }
  return {
    ...(fields as unknown as Suggestion),
    detail: textOf(fields.detail) ?? '',
    evidence: evidenceOf(fields.evidence),
    proposed_action: { ...(action as unknown as Suggestion['proposed_action']), payload: recordOf(action?.payload) },
  }
}

/** GET /api/proactive/suggestions' rows. One without its id, its title and what Start does is left out. */
export function suggestionsOf(body: unknown): Suggestion[] {
  return listOf(recordOf(body)?.suggestions)
    .map(suggestionOf)
    .filter((row): row is Suggestion => row !== null)
}

/** The answer of POST /api/proactive/brief/open: the day's message, else pending. */
export function openedOf(body: unknown): BriefOpened {
  const fields = recordOf(body)
  const chatId = fields?.chat_id
  const messageId = fields?.message_id
  if (!isProactiveId(chatId) || !isProactiveId(messageId)) return { posted: false, pending: true }
  return { chat_id: chatId, message_id: messageId, posted: fields?.posted === true }
}

/** Where the day's brief is, when an open answered it. */
export function briefRefOf(opened: BriefOpened): BriefMessageRef | null {
  return 'chat_id' in opened ? { chat_id: opened.chat_id, message_id: opened.message_id } : null
}

function opensOf(value: unknown): BriefRow['opens'] {
  const opens = recordOf(value)
  const type = textOf(opens?.type)
  const id = textOf(opens?.id)
  return type !== undefined && OPENS_TYPES.has(type) && id !== undefined ? { type: type as NonNullable<BriefRow['opens']>['type'], id } : null
}

function actionOf(value: unknown): BriefRow['action'] {
  const action = recordOf(value)
  const verb = textOf(action?.verb)
  const label = textOf(action?.label)
  if (verb === undefined || !ACTION_VERBS.has(verb) || label === undefined) return null
  return { verb: verb as NonNullable<BriefRow['action']>['verb'], label, done_label: textOf(action?.done_label) ?? null }
}

/** A row with its id, title and why; where it opens, its action and its state only when each is what it says. */
function rowOf(value: unknown): BriefRow | null {
  const fields = recordOf(value)
  if (fields === null) return null
  const ok = typeof fields.id === 'string' && typeof fields.title === 'string' && typeof fields.why === 'string'
  if (!ok) return null
  const state = fields.display_state
  return {
    ...(fields as unknown as BriefRow),
    at: textOf(fields.at) ?? null,
    number: textOf(fields.number) ?? null,
    display_state: isDisplayState(state) ? state : null,
    opens: opensOf(fields.opens),
    action: actionOf(fields.action),
  }
}

function sectionsOf(value: unknown): Record<BriefSection, BriefRow[]> {
  const fields = recordOf(value)
  const rows = (section: BriefSection): BriefRow[] =>
    listOf(fields?.[section]).map(rowOf).filter((row): row is BriefRow => row !== null)
  return Object.fromEntries(BRIEF_SECTIONS.map((section) => [section, rows(section)])) as Record<BriefSection, BriefRow[]>
}

function countsOf(value: unknown): Brief['counts'] {
  const fields = recordOf(value)
  const count = (section: BriefCountedSection): number => {
    const counted = fields?.[section]
    return typeof counted === 'number' && Number.isFinite(counted) ? counted : 0
  }
  return { needs_you: count('needs_you'), suggestions: count('suggestions'), today: count('today'), since: count('since') }
}

/** The server's words per section: only a known section's text is kept. */
function wordsOf(value: unknown): Partial<Record<BriefSection, string>> {
  const fields = recordOf(value)
  const kept = BRIEF_SECTIONS.filter((section) => typeof fields?.[section] === 'string')
  return Object.fromEntries(kept.map((section) => [section, fields?.[section]])) as Partial<Record<BriefSection, string>>
}

/** The brief's copy (US-C23), or undefined when it carries none: only a known section's label and text are kept. */
function copyOf(value: unknown): BriefCopy | undefined {
  const fields = recordOf(value)
  if (fields === null) return undefined
  return {
    section_labels: wordsOf(fields.section_labels),
    empty_sentence: textOf(fields.empty_sentence),
    show_more_template: textOf(fields.show_more_template),
  }
}

/** A `brief@1` part's Brief@1, or null when it isn't one. A row without its id, title and why is left out. */
export function briefOf(value: unknown): Brief | null {
  const fields = recordOf(value)
  if (fields === null || fields.v !== 1) return null
  const words = [fields.date, fields.greeting, fields.prose].every((field) => typeof field === 'string')
  if (!words) return null
  const labels = recordOf(fields.labels)
  return {
    ...(fields as unknown as Brief),
    spoken: textOf(fields.spoken) ?? '',
    sections: sectionsOf(fields.sections),
    counts: countsOf(fields.counts),
    labels: { sections: wordsOf(labels?.sections), more: wordsOf(labels?.more), empty: textOf(labels?.empty) },
    copy: copyOf(fields.copy),
    state_hash: textOf(fields.state_hash) ?? '',
    generated_at: textOf(fields.generated_at) ?? '',
  }
}

/** The brief a message carries in its `brief@1` part, or null. */
export function briefOfParts(parts: unknown): Brief | null {
  const part = listOf(parts).map(recordOf).find((fields) => fields?.type === BRIEF_PART)
  return part ? briefOf(part.brief) : null
}

/** The brief of the message `ref` names, among GET /api/chat/{chat_id}/messages' answer; null when absent. */
export function briefOfMessages(messages: unknown, ref: BriefMessageRef): Brief | null {
  const message = listOf(messages).map(recordOf).find((fields) => fields?.id === ref.message_id)
  return message ? briefOfParts(message.parts) : null
}

/** True for an app path (APP_PATH); a readiness fix links only to one (ReadinessList checks its props too). */
export function appPath(href: unknown): href is string {
  return typeof href === 'string' && APP_PATH.test(href)
}

function fixOf(value: unknown): ReadinessItem['fix'] {
  const fix = recordOf(value)
  const label = textOf(fix?.label)
  const href = fix?.href
  return label !== undefined && appPath(href) ? { label, href } : null
}

function itemOf(value: unknown): ReadinessItem | null {
  const fields = recordOf(value)
  const id = textOf(fields?.id)
  const state = textOf(fields?.state)
  if (fields === null || id === undefined || state === undefined || !READINESS_STATES.has(state)) return null
  return {
    ...(fields as unknown as ReadinessItem),
    label: textOf(fields.label) ?? id,
    detail: textOf(fields.detail) ?? '',
    fix: fixOf(fields.fix),
    needed_by: listOf(fields.needed_by).filter((need): need is ReadinessNeed => typeof need === 'string' && READINESS_NEEDS.has(need)),
  }
}

/** GET /api/readiness' Readiness@1. An item that isn't one is left out. */
export function readinessOf(body: unknown): Readiness {
  const fields = recordOf(body)
  const items = listOf(fields?.items).map(itemOf).filter((item): item is ReadinessItem => item !== null)
  return { v: 1, checked_at: textOf(fields?.checked_at) ?? '', items }
}

/**
 * The refusal a failed proactive call carries (apiClient's ApiHttpError: the status, and the
 * `{code, detail}` body), or null when `error` is not an error. A failure with no body still gives
 * its message, so a card can always say one line.
 */
export function refusalOf(error: unknown): ProactiveRefusal | null {
  if (!(error instanceof Error)) return null
  const failure = error as Error & { status?: unknown; body?: unknown }
  const body = recordOf(failure.body)
  return {
    code: textOf(body?.code) ?? null,
    detail: textOf(body?.detail) ?? failure.message,
    status: typeof failure.status === 'number' ? failure.status : null,
  }
}

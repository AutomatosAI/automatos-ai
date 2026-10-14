import { CHAT_HREF } from '@/lib/ticket-links'

/**
 * PRD-260 §4 — a new chat with words waiting in Play's message box, ported from
 * Automatos Play's lib/chat/params.ts: /chat?draft=<text> (a Deliverable's "Ask
 * Auto about this"). Play's chat reads it once, opens a new chat with the words
 * in its box, and drops it from the URL (hooks/use-play-conversation.ts). The
 * core's own chat page has no such parameter and ignores it.
 */

export const DRAFT_PARAM = 'draft'
/** Longer drafts are cut: the address is not the place for a whole message. */
export const MAX_DRAFT_CHARS = 500
const REPLACEMENT_CHARACTER = '�'
const HIGH_SURROGATES = [0xd800, 0xdbff] as const
const LOW_SURROGATES = [0xdc00, 0xdfff] as const

/** Words for the message box; a new serial puts them in a fresh box, which takes focus. */
export type ComposerFill = { readonly text: string; readonly serial: number }

function within(code: number, [low, high]: readonly [number, number]): boolean {
  return code >= low && code <= high
}

/** Half of a pair on its own: encodeURIComponent throws on one. */
function isLoneSurrogate(char: string): boolean {
  const code = char.charCodeAt(0)
  return char.length === 1 && (within(code, HIGH_SURROGATES) || within(code, LOW_SURROGATES))
}

/**
 * The first MAX_DRAFT_CHARS characters, never half an emoji. Array.from splits by
 * code point, so a lone surrogate (a Deliverable's title is backend text) stands
 * alone and is replaced.
 */
export function cutDraft(text: string): string {
  return Array.from(text)
    .slice(0, MAX_DRAFT_CHARS)
    .map((char) => (isLoneSurrogate(char) ? REPLACEMENT_CHARACTER : char))
    .join('')
}

/** A new chat with these words waiting in the message box. */
export function draftHref(text: string): string {
  return `${CHAT_HREF}?${DRAFT_PARAM}=${encodeURIComponent(cutDraft(text))}`
}

/** The words a link asks a new chat to start with, or null (none, or only spaces). */
export function draftFrom(params: Pick<URLSearchParams, 'get'> | null): string | null {
  const value = params?.get(DRAFT_PARAM)
  return value && value.trim() ? cutDraft(value) : null
}

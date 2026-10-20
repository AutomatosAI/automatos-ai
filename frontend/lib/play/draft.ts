import { CHAT_HREF } from '@automatos/theme-sdk'

/**
 * PRD-260 §4 — a new chat with words waiting in Play's message box, ported from
 * Automatos Play's lib/chat/params.ts: /chat?draft=<text> (a Deliverable's "Ask
 * Auto about this"). Play's chat reads it once, opens a new chat with the words
 * in its box, and drops it from the URL (hooks/use-play-conversation.ts). The
 * core's own chat page has no such parameter and ignores it. Any link can carry
 * one, so the words are always one plain line (oneLine below): nothing hidden
 * on a second line, which one Enter would send to Auto.
 *
 * PRD-261 A-14c: Play reads the core only through '@automatos/theme-sdk'. The
 * one-line rule is Play's own copy of the core's lib/text/plain-line.ts (the
 * surface map copies pure text helpers into Play). The two must stay the same:
 * lib/__tests__/prd261-port-conversations.test.ts runs both over every
 * character and fails on any difference.
 */

export const DRAFT_PARAM = 'draft'
/** Longer drafts are cut: the address is not the place for a whole message. */
export const MAX_DRAFT_CHARS = 500
const REPLACEMENT_CHARACTER = '�'
/**
 * Every format character (Unicode Cf) except ZWNJ and ZWJ, which emoji
 * sequences and some scripts need: bidi controls, zero-width spaces, the BOM,
 * soft hyphens, interlinear annotations and the tag characters (invisible, but
 * a model reads them: a link could carry words the owner never sees).
 */
const FORMAT_BUT_JOINERS = String.raw`(?![\u200C\u200D])\p{Cf}`
/**
 * C0 and C1 controls, line and paragraph separators, private use and
 * unassigned code points, and variation selectors 17 to 256 (U+E0100 to
 * U+E01EF), which can carry hidden data. U+FE0E and U+FE0F stay.
 */
const BREAKING_OR_HIDDEN = String.raw`[\p{Cc}\p{Zl}\p{Zp}\p{Co}\p{Cn}\u{E0100}-\u{E01EF}]`
const WHITE_SPACE = String.raw`\s`
/** A run of unsafe characters and white space, as one. */
const UNSAFE_OR_SPACE_RUN = new RegExp(`(?:${[FORMAT_BUT_JOINERS, BREAKING_OR_HIDDEN, WHITE_SPACE].join('|')})+`, 'gu')
const ONE_SPACE = ' '
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

/** The text as one plain line: each run of unsafe characters and white space becomes one space, the ends trimmed. */
function oneLine(text: string): string {
  return text.replace(UNSAFE_OR_SPACE_RUN, ONE_SPACE).trim()
}

/**
 * The words as one plain line, then its first MAX_DRAFT_CHARS characters, never
 * half an emoji. Array.from splits by code point, so a lone surrogate (a
 * Deliverable's title is backend text) stands alone and is replaced.
 */
export function cutDraft(text: string): string {
  return Array.from(oneLine(text))
    .slice(0, MAX_DRAFT_CHARS)
    .map((char) => (isLoneSurrogate(char) ? REPLACEMENT_CHARACTER : char))
    .join('')
}

/** A new chat with these words waiting in the message box. */
export function draftHref(text: string): string {
  return `${CHAT_HREF}?${DRAFT_PARAM}=${encodeURIComponent(cutDraft(text))}`
}

/** The words a link asks a new chat to start with, or null (none, or nothing but spaces and unsafe characters). */
export function draftFrom(params: Pick<URLSearchParams, 'get'> | null): string | null {
  const words = cutDraft(params?.get(DRAFT_PARAM) ?? '')
  return words === '' ? null : words
}

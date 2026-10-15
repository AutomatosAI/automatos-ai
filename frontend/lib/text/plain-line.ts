/**
 * One line of plain text, for words that come from outside the code (a Look's
 * name, a link's ?draft=): nothing in them may break the line, hide text or
 * reorder what the reader sees.
 */

/**
 * Characters that don't belong in one line of plain text: C0 and C1 controls,
 * line and paragraph separators, bidi overrides and isolates, zero-width spaces
 * and the BOM. ZWJ and ZWNJ stay: emoji sequences and some scripts need them.
 */
export const UNSAFE_TEXT = /[\p{Cc}\p{Zl}\p{Zp}\u061C\u200B\u200E\u200F\u202A-\u202E\u2060-\u2069\uFEFF]/u

/** A run of unsafe characters and white space, as one. */
const UNSAFE_OR_SPACE_RUN = new RegExp(`(?:${UNSAFE_TEXT.source}|\\s)+`, 'gu')
const ONE_SPACE = ' '

/** The text as one plain line: each run of unsafe characters and white space becomes one space, the ends trimmed. */
export function oneLine(text: string): string {
  return text.replace(UNSAFE_OR_SPACE_RUN, ONE_SPACE).trim()
}

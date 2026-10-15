/**
 * PRD-260 P260-FP-1 — text with hidden characters, for the tests of one line of
 * plain text (lib/text/plain-line.ts) and the two places that use it: a Look's
 * name and a link's ?draft=. Every special character is written as an escape.
 */

/** U+E0000 plus an ASCII code is that character's invisible tag twin (U+E0020 to U+E007E). */
const TAG_BASE = 0xe0000
/** Opens a tag sequence: LANGUAGE TAG. */
const TAG_OPEN = '\u{E0001}'
/** Closes one: CANCEL TAG. */
const TAG_CLOSE = '\u{E007F}'

/** Words a reader can't see but a model reads: ASCII written in Unicode tag characters. */
export function inTagCharacters(words: string): string {
  const tags = Array.from(words, (char) => String.fromCodePoint(TAG_BASE + char.charCodeAt(0))).join('')
  return `${TAG_OPEN}${tags}${TAG_CLOSE}`
}

/** A family: man, woman and girl joined by two ZWJs. */
export const FAMILY_EMOJI = '\u{1F468}\u200D\u{1F469}\u200D\u{1F467}'
/** A red heart: U+2764 and VARIATION SELECTOR-16 (emoji presentation). */
export const RED_HEART = '\u2764\uFE0F'
/** SOFT HYPHEN: invisible unless a line breaks at it. */
export const SOFT_HYPHEN = '\u00AD'
/** INTERLINEAR ANNOTATION ANCHOR. */
export const ANNOTATION_ANCHOR = '\uFFF9'
/** VARIATION SELECTOR-17, the first of the 240 that can carry hidden data. */
export const VS17 = '\u{E0100}'
/** VARIATION SELECTOR-256, the last of them. */
export const VS256 = '\u{E01EF}'

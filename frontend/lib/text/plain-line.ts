/**
 * One line of plain text, for words that come from outside the code (a Look's
 * name, a link's ?draft=): nothing in them may break the line, hide text or
 * reorder what the reader sees.
 */

/**
 * Every format character (Unicode Cf) except ZWNJ and ZWJ, which emoji
 * sequences and some scripts need: bidi controls, zero-width spaces, the BOM,
 * soft hyphens, interlinear annotations and the tag characters (invisible, but
 * a model reads them: a link could carry words the owner never sees). It also
 * drops the tags of a subdivision flag, which then shows as a plain black flag.
 */
const FORMAT_BUT_JOINERS = String.raw`(?![\u200C\u200D])\p{Cf}`
/**
 * C0 and C1 controls, line and paragraph separators, private use and
 * unassigned code points, and variation selectors 17 to 256 (U+E0100 to
 * U+E01EF), which can carry hidden data. U+FE0E and U+FE0F (text and emoji
 * presentation) stay.
 */
const BREAKING_OR_HIDDEN = String.raw`[\p{Cc}\p{Zl}\p{Zp}\p{Co}\p{Cn}\u{E0100}-\u{E01EF}]`
const UNSAFE_PIECES: readonly string[] = [FORMAT_BUT_JOINERS, BREAKING_OR_HIDDEN]
const WHITE_SPACE = String.raw`\s`

/** One character matching any of the pieces. */
function anyOf(pieces: readonly string[]): string {
  return `(?:${pieces.join('|')})`
}

/** One character that doesn't belong in one line of plain text. */
export const UNSAFE_TEXT = new RegExp(anyOf(UNSAFE_PIECES), 'u')

/** A run of unsafe characters and white space, as one. */
const UNSAFE_OR_SPACE_RUN = new RegExp(`${anyOf([...UNSAFE_PIECES, WHITE_SPACE])}+`, 'gu')
const ONE_SPACE = ' '

/** The text as one plain line: each run of unsafe characters and white space becomes one space, the ends trimmed. */
export function oneLine(text: string): string {
  return text.replace(UNSAFE_OR_SPACE_RUN, ONE_SPACE).trim()
}

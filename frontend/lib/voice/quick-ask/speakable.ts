/**
 * Theme voice, web quick-ask: what Auto's reply sounds like. toSpeakable turns the markdown Auto writes into words a
 * voice can say: the browser's guarantee whether or not the server speechified the turn (spokenMode). Ported from
 * automatos-go src/voice/speech/speakable.ts (PR #15).
 *
 * It runs on the whole text so far at every streamed update (sentences.ts queues what is new), so it holds up
 * mid-stream: an unclosed code fence is code to the end, a '**' still waiting for its pair is dropped, and a heading
 * or list item gets its full stop only once its line is complete. Blocks first (code fences and tables: one sentence
 * each, pointing to the chat), then each other line: its marks (quote, heading, bullet, number) and its inline
 * markdown are stripped until neither changes it, so toSpeakable(toSpeakable(x)) === toSpeakable(x). Links say their
 * label; a bare URL says "a link".
 */
import { SPOKEN_COPY } from './copy'

type BlockKind = 'code' | 'table' | 'text'
interface PlainLine {
  readonly text: string
  readonly sentence: boolean
}

const FENCE = /^\s*(`{3,}|~{3,})/
const QUOTES = /^(?:\s*>\s?)+/
const QUOTE = /^\s*>\s?/
const HEADING = /^\s*#{1,6}(?:\s+|$)/
const HEADING_CLOSE = /\s+#+\s*$/
const ITEM = /^\s*(?:[-*+•]|\d{1,3}[.)])\s+(?:\[[ xX]\]\s+)?/
const RULE = /^\s*([-*_=])(?:\s*\1){2,}\s*$/
const SETEXT_UNDERLINE = /^\s*(?:={3,}|-{3,})\s*$/
const TABLE_SEPARATOR = /^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*$/
const ENDED = /[.!?…]["'”’)\]]*$/
const SOFT_END = /[,:;]$/

/** A line's marks, outermost first: a quote is not a sentence, a heading or an item is. */
const MARKS: readonly { readonly pattern: RegExp; readonly sentence: boolean }[] = [
  { pattern: QUOTE, sentence: false },
  { pattern: HEADING, sentence: true },
  { pattern: ITEM, sentence: true },
]

/** Inline markdown, in order: links before bare URLs, pairs before strays. */
const INLINE: readonly (readonly [RegExp, string])[] = [
  [/!\[([^\]]*)\]\([^)]*\)/g, '$1'],
  [/\[([^\]]+)\]\([^)]*\)/g, '$1'],
  [/\[([^\]]+)\]\[[^\]]*\]/g, '$1'],
  [/<(?:https?:\/\/|mailto:)[^>\s]+>/gi, SPOKEN_COPY.link],
  [/\b(?:https?:\/\/|www\.)[^\s<>]*[^\s<>.,;:!?'")\]]/gi, SPOKEN_COPY.link],
  [/`([^`]+)`/g, '$1'],
  [/\*\*(?=\S)(.*?\S)\*\*/g, '$1'],
  [/(^|\W)__(?=\S)(.*?\S)__(?!\w)/g, '$1$2'],
  [/~~(?=\S)(.*?\S)~~/g, '$1'],
  [/(^|[^\w*])\*(?=[^\s*])(.*?[^\s*])\*(?![\w*])/g, '$1$2'],
  [/(^|\W)_(?=[^\s_])(.*?[^\s_])_(?!\w)/g, '$1$2'],
  // Strays: a pair still streaming, a marker with no partner.
  [/\*{2,}|~{2,}|`+/g, ''],
  [/(^|\s)[*_]+(?=\S)/g, '$1'],
  [/(\S)[*_]+(?=[\s.,!?;:…]|$)/g, '$1'],
  [/[[\]]/g, ''],
  [/\|/g, ' '],
]

/** Enough rounds for the marks and the inline rules to settle. */
const MAX_ROUNDS = 4

const NOTICES: Readonly<Record<Exclude<BlockKind, 'text'>, string>> = Object.freeze({
  code: SPOKEN_COPY.code,
  table: SPOKEN_COPY.table,
})

function hasWords(text: string): boolean {
  return text.trim().length > 0
}

function isTableSeparator(line: string | undefined): boolean {
  return line !== undefined && line.includes('|') && TABLE_SEPARATOR.test(line)
}

/** A row of a table: a leading pipe, the separator, its header, or a pipe row after them. */
function isTableRow(lines: readonly string[], index: number, afterTable: boolean): boolean {
  const line = lines[index] ?? ''
  if (!line.includes('|')) return false
  if (line.trimStart().startsWith('|') || isTableSeparator(line)) return true
  return isTableSeparator(lines[index + 1]) || (afterTable && hasWords(line))
}

function closesFence(line: string, fence: string): boolean {
  const trimmed = line.trim()
  return trimmed.length >= fence.length && trimmed === fence.charAt(0).repeat(trimmed.length)
}

/** Each line's block: inside a code fence (to its close, or the end), in a table, or text. */
function classify(lines: readonly string[]): readonly BlockKind[] {
  const kinds: BlockKind[] = []
  let fence: string | null = null
  lines.forEach((line, index) => {
    if (fence !== null) {
      kinds.push('code')
      if (closesFence(line, fence)) fence = null
      return
    }
    const opened = FENCE.exec(line)
    if (opened) {
      fence = opened[1] ?? '```'
      kinds.push('code')
      return
    }
    kinds.push(isTableRow(lines, index, kinds[index - 1] === 'table') ? 'table' : 'text')
  })
  return kinds
}

function markOf(text: string): (typeof MARKS)[number] | undefined {
  return MARKS.find((mark) => mark.pattern.test(text))
}

/** The line without its leading marks, and whether one of them makes it a sentence. */
function stripMarks(line: string): PlainLine {
  let text = line
  let sentence = false
  // Each mark removes at least its marker character, so this ends.
  for (let mark = markOf(text); mark !== undefined; mark = markOf(text)) {
    text = text.replace(mark.pattern, '')
    if (mark.pattern === HEADING) text = text.replace(HEADING_CLOSE, '')
    sentence = sentence || mark.sentence
  }
  return { text, sentence }
}

function stripInline(text: string): string {
  return INLINE.reduce((out, [pattern, replacement]) => out.replace(pattern, replacement), text)
}

/** One text line in plain words, and whether it is a sentence of its own (a heading or an item). */
function plainLine(line: string): PlainLine {
  let text = line
  let sentence = false
  for (let round = 0; round < MAX_ROUNDS; round += 1) {
    if (RULE.test(text)) return { text: '', sentence }
    const unmarked = stripMarks(text)
    const next = stripInline(unmarked.text)
    sentence = sentence || unmarked.sentence
    if (next === text) break
    text = next
  }
  return { text: text.replace(/\s+/g, ' ').trim(), sentence }
}

/** The text ending as a sentence: a full stop added, or put in place of a comma, colon or semicolon. */
function endSentence(text: string): string {
  if (ENDED.test(text)) return text
  return SOFT_END.test(text) ? `${text.slice(0, -1)}.` : `${text}.`
}

/** A block's sentence, once for a run of the same block; the words before it end there. */
function withNotice(pieces: readonly string[], notice: string): readonly string[] {
  const last = pieces[pieces.length - 1]
  if (last === undefined) return [notice]
  if (last === notice) return pieces
  return [...pieces.slice(0, -1), endSentence(last), notice]
}

function speakLine(lines: readonly string[], index: number): string {
  const { text, sentence } = plainLine(lines[index] ?? '')
  if (!hasWords(text)) return ''
  // Complete: the next line has begun. A heading underlined with === or --- is a sentence too.
  const next = lines[index + 1]
  const complete = next !== undefined
  const underlined = complete && SETEXT_UNDERLINE.test(next)
  return complete && (sentence || underlined) ? endSentence(text) : text
}

/** Auto's markdown as plain words for text to speech; whitespace collapsed to single spaces. */
export function toSpeakable(text: string): string {
  const lines = text
    .replace(/\r\n?/g, '\n')
    .split('\n')
    .map((line) => line.replace(QUOTES, ''))
  const kinds = classify(lines)
  const pieces = lines.reduce<readonly string[]>((out, _line, index) => {
    const kind = kinds[index] ?? 'text'
    if (kind !== 'text') return withNotice(out, NOTICES[kind])
    const spoken = speakLine(lines, index)
    return spoken ? [...out, spoken] : out
  }, [])
  return pieces.join(' ').replace(/\s+/g, ' ').trim()
}

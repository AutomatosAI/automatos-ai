import { checkLook, freshLookId, type ThemeLook, type ThemeLookCheck, type ThemeLookRules } from './spec'

/**
 * PRD-261 A-19d1 (plan §2.2.4) — a theme's looks as files: what Export writes and what Import
 * accepts, for any theme. A file or pasted text is untrusted: it is size-checked before it is read
 * or parsed, then goes through the theme's checks (./spec.ts checkLook) like anything else from
 * outside the code, and gets a fresh id. What Export writes is the look as those checks keep it:
 * the spec's fields only, in its order.
 */

/** 1 MB. A look file is a few hundred bytes; anything this big is the wrong file. */
export const LOOK_FILE_MAX_BYTES = 1024 * 1024
export const LOOK_FILE_SUFFIX = '.look.json'
export const LOOK_FILE_TYPE = 'application/json'
const JSON_INDENT = 2
/** The file name when the look's name leaves nothing a file system keeps. */
const FALLBACK_FILE_NAME = 'look'
const UNSAFE_REPLACEMENT = '-'
const LEADING_DOTS = /^\.+/

export const TOO_BIG = "That's 1 MB or more. A Look file is well under 1 KB: check it's a .look.json file."
export const NOTHING_TO_IMPORT = "There's nothing to import yet. Paste a Look or choose a file."
export const NOT_JSON = "That isn't JSON. Paste the whole Look, from the first { to the last }."
export const UNREADABLE = "That file couldn't be read. Try choosing it again."

/** Characters a file name can't hold on Windows or macOS. */
const FILE_NAME_UNSAFE: ReadonlySet<string> = new Set(['\\', '/', ':', '*', '?', '"', '<', '>', '|'])

/** A look as a file: what Export offers for download. */
export interface LookFile {
  readonly fileName: string
  readonly text: string
  /** The file's media type. */
  readonly type: string
}

function refused(error: string): ThemeLookCheck {
  return { ok: false, errors: [error] }
}

/** True for a size (file bytes, or pasted characters) at or over the 1 MB limit. */
export function tooBig(size: number): boolean {
  return size >= LOOK_FILE_MAX_BYTES
}

/** '<name>.look.json', with characters a file system refuses turned into dashes. */
export function lookFileName(look: Pick<ThemeLook, 'name'>): string {
  const safe = Array.from(look.name)
    .map((char) => (FILE_NAME_UNSAFE.has(char) ? UNSAFE_REPLACEMENT : char))
    .join('')
    .trim()
    .replace(LEADING_DOTS, '')
  return `${safe === '' ? FALLBACK_FILE_NAME : safe}${LOOK_FILE_SUFFIX}`
}

/** The checked look pretty-printed, ending in a newline. */
function lookText(look: ThemeLook): string {
  return `${JSON.stringify(look, null, JSON_INDENT)}\n`
}

/** The look as a file, once it passes the theme's checks; null otherwise. */
export function lookFile(rules: Pick<ThemeLookRules, 'spec' | 'table'>, input: unknown): LookFile | null {
  const checked = checkLook(rules, input)
  if (!checked.ok) return null
  return Object.freeze({ fileName: lookFileName(checked.look), text: lookText(checked.look), type: LOOK_FILE_TYPE })
}

/** Pasted or read text → a look ready to save under a fresh id (no preset's, none of `saved`), or plain-word errors. */
export function importLook(rules: ThemeLookRules, text: string, saved: readonly ThemeLook[]): ThemeLookCheck {
  if (tooBig(text.length)) return refused(TOO_BIG)
  if (text.trim() === '') return refused(NOTHING_TO_IMPORT)
  let parsed: unknown
  try {
    parsed = JSON.parse(text)
  } catch (error) {
    console.warn('[looks] import: not JSON', error)
    return refused(NOT_JSON)
  }
  const result = checkLook(rules, parsed)
  return result.ok ? { ok: true, look: Object.freeze({ ...result.look, id: freshLookId(rules, saved) }) } : result
}

/** A Blob's text through FileReader (every browser, and jsdom). */
function readText(file: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(typeof reader.result === 'string' ? reader.result : '')
    reader.onerror = () => reject(reader.error ?? new Error('FileReader failed'))
    reader.onabort = () => reject(new Error('FileReader aborted'))
    reader.readAsText(file)
  })
}

export type ReadResult =
  | { readonly ok: true; readonly text: string }
  | { readonly ok: false; readonly errors: readonly string[] }

/** A chosen file's text. Over the limit, it is refused before a byte is read. */
export async function readLookFile(file: Blob): Promise<ReadResult> {
  if (tooBig(file.size)) return { ok: false, errors: [TOO_BIG] }
  try {
    return { ok: true, text: await readText(file) }
  } catch (error) {
    console.warn("[looks] import: the file couldn't be read", error)
    return { ok: false, errors: [UNREADABLE] }
  }
}

/** A chosen file → importLook (readLookFile, then the same checks as pasted text). */
export async function importLookFile(rules: ThemeLookRules, file: Blob, saved: readonly ThemeLook[]): Promise<ThemeLookCheck> {
  const read = await readLookFile(file)
  return read.ok ? importLook(rules, read.text, saved) : read
}

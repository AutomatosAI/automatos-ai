import { freshLookId } from './editor'
import { LOOK_FIELDS, type Look, type ValidationResult } from './types'
import { validateLook } from './validate'

/**
 * PRD-260 — looks as files: what Export writes and what Import accepts. A file
 * or pasted text is untrusted: it is size-checked before it is read or parsed,
 * then goes through validateLook like anything else from outside the code, and
 * gets a fresh id.
 */

/** 1 MB. A look file is a few hundred bytes; anything this big is the wrong file. */
export const LOOK_FILE_MAX_BYTES = 1024 * 1024
export const LOOK_FILE_SUFFIX = '.look.json'
export const LOOK_FILE_TYPE = 'application/json'
const JSON_INDENT = 2

export const TOO_BIG = "That's 1 MB or more. A Look file is well under 1 KB: check it's a .look.json file."
export const NOTHING_TO_IMPORT = "There's nothing to import yet. Paste a Look or choose a file."
export const NOT_JSON = "That isn't JSON. Paste the whole Look, from the first { to the last }."
export const UNREADABLE = "That file couldn't be read. Try choosing it again."

/** Characters a file name can't hold on Windows or macOS. */
const FILE_NAME_UNSAFE = new Set(['\\', '/', ':', '*', '?', '"', '<', '>', '|'])

function refused(error: string): ValidationResult {
  return { ok: false, errors: [error] }
}

/** True for a size (file bytes, or pasted characters) at or over the 1 MB limit. */
export function tooBig(size: number): boolean {
  return size >= LOOK_FILE_MAX_BYTES
}

/** The look as a file holds it: pretty-printed, only the look's own fields, in format order. */
export function lookJson(look: Look): string {
  return `${JSON.stringify(look, [...LOOK_FIELDS], JSON_INDENT)}\n`
}

/** '<name>.look.json', with characters a file system refuses turned into dashes. */
export function lookFileName(look: Look): string {
  const safe = Array.from(look.name)
    .map((char) => (FILE_NAME_UNSAFE.has(char) ? '-' : char))
    .join('')
    .trim()
    .replace(/^\.+/, '')
  return `${safe === '' ? 'look' : safe}${LOOK_FILE_SUFFIX}`
}

/** Pasted or read text → a look ready to save under a fresh id, or plain-word errors. */
export function importLook(text: string, saved: readonly Look[]): ValidationResult {
  if (tooBig(text.length)) return refused(TOO_BIG)
  if (text.trim() === '') return refused(NOTHING_TO_IMPORT)
  let parsed: unknown
  try {
    parsed = JSON.parse(text)
  } catch (error) {
    console.warn('[looks] import: not JSON', error)
    return refused(NOT_JSON)
  }
  const result = validateLook(parsed)
  return result.ok ? { ok: true, look: { ...result.look, id: freshLookId(saved) } } : result
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
export async function importLookFile(file: Blob, saved: readonly Look[]): Promise<ValidationResult> {
  const read = await readLookFile(file)
  return read.ok ? importLook(read.text, saved) : read
}

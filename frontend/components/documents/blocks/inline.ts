// Inline content helpers (PRD-167 S5).
//
// The dependency-free editor authors a block's inline content as a plain string with
// {{path}} variable tokens. These helpers convert between that string and the canonical
// Inline[] run model, so what the user types round-trips losslessly to the schema and
// resolves live in the preview pane. (When Plate lands, this seam is where the adapter
// replaces the string model — see the S1 memo.)

import type { Inline, TextRun, VariableRun } from './types'

const TOKEN_RE = /\{\{\s*([a-zA-Z0-9_.]+)\s*\}\}/g

// "Dear {{user.name}}," -> [text "Dear ", variable user.name, text ","]
export function parseInline(text: string): Inline[] {
  const runs: Inline[] = []
  let lastIndex = 0
  let match: RegExpExecArray | null
  TOKEN_RE.lastIndex = 0
  while ((match = TOKEN_RE.exec(text)) !== null) {
    if (match.index > lastIndex) {
      runs.push({ type: 'text', text: text.slice(lastIndex, match.index) })
    }
    runs.push({ type: 'variable', path: match[1] })
    lastIndex = match.index + match[0].length
  }
  if (lastIndex < text.length) {
    runs.push({ type: 'text', text: text.slice(lastIndex) })
  }
  return runs
}

// Inline[] -> "Dear {{user.name}},"
export function serializeInline(content: Inline[]): string {
  return (content || [])
    .map((run) =>
      run.type === 'variable' ? `{{${(run as VariableRun).path}}}` : (run as TextRun).text,
    )
    .join('')
}

// Insert a {{path}} token into a string at a cursor position.
export function insertToken(text: string, cursor: number, path: string): string {
  const token = `{{${path}}}`
  const at = cursor < 0 || cursor > text.length ? text.length : cursor
  return text.slice(0, at) + token + text.slice(at)
}

let _idCounter = 0
// Stable-ish client id for new blocks (not persisted as meaningful — just a React key
// and a schema `id`). Avoids Math.random for determinism in tests.
export function newBlockId(prefix = 'b'): string {
  _idCounter += 1
  return `${prefix}-${Date.now().toString(36)}-${_idCounter}`
}

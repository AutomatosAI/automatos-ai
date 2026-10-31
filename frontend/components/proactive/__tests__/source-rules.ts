/**
 * PRD-268 K17 (US-C21c) — the source rules the proactive components' tests hold them to: tokens
 * only (no hex colour, no orange-* class, no rounded-3xl) and every React component at most 150 lines.
 */
import { readFileSync } from 'fs'
import path from 'path'

export const MAX_COMPONENT_LINES = 150
/** A hex colour literal: #abc, #aabbcc, #aabbccdd. */
export const HEX_COLOUR = /#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3}(?:[0-9a-fA-F]{2})?)?\b/
export const ORANGE_CLASS = /\borange-/
export const ROUNDED_3XL = /\brounded-3xl\b/

/** A file of components/proactive, read as text. */
export function proactiveSource(file: string): string {
  return readFileSync(path.resolve(__dirname, '..', file), 'utf8')
}

export interface ComponentSpan {
  readonly name: string
  readonly lines: number
}

/** Each top-level React component (a capitalised function) and its length, from `function` to its closing brace. */
export function componentSpans(source: string): ComponentSpan[] {
  const lines = source.split('\n')
  return lines.flatMap((line, index) => {
    const match = /^(?:export )?function ([A-Z]\w*)\(/.exec(line)
    if (!match) return []
    const end = lines.findIndex((other, at) => at > index && other === '}')
    return [{ name: match[1], lines: end - index + 1 }]
  })
}

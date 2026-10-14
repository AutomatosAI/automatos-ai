/**
 * PRD-260 §4 — a tool call in words, ported from Automatos Play's
 * lib/chat/labels.ts: what the call is about, from its input (the search
 * query, the page, the title). The caption under Auto shows it beside
 * "Searching." while the call runs. Tool names stay in lib/mascot/tools.ts.
 */

const DETAIL_KEYS = ['query', 'q', 'url', 'title', 'name', 'filename', 'path'] as const
const DETAIL_CHARS = 80

function recordOf(value: unknown): Readonly<Record<string, unknown>> | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null
}

/** One line, at most `limit` characters, an ellipsis when cut. */
export function clip(text: string, limit: number): string {
  const flat = text.replace(/\s+/g, ' ').trim()
  return flat.length <= limit ? flat : `${flat.slice(0, limit - 1).trimEnd()}…`
}

/** The first non-blank text field of the call's input, by DETAIL_KEYS order; null when it has none. */
export function toolDetail(input: unknown): string | null {
  const fields = recordOf(input)
  if (fields === null) return null
  for (const key of DETAIL_KEYS) {
    const value = fields[key]
    if (typeof value === 'string' && value.trim()) return clip(value, DETAIL_CHARS)
  }
  return null
}

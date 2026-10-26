/**
 * Theme voice — a chat turn's stream, one line at a time: the AI SDK data-stream frames POST /api/chat writes, read
 * into the few a turn sent outside React needs (turn-stream.ts): Auto's text, the new chat's id, narration leaving
 * the answer (F186), the answering agent (PRD-266 US-007b) and an error (PRD-239 S4). useChat (hooks.ts) reads the
 * same frames inline and more of them (tools, receipts, cards); it is not split under the code-shape limit yet, so it
 * keeps its own reader until it is.
 */
import { errorFromDataPayload, parseErrorFrame } from '@/lib/chat/errors'
import type { TurnError } from '@/types'

export type ChatFrame =
  | { readonly kind: 'text'; readonly text: string }
  | { readonly kind: 'chat-id'; readonly chatId: string }
  /** The text just streamed was narration (the round called tools): it leaves the answer. */
  | { readonly kind: 'narration'; readonly text: string }
  | { readonly kind: 'agent'; readonly name: string }
  | { readonly kind: 'error'; readonly message: string; readonly code: string | null }

type Fields = Readonly<Record<string, unknown>>

const TEXT_PREFIX = '0:'
const DATA_PREFIX = 'd:'
const ERROR_PREFIX = 'e:'
const LEGACY_PREFIX = /^data:\s*/

function parseJson(raw: string): unknown {
  try {
    return JSON.parse(raw)
  } catch {
    // A line cut short or not JSON: not a frame this reader knows (useChat skips it too).
    return undefined
  }
}

function fieldsOf(value: unknown): Fields | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? (value as Fields) : null
}

function errorFrame(error: TurnError): ChatFrame {
  return { kind: 'error', message: error.message, code: error.code ?? null }
}

function dataFrame(data: Fields): ChatFrame | null {
  const chatId = data.chatId
  const said = fieldsOf(data.data)?.text
  const agentName = fieldsOf(data.agent)?.name
  if (data.type === 'chat-id' && typeof chatId === 'string' && chatId) return { kind: 'chat-id', chatId }
  if (data.type === 'narration' && typeof said === 'string') return { kind: 'narration', text: said }
  if (data.type === 'agent-info' && typeof agentName === 'string') return { kind: 'agent', name: agentName }
  return null
}

/** The legacy `data: {json}` lines some backends send: a text delta or an error. */
function legacyFrame(payload: Fields): ChatFrame | null {
  const delta = payload.delta
  if (payload.type === 'text-delta' && typeof delta === 'string') return { kind: 'text', text: delta }
  if (payload.type === 'error') return errorFrame(errorFromDataPayload(payload))
  return null
}

/** One line of the turn's stream as a frame, or null for a line this reader does not need. */
export function parseChatLine(line: string): ChatFrame | null {
  if (line.startsWith(TEXT_PREFIX)) {
    const text = parseJson(line.slice(TEXT_PREFIX.length))
    return typeof text === 'string' ? { kind: 'text', text } : null
  }
  if (line.startsWith(DATA_PREFIX)) {
    const data = fieldsOf(parseJson(line.slice(DATA_PREFIX.length)))
    return data ? dataFrame(data) : null
  }
  if (line.startsWith(ERROR_PREFIX)) return errorFrame(parseErrorFrame(line.slice(ERROR_PREFIX.length)))
  if (LEGACY_PREFIX.test(line)) {
    const payload = fieldsOf(parseJson(line.replace(LEGACY_PREFIX, '')))
    return payload ? legacyFrame(payload) : null
  }
  return null
}

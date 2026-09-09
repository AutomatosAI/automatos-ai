/**
 * PRD-239 S4 — the error frames a chat turn can end with, parsed into one shape.
 *
 * The stream carries `e:` lines as `{"message": "...", "code": "..."}` (the
 * streaming handler and the detached-turn producer both emit that) and, on older
 * paths, a bare JSON string. Data frames may also carry `{type: 'error', ...}`.
 * Either way the surfaces need a sentence to show and a code to branch on.
 */
import type { TurnError } from '@/types'

export const FALLBACK_TURN_ERROR = 'The reply failed. Try again.'

export function parseErrorFrame(raw: string): TurnError {
  const text = (raw ?? '').trim()
  if (!text) return { message: FALLBACK_TURN_ERROR, code: null }
  try {
    const value = JSON.parse(text)
    if (typeof value === 'string') return { message: value || FALLBACK_TURN_ERROR, code: null }
    if (value && typeof value === 'object') {
      const message = value.message ?? value.error
      const code = value.code ?? value.error_code ?? null
      return {
        message: typeof message === 'string' && message ? message : FALLBACK_TURN_ERROR,
        code: typeof code === 'string' ? code : null,
      }
    }
  } catch {
    // not JSON — the raw text is the message
  }
  return { message: text, code: null }
}

export function errorFromDataPayload(payload: Record<string, unknown> | null | undefined): TurnError {
  const message = payload?.error ?? payload?.message
  const code = payload?.code ?? payload?.error_code ?? null
  return {
    message: typeof message === 'string' && message ? message : FALLBACK_TURN_ERROR,
    code: typeof code === 'string' ? code : null,
  }
}

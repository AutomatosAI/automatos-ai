/**
 * PRD-277 US-502 — a chat request refused because the session ended.
 *
 * The chat streams through the web app's own route (app/api/chat/route.ts), not apiClient, so its
 * 401 would not reach the listeners apiClient tells. In enterprise that route hands the API's 401
 * body back as it came, and the chat hook passes it here: the listeners hear it exactly as they
 * hear apiClient's, and "Your session ended" opens with the chat box's text kept. Every other
 * edition, and every other status, is left to the chat hook as before.
 */
import { reportUnauthorized } from '@/lib/api-unauthorized'
import { isEnterprise } from '@/lib/edition-name'

const UNAUTHORIZED_STATUS = 401

function parsedOrUndefined(text: string): unknown {
  try {
    return JSON.parse(text)
  } catch (error) {
    console.warn('[chat] a 401 body was not JSON', error)
    return undefined
  }
}

/** True when the refusal was the session's and the listeners were told; the hook then shows no toast. */
export function reportedSessionRefusal(status: number, text: string, enterprise: boolean = isEnterprise): boolean {
  if (!enterprise || status !== UNAUTHORIZED_STATUS) return false
  reportUnauthorized(parsedOrUndefined(text))
  return true
}

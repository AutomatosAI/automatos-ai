/**
 * PRD-265 US-011 / W2 — what a billing call's failure says to the person who clicked.
 *
 * The billing routes refuse in plain sentences (FastAPI `{"detail": "Paid plans aren't on
 * sale yet."}`), so a string `detail` is shown as it is. Anything else (a structured detail,
 * a server error, a network failure) gets the surface's own fallback: the message never shows
 * raw JSON, a status line or a stack.
 *
 * Stripe.js answers a declined card or a failed check with an error whose `message` is
 * written for the buyer ("Your card was declined."), so that is shown as it is too.
 */
import type { ApiHttpError } from '@/lib/api-client'

/** 503 is the one server-side status whose detail is written for people ("billing isn't configured"). */
const SERVICE_UNAVAILABLE = 503
const SERVER_ERROR_FLOOR = 500

/** The API's own sentence when it gave one, else `fallback`. */
export function billingErrorMessage(error: unknown, fallback: string): string {
  const { status, body } = (error ?? {}) as ApiHttpError
  if (typeof status === 'number' && status >= SERVER_ERROR_FLOOR && status !== SERVICE_UNAVAILABLE) return fallback
  const detail = (body as { detail?: unknown } | null | undefined)?.detail
  return typeof detail === 'string' && detail.trim() ? detail.trim() : fallback
}

/** Stripe's own sentence for the buyer ("Your card was declined."), else `fallback`. */
export function stripeErrorMessage(error: { message?: unknown } | null | undefined, fallback: string): string {
  const message = error?.message
  return typeof message === 'string' && message.trim() ? message.trim() : fallback
}
